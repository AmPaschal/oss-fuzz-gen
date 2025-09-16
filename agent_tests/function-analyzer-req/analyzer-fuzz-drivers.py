#!/usr/bin/env python3
"""
Analyze fuzz drivers against enumerated requirements using an LLM.

For every .txt file under with-fa/ and without-fa/, this script:
    - Reads the description, requirements, and fuzz target source from the file.
    - Asks gpt-5 to judge whether the fuzz driver satisfies all requirements.
    - Expects STRICT JSON: {"conclusion": bool, "num_satisfied": "X/Y", "analysis": str}
    - Appends one JSON object per file to an output JSONL file.
    - Prints a summary comparing with-fa vs without-fa per paired filename.

Usage:
    python analyzer-fuzz-drivers.py \
        --with-dir agent_tests/function-analyzer-req/with-fa \
        --without-dir agent_tests/function-analyzer-req/without-fa \
        --output agent_tests/function-analyzer-req/fuzz-driver-req-eval.jsonl \
        --model gpt-5 \
        [--openai-api-key $OPENAI_API_KEY] [--api-base https://api.openai.com/v1]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Standalone: no repo-local dependencies
try:
    import openai
except Exception:  # pragma: no cover - we'll fallback to HTTP
    openai = None  # type: ignore
import urllib.request
import urllib.error


logger = logging.getLogger(__name__)


# ------------------------------ Data types ------------------------------ #


@dataclass
class FileEval:
    group: str  # "with-fa" or "without-fa"
    name: str   # filename base
    path: Path
    result: Dict[str, Any]  # Parsed LLM JSON


# ------------------------------ Helpers ------------------------------ #


REQ_TAG_RE = re.compile(r"<requirement>\s*(.*?)\s*</requirement>", re.DOTALL)
DESC_TAG_RE = re.compile(r"<description>\s*(.*?)\s*</description>", re.DOTALL)
SIG_TAG_RE = re.compile(r"<function_signature>\s*(.*?)\s*</function_signature>", re.DOTALL)
PROJ_TAG_RE = re.compile(r"<project_name>\s*(.*?)\s*</project_name>", re.DOTALL)


def read_text(p: Path) -> str:
    with p.open('r', encoding='utf-8') as f:
        return f.read()


def extract_between_tags(text: str, regex: re.Pattern[str]) -> Optional[str]:
    m = regex.search(text)
    return m.group(1).strip() if m else None


def parse_requirements(text: str) -> List[str]:
    return [m.strip() for m in REQ_TAG_RE.findall(text)]


def sanitize_json_maybe(s: str) -> str:
    """Try to extract a JSON object from s, stripping code fences if present."""
    s = s.strip()
    # Remove Markdown fences if present
    if s.startswith('```'):
        s = re.sub(r"^```[a-zA-Z0-9]*\n|\n```$", "", s)
    # Find first balanced { ... }
    start = s.find('{')
    end = s.rfind('}')
    if start != -1 and end != -1 and end > start:
        return s[start:end + 1]
    return s


def call_openai_chat(messages: List[Dict[str, str]], model: str, api_key: str, api_base: str) -> str:
    """Calls OpenAI's Chat Completions API and returns assistant content.

    Uses openai package if available; otherwise falls back to raw HTTP.
    """
    # Prefer SDK if present
    if openai is not None:
        client = openai.OpenAI(api_key=api_key, base_url=api_base)
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=1,
            n=1,
        )
        return resp.choices[0].message.content or ''

    # Fallback to raw HTTP
    payload = json.dumps({
        'model': model,
        'messages': messages,
        'temperature': 0,
        'n': 1,
    }).encode('utf-8')
    req = urllib.request.Request(
        url=f"{api_base.rstrip('/')}/chat/completions",
        data=payload,
        headers={
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json',
        },
        method='POST',
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode('utf-8'))
    except urllib.error.HTTPError as e:
        detail = e.read().decode('utf-8', errors='replace') if hasattr(e, 'read') else str(e)
        raise RuntimeError(f"OpenAI API HTTPError {e.code}: {detail}") from e
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"OpenAI API request failed: {e}") from e

    try:
        return data['choices'][0]['message']['content'] or ''
    except Exception as e:
        raise RuntimeError(f"Unexpected OpenAI response format: {data}") from e


def build_prompt(full_text: str) -> List[Dict[str, str]]:
    sys_prompt = (
            "You are evaluating whether a fuzz driver satisfies all listed requirements for a target function in a given project. "
            "Read the requirements and the fuzz target source embedded in the text. "
            "Judge each requirement strictly based on what the driver actually does. "
            "Then return ONLY a single JSON object with fields: conclusion (boolean), num_satisfied (string 'X/Y'), analysis (string). "
            "- conclusion = true iff ALL requirements are satisfied; otherwise false. "
            "- num_satisfied must be 'X/Y' where Y = the count of requirements found in the <requirements> section, and X is how many are satisfied. "
            "- analysis must briefly justify, requirement-by-requirement, why each is or is not satisfied. "
            "No extra text, no markdown, no code fences — output just the JSON object."
    )

    return [
        { 'role': 'system', 'content': sys_prompt },
        { 'role': 'user', 'content': full_text },
    ]


def eval_file(model: str, api_key: str, api_base: str, group: str, p: Path) -> Optional[FileEval]:
    text = read_text(p)

    if not "<requirement>" in text or not "Fuzz target source:" in text:
        return None

    messages = build_prompt(text)
    raw = call_openai_chat(messages, model, api_key, api_base)
    raw_json = sanitize_json_maybe(raw)
    try:
        parsed: Dict[str, Any] = json.loads(raw_json)
    except Exception as e:
        print(f"Warning: LLM returned invalid JSON for {p.name}. Error: {e}. Raw: {raw[:500]}...")
        return None

    return FileEval(
            group=group,
            name=p.name,
            path=p,
            result=parsed,
    )


def iter_txt_files(dir_path: Path) -> Iterable[Path]:
    if not dir_path.exists():
        return []
    yield from sorted(dir_path.glob('*.txt'))


def parse_num_satisfied(s: Optional[str]) -> Tuple[int, int]:
    if not s:
        return (0, 0)
    m = re.match(r"\s*(\d+)\s*/\s*(\d+)\s*$", s)
    if not m:
        return (0, 0)
    return int(m.group(1)), int(m.group(2))


def write_jsonl(output: Path, record: Dict[str, Any]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False, indent=2) + "\n")

def compare_and_print(with_evals: Dict[str, FileEval], without_evals: Dict[str, FileEval]) -> None:
    print("\nSummary (with-fa vs without-fa):")
    print("-" * 60)

    wins = draws = losses = 0
    total_with_x = total_with_y = 0
    total_without_x = total_without_y = 0
    total_w_true = total_wo_true = 0

    all_names = sorted(set(with_evals.keys()) | set(without_evals.keys()))
    for name in all_names:
        w = with_evals.get(name)
        wo = without_evals.get(name)
        if not w and not wo:
            continue

        w_x, w_y = parse_num_satisfied(w.result.get('num_satisfied') if w else None)
        wo_x, wo_y = parse_num_satisfied(wo.result.get('num_satisfied') if wo else None)

        total_with_x += w_x
        total_with_y += w_y
        total_without_x += wo_x
        total_without_y += wo_y

        w_conc = w.result.get('conclusion', False) if w else False
        wo_conc = wo.result.get('conclusion', False) if wo else False

        if w_conc:
            total_w_true += 1
        if wo_conc:
            total_wo_true += 1

        delta = (w_x - wo_x) if (w and wo) else 0
        if w and wo:
            if w_conc and not wo_conc:
                wins += 1
            elif w_conc == wo_conc:
                draws += 1
            else:
                losses += 1

        print(f"{name}: with-fa {w_x}/{w_y} (conclusion={w_conc}) vs without-fa {wo_x}/{wo_y} (conclusion={wo_conc}) | Δ={delta}")

    print("-" * 60)
    def ratio(x: int, y: int) -> str:
        return f"{x}/{y}" if y else "0/0"

    print(f"Totals: with-fa {ratio(total_with_x, total_with_y)} ; without-fa {ratio(total_without_x, total_without_y)}")
    print(f"Pairwise: wins={wins}, draws={draws}, losses={losses}")


# ------------------------------ Main ------------------------------ #


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate fuzz drivers against requirements using an LLM")
    default_root = Path('/home/pamusuo/research/oss-fuzz-projects/reports/2025-07-29-weekly-all-1')
    parser.add_argument('--with-dir', type=Path, default=default_root / 'with_fa', help='Directory with FA-assisted requirement files')
    parser.add_argument('--without-dir', type=Path, default=default_root / 'without_fa', help='Directory without FA requirement files')
    parser.add_argument('--output', type=Path, default=default_root / 'fuzz-driver-req-eval.jsonl', help='Output JSONL file')
    parser.add_argument('--model', type=str, default='gpt-5', help='LLM model name (e.g., gpt-5, gpt-4o)')
    parser.add_argument('--openai-api-key', type=str, default=os.getenv('OPENAI_API_KEY', ''), help='OpenAI API key (or set OPENAI_API_KEY env var)')
    parser.add_argument('--api-base', type=str, default=os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1'), help='OpenAI API base URL')
    parser.add_argument('--max', type=int, default=None, help='Max files per directory (for quick runs)')
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    if not args.openai_api_key:
        print('Error: OPENAI_API_KEY not provided. Use --openai-api-key or set env var.', file=sys.stderr)
        return 2

    timestamp = datetime.utcnow().isoformat() + 'Z'
    records: List[Dict[str, Any]] = []

    with_evals: Dict[str, FileEval] = {}
    without_evals: Dict[str, FileEval] = {}

    for group, dir_path in [( 'with-fa', args.with_dir), ( 'without-fa', args.without_dir)]:
        count = 0
        for p in iter_txt_files(dir_path):
            if args.max is not None and count >= args.max:
                break
            logger.info("Evaluating %s file: %s", group, p)
            fe = eval_file(args.model, args.openai_api_key, args.api_base, group, p)
            if fe is None:
                logger.warning("Skipping file %s due to evaluation error.", p)
                continue
            rec = {
                    'ts': timestamp,
                    'group': group,
                    'filename': fe.name,
                    'path': str(p),
                        'model': args.model,
                    'result': fe.result,
            }
            records.append(rec)
            write_jsonl(args.output, rec)
            if group == 'with-fa':
                with_evals[fe.name] = fe
            else:
                without_evals[fe.name] = fe
            count += 1

    # write_jsonl(args.output, records)
    compare_and_print(with_evals, without_evals)

    print(f"\nSaved {len(records)} records to {args.output}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

