#!/usr/bin/env python3
"""
Analyze fuzz drivers against enumerated requirements using an LLM.

For every .txt file under with-fa/ and without-fa/, this script:
    - Reads the description, requirements, and fuzz target source from the file.
    - Asks Gemini 2.5 to judge whether the fuzz driver satisfies all requirements.
    - Expects STRICT JSON: {"conclusion": bool, "num_satisfied": "X/Y", "analysis": str}
    - Appends one JSON object per file to an output JSONL file.
    - Prints a summary comparing with-fa vs without-fa per paired filename.

Usage:
    python analyzer-fuzz-drivers.py \
        --with-dir agent_tests/function-analyzer-req/with-fa \
        --without-dir agent_tests/function-analyzer-req/without-fa \
        --output agent_tests/function-analyzer-req/fuzz-driver-req-eval.jsonl \
    --model gemini-2.5-flash \
    --gcp-project $GOOGLE_CLOUD_PROJECT --location us-central1
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

# Standalone: use Vertex AI Gemini SDK
import vertexai

from llm_toolkit import models, prompts

try:
  # Prefer stable API if available
  from vertexai.generative_models import GenerativeModel
except Exception:  # pragma: no cover
  # Fallback to preview namespace for older installs
  from vertexai.preview.generative_models import GenerativeModel  # type: ignor

# ------------------------------ Data types ------------------------------ #


@dataclass
class FileEval:
  group: str  # "with-fa" or "without-fa"
  name: str  # filename base
  path: Path
  result: Dict[str, Any]  # Parsed LLM JSON


# ------------------------------ Helpers ------------------------------ #

REQ_TAG_RE = re.compile(r"<requirement>\s*(.*?)\s*</requirement>", re.DOTALL)
DESC_TAG_RE = re.compile(r"<description>\s*(.*?)\s*</description>", re.DOTALL)
SIG_TAG_RE = re.compile(r"<function_signature>\s*(.*?)\s*</function_signature>",
                        re.DOTALL)
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


def build_prompt_parts(full_text: str) -> Tuple[str, str]:
  system_instruction = (
      "You are evaluating whether a fuzz driver satisfies all listed requirements for a target function in a given project. "
      "Read the requirements and the fuzz target source embedded in the text. "
      "Judge each requirement strictly based on what the driver actually does. "
      "Then return ONLY a single JSON object with fields: conclusion (boolean), num_satisfied (string 'X/Y'), analysis (string). "
      "- conclusion = true iff ALL requirements are satisfied; otherwise false. "
      "- num_satisfied must be 'X/Y' where Y = the count of requirements found in the <requirements> section, and X is how many are satisfied. "
      "- analysis must briefly justify, requirement-by-requirement, why each is or is not satisfied. "
      "No extra text, no markdown, no code fences — output just the JSON object."
  )

  prompt = prompts.TextPrompt()
  prompt.add_priming(system_instruction)
  prompt.append(full_text)
  return prompt


def eval_file(llm: models.LLM, project: str, location: str, group: str,
              p: Path) -> Optional[FileEval]:
  text = read_text(p)

  if not "<requirement>" in text or not "Fuzz target source:" in text:
    return None

  prompt = build_prompt_parts(text)

  client = llm.get_chat_client(model=llm.get_model())
  response = llm.chat_llm(client, prompt)
  raw_json = sanitize_json_maybe(response)
  try:
    parsed: Dict[str, Any] = json.loads(raw_json)
  except Exception as e:
    print(
        f"Warning: LLM returned invalid JSON for {p.name}. Error: {e}. Raw: {raw_json[:500]}..."
    )
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


def compare_and_print(with_evals: Dict[str, FileEval],
                      without_evals: Dict[str, FileEval]) -> None:
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
    wo_x, wo_y = parse_num_satisfied(
        wo.result.get('num_satisfied') if wo else None)

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

    print(
        f"{name}: with-fa {w_x}/{w_y} (conclusion={w_conc}) vs without-fa {wo_x}/{wo_y} (conclusion={wo_conc}) | Δ={delta}"
    )

  print("-" * 60)

  def ratio(x: int, y: int) -> str:
    return f"{x}/{y}" if y else "0/0"

  print(
      f"Totals: with-fa {ratio(total_with_x, total_with_y)} ; without-fa {ratio(total_without_x, total_without_y)}"
  )
  print(f"Pairwise: wins={wins}, draws={draws}, losses={losses}")


# ------------------------------ Main ------------------------------ #


def main(argv: Optional[List[str]] = None) -> int:
  parser = argparse.ArgumentParser(
      description="Evaluate fuzz drivers against requirements using an LLM")
  parser.add_argument('--with-dir',
                      required=True,
                      help='Directory with FA-assisted requirement files')
  parser.add_argument('--without-dir',
                      required=True,
                      help='Directory without FA requirement files')
  parser.add_argument('--output', required=True, help='Output JSONL file')
  parser.add_argument(
      '--model',
      type=str,
      default='gemini-2.5-flash',
      help='Gemini model name (e.g., gemini-2.5-flash, gemini-2.5-pro)')

  args = parser.parse_args(argv)

  args.llm = models.LLM.setup(
      ai_binary='',
      name=args.model,
  )

  timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  records: List[Dict[str, Any]] = []

  with_evals: Dict[str, FileEval] = {}
  without_evals: Dict[str, FileEval] = {}

  for group, dir_path in [('with-fa', args.with_dir),
                          ('without-fa', args.without_dir)]:
    count = 0
    for p in iter_txt_files(dir_path):
      if args.max is not None and count >= args.max:
        break
      print(f"Evaluating {group} file: {p}")
      fe = eval_file(args.llm, args.gcp_project, args.location, group, p)
      if fe is None:
        print(f"Skipping file {p} due to evaluation error.")
        continue
      rec = {
          'ts': timestamp,
          'group': group,
          'filename': fe.name,
          'path': str(p),
          'model': args.llm.name,
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
