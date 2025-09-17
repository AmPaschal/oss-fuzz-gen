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


def build_prompt_parts(prompt: prompts.Prompt, full_text: str) -> prompts.Prompt:
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

  prompt.add_priming(system_instruction)
  prompt.add_problem(full_text)
  return prompt


def eval_file(llm: models.LLM, p: Path) -> Optional[Dict[str, Any]]:
  text = read_text(p)

  if not "<requirement>" in text or not "Fuzz target source:" in text:
    return None

  prompt = llm.prompt_type()(None)

  prompt = build_prompt_parts(prompt, text)

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

  result = {
    'file_name': p.name,
  }

  result.update(parsed)

  return result


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

def summarize_result(results: list[Dict[str, Any]]) -> dict[str, Any]:
  total = len(results)
  true_count = sum(1 for r in results if r.get('conclusion') is True)
  false_count = sum(1 for r in results if r.get('conclusion') is False)
  print(f"Total fuzz drivers evaluated: {total}")
  print(f"Files satisfying all requirements (conclusion=True): {true_count}")
  print(f"Files NOT satisfying all requirements (conclusion=False): {false_count}")

  summary = {
    "total_fuzz_drivers": total,
    "total_properties": 0,
    "all_satisfied": true_count,
  }

  # Compute percentage satisfying all requirements, and all-but-one requirements
  if total > 0:
    # Total
    percent_true = (true_count / total) * 100
    print(f"Percentage satisfying all requirements: {percent_true:.2f}%")

    # All-but-one
    all_but_one_count = 0
    total_x = total_y = 0
    for result in results:
      num_satisfied = result.get('num_satisfied', '0/0')
      x, y = parse_num_satisfied(num_satisfied)
      total_x += x
      total_y += y
      if y > 0 and x == y - 1:
        all_but_one_count += 1

    summary["total_properties"] = total_y

    percent_all_but_one = (all_but_one_count / total) * 100
    print(f"Percentage satisfying all-but-one requirements: {percent_all_but_one:.2f}%")
    summary["all_but_one"] = all_but_one_count

    percent_satisfied = (total_x / total_y) * 100 if total_y > 0 else 0
    print(f"Overall percentage of requirements satisfied: {percent_satisfied:.2f}%")
    summary["requirements_satisfied"] = percent_satisfied

  return summary

# ------------------------------ Main ------------------------------ #


def main(argv: Optional[List[str]] = None) -> int:
  parser = argparse.ArgumentParser(
      description="Evaluate fuzz drivers against requirements using an LLM")
  parser.add_argument('--input-dir',
                      type=Path,
                      required=True,
                      help='Directory with FA-assisted requirement files')
  parser.add_argument('-o', '--output', help='Output JSONL file')
  parser.add_argument(
      '-l',
      '--model',
      type=str,
      default='gemini-2.5-flash',
      help='Gemini model name (e.g., gemini-2.5-flash, gemini-2.5-pro)')
  parser.add_argument(
      '--max', 
      type=int, 
      default=None, 
      help='Max files per directory (for quick runs)')
    
  args = parser.parse_args(argv)

  if args.output:
    output_file = Path(args.output)
  else:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_file = Path(f"results-fa-influence-{timestamp}.jsonl")

  # clear output file if it exists
  if output_file.exists():
    os.remove(output_file)

  args.llm = models.LLM.setup(
      ai_binary='',
      name=args.model,
  )

  timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  records: List[Dict[str, Any]] = []

  count = 0
  for file in iter_txt_files(args.input_dir):
    if args.max is not None and count >= args.max:
      break
    print(f"Evaluating file: {file}")
    file_result = eval_file(args.llm, file)
    if file_result is None:
      print(f"Skipping file {file} due to evaluation error.")
      continue

    records.append(file_result)
    write_jsonl(output_file, file_result)
    count += 1

  # write_jsonl(args.output, records)
  summary = summarize_result(records)

  write_jsonl(output_file, summary)

  print(f"\nSaved {len(records)} records to {output_file}")
  return 0


if __name__ == '__main__':
  raise SystemExit(main())
