import argparse
import json
import re
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import tiktoken

# Regex to extract feasibility blocks
CHAT_BLOCK_RE = re.compile(
    r"<CHAT RESPONSE:ROUND \d+>.*?<feasible>\s*(True|False)\s*</feasible>.*?</CHAT RESPONSE:ROUND \d+>",
    re.DOTALL,
)


def run_command(yaml_path, func, prompt_file, model, output_dir):
  """Run the given command multiple times and capture output files."""
  results = []
  Path(output_dir).mkdir(parents=True, exist_ok=True)

  prompt_dirs = [
      "prompts/agent", "prompts/agent_context_analyzer_type2",
      "prompts/agent_context_analyzer_type4"
  ]

  for i, prompt_dir in enumerate(prompt_dirs):
    out_file = Path(output_dir) / f"run_v{i+1}.log"
    cmd = [
        "python3",
        "-m",
        "agent_tests.agent_test",
        "-y",
        yaml_path,
        "-f",
        func,
        "-p",
        "ContextAnalyzer",
        "-pf",
        prompt_file,
        "-td",
        prompt_dir,
        "-l",
        model,
    ]
    with open(out_file, "w") as f:
      subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    results.append((prompt_dir, str(out_file)))
  return results


def parse_feasibility(log_file):
  """Extract feasibility values from a log file."""
  with open(log_file, "r") as f:
    content = f.read()
  return CHAT_BLOCK_RE.findall(content)


def extract_final_response_metrics(log_file_path):
  # Read the log file
  with open(log_file_path, "r", encoding="utf-8") as f:
    content = f.read()

  # Regex to capture all response blocks
  pattern = re.compile(
      r"<CHAT RESPONSE:ROUND (\d+)>(.*?)</CHAT RESPONSE:ROUND \1>", re.DOTALL)

  # Find all matches and select the last one
  matches = pattern.findall(content)
  if not matches:
    raise ValueError("No valid <CHAT RESPONSE:ROUND> blocks found in log file.")

  round_number, final_response = matches[-1]

  # Extract <feasible> content
  feasible_match = re.search(r"<feasible>(.*?)</feasible>", final_response,
                             re.DOTALL)
  feasible = feasible_match.group(1).strip() if feasible_match else "Unknown"

  # Token count using cl100k_base tokenizer
  enc = tiktoken.get_encoding("cl100k_base")
  response_tokens = len(enc.encode(final_response))

  # Validate other required tags exist
  for tag in ["analysis", "source_code_evidence", "recommendations"]:
    if not re.search(rf"<{tag}>(.*?)</{tag}>", final_response, re.DOTALL):
      raise ValueError(f"Missing required tag <{tag}> in final response.")

  return {
      "feasible": feasible,
      "round_number": round_number,
      "response_tokens": response_tokens
  }


def process_case(idx, test, model, output_root):
  """Run one test case and return feasibility analysis."""
  yaml_path = test["-y"]
  func = test["-f"]
  prompt_file = test["-pf"]
  prompt_file_name = Path(prompt_file).name
  case_out_dir = Path(output_root) / f"case_{idx}_{prompt_file_name}"
  output_log_file = Path(output_root) / "summary.log"

  with open(output_log_file, "a") as outf:
    outf.write(f"[Case {idx}] Running: {func}, {prompt_file}\n\n")

  log_files = run_command(yaml_path, func, prompt_file, model, case_out_dir)

  with open(output_log_file, "a") as outf:
    outf.write(f"[Case {idx}]: {func} results:\n")
  for prompt_dir, log_file in log_files:
    try:
      metrics = extract_final_response_metrics(log_file)
      # Print metrics in a well formatted way
      data = dict(metrics)
      data["prompt_version"] = prompt_dir
      with open(output_log_file, "a") as outf:
        json.dump(data, outf, indent=2)
        outf.write("\n")
    except Exception as e:
      with open(output_log_file, "a") as outf:
        outf.write(f"[Case {idx}] Error processing {log_file}: {e}\n")


def main():
  parser = argparse.ArgumentParser(description="Batch runner for agent_tests.")
  parser.add_argument("input_json",
                      help="Path to JSON file containing test cases")
  parser.add_argument("-o",
                      "--output",
                      help="Directory to store logs (default: timestamped)")
  parser.add_argument("-l", "--model", required=True, help="Model name")
  parser.add_argument("-j",
                      "--jobs",
                      type=int,
                      default=2,
                      help="Number of parallel jobs")
  args = parser.parse_args()

  # If no output dir provided, create one with timestamp
  if args.output:
    output_root = Path(args.output)
  else:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_root = Path(f"results-ca-prompt-influence-{timestamp}")

  output_root.mkdir(parents=True, exist_ok=True)

  with open(args.input_json, "r") as f:
    tests = json.load(f)

  futures = []
  with ProcessPoolExecutor(max_workers=args.jobs) as executor:
    for idx, test in enumerate(tests, start=1):
      futures.append(
          executor.submit(process_case, idx, test, args.model, output_root))

    for future in as_completed(futures):
      # process_case does not return anything, just ensure completion
      future.result()


if __name__ == "__main__":
  main()
