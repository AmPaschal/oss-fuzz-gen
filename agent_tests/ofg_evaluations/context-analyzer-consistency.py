import argparse
import json
import re
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

# Regex to extract feasibility blocks
CHAT_BLOCK_RE = re.compile(
    r"<CHAT RESPONSE:ROUND \d+>.*?<feasible>\s*(True|False)\s*</feasible>.*?</CHAT RESPONSE:ROUND \d+>",
    re.DOTALL,
)


def run_command(yaml_path, func, prompt_file, model, repeat, output_dir):
  """Run the given command multiple times and capture output files."""
  results = []
  Path(output_dir).mkdir(parents=True, exist_ok=True)

  for i in range(repeat):
    out_file = Path(output_dir) / f"run_{i+1}.log"
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
        "-l",
        model,
    ]
    with open(out_file, "w") as f:
      subprocess.run(cmd, stdout=f, stderr=f)
    results.append(str(out_file))
  return results


def parse_feasibility(log_file):
  """Extract feasibility values from a log file."""
  with open(log_file, "r") as f:
    content = f.read()
  return CHAT_BLOCK_RE.findall(content)


def process_case(idx, test, model, repeat, output_root):
  """Run one test case and return feasibility analysis."""
  yaml_path = test["-y"]
  func = test["-f"]
  prompt_file = test["-pf"]
  prompt_file_name = Path(prompt_file).name
  case_out_dir = Path(output_root) / f"case_{idx}_{prompt_file_name}"
  output_log_file = Path(output_root) / "summary.log"

  with open(output_log_file, "a") as outf:
    outf.write(f"[Case {idx}] Running: {yaml_path}, {func}, {prompt_file}\n")

  log_files = run_command(yaml_path, func, prompt_file, model, repeat,
                          case_out_dir)

  all_feasibles = []
  for lf in log_files:
    all_feasibles.extend(parse_feasibility(lf))

  if not all_feasibles or len(all_feasibles) <= 1:
    text = f"[Case {idx}]: Some feasibility results missing.\n"
    consistent = None
  else:
    consistent = len(set(all_feasibles)) == 1
    status = "consistent" if consistent else "inconsistent"
    text = f"[Case {idx}]: Feasibility results: {all_feasibles} ({status})\n"

  with open(output_log_file, "a") as outf:
    outf.write(text)

  return consistent


def main():
  parser = argparse.ArgumentParser(description="Batch runner for agent_tests.")
  parser.add_argument("input_json",
                      help="Path to JSON file containing test cases")
  parser.add_argument("-r",
                      "--repeat",
                      type=int,
                      default=3,
                      help="Number of times to repeat each test")
  parser.add_argument("-o",
                      "--output",
                      help="Directory to store logs (default: timestamped)")
  parser.add_argument("-l",
                      "--model",
                      required=True,
                      help="Model name (default gpt-5)")
  parser.add_argument("-j",
                      "--jobs",
                      type=int,
                      default=5,
                      help="Number of parallel jobs")
  args = parser.parse_args()

  print("Starting experiments...")

  # If no output dir provided, create one with timestamp
  if args.output:
    output_root = Path(args.output)
  else:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_root = Path(f"results-ca-consistency-{timestamp}")

  output_root.mkdir(parents=True, exist_ok=True)

  with open(args.input_json, "r") as f:
    tests = json.load(f)

  futures = []
  with ProcessPoolExecutor(max_workers=args.jobs) as executor:
    for idx, test in enumerate(tests, start=1):
      futures.append(
          executor.submit(process_case, idx, test, args.model, args.repeat,
                          output_root))

    for future in as_completed(futures):
      future.result()

    print(f"All test cases completed.")

if __name__ == '__main__':
  main()