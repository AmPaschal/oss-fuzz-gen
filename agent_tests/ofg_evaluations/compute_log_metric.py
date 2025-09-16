import re
import sys
import tiktoken

def extract_final_response_metrics(log_file_path):
    # Read the log file
    with open(log_file_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Regex to capture all response blocks
    pattern = re.compile(
        r"<CHAT RESPONSE:ROUND (\d+)>(.*?)</CHAT RESPONSE:ROUND \1>",
        re.DOTALL
    )

    # Find all matches and select the last one
    matches = pattern.findall(content)
    if not matches:
        raise ValueError("No valid <CHAT RESPONSE:ROUND> blocks found in log file.")
    
    round_number, final_response = matches[-1]

    # Extract <feasible> content
    feasible_match = re.search(r"<feasible>(.*?)</feasible>", final_response, re.DOTALL)
    feasible = feasible_match.group(1).strip() if feasible_match else "Unknown"

    # Token count using cl100k_base tokenizer
    enc = tiktoken.get_encoding("cl100k_base")
    response_tokens = len(enc.encode(final_response))

    # Validate other required tags exist
    for tag in ["analysis", "source_code_evidence", "recommendations"]:
        if not re.search(rf"<{tag}>(.*?)</{tag}>", final_response, re.DOTALL):
            raise ValueError(f"Missing required tag <{tag}> in final response.")

    # Print metrics
    print("Feasible:", feasible)
    print("Round to conclusion:", round_number)
    print("Response tokens:", response_tokens)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <log_file_path>")
    else:
        extract_final_response_metrics(sys.argv[1])
