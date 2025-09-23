# Evaluating Function and Context Analyzers

The python scripts in this directory are used to evaluate the following aspects of the function analyzer and the context analyzers.

1. Do function requirements, provided by the function analyzer, influence the generated fuzz drivers? 
[function-analyzer-influence.py](./function-analyzer-influence.py)
2. How consistent is the context analyzer in its conclusion?
[context-analyzer-consistency.py](./context-analyzer-consistency.py)
3. How do detailed and non-detailed prompts influence the context analyzer's conclusion?
[context-analyzer-prompt-influence.py](./context-analyzer-prompt-influence.py)

Here are the commands to execute each of these scripts.

## Execution Commands

You can execute each of the following commands from the root of the OSS-Fuzz-Gen's directory.
Commands can be executed with ```nohup [command] &``` to run it in the background.

### Function Analyzer Influence

This command will be run on two directories. The first directory is testcases created from experiments conducted using the function analyzer, while the second contains testcases from experiments without the function analyzer.

```bash
python3 -m agent_tests.ofg_evaluations.function-analyzer-influence --input-dir agent_tests/ofg_evaluations/2025-07-29-weekly-all-1-testcases -o 2025-07-29-weekly-all-1-output.jsonl -l [model]
```

```bash
python3 -m agent_tests.ofg_evaluations.function-analyzer-influence --input-dir agent_tests/ofg_evaluations/2025-07-30-weekly-all-1-testcases -o 2025-07-30-weekly-all-1-output.jsonl -l [model]
```

The results of both commands will be written to files specified with the -o or --output flags, and located in the OSS-Fuzz-Gen's directory

### Context Analyzer Consistency

There are three input files
agent_tests/ofg_evaluations/context-analyzer-input-part1.json
agent_tests/ofg_evaluations/context-analyzer-input-part2.json
agent_tests/ofg_evaluations/context-analyzer-input-part3.json

Please, run with the first file, and if it succeeds and we have time, we can run the remaining two.
```bash
python -m agent_tests.ofg_evaluations.context-analyzer-consistency agent_tests/ofg_evaluations/context-analyzer-input-part1.json -l [model]
```
The output logs and result file will be written to a default directory ```results-ca-consistency-{timestamp}```

### Context Analyzer Prompt Influence

We will also reuse the same input files.
agent_tests/ofg_evaluations/context-analyzer-input-part1.json
agent_tests/ofg_evaluations/context-analyzer-input-part2.json
agent_tests/ofg_evaluations/context-analyzer-input-part3.json

Please, also run with the first file, and if it succeeds and we have time, we can run the remaining two.

```bash
python -m agent_tests.ofg_evaluations.context-analyzer-prompt-influence agent_tests/ofg_evaluations/context-analyzer-input-part1.json -l [model]
```
The output logs and result file will be written to a default directory ```results-ca-prompt-influence-{timestamp}```