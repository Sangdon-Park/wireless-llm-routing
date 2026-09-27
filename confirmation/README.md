# Confirmation on 450 new tasks

[Back to overview](../README.md)

Four hypotheses were fixed before opening a sealed set of 450 new tasks: 150
native multi-turn IFBench, 150 XSum, and 150 MBPP tasks, with 900 responses from
`qwen3.8:27b` at the edge and `gemini-3.8-flash` in the cloud. The controller
snapshot is supplied in [`frozen_source/qbr/`](frozen_source/qbr/).

The development responses used greedy decoding: temperature 0 for both models
and Ollama seed 20260919. Confirmation responses used sampling: edge temperature
0.7, top-p 0.8, top-k 20, presence penalty 1.5, and seed 20260919. Cloud requests
specified low thinking and otherwise used the API's default temperature. This
change in decoding is a limitation of the comparison.

## Results

The primary analysis uses 2,000 whole-task bootstrap replicates, each with two
traffic seeds. Brackets give central 95% ranges.

| Hypothesis | Decision | Estimate |
| --- | --- | --- |
| H1: queue state lowers cost at ρ = 0.9, more than at ρ = 0.35 | Confirmed | Shortfall–Queue − Intrinsic: −29.4 s [−55.3, −8.2] at ρ = 0.9; −0.009 s [−0.050, 0.026] at ρ = 0.35 |
| H2: oracle gain exceeds the rule-refinement gain | Confirmed | Oracle − CARROT–Queue: −0.89 s [−1.13, −0.66]; CARROT–Queue − Utility: −0.077 s |
| H3: radio correction lowers Mean's cost at ρ_air = 1.43 and is within ±0.7% at 0.12 | Not confirmed | +0.13 s [−0.49, 0.74] at ρ_air = 1.43; relative change at 0.12: [−0.84%, +0.67%] |
| H4: FPI–Queue beats CARROT–Queue with one edge slot | Not confirmed | −0.070 s [−0.209, 0.056] |

These decisions hold when ungraded answers take either extreme score, when the
three task IDs exposed during handling on 2026-09-23 are excluded, and when the
frozen-code task filter is used (448 tasks, 200 replicates). A separate analysis
using 20-seed t intervals also confirms H4. Those intervals describe traffic
variation on a fixed task set, whereas the primary analysis resamples tasks.

The declared grading rule assigns ungraded answers the midpoint of their possible
scores. A deviation found after grading but before analysis was that the frozen
filter dropped two such tasks. The primary analysis follows the declared rule;
`results_frozen448/` reports the filter-based analysis.

## Plans, source files, and verification

The analysis plan, executor specification, predictions, and grader checks were
recorded locally before unsealing. They were first released publicly with the
results. Their recorded timing is therefore based on local records rather than
an external preregistration service.

Public copies use portable paths. The
[publication manifest](publication_manifest.json) distinguishes archived
record hashes from hashes of the public export. The verifier checks the
exported files and published numerical results; it does not establish the original
recording time. Development manifests retain their archived source checksums;
`source_files_lf` in the publication manifest maps them to the public source files.

| File or directory | Contents |
| --- | --- |
| `commitment.json` | Hypotheses, decision rules, inference procedure, and exposure handling |
| `executor_commitment.json` | Executor specification, predictions, grader validation, and declared rules |
| `exposure_receipt.json` | Recorded authorization and scope for opening the task set |
| `exposure_access_log.jsonl.gz` | 7,495 recorded reads, with time, path, file hash, and purpose |
| `deviations.json` | The grading-filter deviation and its treatment |
| `frozen_source/qbr/` | Controller source snapshot used by the confirmation |
| `publication_manifest.json` | Archived hashes and public-export hashes |

From the repository root:

```sh
python confirmation/verify_confirmation.py
```

The verifier checks the published files and recomputes intervals and decisions in
`results/result.json` and `results_frozen448/result.json` from the supplied path
results. The public numerical results can be checked without the private
acquisition store. Running the original acquisition and grading workflow requires
the excluded task and response data. Its entry points use
`WCL_PRIVATE_WORKSPACE` for the acquisition and grading workspace; this variable
is unnecessary for the public verification commands above.

## Data availability

The release includes executor code, CARROT and forest predictions computed from
prompts before unsealing (`predictions/`), grader validation on reservation
canonical solutions (`validation/`), and per-path replay metrics and decisions
(`results*/`). Task prompts (`predictions/prompts.json`), responses, judge outputs,
and per-task grades are not distributed.

## Exploratory static-split comparison

After the confirmation, `exploratory_v18/` compared queue-aware rules with a
capacity-aware static split. Its plan is `data/information_value_review_plan.json`;
`run_static.py` produced 10,840 paths on the frozen new-task bank, stored in
`static_split.jsonl.gz`. This comparison was not one of the four confirmation
hypotheses. Against the best tested static share chosen in hindsight, queue-aware
rules cost 9–12% less with one edge server and 1–2% less with four independent
replicas.

```sh
python confirmation/exploratory_v18/summarize.py
```

This command uses the confirmation's queue-aware paths in `results/full.jsonl.gz`
for the comparison.

## Checks before unsealing

- CARROT and forest predictions reproduced the development-bank predictions exactly.
- The IFBench multi-turn grader scored 450 synthetic answers without an evaluator error.
- The WASI code grader passed 147 of 150 canonical solutions. The other 3 were unsupported and treated as ungraded. It rejected all 150 wrong solutions.
- The replay and analysis ran end to end on a synthetic bank built from development responses.
