# Bao decision-rule transfers

This comparison applies two variants of Bao et al.'s quality–latency routing rule
to 225 recorded task pairs. Each variant uses 12 conditions and 20 traffic seeds,
for 480 replay trajectories in total. Both use the original AP-entry executor,
strict512 instruction responses, full summary/code responses, streaming and
retry accounting, and achieved cost. They require no additional model responses.

## Adapted rule

The rule selects cloud when

`p - alpha * (T_cloud - T_edge) > theta`.

Here `p` is the archived RouteLLM-BERT strong-model preference transferred to the
edge/cloud pair. An exact tie selects edge; the fused score is not clipped.
`T` is observed queue waiting plus calibrated service duration, in seconds.
`Bao-Debt` also adds `qbr.strengthening.radio_correction` to each route:

`max(sum(backlog/capacity) + payload/capacity[user] - serving, 0)`.

This is the earlier airtime-debt correction. Bao's latency estimate excludes the
release-aware correction in `qbr.information_controls`, monetary cost, and the
quality-target term.

The decision rule follows [Bao et al., IEEE/CIC ICCC Workshops 2025,
DOI 10.1109/ICCCWorkshops67136.2025.11147210](https://doi.org/10.1109/ICCCWorkshops67136.2025.11147210)
([author manuscript, Eqs. 11–12](https://arxiv.org/html/2508.11291v1)). The original
local-device/remote-edge topology, context/cache switching, and length-augmented
semantic classifier are outside this comparison. These experiments evaluate the
transferred decision rule within this project's replay model.

## Parameter selection

| Transfer | alpha (1/s) | theta |
| --- | ---: | ---: |
| Bao-Queue | 0.3 | 0.5 |
| Bao-Debt | 1.0 | 0.5 |

Each variant selected parameters on the earlier 660 calibration tasks and three
primary-condition traffic seeds. The fixed grid contained seven alpha values and
six theta values, including the previous native RouteLLM threshold 0.731: 42
points per variant and 252 calibration trajectories. Selection minimized mean
achieved cost, breaking exact ties by `(alpha, theta)`.

All 480 evaluation trajectories used these fixed parameters. They were not retuned
on the 225 evaluation tasks, but the comparison remains retrospective because
those tasks had appeared in earlier experiments.

## Reproduction

```sh
python -m unittest test_bao -v
python reproduce_bao.py verify
python reproduce_bao.py summary --output outputs/bao-summary
python reproduce_bao.py replay --smoke --workers 2 --output outputs/bao-smoke
python reproduce_bao.py replay --workers 4 --output outputs/bao-full
```

`replay` runs all 480 trajectories and compares route hashes, termination counts,
and numerical metrics with the archived results, using absolute tolerance `1e-8`.
`--smoke` checks two primary-condition trajectories. The archived key `bao_radio`
denotes Bao-Debt.

The replay requires only the repository dependencies. Recorded per-request
RouteLLM head times delay admission and incur a $1/hour charge; controller and
radio bookkeeping CPU time is excluded for all controls. Four independent
inference slots per route are a simulation assumption. A single Ollama endpoint's
measured throughput does not support that capacity.

## Available data and scope

`data/bao_plan.json` records the calibration grid, aggregate costs, selected values,
and source hashes. `data/bao_results.json.gz` contains numerical evaluation metrics
and routing hashes. The summary includes every condition and the CARROT-Queue,
CARROT-Radio, and release-aware comparisons. Its paired traffic-seed intervals
measure variation conditional on the fixed task bank.

The public arrays reproduce the fixed-parameter evaluation replay. The original
660 calibration response streams, prompt configurations, and measured training
head times are excluded, so the 252-path parameter search cannot be rerun from
this release alone. The semantic predictor is neither retrained nor remeasured.
The comparison isolates the transferred rule's performance in this setting;
it does not attribute any difference to calibration alone.
