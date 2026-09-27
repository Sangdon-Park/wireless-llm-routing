# Quality, Queue, and Radio Information for Wireless Edge–Cloud LLM Routing

Code and numerical data for the manuscript by **Sangdon Park and Joohyung Lee**.
The study compares the value of quality predictions, inference-queue state, and
radio state when routing LLM requests between an edge model and a cloud model.
Matched policies replay the same recorded response streams.

Python 3.12 · Numerical replay requires no API key or GPU · MIT license

## Quick start

Run from the repository root:

```sh
python -m pip install -r requirements.txt
python reproduce.py verify
python information_value.py --verify
python plot_information_value.py
```

The last command creates `outputs/figures/results_information.pdf` and `.png`
from the supplied results. To rerun one path and compare it with the archived
record:

```sh
python information_value.py --rerun --exp load --method fpi --rho 0.9 --seed 23924013
```

## Findings

The development study uses 225 recorded task pairs. Results below average 12
wireless conditions and 20 paired traffic seeds unless stated otherwise. One
edge server is the primary setting; four replicas represent independent servers.

| Comparison | Change in weighted cost |
| --- | --- |
| Queue-aware routing vs. the best tested static split chosen in hindsight | −8.0% to −8.9% with one edge server; −1.0% to −2.5% with four replicas |
| Queue-aware routing vs. queue-blind scoring | −0.1% (not significant) at load 0.35 and −16.2% at load 0.9 with four replicas; queue-blind scoring overloads the single edge server |
| Realized quality vs. CARROT predictions within the same routing rule | −7.9% with one edge server and −11.5% with four replicas; this oracle also knows response-specific randomness |
| Radio correction | Absolute change below 0.7% for airtime load < 0.5; a 4.8% reduction at 1.43 with four replicas and the Mean predictor. No significant reduction with one edge server or CARROT predictions |
| Full quality distribution vs. its mean, four replicas | +0.55%; the task-bank interval includes zero |
| Target clipping and workload penalty vs. linear utility, four replicas | −1.2% at β = 12; the advantage reverses at β = 8, 16, 24, and 36 |

Predicted and realized route-quality differences correlate only 0.13–0.21. With
four replicas, a noisy oracle with approximately CARROT's mean squared error
costs 6% less, showing that similar prediction errors can yield different routing
performance. See the [study guide](docs/information_value.md) for conditions and
intervals.

Four hypotheses were fixed before opening 450 new tasks. The queue-state and
quality-information hypotheses were confirmed; the radio and load-derived
workload-weight hypotheses were not. A later exploratory static-split comparison
found reductions of 9–12% with one edge server and 1–2% with four replicas.
The [confirmation record](confirmation/README.md) separates these analyses.

Some refinements also fail on the development set: hindsight tuning of the
utility weight beats CARROT–Queue, its advantage is sensitive to load and server
count, and the declared criterion for the load-derived coefficient is not met.
The supplied results include these comparisons.

## Repository guide

| Location | Contents |
| --- | --- |
| [`qbr/`](qbr/) | Routing policies, quality models, wireless replay, and comparison controls |
| [`data/`](data/) | Numerical response records, frozen predictions, plans, and experiment results |
| [`information_value.py`](information_value.py) | Verification, summaries, and replay of the main study |
| [`plot_information_value.py`](plot_information_value.py) | Manuscript figure |
| [Study guide](docs/information_value.md) | Methods, experiment counts, results, and commands |
| [Reproduction guide](docs/reproduction.md) | Earlier comparisons and their replay commands |
| [`confirmation/`](confirmation/README.md) | New-task hypotheses, public analysis files, numerical results, and verification |

## Scope and reproducibility

The development archives contain 52,420 replay and task-bank resampling paths.
They reuse 225 task pairs, with one recorded response per route. Their intervals
are conditional on that bank. The separate confirmation uses 450 new tasks.
Concurrent serving is simulated; a measured endpoint probe supports one serving
slot for the local edge model.

The public release supports numerical replay. It excludes prompt/response text,
raw provider logs, and third-party model weights. Queue-aware controls built from
CARROT and RouteLLM predictions use this project's serving model; the guides
identify the adaptations and the native baselines separately.

[Citation](CITATION.cff) · [License](LICENSE) · [Bao rule transfers](docs/bao.md)
