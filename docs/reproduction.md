# Reproducing the earlier comparisons

[Back to overview](../README.md)

The [study guide](information_value.md) covers the experiments in the current
manuscript. This page documents the earlier baseline, calibration, and adapted-rule
comparisons. Their numerical records and replay commands remain available.

The shared router scores edge and cloud inference using expected quality
shortfall, observed serving workload, and model charges. Later controls separate
these contributions and test alternative quality predictions. Unless stated
otherwise, these earlier comparisons use four independent serving slots per route.

## Bao rule comparisons

Two transfers of Bao et al.'s quality–latency fusion rule, `Bao-Queue` and
`Bao-Debt`, use the same archived RouteLLM score
and predictor runtime; Debt uses the older airtime-debt correction, separately
from the release-aware radio controls. Their 12-condition costs are 5.7529 and
5.9079 s, versus 5.6337 s for CARROT-Queue. These are comparisons of adapted routing
rules within the same replay setting.

The 480 evaluation trajectories cover all 12 conditions and 20 seeds.
The fixed-parameter replay is reproducible from the public arrays. The preceding
252-trajectory parameter search is documented but cannot be rerun from this
release alone because the original 660 calibration streams are not redistributed.
See [the transfer assumptions and full commands](bao.md).

```sh
python -m unittest test_bao -v
python reproduce_bao.py verify
python reproduce_bao.py replay --smoke --workers 2 --output outputs/bao-smoke
python reproduce_bao.py replay --workers 4 --output outputs/bao-full
python plot_bao_comparison.py
```

`python plot_controller_comparison.py` draws the earlier controller comparison,
including the fixed UtilityQueue/ShortfallQueue controls.
`plot_bao_comparison.py` preserves the earlier Bao comparison figure.

## Information-matched controls

These controls separate inference-queue information, quality prediction,
and observed radio state. The 2,880 trajectories use the same 225 task pairs,
20 traffic seeds, 12 conditions, physical executor, and achieved cost. They add
no independent model responses.

`CARROT-Queue` and `CARROT-Radio` combine CARROT's frozen paired quality predictions
with this project's serving score. `RouteLLM-score` uses a separate adapter:
a service-specific one-dimensional KNN converts transferred preference scores
to paired mean qualities. Its neighbor counts (128/32/128) are selected using
five-fold quality MSE on the old 660 calibration tasks, without evaluation labels.
Numeric fitting inputs and recorded extra prediction times are included. These
adapters use this project's routing score rather than the native CARROT or
RouteLLM selection rules.

Within each predictor, `Intrinsic` excludes waiting and the entire workload
penalty; `Static` retains the second-moment penalty but sets waits to zero;
`Queue` uses the original complete serving score. Relative to Intrinsic, Mean
and CARROT Queue costs decrease by 2.49% and 3.39% across all 12 conditions.
Removing only waiting while retaining a route-dependent static penalty is a
different control because part of the workload score remains in the baseline.

The release-aware correction is
`max(max(D, wait) + payload/capacity - wait - duration, 0)` for positive payload,
and zero for zero payload. It prevents the fluid reference from transmitting
before generation starts. It is an own-delay surrogate with constant-rate,
uniform-fluid assumptions. It models the new request's delay, without pricing its
effect on already queued requests. The earlier radio correction and its results
remain in the supplementary inputs.

Mean's release-aware radio effect is a 0.43% cost reduction overall and 3.16%
at 10 kHz; CARROT instead increases by 0.14% overall and 0.59% at 10 kHz.
Mean-Radio's advantage over CARROT-Radio varies across conditions, and
distributional calibration costs more than the matched mean on this bank.
Neither of the two radio candidates passed its predeclared screen. These controls
use the development bank; the separate new-task study is documented in
[confirmation](../confirmation/README.md).

```sh
python information_controls.py verify
python information_controls.py summary
python plot_information.py
# All 12 new controls in two representative conditions, one seed:
python information_controls.py replay --conditions primary bandwidth_10000 --seeds 23924000 --output outputs/info-smoke
# All 2,880 trajectories, with route hashes and every accounting metric checked:
python information_controls.py replay --conditions all --workers 8 --output outputs/info-full
```

`qbr/information_controls.py` contains the controls, numeric adapter-fit check,
and all-condition summaries. `data/information_manifest.json` records the design,
source provenance, failed screens, and numeric-input hashes. The replay
uses frozen prediction costs; controller and scheduler CPU time is excluded.

`data/information_concurrency.csv` contains all 36 cells of a separate actual
endpoint probe (144 responses, 12 reused tasks, widths 1/2/4, two repetitions).
Four local calls took 91.0–99.8% of serial batch time across the three services.
This endpoint therefore provides little parallel speed-up. The four-replica
replay represents independent servers. The probe used a different cap and no
retries; its purpose was to measure endpoint concurrency. The CSV's
independent-replica columns are calculated counterfactuals. Raw text and provider
logs are excluded.

## Reproduce

Python 3.12 is recommended. Run from the repository root:

```sh
python -m pip install -r requirements.txt
python reproduce.py verify
python reproduce.py summary
python reproduce.py figures
python plot_strengthening.py
python information_controls.py verify
python information_controls.py summary
python plot_information.py
```

`verify` checks input hashes, all experiment records, target-integrated mixture
coefficients, and the analytic integrals against numerical quadrature.
`summary` writes numerical results to `outputs/summary.json`.
`figures` redraws the original baseline panels as `outputs/results.pdf` and `.png`.
`plot_strengthening.py` draws the earlier manuscript figure, including matched
controls and task sensitivity, as `outputs/results_comparison.pdf` and `.png`.
These commands use archived observations; they do not rerun the simulator.
`plot_information.py` draws the information-matched comparison, giving external
predictors the same serving and radio information. For the current manuscript
figure, use `plot_information_value.py` as described in the study guide.

To execute the measured-response wireless replay:

```sh
# Quick check: all five methods, one traffic seed, primary condition.
python reproduce.py replay --seeds 23924000 --workers 2 --output outputs/smoke

# Main experiment: 12 conditions × 20 seeds × 5 methods.
python reproduce.py replay --conditions all --workers 4 --output outputs/full

# PF scheduler sensitivity: 20 seeds × 3 internal methods.
python reproduce.py replay --scheduler pf --methods service prompt calibrated --workers 2 --output outputs/pf
```

Each replay recomputes arrivals, channel evolution, inference queues, routing,
stream delivery, and achieved costs. It checks every routing decision against
the archived route hash and seven metrics against the archived values (absolute
tolerance `1e-8`). An existing replay output is never overwritten: use a new
`--output` directory for a repeated run. No API key or GPU is required.

## Methods and files

| CLI name | Method |
| --- | --- |
| `service` | Service-level quality distributions with the workload score |
| `prompt` | Prompt-conditioned quality distributions with the same score |
| `calibrated` | Target-integrated mixture of prompt and service distributions |
| `carrot` | CARROT quality/cost selection using frozen predictions |
| `routellm` | RouteLLM-BERT selection using frozen predictions |

`qbr/quality.py` implements the prompt quality model and mixture fitting.
`qbr/policy.py` implements routing and workload observation. `qbr/radio.py` and
`qbr/replay.py` implement the wireless delivery simulation and accounting.

`data/` contains measured response sizes, chunk times, quality scores, charges,
termination statuses, numerical serving statistics, out-of-fold calibration
weights, fixed predictor outputs/times, and archived evaluation results.
`data/config.json` records all 12 conditions, 20 seeds, model identifiers, and source
hashes. `data/manifest.json` records SHA-256 hashes of the numerical input files.
The internal service label `dialogue` denotes the instruction-following tasks.

## Evaluation scope

The evaluation uses 225 paired tasks (75 each from IFBench, XSum, and MBPP),
450 measured responses, and 1,400 requests per trajectory, including 200 warmup
requests. Truncated responses remain in the evaluation. Each condition reuses
the same response bank. The 60 PF trajectories are a separate sensitivity check.
Traffic-seed intervals measure variation conditional on this fixed task bank.
Concurrent inference is simulated from individually recorded streams. The
separate endpoint probe above measures the effect of simultaneous calls.

This release reproduces calibration from out-of-fold weights and the downstream
wireless replay. Prompt text, response text, third-party model weights, and raw
provider logs are not redistributed. Predictor outputs and measured prediction
times are supplied, so the replay neither retrains the external encoders nor
remeasures inference overhead. `PromptQuality.fit` is included for training on
user-supplied paired data; reproducing the original text-based training requires
the original benchmark inputs. The recorded cloud model name is a mutable alias.

The external comparison was retrospective; the internal comparison was fixed
before response collection. Here CARROT and RouteLLM retain their native
selection rules without the workload score. The resulting differences measure
whole-router performance. The calibrated method's average cost reduction across
the 12 conditions is 3.26% versus CARROT and 8.52% versus RouteLLM-BERT;
the additional reduction versus the internal prompt method is 0.38%.
CARROT has lower quality shortfall and better tail latency in this evaluation.

## External implementations

- [CARROT](https://github.com/somerstep/CARROT), source revision
  `3e6acff6aecf4cbcb8f31a118d04c799c2ea1655`.
- [RouteLLM](https://github.com/lm-sys/RouteLLM), source revision
  `0b64fdafe049e596a3f5657c219329f24af24198`.

Their code and model weights are not bundled here. See their repositories for
training and license information. This repository's code is MIT licensed.

## Matched controls and task sensitivity

The supplementary study adds three controls. `mean` uses the shortfall of the
mean of the **same calibrated quality distribution**, with the same serving
score and measured prediction delay. `radio_mean` and `radio_calibrated` add
the same causal radio correction, computed from visible queued bits, observed
capacities (including the configured CSI delay), and calibration mean payloads.
The controllers see only available channel and backlog observations and
calibration statistics. The radio term approximates delay through a fluid
workload model; routing and scheduling are not jointly optimized.

```sh
python -m unittest test_strengthening -v
python strengthening_summary.py
python plot_strengthening.py
python benefit_diagnostic.py

# Rerun all 720 supplementary controls and all 14,400 task-bank trajectories.
python strengthen.py controls --workers 8
python strengthen.py bootstrap --workers 8
python strengthening_summary.py --rerun-dir outputs/strengthening
```

The summary and plot commands use the published supplementary archive unless
`--rerun-dir` is given to the summary command. A complete rerun is checked for
identical route hashes and numerical metrics within absolute tolerance `1e-8`;
machine-dependent controller wall times are excluded from that comparison.
Bootstrap batches can be split with `--start` and `--stop` (exclusive), using
disjoint ranges within 0–200 and one shared output directory.

`data/strengthening_plan.json` records the design and frozen source hashes.
It was fixed locally before the supplementary runs, after the original
task-bank evaluation. All 200 stratified
resamples retain complete edge/cloud task pairs and rerun the actual queues
and radio delivery. Each bank is shared across all 12 conditions and Mean,
Prompt, and Calibrated, using traffic seeds 23924000 and 23924001. Calibration,
predictors, target/channel paths, and recorded per-request prediction times
remain fixed. Percentile ranges are descriptive task-composition sensitivity
intervals conditional on those two traffic paths.

The full quality distribution costs more than its matched mean. Across the 12
conditions, mean cost is
5.6935 s for Calibrated, 5.6621 s for Mean, 5.6722 s for Radio-Calibrated, and
5.6374 s for Radio-Mean. The calibrated distribution has smaller measured
target-integrated benefit error, but this does not guarantee a better queue
trajectory or network cost. This explains why the earlier whole-router
improvements cannot be attributed to distributional calibration alone. Run the
summary for paired intervals for all comparisons.

## Fixed utility and shortfall queue controls

This retrospective comparison adds 480 trajectories (two controls × all
12 conditions × 20 paired traffic seeds) and retains the archived 240
CARROT-Queue trajectories. CARROT-Queue combines this project's serving controller
with CARROT's frozen paired quality predictions.
The controls share that predictor, its charged prediction time, causal serving
state, response actions, wireless scheduler and original achieved cost.

- `utility_queue`: `wait + mean_RPC + 1000*mean_fee - 12*predicted_mean_quality`.
- `shortfall_queue`: `wait + mean_RPC + 1000*mean_fee + 12*max(target - predicted_mean_quality, 0)`.
- Archived `carrot_queue`: ShortfallQueue plus
  `0.5*(wait*mean_RPC + second_RPC_moment/2)`.

UtilityQueue implements its rule by adding the route-independent constant 12.
Its scoring target is one; achieved cost retains each original request's target.
Exact route ties select cloud. The fixed coefficients were not tuned
in this comparison. The utility rule follows Eq. (5) of [Patel et al., Beyond Accuracy and Cost: Latency-Aware LLM Query Routing for Dynamic Workloads](https://arxiv.org/html/2607.18253),
using predicted full-RPC latency in place of the original SFS/TTFT setting.
No external optimum is fitted. Controller and scheduler CPU time is omitted.

```sh
python -m unittest test_utility_controls -v
python utility_controls.py --verify
python utility_controls.py --summary --output outputs/utility-summary.json
python utility_controls.py --rerun --condition primary --seed 23924000 --method utility_queue
python utility_controls.py --rerun --condition bandwidth_10000 --seed 23924000 --method shortfall_queue
```

The summary includes all conditions, six performance components, and paired
seed contrasts. It reports the primary condition, an equal-weight average over
12 conditions, and a sensitivity average excluding load 0.8. The 12-condition
average remains the main comparison. Intervals use 20 paired seed means and
Student's t with 19 degrees of freedom, conditional on the same 225 task pairs.

Across all 12 conditions, CARROT-Queue costs 5.633704 versus 5.703680 for
UtilityQueue and 5.675040 for ShortfallQueue, reductions of 1.22685% and
0.72839%. CQ-minus-control paired 95% intervals are
[-0.111857, -0.028095] and [-0.075453, -0.007220] seconds, respectively.
CQ's quality is 0.48436 percentage points lower than UtilityQueue's and its
p95 latency is 0.16292 seconds higher. Relative to ShortfallQueue, CQ's mean
latency is 0.02929 seconds higher and p95 is 0.18413 seconds higher. Both
10-kHz and load-0.8 individual-condition cost intervals include zero.

These controls compare score components on the development bank.
`data/utility_results.json.gz` contains numerical records; request traces,
prompt/response text, and provider logs are excluded. The optional replay checks
routing hashes, achieved metrics, and terminal counts without making model calls.
