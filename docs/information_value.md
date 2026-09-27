# Value of quality, queue, and radio information

[Back to overview](../README.md)

This guide describes the development experiments for *Quality, Queue, and Radio
Information for Wireless Edge–Cloud LLM Routing*. The initial study contains 9,500
replay paths and 19,200 task-bank resampling paths. Subsequent payload, diagnostic,
and static-split experiments are described below.

These experiments reuse 225 recorded task pairs and 450 responses, including 62
truncations. They share the wireless replay, downlink scheduler, and achieved cost
`L + beta*shortfall + 1000*fees`; they require no new provider calls or model fitting.
The plan in `data/information_value_plan.json` was fixed before the initial runs.
The comparisons remain retrospective because the task bank and `kappa = 0.5` had
already been used during development.

## Controllers

The queue-policy comparisons use CARROT's frozen paired-quality predictions,
their charged prediction runtime, and the same causal wait estimator. The table
specifies changes to the score; the oracle and fixed-route controls are exceptions
to the shared predictor:
`w + mu + gamma*fee + kappa*(w*mu + second/2) + beta*shortfall`.

| Name | Definition |
| --- | --- |
| `intrinsic` | no waits, `kappa = 0` (queue-blind) |
| `utility` | waits, `kappa = 0`, linear utility `-beta*Q` (target 1) |
| `shortfall` | waits, `kappa = 0`, `beta*max(q - Q, 0)` |
| `cq` | CARROT–Queue, `kappa = 0.5` |
| `fpi` | `kappa_m = rho / ((1 - rho) * mbar_m)` from the M/G/1-FIFO value function |
| `*_release` | adds the release-aware radio term |
| `oracle_cq` | realized qualities replace predictions within the same routing rule |
| `always_edge`, `always_cloud` | fixed route, no predictor charge |
| `kX`, `method@bX` | fixed `kappa = X`; rule quality weight `X` |

The oracle is a clairvoyant reference: it knows the outcome of the single recorded
response for each route, including response-specific randomness. Its improvement
need not be attainable by a predictor.

For a stationary M/G/1-FIFO reference slot, the workload term is the
first-policy-iteration admission cost
`lambda/(1 - rho) * E[((w + S)^2 - w^2)/2]`. This follows from
Proposition 1 of Hyytiä, Penttinen, and Aalto (EJOR 217(2), 2012). `fpi` sets this
coefficient from the configured load with proportional splitting. The reference
model does not fully describe the state-dependent, multi-slot replay.

## Initial experiments

| Experiment | Paths | Design |
| --- | --- | --- |
| `main` | 1,680 | 7 controllers × 12 conditions × 20 seeds |
| `beta` | 5,760 | rule and cost weight `beta` in {4, 8, 12, 16, 24, 36} × 4 controllers |
| `kappa` | 960 | `kappa` in {0.125, 0.25, 1, 2} |
| `load` | 800 | `rho` in {0.35, 0.5, 0.65, 0.8, 0.9} × 8 controllers, primary radio |
| `serial` | 300 | one edge slot, `rho` in {0.5, 0.65, 0.8} × 5 controllers |
| `boot` | 19,200 | 200 task banks × 12 conditions × 2 seeds × 4 controllers |

Rerunning `cq`, `cq_release`, `utility@b12`, and `shortfall@b12` reproduces
every archived routing-choice hash on all 12 × 20 paths.

## Results with four replicas (12 conditions, 20 seeds)

| Router | Cost (s) | Delay | p95 | Quality % | Cloud % |
| --- | --- | --- | --- | --- | --- |
| Always edge | 9.447 | 6.04 | 13.55 | 64.6 | 0.0 |
| Always cloud (overloaded) | 135.646 | 132.11 | 217.48 | 70.0 | 100.0 |
| Intrinsic | 5.832 | 2.36 | 5.91 | 64.7 | 13.9 |
| Utility–Queue | 5.704 | 2.29 | 5.66 | 66.2 | 27.0 |
| Shortfall–Queue | 5.675 | 2.21 | 5.64 | 65.1 | 19.5 |
| CARROT–Queue | 5.634 | 2.24 | 5.82 | 65.7 | 15.4 |
| FPI–Queue | 5.617 | 2.21 | 5.70 | 65.8 | 17.8 |
| CARROT–Radio | 5.641 | 2.23 | 5.79 | 65.6 | 16.2 |
| FPI–Radio | 5.634 | 2.21 | 5.69 | 65.7 | 18.8 |
| Oracle–Queue | 4.986 | 2.30 | 6.12 | 73.9 | 19.1 |

- **Queue state** matters more as load grows. At `rho = 0.35`, the queue-aware rules range from 1.8% below to 1.1% above Intrinsic. At `rho = 0.9`, all four are 15.3–16.2% below it. With one serial edge slot, Intrinsic overloads the edge (93.7–375.6 s).
- **Quality information.** The oracle lowers cost by 11.5%, CI [−0.668, −0.627] s.
- **Radio state** changes cost by less than 0.5% for nominal airtime load ≤ 0.4 (`data/airtime_load.json`). At 1.19 (10 kHz), its sign depends on the predictor.

### Sensitivity to weights, load, and task composition

- **Weight robustness.** CARROT–Queue minus Utility–Queue, with the same `beta` in the rule and the cost:

  | `beta` | Difference (s) | CI |
  | --- | --- | --- |
  | 4 | −0.010 | [−0.023, 0.003] |
  | 8 | +0.118 | [0.089, 0.148] |
  | 12 | −0.070 | [−0.112, −0.028] |
  | 16 | +0.065 | [0.027, 0.103] |
  | 24 | +0.147 | [0.084, 0.209] |
  | 36 | +0.163 | [0.085, 0.241] |

- **Hindsight-tuned baseline.** Utility with rule weight 8, evaluated at `beta = 12`, costs 5.604 s. CARROT–Queue minus this is +0.030 s, CI [0.002, 0.058].
- **Load.** CARROT–Queue minus Utility–Queue is −0.157 s at `rho = 0.35` and +0.008 s at `rho = 0.9`, CI [−0.034, 0.050]. With a serial edge slot, this difference is not significant at any load.
- **Task banks.** The central 95% range of CARROT–Queue minus Utility–Queue is [−0.372, 0.197] s, and 69.5% of the 200 banks are negative.
- **Declared FPI gate: failed.** Over 12 conditions, FPI minus CARROT–Queue is −0.016 s, CI [−0.038, 0.005]. FPI is worse than `kappa = 0.5` at `rho` 0.35, 0.5, and 0.9.

## Payload sweep and one edge server

The payload and one-server experiment plan was fixed before these runs
(`extension_v1_5` in `data/information_value_plan.json`). The 2,900 paths are in
`data/information_value_extension.json.gz`; use `python information_value.py --extension-summary`
to summarize them.
At payload scale 1 all five methods reproduce the archived choice hashes.

**Payload sweep** (100 kHz, rho 0.65; response bytes scaled, timing/quality/fees unchanged, a stand-in for larger outputs):

| Scale | rho_air | Mean: Radio − Queue | CARROT: Radio − Queue | Queue − Intrinsic |
| --- | --- | --- | --- | --- |
| 1 | 0.12 | −0.013 s [−0.023, −0.004] | −0.005 s [−0.015, 0.005] | −0.179 s |
| 2 | 0.24 | −0.014 s [−0.028, −0.001] | +0.006 s [−0.007, 0.019] | −0.181 s |
| 4 | 0.48 | −0.025 s [−0.046, −0.003] | +0.034 s [0.014, 0.053] | −0.185 s |
| 8 | 0.95 | −0.118 s [−0.155, −0.082] | +0.046 s [0.007, 0.084] | −0.210 s |
| 12 | 1.43 | −0.413 s [−0.502, −0.324] | −0.018 s [−0.100, 0.064] | −0.189 s |

At fixed power spectral density, scaling payload and inversely scaling bandwidth
give equivalent airtime scaling. Their agreement at equal rho_air is a consistency
check (manuscript Fig. 2(c)).

**Serial edge, all 12 conditions** (one edge slot, arrivals normalised by the resulting capacity):

| Router | Cost (s) |
| --- | --- |
| Intrinsic (queue-blind) | 251.559 (edge overloaded) |
| Always cloud | 8.016 |
| Utility–Queue | 6.220 |
| Shortfall–Queue | 6.224 |
| CARROT–Queue | 6.208 |
| FPI–Queue | 6.160 |
| CARROT–Radio | 6.214 |
| Mean–Queue / Mean–Radio | 6.235 / 6.234 |
| Oracle–Queue | 5.716 |

FPI − CARROT-Queue = −0.048 s [−0.077, −0.020]; FPI − Utility = −0.060 s [−0.099, −0.020];
CARROT-Queue − Utility = −0.011 s [−0.045, 0.023]. The significant one-server FPI
difference does not change the failed criterion defined for four replicas above.

## Radio decision diagnostic

The diagnostic plan (`radio_diagnostic_v1_6` in `data/information_value_plan.json`)
was fixed before the runs. For each predictor, setting, and seed, Queue and Radio
replay the same traffic. These 400 diagnostic paths reproduce the choice hashes
of the initial and payload experiments. `data/information_value_radio_diagnostic.json` holds one row per pair;
`python information_value.py --diagnostic-summary` recomputes the table. Flips are defined against the Queue
path of the same seed; trajectories diverge after the first flip, so this is a paired descriptive
decomposition, not a causal per-decision effect.

At 100 kHz with 12x payload (rho_air = 1.43), per request:

| Predictor | Routes changed | Toward cloud | Predicted quality change | Realized quality change | Delay change | Total |
| --- | --- | --- | --- | --- | --- | --- |
| CARROT | 16.9% | 65.5% | −0.137 s | +0.043 s | −0.104 s | −0.018 s (n.s.) |
| Mean | 19.3% | 65.0% | −0.144 s | −0.081 s | −0.373 s | −0.413 s |

Both controllers move similar requests in the same direction and expect a quality gain. The gain
materializes for the Mean predictor but reverses for CARROT, so the value of radio state depends on
predictor accuracy for the requests near the decision boundary. The 10 kHz setting shows the same pattern.


## Static splits, noisy oracles, and one-server sweeps

These 20,820 paths are stored in `data/information_value_review.json.gz`. Their
plan, `data/information_value_review_plan.json`, was fixed before the runs, after
the new-task confirmation. The analyses are exploratory and have no pass/fail
criterion. The measured edge server serves one request at a time: four concurrent
calls took 91.0–99.8% of their serial time. One edge server is therefore the primary
setting, and four edge slots represent independent replicas.

| Name | Definition |
| --- | --- |
| `static_prop` | no queue state; the cloud share equals the cloud share of service capacity; requests with the largest queue-free score difference go to the cloud (fixed random tie-break) |
| `static_pX` | the same with cloud share `X` in 0.05–0.90; "best in hindsight" is the lowest-cost share per scope |
| `noisyX` | CARROT–Queue with realized qualities plus fixed Gaussian error `X`, clipped to [0, 1]; `noisy0` reproduces `oracle_cq` exactly |

Averages over 12 conditions and 20 seeds (paired 95% t intervals):

| Setting | Best static share | Static cost | Queue-aware cost | Queue-aware minus best static |
| --- | --- | --- | --- | --- |
| One edge server | 0.75 | 6.764 s | 6.160–6.224 s | FPI −0.604 [−0.659, −0.549]; Utility −0.544 [−0.600, −0.488] |
| Four replicas | 0.20 | 5.759 s | 5.617–5.704 s | FPI −0.142 [−0.184, −0.100]; Utility −0.056 [−0.092, −0.019] |

With four replicas at load 0.35 the best static split beats every queue-aware rule except CARROT–Queue.

Choosing the static share in hindsight separately for each condition leaves the one-server static cost at 6.764 s
(queue-aware rules remain 8.0–8.9% cheaper) and lowers the four-replica static cost to 5.749 s (queue-aware rules
remain 0.8–2.3% cheaper; Utility −0.045 [−0.080, −0.011]). With one edge server,
FPI–Queue lowers the capacity-share split's cost from 6.766 s to 6.160 s (−9.0%).
Its workload coefficient comes from the M/G/1 policy-improvement reference
described above.

Route discrimination (mix-weighted correlation between predicted and realized route-quality differences):
CARROT 0.168, quantile-forest mean 0.214, RouteLLM score 0.132. Noisy oracles with sigma 0.1/0.2/0.3/0.5 correlate
0.91/0.78/0.60/0.49 and cost 5.006/5.042/5.077/5.297 s; `noisy0.5` has about CARROT's mean squared error (0.124
vs. 0.129) but costs 6% less than CARROT–Queue (5.634 s).

One edge server: CARROT–Queue minus Utility–Queue is −0.025, +0.036, −0.011 (n.s.), +0.052, +0.089, +0.140 s at
beta = 4, 8, 12, 16, 24, 36; the Mean-predictor radio correction is not significant at any payload scale, and the
CARROT radio correction raises cost by 0.090 s at 12x. The beta = 12 paths reproduce
the earlier one-server records exactly. The corresponding exploratory analysis on
the new-task bank is in `confirmation/exploratory_v18/`.

## Commands

```sh
python -m unittest test_information_value -v
python information_value.py --verify          # hashes, 52,420 records, full summary recomputation
python information_value.py --gate
python information_value.py --extension-summary
python information_value.py --diagnostic-summary
python information_value.py --review-summary
python information_value.py --rerun --exp static12 --method static_p0.75 --edge-slots 1 --condition users_32 --seed 23924007
python information_value.py --rerun --exp payload --method mean_release --payload 12 --seed 23924011
python information_value.py --rerun --exp beta --method utility@b8 --condition users_32 --seed 23924007
python information_value.py --rerun --exp boot --method fpi --replicate 57 --condition csi_0.3 --seed 23924001
python information_value.py --airtime
python information_value.py --replay main --workers 8 --output outputs/iv-main.jsonl   # about 10 min on 20 cores
python plot_information_value.py
```

## Limitations

- The comparisons are retrospective on one task bank, with one response per route.
- Quality is measured by task-specific proxies.
- Concurrent service is simulated. A local probe found little parallel speed-up, so one edge server is the primary setting and four slots model independent replicas.
- The static-split and noisy-oracle analyses were added after the new-task confirmation.
- Controller and scheduler CPU time is omitted for all methods.
