# Neural ODEs: damped pendulum, Neural ODE against an LSTM baseline

## Contents
- `neural-ode-notebook.ipynb`: data simulation, Neural ODE, LSTM baseline, extrapolation, noise and mitigation experiments (Kaggle, CPU)
- `results/`: `node_trajectories.png`, `node_vs_lstm.png`, `node_longer_training.png`, `node_noise.png`
- `LOG.md`: timestamped experiment log

## Goal and system
Learn the dynamics of a damped pendulum from simulated trajectories with a Neural ODE, and compare it with a
discrete-time LSTM baseline on (1) unseen damping values, (2) extrapolation in time and (3) noisy training data.
System: theta'' + b theta' + (g/L) sin(theta) = 0 with g = 9.81 and L = 1, written as a first-order system for the
state z = (theta, omega) with omega = theta'.

## Data
Simulated with `scipy.integrate.solve_ivp` (DOP853, rtol 1e-10, atol 1e-12), sampled every 0.05 s up to 30 s,
every trajectory starting at rest. 24 damping values drawn uniformly from 0.05 to 0.40 (seed 42) and six start
angles each (0.6, 1.0, 1.4, 1.8, 2.2, 2.6 rad).

**Grouped split by damping value:** 17 damping values for training (102 trajectories) and 7 for testing (42
trajectories); no damping value appears in both (checked by an assertion). The training window is 0 to 10 s
(200 steps) and the extrapolation target is 0 to 30 s (600 steps). The training damping range is 0.083 to 0.390 and
the test range 0.072 to 0.391, so one test value (0.072) lies below the training range and one (0.391) above it by
0.001; this is a consequence of the random split.

**How trajectories change with the parameters** (`results/node_trajectories.png`):
- Larger damping makes the swings die out faster. From a 2.0 rad start, b = 0.05 still swings by about 0.9 rad at
  30 s (theory: 2 x exp(-b t / 2) = 0.94), while b = 0.40 is nearly flat by 10 s. The curves also drift out of
  phase, because large swings are slower.
- Larger start angles give larger and slower swings: the first half-swing takes 1.05 s from 0.6 rad and 1.70 s
  from 2.6 rad (b = 0.05), against 1.00 s from the small-angle formula, which shows the nonlinearity of sin(theta).
  The amplitudes decay at a similar exponential rate.
- The phase portrait is a spiral into the origin: tight and fast for high damping, many loops for low damping.
- Late-time amplitudes of the test trajectories are small (median peak |theta| 0.373 near 10 s and 0.029 near 30 s),
  so a model that predicts zero already has a small late-time error. A zero predictor is therefore used as a
  reference throughout.

## Models
- **Neural ODE.** A small network defines the vector field dz/dt = f(z, b): inputs are the angle, the angular velocity
  divided by 3 and 5 times the damping; two hidden layers of 64 units with tanh; the output is multiplied by 5.
  4,546 parameters. A hand-written fixed-step RK4 solver (dt = 0.05 s) integrates from the true initial state
  (no `torchdiffeq`, no adjoint method). With the true pendulum equations this solver reproduces the reference
  solution to 3.7e-4 over 30 s on all test trajectories, so the solver error is negligible and any large error comes
  from the learned field. The loss is the mean squared error between the rolled-out and the observed trajectory,
  backpropagated through the solver steps. Training uses a curriculum on the window (40, then 100, then 200 steps,
  at 30%, 30% and 40% of the epochs), Adam with learning rate 3e-3 and cosine decay to 3e-4, gradient clipping at
  5, and the full batch of 102 trajectories.
- **LSTM baseline.** Two layers with hidden size 64 (51,074 parameters, about 11 times as many as the Neural ODE)
  and the same inputs (current state and damping). It predicts the change to the next state (divided by 0.25), is
  trained with teacher forcing on the true states of the 0 to 10 s window (same optimiser settings), and is rolled
  out autoregressively to 30 s from the true initial state, feeding back its own predictions and carrying its
  hidden state.
- Both models start the test rollout from the true initial state. CPU only; seed 42.

## Hypotheses (stated before running)
- Both models fit the training window well on unseen damping values.
- The Neural ODE extrapolates better than the LSTM beyond 10 s, because it applies one learned law at every time
  while the LSTM repeats a learned one-step map and compounds its errors.
- Noise degrades both models; I had no strong prior about which suffers more.

## Results: extrapolation on unseen damping values
42 test trajectories; RMSE over states (angle in rad, angular velocity in rad/s).

| Model | 0 to 10 s | 10 to 30 s | 10 to 30 s, low damping (24 traj.) | 10 to 30 s, high damping (18 traj.) |
|---|---|---|---|---|
| Zero predictor (reference) | 1.544 | 0.661 | 0.861 | 0.178 |
| Neural ODE, 300 epochs | 0.381 | 0.672 | 0.883 | 0.120 |
| LSTM, 300 epochs | 0.612 | 0.802 | 1.059 | 0.064 |
| Neural ODE, 1200 epochs | **0.179** | **0.289** | **0.381** | 0.032 |
| LSTM, 1200 epochs | 0.218 | 0.389 | 0.514 | **0.014** |

("Low damping" is the 4 smallest test damping values, b <= 0.275, and "high damping" the 3 largest.)

**Training budget.** The 300-epoch models were undertrained: the Neural ODE's training loss was still falling
(0.073 at the end) and its RMSE on its own training trajectories was 0.289. I retrained both models for 1,200 epochs
(376 s for the Neural ODE, 179 s for the LSTM; training loss 0.0267; LSTM one-step loss 0.00039 against 0.00395).
This choice was made after seeing the 300-epoch test results, motivated by the training loss, so it is not a blind
tuning step, and both runs are reported. Longer training cut the extrapolation error by 57% (Neural ODE) and 52%
(LSTM).

**300 epochs.** The Neural ODE was better than the LSTM overall (0.381 against 0.612 in the window, 0.672 against
0.802 beyond it) but in extrapolation it was no better than predicting zero (0.672 against 0.661), and for the low
damping half both models were worse than zero. The worst absolute errors were 5.2 rad (Neural ODE) and 6.7 rad
(LSTM): phase-flipped predictions on large, lightly damped swings.

**1200 epochs.** The Neural ODE has the lower overall RMSE: 0.289 against 0.389 beyond 10 s (56% and 41% below
the zero predictor) and 0.179 against 0.218 in the window. But this aggregate is dominated by the two lowest damping
values (see below); the LSTM is better on the other five damping values and on the per-trajectory relative error.

**By damping value, RMSE over 10 to 30 s (6 start angles each):**

| b | Truth RMS = zero predictor | Neural ODE, 300 | LSTM, 300 | Neural ODE, 1200 | LSTM, 1200 |
|---|---|---|---|---|---|
| 0.072 (below the training range) | 1.239 | 1.333 | 1.715 | **0.616** | 0.920 |
| 0.095 | 1.024 | 0.888 | 1.218 | **0.403** | 0.455 |
| 0.174 | 0.553 | 0.685 | 0.242 | 0.189 | **0.047** |
| 0.275 | 0.270 | 0.290 | 0.039 | 0.054 | **0.018** |
| 0.321 | 0.200 | 0.143 | 0.058 | 0.025 | **0.015** |
| 0.325 | 0.195 | 0.132 | 0.060 | 0.024 | **0.015** |
| 0.391 (above the training range by 0.001) | 0.129 | 0.073 | 0.072 | 0.044 | **0.012** |

**Per-trajectory relative error** (the model's error divided by the size of the true signal in the window; 1.0
means no better than predicting zero):

| Model | 0 to 10 s median / mean | 10 to 30 s median / mean | Trajectories worse than zero in 10 to 30 s |
|---|---|---|---|
| Neural ODE, 300 epochs | 0.19 / 0.22 | 0.90 / 1.01 | 45% |
| LSTM, 300 epochs | 0.16 / 0.20 | 0.36 / 0.52 | 12% |
| Neural ODE, 1200 epochs | 0.09 / 0.10 | 0.27 / 0.35 | 7% |
| LSTM, 1200 epochs | 0.05 / 0.07 | 0.11 / 0.19 | 2% |

At 300 epochs the answer to "which model extrapolates better" depended on the metric: overall RMSE favoured the
Neural ODE and the per-trajectory relative error favoured the LSTM. At 1200 epochs the same split remains. The
Neural ODE has the lower RMSE on the two lowest damping values (0.072 and 0.095, where the swing persists) and the
LSTM on the other five. 93% of the Neural ODE's squared error and 99.7% of the LSTM's beyond 10 s come from those
two values; excluding them, the 10 to 30 s RMSE is 0.091 (Neural ODE), 0.025 (LSTM) and 0.307 (zero predictor).
The failure on lightly damped trajectories is not only the one value below the training range: b = 0.095 lies
inside it and was also hard at 300 epochs.

**Figures** (`results/node_vs_lstm.png`, `results/node_longer_training.png`; read from the plots). On the lightly
damped test trajectory (b = 0.072, start 0.6 rad) the Neural ODE keeps the right period and phase but does not
damp enough, so its amplitude stays too large at late times, while the LSTM follows the decay (after 300
epochs it decays too fast). On the highly damped trajectory (b = 0.391, start 2.6 rad) all three curves overlap.
In the error-against-time plot the LSTM is more accurate in the first few seconds and the Neural ODE after about
6 s, and by 30 s all errors approach the zero predictor's because the true signal has mostly decayed.

## Results: noise
The training states (including the initial state the models see) were perturbed with Gaussian noise of standard
deviation sigma; both models were retrained from scratch (300 epochs, seed 42) and evaluated on clean test
trajectories with unseen damping values. RMSE 0 to 10 s / 10 to 30 s (`results/node_noise.png`):

| sigma | Neural ODE | LSTM |
|---|---|---|
| 0 | 0.381 / 0.672 | 0.612 / 0.802 |
| 0.02 | 0.400 / 0.705 | 0.633 / 0.810 |
| 0.05 | 0.350 / 0.537 | 0.719 / 0.821 |
| 0.10 | 0.376 / 0.610 | 0.950 / 0.840 |

The LSTM's in-window error rises steadily (55% worse at sigma 0.1). The Neural ODE's stays within about 8% of its
clean value, and it is ahead at every noise level (at sigma 0.1: 60% lower in the window and 27% lower in
extrapolation). The Neural ODE's extrapolation error is not monotonic in sigma; I attribute this to run-to-run
variation (see the seed spread below), so I do not claim that noise helps it. A possible reason for the LSTM's drop
(untested): it learns from differences of consecutive noisy states, whose noise (about 1.4 sigma) is comparable
to the real per-step change (up to about 0.3 rad in angle and 0.5 rad/s in velocity), while the Neural ODE matches
whole trajectories with a smooth vector field that cannot fit independent noise at every point.

**What helps the Neural ODE under noise** (sigma 0.1, 300 epochs, RMSE 0 to 10 s / 10 to 30 s):

| Variant | Parameters | 0 to 10 s | 10 to 30 s |
|---|---|---|---|
| Baseline, seeds 42 / 43 / 44 | 4,546 | 0.376 / 0.395 / 0.394 | 0.610 / 0.523 / 0.606 |
| Weight decay 1e-3 | 4,546 | 0.372 | 0.569 |
| Hidden size 16 | 370 | 0.471 | 0.654 |
| Hidden size 128 | 17,282 | 0.415 | 0.642 |

The baseline's seed-to-seed spread in extrapolation is 0.087 (mean 0.580), and 0.019 in the window. Weight decay lies
inside the spread (no evidence of a benefit). The smaller network is clearly worse in the window and the larger
one marginally worse, and both are above the baseline's extrapolation range. So none of the tried changes helped
beyond seed noise; training longer was the bigger lever (-57% extrapolation error on clean data). Caveat: these
variants were trained at the undertrained 300-epoch budget with one seed each, so a modest effect could be hidden.
The solver was not varied: with fixed-step RK4 at 0.05 s its error (3.7e-4) is far below the noise level, so a more
accurate solver cannot be expected to help.

## Analysis
- **Overall RMSE and the typical trajectory disagree.** With enough training the Neural ODE has the lower overall
  extrapolation RMSE (0.289 against 0.389 beyond 10 s) with 11 times fewer parameters, consistent with the
  expectation that one learned vector field generalises across time better than an iterated one-step map. But that
  aggregate is dominated (93% and 99.7% of the squared error) by the two most lightly damped test values, one of
  which lies below the training range; on the other five damping values and in the per-trajectory relative error
  (median 0.27 against 0.11) the LSTM is better. So the Neural ODE's advantage is supported on the hardest, lightly
  damped trajectories and in the aggregate, not for the typical trajectory.
- **The results are sensitive to the training budget and to the metric.** At 300 epochs the Neural ODE was no
  better than predicting zero in extrapolation and the winner depended on the metric; at 1200 epochs both models
  beat zero on almost every trajectory. The Neural ODE is noticeably more noise-robust than the LSTM inside the
  training window.
- **Both models struggle on lightly damped, large swings**, where the swing period depends on the amplitude, so a
  small error in the learned law makes the prediction drift out of phase over many swings.

## Limitations
- One seed for most runs; the seed spread at sigma 0.1 is 0.087 in extrapolation, comparable to several of the
  differences discussed.
- The noise experiment used the undertrained 300-epoch budget.
- The 1200-epoch choice was made after seeing the 300-epoch test results.
- One system and a 20 s extrapolation horizon; one test damping value lies below the training range, and the two
  lowest damping values account for most of the aggregate error (12 of the 42 test trajectories).
- The baseline is a plain teacher-forced LSTM (no scheduled sampling or noise injection), so a stronger baseline
  might narrow the gap; only fixed-step RK4 was used (no adaptive solver or adjoint method).
- Synthetic, regularly sampled, fully observed data; irregular sampling, where Neural ODEs are expected to have a
  larger advantage, was not tested.

## Reproduce
Kaggle, CPU only. Python with NumPy, SciPy, PyTorch, pandas and matplotlib; seed 42. Run the cells of
`notebook.ipynb` top to bottom (about 30 minutes: Neural ODE 300 epochs about 100 s and 1200 epochs about 380 s;
LSTM 44 s and 179 s; the noise experiment about 7 minutes; the mitigation experiment about 8 minutes).
