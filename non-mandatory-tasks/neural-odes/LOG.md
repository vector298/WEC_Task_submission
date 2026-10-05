# Experiment log: Neural ODEs

## 2026-10-05: data simulation and visualisation
- Hypothesis: larger damping makes the swings die out faster; larger start angles give larger and slower swings
(the sin term is nonlinear); a hand-written RK4 with dt 0.05 s should add negligible solver error.
Setup: damped pendulum (g = 9.81, L = 1), solve_ivp DOP853 (rtol 1e-10, atol 1e-12), dt 0.05 s to 30 s, start at
rest; 24 damping values uniform in 0.05 to 0.40, 6 start angles (0.6 to 2.6 rad); split by whole damping value.
- Result: training window 0 to 10 s (200 steps), horizon 30 s (600 steps); 17 train and 7 test damping values,
disjoint; train b range 0.083 to 0.390, test 0.072 to 0.391; 102 train and 42 test trajectories. First half-swing
at b = 0.05: 1.05 s (0.6 rad) and 1.70 s (2.6 rad), small-angle half period 1.00 s. Test peak |angle| median 0.373
near 10 s and 0.029 near 30 s (minimum 0.0024). RK4 against the reference over 30 s: worst error 5.7e-4.
- Observation: damping controls the decay as expected (b = 0.05 still swings about 1 rad at 30 s from 2 rad, b = 0.40
is flat by 10 s); big swings are slower; the phase portrait spirals into the origin. Late-time amplitudes are small,
so a zero predictor is a necessary reference. One test damping (0.072) lies below the training range.
Next: Neural ODE and LSTM.

## 2026-10-05: Neural ODE versus LSTM, first results (300 epochs)
- Hypothesis: both models fit the training window well on unseen damping values; the Neural ODE extrapolates
better beyond 10 s than the LSTM (one learned law against an iterated one-step map).
- Setup: Neural ODE: MLP vector field (inputs angle, angular velocity / 3, 5 x damping; 2 x 64 tanh; 4,546
parameters), hand-written RK4 (dt 0.05), backpropagation through the solver, 300 epochs, curriculum window (40,
100, 200 steps), Adam 3e-3 with cosine decay, clipping 5. Baseline: 2-layer LSTM (51,074 parameters) predicting
the next-state change, teacher-forced on the 0 to 10 s window, rolled out autoregressively. Same inputs for both;
CPU; seed 42; zero predictor as reference.
- Result: solver check 3.7e-4 (all test trajectories). Test RMSE 0 to 10 s / 10 to 30 s: zero 1.544 / 0.661; Neural
ODE 0.381 / 0.672; LSTM 0.612 / 0.802. 10 to 30 s for the 24 low-damping / 18 high-damping trajectories: zero
0.861 / 0.178; Neural ODE 0.883 / 0.120; LSTM 1.059 / 0.064. Worst absolute error 5.23 (Neural ODE), 6.73 (LSTM).
Training 101 s and 44 s; the Neural ODE loss was still falling (0.093 at epoch 251, 0.073 at epoch 300).
- Observation: the Neural ODE beats the LSTM overall (38% lower in the window, 16% lower beyond it) with 11 times
fewer parameters, but in extrapolation it is no better than predicting zero, and for low damping both models are
worse than zero; on high damping the LSTM is better. For b = 0.072 the Neural ODE keeps the phase but does not
damp enough. Both models are undertrained; the hypotheses are only partly supported.
- Next: noise experiment.

## 2026-10-05: noise experiment
- Hypothesis: noise degrades both models; no strong prior about which suffers more.
- Setup: Gaussian noise (sigma 0.02, 0.05, 0.1) added to every training state value (including the initial state),
both models retrained from scratch (300 epochs, seed 42), evaluated on clean test trajectories.
Result (RMSE 0 to 10 s / 10 to 30 s): Neural ODE 0.381 / 0.672 (sigma 0), 0.400 / 0.705, 0.350 / 0.537,
0.376 / 0.610; LSTM 0.612 / 0.802, 0.633 / 0.810, 0.719 / 0.821, 0.950 / 0.840. About 140 s per level.
- Observation: the LSTM's in-window error rises steadily (55% worse at sigma 0.1) while the Neural ODE's stays within
about 8% of its clean value; at sigma 0.1 the Neural ODE is 60% lower in the window and 27% lower beyond it. The
Neural ODE's extrapolation error is non-monotonic in sigma, which I attribute to run-to-run variation (one seed),
so I do not claim noise helps it. Possible reason for the LSTM's drop (untested): its targets are differences of
consecutive noisy states, with noise about 1.4 sigma, comparable to the real per-step change.
- Next: longer training and what helps the Neural ODE under noise.

## 2026-10-05: longer training and noise mitigation
- Hypothesis: the 300-epoch models are undertrained (the Neural ODE training loss was still falling), so longer
training should lower the test errors of both models; weight decay and a different network size might help the
Neural ODE under noise.
- Setup: both models retrained for 1200 epochs (same budget each; chosen after seeing the 300-epoch test results,
motivated by the training loss). At sigma 0.1, the Neural ODE retrained with weight decay 1e-3, hidden size 16 and
128, and two more seeds of the baseline (300 epochs each).
- Result: 1200 epochs: Neural ODE 376 s, training loss 0.0267 (300 epochs: 0.0727), RMSE on its training
trajectories 0.146 (0.289); LSTM 179 s, one-step loss 0.00039 (0.00395). Test RMSE 0 to 10 s / 10 to 30 s: Neural
ODE 0.179 / 0.289, LSTM 0.218 / 0.389; low / high damping halves in 10 to 30 s: Neural ODE 0.381 / 0.032, LSTM
0.514 / 0.014. Noise at sigma 0.1 (0 to 10 s / 10 to 30 s): baseline seeds 42, 43, 44: 0.376, 0.395, 0.394 / 0.610,
0.523, 0.606 (spread 0.087 in extrapolation); weight decay 0.372 / 0.569; hidden 16 0.471 / 0.654; hidden 128
0.415 / 0.642.
- Observation: longer training cut extrapolation error by 57% (Neural ODE) and 52% (LSTM); the earlier result that
the Neural ODE was no better than predicting zero came from undertraining. At 1200 epochs the Neural ODE is 56%
below the zero predictor and 26% below the LSTM in extrapolation. None of the noise mitigations helped beyond the
seed spread (weight decay inside it; the smaller network clearly worse in the window, the larger one marginally
worse); training longer was the bigger lever. Caveat: single seeds at the undertrained budget. The solver was not
varied because its error (3.7e-4) is far below the noise.
- Next: per-damping breakdown.

## 2026-10-05: breakdown by damping value and relative error
- Hypothesis: the Neural ODE's advantage should hold across damping values.
- Setup: per-damping RMSE (10 to 30 s) and per-trajectory relative error (model error divided by the size of the
true signal; 1.0 = no better than zero) for the 300-epoch and 1200-epoch models; no retraining.
- Result (RMSE 10 to 30 s, zero / Neural ODE 1200 / LSTM 1200): b 0.072: 1.239 / 0.616 / 0.920; 0.095: 1.024 / 0.403 /
0.455; 0.174: 0.553 / 0.189 / 0.047; 0.275: 0.270 / 0.054 / 0.018; 0.321: 0.200 / 0.025 / 0.015; 0.325: 0.195 /
0.024 / 0.015; 0.391: 0.129 / 0.044 / 0.012. Relative error 10 to 30 s: Neural ODE 300 epochs median 0.90 (45% of
trajectories worse than zero), LSTM 0.36 (12%); Neural ODE 1200 epochs median 0.27 (mean 0.35, 7% worse than
zero), LSTM 0.11 (mean 0.19, 2%). Share of the squared error from the two lowest dampings: 93% (Neural ODE) and
99.7% (LSTM); excluding them the 10 to 30 s RMSE is 0.091 (Neural ODE), 0.025 (LSTM), 0.307 (zero).
- Observation: the Neural ODE wins on the two lightly damped values (0.072, below the training range, and 0.095) and
in the aggregate RMSE; the LSTM wins on the other five values and per trajectory. The aggregate advantage is thus
a statement about lightly damped, large-swing trajectories. The hypothesis that the Neural ODE extrapolates
better is only partly supported.
- Next: README and commit.
