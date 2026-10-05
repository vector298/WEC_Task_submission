## 2026-10-05: Neural ODE versus LSTM on the damped pendulum (first results)

>> Setup: damped pendulum (g/L = 9.81, dt 0.05 s), 24 damping values in 0.05 to 0.40 split by whole value (17
train, 7 test, none shared), 6 start angles each (102 / 42 trajectories); training window 0 to 10 s, test
rollout 0 to 30 s from the true initial state. Neural ODE: MLP vector field (3 inputs: angle, angular velocity,
damping; 2 x 64 tanh; 4,546 parameters), hand-written RK4 with dt 0.05, backpropagation through the solver,
300 epochs with a window curriculum (40, 100, 200 steps), Adam 3e-3 with cosine decay. Baseline: 2-layer LSTM
(51,074 parameters) predicting the next-state change, teacher-forced on the training window, rolled out
autoregressively. Same inputs for both. CPU. A zero predictor is the reference.
>> Result: solver check 3.7e-4. Test RMSE (0 to 10 s / 10 to 30 s): zero predictor 1.544 / about 0.661; Neural ODE
0.381 / 0.672; LSTM 0.612 / 0.802. 10 to 30 s by damping half (low 24 trajectories / high 18): zero 0.861 /
0.178; Neural ODE 0.883 / 0.120; LSTM 1.059 / 0.064. Worst absolute error 5.23 (Neural ODE), 6.73 (LSTM).
Training: 101 s and 44 s; the Neural ODE loss was still falling (0.093 at epoch 251, 0.073 at epoch 300).
Noise (sigma 0.02, 0.05; 0 to 10 s / 10 to 30 s): Neural ODE 0.400 / 0.705 and 0.350 / 0.537; LSTM 0.633 / 0.810
and 0.719 / 0.821.
>> Observation: the Neural ODE beats the LSTM overall (38% lower in-window, 16% lower in extrapolation) with 11
times fewer parameters, but in extrapolation it is no better than predicting zero, and for low damping both
models are worse than zero; on high damping the LSTM is better. For b = 0.072 the Neural ODE keeps the phase
but not the decay, and that value lies below the training range (0.083). Both models are undertrained. The
Neural ODE is not monotonic in noise (better at sigma 0.05), probably run-to-run variation (one seed), so no
claim that noise helps. The first hypothesis (fits the window well) and the second (clear extrapolation
advantage) are only partly supported.
>> Next: per-damping breakdown and relative error, longer training, then README.


## 2026-10-05: Neural ODE versus LSTM, noise experiment

>> Setup: both models retrained from scratch on training trajectories with Gaussian noise (sigma 0.02, 0.05,
0.1) added to every state value (including the initial state); evaluated on clean test trajectories with
unseen damping values; 300 epochs each; one seed.

>> Result (RMSE 0 to 10 s / 10 to 30 s): Neural ODE 0.381 / 0.672 (sigma 0), 0.400 / 0.705 (0.02), 0.350 / 0.537
(0.05), 0.376 / 0.610 (0.10); LSTM 0.612 / 0.802, 0.633 / 0.810, 0.719 / 0.821, 0.950 / 0.840. About 140 s per
noise level.

>> Observation: the LSTM's in-window error rises steadily with noise (55% worse at sigma 0.1) while the Neural
ODE's stays within about 8% of its clean value; at sigma 0.1 the Neural ODE is 60% lower in-window and 27%
lower in extrapolation. The Neural ODE's extrapolation error is non-monotonic in sigma, which I attribute to
run-to-run variation (one seed, undertrained), so I do not claim noise helps it. A possible reason for the
LSTM's drop (untested): its targets are differences of consecutive noisy states, with noise about 1.4 times
sigma, comparable to the real per-step change.

>> Next: per-damping breakdown, longer training, and what helps the Neural ODE under noise (weight decay,
network size, seed spread at sigma 0.1).
