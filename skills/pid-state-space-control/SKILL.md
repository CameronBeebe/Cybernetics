---
name: pid-state-space-control
description: "Use when a numeric harness quantity must track a target over time: request rate under a provider limit, token or cost budget, retry pressure, concurrency, or a confidence estimate that gates escalation."
license: MIT
---
# PID and state-space control for harnesses

Use classical (PID) and state-space tools where the controlled quantity in a harness really is numeric.

## PID
- **P** reacts to the current error: throttle in proportion to how far over the rate limit you are.
- **I** accumulates past error and removes steady-state offset: a task chronically 10% over budget gets its per-step allowance trimmed.
- **D** reacts to the error's rate of change: accelerating spend is damped before it overshoots.
- **Integral windup.** When the actuator saturates (max backoff, provider hard limit), the integral keeps growing and the loop overshoots once the constraint lifts. Clamp the integrator, stop integrating while saturated, or use back-calculation.
- **Derivative noise.** Token counts, latencies and judge scores are noisy, and D amplifies noise. Low-pass filter the derivative or drop it; PI is often enough.
- **Tuning.** Start with P, add I, add D only if overshoot is a problem. Ziegler–Nichols (raise gain until sustained oscillation, then back off by fixed ratios) gives aggressive settings; treat it as a starting point and tune on measured step response (overshoot, settling time) using replayed traffic.
- **Backoff as control.** Exponential backoff is multiplicative decrease; AIMD (additive increase, multiplicative decrease) is a well-understood stable rate controller. Add jitter so clients sharing a limit do not synchronize into oscillation.

## State space
- Write the hidden state *x* (true remaining budget, true provider capacity, true task difficulty or agent competence), the inputs *u* (throttle, model choice, step allowance), and the measurements *y* (tokens used, 429 responses, test results).
- An **observer** reconstructs *x* from *y*. Under linear-Gaussian assumptions the **Kalman filter** is the optimal linear estimator. Practical use: track a smoothed estimate of success probability with its uncertainty, instead of reacting to each individual pass or fail.
- **LQR intuition.** Choose actions by trading the cost of state error against the cost of control effort. In a harness: state how much a unit of budget overrun costs relative to a unit of extra intervention (model upgrade, human escalation), then pick the gain that minimizes the sum. Writing the cost down disciplines the design even if no Riccati equation is ever solved.

## Procedure
1. For each numeric loop, write the setpoint, measured variable, actuator, and actuator saturation limits.
2. Implement PI with anti-windup (add D only if replayed data justify it), or a scalar Kalman-style estimator.
3. Test on replayed or simulated traces. Report overshoot, settling time, and behavior under saturation.
4. Never deploy a gain that has not been step-tested.

## Validity limits
Harness plants are non-linear, time-varying and often discrete-event; PID and LQR guarantees rest on linear(ized) models, so treat them as designs to test, not proofs. The Kalman filter is optimal only under its assumptions; for binary pass/fail data a Beta–Bernoulli update is often the more honest estimator.

## Sources
- Åström & Murray, *Feedback Systems: An Introduction for Scientists and Engineers*, Princeton University Press, 2008: ch. 6 "State Feedback" (§6.3, including linear quadratic regulators); ch. 7 "Output Feedback" (§7.1 observability, §7.2 state estimation, §7.4 Kalman filtering); ch. 10 "PID Control" (§10.3 tuning, §10.4 integrator windup).
- Ziegler & Nichols, "Optimum settings for automatic controllers," *Transactions of the ASME* 64(8):759–765, 1942. doi:10.1115/1.4019264
- Kalman, "A new approach to linear filtering and prediction problems," *Journal of Basic Engineering* 82(1):35–45, 1960. doi:10.1115/1.3662552
