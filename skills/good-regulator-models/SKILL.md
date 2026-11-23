---
name: good-regulator-models
description: "Use when deciding what model of its environment (codebase, user, tools) an agent must hold and keep current, and when catching a mismatch between that model and reality."
license: MIT
---
# Good regulator models

Make the agent's model of its environment explicit, measure how wrong it is, and refresh it on evidence rather than letting it drift.

## Core ideas
- **Conant–Ashby (1970):** the simplest optimal regulator of a system is a model of that system (a mapping from system states to regulator states). The theorem is narrower than its title. State its assumptions; do not cite it as a general "agents need world models" claim.
- **Internal model principle (Francis & Wonham 1976):** to track or reject a class of signals robustly, the controller must contain a model of the process that generates them.
- **Cause vs error control (Ashby 1956):** a regulator can act on the cause (it observes the disturbance) or on the error (it observes only the deviation). Choose deliberately.

## Procedure
1. **Name the working model.** What does the agent rely on: a repository map, a test-to-module map, project instructions, memory files, a model of the user's intent?
2. **Predict, then check.** Before acting, the agent records a prediction ("this test will pass," "this function is defined in X," "this command exits 0"). After acting, it logs whether the prediction held.
3. **Track calibration.** The hit rate and calibration of those predictions are the measure of model–plant mismatch.
4. **Refresh on evidence.** When calibration drops below a threshold, trigger a model refresh (re-index, re-read, re-ask) instead of taking more actions on a stale model. This is state estimation, in the spirit of the Kalman filter.

## Output
Prediction-then-check hooks in the harness, a calibration log, and a refresh trigger with an explicit threshold.

## Validity limits
Conant–Ashby assumes a specific setting (optimality and simplicity of the regulator, a deterministic mapping). Prediction hooks cost tokens; sample them if the full cost is too high, and say so.

## Sources
- Conant & Ashby, "Every good regulator of a system must be a model of that system," *International Journal of Systems Science* 1(2):89–97, 1970. doi:10.1080/00207727008920220
- Ashby, *An Introduction to Cybernetics*, Chapman & Hall / Wiley, 1956, ch. 12 ("The Error-controlled Regulator").
- Francis & Wonham, "The internal model principle of control theory," *Automatica* 12(5):457–465, 1976. doi:10.1016/0005-1098(76)90006-6
- Kalman, "A new approach to linear filtering and prediction problems," *Journal of Basic Engineering* 82(1):35–45, 1960. doi:10.1115/1.3662552
