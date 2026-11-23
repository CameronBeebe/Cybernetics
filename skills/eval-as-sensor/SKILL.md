---
name: eval-as-sensor
description: "Use whenever an eval score, test result, or LLM judge drives a decision: prompt tuning, model selection, automated revision, release gating."
license: MIT
---
# Eval as sensor

A controller is only as good as its sensor. Characterize the eval before letting it drive any loop.

## Procedure
1. **Write a sensor card** before any tuning loop runs:
   - *noise*: run-to-run variance on identical inputs
   - *bias*: agreement with human labels on a labeled sample
   - *resolution*: smallest score difference the metric can express
   - *latency and cost* per measurement
2. **Compute the minimum detectable effect (MDE)** for the sample size you can afford. Changing prompts or models on differences inside the noise band puts loop gain on noise: drift dressed up as progress.
3. **Filter and gate.** Average repeated measurements, use paired comparisons on the same items, and add a deadband: changes smaller than the MDE are neither accepted nor reported as wins.
4. **Hold out a check the loop never sees.** Keep a holdout set (or a second sensor) that no optimization step touches.
5. **Treat judge probabilities as credences.** Score them with proper scoring rules (e.g. the Brier score) and recalibrate against human labels on a schedule.

## Output
A sensor card (noise, bias vs human labels, MDE, holdout), and a gate in the tuning loop that refuses sub-MDE changes.

## Second-order point
The evaluator is part of the system. Optimizing against a sensor changes what it measures (Goodhart's law), which is why step 4 is not optional.

## Sources
- Brier, "Verification of forecasts expressed in terms of probability," *Monthly Weather Review* 78(1):1–3, 1950. doi:10.1175/1520-0493(1950)078<0001:VOFEIT>2.0.CO;2
- Standard measurement and experimental-statistics practice; Goodhart's law is named as a known effect, not a formal result.
