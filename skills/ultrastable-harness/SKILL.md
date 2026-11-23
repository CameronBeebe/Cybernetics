---
name: ultrastable-harness
description: "Use when adding an adaptive outer loop (strategy, prompt or model switching, automated prompt optimization) on top of an agent's task loop."
license: MIT
---
# Ultrastable harness

Structure adaptation as Ashby's ultrastability: a fast loop regulates the task, and a slow loop changes the fast loop's parameters only when an essential variable leaves its bounds.

## Procedure
1. **Declare essential variables with numeric bounds**: budget (tokens, dollars, wall-clock), correctness invariants (tests that must stay green), safety and permission limits, trust signals from the human.
2. **Separate the loops in code.**
   - *Inner loop*: the agent working the task.
   - *Outer loop*: changes the inner loop's parameters (strategy, model, prompt, tool set, prompt-optimizer step) step-wise.
3. **Fire the outer loop on bound violations only**, not on every error. This keeps the timescales apart so the outer loop does not fight the inner one.
4. **Make the search explicit.** Ashby's outer loop searched until it found a configuration that kept the essential variables in bounds. Bound the search (max steps, allowed configurations) and log each step with the violation that triggered it.
5. **Check timescale separation.** Verify that outer-loop changes happen slower than inner-loop convergence; if not, lengthen the outer interval.

## Output
Declared essential variables and bounds, separated inner/outer loop code, and an outer-loop log (trigger → change → result).

## Risks
- The outer loop optimizes a proxy instead of the essential variable it stands for (Goodhart). Keep at least one essential variable measured on data the outer loop never optimizes against.
- Overlapping timescales make the two loops interact unpredictably.

## Sources
- Ashby, *Design for a Brain*, Chapman & Hall, 1952, ch. 8 (the principle of ultrastability; essential variables and step-functions are developed in the surrounding chapters).
