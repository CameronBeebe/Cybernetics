---
name: harness-control-loops
description: "Use when designing or debugging any loop where an agent acts, gets feedback, and corrects: retries, test-fix cycles, critic/reviser pairs, planners, self-repair."
license: MIT
---
# Harness control loops

Treat an agent harness as a feedback control system, name every part of the loop, and find where it is open, slow, unobservable, or over-driven.

## Procedure
1. **Name the blocks.** For each loop in the harness, write down:
   - *plant*: what is being changed (repo, environment, document, user state)
   - *actuator*: how the agent changes it (tool calls, edits, messages)
   - *sensor*: what reports back (tests, linters, type checker, LLM judge, user)
   - *setpoint*: what "done" means, as a measurable predicate, not a vibe
   - *comparator*: where setpoint and measurement are compared
   - *controller*: the policy (model + prompt + harness logic) that turns error into action
2. **Characterize the error signal.** Is the full residual fed back (failing test output, diff of expected vs actual), a scalar (score, pass/fail), or nothing (an open-loop segment)?
3. **Estimate gain and delay.**
   - *Gain*: how much the system changes per correction (edit size, how much a prompt is rewritten, retry aggressiveness).
   - *Delay*: steps, wall-clock time, or cost before feedback arrives (CI, slow evals, human review).
   - High gain plus long delay produces oscillation: the agent flip-flops between two fixes, or rewrites what already worked.
4. **Check observability and controllability.**
   - *Observability*: can the controller reconstruct the state it needs from what it actually sees (truncated logs, summarized context, sampled outputs)?
   - *Controllability*: can its actions reach the target variable at all (e.g. a flaky test caused by infrastructure the agent cannot touch)?
5. **Apply stability levers** only where step 3 or 4 found a problem:
   - damping: smaller, more local edits per iteration
   - deadband / hysteresis: ignore differences inside the measurement noise band
   - backoff: lengthen the interval between corrections when errors persist
   - integral-like memory: keep a record of persistent residual error so the same symptom is not re-fixed from scratch each pass

## Output
A loop table (block → concrete component → signal type → delay), a list of flags (open-loop segments, setpoints never measured, delay longer than the correction interval, unobservable or uncontrollable targets), and at most three proposed changes, each tied to a specific flag.

## Validity limits
Agent loops are discrete, stochastic, non-linear and non-stationary. Linear-systems results (gain and phase margins, Nyquist) do not transfer quantitatively. Use them as qualitative design heuristics and confirm each change with an experiment.

## Sources
- Rosenblueth, Wiener & Bigelow, "Behavior, Purpose and Teleology," *Philosophy of Science* 10(1):18–24, 1943. doi:10.1086/286788
- Wiener, *Cybernetics: Or Control and Communication in the Animal and the Machine*, Technology Press / Wiley / Hermann, 1948.
- Åström & Murray, *Feedback Systems: An Introduction for Scientists and Engineers*, Princeton University Press, 2008 (2nd ed. 2021).
- Kalman, "On the general theory of control systems," Proc. 1st IFAC Congress, Moscow, 1960; *IFAC Proceedings Volumes* 1(1):491–502. doi:10.1016/S1474-6670(17)70094-8
