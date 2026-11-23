---
name: agent-phase-space
description: "Use when agent runs stall, loop, diverge across seeds, or change qualitatively when a parameter changes (model, temperature, step budget, context window, prompt)."
license: MIT
---
# Agent phase space

Frame agent runs as trajectories of a dynamical system, then find the stalls, cycles, outcome basins, bifurcations, and divergence that explain the behavior.

## Procedure
1. **Choose state variables and justify them.** Candidates: tests passing, diff size, files touched, context tokens used, plan step, last tool type, open errors. The choice of state is a modelling decision; write it down with the reason.
2. **Use the rigorous framing first.** A run is a trajectory of a stochastic, discrete-time process over that state. Then translate:
   - *fixed point* → absorbing or stalled state (no state change over k steps)
   - *limit cycle* → recurrent state sequence (edit → revert → edit)
   - *attractor / basin* → absorbing outcome class, and the probability of reaching it from a given starting condition
3. **Detect revisits and cycles.** Hash the state at each step; report stalls, revisits, and cycles with their period and the step where they begin.
4. **Sweep for bifurcations.** Vary one parameter at a time over a grid. A bifurcation is a qualitative change in which outcome class runs end in, not a smooth drift in the mean.
5. **Measure divergence.** Run pairs from the same start with different seeds and measure how quickly they separate in state space. This is the empirical stand-in for sensitivity to initial conditions; true Lyapunov exponents are not defined here.

## Output
A small trace analyzer (state hashing, stall/cycle detection) and a phase portrait: outcome-class frequencies per parameter value, the cycles found, and the step where seeds diverge. Fixes then get concrete targets, e.g. "break the 3-cycle in the edit/revert subspace."

## Validity limits
The dynamics are non-autonomous: context grows, and the environment or model can change mid-run. "Attractor" therefore holds only within a fixed configuration. State this in every report, along with the number of seeds behind each estimate.

## Sources
- Strogatz, *Nonlinear Dynamics and Chaos: With Applications to Physics, Biology, Chemistry, and Engineering*, Addison-Wesley, 1994.
- Standard Markov-chain theory for absorbing and recurrent classes.
