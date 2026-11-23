---
name: requisite-variety
description: "Use when deciding whether a harness can cope with the disturbances it will meet, and whether to add responses, attenuate disturbances, or escalate."
license: MIT
---
# Requisite variety

Apply Ashby's law of requisite variety to an agent harness: only variety in the regulator can absorb variety in the disturbances.

## Core idea
In entropy form, the variety of outcomes is at least the variety of disturbances minus the variety of the regulator. Two corollaries matter in practice:
- the regulator must be able to **observe** the disturbance (a channel from disturbance to regulator), and
- its **repertoire** of responses must be large enough.

A regulator with plenty of actions but no view of the disturbance cannot do better than the best single fixed action. (The original toy model in this repo is an instance: its regulator never conditions on the disturbance row.)

## Procedure
1. **Enumerate disturbance classes from real failure logs**, not from imagination: tool/API failures, flaky tests, ambiguous specs, dependency drift, adversarial or malformed input, rate limits.
2. **Enumerate the response repertoire**: tools, strategies, fallbacks, subagents, escalation paths.
3. **Build a coverage matrix** (disturbance × response) and mark, for each disturbance, whether the harness can detect it and whether any response regulates it.
4. **For each uncovered cell, choose one move:**
   - add variety: a new tool, strategy or subagent
   - attenuate the disturbance: sandboxing, pinned dependencies, a constrained spec
   - amplify: escalate to a human or a more capable regulator
5. **Flag dead variety**: responses that never fire. They cost tokens and attention and are candidates for removal.

## Output
The coverage matrix, the list of unregulated and undetected disturbances, the dead responses, and one recommendation per gap. Requisite variety is about where variety must come from, not a reason to add every possible tool.

## Validity limits
The law is a counting/information bound on outcomes, not a design recipe. Disturbance classes in real harnesses overlap and shift over time; re-derive the matrix from fresh logs periodically.

## Sources
- Ashby, *An Introduction to Cybernetics*, Chapman & Hall / Wiley, 1956, ch. 11 ("Requisite Variety").
