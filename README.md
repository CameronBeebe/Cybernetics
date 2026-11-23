# Cybernetics

From Ashby's regulator game to engineering skills for agent harnesses.

This repository has two layers. The first is a 2020–21 toy implementation of the game-theoretic foundation of cybernetic systems theory as outlined by W. Ross Ashby, written alongside a talk arguing that neural networks are best understood as cybernetic regulators. The second, added in 2026, is a set of [cybernetics engineering skills](skills/): procedures that coding agents can follow when they build, debug and evaluate agent harnesses.

## 2020–21: neural networks as cybernetic regulators

In *An Introduction to Cybernetics* (1956), Ashby models regulation as a game. An environment plays a disturbance (a row of an outcome table), a regulator replies with an action (a column), and the regulator succeeds when the resulting outcome lies in a goal set. The central result is the law of requisite variety: only variety in the regulator can absorb variety in the disturbances.

The talk built on that frame to argue that connectionist networks are regulators that learn by error control or by reinforcement, and that this explains both their effectiveness and their opacity.

- **Talk slides:** [Black Boxes or Rube Goldberg Machines? Neural Networks as Cybernetic Regulators](Neural_Networks_Cybernetic_Regulators.pdf) (November 12, 2021)
- **Toy package:** [`cybernetics/`](cybernetics/)
  - [`cybernetics.py`](cybernetics/cybernetics.py) and [`train.py`](cybernetics/train.py): the original functional version. It builds a random game, has the environment play from a Dirichlet distribution over rows, and trains the regulator with an urn-reinforcement rule.
  - [`regulators.py`](cybernetics/regulators.py): an unfinished class-based rewrite (`Regulator`, `Ashby`) with optional MPI or PySpark parallelism ([`mpi_regulator_test.py`](mpi_regulator_test.py), [`mpi_pyspark.yml`](mpi_pyspark.yml)).
- **Notebooks:** [`Cybernetic_Game_Theory.ipynb`](Cybernetic_Game_Theory.ipynb) (the original exploration) and [`test.ipynb`](test.ipynb).

### Status and known limitations
The code is a historical toy model, kept as it was written. It is not maintained as a library.
- The regulator draws its action from a single urn and never conditions on the disturbance. In Ashby's terms it has no channel from the disturbance, so it cannot show requisite variety. The best it can do is match the best single fixed action.
- The environment's distribution uses `np.random.dirichlet(alpha=index)`, which makes later rows systematically more likely.
- The class-based rewrite in `regulators.py` is incomplete. Among other things, it uses a mutable default argument (`history=dict()`) that is shared across instances.

The original to-do list (game input, success-availability analysis, alternative reinforcement rules, parallelism, a single-step `act` method) is preserved in this file's git history.

## 2026: cybernetics engineering skills for agent harnesses

An agent harness is a feedback system. A model acts on a repository or environment, tests, judges and users report back, and the harness decides what to do next. The same questions Ashby and Wiener asked of regulators apply directly. What is the setpoint? What does the regulator observe? Is its repertoire large enough for the disturbances it meets? What does it model? When should it change its own parameters?

[`skills/`](skills/) turns those questions into seven procedures. Each skill is a self-contained `SKILL.md` that gives a coding agent concrete steps, the output it should produce, the limits of the analogy, and primary sources.

| Skill | When to use |
|---|---|
| [`harness-control-loops`](skills/harness-control-loops/SKILL.md) | Designing or debugging any act → feedback → correct loop (retries, test-fix cycles, critic/reviser). |
| [`agent-phase-space`](skills/agent-phase-space/SKILL.md) | Runs stall, cycle, diverge across seeds, or change qualitatively when a parameter changes. |
| [`requisite-variety`](skills/requisite-variety/SKILL.md) | Checking whether a harness can cope with the disturbances it will meet. |
| [`good-regulator-models`](skills/good-regulator-models/SKILL.md) | Deciding what the agent must model and catching a mismatch between that model and reality. |
| [`ultrastable-harness`](skills/ultrastable-harness/SKILL.md) | Adding an adaptive outer loop (strategy, model or prompt switching, prompt optimization). |
| [`eval-as-sensor`](skills/eval-as-sensor/SKILL.md) | An eval, test or LLM judge drives a decision. |
| [`pid-state-space-control`](skills/pid-state-space-control/SKILL.md) | Holding a numeric harness quantity (rate, budget, retry pressure, confidence) near a target. |

The skills use plain Markdown with YAML front matter (`name`, `description`), so they work with agent frameworks that load `SKILL.md` files, or as checklists for a human engineer. Every skill states where its formal concepts stop applying: agent loops are discrete, stochastic, non-stationary and only partly observed.

## License
MIT. See [LICENSE.txt](LICENSE.txt).
