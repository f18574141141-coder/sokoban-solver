# Sokoban Solver

![make.fun bounty](https://make.fun/badge/sokoban-solver-01-run-20261121.svg)
[![rank](https://make.fun/badge/sokoban-solver-01-run-20261121/f18574141141-coder/sokoban-solver/rank.svg)](https://make.fun/bounty/sokoban-solver-01-run-20261121)
[![reward](https://make.fun/badge/sokoban-solver-01-run-20261121/f18574141141-coder/sokoban-solver/reward.svg)](https://make.fun/bounty/sokoban-solver-01-run-20261121)

A push-optimal Sokoban solver, built for the [make.fun bounty "Build the best Sokoban solver, 21 November 2026"](https://make.fun/bounty/sokoban-solver-01-run-20261121). One program answers all 5 instances of a set in a single run.

## How it works

The levels are a hallway with 1–4 rooms off it, and a box pushed out of a
room's door can never come back — so each room is an independent problem and
the level optimum is the sum of the rooms' optima. The solver:

1. Finds the hallway (the player's row) and splits the board into room
   components, each entered from the hallway cell beside its door.
2. Solves each room push-optimally with A* over pushes: Hungarian
   assignment of boxes to goals over walking distances as the heuristic,
   plus static dead squares, 2×2 blocks and frozen-box deadlock pruning.
3. Walks the player between rooms through the hallway and concatenates the
   move strings, then replays the whole solution to verify it. A whole-level
   A* is the fallback when decomposition fails.

The search budget is counted in state expansions, never in wall-clock time,
so two runs write the same bytes.

## Run it

```sh
node main.ts   # reads /input/<id>/level.txt, writes /output/<id>/solution.txt
```

Local run on the kit's practice set:

```sh
MAKEFUN_INPUT_DIR=practice/input MAKEFUN_OUTPUT_DIR=out node main.ts
```

## Results

Practice set: **1,000,000 / 1,000,000** — every instance solved in its
push-optimal count (13 / 16 / 29 / 47 / 60 pushes), in under a second for
the whole set. The same score held on 10 further freshly drawn sets. The
house baseline (plain-astar 1.0.0) scores 200,000 on the practice set: it
solves only the first instance and gives up on the rest.

## License

MIT — see [LICENSE](LICENSE).
