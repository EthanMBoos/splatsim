# AGENTS.md

splatsim is a prototype: research and student code that changes quickly.
Nothing here is fixed.

## How to work here

- Write the simple, direct version. Add an abstraction, config layer or
  interface only when a present need requires it.
- Experiments are welcome. Replace what does not fit.
- `docs/` records where we stand, not rules. If a decision is in the way, change
  it and update the doc.
- Do not gold-plate: no exhaustive tests, no handling of cases that cannot
  happen, no compatibility shims. Keep a test if it catches a real mistake.
- State what was run and what was not. Training and replay need an NVIDIA
  GPU, and our only one is Georgia Tech's PACE ICE cluster, so code that
  touches 3DGRUT is unverified until it has run there.
- The README is written for any Linux machine with an NVIDIA GPU. ICE
  specifics go in `docs/PACE.md` and `scripts/pace.sh`.

## How to write here

Comments and docs follow plain deductive order: the claim, then the reason,
then what follows from it.

- State each fact once, in one place. Elsewhere, point to it.
- Give the measurement and what it was measured on, not an adjective.
- Mark what is measured, what is read from documentation, and what is untried.
- A comment says why the code is so. It does not restate the code or tell the
  story of how we found out.
- Wrap Markdown at 80 columns. Use prose and few headings.

## Where we stand

The measurements behind these are in `docs/TRACKS.md`.

- The goal is to replay a drone's flight log inside a Gaussian splat of the
  place and score the replay against the log. The steps are export, refine,
  train and replay, as the README sets out.
- Training and rendering are NVIDIA's 3DGRUT, used as an installed program
  and not vendored. `src/splatsim/replay.py` runs under 3DGRUT's Python and
  imports nothing from splatsim; the rest of the package imports nothing from
  3DGRUT.
- An export folder (`images/`, `sparse/0/`, `trajectory.tum`, `camera.json`)
  is the interface between the steps.
- Quaternions are `wxyz`, units are metres and radians. Names carry the
  order, as in `quat_wxyz`. The world frame is the log's.

## Practical

- `uv sync --extra sfm` works anywhere, macOS included.
- `uv run pytest` and `uv run ruff check . && uv run ruff format .` before
  committing.
- On ICE, everything goes through `scripts/pace.sh`. See `docs/PACE.md`.
- 3DGRUT trains on an H100, L40S or A40. Our build cannot use a Blackwell.
- One-off measurement scripts stay out of `main`. TRACKS.md records the
  method of each measurement.
- Open work is in `docs/TODO.md`.
