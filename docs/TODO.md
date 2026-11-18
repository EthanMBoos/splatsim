# To do

Open work, updated 2026-11-16. What has been measured, and the research
tracks that follow from it, are in [TRACKS.md](TRACKS.md).

## Next, smallest first

- Run `splatsim.refine` on ICE. Its code ran once outside the package; the
  module itself has not run.
- Replay the splat trained from refined poses. `refine` writes a COLMAP model
  with a radial lens term, which 3DGRUT reads; a replay of it is untried.
- Write `trajectory.tum` from `refine`, so that a refined folder is a full
  export folder.
- Replay from a trajectory file, not only a shift of the logged poses, so
  that a counterfactual flight can be authored.
- A task score beside PSNR, SSIM and LPIPS: run a detector on the logged and
  the rendered frame and compare what it finds. Track 3 states what a first
  version measured and how.
- A trust score per rendered frame from camera geometry alone (track 1).
- A viewer that shows the logged frame beside the render along a timeline.
- Read a ROS 2 log, and a log from our own drone.

## Verified on PACE ICE

2026-11-16: `flightlog export`, a 3DGUT training and `replay.py` ran through
`scripts/pace.sh` as it is on `main`. `uv run pytest` reported 1 passed on
macOS; the test needs no GPU.
