# splatsim

Replay a drone's flight inside a Gaussian splat of the place it flew through,
and score the replay against the log. A Gaussian splat is a 3D scene stored as
millions of small coloured blobs fitted to photos of a real place, so a render
of it looks like the place. splatsim is a prototype for aerial drones with one
camera.

The workflow has four steps:

1. **Export.** Read the camera frames, lens and poses from the flight log, an
   MCAP file, into a folder.
2. **Refine.** If the poses are the drone's own GPS and gimbal estimate, they
   are too coarse to train on; COLMAP refines them, using them as priors.
3. **Train.** NVIDIA's [3DGRUT](https://github.com/nv-tlabs/3dgrut) fits a
   splat to the frames.
4. **Replay.** Render the splat at every logged pose with the logged lens,
   using 3DGRUT's renderer, and score each render against the logged frame.
   The flight replayed need not be one the splat was trained on.

The purpose is step 4 with something changed: the same moment from another
height, or a different detector run on both the logged and the rendered frame.
Whether such a render can be trusted is the open research question, and what
we have measured so far is in [docs/TRACKS.md](docs/TRACKS.md).

## Requirements

Steps 1 and 2 need only a CPU and run on Linux or macOS. Steps 3 and 4 need an
NVIDIA GPU with CUDA and an installed 3DGRUT, which means Linux. Our only such
machine is Georgia Tech's PACE ICE cluster; everything specific to it is in
[docs/PACE.md](docs/PACE.md) and `scripts/pace.sh`.

## Setup

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra sfm     # steps 1 and 2; --extra sfm adds COLMAP for step 2
uv run pre-commit install
```

For steps 3 and 4, install 3DGRUT as its README describes. It keeps its own
Python environment, separate from this one. We use commit `dfd3d54`. On ICE,
`scripts/pace.sh trainer` does this.

## Use

A flight log to test with: [PIVOT-mcap](https://huggingface.co/datasets/Voxel51/PIVOT-mcap)
holds drone flights as MCAP files, with both the drone's own poses and poses
solved offline by COLMAP. Its licence is CC BY-NC 4.0. Logs go in `data/` and
outputs in `runs/`; git ignores both.

```bash
# 1. Frames, lens and poses out of the logs of one scene
uv run python -m splatsim.flightlog export data/street \
  data/pivot/orbit_low.mcap data/pivot/orbit_high.mcap

# 1 and 2. The same from the drone's own poses, then refined
uv run python -m splatsim.flightlog export data/street-raw \
  data/pivot/orbit_low.mcap data/pivot/orbit_high.mcap --poses camera_measured
uv run python -m splatsim.refine data/street-raw data/street-refined

# 3. Train, inside 3DGRUT's folder and environment
python train.py --config-name apps/colmap_3dgut.yaml path=/abs/data/street \
  out_dir=/abs/runs experiment_name=street export_ply.enabled=true

# 4. Replay another flight of the same scene, with 3DGRUT's Python
python /abs/splatsim/src/splatsim/replay.py \
  /abs/runs/street/RUN/ckpt_last.pt /abs/data/street-loop /abs/runs/replay-loop
```

| Step | Reads | Writes |
|---|---|---|
| `flightlog export` | MCAP files of one scene | `images/`, `sparse/0/` (a COLMAP text model), `trajectory.tum`, `camera.json` |
| `refine` | an export folder | a folder with refined `sparse/0/` and a link to the images |
| 3DGRUT `train.py` | an export or refined folder | `ckpt_last.pt`, `export_last.ply` |
| `replay.py` | a checkpoint and an export folder | `frames/NAME.png`, `scores.csv` (PSNR, SSIM, LPIPS per frame) |

`replay.py --shift X Y Z` moves every camera by that many metres before
rendering, for views the drone did not fly. Those frames have no logged frame
to compare with, so they are not scored.

The log reader covers one convention: Foxglove protobuf messages as PIVOT-mcap
writes them. A ROS 2 log needs another decoder.

## What to expect

Measured on PIVOT's `village_street`, a street 52 m across, on an H100. Each
figure is one run; the details are in [docs/TRACKS.md](docs/TRACKS.md).

| Step | Time | Result |
|---|---|---|
| Export, 661 frames at 1920x1080 | 3 min | |
| Refine, 661 frames, 8 CPUs | 2 h | camera path within 6 cm of the offline solution, after alignment |
| Train | 20 min | 23.0 dB on held-out frames of the training flights |
| Replay, 291 frames | 12 ms per frame to render | 12.6 dB on a flight the splat was not trained on |

The last row is the main finding so far: a splat replays the flights it was
built from and fails on another flight through the same street.

## Tests and lint

```bash
uv run pytest
uv run ruff check . && uv run ruff format .
```

## Layout

```text
src/splatsim/
  flightlog.py     # step 1: MCAP to an export folder
  refine.py        # step 2: COLMAP with the log's poses as priors
  replay.py        # step 4: runs under 3DGRUT's Python, imports no splatsim
scripts/pace.sh    # every step on PACE ICE
docs/              # TRACKS.md, RESEARCH.md, IDEAS.md, PACE.md, TODO.md
```

## Conventions

Quaternions are `wxyz` in our code and names carry the order, as in
`quat_wxyz`. Units are metres and radians. The world frame is the log's own;
PIVOT's is metric with Z up. The lens after export is a pinhole with square
pixels and a centred principal point.
