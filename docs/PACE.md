# Running splatsim on Georgia Tech PACE ICE

ICE is the only machine we have with NVIDIA GPUs, so training and replay run
here. Every step goes through `scripts/pace.sh`, run from the repo on a login
node.

## Quickstart

Connect to the campus network or VPN.

First time:

```bash
ssh YOUR_USERNAME@login-ice.pace.gatech.edu
git clone git@github.com:EthanMBoos/splatsim.git ~/splatsim  # needs a key
cd ~/splatsim
pace-quota               # need about 20 GB free in scratch, plus your data
scripts/pace.sh setup    # a minute: uv and splatsim's environment
scripts/pace.sh trainer  # queues a GPU job that installs 3DGRUT
squeue --me              # PD = waiting, R = running
```

The trainer job is done when its log ends with `splatsim: trainer installed`.

A flight log to a scored replay. Each line queues a job and returns; wait for
one to finish before starting the next. `D` and `R` shorten the paths:

```bash
D=~/scratch/splatsim/data R=~/scratch/splatsim/runs
scripts/pace.sh hf download Voxel51/PIVOT-mcap --repo-type dataset \
  --include "data/*-5.mcap" --local-dir pivot   # one scene: 17 flights, 6 GB
scripts/pace.sh cpu uv run python -m splatsim.flightlog export $D/street \
  $D/pivot/data/orbit_inward_low-5.mcap $D/pivot/data/orbit_inward_high-5.mcap
scripts/pace.sh cpu uv run python -m splatsim.flightlog export $D/loop \
  $D/pivot/data/traverse_loop_low-5.mcap
scripts/pace.sh train street $D/street
scripts/pace.sh replay $R/street/*/ckpt_last.pt $D/loop $R/replay-loop
tail ~/scratch/splatsim/logs/JOBID.log     # the line each command prints
```

Work interactively, see what is using space, and clear it:

```bash
scripts/pace.sh gpu           # a shell on an H100 node for four hours
scripts/pace.sh cards         # card types and what is busy
scripts/pace.sh space         # quota, and the size of data/ and runs/
scripts/pace.sh clean runs    # or data, or build; asks first
```

## What has been verified

All on 2026-11-16. The numbers the steps produce are in
[TRACKS.md](TRACKS.md).

- `setup`, `trainer`, `cpu`, `train`, `replay`, `cards`, `hf` and `space`
  have each run as written here. `trainer` took 6 minutes. A training cut to
  7000 steps took 4 minutes, and a replay of 291 frames under 2 minutes as a
  job, of which rendering is 12 ms per frame or less.
- `replay --shift 0 0 2` raised the camera 2 m, checked by eye on two frames.
- `node`, `gpu` and `clean` have not been run in the script's current form.
- The code in `splatsim.refine` ran once outside the package; the module
  itself has not run.

## Where things live

Home on ICE is 30 GB. Scratch is 300 GB and 1,000,000 files. The repo
therefore goes in home and everything large in scratch.

```text
~/splatsim/                     the git clone (home); code only
~/scratch/splatsim/
  data/                         yours: logs and export folders
  runs/                         yours: trained splats and replays
  env/                          splatsim's Python environment (0.4 GB)
  3dgrut/                       3DGRUT and its environment (about 8 GB)
  uv/                           uv, the Pythons it downloads, its cache
  logs/                         output of jobs, one file per job
```

Set `SPLATSIM_SCRATCH` to move it. Three limits shape how to use scratch:

- It is not backed up, and PACE clears old files from it. A log can be
  downloaded again; a run cannot, so copy off what matters.
- The file limit can bind before the space limit. A replay writes one PNG per
  frame.
- Sizes add up quickly. One scene of 661 frames is 1 GB exported; its trained
  splat is a 1.7 GB PLY and a larger checkpoint.

`scripts/pace.sh clean build` deletes both environments and leaves `data/`,
`runs/`, `logs/` and `uv/`. Nothing in the build is hand-made: rebuild with
`setup` and `trainer`.

## Notes on the first-time steps

**Cloning.** The repo is private, so ICE needs a key that GitHub knows. ICE
accounts come with `~/.ssh/id_rsa`, which PACE uses between cluster nodes.
Leave it alone and make a second key:

```bash
ssh-keygen -t rsa -b 4096 -f ~/.ssh/id_rsa_github -C "ice-github"
printf 'Host github.com\n  User git\n' >> ~/.ssh/config
printf '  IdentityFile ~/.ssh/id_rsa_github\n' >> ~/.ssh/config
printf '  IdentitiesOnly yes\n' >> ~/.ssh/config
chmod 600 ~/.ssh/config
cat ~/.ssh/id_rsa_github.pub
```

Add the printed line at <https://github.com/settings/keys>. `ssh -T
git@github.com` then greets you by your GitHub username.

**The trainer.** 3DGRUT is NVIDIA's splat trainer and renderer. It is a
separate program with its own Python environment, and splatsim does not
install it with `uv sync`. Two facts shape its install here:

- One of its packages links against CUDA while it builds, so the install
  fails on a login or CPU node with `cannot find -lcuda`. `trainer` therefore
  runs it in a GPU job.
- Its installer builds for Blackwell cards by default, and the CUDA 12.6
  compiler on ICE rejects that with `Unsupported gpu architecture
  'compute_100'`. The script passes `TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"`,
  which covers the A40, L40S and H100. A Blackwell therefore cannot train.

The script checks out the commit our measurements used. To move to a newer
3DGRUT, change `TRAINER_COMMIT` in `scripts/pace.sh`, then `clean build`,
`setup` and `trainer`.

**Hugging Face.** `scripts/pace.sh hf` runs the Hugging Face CLI on the login
node with `data/` as its folder and its caches in scratch. Some datasets need
`scripts/pace.sh hf auth login` first, which stores a token in
`~/.cache/huggingface/`.

**Updating.** After pulling commits that change dependencies, run
`scripts/pace.sh setup` again.

## Getting a GPU node

The script knows three card types. The counts are from `sinfo` on 2026-11-14:

| Name | Card | Partition | Cards |
|---|---|---|---|
| `h100` | H100 | `coe-gpu` | 19 nodes of 8 |
| `l40s` | L40S | `ice-gpu` | 4 nodes of 8 |
| `a40` | A40 | `ice-gpu` | 2 nodes of 2 |

`train` and `replay` use an H100 unless `SPLATSIM_CARD` names another. H100s
are the most numerous, so a job usually starts within seconds. Only the H100
has been used for training. ICE also has Blackwell cards, which our 3DGRUT
build cannot use.

Jobs get four hours. They wait in the queue unattended and write to
`~/scratch/splatsim/logs/JOBID.log`. `scripts/pace.sh gpu [CARD]` gives a
shell on a node for the same time; an interactive request is lost if the
terminal closes while it waits.

The script names the partition explicitly. Otherwise ICE assigns the job to
several partitions, including ones with no matching card, and the job stays at
`PD` with reason `BadConstraints` even when a card is free. A healthy waiting
job shows `(Priority)` or `(Resources)`.

**Two H100 nodes are excluded.** On 2026-11-14 one card on each of
`atl1-1-03-010-10-0` and `atl1-1-03-010-25-0` failed at startup on all nine
runs it was given, with another program. Slurm tends to return the same card,
so a resubmitted job fails again. We have not tried 3DGRUT on them. PACE may
repair the cards, so remove the exclusion in `card_request` occasionally.

**Training can fail at random.** One training of eight stopped after a minute
with `CUDA error: an illegal memory access` and ran when resubmitted, on
another node.

## Copying files off ICE

ICE cannot reach your computer, so every copy starts from your computer, in a
terminal not logged in to ICE. `scp` asks for the same password and Duo prompt
as `ssh` and needs the campus network or VPN.

The form is `scp FROM TO`. A path on ICE is
`YOUR_USERNAME@login-ice.pace.gatech.edu:PATH`. A `PATH` without a leading
slash starts in your ICE home directory.

```bash
# A replay's scores
scp YOUR_USERNAME@login-ice.pace.gatech.edu:scratch/splatsim/runs/NAME/scores.csv .

# A job's log
scp YOUR_USERNAME@login-ice.pace.gatech.edu:scratch/splatsim/logs/JOBID.log .

# A whole folder
scp -r YOUR_USERNAME@login-ice.pace.gatech.edu:scratch/splatsim/runs/NAME .
```
