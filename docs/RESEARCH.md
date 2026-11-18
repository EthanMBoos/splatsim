# What NVIDIA's tools cover, and what they leave to us

splatsim bets that replaying real flight data inside a splat is more useful
than simulating a scene. This document records what NVIDIA's code and forum
say about that route, where they stop, and what we decided as a result. It is
the basis for choosing student work.

The target is a drone whose only imaging sensor is one camera: no lidar, no
stereo. We assume its log also carries the flight controller's position
estimate. Without that, a single camera gives a scene of unknown scale and the
log cannot fix it.

Everything under the first two headings was read on 2026-11-16, not run. The
repos read were `nv-tlabs/3dgrut` and `NVIDIA/ncore`. What we measured is in
[TRACKS.md](TRACKS.md).

## What NVIDIA's code shows

**Flight logs are outside NVIDIA's pipeline.** 3DGRUT, the trainer, reads
COLMAP folders, NeRF and ScanNet++ layouts, and NVIDIA's NCore format. It reads
no MCAP or rosbag. A request for ROS 2 bag support in NCore
([Issue 90](https://github.com/NVIDIA/ncore/issues/90)) was closed with this
reply from its maintainer: "A general-purpose ROS2 bag converter isn't
feasible ... Users with ROS2 bags would need to write a custom converter
tailored to their specific recording setup." NCore's converters cover
automotive datasets: Waymo, nuScenes, Argoverse 2, KITTI. NVIDIA's guide for a
single camera assumes still photos with locked focus and exposure, the
opposite of flight video.

**NCore has fields for rolling shutter.** It stores the start and end of each
frame's exposure and a shutter delay, and these are the first things forum
staff check when moving objects blur.

**3DGRUT has no camera pose optimisation.** Poses must therefore be right
before training; track 2 of TRACKS.md measures what wrong ones cost.

## What the NuRec forum shows

NuRec is NVIDIA's product for reconstructing a scene from a sensor log; 3DGRUT
is its open trainer. The
[NuRec forum](https://forums.developer.nvidia.com/c/omniverse/platform/nurec/752)
held 35 topics on 2026-11-16, the first from February 2026, and we read all of
them. Quotations are from NVIDIA moderators or employees unless stated.

**It is a forum for autonomous vehicles.** Across all posts, "drone", "UAV",
"aerial", "MCAP" and "rosbag" appear zero times; "LiDAR" appears 59 times and
"CARLA" 25. No topic concerns replay fidelity or whether a detector behaves
the same on rendered frames.

**NVIDIA makes the same bet for cars.** Its users reconstruct a driving log
with NuRec and replay it in CARLA or in AlpaSim, NVIDIA's driving simulator.
splatsim is that loop for a drone. We have not read AlpaSim.

**NuRec expects clean logs from several sensors.** Lidar "is not required but
is currently recommended to resolve scale ambiguity or improve pose
optimization". The community manager states that camera-only input gives
"significant degradation" and that "NuRec really requires a lot of super
clean, super organized data to work well." A camera-only drone is therefore
the case NVIDIA calls degraded.

**A single camera cannot use the NuRec container yet.** Data converted from
COLMAP fails in training with "Rig-to-world poses are currently required"; a
fix is planned. The supported single-camera route is COLMAP into open 3DGRUT.

**Low twenties of PSNR is ordinary on NVIDIA's robot data.** A user following
NVIDIA's guide on the Galileo scene reports a mean PSNR of about 22 dB, and
about 14 dB with the alternative MCMC configuration. The scene differs from
ours, so this compares only roughly with our 23 dB.

**Reconstruction is slow.** A driving scene takes 6 to 7 hours on an A100,
which staff call normal. NVIDIA's Instant NuRec reconstructs in seconds, but
only from multi-camera driving logs.

## Decision

**We train with open 3DGRUT, not the NuRec container.** The container rejects
single-camera data today.

**We replay with 3DGRUT's renderer and do not build our own.** It is the
renderer the splat was trained against, and it draws a 1920x1080 frame in
about 10 ms on an H100.

**The aim is a measured result on real flight data.** NVIDIA's tooling is
built for cars and ground robots with several sensors. A result it cannot
produce, on the camera-only data it calls degraded, is worth more than a
patch.

**Upstream work goes to 3DGRUT or NCore.** 3DGRUT merges pull requests from
outside contributors regularly; what fits there is training from flight video
and pose optimisation. NCore is written nearly all by one maintainer, with a
few outside contributions; what fits there is a flight-log converter. We do
not plan to post on the forum.

## Work that stays ours

The tracks, each with its first measurements, are in [TRACKS.md](TRACKS.md).
Three ideas from the reading are not yet a track because nothing was tried:
modelling the drone camera's lens, rolling shutter and exposure before
blaming the splat for a difference; using the difference between a logged
frame and its render as a change detector; and collision from a mesh of the
same capture.

## Sources

- [3DGRUT](https://github.com/nv-tlabs/3dgrut) and
  [NCore](https://github.com/NVIDIA/ncore)
- [NuRec forum](https://forums.developer.nvidia.com/c/omniverse/platform/nurec/752),
  35 topics read through its JSON endpoints
- [NuRec: reconstruct scenes from mono camera data](https://docs.nvidia.com/nurec/robotics/neural_reconstruction_mono.html)
