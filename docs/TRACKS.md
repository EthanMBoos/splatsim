# Research tracks

Written 2026-11-16. Each track below was probed for a few hours on PACE ICE
that day, to find out whether it holds a hard wall, an easy win or an open
question before anyone is assigned to it. A track states what was measured,
what that rules in or out, and a first step. The reasons the tracks exist are
in [RESEARCH.md](RESEARCH.md); how to use the code is in the
[README](../README.md).

Every number is one run unless a count is given, so treat differences under
0.5 dB as noise. A number that `replay.py` does not print came from a one-off
script that is not in the repository, so repeating it means writing the
script from the track's description.

## The data all tracks share

**PIVOT-mcap is the flight-log dataset.** `Voxel51/PIVOT-mcap` on Hugging Face
holds drone flights as MCAP files with a 4K camera feed, the drone's own GPS
and gimbal poses, COLMAP poses solved offline, and calibration for both. It is
the only public dataset we found with a flight log in the form our goal
assumes. Its licence is CC BY-NC 4.0. Its timestamps are synthetic, at 10
frames per second, and its messages use Foxglove's protobuf schemas, not ROS 2.

**One scene was used: `village_street`.** It is a street 52 m across with 17
flights, of which we used seven. Six of them (three inward orbits at
different heights, an outward orbit and two overhead passes; 661 frames)
trained the splat. The
seventh, `traverse_loop_low`, flies along the street at low height (291
frames) and was held out as the flight to replay. The other ten, which
include four more low traverses, are untouched.

**The pipeline runs end to end.** `splatsim.flightlog export` wrote 661
undistorted 1920x1080 frames with poses in about 3 minutes. 3DGRUT trained a
3DGUT splat of 6.7 million Gaussians from them in 20 minutes on an H100.
`replay.py` renders a frame of the held-out flight in 11.5 ms on an H100
(median of 291) and scores the flight at 12.56 dB.

**The trainer reproduces a published figure.** On a standard indoor scene, a
room of 311 photos, 3DGUT trained in 8 minutes on an H100 and scored 31.7 dB
on the 39 held-out photos, which matches the figure its authors publish. The
village's 23 dB is therefore a property of the capture, not of our setup.

## 1. How far from the capture a replay can be trusted

**This is the hard wall, and it is closer than we assumed.** On frames the
splat was trained near, it scores 23 dB. On the held-out flight through the
same street it scores 12.5 dB, which is an unusable image: smeared facades
and floating debris.

| Frames | PSNR |
|---|---|
| Six training flights, every 8th frame (83, not trained on) | 22.98 dB |
| Held-out flight, every 8th frame (37) | 12.47 dB |

The held-out flight's median distance to the nearest training camera is
6.2 m. PSNR falls with that distance (correlation -0.76): 14.6 dB
for the 30 frames within 0.25 to 0.5 m, 11.5 dB for the 176 frames beyond 4 m.
Position is not the whole cause: 286 of the 291 frames look more than 40
degrees away from the direction of their nearest training camera, and frames
within half a metre of one still score under 15 dB. An orbit looks inward and
down; a traverse looks along the street.

**A splat trained on the flight itself replays it.** Trained on
`traverse_loop_low` alone (12 minutes), 3DGUT scores 20.92 dB on that
flight's own held-out frames. So the goal's loop, a splat from the flight
that is then replayed, is feasible. What fails is leaving that flight's path,
which is what swapping a height, heading or lens requires.

**Adding the flight to the training set repairs its replay.** A splat trained
on all seven flights (952 frames, 18 minutes) replays the traverse at 18.5 dB
on that flight's 36 held-out frames, against 12.5 dB without it. One pass
along a path is therefore enough to replay that path, and six other flights
of the same street are not.

Open questions, each a project:

- A trust score per rendered frame, predicted from camera geometry alone
  (distance and angle to the training cameras, or how many of them saw each
  visible surface), checked against measured PSNR. It would tell a user of the
  tool which parts of a counterfactual flight to believe.
- The capture pattern a replay needs. Which extra flight buys the most: a
  second traverse at another height, the same one reversed, or an orbit?
- Whether priors help off the path: depth or normal regularisation, or the
  generative repair that NVIDIA's Fixer-style models do. Untried.

First step: reproduce the table, then plot PSNR against distance and angle for
splats trained on one, two and four flights.

## 2. Poses from the log

**The drone's own poses cannot train a splat as they are.** PIVOT's GPS and
gimbal poses lie 1.39 m (mean) from its COLMAP poses, and the measured focal
length is 12% shorter than the one COLMAP solved (2524 against 2836 pixels at
4K). 3DGUT trained on them reaches 14.80 dB and SSIM 0.33, against 22.98 dB
and 0.72 with COLMAP's. 3DGRUT has no camera pose optimisation, which we
read in its source, so it cannot absorb the error.

**COLMAP repairs the log's poses when given them as priors, slowly.** The
log's positions went into COLMAP 4.2 (pycolmap, CPU only) as priors with a 2 m
standard deviation, with matching limited to neighbours in space. All 661
images registered. After the best similarity fit the solved camera positions
lie 0.063 m (mean) from PIVOT's offline COLMAP positions, with a scale factor
of 0.984; without the fit they lie 1.17 m away, because the solution stays
in the GPS frame, which is offset from PIVOT's. COLMAP also corrected the
lens: a focal length of 1437 pixels at 1920 wide against the log's 1285, and
a radial term of 0.11 that the log's calibration had not removed. It took
152 s for features, 740 s for matching and 6080 s for mapping on 8 CPUs.

**The refined poses train as good a splat as the reference poses.** With the
solved radial term kept, 3DGUT reaches 23.64 dB and SSIM 0.75 on its held-out
frames (21 minutes), against 22.98 dB and 0.72 from PIVOT's poses. With the
radial term dropped and the lens treated as a pinhole it reaches 19.09 dB, so
the lens matters as much as the poses. The two test sets are the same frames
undistorted with different calibrations, so the 0.7 dB difference is not a
ranking. The path from a log's own poses to a metric splat therefore exists:
it costs two hours of COLMAP on 8 CPUs per 661 frames, and none of it is in
`flightlog.py` yet. That run's USD export failed validation after training;
its PLY and checkpoint were written.

Open questions:

- Pose and lens optimisation inside 3DGRUT, started from the log's values,
  to replace the two hours of COLMAP. 3DGRUT merges outside pull requests,
  so this is also the most plausible upstream contribution in this document.
- How much pose error training tolerates. PIVOT gives both pose sets, so the
  error can be dialled between them.
- Real flight video. PIVOT's frames are sharp stills at 10 Hz. Motion blur and
  rolling shutter are untested here. NCore's COLMAP converter (787 lines)
  carries shutter type and per-frame exposure times, and is the template for a
  converter that keeps them.

First step: run the prior-seeded COLMAP step, now `splatsim.refine`, on a
second PIVOT scene, and replay the result, which we did not do for the
refined splat.

## 3. Whether a detector sees the render as it sees the log

**At 21 dB, half the detections and three quarters of the features are
lost.** We ran two off-the-shelf algorithms on each logged frame and on a
render of it, every fourth frame. The renders came from NVIDIA's RTX
renderer, which scores the same splat about 2 dB below `replay.py`; the
measurement has not been repeated on `replay.py`'s frames.

| Replay | PSNR | SIFT keypoints matched at the same place | Detections found again |
|---|---|---|---|
| Training flights (166 frames) | 21.4 dB | 22.7% | 53.8% |
| Held-out flight (73 frames) | 12.6 dB | 1.6% | 27.2% |

A keypoint counts if its match lies within 3 pixels of the same position. A
detection (Faster R-CNN on COCO classes, score above 0.5) counts if the render
has one of the same class with IoU above 0.5. The detector is a poor fit for
the scene: over half of its detections are "potted plant".

This is the direct test of the bet against simulation, and the first reading
is not favourable: a replay that looks plausible changes what a detector
reports. Open questions: which image score predicts detector agreement; how
agreement rises with splat quality; whether a detector fine-tuned on renders
transfers. First step: repeat with a detector and classes that suit aerial
footage, on frames from `replay.py`.

## 4. Reading other logs

**The reader covers one convention.** `flightlog.py` decodes Foxglove
protobuf messages as PIVOT writes them. A ROS 2 log needs another decoder.

Untried and open: a real log from our own drone; lens models beyond the
undistortion to a pinhole that `export` does; automatic exposure; and a
converter to NCore that keeps rolling-shutter timing.
