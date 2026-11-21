# Paper ideas

Written 2026-11-21, from a literature search on where this work could be
published. The claims about what exists rest on a few dozen searches each, not
an exhaustive review, so the two checks at the end come first. The
measurements these ideas build on are in [TRACKS.md](TRACKS.md).

The paper in this repository is a measurement paper, not a new method and not
a simulator.

## The gap

Drone papers train policies inside splats and then count successful real
flights. None of them checks whether a detector sees a rendered frame the same
way it sees the real one. Driving research has measured this since 2023. No
equivalent was found for drones, or for replaying a flight log.

## Options

1. **How far can a replay be trusted? (the main paper)**
   Measure how detector and keypoint agreement fall off as the replayed camera
   moves away from the flights the splat was trained on. Compare a simple
   distance-and-angle score against existing uncertainty methods.
   Needs: 5 to 10 places, each flown several times. Venue: RA-L.

2. **Fly again or repair?**
   When a replay is bad, which fixes it best: one more real flight, depth
   priors, or a generative fixer (Difix3D+)? Judge by detector agreement, not
   by how the image looks. Uses the same data as option 1.

3. **Mesh or splat?**
   Isaac Sim can now render a splat and a mesh of the same place in one scene.
   Compare which one predicts real perception behavior better. This is the
   link to [isaacdrones](https://github.com/EthanMBoos/isaacdrones).

4. **A small first step**
   A workshop paper with 2 to 3 places. It grows into option 1.

## Not papers

- Another drone simulator.
- Pose optimization from telemetry in 3DGRUT. Good pull request, not a paper.
- A trust score as a new method. Others already have one (PRIMU, OpenFlyScan).
- World models or language policies for drones. Crowded, and all need real
  flights.

## What is missing today

- Flight data collected for this purpose. One street from a public dataset is
  only a pilot.
- A new name. "SplatSim" is already an ICRA 2025 paper on manipulation
  (arXiv 2409.10161).

## Check before committing

Either of these could already cover option 1:

- Lindström's 2026 thesis (the driving version of this study).
- OpenFlyScan, arXiv 2609.24253 (the closest drone work, September 2026).
