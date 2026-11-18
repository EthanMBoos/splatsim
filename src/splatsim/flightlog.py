"""Flight logs: turn MCAP episodes into what the trainer and the replay need.

    python -m splatsim.flightlog export OUT a.mcap [b.mcap ...] [--poses FRAME]

`export` writes into OUT:

    images/NAME.jpg       frames, undistorted to an ideal pinhole
    sparse/0/*.txt        a COLMAP text model, which 3DGRUT trains from
    trajectory.tum        one line per frame: time tx ty tz qx qy qz qw
    camera.json           width, height, hfov_deg and the image names in order

FRAME names the transform to take poses from: `camera_colmap` (the default)
or, for the drone's own estimate, `camera_measured`. The lens comes from the
same frame's calibration topic.

The frames are undistorted once, here, to a pinhole camera with square pixels
and a centred principal point, so that every later step shares one simple
lens.

`trajectory.tum` holds camera-to-world poses with the camera looking down its
own -Z and +Y up, which is USD's convention and the one NVIDIA's tools write.
The COLMAP model holds the same poses in COLMAP's convention. The world frame
is the log's, unchanged.

Reads logs that use Foxglove's protobuf schemas: CompressedImage,
CameraCalibration and FrameTransform. A ROS 2 log needs another decoder.
Verified on PIVOT-mcap episodes only (docs/TRACKS.md).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def read_episode(mcap_path: str | Path, pose_frame: str = "camera_colmap"):
    """Read one MCAP episode. Returns `(lens, frames, points)`.

    `lens` is the calibration of `/{pose_frame}/calibration`: a dict with
    `width`, `height`, `K` (3x3), `D` and `model`. `frames` is a list of
    `(time_s, jpeg_bytes, camera_to_world)` for every image that has a
    `world -> pose_frame` transform at the same log time; the 4x4 matrix uses
    OpenCV camera axes, +Z forward and +Y down. `points` is the log's point
    cloud as `(xyz float32 (n, 3), rgb uint8 (n, 3))`, or None.
    """
    import numpy as np
    from mcap.reader import make_reader
    from mcap_protobuf.decoder import DecoderFactory

    lens, points = None, None
    images, poses = {}, {}
    with open(mcap_path, "rb") as stream:
        reader = make_reader(stream, decoder_factories=[DecoderFactory()])
        for schema, channel, message in reader.iter_messages():
            # Decode only what is used; the log also carries JSON topics.
            if schema.name == "foxglove.CompressedImage":
                images[message.log_time] = _decode(reader, schema, channel, message).data
            elif schema.name == "foxglove.FrameTransform":
                tf = _decode(reader, schema, channel, message)
                if tf.child_frame_id == pose_frame:
                    q, t = tf.rotation, tf.translation
                    matrix = np.eye(4)
                    matrix[:3, :3] = matrix_from_quat_wxyz((q.w, q.x, q.y, q.z))
                    matrix[:3, 3] = (t.x, t.y, t.z)
                    poses[message.log_time] = matrix
            elif channel.topic == f"/{pose_frame}/calibration" and lens is None:
                cal = _decode(reader, schema, channel, message)
                lens = {
                    "width": cal.width,
                    "height": cal.height,
                    "K": np.array(cal.K).reshape(3, 3),
                    "D": np.array(cal.D),
                    "model": cal.distortion_model,
                }
            elif schema.name == "foxglove.PointCloud" and points is None:
                cloud = _decode(reader, schema, channel, message)
                fields = {f.name: f.offset for f in cloud.fields}
                raw = np.frombuffer(cloud.data, dtype=np.uint8).reshape(-1, cloud.point_stride)
                xyz = np.stack(
                    [raw[:, fields[a] : fields[a] + 4].copy().view("<f4")[:, 0] for a in "xyz"], 1
                )
                rgb = np.stack([raw[:, fields[c]] for c in ("red", "green", "blue")], 1)
                points = (xyz, rgb)
    frames = [(t / 1e9, images[t], poses[t]) for t in sorted(images) if t in poses]
    return lens, frames, points


_decoders = {}


def _decode(reader, schema, channel, message):
    from mcap_protobuf.decoder import DecoderFactory

    if schema.id not in _decoders:
        _decoders[schema.id] = DecoderFactory().decoder_for(channel.message_encoding, schema)
    return _decoders[schema.id](message.data)


def matrix_from_quat_wxyz(quat_wxyz):
    import numpy as np

    w, x, y, z = quat_wxyz
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def quat_wxyz_from_matrix(m):
    import numpy as np

    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0:
        s = 2 * np.sqrt(trace + 1)
        return (s / 4, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s)
    i = int(np.argmax(np.diag(m)))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = 2 * np.sqrt(1 + m[i, i] - m[j, j] - m[k, k])
    xyz = [0.0, 0.0, 0.0]
    xyz[i] = s / 4
    xyz[j] = (m[j, i] + m[i, j]) / s
    xyz[k] = (m[k, i] + m[i, k]) / s
    return ((m[k, j] - m[j, k]) / s, *xyz)


def export(
    mcap_paths: list[str],
    out: str | Path,
    *,
    pose_frame: str = "camera_colmap",
    width: int = 1920,
    max_points: int = 200_000,
) -> None:
    """Write the folder described in the module docstring from one or more
    episodes of one scene. All episodes must share a lens, and the output is
    `width` pixels wide.
    """
    import cv2
    import numpy as np

    out = Path(out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "sparse/0").mkdir(parents=True, exist_ok=True)
    names, tum, colmap_images = [], [], []
    pinhole, points = None, None
    for path in mcap_paths:
        lens, frames, cloud = read_episode(path, pose_frame)
        if lens is None or not frames:
            print(f"flightlog: {path} has no {pose_frame} poses, skipped", flush=True)
            continue
        points = points or cloud
        size = (lens["width"], lens["height"])
        fisheye = "fisheye" in lens["model"].lower()
        if fisheye:
            new_k = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
                lens["K"], lens["D"][:4], size, np.eye(3), balance=0.0
            )
        else:
            # alpha 0 keeps only pixels the lens saw, so no black border.
            new_k, _ = cv2.getOptimalNewCameraMatrix(
                lens["K"], lens["D"], size, 0, size, centerPrincipalPoint=True
            )
        # Square pixels and a centred principal point. The larger focal
        # length keeps the view inside the valid region.
        focal = max(new_k[0, 0], new_k[1, 1])
        new_k = np.array([[focal, 0, size[0] / 2], [0, focal, size[1] / 2], [0, 0, 1]])
        height = round(width * size[1] / size[0])
        this = (width, height, round(focal * width / size[0], 3))
        if pinhole is None:
            pinhole = this
        elif abs(this[2] - pinhole[2]) > 0.01 * pinhole[2] or this[:2] != pinhole[:2]:
            raise ValueError(f"{path} has another lens: {this} against {pinhole}")
        stem = Path(path).stem
        # Log times restart in each episode; keep the file monotonic.
        episode_start = tum[-1][0] + 1.0 if tum else 0.0
        for index, (time_s, jpeg, cam_to_world) in enumerate(frames):
            image = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
            if fisheye:
                image = cv2.fisheye.undistortImage(image, lens["K"], lens["D"][:4], Knew=new_k)
            else:
                image = cv2.undistort(image, lens["K"], lens["D"], None, new_k)
            image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
            name = f"{stem}_{index:04d}.jpg"
            cv2.imwrite(str(out / "images" / name), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
            names.append(name)
            # USD camera axes: X right, Y up, Z back.
            usd = cam_to_world[:3, :3] @ np.diag([1.0, -1.0, -1.0])
            w, x, y, z = quat_wxyz_from_matrix(usd)
            tum.append((episode_start + time_s - frames[0][0], *cam_to_world[:3, 3], x, y, z, w))
            world_to_cam = np.linalg.inv(cam_to_world)
            colmap_images.append((quat_wxyz_from_matrix(world_to_cam[:3, :3]), world_to_cam[:3, 3]))
        print(f"flightlog: {stem} frames={len(frames)}", flush=True)
    if pinhole is None:
        raise ValueError("No frames with poses in any episode")

    width, height, focal = pinhole
    sparse = out / "sparse/0"
    (sparse / "cameras.txt").write_text(
        f"1 PINHOLE {width} {height} {focal} {focal} {width / 2} {height / 2}\n"
    )
    with open(sparse / "images.txt", "w") as f:
        for index, (name, (quat, t)) in enumerate(zip(names, colmap_images, strict=True)):
            numbers = " ".join(f"{v:.9f}" for v in (*quat, *t))
            # The second line of each pair lists 2D points; none are kept.
            f.write(f"{index + 1} {numbers} 1 {name}\n\n")
    with open(sparse / "points3D.txt", "w") as f:
        if points is not None:
            xyz, rgb = points
            keep = slice(None, None, max(1, len(xyz) // max_points))
            for index, (p, c) in enumerate(zip(xyz[keep], rgb[keep], strict=True)):
                f.write(f"{index + 1} {p[0]:.5f} {p[1]:.5f} {p[2]:.5f} {c[0]} {c[1]} {c[2]} 0\n")
    with open(out / "trajectory.tum", "w") as f:
        for row in tum:
            f.write(" ".join(f"{v:.9f}" for v in row) + "\n")
    hfov_deg = float(np.degrees(2 * np.arctan(width / (2 * focal))))
    (out / "camera.json").write_text(
        json.dumps({"width": width, "height": height, "hfov_deg": hfov_deg, "images": names})
    )
    print(f"flightlog: frames={len(names)} {width}x{height} hfov_deg={hfov_deg:.2f} out={out}")


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "export":
        arguments = sys.argv[2:]
        frame = "camera_colmap"
        if "--poses" in arguments:
            at = arguments.index("--poses")
            frame = arguments[at + 1]
            del arguments[at : at + 2]
        export(arguments[1:], arguments[0], pose_frame=frame)
    else:
        sys.exit(__doc__)
