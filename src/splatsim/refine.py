"""Refine a flight log's own poses with COLMAP, using them as priors.

    python -m splatsim.refine EXPORT OUT [--sigma-m 2.0]

EXPORT is a `flightlog export --poses FRAME` folder whose poses come from the
drone itself (GPS and gimbal), which are too coarse to train a splat on. OUT
gets `images` (a link to EXPORT's) and `sparse/0`, a COLMAP text model with
refined poses and a refined lens, which 3DGRUT trains from.

The log's positions enter COLMAP as priors with a standard deviation of
`--sigma-m` metres, so the result stays in the log's frame and at its scale.
Each image is matched only against its neighbours in space and in time, which
the priors make possible. The lens is solved as SIMPLE_RADIAL: one focal
length and one radial term, because a log's calibration can be off in both.

Needs the `sfm` extra. Measured once, on 661 frames at 1920x1080 with 8 CPUs:
about two hours, nearly all of it in mapping (docs/TRACKS.md, track 2).
"""

import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pycolmap


def refine(export: str | Path, out: str | Path, sigma_m: float = 2.0) -> None:
    export, out = Path(export).resolve(), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if not (out / "images").exists():
        (out / "images").symlink_to(export / "images")
    camera = json.loads((export / "camera.json").read_text())
    width, height = camera["width"], camera["height"]
    focal = width / (2 * np.tan(np.radians(camera["hfov_deg"]) / 2))
    # trajectory.tum rows are in the order of camera.json's image names.
    centres = np.loadtxt(export / "trajectory.tum", ndmin=2)[:, 1:4]
    prior = dict(zip(camera["images"], centres, strict=True))

    database = out / "database.db"
    database.unlink(missing_ok=True)
    start = time.monotonic()
    reader = pycolmap.ImageReaderOptions()
    reader.camera_model = "SIMPLE_RADIAL"
    reader.camera_params = f"{focal},{width / 2},{height / 2},0"
    pycolmap.extract_features(
        database, export / "images", camera_mode=pycolmap.CameraMode.SINGLE, reader_options=reader
    )
    print(f"refine: features in {time.monotonic() - start:.0f} s", flush=True)

    db = pycolmap.Database.open(database)
    for image in db.read_all_images():
        p = pycolmap.PosePrior()
        p.position = np.asarray(prior[image.name], dtype=np.float64).reshape(3, 1)
        p.position_covariance = np.eye(3) * sigma_m**2
        p.coordinate_system = pycolmap.PosePriorCoordinateSystem.CARTESIAN
        p.corr_data_id = image.data_id
        db.write_pose_prior(p)
    db.close()

    start = time.monotonic()
    spatial = pycolmap.SpatialPairingOptions()
    spatial.max_num_neighbors = 40
    spatial.max_distance = 30.0
    spatial.ignore_z = False
    pycolmap.match_spatial(database, pairing_options=spatial)
    sequential = pycolmap.SequentialPairingOptions()
    sequential.overlap = 8
    pycolmap.match_sequential(database, pairing_options=sequential)
    print(f"refine: matching in {time.monotonic() - start:.0f} s", flush=True)

    start = time.monotonic()
    options = pycolmap.IncrementalPipelineOptions()
    options.use_prior_position = True
    maps = pycolmap.incremental_mapping(
        database, export / "images", out / "models", options=options
    )
    # COLMAP returns several models when the images do not all connect.
    best = max(maps.values(), key=lambda m: m.num_reg_images())
    print(
        f"refine: mapping in {time.monotonic() - start:.0f} s, "
        f"registered {best.num_reg_images()} of {len(prior)} images",
        flush=True,
    )

    sparse = out / "sparse/0"
    if sparse.exists():
        shutil.rmtree(sparse)
    sparse.mkdir(parents=True)
    best.write_text(str(sparse))
    moved = [
        np.linalg.norm(image.projection_center() - prior[image.name])
        for image in best.images.values()
        if image.has_pose
    ]
    lens = next(iter(best.cameras.values()))
    print(
        f"refine: cameras moved {np.mean(moved):.2f} m (mean) from the log's positions; "
        f"focal {focal:.1f} -> {lens.params[0]:.1f} px, radial {lens.params[3]:.4f}; out={out}"
    )


if __name__ == "__main__":
    arguments = sys.argv[1:]
    sigma_m = 2.0
    if "--sigma-m" in arguments:
        at = arguments.index("--sigma-m")
        sigma_m = float(arguments[at + 1])
        del arguments[at : at + 2]
    if len(arguments) != 2:
        sys.exit(__doc__)
    refine(arguments[0], arguments[1], sigma_m)
