"""Replay: render a trained splat at every frame of a flight and score it.

    python src/splatsim/replay.py CHECKPOINT DATA OUT [--shift X Y Z]

CHECKPOINT is a 3DGRUT checkpoint (`ckpt_last.pt`). DATA is a
`flightlog export` folder; it need not be the one the splat was trained on.
OUT gets `frames/NAME.png` for every frame of DATA and `scores.csv` with PSNR,
SSIM and LPIPS of each render against its logged frame.

`--shift` moves every camera by that many metres in the world frame before
rendering. No logged frame exists for those views, so nothing is scored.

This file runs under 3DGRUT's Python, not splatsim's: it imports `threedgrut`
and torch and nothing from splatsim. 3DGRUT's own `render.py` does the same
job for every 8th frame only and names its output by index.
"""

import argparse
import csv
import time
from pathlib import Path

import torch
import torchvision
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

import threedgrut.datasets as datasets
from threedgrut.model.model import MixtureOfGaussians
from threedgrut.utils.render import apply_background

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("checkpoint")
parser.add_argument("data")
parser.add_argument("out")
parser.add_argument("--shift", type=float, nargs=3, metavar=("X", "Y", "Z"))
args = parser.parse_args()

checkpoint = torch.load(args.checkpoint, weights_only=False)
conf = checkpoint["config"]
if conf.post_processing.method is not None or "feature_decoder" in checkpoint:
    raise SystemExit("replay: this checkpoint needs post-processing, which replay does not apply")
conf.path = args.data
# Zero makes the test split every frame; the trainer's default is every 8th.
conf.dataset.test_split_interval = 0

model = MixtureOfGaussians(conf)
model.init_from_checkpoint(checkpoint, setup_optimizer=False)
model.build_acc()
dataset = datasets.make_test(name=conf.dataset.type, config=conf)
loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8)

frames = Path(args.out) / "frames"
frames.mkdir(parents=True, exist_ok=True)
ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to("cuda")
lpips = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True).to("cuda")
shift = torch.tensor(args.shift, device="cuda") if args.shift else None

rows, render_ms = [], []
with torch.no_grad():
    for index, batch in enumerate(loader):
        gpu = dataset.get_gpu_batch_with_intrinsics(batch)
        if shift is not None:
            gpu.T_to_world[..., :3, 3] += shift
        torch.cuda.synchronize()
        start = time.perf_counter()
        outputs = apply_background(model.background, model(gpu), gpu, training=False)
        torch.cuda.synchronize()
        render_ms.append((time.perf_counter() - start) * 1000)

        render = outputs["pred_features"].clip(0, 1)  # [1, H, W, 3]
        name = Path(str(dataset.image_paths[index])).stem
        torchvision.utils.save_image(render[0].permute(2, 0, 1), frames / f"{name}.png")
        if shift is None:
            logged = gpu.rgb_gt
            psnr = -10 * torch.log10(torch.mean((render - logged) ** 2))
            a, b = render.permute(0, 3, 1, 2), logged.permute(0, 3, 1, 2)
            rows.append(
                (
                    name,
                    f"{psnr.item():.3f}",
                    f"{ssim(a, b).item():.4f}",
                    f"{lpips(a, b).item():.4f}",
                )
            )

# The first frame includes one-time setup, so the median is the honest figure.
print(
    f"replay: frames={len(render_ms)} render_ms_median={sorted(render_ms)[len(render_ms) // 2]:.1f}"
)
if rows:
    with open(Path(args.out) / "scores.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("image", "psnr_db", "ssim", "lpips"))
        writer.writerows(rows)
    mean = [sum(float(r[k]) for r in rows) / len(rows) for k in (1, 2, 3)]
    print(f"replay: psnr_db={mean[0]:.2f} ssim={mean[1]:.3f} lpips={mean[2]:.3f} out={args.out}")
