from pathlib import Path
import argparse, json
import numpy as np, torch
from PIL import Image, ImageDraw, ImageFont

parser = argparse.ArgumentParser()
parser.add_argument("run", type=Path)
parser.add_argument("--step", type=int, required=True)
args = parser.parse_args()
p = args.run / f"prior_samples_{args.step:06d}.pt"
d = torch.load(p, weights_only=True)
font = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
large = ImageFont.truetype(font, 22)
small = ImageFont.truetype(font, 15)
size = 224
gap = 14
canvas = Image.new("RGB", (4 * (size + gap) + gap, 620), (18, 22, 27))
draw = ImageDraw.Draw(canvas)
draw.text(
    (gap, 10),
    f"Normal MRI prior: training step {args.step:,}",
    font=large,
    fill="white",
)
draw.text(
    (gap, 43),
    "Random samples given MNI position only. These are NOT patient-matched inpainting results.",
    font=small,
    fill=(255, 203, 120),
)
for row, (label, key) in enumerate(
    [
        ("Nonlesional MRI references at these positions", "reference"),
        ("Generated normal-prior samples", "images"),
    ]
):
    y = 85 + row * 260
    draw.text((gap, y), label, font=small, fill="white")
    for i, a in enumerate(d[key]):
        im = np.flipud(a[0].float().numpy().T)
        tile = Image.fromarray(np.uint8(np.clip((im + 1) * 127.5, 0, 255))).resize(
            (size, size), Image.Resampling.NEAREST
        )
        canvas.paste(tile, (gap + i * (size + gap), y + 25))
draw.text(
    (gap, 605),
    "Superior is up. Identical intensity display; no output sharpening or averaging.",
    font=small,
    fill=(220, 223, 228),
)
out = args.run / f"prior_samples_{args.step:06d}.png"
canvas.save(out)
print(out)
