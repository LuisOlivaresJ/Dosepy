from pathlib import Path
from image import load

demo_path = Path(__file__).parent.parent / "data" / "demo_calib.tif"
img  = load(demo_path)
print(img.array.shape)