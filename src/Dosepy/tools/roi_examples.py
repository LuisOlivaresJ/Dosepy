from pathlib import Path
from image import load
import matplotlib.pyplot as plt
from tifffile import TiffFile

demo_path = Path(__file__).parent.parent / "data" / "demo_calib.tif"
img  = load(demo_path)
#print(img.array.shape)
#print(img.array.dtype)
#print(img.info)

# Get information about the image stack in the TIFF file without reading any image data:
tif = TiffFile(demo_path)
page = tif.pages[0]
print(page.shape)
print(page.dtype)
print(page.tags['XResolution'].value[0])
"""
fig, axes = plt.subplots(ncols=3, figsize=(8, 2.5))
ax = axes.ravel()
ax[0] = plt.subplot(1, 1, 1)

ax[0].hist(img.array.ravel(), bins = 256)

plt.show()
"""