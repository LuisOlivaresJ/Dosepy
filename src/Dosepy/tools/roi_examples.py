from pathlib import Path
from image import load
import matplotlib.pyplot as plt
from tifffile import TiffFile, imread

from skimage.filters import threshold_otsu
from skimage.color import rgb2gray
from skimage.morphology import closing, square, erosion
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops

import matplotlib.patches as mpatches


demo_path = Path(__file__).parent.parent / "data" / "demo_calib.tif"

print("==================Pillow====================")
#img  = load(demo_path)
#print(img.array.shape)
#print(img.array.dtype)
#print(img.info)

print("==================TIFF====================")
tImage = load(demo_path)
#print(tImage.dpmm)

fig, axes = plt.subplots(ncols=4, figsize=(8, 2.5))
ax = axes.ravel()
ax[0] = plt.subplot(1, 4, 1)
ax[1] = plt.subplot(1, 4, 2)
ax[2] = plt.subplot(1, 4, 3)
ax[3] = plt.subplot(1, 4, 4)

grayscale = rgb2gray(tImage.array)
thresh = threshold_otsu(grayscale)

# Apply threshold and close small holes with binary closing.
# 9 = 75dpi * 3 mm, 9 is used to close smaller holes than 3 mm.

#binary = grayscale > thresh
binary = closing(grayscale < thresh, square(9)) 

# remove artifacts connected to image border
cleared = clear_border(binary)

# label image regions
label_image = label(cleared)
label_image = erosion(label_image, square(24))

ax[0].imshow(grayscale)
#ax[1].hist(img.array.ravel(), bins = 256)
ax[1].hist(grayscale.ravel(), bins = 20)
ax[1].axvline(thresh, color='r')

ax[2].imshow(binary, cmap=plt.cm.gray)
ax[2].set_title('Binary')
ax[2].axis('off')

#plt.show()

ax[3].imshow(tImage.array/(2**16))

#regions = regionprops(label_image, tImage, offset = (50,50))
regions = regionprops(label_image, tImage.array)
for region in regions:
    # take regions with large enough areas
    if region.area >= 100:
        # draw rectangle around segmented coins
        minr, minc, maxr, maxc = region.bbox
        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                  fill=False, edgecolor='red', linewidth=2)
        ax[3].add_patch(rect)
        print(region.intensity_mean[0])

ax[3].set_axis_off()
plt.tight_layout()
plt.show()

print(len(regions))