from pathlib import Path
from image import load, CalibImage
import matplotlib.pyplot as plt

from skimage.filters import threshold_otsu
from skimage.color import rgb2gray
from skimage.morphology import closing, square, erosion
from skimage.segmentation import clear_border
from skimage.measure import label

import matplotlib.patches as mpatches

demo_path = Path(__file__).parent.parent / "data" / "demo_calib.tif"

print("==================TIFF====================")

#cal_image = CalibImage(demo_path)
cal_image = load(demo_path, for_calib = True)
print(cal_image.tags)

fig, axes = plt.subplots(ncols=3, figsize=(8, 2.5))
ax = axes.ravel()
ax[0] = plt.subplot(1, 3, 1)
ax[1] = plt.subplot(1, 3, 2)
ax[2] = plt.subplot(1, 3, 3)

grayscale = rgb2gray(cal_image.array)
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

ax[0].hist(grayscale.ravel(), bins = 20)
ax[0].axvline(thresh, color='r')

#ax[1].imshow(grayscale, cmap = "gray")
ax[1].imshow(cal_image.array[:,:,2], cmap = "gray")

#regions = regionprops(label_image, tImage, offset = (50,50))
#regions = regionprops(label_image, cal_image.array)
regions = cal_image.region_properties()
for region in regions:
    # take regions with large enough areas
    if region.area >= 100:
        # draw rectangle around segmented coins
        minr, minc, maxr, maxc = region.bbox
        rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                  fill=False, edgecolor='red', linewidth=2)
        ax[1].add_patch(rect)
        print(region.intensity_mean)

ax[1].set_axis_off()
plt.tight_layout()

cal_image = CalibImage(demo_path)
cal = cal_image.get_calibration(doses = [0, 0.5, 1, 2, 4, 6, 8, 10], func = "P3", channel = "G")

cal.plot(ax[2], show = False)

plt.show()

