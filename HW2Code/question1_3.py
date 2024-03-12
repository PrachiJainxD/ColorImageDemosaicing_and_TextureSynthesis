from skimage import io, color
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

img = io.imread("../data/cat.png")
img = color.rgb2gray(img[:,:,:])

height, width = img.shape
pixel = []
topNeigh = []
allNeigh = []
#For each pixel in the image except those on the borders (removing layer of 1 pixel around the borders)
for i in range(1, height - 1):
    for j in range(1, width - 1):
        pixel.append(img[i,j])
        topNeigh.append(img[i - 1,j])
        allNeigh.append(np.mean([img[i - 1, j], img[i, j - 1], img[i + 1, j], img[i, j + 1]]))

sns.set_style(style='whitegrid')
p1 = sns.scatterplot(pixel, topNeigh)
p1.set(xlabel ="Pixel Value at [i, j] position", ylabel = "Left Neighbor: Pixel Value at [i-1, j] position", title ='Pixel vs Top neighbor')
plt.show()
p2= sns.scatterplot(pixel, allNeigh)
p2.set(xlabel ="Pixel Value at [i, j] position", ylabel = "Avg of Left, Right, Top and Bottom Neighbors", title='Pixel vs Average of all neighbor')
plt.show()

