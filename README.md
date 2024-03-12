**1. Color image demosaicing**

- In digital cameras the red, blue, and green sensors are interlaced in the Bayer pattern and missing values are interpolated to obtain a color image.
- In this project I implemented several interpolation algorithms such as:
  a. Nearest-neighbour interpolation
  b. Linear interpolation
  c. Adaptive Gradient interpolation
  
**2. Texture synthesis**
- A simple approach for generating a new texture image is to randomly tile the target image using patches from a source image. [Random tiling + Non-parametric sampling]
- It results in an image with artifacts around the edge of the tiles. This is because the tiles do not align well.
- The approach of Efros and Leung avoids this by generating the ouput, one pixel at a time, by matching the local neighborhood
of the pixels.
