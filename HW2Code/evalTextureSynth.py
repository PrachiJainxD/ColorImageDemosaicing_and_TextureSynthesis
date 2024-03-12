import numpy as np
from skimage import io
from skimage import color
import matplotlib.pyplot as plt 
from scipy import ndimage
import random
import time
import warnings

import warnings
warnings.filterwarnings('ignore')

def synthRandomPatch(image, tilesize, numtile, outsize):
    '''
    Texture Synthesis using Random tiling  
    '''
    image2 = color.rgb2gray(image[:,:,:])
    synthimage = np.zeros((outsize,outsize))
    height, width = image2.shape

    for i in range(0, outsize, tilesize): 
        for j in range(0,outsize,tilesize): 
            tileX, tileY = tilesize, tilesize
            if(i + tilesize) >= outsize:
                tileX = outsize - i
            if(j + tilesize) >= outsize:
                tileY = outsize - j

            rowX = random.randint(0, height - tileX)
            colY = random.randint(0, width - tileY)

            image3 = image2[rowX:rowX + tileX, colY:colY + tileY]
            synthimage[i:i + tileX, j:j + tileY] = image3
    
    return synthimage
  
def synthEfrosLeung(image, winsize, outsize):
    '''
    Non-parametric Texture Synthesis using Efros & Leung algorithm 
    '''
    image1 = color.rgb2gray(image[:,:,:])
    ErrThreshold = 0.1

    height, width = image1.shape  
    synthimage = np.zeros((outsize, outsize))

    for i in range(height):
        for j in range(width):
            if image1[i][j] == 0:
                image1[i][j] = 1e-6
    
    centerX, centerY = int(outsize/2), int(outsize/2)
    rowX = random.randint(0, height - 3)
    columnY = random.randint(0, width -3)
    synthimage[centerX - 1:centerX + 2, centerY - 1:centerY + 2] = image1[rowX:rowX + 3, columnY:columnY + 3]
    
    pixelList = []
    structure = ndimage.generate_binary_structure(2, 2)
    NMask = ndimage.binary_dilation(synthimage, structure = structure) 

    counter = 0
    for i in range(outsize):
        for j in range(outsize):
            if NMask[i][j]!=0 and synthimage[i][j]==0:
                pixelList.append((i,j))
                counter += 1
                
    prevList = None
    
    while pixelList:
        prevList = pixelList
        pixel = {}
        for p in pixelList:
            score = np.sum(synthimage[p[0]-1:p[0]+2, p[1]-1:p[1]+2] >0)
            pixel[(p[0], p[1])] = score
        
        sortedPixList = [k for k, v in sorted(pixel.items(), key= lambda x: x[1], reverse=True)]
        
        for p in sortedPixList:
            winXS, winXE = p[0] - int(winsize/2), p[0] + int(winsize/2) + 1
            winYS,winYE  = p[1] - int(winsize/2), p[1] + int(winsize/2) + 1
            if p[0] - int(winsize/2) < 0:
                winXS = 0
                winXE = winsize
            if p[0] + int(winsize/2) + 1 > outsize:
                winXS = outsize - winsize
                winXE = outsize
            if p[1] - int(winsize/2) < 0:
                winYS = 0
                winYE = winsize
            if p[1] + int(winsize/2) + 1 > outsize:
                winYS = outsize - winsize
                winYE = outsize
            
            outputPatch = synthimage[winXS:winXE, winYS:winYE]
            ValidMask = outputPatch > 0
            minSSD = float('inf')            
            SSD = {}
            
            for row in range(0, height - winsize):
                for col in range(0,width - winsize):
                    patch = image1[row:row + winsize, col:col + winsize]
                    difference = (patch * ValidMask) - outputPatch
                    SSD[(row,col)] = np.sum(np.multiply(difference, difference))
                    minSSD = min(minSSD, np.sum(np.multiply(difference, difference)))
                
            BestMatches = []
            for pixel in list(SSD.keys()):
                if SSD[pixel] >= minSSD and SSD[pixel] <= (minSSD + ErrThreshold):
                    BestMatches.append(pixel)

            bmlegth = len(BestMatches)
            pickRPixel = BestMatches[random.randint(0, bmlegth -1)]           
            pickPatch = image1[pickRPixel[0]:pickRPixel[0] + winsize, pickRPixel[1]:pickRPixel[1] + winsize]

            outputPatchHeight, outputPatchWidth = outputPatch.shape
            
            for row in range(outputPatchHeight):
                for col in range(outputPatchWidth):
                    if outputPatch[row,col]==0:
                        outputPatch[row,col] = pickPatch[row,col]
                        
            synthimage[winXS:winXE, winYS:winYE] = outputPatch
                
        updateList = []
        structure1 = ndimage.generate_binary_structure(2, 2)
        NMask1 = ndimage.binary_dilation(synthimage, structure = structure1) 

        counter1 = 0
        for i in range(outsize):
            for j in range(outsize):
                if NMask1[i][j]!=0 and synthimage[i][j]==0:
                    updateList.append((i,j))
                    counter1 += 1
        
        pixelList = updateList
        
    return synthimage

##########################################################################################################
# TESTING                                                                                                #
##########################################################################################################

#Load images
img1 = io.imread('../data/texture/D20.png')
img2 = io.imread('../data/texture/Texture2.bmp')
img3 = io.imread('../data/texture/english.jpg')

#UNCOMMENT TO TEST RANDOM PATCHES
"""
randomPaths = [img1, img2, img3]
name = ['D20.png','Texture2.png','english.png']
counter = 0
for img in randomPaths:
    iname = name[counter]
    counter = counter + 1
    outsize = 480
    #Different Block sizes
    for tilesize in [5, 15, 20, 30, 40]:
        numtile = int(outsize/tilesize)
        #Record Runtimes
        start = time.time()
        randomPatch = synthRandomPatch(img, tilesize, numtile, outsize)
        end = time.time()
        outfile_path = "RandomTextureSynthesis{}x{}_{}".format(tilesize, tilesize,iname) 
        print("{}, Runtime ={:.6f}".format(outfile_path, (end - start)))  
        #Displaying Random Patch output
        plt.subplot(3, 4, 1)        
        plt.imshow(randomPatch, cmap = 'gray')
        plt.title("Synthesized Random Image = {}".format(iname))
        #Save Random Patch output
        plt.imsave(outfile_path, randomPatch, cmap = "gray")

"""
#Testing Non-parametric Texture Synthesis using Efros & Leung algorithm  
EfrosLeungPaths = [img1, img2, img3]
name = ['D20.png','Texture2.png','english.png']
counter1 = 0
for img in EfrosLeungPaths:
    iname = name[counter1]
    counter1 = counter1 + 1
    outsize = 70
    #Different Window sizes
    for winsize in [5, 7, 11, 15]:
        #Record Runtimes
        start = time.time()
        Patch = synthEfrosLeung(img, winsize, outsize)
        end = time.time()
        outfile_path = "EfrosLeungTextureSynthesis{}x{}_{}".format(winsize, winsize, iname) 
        print("{}, Runtime ={:.6f}".format(outfile_path, (end - start)))  
        #Displaying Patch output
        plt.subplot(3, 4, 1)        
        plt.imshow(Patch, cmap = 'gray')
        plt.title("Synthesized Efros & Leung Image = {}".format(iname))
        #Save Patch output
        plt.imsave(outfile_path, Patch, cmap = "gray")
    