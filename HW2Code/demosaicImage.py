# This code is part of:
#
#   CMPSCI 670: Computer Vision, Fall 2022
#   University of Massachusetts, Amherst
#   Instructor: Subhransu Maji

import numpy as np
import matplotlib.pyplot as plt 
import math

def rescale(A):
    return (A - np.min(A))/(np.max(A) - np.min(A))

def descale(A):
    return A*(np.max(A) - np.min(A)) + np.min(A)
   
    
def demosaicImage(image, method):
    ''' Demosaics image.

    Args:
        img: np.array of size NxM.
        method: demosaicing method (baseline or nn).

    Returns:
        Color image of size NxMx3 computed using method.
    '''

    if method.lower() == "baseline":
        return demosaicBaseline(image.copy())
    elif method.lower() == 'nn':
        return demosaicNN(image.copy()) # Implement this
    elif method.lower() == 'linear':
        return demosaicLinear(image.copy()) # Implement this
    elif method.lower() == 'adagrad':
        return demosaicAdagrad(image.copy()) # Implement this
    elif method.lower() == 'divide':
        return demosaicDivide(image.copy()) # Implement this
    elif method.lower() == 'logdivide':
        return demosaicLogDivide(image.copy()) # Implement this        
    else:
        raise ValueError("method {} unkown.".format(method))


def demosaicBaseline(img):
    '''Baseline demosaicing.
    
    Replaces missing values with the mean of each color channel.
    
    Args:
        img: np.array of size NxM.

    Returns:
        Color image of sieze NxMx3 demosaiced using the baseline 
        algorithm.
    '''
    mos_img = np.tile(img[:, :, np.newaxis], [1, 1, 3])
    image_height, image_width = img.shape

    red_values = img[1:image_height:2, 1:image_width:2]
    mean_value = red_values.mean()
    mos_img[:, :, 0] = mean_value
    mos_img[1:image_height:2, 1:image_width:2, 0] = img[1:image_height:2, 1:image_width:2]

    blue_values = img[0:image_height:2, 0:image_width:2]
    mean_value = blue_values.mean()
    mos_img[:, :, 2] = mean_value
    mos_img[0:image_height:2, 0:image_width:2, 2] = img[0:image_height:2, 0:image_width:2]

    mask = np.ones((image_height, image_width))
    mask[0:image_height:2, 0:image_width:2] = -1
    mask[1:image_height:2, 1:image_width:2] = -1
    green_values = mos_img[mask > 0]
    mean_value = green_values.mean()

    green_channel = img
    green_channel[mask < 0] = mean_value
    mos_img[:, :, 1] = green_channel

    return mos_img


def demosaicNN(img):
    '''
    Nearest neighbor demosaicing.
    
    Args:
        img: np.array of size NxM.
        
    '''
    mos_img = np.tile(img[:, :, np.newaxis], [1, 1, 3])
    image_height, image_width = img.shape
    
    #Blue channel (odd rows and columns)
    blue_values = np.ones([image_height, image_width])#*(-1)
    blue_values[0:image_height:2, 0:image_width:2] = img[0:image_height:2, 0:image_width:2] 
    
    #For every alternate row, starting from 1st row
    for i in range(0, image_height, 2):
    #For every alternate column, starting from 1st column
        for j in range(0, image_width, 2):
        #Put the same value as that of known pixel's, to the pixel at right
            if( i + 1 < image_height):
                blue_values[i+1, j] = blue_values[i, j]
        #Put the same value as that of known pixel's, to the pixel at bottom        
            if( j + 1 < image_width):
                blue_values[i, j+1] = blue_values[i, j]              
        #Put the same value as that of known pixel's, to the pixel at bottom right                
            if( i + 1 < image_height) and ( j + 1 < image_width):
                blue_values[i+1, j+1] = blue_values[i, j]      
    
    mos_img[:, :, 2] = blue_values
    
    #Red channel (even rows and columns)
    red_values = np.ones([image_height, image_width])#*(-1)
    red_values[1:image_height:2, 1:image_width:2] = img[1:image_height:2, 1:image_width:2]    

    #For every alternate row, starting from 2nd row    
    for i in range(1, image_height, 2):
    #For every alternate column, starting from 2nd column
        for j in range(1, image_width, 2):
        #Put the same value as that of known pixel's, to the pixel at right
            if( i + 1 < image_height):
                red_values[i+1, j] = red_values[i, j]
        #Put the same value as that of known pixel's, to the pixel at bottom        
            if( j + 1 < image_width):
                red_values[i, j+1] = red_values[i, j]              
        #Put the same value as that of known pixel's, to the pixel at bottom right                
            if( i + 1 < image_height) and ( j + 1 < image_width):
                red_values[i+1, j+1] = red_values[i, j]
    
    # For the first row, copy the values from 2nd row, they are the "nearest" 
    red_values[1,:] = red_values[2,:]
    # For the first column, copy the values from 2nd column, they are the "nearest" 
    red_values[:,1] = red_values[:,2]

    #Make the interpolated red value matrix, the red channel of the final output image    
    mos_img[:, :, 0] = red_values
    
    #Green Channel
    green_values = np.ones([image_height, image_width])#*(-1)
    
    #Odd rows have green pixel's known values at even columns 
    green_values[0:image_height:2, 1:image_width:2] = img[0:image_height:2, 1:image_width:2] 
    #Even rows have green pixel's known values at odd columns     
    green_values[1:image_height:2, 0:image_width:2] = img[1:image_height:2, 0:image_width:2]
    
    #Odd rows
    for i in range(0, image_height, 2):
        #Even columns
        for j in range(1, image_width, 2):
        #Put the same value as that of known pixel's, to the pixel at left
            if(i-1 > 0):
               green_values[i-1, j] = green_values[i, j]
               
    #Even rows
    for i in range(1, image_height, 2):
        #Odd columns
        for j in range(0, image_width, 2):
        #Put the same value as that of known pixel's, to the pixel at right
            if(i+1< image_height):
               green_values[i+1, j] = green_values[i, j]
               
    mos_img[:, :, 1] = green_values   
    
    return  mos_img


def demosaicLinear(img):
    '''
    Linear Interpolation
    Args:
        img: np.array of size NxM.
    '''
    mos_img = np.tile(img[:, :, np.newaxis], [1, 1, 3])
    image_height, image_width = img.shape
    
    #Blue
    blue_values = np.zeros((image_height, image_width))
    blue_values[0:image_height:2, 0:image_width:2] = 1
    blue = mos_img[:, :, 2] * blue_values
    
    #Red
    red_values = np.zeros((image_height, image_width))
    red_values[1:image_height:2, 1:image_width:2] = 1
    red = mos_img[:, :, 0] * red_values
    
    #Green    
    green = mos_img[:,:,1] - (blue + red)
    
    #Interpolation
    interpolateB  = np.argwhere(blue!=0)

    blue = np.hstack((blue, np.zeros((image_height, 2))))
    blue = np.vstack((blue, np.zeros((2, image_width + 2))))
    
    array1= np.zeros_like(blue)
    array2= np.zeros_like(blue)
    array3= np.zeros_like(blue)
    
    for x in interpolateB:
        xh, xw = x[0], x[1]
        array1[xh+1, xw+1] = np.divide((blue[xh,xw]+blue[xh,xw+2]+blue[xh+2,xw]+blue[xh+2,xw+2]), 4)
        array2[xh+1, xw] = np.divide((blue[xh,xw]+blue[xh+2,xw]), 2)
        array3[xh, xw+1] = np.divide((blue[xh,xw]+blue[xh,xw+2]), 2)

    blue = array1 + array2 + array3 + blue
    
    blue[:,image_width - 1] = blue[:, image_width - 2]
    blue[image_height-1,:]= blue[image_height-2,:]    
    mos_img[:, :, 2] = blue[0:image_height, 0:image_width]
    
    interpolateR = np.argwhere(red!=0)
    
    red = np.hstack((red, np.zeros((image_height, 2))))
    red = np.vstack((red, np.zeros((2,image_width + 2))))
    
    array1= np.zeros_like(red)
    array2= np.zeros_like(red)
    array3= np.zeros_like(red)
    
    for x in interpolateR :
        xh, xw = x[0], x[1]
        array1[xh+1, xw+1] = np.divide((red[xh,xw]+red[xh,xw+2]+red[xh+2,xw]+red[xh+2,xw+2]), 4)
        array2[xh+1, xw] = np.divide((red[xh,xw]+red[xh+2,xw]), 2)
        array3[xh, xw+1] = np.divide((red[xh,xw]+red[xh,xw+2]), 2)
        
    red = array1 + array2 + array3 + red
    red[0,:] = red[1,:]
    red[:,0] = red[:,1]
    mos_img[:, :, 0] = red[0:image_height, 0:image_width]
    
    interpolateG = np.argwhere(green==0)
    green = np.hstack((green, np.zeros((image_height,1))))
    green = np.vstack((green, np.zeros((1,image_width+1))))
    
    for x in interpolateG:
        xh, xw = x[0], x[1]
        if((xh!=image_height and xw!=image_width)):
            green[xh,xw]= np.divide((green[xh,xw-1] + green[xh-1,xw] + green[xh+1,xw]+ green[xh,xw+1]),4)
            
        if (xh == 0 and xw!=image_width) or (xh!=image_height and xw==0) or (xh ==image_height-1 and xw!=image_width)or (xh!=image_height and xw==image_width-1):
            green[xh,xw]=np.divide((green[xh,xw-1]+green[xh-1,xw]+green[xh+1,xw]+green[xh,xw+1]), 3)
            
        if (xh in [0,image_height - 1]) and (xw in [0, image_width - 1]):
            green[xh,xw]=np.divide((green[xh,xw-1] + green[xh-1,xw] + green[xh+1,xw] + green[xh,xw+1]), 2)
                                    
    mos_img[:, :, 1] = green[0:image_height, 0:image_width]
    
    return mos_img



def demosaicAdagrad(img):
    '''adaptive gradient interpolation demosaicing.
    
    Args:
        img: np.array of size NxM.
    '''
    mos_img = np.tile(img[:, :, np.newaxis], [1, 1, 3])
    image_height, image_width = img.shape

    blue_values = np.zeros((image_height, image_width))
    blue_values[0:image_height:2, 0:image_width:2] = 1
    blue = mos_img[:, :, 2] * blue_values
    
    red_values = np.zeros((image_height, image_width))
    red_values[1:image_height:2, 1:image_width:2] = 1
    red = mos_img[:, :, 0] * red_values
    
    green = mos_img[:,:,1] - (blue + red)
   
    #Interpolation
    interpolateB  = np.argwhere(blue!=0)
    blue = np.hstack((blue, np.zeros((image_height, 2))))
    blue = np.vstack((blue, np.zeros((2, image_width + 2))))
    
    array1= np.zeros_like(blue)
    array2= np.zeros_like(blue)
    array3= np.zeros_like(blue)
    
    for x in interpolateB:
        xh, xw = x[0], x[1]
        array1[xh+1, xw+1] = np.divide((blue[xh,xw]+blue[xh,xw+2]+blue[xh+2,xw]+blue[xh+2,xw+2]), 4)
        array2[xh+1, xw] = np.divide((blue[xh,xw]+blue[xh+2,xw]), 2)
        array3[xh, xw+1] = np.divide((blue[xh,xw]+blue[xh,xw+2]), 2)
    
    blue = array1 + array2 + array3 + blue
    
    blue[:,image_width-1]= blue[:,image_width-2]
    blue[image_height-1,:]= blue[image_height-2,:]
    mos_img[:, :, 2] = blue[0:image_height, 0:image_width]
    
    interpolateR = np.argwhere(red!=0)
    red = np.hstack((red, np.zeros((image_height, 2))))
    red = np.vstack((red, np.zeros((2,image_width + 2))))
    
    array1= np.zeros_like(red)
    array2= np.zeros_like(red)
    array3= np.zeros_like(red)
    
    for x in interpolateR :
        xh, xw = x[0], x[1]
        array1[xh+1, xw+1] = np.divide((red[xh,xw]+red[xh,xw+2]+red[xh+2,xw]+red[xh+2,xw+2]), 4)
        array2[xh+1, xw] = np.divide((red[xh,xw]+red[xh+2,xw]), 2)
        array3[xh, xw+1] = np.divide((red[xh,xw]+red[xh,xw+2]), 2)
        
    red = array1 + array2 + array3 + red
    red[0,:] = red[1,:]
    red[:,0] = red[:,1]
    mos_img[:, :, 0] = red[0:image_height, 0:image_width]
    
    interpolateG = np.argwhere(green==0)
    green = np.hstack((green, np.zeros((image_height,1))))
    green = np.vstack((green, np.zeros((1,image_width+1))))
    
    for x in interpolateG:
        xh, xw = x[0], x[1]
        if((xh!=image_height and xw!=image_width)):
            if abs(green[xh,xw-1]- green[xh,xw+1]) < abs(green[xh-1,xw]-green[xh+1,xw]):    
                green[xh,xw]= np.divide((green[xh,xw-1] + green[xh,xw+1]),2)
            else:
                green[xh,xw]=np.divide((green[xh-1,xw] + green[xh+1,xw]), 2)
                                    
    mos_img[:, :, 1] = green[0:image_height, 0:image_width]
    
    return mos_img


def demosaicDivide(img):
    '''
    Tranformed Color Space Divide demosaicing
    
    Args:
        img: np.array of size NxM.
    ''' 
    #Dimensions of Filter values 
    image_height, image_width = img.shape
    
    #Green Channel Interpolation by calling Adaptive Gradient  
    mos_img = demosaicAdagrad(img)
    greenChannel = mos_img[:,:,1]
    
    #Replacing all zeros by 1e-6 to tackle divide by zero error
    greenChannel[np.where(greenChannel==0)] = 1e-6
    
    #Blue Filter Values
    blue_values = np.zeros([image_height, image_width])
    blue_values[0:image_height:2, 0:image_width:2] = img[0:image_height:2, 0:image_width:2] 
    
    #Divide Blue Filter values with Interpolated Green Channel
    blue_values = blue_values/greenChannel
    
    #Red Filter Values
    red_values = np.zeros([image_height, image_width])
    red_values[1:image_height:2, 1:image_width:2] = img[1:image_height:2, 1:image_width:2]  
    
    #Divide Red Filter values with Interpolated Green Channel
    red_values = red_values/greenChannel
    
    #Making new Green Filter
    green_values = np.zeros([image_height, image_width])
    green_values[0:image_height:2, 1:image_width:2] = img[0:image_height:2, 1:image_width:2]   
    green_values[1:image_height:2, 0:image_width:2] = img[1:image_height:2, 0:image_width:2]
    
    #RGB Channels
    rgb = red_values + green_values + blue_values
    
    #Finding Interpolated RGB Channels by calling Adaptive Gradient 
    mos_img = demosaicAdagrad(rgb)
    
    #Multiply Blue and Red Filter values with Interpolated Green Channel
    mos_img[:, :, 2] = np.multiply(mos_img[:, :, 2] , greenChannel)
    mos_img[:, :, 0] = np.multiply(mos_img[:, :, 0] , greenChannel)

    #Handling all outlier values greater than 1 in mos_img  
    mos_img[np.where(mos_img>1)] = 1   
    return mos_img

def demosaicLoglDivide(img):
    '''
    Tranformed Color Space Divide and Log demosaicing
    
    Args:
        img: np.array of size NxM.
    '''
    mos_img = np.tile(img[:, :, np.newaxis], [1, 1, 3])
    
    #Dimensions of the filter values    
    image_height, image_width = img.shape
    
    #Green Channel Interpolation by calling Adaptive Gradient  
    mos_img = demosaicAdagrad(img)
    greenChannel = mos_img[:,:,1]
    
    #Replacing all zeros by 1e-6 to tackle divide by zero error
    greenChannel[np.where(greenChannel == 0)] = 1e-6

    #Blue Filter Values
    blue_values = np.zeros([image_height, image_width])
    blue_values[0:image_height:2, 0:image_width:2] = img[0:image_height:2, 0:image_width:2] 
    
    #Divide Blue Filter values with Interpolated Green Channel
    #Add 1 inside log to tackle log(0) edge case                  
    blue_values = np.log(1+ blue_values/greenChannel)
    
    #Get Red Filter Values
    red_values = np.zeros([image_height, image_width])
    red_values[1:image_height:2, 1:image_width:2] = img[1:image_height:2, 1:image_width:2]  
    
    #Divide Red Filter values with Interpolated Green Chzannel
    #Add 1 inside log to tackle log(0) edge case 
    red_values = np.log(1+ red_values/greenChannel)
    
    #Making new Green Filter
    green_values = np.zeros([image_height, image_width])
    green_values[0:image_height:2, 1:image_width:2] = img[0:image_height:2, 1:image_width:2]   
    green_values[1:image_height:2, 0:image_width:2] = img[1:image_height:2, 0:image_width:2]
       
    #RGB Channels       
    rgb = red_values + green_values + blue_values
    
    #Finding Interpolated RGB Channels by calling Adaptive Gradient 
    mos_img = demosaicAdagrad(rgb)
    
    #Multiply Blue and Red Filter values with Interpolated Green Channel
    #Take exponential which is the inverse of log 
    #Subtract 1 because 1 was added before taking log of the ratio
    mos_img[:, :, 2] = np.multiply(np.exp(mos_img[:, :, 2]-1) , greenChannel)
    mos_img[:, :, 0] = np.multiply(np.exp(mos_img[:, :, 0]-1) , greenChannel)
    
    #Handling all outlier values greater than 1 in mos_img  
    mos_img[np.where(mos_img > 1)] = 1    
    return mos_img