# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

#For 3D plotting
from mpl_toolkits import mplot3d

def Sphere3D(x, y, z, radius = 1):
  #Sampling 500 points- a large number of points on the surface of the sphere
  _pi = np.pi
  _phi = np.linspace(0, 2 * _pi, 50)
  _theta = np.linspace(0, _pi, 50) 
  _countPhi = np.ones(np.size(_phi))

  a = x + radius * np.cos(_phi)[:,None] * np.sin(_theta)[None,:]
  b = y + radius * np.sin(_phi)[:,None] * np.sin(_theta)[None,:]
  c = z + radius * _countPhi[:,None] * np.cos(_theta)[None,:] 

  return a, b, c

def PinholeCamera2D(x, y, z, focal_length = 1):
  #Using Perespective Equations
  pinholeX = (focal_length * x)/z
  pinholeY = (focal_length * y)/z

  return pinholeX, pinholeY

locations = [[0, 0, 5], [2, 0, 5], [2, 2, 5],[0, 2, 5]]

for coords in locations:
  X, Y, Z = Sphere3D(coords[0], coords[1], coords[2])
  pinholeX, pinholeY = PinholeCamera2D(X, Y, Z)

  a , b = pinholeX.shape
  numSamplePoints = a * b
  cameraX = list(pinholeX.reshape(numSamplePoints,))
  cameraY = list(pinholeY.reshape(numSamplePoints,))
  
  fig = plt.figure(figsize =(14, 9))
  ax1 = fig.add_subplot(121, projection='3d')
  ax1.plot_surface(X, Y, Z, color='purple')
  ax1.set_title('3D Sphere at {}'.format(coords), fontsize = 20)

  ax2 = fig.add_subplot(122)
  ax2.scatter(cameraX, cameraY,  color='purple', marker='x')
  ax2.set_title('2D Camera Perspective Projection', fontsize = 20)
  ax2.set_aspect('equal')
  plt.show()