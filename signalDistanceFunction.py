# -*- coding: utf-8 -*-
"""SDF.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1LtIiSdYAC4Fka-jFo93hIbpyuM1llv-L
"""

#!pip install scikit-fmm
#!pip install airfoils
#!pip install --upgrade airfoils
#!pip install meshio[all]


import numpy as np
import skfmm
import matplotlib.pyplot as plt
import meshio

def getSDF(datFile, AoA, plot=False):
   # plt.close('all')
    
    np.set_printoptions(threshold=np.inf)
    
    AoA = 0
    
    
    
    
    # Define the angle of attack in degrees
    angle_of_attack = AoA
    
    # Convert the angle to radians
    angle_rad = np.radians(-angle_of_attack)
    
    # Create the rotation matrix
    rotation_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                                [np.sin(angle_rad), np.cos(angle_rad)]])
    
    # Define the grid for the signed distance function calculation
    nx, ny = 300,300
    x = np.linspace(-0.5, 1.5, nx)
    y = np.linspace(-0.5, 0.5, ny)
    X, Y = np.meshgrid(x, y)
    def create_airfoil_mask(x_coords, y_coords, X, Y):
        from matplotlib.path import Path
    
        airfoil_path = Path(list(zip(x_coords, y_coords)))
        mask = airfoil_path.contains_points(np.vstack([X.ravel(), Y.ravel()]).T).reshape(X.shape)
    
        return mask
    
    # Replace these with your airfoil coordinates
    x_coords = np.array(datFile[:,0])
    y_coords = np.array(datFile[:,1])
    
    #Subtract the leading edge coordinates to center the airfoil at the origin
    x_coords_centered = x_coords - x_coords[0]
    y_coords_centered = y_coords - y_coords[0]
    
    # Apply the rotation matrix to the centered coordinates
    rotated_coords = rotation_matrix @ np.array([x_coords_centered, y_coords_centered])
    
    # Add the leading edge coordinates back to shift the airfoil back to its original position
    x_coords_rotated = rotated_coords[0] + x_coords[0]
    y_coords_rotated = rotated_coords[1] + y_coords[0]
    
    # Now you can use x_coords_rotated and y_coords_rotated in your code
    airfoil_mask = create_airfoil_mask(x_coords_rotated, y_coords_rotated, X, Y)
    
    #airfoil_mask = create_airfoil_mask(x_coords, y_coords, X, Y)
    # Create the initial distance field
    phi = np.where(airfoil_mask, -300, 300)
    
    # Compute the signed distance function using the Fast Marching Method
    distance = skfmm.distance(phi,dx=1/150)
    
    
    
    #mesh = meshio.read('meshFileFromFluent.msh')
    
    #points, _, _ = mesh.points, mesh.cells, mesh.cells_dict
    
    if plot!=False:
            
        # Plot the signed distance function
        plt.figure()
        #plt.contourf(X, Y, distance, levels=5,cmap=plt.cm.jet)
        #plt.colorbar(label='Distância')
        plt.plot(x_coords_rotated, y_coords_rotated, 'r-', linewidth=1.5)
        plt.plot(X,Y,'k.',markersize=2)
        #plt.axis('equal')
        #plt.xlim([-0.5,1.5])
        #plt.ylim([-0.5,0.5])
        plt.tight_layout()
        plt.show()
        
        

        """plt.figure()
        plt.plot(points[:,0],points[:,1],'ko',markersize=1)
        plt.axis('equal')
        plt.xlim([-0.5,1.5])
        plt.ylim([-0.5,0.5])
        plt.show()"""
    return distance, X, Y