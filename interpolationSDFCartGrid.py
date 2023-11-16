#Interpola a SDF com a Cartesian Grid com os dados do Ansys
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata

def interpSDFCart(sdf, grid_x, grid_y,results, plot=False):
    x,y,pressure,mach_number,temperature = results
    mask = np.zeros_like(sdf) # create an array of zeros with the same shape as sdf
    mask[sdf > 0] = 1 # set the values to 1 where sdf is positive
    
    
    
    
    # Interpolate pressure
    grid_pressure = griddata((x, y), pressure, (grid_x, grid_y), method='cubic')
    
    # Interpolate mach_number
    grid_mach_number = griddata((x, y), mach_number, (grid_x, grid_y), method='cubic')
    
    # Interpolate temperature
    grid_temperature = griddata((x, y), temperature, (grid_x, grid_y), method='cubic')
    
    # Interpolate e
    #grid_e = griddata((x, y), e, (grid_x, grid_y), method='cubic')
    
    grid_pressure = grid_pressure*mask
    grid_mach_number = grid_mach_number*mask
    grid_temperature = grid_temperature*mask
    #grid_e = grid_e*mask
    
    
    if plot != False:
    
                # Plot the grid_mach_number
        plt.figure()
        c = plt.contourf(grid_x, grid_y, grid_mach_number, cmap=plt.cm.jet, levels=200)
        plt.colorbar(c)
        plt.title('Mach Number Distribution')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()
        
        # Plot the grid_pressure
        plt.figure()
        c = plt.contourf(grid_x, grid_y, grid_pressure, cmap=plt.cm.jet, levels=200)
        plt.colorbar(c)
        plt.title('Pressure Distribution')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()
        
        # Plot the grid_temperature
        plt.figure()
        c = plt.contourf(grid_temperature, cmap=plt.cm.jet, levels=200)
        
        # Add a colorbar to the plot
        plt.colorbar(c)
        
        # Set the title and labels
        plt.title('Temperature Distribution')
        plt.xlabel('X')
        plt.ylabel('Y')
        
        # Display the plot
        plt.show()
        
        # Plot the grid_e
        """plt.figure()
        c = plt.contourf(grid_x, grid_y, grid_e, cmap=plt.cm.jet, levels=200)
        plt.colorbar(c)
        plt.title('E Distribution')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()"""

    return [grid_temperature, grid_pressure, grid_mach_number]
        
