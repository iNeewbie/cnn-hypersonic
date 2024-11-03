import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata

def interpSDFCart(sdf, grid_x, grid_y, results, plot=False):
    x, y, pressure, mach_number, temperature = results
    mask = np.zeros_like(sdf)  # create an array of zeros with the same shape as sdf
    mask[sdf > 0] = 1  # set the values to 1 where sdf is positive
    
    # Interpolate pressure
    grid_pressure = griddata((x, y), pressure, (grid_x, grid_y), method='linear')
    
    # Interpolate mach_number
    grid_mach_number = griddata((x, y), mach_number, (grid_x, grid_y), method='linear')
    
    # Interpolate temperature
    grid_temperature = griddata((x, y), temperature, (grid_x, grid_y), method='linear')
    
    # Apply mask
    grid_pressure = grid_pressure * mask
    grid_mach_number = grid_mach_number * mask
    grid_temperature = grid_temperature * mask
    
    if plot:
        # Plot grid_temperature if requested
        plt.figure()
        c = plt.contourf(grid_temperature, cmap=plt.cm.jet, levels=200)
        plt.colorbar(c)
        plt.title('Temperature Distribution')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()
        
        plt.figure()
        c = plt.contourf(grid_pressure, cmap=plt.cm.jet, levels=200)
        plt.colorbar(c)
        plt.title('Pressure Distribution')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()
        
        plt.figure()
        c = plt.contourf(grid_mach_number, cmap=plt.cm.jet, levels=200)
        plt.colorbar(c)
        plt.title('Mach Distribution')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()
    return [grid_temperature, grid_pressure, grid_mach_number]
