import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path

def importResults(data,plot=False):
    
    # Extract columns
    x = data[:, 1]
    y = data[:, 2]
    pressure = data[:, 3]
    mach_number = data[:, 4]
    temperature = data[:, 5]
    #e = data[:, 6]
    
    
    if plot!=False:
        # Create a contour plot for pressure
        plt.figure()
        plt.tricontourf(x, y, pressure, cmap=plt.cm.jet,levels=200)
        plt.colorbar()
        plt.title('Pressure Contour Plot')
        plt.xlabel('X-coordinate')
        plt.ylabel('Y-coordinate')
        plt.axis('equal')
        plt.xlim([-0.5,1.5])
        plt.ylim([-0.5,0.5])
        plt.show()
        
        # Create a contour plot for mach-number
        plt.figure()
        plt.tricontourf(x, y, mach_number, cmap=plt.cm.jet,levels=200)
        plt.colorbar()
        plt.title('Mach-Number Contour Plot')
        plt.xlabel('X-coordinate')
        plt.ylabel('Y-coordinate')
        plt.axis('equal')
        plt.xlim([-0.5,1.5])
        plt.ylim([-0.5,0.5])
        plt.show()
        
        # Create a contour plot for temperature
        plt.figure()
        plt.tricontourf(x, y, temperature, cmap=plt.cm.jet,levels=200)
        plt.colorbar()
        plt.title('Temperature Contour Plot')
        plt.xlabel('X-coordinate')
        plt.ylabel('Y-coordinate')
        plt.axis('equal')
        plt.xlim([-0.5,1.5])
        plt.ylim([-0.5,0.5])
        plt.show()
    
        """# Create a contour plot for e
        plt.figure()
        plt.tricontourf(x, y, e, cmap=plt.cm.jet,levels=200)
        plt.colorbar()
        plt.title('E Contour Plot')
        plt.xlabel('X-coordinate')
        plt.ylabel('Y-coordinate')
        plt.axis('equal')
        plt.xlim([-0.5,1.5])
        plt.ylim([-0.5,0.5])
        plt.show()"""

    return [x,y,pressure,mach_number,temperature]#,e
