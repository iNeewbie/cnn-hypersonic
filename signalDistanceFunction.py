import numpy as np
import skfmm
import matplotlib.pyplot as plt
import meshio

def getSDF(datFile, AoA, plot=False):
    np.set_printoptions(threshold=np.inf)
    
    # Ângulo de ataque
    angle_of_attack = AoA
    angle_rad = np.radians(-angle_of_attack)
    rotation_matrix = np.array([[np.cos(angle_rad), -np.sin(angle_rad)],
                                [np.sin(angle_rad), np.cos(angle_rad)]])
    
    # Configuração da grade para o domínio de 2m x 1m com resolução de 5mm (0.005)
    nx, ny = 400,400
    x = np.linspace(-0.5, 1.5, nx)  # Domínio de -0.5 a 1.5m em x (2m no total)
    y = np.linspace(-0.5, 0.5, ny)  # Domínio de -0.5 a 0.5m em y (1m no total)
    X, Y = np.meshgrid(x, y)
    
    # Função para criar a máscara do aerofólio
    def create_airfoil_mask(x_coords, y_coords, X, Y):
        from matplotlib.path import Path
        airfoil_path = Path(list(zip(x_coords, y_coords)))
        mask = airfoil_path.contains_points(np.vstack([X.ravel(), Y.ravel()]).T).reshape(X.shape)
        return mask
    
    # Coordenadas do aerofólio (datFile)
    x_coords = np.array(datFile[:,0])
    y_coords = np.array(datFile[:,1])
    
    # Centralizar e rotacionar as coordenadas do aerofólio
    x_coords_centered = x_coords - x_coords[0]
    y_coords_centered = y_coords - y_coords[0]
    rotated_coords = rotation_matrix @ np.array([x_coords_centered, y_coords_centered])
    x_coords_rotated = rotated_coords[0] + x_coords[0]
    y_coords_rotated = rotated_coords[1] + y_coords[0]
    
    # Criar a máscara do aerofólio
    airfoil_mask = create_airfoil_mask(x_coords_rotated, y_coords_rotated, X, Y)
    
    # Campo de distância inicial com valores de -300 (dentro do aerofólio) e 300 (fora)
    phi = np.where(airfoil_mask, -300, 300)
    
    # Computa o campo de distância com dx de 5mm
    distance = skfmm.distance(phi, dx=2/nx)

    
    # Visualização opcional
    if plot:
        plt.figure()
        plt.plot(x_coords_rotated, y_coords_rotated, 'r-', linewidth=1.5)
        plt.plot(X, Y, 'k.', markersize=2)
        plt.tight_layout()
        plt.show()
    
    return distance, X, Y
