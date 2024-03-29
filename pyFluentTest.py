import ansys.fluent.core as pyfluent
import os
import numpy as np
from tqdm import tqdm

print("Iniciando solver")
solver = pyfluent.launch_fluent(
    precision="double",
    processor_count=8,
    mode="solver",
    show_gui=False,
    version="2d"
    )
print("Solver iniciado")
solver.transcript.stop()
print("Carregando case")
solver.file.read_case(file_type = "case", file_name = "H:/Meu Drive/TCC/Programming/cnn-hypersonic/pyFluent/caso.cas.h5")
#solver.setup.models.print_state()

#Iterações:
MachNumbers = [5,6,7,8,9,10]
WedgeAngles = [5, 7, 10, 12, 15]
AoAs = [-5, -3, 0, 5, 10, 15]

base_path = "H:\\Meu Drive\\TCC\\Programming\\cnn-hypersonic\\DataCFD"
pasta = [os.path.join(base_path, str(WedgeAngle) + "-WedgeAngle", "meshFile.msh") for WedgeAngle in WedgeAngles]
pasta2 = [os.path.join(base_path, str(WedgeAngle) + "-WedgeAngle", str(AoA) + "-AoA", str(MachNumber) + "-Mach","solData") 
         for WedgeAngle in WedgeAngles for AoA in AoAs for MachNumber in MachNumbers]
index = 0
index_mach = 0
for wa_it in tqdm(range(len(WedgeAngles))):
    solver.tui.file.replace_mesh(f'"{pasta[wa_it]}"', "yes")
    for aoa_it in tqdm(range(0, len(AoAs))):
        for mn_it in tqdm(range(len(MachNumbers))):            
            # Set Mach number based on angle of attack
            mach_x = np.cos(np.deg2rad(AoAs[aoa_it]))
            mach_y = np.sin(np.deg2rad(AoAs[aoa_it]))
            solver.setup.boundary_conditions.pressure_far_field['pressure-far-field'].m.value = MachNumbers[mn_it]
            solver.setup.boundary_conditions.pressure_far_field['pressure-far-field'].ni.value = mach_x 
            solver.setup.boundary_conditions.pressure_far_field['pressure-far-field'].nj.value = mach_y
            if index_mach%(len(MachNumbers)*len(AoAs)) == 0:
                solver.solution.initialization.hybrid_initialize()
            solver.solution.run_calculation.iterate(iter_count = 5000)
            solver.tui.file.export.ascii(f'"{pasta2[index_mach]}"', "air", "()", "yes", "pressure", "mach-number", "temperature","quit","no","yes")
            solver.tui.file.write_case_data(f'"{pasta2[index_mach]}"','yes')
            #print(f'Iteração completa Mach: {MachNumbers[mn_it]}, Wedge Angle: {WedgeAngles[wa_it]}, AoA: {AoAs[aoa_it]}')
            index_mach+=1
solver.exit()
            


        
#Trocar malha
#solver.tui.file.replace_mesh('"H:\\Meu Drive\\TCC\\Programming\\cnn-hypersonic\\DataCFD\\-5-AoA\\15-WedgeAngle\\meshFile.msh"', "ok")
#solver.tui.file.replace_mesh(f'"{pasta[5]}"', "ok")

#Trocar Mach
#solver.setup.boundary_conditions.pressure_far_field['pressure-far-field'].m.value= 10

#Exporta dados
#solver.tui.file.export.ascii("certo", "air", "()", "yes", "pressure", "mach-number", "temperature")