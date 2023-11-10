import ansys.fluent.core as pyfluent
from pprint import pprint
import os

print("Iniciando solver")
solver = pyfluent.launch_fluent(
    precision="double",
    processor_count=8,
    mode="solver",
    show_gui=True,
    version="2d"
    )
print("Solver iniciado")


print("Carregando case")
solver.file.read_case(file_type = "case", file_name = "caso.cas.h5")
solver.setup.models.print_state()


#Iterações:
MachNumbers = [5,6,7,8,9,10]
WedgeAngles = [5, 7, 10, 12, 15]
AoAs = [-5, -3, 0, 5, 10, 15]


base_path = "H:\\Meu Drive\\TCC\\Programming\\cnn-hypersonic\\DataCFD"

pasta = [os.path.join(base_path, str(AoA) + "-AoA", str(WedgeAngle) + "-WedgeAngle", "meshFile.msh") 
         for AoA in AoAs for WedgeAngle in WedgeAngles]

pasta2 = [os.path.join(base_path, str(AoA) + "-AoA", str(WedgeAngle) + "-WedgeAngle", str(MachNumber) + "-Mach","solData") 
         for AoA in AoAs for WedgeAngle in WedgeAngles for MachNumber in MachNumbers]


index = 0
for aoa_it in AoAs:
    if index == 2:
        break
    for wa_it in WedgeAngles:
        solver.tui.file.replace_mesh(f'"{pasta[index]}"', "ok")
        for mn_it in MachNumbers:
            solver.setup.boundary_conditions.pressure_far_field['pressure-far-field'].m.value= MachNumbers[mn_it]
            solver.tui.solve.initialize.hyb_initialization()
            solver.tui.solve.iterate("5000")
            solver.tui.file.export.ascii(f'"{pasta2[index]}"', "air", "()", "yes", "pressure", "mach-number", "temperature")
        index += 1
        
#Trocar malha
#solver.tui.file.replace_mesh('"H:\\Meu Drive\\TCC\\Programming\\cnn-hypersonic\\DataCFD\\-5-AoA\\15-WedgeAngle\\meshFile.msh"', "ok")
#solver.tui.file.replace_mesh(f'"{pasta[5]}"', "ok")

#Trocar Mach
#solver.setup.boundary_conditions.pressure_far_field['pressure-far-field'].m.value= 10

#Exporta dados
#solver.tui.file.export.ascii("certo", "air", "()", "yes", "pressure", "mach-number", "temperature")