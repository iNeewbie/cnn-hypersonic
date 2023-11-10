import ansys.fluent.core as pyfluent
from pprint import pprint

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






#Trocar malha
solver.tui.file.replace_mesh('"H:\\Meu Drive\\TCC\\Programming\\cnn-hypersonic\\DataCFD\\-5-AoA\\15-WedgeAngle\\meshFile.msh"', "ok")

#Trocar Mach
solver.setup.boundary_conditions.pressure_far_field['pressure-far-field'].m.value= 10
