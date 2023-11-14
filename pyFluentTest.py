import ansys.fluent.core as pyfluent
import os

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
index_mach = 0
for aoa_it in range(len(AoAs)):
    if index == 2:
        break
    for wa_it in range(len(WedgeAngles)):
        solver.tui.file.replace_mesh(f'"{pasta[index]}"', "ok")
        for mn_it in range(len(MachNumbers)):            
            solver.setup.boundary_conditions.pressure_far_field['pressure-far-field'].m.value= MachNumbers[mn_it]
            print(solver.setup.boundary_conditions.pressure_far_field['pressure-far-field'].m.value())
            solver.tui.solve.initialize.hyb_initialization()
            solver.tui.solve.iterate("200")
            solver.tui.file.export.ascii(f'"{pasta2[index_mach]}"', "air", "()", "yes", "pressure", "mach-number", "temperature")
            print(f'Iteração completada Mach: {MachNumbers[mn_it]}, Wedge Angle: {WedgeAngles[wa_it]}, AoA: {AoAs[aoa_it]}')
            index_mach+=1
        index += 1
        
#Trocar malha
#solver.tui.file.replace_mesh('"H:\\Meu Drive\\TCC\\Programming\\cnn-hypersonic\\DataCFD\\-5-AoA\\15-WedgeAngle\\meshFile.msh"', "ok")
#solver.tui.file.replace_mesh(f'"{pasta[5]}"', "ok")

#Trocar Mach
#solver.setup.boundary_conditions.pressure_far_field['pressure-far-field'].m.value= 10

#Exporta dados
#solver.tui.file.export.ascii("certo", "air", "()", "yes", "pressure", "mach-number", "temperature")