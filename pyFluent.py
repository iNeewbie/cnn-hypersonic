import ansys.fluent.core as pyfluent
import os
import numpy as np
from tqdm import tqdm

# Iniciando o solver
print("Iniciando solver...")
solver = pyfluent.launch_fluent(
    version="23.1",
    precision="double",
    processor_count=4,
    mode="solver",
    ui_mode="gui",
    dimension=2
)
solver.transcript.stop()
print("Solver iniciado.")

# Carregando o caso inicial
case_file = "C:\\Users\\guilh\\OneDrive\\Documentos\\cnn-hypersonic\\pyFluent\\caso.cas.h5"
print("Carregando caso:", case_file)
solver.file.read_case(file_type="case", file_name=case_file)
print("Caso carregado.")

# Parâmetros para iteração
MachNumbers = [5, 6, 7, 8, 9, 10]
WedgeAngles = [5, 7, 10, 12, 15]
AoAs = [-5, -3, 0, 5, 10, 15]

base_path = "C:\\Users\\guilh\\OneDrive\\Documentos\\cnn-hypersonic\\DataCFD2"
pasta = [os.path.join(base_path, str(WedgeAngle) + "-WedgeAngle", "meshFile.msh") for WedgeAngle in WedgeAngles]
pasta2 = [os.path.join(base_path, str(WedgeAngle) + "-WedgeAngle", str(AoA) + "-AoA", str(MachNumber) + "-Mach","solData") 
         for WedgeAngle in WedgeAngles for AoA in AoAs for MachNumber in MachNumbers]

# Iterando sobre parâmetros
index_mach = 0
for wa_it in tqdm(range(len(WedgeAngles)), desc="Wedge Angles"):
    print(f"Carregando malha: {pasta[wa_it]}")
    solver.tui.file.replace_mesh(f'"{pasta[wa_it]}"')
    
    for aoa_it in tqdm(range(len(AoAs)), desc="Angles of Attack", leave=False):
        for mn_it in tqdm(range(len(MachNumbers)), desc="Mach Numbers", leave=False):
                # Configurar Mach e ângulo de ataque
                mach_x = np.cos(np.deg2rad(AoAs[aoa_it]))
                mach_y = np.sin(np.deg2rad(AoAs[aoa_it]))
                
              
                print("Setando nova boundary condition...")
                bc = solver.setup.boundary_conditions.pressure_far_field["far-field"]
                bc.m.value = MachNumbers[mn_it]
                bc.ni.value = mach_x
                bc.nj.value = mach_y
                print("Setado com sucesso!")
                
                # Inicialização híbrida se necessário
                if index_mach % (len(MachNumbers) * len(AoAs)) == 0:
                    print("Inicializando solução...")
                    solver.solution.initialization.hybrid_initialize()

                # Executar cálculo
                print(f"Executando cálculo: Mach={MachNumbers[mn_it]}, AoA={AoAs[aoa_it]}, WedgeAngle={WedgeAngles[wa_it]}")
                solver.solution.run_calculation.iterate(iter_count=10)

                # Exportar dados

                solver.tui.file.export.ascii(f'"{pasta2[index_mach]}"', "flow", "()", "yes", "pressure", "mach-number", "temperature","quit","no","yes")
                solver.tui.file.write_case_data(f'"{pasta2[index_mach]}"','yes')

                index_mach += 1
    break

# Finalizando solver
print("Finalizando solver...")
solver.exit()
print("Solver finalizado.")
