import subprocess
import time
from apscheduler.schedulers.background import BackgroundScheduler
import gpustat

# Função para verificar o uso da GPU com gpustat
def verificar_gpu():
    try:
        # Obtém o estado atual das GPUs com gpustat
        gpus = gpustat.GPUStatCollection.new_query()

        # Itera sobre todas as GPUs e verifica o uso da primeira
        for gpu in gpus.gpus:
            gpu_usage = gpu.utilization  # Percentual de uso da GPU
            print(f"Uso da GPU {gpu.index}: {gpu_usage}%")
            
            # Se o uso da GPU for abaixo de 20%, executa o comando desejado
            if gpu_usage < 20:
                print("GPU com uso abaixo de 20%, executando o script Python...")
                subprocess.run(['python', 'bayesianOptimization.py'])
            else:
                print("GPU com uso acima de 20%, aguardando...")

    except Exception as e:
        print(f"Erro ao verificar o uso da GPU: {e}")

# Cria o agendador
scheduler = BackgroundScheduler()

# Agendar a execução da função de verificação da GPU a cada 10 segundos
scheduler.add_job(verificar_gpu, 'interval', seconds=60)

# Iniciar o agendador
scheduler.start()

# Manter o script rodando para que o agendador funcione
try:
    while True:
        time.sleep(1)  # Aguarda 1 segundo e continua verificando
except (KeyboardInterrupt, SystemExit):
    scheduler.shutdown()
