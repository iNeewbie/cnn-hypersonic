import subprocess
import time
from apscheduler.schedulers.background import BackgroundScheduler
import gpustat
import os
import logging

# Configuração do logging para registrar mensagens no arquivo output.log
logging.basicConfig(filename='output.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# Função para verificar o uso da GPU com gpustat
def verificar_gpu():
    try:
        # Obtém o estado atual das GPUs com gpustat
        gpus = gpustat.GPUStatCollection.new_query()

        # Itera sobre todas as GPUs e verifica o uso da primeira
        for gpu in gpus.gpus:
            gpu_usage = gpu.utilization  # Percentual de uso da GPU
            logging.info(f"Uso da GPU {gpu.index}: {gpu_usage}%")
            
            # Se o uso da GPU for abaixo de 20%, executa o comando desejado
            if gpu_usage < 20:
                logging.info("GPU com uso abaixo de 20%, executando o script Python...")

                # Executa o script bayesianOptimization.py e espera ele terminar
                result = subprocess.run(['python', 'bayesianOptimization.py'])

                # Se o script foi executado com sucesso, encerra o script principal
                if result.returncode == 0:
                    logging.info("Execução de bayesianOptimization.py concluída, encerrando o queueScript.py.")
                    # Encerra o script imediatamente
                    os._exit(0)  # Encerra o script imediatamente
            else:
                logging.info("GPU com uso acima de 20%, aguardando...")

    except Exception as e:
        logging.error(f"Erro ao verificar o uso da GPU: {e}")

# Cria o agendador
scheduler = BackgroundScheduler()

# Agendar a execução da função de verificação da GPU a cada 60 segundos
scheduler.add_job(verificar_gpu, 'interval', seconds=60*20)

# Iniciar o agendador
scheduler.start()

# Manter o script rodando para que o agendador funcione
try:
    while True:
        time.sleep(1)  # Aguarda 1 segundo e continua verificando
except (KeyboardInterrupt, SystemExit):
    scheduler.shutdown()
