# Execute o script Python
python pyFluent.py

# Verifique se o script Python terminou com sucesso
if ($LASTEXITCODE -ne 0) {
    Write-Output "Erro ao executar pyFluent.py. Abandonando o processo."
    exit 1
}

# Adicione todas as modifica��es ao commit
git add .

# Fa�a o commit com a mensagem espec�fica
git commit -m "todas simula��es feitas trab"

# Envie as altera��es para o reposit�rio remoto
git push origin maquinaTrab


