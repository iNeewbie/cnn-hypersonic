# Execute o script Python
python pyFluent.py

# Verifique se o script Python terminou com sucesso
if ($LASTEXITCODE -ne 0) {
    Write-Output "Erro ao executar pyFluent.py. Abandonando o processo."
    exit 1
}

# Adicione todas as modificações ao commit
git add .

# Faça o commit com a mensagem específica
git commit -m "todas simulações feitas trab"

# Envie as alterações para o repositório remoto
git push origin maquinaTrab


