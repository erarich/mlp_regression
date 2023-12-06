import matplotlib.pyplot as plt

# Dados para as linhas
x = [1, 2, 3, 4, 5]
y1 = [2, 4, 6, 8, 10]
y2 = [1, 2, 1, 2, 1]

# Criar o gráfico
plt.plot(x, y1, label='Linha 1', marker='o')  # Linha 1 com marcadores 'o'
# Linha 2 com marcadores 's' e estilo de linha '--'
plt.plot(x, y2, label='Linha 2', linestyle='--', marker='s')

# Adicionar rótulos e título
plt.xlabel('Eixo X')
plt.ylabel('Eixo Y')
plt.title('Gráfico com Duas Linhas')
plt.legend()  # Adicionar a legenda com base nos rótulos fornecidos em plt.plot

# Mostrar o gráfico
plt.show()
