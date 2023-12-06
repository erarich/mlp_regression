from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# # Valores reais
# y_true = np.array([380512.68595683837, 221618.58321806978,
#                   100080.86589451409, 331497.0913073626, 163683.6754340198])

# # Valores previstos pelo modelo
# y_pred = np.array([400000.5, 600000.5, 120000, 700000, 98000.2])
# Valores reais

y_true = np.array([2, 4, 5, 4, 5])

# Valores previstos pelo modelo
y_pred = np.array([1.5, 3.5, 4.5, 4.0, 5.2])

# Calcula o erro médio quadrático
mse = mean_squared_error(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)

print(f'O erro médio quadrático é: {mse}')
print(f'O Erro Absoluto Médio (MAE) é: {mae}')
