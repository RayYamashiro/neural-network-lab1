import numpy as np

def threshold_activation(tn, threshold=0):
    yn = np.where(tn <= threshold, 0, 1)
    return yn

def piecewise_activation(tn):
    yn = np.piecewise(v, [v < 0, (v >= 0) & (v <= 1), v > 1], [0, lambda v: v, 1])
    return yn

def sigmoid_activation(tn, a=1):
    yn = 1 / (1 + np.exp(-a * tn))
    return yn

# Создание вектора времени
# N = 50
N = int(input('Задайте N '))
v = np.arange(-3, N, 0.1)

# Вычисление значений функций активации
threshold_data = np.column_stack((v, threshold_activation(v)))
piecewise_data = np.column_stack((v, piecewise_activation(v)))
sigmoid_data = np.column_stack((v, sigmoid_activation(v)))

# Вывод результатов
print("Результаты пороговой функции активации:")
print(threshold_data)

print("\nРезультаты кусочно-линейной функции активации:")
print(piecewise_data)

print("\nРезультаты сигмоидной функции активации:")
print(sigmoid_data)