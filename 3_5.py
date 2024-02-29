import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def sigmoid_derivative_numerical(x, a = 1):
    f = 1 / (1 + np.exp(-a * x))
    return f * (1 - f)

def sigmoid_derivative_theoretical(x, a = 1):
    return a * np.exp(-a * x) / (1 + np.exp(-a * x))**2

# Генерируем данные для графика
x = np.linspace(-10, 10, 100)
y = sigmoid_derivative_numerical(x)
dy_dx = sigmoid_derivative_theoretical(x)

# Отображение графика
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(x, y)
plt.grid(True)
plt.title('Численная производная')

plt.subplot(2, 1, 2)
plt.plot(x, dy_dx, 'r--')
plt.grid(True)
plt.title('Теоритическая производная')
plt.show()