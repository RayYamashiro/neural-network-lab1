import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def plot_activation_function(func_type, v, a=None):
    if func_type == 1:  # Единичный скачок или пороговая функция
        threshold = 0  # Пороговое значение
        ymin = 0
        ymax = 1
        activation = np.where(v <= threshold, ymin, ymax)

    elif func_type == 2:  # Кусочно-линейная функция
        activation = np.piecewise(v, [v < 0, (v >= 0) & (v <= 1), v > 1], [0, lambda v: v, 1])

    elif func_type == 3:  # Сигмоидная функция
        activation = 1 / (1 + np.exp(-a * v))

    elif func_type == 4:  # Гиперболический тангенс
        activation = np.tanh(v / a)

    # Отображение функции активации
    plt.plot(v, activation)
    plt.xlabel('Входные данные')
    plt.ylabel('Активация')
    plt.title('Функция активации')

    plt.grid(True)
    plt.show()


# Пример использования:
v = np.linspace(-10, 10, 100)  # Значения входных данных для проверки функций
plot_activation_function(1, v)  # Пороговая функция
plot_activation_function(2, v)  # Кусочно-линейная функция
plot_activation_function(3, v, a=1)  # Сигмоидная функция
plot_activation_function(4, v, a=1)  # Гиперболический тангенс