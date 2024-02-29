import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import time


# Параметры сигнала
A = 0.005 # интервал наблюдения
f0 = 2000 # частота

# Параметры для дискретизации
fdn = 2 * f0  # Частота дискретизации (с Найквистом)
mvis = 5
fdv = mvis * fdn  # Частота дискретизации для визуализации
dt = 1 / fdv  # Интервал дискретизации по времени
T = 1 / f0  # Период сигнала
NT = f0 * A
points = 300

# Дискретизация сигнала
t_interval = np.arange(0, NT * T, dt)
t_points = np.arange(0, (points - 1) * dt, dt)

cos_signal_interval = np.cos(2 * np.pi * f0 * t_interval)
sin_signal_interval = np.sin(2 * np.pi * f0 * t_interval)

cos_signal_points = np.cos(2 * np.pi * f0 * t_points)
sin_signal_points =np.sin(2 * np.pi * f0 * t_points)

# Построение графиков
plt.figure(figsize=(15, 10))
plt.subplot(2, 2, 1)
plt.plot(t_interval, cos_signal_interval)
plt.xlabel('Время (с)')
plt.ylabel('Амплитуда')
plt.title('Дискретизация и визуализация COS сигнала (интервал)')
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(t_points, cos_signal_points, 'r--')
plt.xlabel('Время (с)')
plt.ylabel('Амплитуда')
plt.title('Дискретизация и визуализация COS сигнала (точки)')
plt.grid(True)

plt.subplot(2, 2, 3)
plt.plot(t_interval, sin_signal_interval)
plt.xlabel('Время (с)')
plt.ylabel('Амплитуда')
plt.title('Дискретизация и визуализация SIN сигнала (интервал)')
plt.grid(True)

plt.subplot(2, 2, 4)
plt.plot(t_points, sin_signal_points, 'r--')
plt.xlabel('Время (с)')
plt.ylabel('Амплитуда')
plt.title('Дискретизация и визуализация SIN сигнала (точки)')
plt.grid(True)

plt.tight_layout()
plt.show()

# Вычисление Фурье-образов
N = len(cos_signal_points)
k = np.arange(N)


Ex = np.exp(-1j * 2 * np.pi / N * np.outer(k, k))
Y_DFT_cos = np.dot(cos_signal_points, Ex)
Y_DFT_sin = np.dot(sin_signal_points, Ex)

# Вычисление с помощью np.fft.fft
fft_cos = np.fft.fft(cos_signal_points)
fft_sin = np.fft.fft(sin_signal_points)

# Построение графиков Фурье-образов
plt.figure(figsize=(15, 15))
plt.subplot(4, 2, 1)
plt.plot(k, np.real(Y_DFT_cos), 'r')
plt.title('COS сигнал (ДПФ, реальная часть)')
plt.grid(True)

plt.subplot(4, 2, 2)
plt.plot(k, np.imag(Y_DFT_cos), 'b')
plt.title('COS сигнал (ДПФ, мнимая часть)')
plt.grid(True)

plt.subplot(4, 2, 3)
plt.plot(k, np.real(Y_DFT_sin), 'g')
plt.title('SIN сигнал (ДПФ, реальная часть)')
plt.grid(True)

plt.subplot(4, 2, 4)
plt.plot(k, np.imag(Y_DFT_sin), 'y')
plt.title('SIN сигнал (ДПФ, мнимая часть)')
plt.grid(True)

plt.subplot(4, 2, 5)
plt.plot(k, np.real(fft_cos), 'r')
plt.title('COS сигнал (БПФ, реальная часть)')
plt.grid(True)

plt.subplot(4, 2, 6)
plt.plot(k, np.imag(fft_cos), 'b')
plt.title('COS сигнал (БПФ, мнимая часть)')
plt.grid(True)

plt.subplot(4, 2, 7)
plt.plot(k, np.real(fft_sin), 'g')
plt.title('SIN сигнал (БПФ, реальная часть)')
plt.grid(True)

plt.subplot(4, 2, 8)
plt.plot(k, np.imag(fft_sin), 'y')
plt.title('SIN сигнал (БПФ, мнимая часть)')
plt.grid(True)

plt.tight_layout()
plt.show()

# Вычисление квадратов модулей Фурье-образов
cos_square_DFT = Y_DFT_cos * np.conj(Y_DFT_cos)
sin_square_DFT = Y_DFT_sin * np.conj(Y_DFT_sin)

cos_square_FFT = fft_cos * np.conj(fft_cos)
sin_square_FFT = fft_sin * np.conj(fft_sin)

# Построение графиков квадратов модулей Фурье-образов
plt.figure(figsize=(15, 15))
plt.subplot(2, 2, 1)
plt.plot(k, cos_square_DFT, 'r')
plt.title('COS сигнал (ДПФ, квадрат модуля)')
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(k, sin_square_DFT, 'g')
plt.title('SIN сигнал (ДПФ, квадрат модуля)')
plt.grid(True)

plt.subplot(2, 2, 3)
plt.plot(k, cos_square_FFT, 'b')
plt.title('COS сигнал (БПФ, квадрат модуля)')
plt.grid(True)

plt.subplot(2, 2, 4)
plt.plot(k, sin_square_FFT, 'y')
plt.title('SIN сигнал (БПФ, квадрат модуля)')
plt.grid(True)

plt.tight_layout()
plt.show()

#бенчмарк
def time_dft(size):
    y = np.random.rand(size)
    start_time = time.time()
    N = len(y)
    k = np.arange(N)
    Ex = np.exp(-1j * 2 * np.pi / N * np.outer(k, k))
    Y = np.dot(y, Ex)
    return time.time() - start_time

# Функция для вычисления времени выполнения БПФ
def time_fft(size):
    y = np.random.rand(size)
    start_time = time.time()
    Y = np.fft.fft(y)
    return time.time() - start_time

# Создаем массив размеров
sizes = [2 ** i for i in range(7, 13)]

# Списки для времен выполнения ДПФ и БПФ
times_dft = []
times_fft = []

# Вычисляем время выполнения для каждого размера массива
for size in sizes:
    times_dft.append(time_dft(size))
    times_fft.append(time_fft(size))

# Построение графика
plt.figure(figsize=(10, 6))
plt.plot(sizes, times_dft, label='ДПФ')
plt.plot(sizes, times_fft, label='БПФ')
plt.title('Зависимость времени обработки от размерности массива')
plt.xlabel('Размерность массива (2^s)')
plt.ylabel('Время выполнения (с)')
plt.legend()
plt.grid(True)
plt.show()