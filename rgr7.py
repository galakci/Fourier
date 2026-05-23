import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import time
from scipy.fft import fft, ifft, fftfreq

x = np.linspace(-np.pi, np.pi, 1000)

#  f(x) = x
def f1(x):
    x_norm = ((x + np.pi) % (2*np.pi)) - np.pi
    return x_norm
# Аналитические коэффициенты Фурье для f(x) = x
def fourier_sum_f1(x, N):
    S = np.zeros_like(x)
    for n in range(1, N+1):
        b_n = 2 * (-1)**(n+1) / n
        S += b_n * np.sin(n * x)
    return S


#  f(x) = |x|

def f2(x):
    x_norm = ((x + np.pi) % (2*np.pi)) - np.pi
    return np.abs(x_norm)

# Аналитические коэффициенты Фурье для f(x) = |x|
def fourier_sum_f2(x, N):
    S = np.full_like(x, np.pi/2)  # a0/2
    for n in range(1, N+1):
        if n % 2 == 1:
            a_n = -4 / (np.pi * n**2)
            S += a_n * np.cos(n * x)
    return S


#  f(x) = sign(x)
def f3(x):
    x_norm = ((x + np.pi) % (2*np.pi)) - np.pi
    return np.where(x_norm > 0, 1, np.where(x_norm < 0, -1, 0))
# Аналитические коэффициенты Фурье для sign(x)
def fourier_sum_f3(x, N):

    S = np.zeros_like(x)
    for n in range(1, N+1):
        if n % 2 == 1:
            b_n = 4 / (np.pi * n)
            S += b_n * np.sin(n * x)
    return S


# Построение графиков


N_values = [1, 3, 5, 10, 30]

#  f(x) = x 
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Разложение в ряд Фурье: $f(x) = x$', fontsize=16, fontweight='bold')

for idx, N in enumerate(N_values):
    row, col = idx // 3, idx % 3
    ax = axes[row, col]
    
    ax.plot(x, f1(x), 'k-', linewidth=1.5, alpha=0.4, label='$f(x)=x$')
    S = fourier_sum_f1(x, N)
    ax.plot(x, S, 'r-', linewidth=2, label=f'$S_{{{N}}}(x)$')
    
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-4, 4)
    ax.set_title(f'$N = {N}$')
    ax.axhline(y=0, color='gray', linewidth=0.5)
    ax.axvline(x=0, color='gray', linewidth=0.5)
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)

ax = axes[1, 2]
ax.plot(x, f1(x), 'k-', linewidth=2, alpha=0.5, label='$f(x)=x$')
colors = plt.cm.viridis(np.linspace(0, 1, len(N_values)))
for N, color in zip(N_values, colors):
    S = fourier_sum_f1(x, N)
    ax.plot(x, S, '-', color=color, linewidth=1.5, alpha=0.7, label=f'$N={N}$')
ax.set_xlim(-np.pi, np.pi)
ax.set_ylim(-4, 4)
ax.set_title('Сравнение частичных сумм')
ax.legend(loc='upper left', fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('fourier_f1_x.png', dpi=150, bbox_inches='tight')
plt.show()

#  f(x) = |x| 
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Разложение в ряд Фурье: $f(x) = |x|$', fontsize=16, fontweight='bold')

for idx, N in enumerate(N_values):
    row, col = idx // 3, idx % 3
    ax = axes[row, col]
    
    ax.plot(x, f2(x), 'k-', linewidth=1.5, alpha=0.4, label='$f(x)=|x|$')
    S = fourier_sum_f2(x, N)
    ax.plot(x, S, 'b-', linewidth=2, label=f'$S_{{{N}}}(x)$')
    
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-0.5, 3.5)
    ax.set_title(f'$N = {N}$')
    ax.axhline(y=0, color='gray', linewidth=0.5)
    ax.axvline(x=0, color='gray', linewidth=0.5)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

ax = axes[1, 2]
ax.plot(x, f2(x), 'k-', linewidth=2, alpha=0.5, label='$f(x)=|x|$')
colors = plt.cm.viridis(np.linspace(0, 1, len(N_values)))
for N, color in zip(N_values, colors):
    S = fourier_sum_f2(x, N)
    ax.plot(x, S, '-', color=color, linewidth=1.5, alpha=0.7, label=f'$N={N}$')
ax.set_xlim(-np.pi, np.pi)
ax.set_ylim(-0.5, 3.5)
ax.set_title('Сравнение частичных сумм')
ax.legend(loc='upper right', fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('fourier_f2_absx.png', dpi=150, bbox_inches='tight')
plt.show()

# f(x) = sign(x) 
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Разложение в ряд Фурье: $f(x) = \\operatorname{sign}(x)$', fontsize=16, fontweight='bold')

for idx, N in enumerate(N_values):
    row, col = idx // 3, idx % 3
    ax = axes[row, col]
    
    ax.plot(x, f3(x), 'k-', linewidth=1.5, alpha=0.4, label='$f(x)=\\operatorname{sign}(x)$')
    S = fourier_sum_f3(x, N)
    ax.plot(x, S, 'g-', linewidth=2, label=f'$S_{{{N}}}(x)$')
    
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-1.5, 1.5)
    ax.set_title(f'$N = {N}$')
    ax.axhline(y=0, color='gray', linewidth=0.5)
    ax.axvline(x=0, color='gray', linewidth=0.5)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

ax = axes[1, 2]
ax.plot(x, f3(x), 'k-', linewidth=2, alpha=0.5, label='$f(x)=\\operatorname{sign}(x)$')
colors = plt.cm.viridis(np.linspace(0, 1, len(N_values)))
for N, color in zip(N_values, colors):
    S = fourier_sum_f3(x, N)
    ax.plot(x, S, '-', color=color, linewidth=1.5, alpha=0.7, label=f'$N={N}$')
ax.set_xlim(-np.pi, np.pi)
ax.set_ylim(-1.5, 1.5)
ax.set_title('Сравнение частичных сумм')
ax.legend(loc='upper right', fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('fourier_f3_sign.png', dpi=150, bbox_inches='tight')
plt.show()


# График эффекта Гиббса для sign(x)

fig, ax = plt.subplots(figsize=(12, 6))

x_zoom = np.linspace(-0.5, 0.5, 1000)
N_zoom = [5, 10, 20, 50, 100]

ax.plot(x_zoom, f3(x_zoom), 'k-', linewidth=2.5, alpha=0.4, label='$\\operatorname{sign}(x)$')
ax.axhline(y=0, color='black', linestyle=':', linewidth=1, alpha=0.5)

colors = plt.cm.plasma(np.linspace(0, 0.9, len(N_zoom)))
for N, color in zip(N_zoom, colors):
    S = fourier_sum_f3(x_zoom, N)
    ax.plot(x_zoom, S, '-', color=color, linewidth=1.5, label=f'$S_{{{N}}}(x)$')

overshoot_level = 1.17898
ax.axhline(y=overshoot_level, color='red', linestyle='--', linewidth=1, alpha=0.7)
ax.axhline(y=-overshoot_level, color='red', linestyle='--', linewidth=1, alpha=0.7)
ax.annotate('Эффект Гиббса\n(~1.18)', xy=(0.15, 1.18), xytext=(0.3, 1.35),
            arrowprops=dict(arrowstyle='->', color='red'),
            fontsize=10, color='red', fontweight='bold')

ax.set_xlim(-0.5, 0.5)
ax.set_ylim(-1.5, 1.5)
ax.set_xlabel('$x$', fontsize=14)
ax.set_ylabel('$S_N(x)$', fontsize=14)
ax.set_title('Эффект Гиббса для $f(x) = \\operatorname{sign}(x)$ вблизи разрыва', fontsize=14, fontweight='bold')
ax.legend(loc='upper left', fontsize=9, ncol=2)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('gibbs_effect.png', dpi=150, bbox_inches='tight')
plt.show()


# График ошибки аппроксимации в зависимости от N

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

N_range = np.arange(1, 51)
x_dense = np.linspace(-np.pi, np.pi, 2000)

functions = [
    (f1, fourier_sum_f1, x_dense, '$f(x)=x$'),
    (f2, fourier_sum_f2, x_dense, '$f(x)=|x|$'),
    (f3, fourier_sum_f3, x_dense, '$f(x)=\\operatorname{sign}(x)$')
]

for ax, (f, fourier_sum, x_pts, title) in zip(axes, functions):
    errors = []
    f_exact = f(x_pts)
    
    for N in N_range:
        S = fourier_sum(x_pts, N)
        mse = np.mean((f_exact - S)**2)
        errors.append(mse)
    
    ax.plot(N_range, errors, 'b-o', markersize=3, linewidth=1.5)
    ax.set_xlabel('Число гармоник $N$', fontsize=12)
    ax.set_ylabel('Среднеквадратичная ошибка', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

plt.suptitle('Сходимость частичных сумм ряда Фурье', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('convergence_error.png', dpi=150, bbox_inches='tight')
plt.show()




# ЧИСЛЕННЫЙ МЕТОД ТРАПЕЦИЙ


def fourier_coeffs_trapezoid(f, N, M=10000):
    x = np.linspace(-np.pi, np.pi, M)
    f_vals = f(x)
    
    a0 = np.trapezoid(f_vals, x) / np.pi
    
    a = np.zeros(N)
    b = np.zeros(N)
    
    for n in range(1, N + 1):
        a[n-1] = np.trapezoid(f_vals * np.cos(n * x), x) / np.pi
        b[n-1] = np.trapezoid(f_vals * np.sin(n * x), x) / np.pi
    
    return a0, a, b, x


def fourier_sum_from_coeffs(a0, a, b, x_pts, N_use):

    S = np.full_like(x_pts, a0 / 2.0)
    for n in range(N_use):
        S += a[n] * np.cos((n+1) * x_pts) + b[n] * np.sin((n+1) * x_pts)
    return S



# БЫСТРОЕ ПРЕОБРАЗОВАНИЕ ФУРЬЕ (FFT)


def fourier_coeffs_fft(f, N_harmonics, M=1024):
    M = 2 ** int(np.ceil(np.log2(M)))
    
    x_fft = np.linspace(0, 2*np.pi, M, endpoint=False)
    f_vals = f(x_fft)
    
    spectrum = fft(f_vals)
    freqs = fftfreq(M, d=(x_fft[1] - x_fft[0]) / (2*np.pi))
    
    c = spectrum / M
    
    a0 = 2.0 * np.real(c[0])
    
    a = np.zeros(N_harmonics)
    b = np.zeros(N_harmonics)
    
    for n in range(1, N_harmonics + 1):
        idx = np.argmin(np.abs(freqs - n))
        a[n-1] = 2.0 * np.real(c[idx])
        b[n-1] = -2.0 * np.imag(c[idx])
    
    return a0, a, b, freqs, spectrum


def fourier_sum_fft(a0, a, b, x_pts, N_use):

    S = np.full_like(x_pts, a0 / 2.0)
    for n in range(N_use):
        S += a[n] * np.cos((n+1) * x_pts) + b[n] * np.sin((n+1) * x_pts)
    return S


# СРАВНЕНИЕ МЕТОДОВ: ТРАПЕЦИИ vs FFT








data = np.array([
    1.0, 1.34, 1.75, 2.18, 2.53, 2.71, 2.65, 2.37, 1.97, 1.54,
    1.16, 0.86, 0.64, 0.5, 0.42, 0.37, 0.36, 0.39, 0.45, 0.56, 0.74
])

n_points = len(data)
t = np.arange(n_points)
t_normalized = np.linspace(0, 2*np.pi, n_points, endpoint=False)


def compute_fft_spectrum(signal):
    n = len(signal)
    spectrum = fft(signal)
    freqs = fftfreq(n)
    amplitudes = np.abs(spectrum) / n
    amplitudes[0] = amplitudes[0] / 2
    phases = np.angle(spectrum)
    return freqs, spectrum, amplitudes, phases


def find_dominant_harmonics(amplitudes, freqs, top_n=5):

    n = len(amplitudes)
    pos_indices = np.arange(1, n//2 + 1)
    pos_amps = amplitudes[pos_indices]
    pos_freqs = freqs[pos_indices]
    sorted_idx = np.argsort(pos_amps)[::-1]
    top_indices = sorted_idx[:top_n]
    dominant_freqs = pos_freqs[top_indices]
    dominant_amps = pos_amps[top_indices]
    return dominant_freqs, dominant_amps


def filter_signal_fft(signal, n_harmonics):

    n = len(signal)
    spectrum = fft(signal)
    mask = np.zeros(n, dtype=complex)
    mask[0] = spectrum[0]
    
    pos_range = np.arange(1, n//2 + 1)
    pos_amps = np.abs(spectrum[pos_range])
    top_idx = np.argsort(pos_amps)[::-1][:n_harmonics]
    
    for idx in top_idx:
        mask[pos_range[idx]] = spectrum[pos_range[idx]]
        if pos_range[idx] != n - pos_range[idx]:
            mask[n - pos_range[idx]] = spectrum[n - pos_range[idx]]
    
    filtered_spectrum = mask
    filtered_signal = np.real(ifft(filtered_spectrum))
    
    return filtered_signal, filtered_spectrum


#  РАЗЛОЖЕНИЕ СИГНАЛА


def plot_spectrum_analysis():
    freqs, spectrum, amplitudes, phases = compute_fft_spectrum(data)
    dom_freqs, dom_amps = find_dominant_harmonics(amplitudes, freqs, top_n=5)

    
    harmonics_list = [1, 2, 3, 5]
    filtered_signals = {}
    for n in harmonics_list:
        filtered_signals[n], _ = filter_signal_fft(data, n)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    ax = axes[0, 0]
    ax.plot(t, data, 'b-o', linewidth=2, markersize=8, markerfacecolor='white')
    ax.set_xlabel('t')
    ax.set_ylabel('f(t)')
    ax.set_title('Исходный сигнал')
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    n = len(data)
    pos_mask = freqs[:n//2 + 1] >= 0
    ax.stem(freqs[:n//2 + 1][pos_mask], amplitudes[:n//2 + 1][pos_mask],
            linefmt='b-', markerfmt='bo', basefmt='k-')
    ax.set_xlabel('Частота (номер гармоники)')
    ax.set_ylabel('Амплитуда')
    ax.set_title('Амплитудный спектр (FFT)')
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 2]
    ax.stem(freqs[:n//2 + 1][pos_mask], phases[:n//2 + 1][pos_mask],
            linefmt='r-', markerfmt='ro', basefmt='k-')
    ax.set_xlabel('Частота (номер гармоники)')
    ax.set_ylabel('Фаза (рад)')
    ax.set_title('Фазовый спектр')
    ax.grid(True, alpha=0.3)
    
    filter_plots = [1, 2, 3]
    for i, n_harm in enumerate(filter_plots):
        ax = axes[1, i]
        ax.plot(t, data, 'k-o', linewidth=1.5, markersize=6, alpha=0.4, label='Исходный')
        ax.plot(t, filtered_signals[n_harm], 'r-s', linewidth=2, markersize=8,
                markerfacecolor='white', label=f'Фильтрованный ({n_harm} гарм.)')
        ax.set_xlabel('t')
        ax.set_ylabel('f(t)')
        ax.set_title(f'Фильтрация сигнала ({n_harm} гарм.)')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Разложение сигнала: спектральный анализ', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('spectrum_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return freqs, amplitudes, filtered_signals




def plot_methods_comparison():

    
    M_range = np.array([33, 65, 129, 257, 513, 1025, 2049, 4097])
    b1_exact = 2.0  
    
    errors_trap = []
    errors_fft  = []
    errors_analytic = []
    times_trap  = []
    times_fft   = []
    times_analytic = []
    
    for M in M_range:
        # Аналитический метод 

        S_analytic = fourier_sum_f1(np.array([0.0]), 1)

        errors_analytic.append(0.0)  
        
        # Трапеции

        x_t = np.linspace(-np.pi, np.pi, M)
        f_t = f1(x_t)
        b1_t = np.trapezoid(f_t * np.sin(x_t), x_t) / np.pi

        errors_trap.append(abs(b1_t - b1_exact))
        
        # FFT

        _, _, b_f, _, _ = fourier_coeffs_fft(f1, 1, M)

        errors_fft.append(abs(b_f[0] - b1_exact))
    
    errors_trap = np.array(errors_trap)
    errors_fft  = np.array(errors_fft)
    errors_analytic = np.array(errors_analytic)

    

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    

    x_plot = np.linspace(-np.pi, np.pi, 500)
    N_plot = 10
    
    S_analytic = fourier_sum_f1(x_plot, N_plot)
    
    a0_trap, a_trap, b_trap, _ = fourier_coeffs_trapezoid(f1, N_plot, 20)
    S_trap = fourier_sum_from_coeffs(a0_trap, a_trap, b_trap, x_plot, N_plot)
    
    a0_fft, a_fft, b_fft, _, _ = fourier_coeffs_fft(f1, N_plot, 20)
    S_fft = fourier_sum_fft(a0_fft, a_fft, b_fft, x_plot, N_plot)
    
    # График 1: Сравнение частичных сумм
    ax = axes[0]
    ax.plot(x_plot, f1(x_plot), 'k-', linewidth=4, alpha=0.3, label='$f(x)=x$ (точная)')
    ax.plot(x_plot, S_analytic, 'g-', linewidth=3, alpha=0.6, label=f'Аналитический (N={N_plot})')
    ax.plot(x_plot, S_trap, 'r--', linewidth=3, alpha=0.6, label=f'Трапеции ')
    ax.plot(x_plot, S_fft, 'b:', linewidth=3, alpha=0.6, label=f'FFT ')
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-4, 4)
    ax.set_xlabel('$x$', fontsize=13)
    ax.set_ylabel('$S_N(x)$', fontsize=13)
    ax.set_title(f'Сравнение частичных сумм (N={N_plot})', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # График 2: Ошибка частичных сумм относительно аналитической
    ax = axes[1]
    error_trap = np.abs(S_trap - S_analytic)
    error_fft = np.abs(S_fft - S_analytic)
    
    ax.semilogy(x_plot, error_trap, 'r-', linewidth=2, alpha=1, 
                label='|Трапеции - Аналитический|')
    ax.semilogy(x_plot, error_fft, 'b-', linewidth=1.5, alpha=0.8, 
                label='|FFT - Аналитический|')
    ax.set_xlim(-np.pi, np.pi)
    ax.set_xlabel('$x$', fontsize=13)
    ax.set_ylabel('Абсолютная ошибка', fontsize=13)
    ax.set_title('Ошибка относительно аналитического разложения', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Сравнение методов вычисления коэффициентов Фурье\n'
                 '(аналитический, трапеции, FFT) для $f(x)=x$', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('methods_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
   











    

freqs, amplitudes, filtered = plot_spectrum_analysis()

plot_methods_comparison()


    


    
   
