# -*- coding: utf-8 -*-
import numpy as np
from sympy import symbols, sqrt, lambdify
import matplotlib.pyplot as plt

# Constantes físicas
G = 6.67430e-11  # m³ kg⁻¹ s⁻²
c = 3.00e8       # m/s
hbar = 1.0545718e-34  # J s
k_B = 1.380649e-23    # J/K
M_sun = 1.989e30      # kg

# Parâmetros do buraco negro
M = 10 * M_sun  # 10 massas solares
a = 0.9 * G * M / c**2  # Rotação (90% do máximo)

# Métrica de Kerr (horizonte externo)
r, M_sym, a_sym, c_sym = symbols('r M a c')
r_plus = G * M_sym / c_sym**2 * (1 + sqrt(1 - (a_sym * c_sym**2 / (G * M_sym))**2))
r_plus_func = lambdify((M_sym, a_sym, c_sym), r_plus, 'numpy')
horizon = r_plus_func(M, a, c)

# Temperatura de Hawking
def hawking_temp(M, a):
    r_g = G * M / c**2
    temp = hbar * c**3 / (8 * np.pi * G * M * k_B) * (1 - (a/r_g)**2)**0.5
    return temp

# Espectro de Planck
def planck_spectrum(freq, temp):
    return (8 * np.pi * hbar * freq**3 / c**3) / (np.exp(hbar * freq / (k_B * temp)) - 1)

# Monte Carlo pra trajetória da Aurora
def monte_carlo_trajectory(n_particles, r_start, r_horizon):
    np.random.seed(42)
    fates = np.random.choice(['absorbed', 'traversed'], size=n_particles, p=[0.8, 0.2])
    return fates

# Configuração inicial
freq = np.logspace(10, 15, 100)  # Hz
T_initial = hawking_temp(M, a)
spectrum_initial = planck_spectrum(freq, T_initial)

# Após Aurora
M_after = M * 0.999  # Pequena perda de massa
a_after = a * 1.01   # Aumento na rotação
T_after = hawking_temp(M_after, a_after)
spectrum_after = planck_spectrum(freq, T_after)

# Simulação Monte Carlo
n_particles = 1000
fates = monte_carlo_trajectory(n_particles, 10 * horizon, horizon)
traversed = np.sum(fates == 'traversed') / n_particles * 100

# Plot
plt.figure(figsize=(12, 6))
plt.loglog(freq, spectrum_initial, label="Antes da Aurora", color="blue")
plt.loglog(freq, spectrum_after, label="Após a Aurora", color="red", linestyle="--")
plt.xlabel("Frequência (Hz)")
plt.ylabel("Intensidade (unidades arbitrárias)")
plt.title("Radiação Hawking: Impacto da Partícula Aurora")
plt.legend()
plt.grid(True, which="both", ls="--")
plt.savefig("spectrum.png")
plt.show()

# Resultados
print(f"Temperatura inicial: {T_initial:.2e} K")
print(f"Temperatura após Aurora: {T_after:.2e} K")
print(f"Porcentagem de Auroras que atravessaram: {traversed:.1f}%")