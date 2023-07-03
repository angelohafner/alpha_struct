import numpy as np
from engineering_notation import EngNumber
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from funcoes import *


constants = define_constants()
deg, Ah, epsr_sal, As, hs, sigma, Tc, e, eps0, kB, NA, I = constants


Vo1 = 0.035700
Vc1 = 0.044607
Vc2 = 0.048602
Vd1 = 0.034571
Vd2 = 0.034474
Vq1 = 0.036637
Vq2 = 0.038070


Qc1 = 32.88
Qc2 = 62.57
Qd1 = 55.66
Qd2 = 36.07
Qq1 = 124.37
Qq2 = 66.82

V1 = Vo1
V2 = np.array([Vc1, Vc2, Vd1, Vd2, Vq1, Vd2])
angle_input = np.array([Qc1, Qc2, Qd1, Qd2, Qq1, Qd2])
ALPHAS = np.zeros(len(V2))
ERRORS = np.zeros(len(V2))

kappa = calcula_kappa(e, NA, I, epsr_sal, eps0, kB, Tc)

ALPHAS, ERRORS = calculate_alphas(V2, kappa, V1, eps0, epsr_sal, Ah, As, 10e-12, hs, sigma, angle_input)

print('ALPHAS =', ALPHAS)
print('ERRORS =', ERRORS)


mean_alphas = np.mean(ALPHAS)
std_dev_alphas = np.std(ALPHAS)

fig_barras = plot_bar_chart_with_mean_line(ALPHAS)
fig_barras.show()

fig_gauss = plot_gaussian(ALPHAS)
fig_gauss.show()



