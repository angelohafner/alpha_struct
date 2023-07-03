import numpy as np
from engineering_notation import EngNumber
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm


def define_constants():
    deg = np.pi / 180
    Ah = 1.745276391348672e-20
    epsr_sal = 71.715238
    As = 1.5e10
    hs = 0.05e-9
    sigma = 25e-3
    Tc = 338.0 - 273.15
    e = 1.602217663e-19
    eps0 = 8.854e-12
    kB = 1.380649e-23
    NA = 6.02214076e23
    I = 4.142

    return deg, Ah, epsr_sal, As, hs, sigma, Tc, e, eps0, kB, NA, I

# %%
def calcula_kappa(e, NA, I, epsr, eps0, kB, Tc):
    # kappa = np.sqrt(((2.0e03 * elec * elec * N_A) / (epson_0 * epson_3 * K_b)) * (I_ion / Tk))
    num = 2e3 * (e**2) * NA * I
    den = epsr * eps0 * kB * (Tc + 273.15)
    return np.sqrt(num/den)



 # %%
def f(x0, V1, V2, eps0, epsr_sal, Ah, As, kappa, hs):
    t1 = eps0 * epsr_sal * (kappa ** 2.0)
    t2 = 2 * V1 * V2 * np.cosh(kappa * x0)
    t3 = (V1 ** 2) + (V2 ** 2)
    num = t1 * (t2 - t3)
    den = (2 * np.sinh(kappa * x0) * np.sinh(kappa * x0))
    Pedl = num / den
    Ps = As * np.exp(-x0 / hs)
    Pvdw = Ah / (6 * np.pi * (x0 ** 3))
    soma = Pedl + Ps - Pvdw

    return soma

# %%
def newton_raphson(kappa, x0, V1, V2, eps0, epsr_sal, Ah, As, hs, N=100, erro=1):
    cont = 0
    dx = 1e-12
    while True:
        f0 = f(x0, V1, V2, eps0, epsr_sal, Ah, As, kappa, hs)
        g0 = (f(x0+dx, V1, V2, eps0, epsr_sal, Ah, As, kappa, hs) - f(x0-dx, V1, V2, eps0, epsr_sal, Ah, As, kappa, hs)) / (2*dx)

        if g0 == 0:
            g0 = 1e-3

        x1 = x0 - f0 / g0
        x0 = x1
        f1 = f(x1, V1, V2, eps0, epsr_sal, Ah, As, kappa, hs)

        cont += 1
        if cont > N:
            continua = False
            break

        if abs(f1) <= erro:
            continua = True
            break
    return [x1, continua]

# %%
def espessura_de_equilibrio(kappa, V1, V2, eps0, epsr_sal, Ah, As, hs, xhmin=0.01e-9):
    x1, continua = newton_raphson(kappa=kappa, x0=xhmin, V1=V1, V2=V2, eps0=eps0, epsr_sal=epsr_sal, Ah=Ah, As=As, hs=hs, N=100, erro=1)
    if continua:
        root = x1
        saida = x1
    else:
        saida = 'Not Convergent.'
    return saida


# %%
def graficos(xx, yy, titulo_y="Pressão [Pa]", titulo_x="Posição [m]"):
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Scatter(x=xx, y=yy, mode='lines', name='Structural'), row=1, col=1)
    fig.update_yaxes(title_text=titulo_y, title_font=dict(size=10), row=1, col=1)
    fig.update_xaxes(title_text=titulo_x, title_font=dict(size=10), row=1, col=1)
    lim = find_extremos(yy)
    if lim is not None:
        fig.update_yaxes(range=[lim[0], lim[1]], row=1, col=1)


    return fig

# %%
def find_extremos(vector):
    local_max = None
    local_min = None

    for i in range(1, len(vector) - 1):  # Vamos ignorar os pontos finais
        # Encontre o máximo local
        if vector[i - 1] < vector[i] > vector[i + 1]:
            if local_max is None or vector[i] > vector[local_max]:
                local_max = i

        # Encontre o mínimo local
        if vector[i - 1] > vector[i] < vector[i + 1]:
            if local_min is None or vector[i] < vector[local_min]:
                local_min = i

    return (vector[local_max], vector[local_min]) if local_max is not None and local_min is not None else None

# %%
def calculo_do_alpha(h0, V1, V2, eps0, epsr_sal, Ah, As, kappa, hs, sigma, theta_rads, fit):
    temp1 = eps0 * epsr_sal * kappa * V1 * V2 * csch(kappa*h0)
    temp2 = eps0 * epsr_sal * kappa / 2 * (V1**2 + V2**2) * coth(kappa*h0)
    Xedl = (temp1 - temp2)/sigma
    Xvdw = -Ah / (12 * np.pi * sigma * h0**2)
    Xst = fit * As * hs * sigma * np.exp(-h0 / hs)
    #ARG = 1.0 + Xedl + Xvdw + Xst
    alpha = (np.cos(theta_rads) - 1 - Xedl - Xvdw) / Xst
    return alpha

def coth(x):
    return 1. / np.tanh(x)

def csch(x):
    return 1. / np.sinh(x)

def alpha_recursivity(h0, V1, V2, eps0, epsr_sal, Ah, As, kappa, hs, sigma, theta_rads, fit, alpha0):
    temp1 = eps0 * epsr_sal * kappa * V1 * V2 * csch(kappa*h0)
    temp2 = eps0 * epsr_sal * kappa / 2 * (V1**2 + V2**2) * coth(kappa*h0)
    Xedl = (temp1 - temp2)/sigma
    Xvdw = -Ah / (12 * np.pi * sigma * h0**2)
    erro = 10
    cont = 0
    NN = 100000
    ALPHA = np.zeros(NN)
    ERRO = np.zeros(NN)
    ALPHA[0] = alpha0
    for ii in range(NN-1):
        Xst = fit * alpha0 * As * hs * sigma * np.exp(-h0 / hs)
        #ARG = 1.0 + Xedl + Xvdw + Xst
        alpha1 = (np.cos(theta_rads) - 1 - Xedl - Xvdw) / Xst
        erro = np.abs(alpha0 - alpha1)
        if (alpha0 - alpha1) > 0.01:
            alpha0 = alpha0 - 0.01
        elif (alpha0 - alpha1) < -0.01:
            alpha0 = alpha0 + 0.01
        else:
            alpha0 = alpha1

        ALPHA[ii + 1] = alpha0
        ERRO[ii + 1] = erro


    return [ALPHA, ERRO]


def plot_vector(vector, plot_title='Line plot from a vector', xaxis_title='Index', yaxis_title='Value'):
    # Create a trace
    trace = go.Scatter(
        x = list(range(len(vector))), # X values - the index of the vector
        y = vector, # Y values - the actual vector data
        mode = 'lines' # The type of plot you want to draw
    )

    data = [trace]

    # Create a layout
    layout = go.Layout(
        title=plot_title,
        xaxis=dict(title=xaxis_title),
        yaxis=dict(title=yaxis_title, type='log')  # Corrected here
    )
    # Create a Figure object
    fig = go.Figure(data=data, layout=layout)

    return fig



def plot_vector_loop(vector, fig, plot_title='Line plot from a vector', xaxis_title='Index', yaxis_title='Value'):
    # Create a trace
    trace = go.Scatter(
        x = list(range(len(vector))), # X values - the index of the vector
        y = vector, # Y values - the actual vector data
        mode = 'lines', # The type of plot you want to draw
        name = plot_title # This will be used in the legend
    )

    # Add the trace to the figure
    fig.add_trace(trace)

    # Atualizando os eixos x e y após adicionar todos os traces
    fig.update_layout(
        xaxis_title='Iteration',
        yaxis_title='Value',
        yaxis_type="log",
        title='Alpha'
    )


def plot_gaussian(vector, bins=5):
    # Calcula a média e o desvio padrão
    mu = np.mean(vector)
    sigma = np.std(vector)

    # Cria um vetor de pontos para o eixo x
    x = np.linspace(min(vector)-1, max(vector)+1, 100)

    # Calcula a PDF para o eixo y
    y = norm.pdf(x, mu, sigma)

    # Cria a figura
    fig = go.Figure()

    # Adiciona a curva gaussiana
    fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name='Curva Gaussiana'))

    # Adiciona o histograma
    fig.add_trace(go.Histogram(x=vector, nbinsx=bins, histnorm='probability density', name='Dados'))

    # Configura o layout
    fig.update_layout(
        title='Curva Gaussiana',
        xaxis_title='Valores',
        yaxis_title='Densidade',
        bargap=0.01, # gap between bars of adjacent location coordinates
        bargroupgap=0.1 # gap between bars of the same location coordinates
    )

    # Mostra o gráfico

    return fig


def plot_bar_chart_with_mean_line(data):
    fig = go.Figure()
    # Gráfico de barras
    fig.add_trace(go.Bar(x=list(range(len(data))), y=data, name="Alphas", text=data, textposition='auto'))

    # Linha da média
    mean_value = np.mean(data)
    fig.add_shape(
        type="line",
        x0=0,
        y0=mean_value,
        x1=len(data) - 1,
        y1=mean_value,
        line=dict(color="red", width=3, dash="dash"),
        name="Mean"
    )
    # Rótulos dos eixos
    fig.update_layout(
        xaxis_title="#",
        yaxis_title="Valor de $\\alpha$"
    )
    return fig


import numpy as np
import plotly.graph_objects as go

def calculate_alphas(V2, kappa, V1, eps0, epsr_sal, Ah, As, xhmin, hs, sigma, angle_input):
    ALPHAS = np.zeros(len(V2))
    ERRORS = np.zeros(len(V2))

    for ii in range(len(V2)):
        ee = espessura_de_equilibrio(kappa=kappa, V1=V1, V2=V2[ii], eps0=eps0, epsr_sal=epsr_sal, Ah=Ah, As=As, xhmin=xhmin, hs=hs)
        print('h0 = ', ee)

        xx = np.geomspace(1e-12, 10e-9, 100000)
        yy = f(xx, V1, V2[ii], eps0, epsr_sal, Ah, As, kappa, hs)
        # fig1 = graficos(xx, yy)
        # fig1.show()

        aa0 = calculo_do_alpha(ee, V1, V2[ii], eps0, epsr_sal, Ah, As, kappa, hs, sigma, angle_input[ii], 1)

        aan, eee = alpha_recursivity(ee, V1, V2[ii], eps0, epsr_sal, Ah, As, kappa, hs, sigma, angle_input[ii], 1, aa0)
        print(aan[-1], eee[-1])

        ALPHAS[ii] = np.round(aan[-1], 1)
        ERRORS[ii] = eee[-1]

        print("----------------------------")

    return ALPHAS, ERRORS
