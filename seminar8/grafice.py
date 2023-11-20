import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np


def corelograma(t, titlu, vmin=-1, vmax=1):
    fig = plt.figure(figsize=(9, 8))
    assert isinstance(fig, plt.Figure)
    ax = fig.add_subplot(1, 1, 1)
    assert isinstance(ax, plt.Axes)
    ax.set_title(titlu, fontsize=16)
    sb.heatmap(t, vmin=vmin, vmax=vmax,
               cmap="RdYlBu", annot=True, ax=ax)
    #plt.show()


# Functia intoarce numarul de componente semnificative
# Functia traseaza graficul variantei
def plot_varianta(alpha, eticheta_axax="Componente"):
    fig = plt.figure(figsize=(12, 8))
    assert isinstance(fig, plt.Figure)
    ax = fig.add_subplot(1, 1, 1)
    assert isinstance(ax, plt.Axes)
    ax.set_title("Plot varianta", fontsize=16)
    ax.set_xlabel(eticheta_axax)
    ax.set_ylabel("Varianta")
    m = len(alpha)
    x = np.array([i for i in range(1, m + 1)])
    ax.set_xticks(x)
    ax.plot(x, alpha, c='b')
    ax.scatter(x, alpha, c='r')
    # Criteriul Kaiser
    ax.axhline(1, c='g')
    ncomp_k = np.shape(np.where(alpha > 1)[0])[0]
    # Criteriul Cattell
    eps = alpha[:(m - 1)] - alpha[1:]
    sigma = eps[:(m - 2)] - eps[1:]
    exista_negative = sigma < 0
    if any(exista_negative):
        k = np.where(exista_negative)
        ncomp_c = k[0][0] + 2
        ax.axhline(alpha[k[0][0] + 1])
    else:
        ncomp_c = ncomp_k
    return min(ncomp_k, ncomp_c)


def plot_scoruri(x, titlu, nume_instante, k1=0, k2=1):
    fig = plt.figure(figsize=(12, 8))
    assert isinstance(fig, plt.Figure)
    ax = fig.add_subplot(1, 1, 1)
    assert isinstance(ax, plt.Axes)
    ax.set_aspect(1)
    ax.set_title(titlu, fontsize=16)
    ax.set_xlabel("a" + str(k1 + 1))
    ax.set_ylabel("a" + str(k2 + 1))
    ax.scatter(x[:, k1], x[:, k2], c='r')
    ax.axhline(0, c='k')
    ax.axvline(0, c='k')
    n = len(nume_instante)
    for i in range(n):
        ax.text(x[i, k1], x[i, k2], nume_instante[i])
        print(x[i, k1], x[i, k2], nume_instante[i])


def plot_variabile(r, titlu, nume_variabile, k1=0, k2=1, eticheta_axe="Comp"):
    fig = plt.figure(figsize=(8, 8))
    assert isinstance(fig, plt.Figure)
    ax = fig.add_subplot(1, 1, 1)
    assert isinstance(ax, plt.Axes)
    ax.set_title(titlu, fontsize=16)
    ax.set_xlabel(eticheta_axe + str(k1 + 1))
    ax.set_ylabel(eticheta_axe + str(k2 + 1))
    theta = np.arange(0, 2 * np.pi, 0.01)
    ax.plot(np.cos(theta), np.sin(theta))
    ax.plot(0.6 * np.cos(theta), 0.6 * np.sin(theta))
    ax.scatter(r[:, k1], r[:, k2], c='r')
    ax.axhline(0, c='k')
    ax.axvline(0, c='k')
    n = len(nume_variabile)
    for i in range(n):
        ax.text(r[i, k1], r[i, k2], nume_variabile[i])


def show():
    plt.show()
