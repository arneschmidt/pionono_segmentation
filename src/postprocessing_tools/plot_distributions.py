"""
This is a script to plot the different distributions q(z|r) of the raters.
"""


import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from matplotlib.ticker import MaxNLocator

matplotlib.rcParams.update({'font.size': 11})

def plot_and_save_distributions(mu_list, cov_list):
    # plt.figure(figsize=(4, 4.2))
    ax = plt.figure(figsize=(4, 4.2)).gca()
    no_annotators = mu_list.shape[0]

    twodim_mu_list = np.zeros(shape=[no_annotators, 2])
    twodim_cov_list = np.zeros(shape=[no_annotators, 2, 2])
    for i in range(no_annotators):
        twodim_mu_list[i] = mu_list[i][0:2]
        twodim_cov_list[i] = cov_list[i][0:2, 0:2]

    # Initializing the random seed
    random_seed = 0

    x = np.linspace(-8, 5, num=100)
    y = np.linspace(-8, 7, num=100)
    X, Y = np.meshgrid(x, y)
    # ...

    pdf_list = []

    for i in range(no_annotators):
        mean = twodim_mu_list[i]
        cov = twodim_cov_list[i]
        distr = multivariate_normal(cov=cov, mean=mean,
                                    seed=random_seed)
        pdf = np.zeros(X.shape)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                pdf[i, j] = distr.pdf([X[i, j], Y[i, j]])
        pdf_list.append(pdf)

   # Plotting contour plots
    annotators = ['r=gold', 'r=1', 'r=2', 'r=3', 'r=4', 'r=5', 'r=6']

    colors = ['gold', 'tab:orange', 'tab:blue', 'tab:red', 'tab:purple', 'tab:cyan', 'tab:green']
    legend_list =[]
    for idx, val in reversed(list(enumerate(pdf_list))):
        print(idx)
        print()
        contourline = np.max(val) * (1/2)
        plt.plot(twodim_mu_list[idx][0], twodim_mu_list[idx][1], color=colors[idx], marker='o', alpha=0.8)
        cntr = plt.contour(X, Y, val, levels=[contourline], colors=colors[idx], alpha=0.8)
        # cntr = plt.contourf(X, Y, val, cmap=colormaps[idx], alpha=0.7)
        # surf = ax.plot_surface(X, Y, val, rstride=1, cstride=1, cmap=colormaps[idx], alpha=0.7)
        h, _ = cntr.legend_elements()
        legend_list.append(h[0])
    plt.legend(legend_list, reversed(annotators), loc='upper left')
    plt.ylabel('Latent Space (dim 2)')
    plt.xlabel('Latent Space (dim 1)')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.tight_layout()
    plt.savefig("dist_plot.png")


path = '/home/arne/projects/segmentation_crowdsourcing/distributions/'

files = os.listdir(path)
mu_list = []
sigma_list = []

for f in sorted(files):
    if 'mu' in f:
        z = pd.read_csv(path + f, sep=',', header=None)
        mu_list.append(np.array(z).flatten())
    elif 'cov' in f:
        z = pd.read_csv(path + f, sep=',', header=None)
        sigma_list.append(np.array(z))

plot_and_save_distributions(np.array(mu_list), np.array(sigma_list))


