"""
This script is used to demonstrate how GP works for my thesis

Author: Yaolin Ge
Email: geyaolin@gmail.com
Date: 2023-11-26
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from matplotlib.cm import get_cmap
from scipy.stats import norm
from src.usr_func.interpolate_2d import interpolate_2d
from pykdtree.kdtree import KDTree
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['font.size'] = 20
mpl.rcParams['font.family'] = 'Times New Roman'
np.random.seed(0)
border = np.array([[0, 0], [0, 1], [1, 1], [1, 0], [0, 0]])

figpath = os.getcwd() + "/../../OneDrive - NTNU/MASCOT_PhD/Thesis/fig/"

def plotf(grid, value, cmap: str = "BrBG", title: str = None, vmin: float = None, vmax: float = None):
    gx, gy, gv = interpolate_2d(grid[:, 0], grid[:, 1], 100, 100, value, "cubic")
    plt.scatter(gx, gy, c=gv, cmap=get_cmap(cmap, 10))
    plt.colorbar()
    plt.clim(vmin, vmax)
    plt.xlabel("East")
    plt.ylabel("North")
    plt.title(title)
    plt.xlim(np.amin(grid[:, 0]), np.amax(grid[:, 0]))
    plt.ylim(np.amin(grid[:, 1]), np.amax(grid[:, 1]))
    ax = plt.gca()
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    # ax.set_aspect('equal')

# s0, set up the parameters
gridsize = .1
xv = np.arange(0, 1, gridsize)
yv = np.arange(0, 1, gridsize)
xx, yy = np.meshgrid(xv, yv)
grid = np.hstack((xx.reshape(-1, 1), yy.reshape(-1, 1)))
grid_kdtree = KDTree(grid)
threshold = 0.5
sigma = .2
eta = 4.5 / .6
nugget = .01
tau = np.sqrt(nugget)
dm = cdist(grid, grid)
cov = sigma ** 2 * ((1 + eta * dm) * np.exp(-eta * dm))

# s2, generate random field, make the prior mean to be increasing linearly on the x axis.
mu = grid[:, 0]
L = np.linalg.cholesky(cov)
mu_truth = mu + np.dot(L, np.random.normal(size=mu.shape[0]))

def make_plots(mu, cov, filename: str = None, loc: np.ndarray = None):
    """ Make plots for conditional mean, conditional variance, excursion set and excursion probability. """
    fig = plt.figure(figsize=(18, 5))
    gs = GridSpec(nrows=1, ncols=3)
    ax = fig.add_subplot(gs[0])
    plotf(grid, mu, "BrBG", "Mean", vmin=0, vmax=1)
    if loc is not None:
        plt.plot(loc[:, 0], loc[:, 1], "k-", markersize=10)
    # ax = fig.add_subplot(gs[0, 1])
    # plotf(grid, np.sqrt(np.diag(cov)), "RdBu", "Uncertainty", .075, .2)
    es = mu < threshold
    ax = fig.add_subplot(gs[1])
    plotf(grid, es, "Blues", "Excursion set")
    if loc is not None:
        plt.plot(loc[:, 0], loc[:, 1], "k-", markersize=10)
    ep = norm.cdf(threshold, mu, np.sqrt(np.diag(cov)))
    ax = fig.add_subplot(gs[2])
    plotf(grid, ep, "Reds", "Excursion probability")
    if loc is not None:
        plt.plot(loc[:, 0], loc[:, 1], "k-", markersize=10)
    plt.tight_layout()
    # plt.show()
    plt.savefig(figpath + filename)
    plt.close("all")

# make_plots(mu, cov, "gp_prior.png")

#%% plot the realizations of the ground truth
# fig = plt.figure(figsize=(18, 5))
# gs = GridSpec(nrows=1, ncols=3)
seeds = [0, 1, 2]
names = ["I", "II", "III"]
for seed in seeds:
    np.random.seed(seed)
    mu_truth = mu + np.dot(L, np.random.normal(size=mu.shape[0]))
    # ax = fig.add_subplot(gs[seed])
    plotf(grid, mu_truth, "BrBG", "Realization " + names[seed], vmin=0, vmax=1)
    plt.show()

# plt.tight_layout()
# plt.savefig(figpath + "gp_realizations.png")
# plt.show()

#%%
np.random.seed(2)
mu_truth = mu + np.dot(L, np.random.normal(size=mu.shape[0]))
# s6, test updating the GP using the measured data.
yloc = np.arange(0, 1, gridsize)
xloc = np.ones(yloc.shape) * .5
loc = np.stack((xloc, yloc), axis=1)
data = np.hstack((loc, mu_truth[grid_kdtree.query(loc)[1]].reshape(-1, 1)))

msamples = data.shape[0]
F = np.zeros([msamples, grid.shape[0]])
ind_measured = grid_kdtree.query(data[:, :2])[1]
for i in range(msamples):
    F[i, ind_measured[i]] = True
R = np.eye(msamples) * tau ** 2
C = F @ cov @ F.T + R
mu_cond = mu + cov @ F.T @ np.linalg.solve(C, (data[:, -1] - F @ mu))
cov_cond = cov - cov @ F.T @ np.linalg.solve(C, F @ cov)

make_plots(mu_cond, cov_cond, "gp_posterior2.png", loc=data[:, :2])
print("finished")

#%% Calculate EIBV and IVR field
from numba import jit
from scipy.stats import multivariate_normal
from src.usr_func.normalize import normalize
from tqdm import tqdm

# s0, load cdf table to prepare the fast calculation of EIBV.
cdf_table_raw = np.load(os.getcwd() + "/prior/cdf.npz")
cdf_z1 = cdf_table_raw["z1"]
cdf_z2 = cdf_table_raw["z2"]
cdf_rho = cdf_table_raw["rho"]
cdf_table = cdf_table_raw["cdf"]

@jit
def get_eibv_analytical_fast(mu: np.ndarray, sigma_diag: np.ndarray, vr_diag: np.ndarray,
                               threshold: float, cdf_z1: np.ndarray, cdf_z2: np.ndarray,
                               cdf_rho: np.ndarray, cdf_table: np.ndarray) -> float:
    """
    Calculate the eibv using the analytical formula but using a loaded cdf dataset.
    """
    eibv = .0
    for i in range(len(mu)):
        sn2 = sigma_diag[i]
        vn2 = vr_diag[i]

        sn = np.sqrt(sn2)
        m = mu[i]

        mur = (threshold - m) / sn

        sig2r_1 = sn2 + vn2
        sig2r = vn2

        z1 = mur
        z2 = -mur
        rho = -sig2r / sig2r_1

        ind1 = np.argmin(np.abs(z1 - cdf_z1))
        ind2 = np.argmin(np.abs(z2 - cdf_z2))
        ind3 = np.argmin(np.abs(rho - cdf_rho))
        eibv += cdf_table[ind1][ind2][ind3]
    return eibv

# s1, calculate the EIBV and IVR field.
eibv_field = np.zeros([grid.shape[0]])
ivr_field = np.zeros([grid.shape[0]])
for i in tqdm(range(grid.shape[0])):
    SF = cov[:, i].reshape(-1, 1)
    MD = 1 / (cov[i, i] + nugget)
    VR = SF @ SF.T * MD
    SP = cov - VR
    sigma_diag = np.diag(SP).reshape(-1, 1)
    vr_diag = np.diag(VR).reshape(-1, 1)
    eibv_field[i] = get_eibv_analytical_fast(mu=mu_truth, sigma_diag=sigma_diag, vr_diag=vr_diag,
                                             threshold=threshold, cdf_z1=cdf_z1, cdf_z2=cdf_z2,
                                             cdf_rho=cdf_rho, cdf_table=cdf_table)
    ivr_field[i] = np.sum(np.diag(VR))

# s2, normalize the EIBV and IVR field.
eibv_field = normalize(eibv_field)
ivr_field = 1 - normalize(ivr_field)

#%%
# s3, plot the EIBV and IVR field.
fig = plt.figure(figsize=(18, 5))
gs = GridSpec(nrows=1, ncols=3)
ax = fig.add_subplot(gs[0])
plotf(grid, mu_truth, "BrBG", "Mean", vmin=0, vmax=1)

ax = fig.add_subplot(gs[1])
plotf(grid, eibv_field, "YlGn", "EIBV", vmin=0, vmax=1)

ax = fig.add_subplot(gs[2])
plotf(grid, ivr_field, "YlGn", "IVR", vmin=0, vmax=1)

plt.tight_layout()
plt.savefig(figpath + "gp_ei.png")
# plt.show()
plt.close("all")

