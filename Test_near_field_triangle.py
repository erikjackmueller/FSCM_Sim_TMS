import numpy as np
import sys
import matplotlib.pyplot as plt
plt.rcParams.update({"text.usetex": True})
import os
import pathlib
project_path = pathlib.Path('C:\\Users\\emueller\\PycharmProjects\\TMS_Sim_MA')
sys.path.append(project_path)
from functions import*

# triangle points
p1 = np.array([-1., -0.333, 0.])
p2 = np.array([1., -0.333, 0.])
p3 = np.array([0., 0.667, 0.])
ael = np.mean([np.linalg.norm(p1-p2), np.linalg.norm(p1-p3), np.linalg.norm(p3-p2)])
print(f'average edge length = {ael}')
z1 = 0.5
z2 = 1.0
z3 = 3.0
z4 = 5.0

# create line along x-axis with z_value as distance above the triangle
xs = np.arange(-1., 1, 0.01) # 200 values
ys = np.zeros_like(xs)
zs_1 = z1*np.ones_like(xs)
zs_2 = z2*np.ones_like(xs)
zs_3 = z3*np.ones_like(xs)
zs_4 = z4*np.ones_like(xs)

rs_1 = np.array([xs, ys, zs_1])
rs_2 = np.array([xs, ys, zs_2])
rs_3 = np.array([xs, ys, zs_3])
rs_4 = np.array([xs, ys, zs_4])

E_ana_1 = np.zeros_like(xs)
E_pp_1 = np.zeros_like(xs)
E_ana_2 = np.zeros_like(xs)
E_pp_2 = np.zeros_like(xs)
E_ana_3 = np.zeros_like(xs)
E_pp_3 = np.zeros_like(xs)
E_ana_4 = np.zeros_like(xs)
E_pp_4 = np.zeros_like(xs)

Q = 1e-15
eps0 = 8.854187812813e-12
for i in range(xs.shape[0]):
    E_ana_1[i] = np.linalg.norm(E_near(Q, p1, p2, p3, rs_1[:, i]))
    E_ana_2[i] = np.linalg.norm(E_near(Q, p1, p2, p3, rs_2[:, i]))
    E_ana_3[i] = np.linalg.norm(E_near(Q, p1, p2, p3, rs_3[:, i]))
    E_ana_4[i] = np.linalg.norm(E_near(Q, p1, p2, p3, rs_4[:, i]))
    E_pp_1[i] = np.linalg.norm((Q / (eps0 * 4 * np.pi) / np.linalg.norm(rs_1[:, i]) ** 3) * rs_1[:, i])
    E_pp_2[i] = np.linalg.norm((Q / (eps0 * 4 * np.pi) / np.linalg.norm(rs_2[:, i]) ** 3) * rs_2[:, i])
    E_pp_3[i] = np.linalg.norm((Q / (eps0 * 4 * np.pi) / np.linalg.norm(rs_3[:, i]) ** 3) * rs_3[:, i])
    E_pp_4[i] = np.linalg.norm((Q / (eps0 * 4 * np.pi) / np.linalg.norm(rs_4[:, i]) ** 3) * rs_4[:, i])

fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
ax[0, 0].plot(xs, E_ana_1)
ax[0, 0].plot(xs, E_pp_1)
ax[0, 0].set_xlabel(r'$x$')
ax[0, 0].set_ylabel(r'$|E|$')
ax[0, 0].set_title(f'$|E|$ at $z={str(z1)}$')
ax[0, 0].legend([r'$|E|_{ana}$', '$|E|_{pp}$'])
ax[0, 0].grid(True)

ax[0, 1].plot(xs, E_ana_2)
ax[0, 1].plot(xs, E_pp_2)
ax[0, 1].set_xlabel(r'$x$')
ax[0, 1].set_ylabel(r'$|E|$')
ax[0, 1].set_title(f'$|E|$ at $z={str(z2)}$')
ax[0, 1].legend([r'$|E|_{ana}$', '$|E|_{pp}$'])
ax[0, 1].grid(True)

ax[1, 0].plot(xs, E_ana_3)
ax[1, 0].plot(xs, E_pp_3)
ax[1, 0].set_xlabel(r'$x$')
ax[1, 0].set_ylabel(r'$|E|$')
ax[1, 0].set_title(f'$|E|$ at $z={str(z3)}$')
ax[1, 0].legend([r'$|E|_{ana}$', '$|E|_{pp}$'])
ax[1, 0].grid(True)

ax[1, 1].plot(xs, E_ana_4)
ax[1, 1].plot(xs, E_pp_4)
ax[1, 1].set_xlabel(r'$x$')
ax[1, 1].set_ylabel(r'$|E|$')
ax[1, 1].set_title(f'$|E|$ at $z={str(z4)}$')
ax[1, 1].legend([r'$|E|_{ana}$', '$|E|_{pp}$'])
ax[1, 1].grid(True)

fig.suptitle(f'Comparison of electric fields above \n a triangle in $x, y$-plane with average edge length ${ael:.2f}$', fontsize=14)

plt.show()