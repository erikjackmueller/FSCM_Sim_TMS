from functions import*
import numpy as np
import time
import matplotlib
import datetime
from cupyx import optimizing
matplotlib.use("TkAgg")
import multiprocessing.managers

# class MyManager(multiprocessing.managers.BaseManager):
#     pass
# MyManager.register('np_zeros', np.zeros, multiprocessing.managers.ArrayProxy)
# man = MyManager()
n = 100
scaling_factor = 1
r = 0.85 * scaling_factor
phi1 = np.linspace(0, np.pi, n)
theta1 = np.linspace(0, 2 * np.pi, n)
phi2, theta2 = np.meshgrid(phi1, theta1)
phi, theta = phi2.T, theta2.T
direction = np.array([0, 0, 1])
d_norm = direction/np.linalg.norm(direction)
r0 = 1.05 * d_norm * scaling_factor
omega = 19e3
# m = d_norm
m = np.array([1, 0, 0])
r_target = sphere_to_carthesian(r=r, phi=phi.flatten(), theta=theta.flatten())

# samples = [1000, 5000, 20000, 50000, 100000, 250000, 850000]
samples = [20000]
t_numpy = []
t_jacobi = []
errors = []
compare = False

res1 = reciprocity_three_D(r, theta, r0_v=r0, m=m, phi=phi, projection="sphere_surface", omega=omega)

# if __name__ == "__main__":
for i in range(len(samples)):

        tc, areas, tri_points, n_v = sphere_mesh(samples[i], scaling=scaling_factor)[:4]

        ig = False
        # print("--------no initial guess---------")
        print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        print(f"elements: {tc.shape[0]}")
        n_elem = tc.shape[0]

        if compare:
            start = time.time()
            Q, rs = SCSM_tri_sphere_numba(tc, tri_points, areas, r0=r0, m=m)
            end = time.time()
            t = t_format(end - start)
            t_numpy.append(t)
            print(f"{t[0]:.2f}" + t[1] + "  Q linalg.solve()")
        else:
            rs = tc
        # elif j == 1:
        #     ig = True
        #     print("--------with initial guess---------")

        start = time.time()
        start_sub = start
        b_im = jacobi_vectors_numpy(tc, n_v, r0, m, omega=omega)
        end_sub = time.time()
        t_sub = t_format(end_sub - start_sub)
        print(f"{t_sub[0]:.2f}" + t_sub[1] + "  b calculation")
        # Q = SCSM_tri_sphere_numba(tc, tri_points, areas, r0=r0, m=m, omega=omega)[0]
        Q = SCSM_jacobi_iter_cupy(tc, areas, n_v, b_im, tol=1e-15, n_iter=1000, omega=omega, high_precision=True)
        # print(f"{Q[:10]}")
        # Q = SCSM_jacobi_iter_debug(tc, areas, n=n_v, r0=r0, m=m, tol=1e-10, initial_guess=ig, n_iter=20,
        #                      b_im=b_im)

        # jac = cp.vectorize(SCSM_jacobi_iter_cupy_test)
        # Q = jac(cp.asarray(tc), cp.asarray(areas), cp.asarray(n_v), cp.asarray(b_im), n_elem,
        #         cp.zeros(n_elem, cp.complex_), cp.zeros(n_elem, cp.complex_), 0.33, 1, 20, 1e-10)
        # if i == 0:
        #     man.start()
        # Q = SCSM_jacobi_iter_cupy_test(man, cp.asarray(tc), cp.asarray(areas), cp.asarray(n_v),
        #                                cp.asarray(b_im), n_elem, 0.33, 1, 20, 1e-10)
        # with optimizing.optimize():
        #     Q = SCSM_jacobi_iter_cupy(tc, areas, n_v, b_im, tol=1e-10, n_iter=20)
        # jac_cupy_vec = cp.vectorize(SCSM_jacobi_iter_cupy)
        # Q = jac_cupy_vec(tc, areas, n_v, b_im, tol=1e-10, n_iter=20)
        # Q = Q_jacobi_numba_cuda(tc, areas, n_v, b_im, tol=1e-10, n_iter=20)
        end = time.time()
        t = t_format(end - start)
        print(f"{t[0]:.2f}" + t[1] + "  Q calculation")
        t_jacobi.append(t)
        start = time.time()

        res_flat = SCSM_FMM_E(Q=Q, r_source=rs, r_target=r_target, eps=1e-2, m=m, r0=r0, omega=omega)
        res = array_unflatten(res_flat, n_rows=n)
        res2 = res.copy()
        diff = np.abs(res1 - res2)
        relative_diff = diff / np.linalg.norm(res1)
        rerror_imag = np.linalg.norm(diff) * 100 / np.linalg.norm(res1)
        print(f"relative error: {rerror_imag:.7f}%")
        errors.append(rerror_imag)
        print("---------------------------")







