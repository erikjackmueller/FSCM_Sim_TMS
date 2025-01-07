from functions import*
import numpy as np
import time
import matplotlib
matplotlib.use("TkAgg")
import multiprocessing.managers



# functions.plot_default()
# path = os.path.realpath(Path("C:/Users/Besitzer/Downloads/Sphere_642"))
# path = "Sphere_10242"
# sizes = [10242, 1280]
# path = "Sphere_2964"
# sizes = [2964, 1280]
path = "Sphere_642"
sizes = [642, 1280]

class MyManager(multiprocessing.managers.BaseManager):
    pass
MyManager.register('np_zeros', np.zeros, multiprocessing.managers.ArrayProxy)
man = MyManager()
if __name__ == '__main__':
    # functions.plot_default()
    man.start()
    n = 400
    scaling_factor = 1
    r_max = 0.9 * scaling_factor
    r = np.linspace(0.01 * scaling_factor, r_max, n)
    theta = np.linspace(0, 2*np.pi, n)
    phi = (1/2)*np.pi
    # r0 = np.array([0, 1.05, 0])
    direction = np.array([1, 1, 0])
    d_norm = direction / np.linalg.norm(direction)
    r0 = 1.05 * d_norm * scaling_factor
    m = d_norm
    r_t, t_t = np.meshgrid(r, theta)
    r_target = circle_to_carthesian(r=r_t.flatten(), theta=t_t.flatten())

    start = time.time()
    time_0 = start
    res1 = reciprocity_three_D(r, theta, r0_v=r0, m=m, phi=phi, omega=1e3)
    end = time.time()
    t = t_format(end - start)
    print(f"{t[0]:.2f}" + t[1] + " receprocity")

    start = time.time()
    # tc, areas = functions.read_sphere_mesh_from_txt(sizes, path)
    tc, areas, tri_points = read_sphere_mesh_from_txt_locations_only(sizes, path, scaling=scaling_factor)
    tc, areas, tri_points, n_v, avg_len = sphere_mesh(1000, scaling=scaling_factor)
    end = time.time()
    t = t_format(end - start)
    print(f"{t[0]:.2f}" + t[1] + " triangulation")

    start = time.time()
    # Q, rs = functions.SCSM_tri_sphere(tc, areas, r0=r0, m=m, sig=1)
    b_im = jacobi_vectors_numpy(tc, n_v, r0, m)
    Q = SCSM_jacobi_iter_cupy(tc, areas, n_v, b_im, tol=1e-10, n_iter=20, omega=1e3)
    # Q, rs = SCSM_tri_sphere_numba(tc, tri_points, areas, r0=r0, m=m)
    end = time.time()
    t = t_format(end - start)
    print(f"{t[0]:.2f}" + t[1] + "  Q calculation")

    start = time.time()

    res_flat = SCSM_FMM_E(Q=Q, r_source=tc, r_target=r_target, eps=1e-2, m=m, r0=r0, omega=1e3)
    res = array_unflatten(res_flat, n_rows=n).T
    # res = functions.parallel_SCSM_E_sphere(man, Q, rs, r, theta, r0=r0, m=m, phi=phi)
    # res = functions.SCSM_E_sphere(Q, rs, r, theta, r0=r0, m=m)
    # res = functions.SCSM_E_sphere_numba_polar(Q, rs, r, theta, r0=r0, m=m)
    end = time.time()
    t = t_format(end - start)
    print(f"{t[0]:.2f}" + t[1] + " E calculation")
    time_last = end
    t = t_format(end - time_0)
    print(f"{t[0]:.2f}" + t[1] + " complete simulation")

    res2 = res.copy()

    diff = np.abs(res1 - res2)
    relative_diff = diff / np.linalg.norm(res1)

    rerror_imag = np.linalg.norm(diff) * 100 / np.linalg.norm(res1)

    print(f"relative error: {rerror_imag:.7f}%")
    # print(res1[0, 0])
    # functions.plot_E(res2, r, theta, r_max)
    # functions.plot_E(res2, r, theta, r_max)
    # functions.plot_E(diff_to_imag, r, theta, r_max)
    plot_E_diff(res1, res2, r, theta, r_max, r0, m)
    # functions.plot_E(relative_diff, r, theta, r_max)
    # plt.savefig("sample" + ".png")
    # functions.plot_E_diff(res2, res3, r, theta, r_max, r0, m)
    # functions.plot_E(res4, r, theta, r_max)
    # functions.plot_E(res1, r, theta, r_max)
    # functions.plot_E(diff, r, theta, r_max)





