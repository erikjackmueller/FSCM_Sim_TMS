from functions import*
import numpy as np
import time
import matplotlib
matplotlib.use("TkAgg")
import multiprocessing.managers

# path = "Sphere_642"
# sizes = [642, 1280]
path = "Sphere_10242"
sizes = [10242, 1280]

class MyManager(multiprocessing.managers.BaseManager):
    pass
MyManager.register('np_zeros', np.zeros, multiprocessing.managers.ArrayProxy)
man = MyManager()

if __name__ == '__main__':

    man.start()
    n = 100
    r_max = 0.93
    r = np.linspace(0.61, r_max, n)
    theta = np.linspace(0, np.pi, n)
    phi = (1/2)*np.pi
    # r0 = np.array([0, 1.05, 0])
    r0 = 1.1*np.array([0, 1, 0])
    m = np.array([-1, 0, 0])

    start = time.time()
    time_0 = start
    res1 = reciprocity_three_D(r, theta, r0_v=r0, m=m, phi=phi)
    end = time.time()
    print(f"{end - start:.2f}s receprocity")

    start = time.time()
    tc, areas, tri_points = read_sphere_mesh_from_txt_locations_only(sizes, path)
    # functions.triangulateSphere(20)
    end = time.time()
    print(f"{end - start:.2f}s triangulation")

    start = time.time()
    Q, rs = SCSM_tri_sphere_numba(tc, tri_points, areas, r0=r0, m=m)
    end = time.time()
    print(f"{end - start:.2f}s  Q calculation")

    start = time.time()

    res_fine = parallel_SCSM_E_sphere(man, Q, rs, r, theta, r0=r0, m=m, phi=phi, near_field=True,
                                      tri_points=tri_points, near_radius=0.5)
    res = SCSM_E_sphere_numba_polar(Q, rs, r, theta, r0=r0, m=m)
    end = time.time()
    print(f"{(end - start)/60:.2f}minutes E calculation")
    time_last = end
    print(f"{(time_last - time_0) / 60:.2f}minutes complete simulation")

    res2 = res.copy()

    diff = np.abs(res1 - res2)
    dif2 = np.abs(res1 - res_fine)
    relative_diff = diff / np.linalg.norm(res1)
    #
    rerror_imag = np.linalg.norm(diff) * 100 / np.linalg.norm(res1)
    rerror_fine = np.linalg.norm(dif2) * 100 / np.linalg.norm(res1)

    print(f"relative error with near field: {rerror_fine:.7f}%")
    print(f"relative error no near field: {rerror_imag:.7f}%")

    plot_E_diff(res1, res_fine, r, theta, r_max, r0, m)
    plot_E_diff(res1, res2, r, theta, r_max, r0, m)
    plot_E_diff(res2, res_fine, r, theta, r_max, r0, m)

