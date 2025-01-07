from functions import*
import numpy as np
import time
import matplotlib
# matplotlib.use("TkAgg")

from matplotlib import cm
import tracemalloc
import datetime



small_samples = True

print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

n = 100
phi1 = np.linspace(0, np.pi, n)
theta1 = np.linspace(0, 2 * np.pi, n)
phi2, theta2 = np.meshgrid(phi1, theta1)
phi, theta = phi2.T, theta2.T
scaling_factor = 1
r = 0.9
xyz_grid = xyz_grid(r, phi, theta)
# define dipole
direction = np.array([0, 0, 1])
d_norm = direction/np.linalg.norm(direction)
r0 = 1.05 * d_norm
m = np.array([0, 0, 1])
omega = 18.85956e3
r_target = sphere_to_carthesian(r=r, phi=phi.flatten(), theta=theta.flatten())

start = time.time()

# calculate analytic solution
res1 = reciprocity_sphere(grid=xyz_grid, r0_v=r0, m=m, omega=omega)[1]
end = time.time()
t = t_format(end - start)
print(f"{t[0]:.2f}" + t[1] + " receprocity")
start = time.time()


if small_samples:
    sample_list_small = [2000, 4000, 6000, 8000, 10000]
    length = len(sample_list_small)
    errors = np.zeros((4, length))
    memories = np.zeros_like(errors)
    times = np.zeros_like(errors)
    elements = np.zeros(length)
    # initialize all methods for JIT reset
    tc, areas, tri_points, n_v, avg_len = read_sphere_mesh_from_txt(sizes=None, path="spheres/", n=2000)
    b_im = jacobi_vectors_numpy(tc, n_v, r0, m, omega=omega)
    # A0 = v_FSCM_matrix_slim(tri_centers=tc, areas=areas, n=n_v)
    # Q1 = SCSM_tri_sphere_numba(tc, tri_points, areas, r0=r0, m=m, omega=omega)[0]
    # A1 = FSCM_matrix_point_to_point(tri_centers=tc, areas=areas, n=n_v, idxs=get_matrix_idxs(n_elements))
    # Q2 = SCSM_jacobi_iter_numpy(tc, areas, n_v, b_im, tol=0.1)
    Q3 = q_jac_vec(tc, areas, n_v, b_im, tol=0.1)
    Q4 = q_jac_cu(tc, areas, n_v, b_im, tol=0.1)
    b = vector_potential_for_E_single_m(rs=r_target, m=m, m_pos=r0, omega=omega)
    res0 = SCSM_FMM_E2(Q=Q3, r_source=tc, r_target=r_target, eps=1e-15, b_im=b)
    print("--------------------------------------------------")
    print("finished initialization")
    print("--------------------------------------------------")
    for i_samples, samples in enumerate(sample_list_small):
        #tc, areas, tri_points, n_v, avg_len = sphere_mesh(samples)
        tc, areas, tri_points, n_v, avg_len = read_sphere_mesh_from_txt(sizes=None, path="spheres/", n=samples)
        n_elements = tc.shape[0]
        print(f"average length: {avg_len:.5f}")
        end = time.time()
        t = t_format(end - start)
        print(f"{t[0]:.2f}" + t[1] + " triangulation and initialization")
        print(f"elements: {n_elements}")
        print("--------------------------------------------------")
        elements[i_samples] = n_elements
        for method in [0, 1, 2, 3]:
            start = time.time()
            time_0 = start
            tracemalloc.start()
            print("--------------------------------------------------")
            b_im = jacobi_vectors_numpy(tc, n_v, r0, m, omega=omega)
            # mask_E_near = get_near_field_mask(tc, avg_len)
            # E_nears = dot_fun1(E_near_vec(Q=1, triangle_points=tri_points[mask_E_near[1]], c=tc[mask_E_near[1]],
            #                               A=areas[mask_E_near[1]], r=tc[mask_E_near[0]], n=-n_v[mask_E_near[1]]),
            #                    n_v[mask_E_near[0]])
            if method == 0:
                print(f"matrix, number of samples: {samples}")
                A = v_FSCM_matrix_slim(tri_centers=tc, areas=areas, n=n_v)
                Q = np.linalg.solve(A, b_im)
            elif method == 1:
                print(f"matrix gpu, number of samples: {samples}")
                dtype='float32'
                if samples == 10000:
                    dtype = 'float16'
                A = FSCM_matrix_gpu(tri_centers=tc, areas=areas, n=n_v, dtype=dtype)
                Q = np.linalg.solve(A, b_im)
            elif method == 2:
                print(f"vectorized jacobi, number of samples: {samples}")
                # Q = SCSM_jacobi_iter_numpy(tc, areas, n_v, b_im, tol=1e-3, n_iter=5)
                Q = q_jac_vec(tc, areas, n_v, b_im, tol=1e-3)
            elif method == 3:
                print(f"cupy-jacobi, number of samples: {samples}")
                Q = q_jac_cu(tc, areas, n_v, b_im, tol=1e-3)
            #
            # elif method == 5:
            #     print(f"vector matrix, number of samples: {samples}")
            #     A = v_FSCM_matrix_slim(tri_centers=tc, areas=areas, n=n_v)
            #     Q = np.linalg.solve(A, b_im)
            # Q = SCSM_matrix(tc, areas, n=n_v, b_im=b_im, omega=omega)
            mem = tracemalloc.get_traced_memory()[1] / (1024**2)
            tracemalloc.stop()
            print(f"Memory peak: {mem:.4f}MB")
            rs = tc
            end = time.time()
            t = t_format(end - start)
            print(f"{t[0]:.2f}" + t[1] + " Q calc")

            start = time.time()

            b_im_ = vector_potential_for_E_single_m(rs=r_target, m=m, m_pos=r0, omega=omega)
            res_flat = SCSM_FMM_E2(Q=Q, r_source=tc, r_target=r_target, eps=1e-15, b_im=b_im_)
            res = array_unflatten(res_flat, n_rows=n)

            end = time.time()
            t = t_format(end - start)
            print(f"{t[0]:.2f}" + t[1] + "  E calculation")

            time_last = end
            t_full = time_last - time_0
            t = t_format(end - time_0)
            print(f"{t[0]:.2f}" + t[1] + "  complete simulation")


            #
            nrmse_val = nrmse(res1, res) * 100
            print(f"nrmse: {nrmse_val:.7f}%")
            # plot_E_sphere_surf_diff(res1, res, xyz_grid=xyz_grid, c_map=cm.jet)
            errors[method, i_samples] = nrmse_val
            times[method, i_samples] = t_full
            memories[method, i_samples] = mem

    np.savetxt("Q_benchmark_error_with_gpu_1", errors, delimiter=",")
    np.savetxt("Q_benchmark_memory_with_gpu_1", memories, delimiter=",")
    np.savetxt("Q_benchmark_time_with_gpu_1", times, delimiter=",")
    np.savetxt("Q_benchmark_elements_with_gpu_1", elements, delimiter=",")
    print(f"saved results of elements {elements, memories, times}")
else:
    sample_list_large = [10000, 20000, 40000, 100000, 200000]
    length = len(sample_list_large)
    errors = np.zeros((2, length))
    memories = np.zeros_like(errors)
    times = np.zeros_like(errors)
    elements = np.zeros(length)
    for i_samples, samples in enumerate(sample_list_large): #
        # tc, areas, tri_points, n_v, avg_len = sphere_mesh(samples)
        print("--------------------------------------------------")
        print("--------------------------------------------------")
        tc, areas, tri_points, n_v, avg_len = read_sphere_mesh_from_txt(sizes=None, path="spheres/", n=samples)
        # initialize all methods for JIT reset
        if i_samples == 0:
            b_im = jacobi_vectors_numpy(tc, n_v, r0, m, omega=omega)
            Q3 = q_jac_vec(tc, areas, n_v, b_im, tol=0.1)
            Q4 = q_jac_cu(tc, areas, n_v, b_im, tol=0.1)
        print(f"average length: {avg_len:.5f}")
        end = time.time()
        t = t_format(end - start)
        print(f"{t[0]:.2f}" + t[1] + " triangulation")
        n_elements = tc.shape[0]
        print(f"elements: {n_elements}")
        elements[i_samples] = n_elements
        for method in [0, 1]:
            time_0 = start
            tracemalloc.start()
            start = time.time()
            print("--------------------------------------------------")
            mask_E_near = get_near_field_mask(tc, avg_len)
            E_nears = dot_fun1(E_near_vec(Q=1, triangle_points=tri_points[mask_E_near[1]], c=tc[mask_E_near[1]],
                                          A=areas[mask_E_near[1]], r=tc[mask_E_near[0]], n=-n_v[mask_E_near[1]]),
                               n_v[mask_E_near[0]])
            if method == 0:
                b_im = jacobi_vectors_numpy(tc, n_v, r0, m, omega=omega)
                Q = q_jac_vec(tc, areas, n_v, b_im, tol=1e-3)
                print(f"vec-jacobi, number of samples: {samples}")
            elif method == 1:
                b_im = jacobi_vectors_cupy(tc, n_v, m, omega=omega, r0=r0)
                Q = q_jac_cu(tc, areas, n_v, b_im, tol=1e-3)
                print(f"cupy-jacobi, number of samples: {samples}")
            # Q = SCSM_matrix(tc, areas, n=n_v, b_im=b_im, omega=omega)
            mem = tracemalloc.get_traced_memory()[1] / (1024 ** 2)
            tracemalloc.stop()
            print(f"Memory peak: {mem:.4f}MB")
            rs = tc
            end = time.time()
            t = t_format(end - start)
            print(f"{t[0]:.2f}" + t[1] + " Q calc")

            start = time.time()

            b_im_ = vector_potential_for_E_single_m(rs=r_target, m=m, m_pos=r0, omega=omega)
            res_flat = SCSM_FMM_E2(Q=Q, r_source=tc, r_target=r_target, eps=1e-15, b_im=b_im_)
            res = array_unflatten(res_flat, n_rows=n)

            end = time.time()
            t = t_format(end - start)
            print(f"{t[0]:.2f}" + t[1] + "  E calculation")

            time_last = end
            t_full = time_last - time_0
            t = t_format(end - time_0)
            print(f"{t[0]:.2f}" + t[1] + "  complete simulation")

            #
            nrmse_val = nrmse(res1, res) * 100
            print(f"nrmse: {nrmse_val:.7f}%")
            # plot_E_sphere_surf_diff(res1, res, xyz_grid=xyz_grid, c_map=cm.jet)
            errors[method, i_samples] = nrmse_val
            times[method, i_samples] = t_full
            memories[method, i_samples] = mem

    np.savetxt("Q_benchmark_large_samples_error", errors, delimiter=",")
    np.savetxt("Q_benchmark_large_samples_memory", memories, delimiter=",")
    np.savetxt("Q_benchmark_large_samples_time", times, delimiter=",")
    np.savetxt("Q_benchmark_large_samples_elements", elements, delimiter=",")
    print(f"saved results of elements {elements}")




