from functions import*
import numpy as np
import time
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import cm

path = "Sphere_2964"
sizes = [2964, 5924]
n = 101
phi1 = np.linspace(0, np.pi, n)
theta1 = np.linspace(0, 2 * np.pi, n)
phi2, theta2 = np.meshgrid(phi1, theta1)
phi, theta = phi2.T, theta2.T
scaling_factor = 1
r = 0.9
direction = np.array([0, 0, 1])
d_norm = direction/np.linalg.norm(direction)
r0 = 1.05 * d_norm
m = np.array([0, 0, 1])
omega = 18.85956e3

xyz_grid = xyz_grid(r, phi, theta)
r_target = sphere_to_carthesian(r=r, phi=phi.flatten(), theta=theta.flatten())
start = time.time()
time_0 = start
res1 = reciprocity_three_D(r, theta, r0_v=r0, m=m, phi=phi, omega=omega, projection='sphere_surface')
end = time.time()
t = t_format(end - start)
print(f"{t[0]:.2f}" + t[1] + " receprocity")


start = time.time()
tc, areas, tri_points, n_v, avg_len = read_sphere_mesh_from_txt(sizes=None, path="spheres/", n=3500)
print(f"average length: {avg_len}")
end = time.time()
t = t_format(end - start)
print(f"{t[0]:.2f}" + t[1] + " triangulation")
n_elements = tc.shape[0]
print(f"elements: {n_elements}")

start = time.time()
b_im = jacobi_vectors_numpy(tc, n_v, r0, m, omega=omega)
mask_E_near = get_near_field_mask(tc, avg_len)
A_pp = v_FSCM_matrix_slim(tri_centers=tc, areas=areas, n=n_v)
A_ana = SCSM_matrix_analytical(A_pp, tri_centers=tc, tri_points=tri_points, areas=areas, n=n_v, idxs=mask_E_near)
Q = np.linalg.solve(A_ana, b_im)
Q = SCSM_jacobi_iter_vec_numpy(tc, areas, n_v, b_im, tol=1e-3, verbose=True, with_E_near=True,
                               E_near_mask=get_near_field_mask(triangle_centers=tc, avg_length=avg_len),
                               tri_points=tri_points)

rs = tc
end = time.time()
t = t_format(end - start)
print(f"{t[0]:.2f}" + t[1] + " Q jacobi")
b_im_ = vector_potential_for_E_single_m_numpy(rs=r_target, m=m, m_pos=r0, omega=omega)
start = time.time()

near_field_mask = get_near_field_mask(triangle_centers=tc, avg_length=avg_len, r_target=r_target)
res_flat = SCSM_FMM_E2(Q=Q, r_source=tc, r_target=r_target, eps=1e-15, b_im=b_im_, with_near_field=True,
                       near_mask=near_field_mask, tri_points=tri_points, n=n_v, areas=areas)
res = array_unflatten(res_flat, n_rows=n)

end = time.time()
t = t_format(end - start)
print(f"{t[0]:.2f}" + t[1] + "  E calculation")

time_last = end
t = t_format(end - time_0)
print(f"{t[0]:.2f}" + t[1] + "  complete simulation")
res2 = res.copy()
diff = np.abs(res1 - res2)
relative_diff = diff / np.linalg.norm(res1)
max_analytic = np.max(res1)
min_analytic = np.min(res1)
max_numeric = np.max(res)
min_numeric = np.min(res)

print(f"max_analytic = {max_analytic}, min_analytic = {min_analytic}, max_numeric = {max_numeric}, min_numeric = {min_numeric}")
error_imag = nrmse(res1, res2) * 100
print(f"relative error: {error_imag:.7f}%")

plot_E_sphere_surf_diff(res1, res2, xyz_grid=xyz_grid, c_map=cm.bwr, title=False, plot_difference=False)
