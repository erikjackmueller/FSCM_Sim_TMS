from functions import*
import numpy as np
import time
import matplotlib
import datetime
from pathlib import Path
import os
matplotlib.use("TkAgg")

file_path = "head_models/MPI_head_01/"
# file_path = os.path.realpath(Path("C:/Users/User/Downloads"))
fn = os.path.join(file_path, "15484.08.hdf5")
fn2 = os.path.join(file_path, "e.hdf5")
fn3 = "MagVenture_MCF_B65_REF_highres.ccd"

start = time.time()
print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
# initialize values
n = 100
scaling_factor = 1
r = 0.9 * scaling_factor
phi1 = np.linspace(0, np.pi, n)
theta1 = np.linspace(0, 2 * np.pi, n)
phi2, theta2 = np.meshgrid(phi1, theta1)
phi, theta = phi2.T, theta2.T
xyz_grid = xyz_grid(r, phi, theta)
omega = 18.85956e3

# read data for transformation matrix and sigmas
transformation_matrix, sigmas = read_mesh_from_hdf5(fn2, mode="coil")
# read coil data
m, m_pos = read_from_ccd(file_path)
# moving the coil position according to the FEM calculation values
m_pos = translate_old(m_pos, transformation_matrix)
# m_pos = translate(m_pos, transformation_matrix)
# np.savetxt("coil_new.csv", m_pos, delimiter=",")

# creating a shrinking and moving matrix for the small sphere
trans_mat1 = np.eye(4)
trans_mat1[0, 0] = 1/200
trans_mat1[1, 1] = 1/200
trans_mat1[2, 2] = 1/200
trans_mat2 = np.eye(4)
trans_mat2[3, 0] = 0.35
trans_mat2[3, 1] = -0.35
trans_mat2[3, 2] = 0.90
trans_mat = trans_mat1 @ trans_mat2
# transforming the coil for the sphere
m_pos1 = translate(m_pos, trans_mat)


# m_pos1 = m_pos1[:50]
# m = m[:50]
r_target = sphere_to_carthesian(r=r, phi=phi.flatten(), theta=theta.flatten())
res1_vector_flat, res1_flat = reciprocity_surface(rs=r_target, r0_v=m_pos1, m=m, omega=omega)
res1_vector = array_unflatten3d(res1_vector_flat, n_rows=n)
res1 = Norm_x_y(res1_vector, n)
res12 = array_unflatten(res1_flat, n_rows=n)

#create sphere mesh
tc, areas, tri_points, n_v = sphere_mesh(2000, scaling=scaling_factor)[:4]
# array_3d_plot(tc, m_pos1)

print(f"elements: {tc.shape[0]}")
n_elem = tc.shape[0]
rs = tc
end = time.time()
t = t_format(end - start)
print(f"{t[0]:.2f}" + t[1] + " set-up + analytical solution")
start = time.time()
start_sub = start
# b_im = jacobi_vectors_cupy(rs=tc, n=n_v, m=m, m_pos=m_pos1, omega=omega)
b_im = jacobi_vectors_numpy(rs=tc, n=n_v, m=m, m_pos=m_pos1, omega=omega)
end_sub = time.time()
t_sub = t_format(end_sub - start_sub)
print(f"{t_sub[0]:.2f}" + t_sub[1] + "  b calculation")
Q = SCSM_jacobi_iter_cupy(tc, areas, n_v, b_im, sig_in=0.33, sig_out=0.0, tol=5e-13, n_iter=40,
                          omega=omega)
end = time.time()
t = t_format(end - start)
print(f"{t[0]:.2f}" + t[1] + "  Q jacobi")
start = time.time()

b_im_ = vector_potential_for_E(r_target, m=m, m_pos=m_pos1, omega=omega)
res_flat = SCSM_FMM_E2(Q=Q, r_source=rs, r_target=r_target, eps=1e-20, b_im=b_im_)
res = array_unflatten(res_flat, n_rows=n)
print(f"error: {nrmse(res1, res)*100:.2f}%")
# plot_E_sphere_surf(res, phi, theta, r, c_map=cm.jet)
plot_E_sphere_surf_diff(res1, res, xyz_grid=xyz_grid, c_map=cm.jet, title=False)

print("---------------------------")
# print(np.linalg.norm(res - res1))







