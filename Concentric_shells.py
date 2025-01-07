from functions import*
import numpy as np
import time
import matplotlib
import datetime
from pathlib import Path
import os
matplotlib.use("TkAgg")

# set option of using realistic Coil model
Coil = False
if Coil:
    # file_path = os.path.realpath(Path("C:/Users/ermu8317/Downloads"))
    file_path = os.path.realpath(Path("C:/Users/User/Downloads"))
    fn = os.path.join(file_path, "15484.08.hdf5")
    fn2 = os.path.join(file_path, "e.hdf5")
    fn3 = "MagVenture_MCF_B65_REF_highres.ccd"

start = time.time()
print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
n = 100
scaling_factor = 1
r = 0.90 * scaling_factor
phi1 = np.linspace(0, np.pi, n)
theta1 = np.linspace(0, 2 * np.pi, n)
phi2, theta2 = np.meshgrid(phi1, theta1)
phi, theta = phi2.T, theta2.T
direction = np.array([0, 0, 1])
d_norm = direction/np.linalg.norm(direction)
r0 = 1.05 * np.array([0, 0, 1]) # * d_norm
r_target = sphere_to_carthesian(r=r, phi=phi.flatten(), theta=theta.flatten())
# m = np.array([0, 0, 1])
m = d_norm
omega = 18.85956e3
xyz_grid = xyz_grid(r, phi, theta)

if not Coil:
    # calculate analytic solution
    res1 = reciprocity_sphere(grid=xyz_grid, r0_v=r0, m=m, omega=omega)[1]
else:
    # read data for transformation matrix and sigmas
    transformation_matrix, sigmas = read_mesh_from_hdf5(fn2, mode="coil")
    # read coil data
    m, m_pos = read_from_ccd(file_path)
    # moving the coil position according to the FEM calculation values
    m_pos = translate(m_pos, transformation_matrix)
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

    res1_vector_flat, res1_flat = reciprocity_surface(rs=r_target, r0_v=m_pos1, m=m, omega=omega)
    res1_vector = array_unflatten3d(res1_vector_flat, n_rows=n)
    res1 = Norm_x_y(res1_vector, n)

# create concentric sphere meshes with different radii
radii = np.array([0.2, 0.4, 0.69, 0.7, 1.0])
sigmas = np.array([0.126, 0.275, 1.654, 0.001, 0.456])
tc, areas, tri_points, n_v, avg_lens, sigmas_in, sigmas_out = layered_sphere_mesh(n_samples=2000, sigmas=sigmas, radii=radii)
n_elem = tc.shape[0]
print(f"elements: {n_elem}")
print(f"average edge length: {avg_lens:.5f}")


# calculate solution using jacobi method
if not Coil:
    b_im = jacobi_vectors_numpy(tc, n=n_v, m=m, omega=omega, r0=r0)
else:
    b_im = jacobi_vectors_cupy(rs=tc, n=n_v, m=m, m_pos=m_pos1, omega=omega)
Q = SCSM_jacobi_iter_cupy(tc, areas, n_v, b_im, sig_in=sigmas_in, sig_out=sigmas_out, tol=7e-17, n_iter=5,
                         omega=omega, high_precision=True, verbose=True)

# tc1, areas1, tri_points1, n_v1, avg_length1 = sphere_mesh(samples=2000, scaling=scaling_factor)
# b_im1 = jacobi_vectors_numpy(tc1, n=n_v1, m=m, omega=omega, r0=r0)
# Q2 = SCSM_jacobi_iter_cupy(tc1, areas1, n_v1, b_im1, sig_in=0.33, sig_out=0.00, tol=7e-17, n_iter=5,
#                          omega=omega, high_precision=True)
end = time.time()
t = t_format(end - start)
print(f"{t[0]:.2f}" + t[1] + " jacobi  Q")
# calculate solution using the matrix version (LU)
start = time.time()
# Q1 = SCSM_matrix(tc, areas, n=n_v, sig_in=sigmas_in, sig_out=sigmas_out, b_im=b_im, omega=omega)
end = time.time()
t = t_format(end - start)
print(f"{t[0]:.2f}" + t[1] + " matrix  Q")
start = time.time()


# calculate Electric fields
if not Coil:
    b_im_ = vector_potential_for_E_single_m(rs=r_target, m=m, m_pos=r0, omega=omega)
else:
    b_im_ = vector_potential_for_E(r_target, m=m, m_pos=m_pos1, omega=omega)
res_flat = SCSM_FMM_E2(Q=Q, r_source=tc, r_target=r_target, eps=1e-15, b_im=b_im_)
res = array_unflatten(res_flat, n_rows=n)

nrmse = nrmse(res1, res) * 100
print(f"nrmse: {nrmse:.7f}%")
plot_E_sphere_surf_diff(res1, res, xyz_grid=xyz_grid, names=["analytic", "jacobi"], c_map=cm.jet, title=False)
print("---------------------------")







