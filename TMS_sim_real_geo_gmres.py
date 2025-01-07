import functions
import numpy as np
import time
import os
import matplotlib
# matplotlib.use("TkAgg")
import multiprocessing.managers
from pathlib import Path
import datetime
from functions import*

# path_string_original = "C:\Users\ermu8317\Downloads" #cannot be handled by python
# path_string = path_string_original.replace("\\", "/")
path = "head_models/MPI_head_01/"
# path = os.path.realpath(Path("C:/Users/User/Downloads"))
fn = os.path.join(path, "15484.08.hdf5")
fn2 = os.path.join(path, "e.hdf5")
fn3 = "MagVenture_MCF_B65_REF_highres.ccd"

print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
start = time.time()
time_0 = start
tc, areas, tri_points, n_v, tissue_types = read_mesh_from_hdf5(fn)
r_targets = read_mesh_from_hdf5(fn, mode="target")
transformation_matrix, sigmas = read_mesh_from_hdf5(fn2, mode="coil")
print(f"# of elements: {tc.shape[0]}")
sigma_out = np.zeros(tc.shape[0])
sigma_in = np.zeros_like(sigma_out)
tissue_number = np.max(tissue_types) - 1000
for i in range(tissue_number + 1):
    sigma_in[np.where(tissue_types == i + 1000)] = sigmas[i - 1]
    sigma_out[np.where(tissue_types == i + 1000)] = sigmas[i]
m, m_pos_raw = read_from_ccd(path)
m_pos = translate_old(m_pos_raw, transformation_matrix)
omega = 18.85956e3
end = time.time()
t = t_format(end - start)
print(f"{t[0]:.2f}" + t[1] + " triangulation")

start = time.time()
# b_im = jacobi_vectors_numpy(rs=tc, n=n_v, m=m, m_pos=m_pos, omega=omega)
b_im = jacobi_vectors_cupy(rs=tc, n=n_v, m=m, m_pos=m_pos, omega=omega)
end = time.time()
t = t_format(end - start)
print(f"{t[0]:.2f}" + t[1] + " b calculation")

start_q = time.time()
Q = GMRES_FSCM(tc, areas, n_v=n_v, b=b_im, sig_in=sigma_in, sig_out=sigma_out, n_inner=200, n_outer=20, tol=1e-5,
                    mem=8, dtype='float64', use_gpu=True, matrix=None)
end_q = time.time()
t = t_format(end_q - start_q)
print(f"{t[0]:.2f}" + t[1] + " Q calculation")
print(f"Q: {Q[0], Q[-1]}")

start = time.time()
b_im_ = vector_potential_for_E(rs=r_targets, m=m, m_pos=m_pos, omega=omega)
# b_im_ = vector_potential_for_E_numpy(rs=r_targets, m=m, m_pos=m_pos, omega=omega)
# b_im_ = vector_potential_for_E(rs=r_targets, n=n_v, m=m, m_pos=m_pos)
print(f"b for E: {b_im_[0], b_im_[-1]}")
res = SCSM_FMM_E2(Q=Q, r_source=tc, r_target=r_targets, eps=1e-5, b_im=b_im_)
print(f"E: {res[0], res[-1]}")
# res = SCSM_FMM_E(Q=Q, r_source=tc, r_target=r_targets, eps=1e-15, m=m, r0=r0)
with h5py.File('real_geo_gmres_inner_200_outer_20_tol_1em5.hdf5', 'w') as f:
    f.create_dataset('e', data=res)
end = time.time()
t = t_format(end - start)
print(f"{t[0]:.2f}" + t[1] + " E calculation")
time_last = end
t = t_format(time_last - time_0)
print(f"{t[0]:.2f}" + t[1] + " complete simulation")
print(f'max E = {np.max(res)}, should be 1.7982708')

# simnibs_path = os.path.realpath(Path("C:/Users/emueller/Downloads"))
# fn_simnibs_results = os.path.join(simnibs_path, "e.hdf5")
# # load reference e-fields from SimNIBS
# with h5py.File(fn_simnibs_results, "r") as f:
#     e_mag_simnibs_roi = f["data"]["midlayer"]["roi_surface"]["midlayer_m1s1pmd"]["E_mag"][:]
# print(f'nrmse to simnbs result: {nrmse(e_mag_simnibs_roi, res) * 100} %')