from functions import*
import numpy as np
import time
import datetime
import scipy as sc
import sys
import pathlib
project_path = pathlib.Path('C:\\Users\\emueller\\PycharmProjects\\TMS_Sim_MA')
sys.path.append(project_path)

############# Init file and geo setup ###########################
gpu_support(False)
date = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
print(date)
res_folder = 'results/head_models/TUI_head_model/'
res_path = res_folder + date[:10] + '-TUI_head_03_matrix_test.hdf5'
res_path1 = res_folder + date[:10] + '-TUI_head_03_gmres_test.hdf5'
reference_path = res_folder + 'test_22_05_23_matrix_QE_ana.hdf5'


# dipole and frequency
m = np.array([0, 0, 1])
omega = 18.85956e3

# sigmas
sigma_brain = 0.33
sigma_skull = 0.01
sigma_scalp = 0.465
sigma_csf = 1.654
sigma_gm = 0.274
sigma_wm = 0.126
sigmas = np.array([sigma_wm, sigma_gm, sigma_csf, sigma_skull, sigma_scalp])

# init timing
start = time.time()
time_0 = start

file_names = ["01_Scalp_07mm_NE=8202", "02_Skull_04mm_NE=22226", "03_CSF_04mm_NE=14394",
              "04_GM_04mm_NE=15596",  "05_WM_03mm_NE=33066"]
deflated_scale = 1.0
tc, areas, tri_points, n_v, avg_len, sigmas_in, sigmas_out, roi_deflated = read_head_mesh_from_txt(
    path="head_models/TUI_head_03/", layer_names=file_names, sigmas=sigmas[:], file_format='stl', unit='mm',
    roi_deflated_smallest=True)

# geometry transform for COG model
COG = np.array([97.9839,  110.5508,  140.5215])
d = 10
Zmax = 154.1698
# mm to m transform
COG, d, Zmax = 0.001*COG, 0.001*d, 0.001*Zmax
# shift all by COG
tc, roi_deflated.triangle_centers, roi_deflated.locations = shift_points(
    [tc, roi_deflated.triangle_centers, roi_deflated.locations.T], COG)
roi_deflated.locations = roi_deflated.locations.T
tri_points = tri_points + np.repeat([COG], 3, axis=0)
r0 = COG + np.array([0, 0, Zmax + d])
r_target = roi_deflated.triangle_centers

print(f"average length: {avg_len}")
end = time.time()
t = t_format(end - start)
print(f"{t[0]:.2f}" + t[1] + " triangulation")
n_elements = tc.shape[0]
print(f"elements: {n_elements}")


############# Simulation ###########################
def head_simulation(time_0):


    start_q = time.time()
    b_im = jacobi_vectors_numpy(tc, n_v, r0, m, omega=omega)
    start = time.time()
    # calculate analytical solutions
    mask_E_near = get_near_field_mask(tc, avg_len, area_measure=np.sqrt(np.mean(areas)), memory=32)
    s_A_ana = sc.sparse.lil_array((tc.shape[0], tc.shape[0]), dtype=np.float32)
    s_A_ana[mask_E_near[0].tolist(), mask_E_near[1].tolist()] = build_sparse_mat_ana(tri_points, tc, areas, n_v,
                                                                                     mask_E_near)
    end = time.time()
    t = t_format(end - start)
    print(f"{t[0]:.2f}" + t[1] + " calculate E_ana")

    ##### matrix version for hpc high mem ######

    # A_pp = v_FSCM_matrix_slim(tri_centers=tc, areas=areas, n=n_v, sig_in=sigmas_in, sig_out=sigmas_out)
    # A_ana = FSCM_matrix_analytical(A_pp, tri_centers=tc, tri_points=tri_points, areas=areas, n=n_v, sig_in=sigmas_in,
    #                                sig_out=sigmas_out, idxs=mask_E_near)
    # Matrix_update(A_pp, s_A_ana, mask_E_near)
    # end = time.time()
    # t = t_format(end - start)
    # print(f"{t[0]:.2f}" + t[1] + " build matrix")
    # start = time.time()
    # Q = np.linalg.solve(A_ana, b_im)
    # end = time.time()
    # t = t_format(end - start)
    # print(f"{t[0]:.2f}" + t[1] + " solve for Q (LU)")

    start = time.time()
    s_A_pp = build_FSCM_matrix_sparse(tri_centers=tc, areas=areas, n=n_v, mask=mask_E_near)
    Q = GMRES_FSCM(tc, areas, n_v=n_v, b=b_im, sig_in=sigmas_in, sig_out=sigmas_out, n_inner=200, n_outer=10, tol=1e-4,
                    mem=32, dtype='float64', use_gpu=False, matrix=None, A_ana=s_A_ana, A_pp=s_A_pp)
    # print(f'nrmse between Q: LU and GMRES: {nrmse(Q, Q1) * 100:.2f}%')
    end = time.time()
    t = t_format(end - start)
    print(f"{t[0]:.2f}" + t[1] + " solve for Q (GMRES)")
    # Q = SCSM_jacobi_iter_vec_numpy(tc, areas, n_v, b_im, tol=0.1, n_iter=50, omega=omega, complex_values=False,
    #                                sig_in=sigmas_in, sig_out=sigmas_out, verbose=True, E_nears=E_nears,
    #                                E_near_mask=mask_E_near)
    # Q = SCSM_jacobi_iter_vec_numpy(tc, areas, n_v, b_im, tol=0.1, n_iter=500, omega=omega, complex_values=False,
    #                                sig_in=sigmas_in, sig_out=sigmas_out, verbose=True)
    end_q = time.time()
    t = t_format(end_q - start_q)
    # print(f'nrmse(Q_mat, Q_jac): {nrmse(Q, Q1) * 100:.2g}%')
    print(f"{t[0]:.2f}" + t[1] + " Q calculation")
    b_im_ = vector_potential_for_E_single_m_numpy(rs=r_target, m=m, m_pos=r0, omega=omega)
    start = time.time()
    near_field_mask = get_near_field_mask(triangle_centers=tc, avg_length=avg_len, r_target=r_target)
    res = SCSM_FMM_E2(Q=Q, r_source=tc, r_target=r_target, eps=1e-15, b_im=b_im_, with_near_field=True,
                       near_mask=near_field_mask, tri_points=tri_points, n=n_v, areas=areas)
    # res = SCSM_FMM_E2(Q=Q, r_source=tc, r_target=r_target, eps=1e-15, b_im=b_im_)
    # res1 = SCSM_FMM_E2(Q=Q1, r_source=tc, r_target=r_target, eps=1e-15, b_im=b_im_, with_near_field=True,
    #                    near_mask=near_field_mask, tri_points=tri_points, n=n_v, areas=areas)
    # print(f'nrmse between E: LU and GMRES: {nrmse(res, res1) * 100:.5f}%')

    print(f"max |E| = {np.max(res)}")
    end = time.time()
    t = t_format(end - start)
    print(f"{t[0]:.2f}" + t[1] + "  E calculation")

    time_last = end
    t = t_format(end - time_0)
    print(f"{t[0]:.2f}" + t[1] + "  complete simulation")


    # deflated roi on smallest region (wm in this case)
    save_results(res_path1, res, Q, r0, deflated_scale, m, file_names=file_names, roi=roi_deflated,
                 roi_type='deflated')
    # save_results(res_path, res1, Q1, r0, deflated_scale, m, file_names=file_names, roi=rectangle_roi,
    #              roi_type='rectangle')
    # compare_E(project_path, reference_path, res_path)
    # compare_E(project_path, reference_path, res_path1)

head_simulation(start)


with h5py.File(res_path, 'r') as f:
    res_read = np.array(f['e'])
    dip_loc = np.array(f["dipole_locations"])
    # read roi into roi_read dictionary
    roi_read = {}
    roi_dict = f["roi"]
    for k in roi_dict.keys():
        roi_read[k] = roi_dict[k][:]


trisurf_color_plot(res=res_read.flatten(), roi=roi_read, plot_cmap='bwr', show_dipoles=True, dipole_locations=dip_loc)

