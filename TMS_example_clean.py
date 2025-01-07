from TMS import TMS, head_mesh
import numpy as np
import time
import meshio
from functions import*


res_path = 'results/head_models/TUI_head_model/single_dipole_test.hdf5'
reference_path = 'results/head_models/TUI_head_model/test_22_05_23_matrix_QE_ana.hdf5'

# set up simulation for single dipole source and three layer head model
r0 = np.array([0.003334521, 0.02432317, 0.16170937])
m = np.array([0, 0, 1])
omega = 18.85956e3
sigma_brain = 0.33
sigma_skull = 0.01
sigma_scalp = 0.465

# set up parameters for incident electric field
params = dict()
params['m'] = np.array([0, 0, 1])
params['source_locations'] = np.array([0.003334521, 0.02432317, 0.16170937])
params['omega'] = 18.85956e3
# params['roi']

# select mesh model
file_names = ["scalp_009mm_NE=3800", "skull_008mm_NE=2780", "brain_008mm_NE=2266"]
# file_names = ["scalp_005mm_NE=13952", "skull_004mm_NE=9844", "brain_004mm_NE=7924"]

# start TMS mesh and TMS object
TMS_mesh = head_mesh(path="head_models/TUI_head_02/", file_names=file_names,
                 sigmas=np.array([sigma_brain, sigma_skull, sigma_scalp]),
                 file_format='stl', unit='mm')
TMS_mesh.read_mesh()
TMS_object = TMS(head_mesh=TMS_mesh, algorithm='matrix', params=params, res_path=res_path, save_results=True)
TMS_object.simulate()
error = compare_E('', reference_path, res_path)



