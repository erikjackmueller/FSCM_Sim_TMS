import numpy as np
from mpl_toolkits import mplot3d
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
import h5py
import numba
import math
import os
import time
from functions import*


class TMS:
    # idea TMS --> super class
    # head mesh --> class for TMS (TMS.head_mesh.read(), TMS.head_mesh.triangle_centers ...)
    # achieve this by class head_mesh, TMS(head_mesh=head_mesh)

    """
    docu here
    """
    def __init__(self, head_mesh=None, algorithm="matrix", params=None, res_path=None, save_results=False):
        self.head_mesh = head_mesh
        self.algorithm = algorithm
        self.params = dict(params)
        self.res_path = ''
        if 'm' in params.keys():
            self.m = params['m']
        if 'source_locations' in params.keys():
            self.r_source = params['source_locations']
        if 'omega' in params.keys():
            self.omega = params['omega']
        elif 'frequency' in params.keys():
            self.omega = 2 * np.pi * params['frequency']
        if res_path is not None:
            self.res_path = res_path
        self.save_results = save_results
        # self.connections = np.array(connections)
    @timer
    def simulate(self):

        # this is a place holder for any ROI function
        self.r_target = rectangle_points_xy(width=0.05, length=0.08, z=0.075, n=10, center=np.array([0, 0.01])).T
        # calculate Q which has a lot of versions actually
        self.b = jacobi_vectors_numpy(self.head_mesh.triangle_centers, self.head_mesh.n_v, self.r_source, self.m,
                                      self.omega)
        self.charge_matrix, d = v_FSCM_matrix_slim(tri_centers=self.head_mesh.triangle_centers,
                                                   areas=self.head_mesh.areas,
                                                   n=self.head_mesh.n_v, sig_in=self.head_mesh.sigmas_in,
                                                   sig_out=self.head_mesh.sigmas_out, d_out=True)
        self.x0 = self.b/d
        self.charges = GMRES_FSCM(matrix=self.charge_matrix, x0=self.x0, b=self.b, n_inner=150, n_outer=10, tol=1e-4)
        self.b_v = vector_potential_for_E_single_m_numpy(self.r_target, self.m, self.omega, self.r_source)
        self.electric_field = SCSM_FMM_E2(self.charges, self.head_mesh.triangle_centers, self.r_target, eps=1e-15,
                                          b_im=self.b_v)
        save_results(self.res_path, self.electric_field, self.charges)

class head_mesh:

    def __init__(self, path, file_names, sigmas, file_format=None, unit=None):
        # set standard parameters for processing
        self.roi = None
        self.triangle_centers = None
        self.areas = None
        self.triangle_points = None
        self.n_v = None
        self.avg_length = None
        self.connections = None
        self.locations = None
        self.sigmas_in = None
        self.sigmas_out = None

        # read in input parameters
        self.path = path
        self.file_names = file_names
        self.sigmas = sigmas

        if file_format is not None:
            self.file_format = file_format
        else:
            self.file_format = "stl"

        if unit is not None:
            self.unit = unit
        else:
            self.unit = 'm'
    @timer
    def read_mesh(self):

        layer_sizes = [0]
        layers = self.file_names

        unit_factor = 1.0
        if self.unit == 'cm':
            unit_factor = 0.01
        if self.unit == 'mm':
            unit_factor = 0.001
        elif not self.unit == 'mm':
            warnings.warn('ValueError: not implemented unit used! \n implemented units are : m, cm, mm')


        for i_layer, layer_str in enumerate(layers):
            if self.file_format == 'txt':
                files_con = ["connections_1_" + str(layers[i_layer]), "connections_2_" + str(layers[i_layer]),
                             "connections_3_" + str(layers[i_layer])]
                files_loc = ["x_positions_" + str(layers[i_layer]), "y_positions_" + str(layers[i_layer]),
                             "z_positions_" + str(layers[i_layer])]

                # test sizes before defining array with ..._dummy
                connections_dummy = np.genfromtxt(os.path.join(self.path, files_con[0] + ".txt"), dtype=float) - 1
                con_size = connections_dummy.shape[0]
                locations_dummy = np.genfromtxt(os.path.join(self.path, files_loc[0] + ".txt"), dtype=float)
                loc_size = locations_dummy.shape[0]

                # define connections and locations and use the dummy size
                connections = np.zeros([3, con_size])
                locations = np.zeros([3, loc_size])

                # add the dummy points
                connections[0, :] = connections_dummy
                locations[0, :] = locations_dummy

                # add connections and locations that weren't loaded through the dummy (points 2 and 3)
                for i in [1, 2]:
                    connections[i, :] = np.genfromtxt(os.path.join(self.path, files_con[i] + ".txt"), dtype=float) - 1
                for i in [1, 2]:
                    locations[i, :] = np.genfromtxt(os.path.join(self.path, files_loc[i] + ".txt"), dtype=float)
            elif self.file_format == 'stl':
                file_path = os.path.join(self.path, layer_str + '.stl')
                stl_mesh = meshio.read(file_path)
                locations = stl_mesh.points.T
                connections = stl_mesh.cells_dict['triangle'].T
            else:
                warnings.warn('TypeError: File format mot implemented yet')

            # add unit_factor for cm/mm values
            norm_locations = unit_factor * locations

            # define current output for mesh
            self.triangle_mesh(norm_locations, connections)
            # add number of elements in layer to layers_sizes = con_sizes for redundancy
            layer_sizes.append(self.triangle_centers_i.shape[0])

            # use layered_sphere formulation to concatenate tc, area, points, n, avg_l
            if i_layer == 0:
                self.triangle_centers = self.triangle_centers_i
                self.areas = self.areas_i
                self.triangle_points = self.triangle_points_i
                self.n_v = self.n_v_i
                self.avg_length = np.array([self.avg_length_i])
            else:
                self.triangle_centers = np.vstack((self.triangle_centers, self.triangle_centers_i))
                self.areas = np.concatenate((self.areas, self.areas_i))
                self.triangle_points = np.vstack((self.triangle_points, self.triangle_points_i))
                self.n_v = np.vstack((self.n_v, self.n_v_i))
                self.avg_length = np.concatenate((self.avg_length, np.array([self.avg_length_i])))
            if (i_layer == len(layers)-1):
                self.norm_locations_last = norm_locations
                self.connections_last = connections


        self.avg_length = np.mean(self.avg_length)

        # set up realistic sigma values using layer_sizes

        # hard coded manual cumulative sum (numpy arrays have a function for this: np.cumsum())
        num_elements = []
        num_elements.append(layer_sizes[0])
        num_elements.append(layer_sizes[1])
        num_elements.append(layer_sizes[1] + layer_sizes[2])
        num_elements.append(layer_sizes[1] + layer_sizes[2] + layer_sizes[3])

        self.sigmas_in = np.zeros(self.triangle_centers.shape[0])
        self.sigmas_out = np.zeros_like(self.sigmas_in)
        for j in range(self.sigmas.shape[0]):
            if j < (self.sigmas.shape[0] - 1):
                self.sigmas_in[num_elements[j]:num_elements[j+1]] = self.sigmas[j]
                self.sigmas_out[num_elements[j]:num_elements[j+1]] = self.sigmas[j + 1]
            else:
                self.sigmas_in[num_elements[j]:num_elements[j+1]] = self.sigmas[j]
                self.sigmas_out[num_elements[j]:num_elements[j+1]] = 0.0



        # return triangle_centers, areas, triangle_points, n_v, avg_length, sigmas_in, sigmas_out, roi


    def triangle_mesh(self, locations, connections):

        triangle_centers = np.zeros([len(connections[0, :]), 3])
        areas = np.zeros(len(connections[0, :]))
        n_v = np.zeros_like(triangle_centers)
        n = len(connections[0, :])
        triangle_points = np.zeros((n, 3, 3))
        edge_lens = np.zeros(n)
        for i in range(n):
            p1 = locations[:, int(connections[0, i])]
            p2 = locations[:, int(connections[1, i])]
            p3 = locations[:, int(connections[2, i])]
            triangle_points[i][:][:] = np.vstack((p1, p2, p3))
            p_c_1 = (p1[0] + p2[0] + p3[0]) / 3
            p_c_2 = (p1[1] + p2[1] + p3[1]) / 3
            p_c_3 = (p1[2] + p2[2] + p3[2]) / 3
            triangle_centers[i, :] = np.array([p_c_1, p_c_2, p_c_3])
            line1_2 = p2 - p1
            line1_3 = p3 - p1
            line2_3 = p3 - p2
            areas[i] = 0.5 * vnorm(np.cross(line1_2, line1_3))
            n_v[i] = - np.cross((p3 - p1), (p2 - p3)) / (2 * areas[i])
            edge_lens[i] = 1 / 3 * (np.linalg.norm(line1_2) + np.linalg.norm(line1_3) + np.linalg.norm(line2_3))
        avg_length = np.mean(edge_lens)
        self.triangle_centers_i = triangle_centers
        self.areas_i = areas
        self.triangle_points_i = triangle_points
        self.n_v_i = n_v
        self.avg_length_i = avg_length

    def create_test_roi(self, type='defalted', defalted_scaling=0.9):

        if type == 'deflated':
            # create roi_object as deflated surface
            roi_triangle_centres = defalted_scaling * self.triangle_centers_i
            roi_locations = defalted_scaling * self.norm_locations_last
            roi_connections = self.connections_last
            self.roi = head_roi(triangle_centers=roi_triangle_centres, locations=roi_locations,
                                    connections=roi_connections)

