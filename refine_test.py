from scipy.spatial import Delaunay
import numpy as np
import matplotlib.pyplot as plt

n = 5
a = np.linspace(0, 1, n)
b = np.linspace(0, 1, n)
u, v = np.meshgrid(a, b)
points = []
for i in range(n):
    for j in range(n):
        points.append(np.array((a[i], a[j])))
points = np.array(points)

tri = Delaunay(points)

tc = np.zeros((tri.simplices.shape[0], 2))
edge_lengths = np.zeros((tri.simplices.shape[0], 3))
for i in range(tri.simplices.shape[0]):
    i_p1 = tri.simplices[i, 0]
    i_p2 = tri.simplices[i, 1]
    i_p3 = tri.simplices[i, 2]
    p1 = points[i_p1]
    p2 = points[i_p2]
    p3 = points[i_p3]
    tc[i] = (1/3) * (p1 + p2 + p3)
    line12 = p2 - p1
    line13 = p3 - p1
    line23 = p3 - p2
    edge_lengths[i] = np.array((np.linalg.norm(line12), np.linalg.norm(line23), np.linalg.norm(line13)))
avg_edge_length = np.average(edge_lengths)


def refine_trianlge(i, points, tris, tc, idxs_refined):
    I = points.shape[0]
    J = tris.shape[0]
    i_p1 = tris[i, 0]
    i_p2 = tris[i, 1]
    i_p3 = tris[i, 2]
    p1 = points[i_p1]
    p2 = points[i_p2]
    p3 = points[i_p3]
    p12 = p1 + (1 / 2) * (p2 - p1)
    p23 = p2 + (1 / 2) * (p3 - p2)
    p13 = p1 + (1 / 2) * (p3 - p1)
    new_points = np.vstack((p12, p23, p13))  # idxs: I, I+1, I+2
    points = np.vstack((points, new_points))
    i_p12 = I
    i_p23 = I + 1
    i_p13 = I + 2
    tris[i, :] = np.array((i_p1, i_p12, i_p13))  # set first new point at original index i
    new_tris = np.array([[i_p12, i_p2, i_p23], [i_p12, i_p13, i_p23], [i_p13, i_p23, i_p3]])
    tris = np.vstack((tris, new_tris))
    new_tcs = np.array([(1 / 3) * (p12 + p2 + p23), (1 / 3) * (p12 + p13 + p23),
                        (1 / 3) * (p13 + p23 + p3)])
    new_tc_i = (1 / 3) * (p1 + p12 + p13)
    tc[i] = new_tc_i
    tc = np.vstack((tc, new_tcs))
    # idxs_refined = np.vstack((idxs_refined, np.array([i, J, J + 1, J + 2])))
    idxs_refined = idxs_refined + [i, J, J + 1, J + 2]
    return tris, points, tc, idxs_refined


def refine_trianlge_half(i, points, tris, tc, shared_points, idxs_refined):
    I = points.shape[0]
    J = tris.shape[0]
    i_p1 = tris[i, 0]
    i_p2 = tris[i, 1]
    i_p3 = tris[i, 2]
    p1 = points[i_p1]
    p2 = points[i_p2]
    p3 = points[i_p3]
    if shared_points[0] == 1:
        p12 = p1 + (1 / 2) * (p2 - p1)
        points = np.vstack((points, p12))
        i_p12 = I
        tris[i, :] = np.array((i_p1, i_p12, i_p3))
        new_tri = np.array((i_p12, i_p2, i_p3))
        tc[i] = (1 / 3) * (p1 + p12 + p3)
        new_tc = (1 / 3) * (p12 + p2 + p3)
    elif shared_points[1] == 1:
        p23 = p2 + (1 / 2) * (p3 - p2)
        points = np.vstack((points, p23))
        i_p23 = I
        tris[i, :] = np.array((i_p1, i_p2, i_p23))
        new_tri = np.array((i_p1, i_p23, i_p3))
        tc[i] = (1 / 3) * (p1 + p2 + p23)
        new_tc = (1 / 3) * (p1+ p23 + p3)
    elif shared_points[2] == 1:
        p13 = p1 + (1 / 2) * (p3 - p1)
        points = np.vstack((points, p13))
        i_p13 = I
        tris[i, :] = np.array((i_p1, i_p2, i_p13))
        new_tri = np.array((i_p13, i_p2, i_p3))
        tc[i] = (1 / 3) * (p1 + p2 + p13)
        new_tc = (1 / 3) * (p13 + p2 + p3)
    tris = np.vstack((tris, new_tri))
    tc = np.vstack((tc, new_tc))
    idxs_refined.append(J)
    return tris, points, tc, idxs_refined


def point_exists(p, points, dim=2):
    mask1 = np.isin(points[:, 0], p[0]).astype(int)
    mask2 = np.isin(points[:, 1], p[1]).astype(int)
    if dim == 2:
        mask_sum = np.vstack((mask1, mask2)).T.sum(axis=1)
    elif dim == 3:
        mask3 = np.isin(points[:, 2], p[2]).astype(int)
        mask_sum = np.vstack((mask1, mask2, mask3)).T.sum(axis=1)
    int_bool = (1/dim)*mask_sum[np.where(mask_sum > 1)].sum()
    return int_bool

def refine(idxs, tris, points, tc):
    # idxs_refined = np.zeros((len(idxs), 4), dtype=int)
    idxs_refined = []
    K = tris.shape[0]
    j_idxs = list(range(K))
    for i in range(len(idxs)):
        j_idxs.remove(idxs[i])

    for i in idxs:
        tris, points, tc, idxs_refined = refine_trianlge(i, points, tris, tc, idxs_refined)

    idxs_half_angle_refinement = []
    idxs2 = []
    if not len(j_idxs) == 0:
        shared_points = np.zeros((np.array(j_idxs).max() + 1, 3))
        for j in j_idxs:
            j_p1 = tris[j, 0]
            j_p2 = tris[j, 1]
            j_p3 = tris[j, 2]
            p1 = points[j_p1]
            p2 = points[j_p2]
            p3 = points[j_p3]
            p12 = p1 + (1 / 2) * (p2 - p1)
            p23 = p2 + (1 / 2) * (p3 - p2)
            p13 = p1 + (1 / 2) * (p3 - p1)
            shared_points[j, 0] = point_exists(p12, points)
            shared_points[j, 1] = point_exists(p23, points)
            shared_points[j, 2] = point_exists(p13, points)
            if shared_points[j].sum() > 1:
                idxs2.append(j)

        for i in idxs2:
            tris, points, tc, idxs_refined = refine_trianlge(i, points, tris, tc, idxs_refined)

        for i in range(len(idxs_refined)):
            if idxs_refined in j_idxs:
                j_idxs.remove(idxs_refined[i])

        idxs3 = []
        for j in j_idxs:
            j_p1 = tris[j, 0]
            j_p2 = tris[j, 1]
            j_p3 = tris[j, 2]
            p1 = points[j_p1]
            p2 = points[j_p2]
            p3 = points[j_p3]
            p12 = p1 + (1 / 2) * (p2 - p1)
            p23 = p2 + (1 / 2) * (p3 - p2)
            p13 = p1 + (1 / 2) * (p3 - p1)
            shared_points[j, 0] = point_exists(p12, points)
            shared_points[j, 1] = point_exists(p23, points)
            shared_points[j, 2] = point_exists(p13, points)
            if shared_points[j].sum() == 1:
                idxs_half_angle_refinement.append(j)
            elif shared_points[j].sum() > 1:
                print("starting second loop")
                idxs3.append(j)

        for i in idxs_half_angle_refinement:
            tris, points, tc, idxs_refined = refine_trianlge_half(i, points, tris, tc, shared_points[i], idxs_refined)

        if not len(idxs3) == 0:

            shared_points = np.zeros((np.array(j_idxs).max() + 1, 3))
            for j in j_idxs:
                j_p1 = tris[j, 0]
                j_p2 = tris[j, 1]
                j_p3 = tris[j, 2]
                p1 = points[j_p1]
                p2 = points[j_p2]
                p3 = points[j_p3]
                p12 = p1 + (1 / 2) * (p2 - p1)
                p23 = p2 + (1 / 2) * (p3 - p2)
                p13 = p1 + (1 / 2) * (p3 - p1)
                shared_points[j, 0] = point_exists(p12, points)
                shared_points[j, 1] = point_exists(p23, points)
                shared_points[j, 2] = point_exists(p13, points)
                if shared_points[j].sum() > 1:
                    idxs2.append(j)

            for i in idxs3:
                tris, points, tc, idxs_refined = refine_trianlge(i, points, tris, tc, idxs_refined)
                idxs4 = []
            idxs_half_angle_refinement2 = []
            for j in j_idxs:
                j_p1 = tris[j, 0]
                j_p2 = tris[j, 1]
                j_p3 = tris[j, 2]
                p1 = points[j_p1]
                p2 = points[j_p2]
                p3 = points[j_p3]
                p12 = p1 + (1 / 2) * (p2 - p1)
                p23 = p2 + (1 / 2) * (p3 - p2)
                p13 = p1 + (1 / 2) * (p3 - p1)
                shared_points[j, 0] = point_exists(p12, points)
                shared_points[j, 1] = point_exists(p23, points)
                shared_points[j, 2] = point_exists(p13, points)
                if shared_points[j].sum() == 1:
                    idxs_half_angle_refinement2.append(j)
                elif shared_points[j].sum() > 1:
                    print("third loop needed")
                    idxs4.append(j)

            for i in idxs_half_angle_refinement2:
                tris, points, tc, idxs_refined = refine_trianlge_half(i, points, tris, tc, shared_points[i],
                                                                      idxs_refined)

    return tris, points, tc, idxs_refined
triangles, points, tc, idxs_refined = refine([11], tri.simplices, points, tc)
# triangles, points, tc, idxs_refined = refine(idxs_refined, triangles, points, tc)
# triangles, points, tc, idxs_refined = refine([0, 5], tri.simplices, points, tc)
# triangles, points, tc, idxs_refined = refine(idxs_refined, triangles, points, tc)
# triangles, points, tc, idxs_refined = refine(idxs_refined, triangles, points, tc)
# triangles, points = refine(idxs_refined, triangles, points, tc)[:2]

# triangles = tri.simplices
plt.triplot(points[:, 0], points[:, 1], triangles)
plt.plot(points[:, 0], points[:, 1], 'o')
plt.show()
