import numpy as np
from functions import*

M = 100
eps0 = 3e-12
omega = 1
sig = 1
rs = np.random.rand(M, 3)
tri_points = np.random.rand(M, 3, 3)
areas = np.random.rand(M)
A_mat = np.zeros((M, M), dtype=np.complex_)
A_mat_r = np.zeros((M, M))
B = np.zeros(M, dtype=np.complex_)
B_r = np.zeros(M)
m = np.array([0, 1, 0])
r0 = np.array([0, 1, 0])

for i in range(M):
    r_norm_i = rs[i] / vnorm(rs[i])
    p1 = tri_points[i][0]
    p2 = tri_points[i][1]
    p3 = tri_points[i][2]
    n = - np.cross((p3 - p1), (p2 - p3)) / (2 * areas[i])
    for j in range(M):
        A11 = np.dot((rs[i, :] - rs[j, :]), n)
        A12 = (4 * np.pi * eps0 * vnorm(rs[i, :] - rs[j, :]) ** 3 + kroen(i, j))
        A1 = A11 / A12
        A2 = kroen(i, j) / (eps0 * areas[i]) * ((1 / 2) + ((1j * omega * eps0) / sig))
        A2_r = kroen(i, j) / (eps0 * areas[i]) * ((1 / 2) + ((1 * omega * eps0) / sig))
        A_mat[i, j] = A1 - A2
        A_mat_r[i, j] = A1 - A2_r
    B[i] = 1j * omega * 1e-7 * np.dot(np.cross(m, (rs[i] - r0)), n) / (vnorm(rs[i] - r0) ** 3)
    B_r[i] = omega * 1e-7 * np.dot(np.cross(m, (rs[i] - r0)), n) / (vnorm(rs[i] - r0) ** 3)
x_ref = np.linalg.solve(A_mat, B)
x_ref_r = np.linalg.solve(A_mat_r, B_r)
print(f"reference x_ref = {x_ref_r[:5]}")

ITERATION_LIMIT = 1000


def A(i):
    a = np.zeros(M, dtype=np.complex_)
    for j in range(M):
        a11 = np.dot((rs[i, :] - rs[j, :]), n)
        a12 = (4 * np.pi * eps0 * vnorm(rs[i, :] - rs[j, :]) ** 3 + kroen(i, j))
        a1 = a11 / a12
        a2 = kroen(i, j) / (eps0 * areas[i]) * ((1 / 2) + ((1j * omega * eps0) / sig))
        a[j] = a1 - a2
    return a


def A_r(i):
    a = np.zeros(M)
    for j in range(M):
        a11 = np.dot((rs[i, :] - rs[j, :]), n)
        a12 = (4 * np.pi * eps0 * vnorm(rs[i, :] - rs[j, :]) ** 3 + kroen(i, j))
        a1 = a11 / a12
        a2 = kroen(i, j) / (eps0 * areas[i]) * ((1 / 2) + ((omega * eps0) / sig))
        a[j] = a1 - a2
    return a

def A_guess(i):
    a = 1 / (eps0 * areas[i]) * ((1 / 2) + ((1j * omega * eps0) / sig))
    return a

def A_guess_r(i):
    a = 1 / (eps0 * areas[i]) * ((1 / 2) + ((omega * eps0) / sig))
    return a

def jacobi(x_0, b, A_fun, n, n_iter, tol, A):
    for i_iter in range(n_iter):
        x_new = np.zeros_like(x_0)
        if i_iter != 0:
            print("Iteration {0}: {1}".format(i_iter, x_old[:3]))
        if i_iter == 0:
            x_old = x_0
        for i in range(n):
            a_i = A_fun(i)
            a_ii = a_i[i]
            a_i[i] = 0
            s1 = np.dot(A[i, :i], x[:i])
            s2 = np.dot(A[i, i + 1:], x[i + 1:])
            x_new[i] = (b[i] - s1 - s2) / A[i, i]
            # x_new[i] = (b[i] - np.dot(a_i, x_old)) / a_ii
            if x_new[i] == x_new[i - 1]:
                    print("broke after x_new_i = x_new_i-1")
                    break
        if np.allclose(x_old, x_new, atol=tol, rtol=0.):
            print("broke after x close to x_new")
            break

        if not np.alltrue(np.abs(x_new) < 1e-11):
            print("broke after x not converging")
            break
        else:
            x_old = x_new
    return x_new


# x = np.zeros(M, dtype=np.complex_)
# x = np.zeros(M)
# b = np.zeros_like(x)
# for i in range(M):
#     b[i] = 1 * omega * 1e-7 * np.dot(np.cross(m, (rs[i] - r0)), n) / (vnorm(rs[i] - r0) ** 3)
#     x[i] = b[i] / A_guess_r(i)
#
# x = jacobi(x_0=x, b=B_r, A_fun=A_r, n=M, n_iter=1000, tol=1e-27, A=A_mat_r)
#
# print("Solution: ")
# print(x[:5])
# error = x_ref_r - x
# rel_error = np.linalg.norm(error / x_ref_r)
# print("Error:")
# print(rel_error)

# for it_count in range(ITERATION_LIMIT):
#     x_new = np.zeros_like(x)
#     if it_count != 0:
#         print("Iteration {0}: {1}".format(it_count, x))
#
#     for i in range(M):
#         p1 = tri_points[i][0]
#         p2 = tri_points[i][1]
#         p3 = tri_points[i][2]
#         n = - np.cross((p3 - p1), (p2 - p3)) / (2 * areas[i])
#         s1 = np.dot(A(i)[:i], x[:i])
#         s2 = np.dot(A(i)[i + 1:], x[i + 1:])
#         x_new[i] = (b[i] - s1 - s2) / A(i)[i]
#         if x_new[i].imag == x_new[i-1].imag:
#             print("broke after x = x_new")
#             break
#
#     if np.allclose(x.imag, x_new.imag, atol=1e-19, rtol=0.):
#         print("broke after x close to x_new")
#         break
#
#     x = x_new
#
# print("Solution: ")
# print(x)
# error = x_ref - x
# rel_error = np.linalg.norm(error / x_ref)
# print("Error:")
# print(rel_error)
#


x = 1e-20 * np.ones_like(x_ref_r)
x = np.zeros_like(x_ref_r)
A = A_mat_r
b = B_r
for it_count in range(ITERATION_LIMIT):
    if it_count != 0:
        print("Iteration {0}: {1}".format(it_count, x[:5]))
    x_new = np.zeros_like(x)

    for i in range(A.shape[0]):
        s1 = np.dot(A[i, :i], x[:i])
        s2 = np.dot(A[i, i + 1:], x[i + 1:])
        x_new[i] = (b[i] - s1 - s2) / A[i, i]
        if x_new[i] == x_new[i-1]:
            print("broke after x = x_new")
            break


    if np.allclose(x, x_new, atol=1e-20, rtol=0.):
        print("broke after x close to x_new")
        break
    if not np.alltrue(np.abs(x_new) < 1e-11):
        print("broke after x not converging")
        break
    x = x_new

print("Solution: ")
print(x[:5])
error = x_ref_r - x
rel_error = np.linalg.norm(error / x_ref_r)
print("Error:")
print(rel_error)

# A = A_mat.imag
# x = np.zeros(M)
# for it_count in range(ITERATION_LIMIT):
#     if it_count != 0:
#         print("Iteration {0}: {1}".format(it_count, x[:5]))
#     x_new = np.zeros_like(x)
#
#     for i in range(A.shape[0]):
#         s1 = np.dot(A[i, :i], x[:i])
#         s2 = np.dot(A[i, i + 1:], x[i + 1:])
#         x_new[i] = (b[i] - s1 - s2) / A[i, i]
#         if x_new[i] == x_new[i-1]:
#             print("broke after x = x_new")
#             break
#
#
#     if np.allclose(x, x_new, atol=1e-23, rtol=0.):
#         print("broke after x close to x_new")
#         break
#     if not np.alltrue(np.abs(x_new.imag) < 1e-11):
#         print("broke after x not converging")
#         break
#     x = x_new
