import numpy as np
import scipy as sp
from scipy.linalg import get_blas_funcs, get_lapack_funcs
from scipy.sparse.sputils import upcast
from functools import wraps
from inspect import signature
from inspect     import signature
from collections import namedtuple
class Singleton(type):
    """
    class MySingleton(metaclass=Singleton):
        ...
    Setting `metaclass=Singleton` in the classes meta descriptor marks it as a
    singleton object: if the object has already been constructed elsewhere in
    the code, subsequent calls to the constructor just return this original
    instance.
    """

    # Stores instances in a dictionary:
    # {class: instance}
    _instances = dict()

    def __call__(cls, *args, **kwargs):
        """
        Metclass __call__ operator is called before the class constructor -- so
        this operator will check if an instance already exists in
        Singleton._instances. If it doesn't call the constructor and add the
        instance to Singleton._instances. If it does, then don't call the
        constructor and return the instance instead.
        """
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(
                *args, **kwargs
            )

        return cls._instances[cls]

class RC(metaclass=Singleton):
    """
    Stores settings used by the compiler
    """
    def __init__(self):
        self._enable_numba = True
        self._lock = False

    @property
    def enable_numba(self):
        return self._enable_numba

    @enable_numba.setter
    def enable_numba(self, val):
        if not self._lock:
            self._enable_numba = val
        else:
            raise RuntimeError(
                "Cannot set enable_numba after compiler has been loaded"
            )

    def lock(self):
        self._lock = True


class Decorated(metaclass=Singleton):
    """
    Stores a record of decorated function
    """
    def __init__(self):
        self.descriptor = namedtuple(
            "FunctionDescriptor", ("module", "name", "signature", "func")
        )
        self._record = set()

    @property
    def record(self):
        return self._record

    def add(self, func):
        self._record.add(
            self.descriptor(
                module=func.__module__,
                name=func.__name__,
                signature=signature(func),
                func=func
            )
        )

    def has_name(self, name):
        return filter(lambda e:e.name==name, self._record)


def get_undecorated_fn(name):
    return next(Decorated().has_name(name)).func


RC().lock()

if RC().enable_numba:
    import numba
    from numba.typed import List
else:
    List = list()


def jit(**kwargs):
    """
    Conditional Numba compiler decorator that invokes the compiler iff
    RC().enable_numba = True when decorator is invoked (i.e. when the decorated
    function is first defined.)
    """

    def noop(func):
        return func

    def op(func):
        _op = wraps(func)(
            numba.jit(**kwargs)(
                func
            )
        )
        _op.__signature__ = signature(func)
        Decorated().add(func)
        return _op

    if RC().enable_numba:
        return op
    else:
        return noop




@jit(nogil=True, nopython=True)
def mat_to_a(a):
    return np.asarray(a)

@jit(nogil=True, nopython=True)
def update_solution(x, y, q):
    g = np.zeros_like(x)

    for i, iy in enumerate(y):
        # Avoid += => observed to reduction in precision.
        g = g + q[i] * iy

    return g


@jit(nogil=True, nopython=True)
def GMRES(A, b, x0, e, nmax_iter, restart=None, debug=False):
    """
    Quick and dirty GMRES -- TODO: optimize going to larger systems.
    """

    b = b.astype(np.float64)
    x0 = x0.astype(np.float64)

    # TODO: you can use this to make the problem agnostic to complex numbers
    # # Defining xtype as dtype of the problem, to decide which BLAS functions
    # # import.
    # xtype = upcast(x0.dtype, b.dtype)

    # Defining dimension
    dimen = x0.shape[0]

    # TODO: use BLAS functions
    # # Get fast access to underlying BLAS routines
    # [lartg] = get_lapack_funcs(['lartg'], [x0] )
    # if np.iscomplexobj(np.zeros((1,), dtype=xtype)):
    #     [axpy, dotu, dotc, scal] =\
    #         get_blas_funcs(['axpy', 'dotu', 'dotc', 'scal'], [x0])
    # else:
    #     # real type
    #     [axpy, dotu, dotc, scal] =\
    #         get_blas_funcs(['axpy', 'dot', 'dot', 'scal'], [xO])

    # TODOs for this function:
    # 1. list -> numpy.array <= better memory access
    # 2. don't append to lists -> prealoc and slice
    # 3. add documentation -- this will probably never happen :P

    normb = np.linalg.norm(b)
    if normb == 0.0:
        normb = 1.0

    r = b - A(x0)

    # Set number of outer loops based on the value of `restart`
    n_outer = 1
    if restart is not None:
        n_outer = int(restart)

    x = List()
    x.append(x0)
    x_sol = x0

    # for l in mrange(n_outer):
    for l in range(n_outer):
        q = [x0] * (nmax_iter)
        q[0] = r / np.linalg.norm(r)

        h = np.zeros((nmax_iter + 1, nmax_iter))

        # for k in mrange(min(nmax_iter, dimen)):
        for k in range(min(nmax_iter, dimen)):
            y = A(q[k])

            # Modified Grahm-Schmidt
            for j in range(k + 1):
                # use flatten -> enable N-D dot product
                h[j, k] = np.dot(q[j].flatten(), y.flatten())
                y = y - h[j, k] * q[j]

            h[k + 1, k] = np.linalg.norm(y)

            if (h[k + 1, k] != 0 and k != nmax_iter - 1):
                q[k + 1] = y / h[k + 1, k]

            # Debug-mode tracks inner-loop convergence
            if debug:
                beta = np.zeros(nmax_iter + 1)
                beta[0] = np.linalg.norm(r)
                y = np.linalg.lstsq(h, beta)[0]
                g = update_solution(x_sol, y[:k], q[:k])
                x.append(x_sol + g)

        beta = np.zeros(nmax_iter + 1)
        beta[0] = np.linalg.norm(r)
        y = np.linalg.lstsq(h, beta)[0]
        g = update_solution(x_sol, y, q)

        x_sol = x_sol + g
        x.append(x_sol)

        r = b - A(x_sol)

        # Break out if the residual is lower than threshold
        if np.linalg.norm(r) / normb < e:
            break

    return x


def apply_givens(Q, v, k):
    """
    Apply the first k Givens rotations in Q to the vector v.
    Arguments
    ---------
        Q: list, list of consecutive 2x2 Givens rotations
        v: array, vector to apply the rotations to
        k: int, number of rotations to apply
    Returns
    -------
        v: array, that is changed in place.
    """

    for j in range(k):
        Qloc = Q[j]
        # TODO: why sp.dot and not np.dot?
        v[j:j + 2] = sp.dot(Qloc, v[j:j + 2])


def GMRES_R(A, b, x0, tol, max_outer, max_inner, restart=None):
    """
    Quick and dirty GMRES -- TODO: optimize mem footprint when going to larger
    systems.
    """

    X = x0

    # Defining xtype as dtype of the problem, to decide which BLAS functions
    # import.
    xtype = upcast(X.dtype, b.dtype)

    # Get fast access to underlying BLAS routines
    # dotc is the conjugate dot, dotu does no conjugation

    [lartg] = get_lapack_funcs(['lartg'], [X])
    if np.iscomplexobj(np.zeros((1,), dtype=xtype)):
        [axpy, dotu, dotc, scal] = \
            get_blas_funcs(['axpy', 'dotu', 'dotc', 'scal'], [X])
    else:
        # real type
        [axpy, dotu, dotc, scal] = \
            get_blas_funcs(['axpy', 'dot', 'dot', 'scal'], [X])

    # Make full use of direct access to BLAS by defining own norm
    def norm(z):
        return np.sqrt(np.real(dotc(z, z)))

    # Defining dimension
    dimen = len(X)

    # TODOs for this function:
    # 1. list -> numpy.array <= better memory access
    # 2. don't append to lists -> prealoc and slice
    # 3. lapack replacemnt for matmul_a?
    # 4. clean up this function!
    # 3. add documentation -- this will probably never happen :P

    r = b - matmul_a(A, x0)

    normr = norm(r)
    normb = norm(b)
    if normb == 0.0:
        normb = 1.0

    iteration = 0
    x = list()

    # Here start the GMRES
    for outer in range(max_outer):
        # Preallocate for Givens Rotations, Hessenberg matrix and Krylov Space
        # Space required is O(dimen*max_inner).
        # NOTE:  We are dealing with row-major matrices, so we traverse in a
        #        row-major fashion,
        #        i.e., H and V's transpose is what we store.

        Q = []  # Initialzing Givens Rotations
        # Upper Hessenberg matrix, which is then
        # converted to upper triagonal with Givens Rotations

        H = np.zeros((max_inner + 1, max_inner + 1), dtype=xtype)
        V = np.zeros((max_inner + 1, dimen), dtype=xtype)  # Krylov space

        # vs store the pointers to each column of V.
        # This saves a considerable amount of time.
        vs = []

        # v = r/normr
        V[0, :] = scal(1.0 / normr, r)  # scal wrapper of dscal --> x = a*x
        vs.append(V[0, :])

        # Saving initial residual to be used to calculate the rel_resid
        if iteration == 0:
            res_0 = normb

        # RHS vector in the Krylov space
        g = np.zeros((dimen,), dtype=xtype)
        g[0] = normr

        for inner in range(max_inner):
            # New search direction
            v = V[inner + 1, :]  # pointer!
            v[:] = matmul_a(A, vs[-1])
            vs.append(v)

            # Modified Gram Schmidt
            for k in range(inner + 1):
                vk = vs[k]
                alpha = dotc(vk, v)
                H[inner, k] = alpha
                v[:] = axpy(vk, v, dimen, -alpha)  # y := a*x + y
                # axpy is a wrapper for daxpy (blas function)

            normv = norm(v)
            H[inner, inner + 1] = normv

            # Check for breakdown
            if H[inner, inner + 1] != 0.0:
                v[:] = scal(1.0 / H[inner, inner + 1], v)

            # Apply for Givens rotations to H
            if inner > 0:
                apply_givens(Q, H[inner, :], inner)

            # Calculate and apply next complex-valued Givens rotations

            # If max_inner = dimen, we don't need to calculate, this
            # is unnecessary for the last inner iteration when inner = dimen -1

            if inner != dimen - 1:
                if H[inner, inner + 1] != 0:
                    # lartg is a lapack function that computes the parameters
                    # for a Givens rotation
                    [c, s, _] = lartg(H[inner, inner], H[inner, inner + 1])
                    Qblock = np.array([[c, s], [-np.conjugate(s), c]], dtype=xtype)
                    Q.append(Qblock)

                    # Apply Givens Rotations to RHS for the linear system in
                    # the krylov space. TODO: why sp.dot and not np.dot?
                    g[inner:inner + 2] = sp.dot(Qblock, g[inner:inner + 2])

                    # Apply Givens rotations to H
                    H[inner, inner] = dotu(Qblock[0, :], H[inner, inner:inner + 2])
                    H[inner, inner + 1] = 0.0

            iteration += 1

            if inner < max_inner - 1:
                normr = abs(g[inner + 1])
                rel_resid = normr / res_0

                if rel_resid < tol:
                    break

        # end inner loop, back to outer loop

        # Find best update to X in Krylov Space V.  Solve inner X inner system.
        y = sp.linalg.solve(H[0:inner + 1, 0:inner + 1].T, g[0:inner + 1])
        update = np.ravel(sp.mat(V[:inner + 1, :]).T.dot(y.reshape(-1, 1)))
        X = X + update
        aux = matmul_a(A, X)
        r = b - aux

        normr = norm(r)
        rel_resid = normr / res_0

        x.append(X)

        # test for convergence
        if rel_resid < tol:
            print('GMRES solve')
            print(f'Converged after {iteration} iterations to a residual of {rel_resid}')
            return x

    # end outer loop

    return x