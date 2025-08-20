import numpy as np
from scipy.fft import fft, ifft, fftn, ifftn
import scipy.linalg as spl
from scipy.linalg import circulant

def build_circulant_col(size):
    col = np.zeros(size)
    col[0] = 1
    col[1] = -1
    return col

def build_C_3D(n_x, n_y, n_z, lmbda_x, lmbda_y, lmbda_z):
    Cx_nx = spl.circulant(build_circulant_col(n_x))
    Cx_nx_ny_nz = np.kron(np.eye(n_y * n_z), Cx_nx)

    Cx_ny = spl.circulant(build_circulant_col(n_y))
    Cy_nx_ny = np.kron(Cx_ny, np.eye(n_x))
    Cy_nx_ny_nz = np.kron(np.eye(n_z), Cy_nx_ny)

    Cx_nz = spl.circulant(build_circulant_col(n_z))
    Cz_nx_ny_nz = np.kron(Cx_nz, np.eye(n_x * n_y))
    
    C = np.eye(n_x * n_y * n_z) + lmbda_x * Cx_nx_ny_nz + lmbda_y * Cy_nx_ny_nz + lmbda_z * Cz_nx_ny_nz
    return C

def build_diag_mat_vec_3D(n_x, n_y, n_z, lmbda_x, lmbda_y, lmbda_z):
    return

def vectorized_fftn(v, n_x, n_y):
    v = v.reshape((n_y, n_x))
    v_hat = fftn(v)
    return v_hat.flatten()

def vectorized_ifftn(v, n_x, n_y):
    v = v.reshape((n_y, n_x))
    v_hat = ifftn(v)
    return v_hat.flatten()

def solve_circulant_system_3D(Diag, b, n_x, n_y, n_z):
    return

def test_fft_precond_3D(n_x=5, n_y=5, n_z=5, a_x=1, a_y=1, a_z=1, delta_t=0.01, delta_x=0.1, delta_y=0.1, delta_z=0.1, seed=123):
    # Matrice circulante par bloc
    lmbda_x = a_x * delta_t / delta_x
    lmbda_y = a_y * delta_t / delta_y
    lmbda_z = a_z * delta_t / delta_z
    C = build_C_3D(n_x, n_y, n_z, lmbda_x, lmbda_y, lmbda_z)
    print(C)

    # Génerer un X random qui sera le même à chaque execution
    rng = np.random.default_rng(seed)
    X_ref = rng.random((n_x, n_y, n_z))
    b = C @ X_ref.flatten()

    # Matrice diagonale
    # Diag = build_diag_mat_vec_3D(n_x, n_y, lmbda_x, lmbda_y)
    
    # Résolution
    # X = solve_circulant_system_3D(Diag, b, n_x, n_y)

    # Vérification
    # error_rel = np.linalg.norm(X_ref - X.reshape(n_x, n_y)) / np.linalg.norm(X_ref)
    # residual_rel = np.linalg.norm(C @ X.flatten() - b) / np.linalg.norm(b)
    # print("error_rel     =", error_rel)
    # print("residual_rel  =", residual_rel)
    # print("X_ref =\n", X_ref)
    # print("X =\n", X.reshape(size, size))
    # print("X_ref - X =\n", X_ref - X.reshape(size, size))


size=100
n_x=5
n_y=2
n_z=3
a_x=30
a_y=3
a_z=1
delta_t=0.01
delta_x=0.1
delta_y=0.1
delta_z=0.1

test_fft_precond_3D(n_x=n_x, n_y=n_y, n_z=n_z, a_x=a_x, a_y=a_y, a_z=a_z, delta_t=delta_t, delta_x=delta_x, delta_y=delta_y, delta_z=delta_z)