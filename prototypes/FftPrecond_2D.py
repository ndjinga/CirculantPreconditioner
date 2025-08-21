import numpy as np
from scipy.fft import fft, ifft, fft2, ifft2
import scipy.linalg as spl
from scipy.linalg import circulant

def build_circulant_col(size):
    col = np.zeros(size)
    col[0] = 1
    col[1] = -1
    return col

def build_C_2D(n_x, n_y, lmbda_x, lmbda_y):
    Cx_nx = spl.circulant(build_circulant_col(n_x))
    Cx_nx_ny = np.kron(np.eye(n_y), Cx_nx)

    Cx_ny = spl.circulant(build_circulant_col(n_y))
    Cy_nx_ny = np.kron(Cx_ny, np.eye(n_x))
    
    C = np.eye(n_x * n_y) + lmbda_x * Cx_nx_ny + lmbda_y * Cy_nx_ny
    return C

def build_diag_mat_vec_2D(n_x, n_y, lmbda_x, lmbda_y):
    c_nx = build_circulant_col(n_x)
    c_ny = build_circulant_col(n_y)

    # print("c_nx =\n", c_nx)
    # print("c_ny =\n", c_ny)

    c_nx_hat = fft(c_nx)
    c_ny_hat = fft(c_ny)

    # print("c_nx_hat =\n", c_nx_hat)
    # print("c_ny_hat =\n", c_ny_hat)

    Diag = 1 + lmbda_x * np.tile(c_nx_hat, n_y) + lmbda_y * np.repeat(c_ny_hat, n_x)
    # print("Diag =\n", Diag)
    return Diag

def vectorized_fft2(v, n_x, n_y):
    v = v.reshape((n_y, n_x))
    v_hat = fft2(v)
    return v_hat.flatten()

def vectorized_ifft2(v, n_x, n_y):
    v = v.reshape((n_y, n_x))
    v_hat = ifft2(v)
    return v_hat.flatten()

def solve_circulant_system_2D(Diag, b, n_x, n_y):
    b_hat = vectorized_fft2(b, n_x, n_y)
    x_hat = b_hat / Diag # np.where(Diag != 0, b_hat / Diag, 0)
    X = vectorized_ifft2(x_hat, n_x, n_y)
    return X

def test_fft_precond_2D(n_x=5, n_y=5, a_x=1, a_y=1, delta_t=0.01, delta_x=0.1, delta_y=0.1, seed=123):
    # Matrice circulante par bloc
    lmbda_x = a_x * delta_t / delta_x
    lmbda_y = a_y * delta_t / delta_y
    C = build_C_2D(n_x, n_y, lmbda_x, lmbda_y)

    # Génerer un X random qui sera le même à chaque execution
    rng = np.random.default_rng(seed)
    X_ref = rng.random((n_x, n_y))
    b = C @ X_ref.flatten()

    # Matrice diagonale
    Diag = build_diag_mat_vec_2D(n_x, n_y, lmbda_x, lmbda_y)
    
    # Résolution
    X = solve_circulant_system_2D(Diag, b, n_x, n_y)

    # Vérification
    error_rel = np.linalg.norm(X_ref - X.reshape(n_x, n_y)) / np.linalg.norm(X_ref)
    residual_rel = np.linalg.norm(C @ X.flatten() - b) / np.linalg.norm(b)
    print("Relative error     =", error_rel)
    print("Relative residual  =", residual_rel)
    # print("X_ref =\n", X_ref)
    # print("X =\n", X.reshape(size, size))
    # print("X_ref - X =\n", X_ref - X.reshape(size, size))


n_x=50
n_y=200
a_x=30
a_y=3
delta_t=0.01
delta_x=0.1
delta_y=0.1

test_fft_precond_2D(n_x=n_x, n_y=n_y, a_x=a_x, a_y=a_y, delta_t=delta_t, delta_x=delta_x, delta_y=delta_y)