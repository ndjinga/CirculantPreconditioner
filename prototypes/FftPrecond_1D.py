import numpy as np
from scipy.fft import fft, ifft
import scipy.linalg as spl

def build_circulant_col(size, lmbda):
    col = np.zeros(size)
    col[0] = 1 + lmbda
    col[1] = -lmbda
    return col

def solve_circulant_system(C_col, b):
    lambdas = fft(C_col)
    b_hat = fft(b)

    x_hat = b_hat / lambdas
    x = ifft(x_hat).real
    return x

def test_fft_precond_1D(size=5, a=3, delta_t=1, delta_x=1, seed=123):
    rng = np.random.default_rng(seed)
    lmbda = a * delta_t / delta_x
    C_col = build_circulant_col(size, lmbda)
    C_mat = spl.circulant(C_col)
    
    x_ref = rng.random(size)
    b = C_mat @ x_ref

    x = solve_circulant_system(C_col, b)
    
    rel_error = np.linalg.norm(x - x_ref) / np.linalg.norm(x_ref)
    rel_residual = np.linalg.norm(C_mat @ x - b) / np.linalg.norm(b)
    # print("C_mat @ x =", C_mat @ x)
    # print("C_mat    =", C_mat)
    # print("x         =", x)
    # print("b         =", b)
    print("Relative error =", rel_error)
    print("Relative residual =", rel_residual)

size=8
a=1
delta_t=1
delta_x=1
test_fft_precond_1D(size=size, a=a, delta_t=delta_t, delta_x=delta_x)