#include <petscmat.h>
#include <petscvec.h>

PetscErrorCode build_circulant_col_1D(Vec c, PetscInt size) {
    PetscFunctionBeginUser;
    VecSet(c, 0.0);
    VecSetValue(c, 0, 1.0, INSERT_VALUES);
    VecSetValue(c, 1, -1.0, INSERT_VALUES);
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode build_circulant_mat_1D( Mat C, PetscInt size, PetscScalar c1, PetscScalar c2) {
    PetscFunctionBeginUser;
    for (PetscInt i = 0; i < size; i++) 
        MatSetValue( C, i, (i+size-1)%size, c2, INSERT_VALUES);
    
    MatAssemblyBegin(C, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(C, MAT_FINAL_ASSEMBLY);

    MatShift(C,c1);
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode create_identity_matrix(Mat Id, int n) {
    PetscFunctionBeginUser;

    for (PetscInt i = 0; i < n; ++i) {
        PetscCall(MatSetValue(Id, i, i, 1.0, INSERT_VALUES));
    }

    PetscCall(MatAssemblyBegin(Id, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(Id, MAT_FINAL_ASSEMBLY));

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode build_C_2D(Mat *C, PetscInt n_x, PetscInt n_y, PetscScalar lambda_x, PetscScalar lambda_y) {
    PetscFunctionBeginUser;
    Mat Cx_nx, Cx_ny;
    Mat Id_nx, Id_ny, Id_nx_ny;
    Mat Cx_nx_ny, Cy_nx_ny;

    PetscCall(MatCreateAIJ(PETSC_COMM_WORLD, n_x, n_x, n_x, n_x, 2, NULL, 1, NULL, &Cx_nx));
    PetscCall(build_circulant_mat_1D(Cx_nx, n_x, 1.0, -1));

    PetscCall(MatCreateAIJ(PETSC_COMM_WORLD, n_y, n_y, n_y, n_y, 2, NULL, 1, NULL, &Cx_ny));
    PetscCall(build_circulant_mat_1D(Cx_ny, n_y, 1.0, -1));

    PetscCall(MatCreateAIJ(PETSC_COMM_WORLD, n_x, n_x, n_x, n_x, 2, NULL, 1, NULL, &Id_nx));
    PetscCall(create_identity_matrix(Id_nx, n_x));

    PetscCall(MatCreateAIJ(PETSC_COMM_WORLD, n_y, n_y, n_y, n_y, 2, NULL, 1, NULL, &Id_ny));
    PetscCall(create_identity_matrix(Id_ny, n_y));

    PetscCall(MatCreateAIJ(PETSC_COMM_WORLD, n_x * n_y, n_x * n_y, n_x * n_y, n_x * n_y, 2, NULL, 1, NULL, &Id_nx_ny));
    PetscCall(create_identity_matrix(Id_nx_ny, n_x * n_y));

    PetscCall(MatSeqAIJKron(Id_ny, Cx_nx, MAT_INITIAL_MATRIX, &Cx_nx_ny));
    PetscCall(MatSeqAIJKron(Cx_ny, Id_nx, MAT_INITIAL_MATRIX, &Cy_nx_ny));

    PetscCall(MatDuplicate(Id_nx_ny, MAT_COPY_VALUES, C));
    PetscCall(MatAXPY(*C, lambda_x, Cx_nx_ny, DIFFERENT_NONZERO_PATTERN));
    PetscCall(MatAXPY(*C, lambda_y, Cy_nx_ny, DIFFERENT_NONZERO_PATTERN));

    PetscCall(MatDestroy(&Cx_nx));
    PetscCall(MatDestroy(&Cx_ny));
    PetscCall(MatDestroy(&Id_nx));
    PetscCall(MatDestroy(&Id_ny));
    PetscCall(MatDestroy(&Cx_nx_ny));
    PetscCall(MatDestroy(&Cy_nx_ny));

    PetscCall(MatAssemblyBegin(*C, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(*C, MAT_FINAL_ASSEMBLY));
    
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode vec_kronecker_product_identity_left(Vec c, Vec res, PetscInt c_size, PetscInt id_size, PetscScalar lambda) {
    PetscFunctionBeginUser;
    PetscScalar *c_array;
    PetscScalar cur;

    VecGetArray(c, &c_array);

    for (PetscInt i = 0; i < c_size; i++) {
        cur = c_array[i] * lambda;
        for (PetscInt j = 0; j < id_size; j++) {
            VecSetValue(res, j*c_size + i, cur, INSERT_VALUES);
        }
    }

    VecRestoreArray(c, &c_array);

    VecAssemblyBegin(c);
    VecAssemblyEnd(c);

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode vec_kronecker_product_identity_right(Vec c, Vec res, PetscInt c_size, PetscInt id_size, PetscScalar lambda) {
    PetscFunctionBeginUser;
    PetscScalar *c_array;
    PetscScalar cur;

    VecGetArray(c, &c_array);

    for (PetscInt i = 0; i < c_size; i++) {
        cur = c_array[i] * lambda;
        for (PetscInt j = 0; j < id_size; j++) {
            VecSetValue(res, i*id_size + j, cur, INSERT_VALUES);
        }
    }

    VecRestoreArray(c, &c_array);

    VecAssemblyBegin(c);
    VecAssemblyEnd(c);

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode build_diag_mat_vec_2D(Vec c, PetscInt n_x, PetscInt n_y, PetscScalar lambda_x, PetscScalar lambda_y) {
    PetscFunctionBeginUser;
    Mat FFT_cx, FFT_cy;
    Vec c_x, c_y, c_x_hat, c_y_hat, sum;
    PetscInt ndim_x = 1;
    PetscInt ndim_y = 1;
    PetscInt dims_x[1] = {n_x};
    PetscInt dims_y[1] = {n_y};


    PetscCall(MatCreateFFT(PETSC_COMM_WORLD, ndim_x, dims_x, MATFFTW, &FFT_cx));
    PetscCall(MatCreateFFT(PETSC_COMM_WORLD, ndim_y, dims_y, MATFFTW, &FFT_cy));

    PetscCall(MatCreateVecsFFTW(FFT_cx, &c_x, &c_x_hat, NULL));
    PetscCall(MatCreateVecsFFTW(FFT_cy, &c_y, &c_y_hat, NULL));
    
    build_circulant_col_1D(c_x, n_x);
    build_circulant_col_1D(c_y, n_y);

    MatMult(FFT_cx, c_x, c_x_hat);
    MatMult(FFT_cy, c_y, c_y_hat);

    Vec kpi_x, kpi_y;
    VecDuplicate(c, &kpi_x);
    VecDuplicate(c, &kpi_y);

    vec_kronecker_product_identity_left(c_x_hat, kpi_x, n_x, n_y, lambda_x);
    vec_kronecker_product_identity_right(c_y_hat, kpi_y, n_y, n_x, lambda_y);

    PetscCall(VecDuplicate(kpi_x, &sum));
    PetscCall(VecAXPY(sum, 1.0, kpi_x));
    PetscCall(VecAXPY(sum, 1.0, kpi_y));
    PetscCall(VecShift(sum, 1.0));

    PetscCall(VecCopy(sum, c));


    PetscCall(VecDestroy(&kpi_x));
    PetscCall(VecDestroy(&kpi_y));
    PetscCall(VecDestroy(&c_x));
    PetscCall(VecDestroy(&c_y));
    PetscCall(VecDestroy(&sum));
    PetscCall(MatDestroy(&FFT_cx));
    PetscCall(MatDestroy(&FFT_cy));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode test_2D_2(int argc, char **argv) {
    PetscInitialize(&argc, &argv, NULL, NULL);
    Mat C;
    PetscInt n_x = 3, n_y = 3;
    PetscScalar lambda_x = 1, lambda_y = 1;
    PetscCall(MatCreateAIJ(PETSC_COMM_WORLD, n_x * n_y, n_x * n_y, n_x * n_y, n_x * n_y, 2, NULL, 1, NULL, &C));
    PetscCall(build_C_2D(&C, n_x, n_y, lambda_x, lambda_y));
    PetscCall(MatView(C, PETSC_VIEWER_STDOUT_WORLD));
    PetscCall(PetscFinalize());
    return 0;
}

PetscErrorCode test_2D(int argc, char **argv) {
    PetscInt n_x = 3;
    PetscInt n_y = 3;
    PetscScalar lambda_x = 1;
    PetscScalar lambda_y = 1;
    Vec c, b;// input vectors of forward fft
    Vec lambdas, b_hat;// output vectors of forward fft = input vectors of backward fft
    Vec x;// output vectors of backward fft
    //Vec r;// residual vector r=AX-b
    //PetscReal rnorm;// residual norm
    Mat FFT_MAT;//, C;
    PetscInt ndim = 1;
    PetscInt dims[1];

    PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));

    dims[0] = n_x * n_y;
    PetscCall(MatCreateFFT(PETSC_COMM_WORLD, ndim, dims, MATFFTW, &FFT_MAT));
    
    PetscCall(MatCreateVecsFFTW( FFT_MAT, &c, &lambdas, &x));
    PetscCall(MatCreateVecsFFTW( FFT_MAT, &b, &b_hat  , NULL));

    build_diag_mat_vec_2D(c, n_x, n_y, lambda_x, lambda_y);
    PetscCall(VecView(c, PETSC_VIEWER_STDOUT_WORLD));

    for (PetscInt i = 0; i < n_x * n_y; i++) 
        VecSetValue( b, i, i*i*i, INSERT_VALUES);

    VecAssemblyBegin(b);
    VecAssemblyEnd(b);

    //PetscCall(solve_circulant_system_2D(FFT_MAT, c, b, x, lambdas, b_hat));

    // Clean up
    PetscCall(VecDestroy(&b));
    PetscCall(VecDestroy(&x));
    PetscCall(VecDestroy(&c));
    PetscCall(VecDestroy(&lambdas));
    PetscCall(VecDestroy(&b_hat));
    PetscCall(MatDestroy(&FFT_MAT));
    PetscCall(PetscFinalize());
    return 0;
}

int main(int argc, char **argv) {
    test_2D(argc, argv);
    return 0;
}