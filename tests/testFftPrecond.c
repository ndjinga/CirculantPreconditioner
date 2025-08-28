#include "FftPrecond_3D.h"

PetscErrorCode build_circulant_mat_1D( Mat C, PetscInt size, PetscScalar c1, PetscScalar c2) {
    PetscFunctionBeginUser;
    for (PetscInt i = 0; i < size; i++) 
        MatSetValue( C, i, (i+size-1)%size, c2, INSERT_VALUES);
    
    MatAssemblyBegin(C, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(C, MAT_FINAL_ASSEMBLY);

    MatShift(C,c1);
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode build_identity_matrix(Mat Id, int n) {
    PetscFunctionBeginUser;

    for (PetscInt i = 0; i < n; ++i) {
        PetscCall(MatSetValue(Id, i, i, 1.0, INSERT_VALUES));
    }

    PetscCall(MatAssemblyBegin(Id, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(Id, MAT_FINAL_ASSEMBLY));

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode build_C_3D(Mat *C, PetscInt n_x, PetscInt n_y, PetscInt n_z, PetscScalar lambda_x, PetscScalar lambda_y, PetscScalar lambda_z) {
    PetscFunctionBeginUser;
    Mat Cx_nx, Cx_ny, Cx_nz;
    Mat Id_nx, Id_nz, Id_nx_ny, Id_ny_nz, Id_nx_ny_nz;
    Mat Cy_nx_ny;
    Mat Cx_nx_ny_nz, Cy_nx_ny_nz, Cz_nx_ny_nz;
    PetscInt size;

    // Cx_nx, Cx_ny, C_x,nz
    PetscCall(MatCreateAIJ(PETSC_COMM_WORLD, n_x, n_x, n_x, n_x, 2, NULL, 1, NULL, &Cx_nx));
    PetscCall(build_circulant_mat_1D(Cx_nx, n_x, 1.0, -1));

    PetscCall(MatCreateAIJ(PETSC_COMM_WORLD, n_y, n_y, n_y, n_y, 2, NULL, 1, NULL, &Cx_ny));
    PetscCall(build_circulant_mat_1D(Cx_ny, n_y, 1.0, -1));

    PetscCall(MatCreateAIJ(PETSC_COMM_WORLD, n_z, n_z, n_z, n_z, 2, NULL, 1, NULL, &Cx_nz));
    PetscCall(build_circulant_mat_1D(Cx_nz, n_z, 1.0, -1));

    // Id_nx, Id_nz, Id_nx_ny, Id_ny_nz, Id_nx_ny_nz
    size = n_x;
    PetscCall(MatCreateAIJ(PETSC_COMM_WORLD, size, size, size, size, 2, NULL, 1, NULL, &Id_nx));
    PetscCall(build_identity_matrix(Id_nx, size));

    size = n_z;
    PetscCall(MatCreateAIJ(PETSC_COMM_WORLD, size, size, size, size, 2, NULL, 1, NULL, &Id_nz));
    PetscCall(build_identity_matrix(Id_nz, size));

    size = n_x * n_y;
    PetscCall(MatCreateAIJ(PETSC_COMM_WORLD, size, size, size, size, 2, NULL, 1, NULL, &Id_nx_ny));
    PetscCall(build_identity_matrix(Id_nx_ny, size));

    size = n_y * n_z;
    PetscCall(MatCreateAIJ(PETSC_COMM_WORLD, size, size, size, size, 2, NULL, 1, NULL, &Id_ny_nz));
    PetscCall(build_identity_matrix(Id_ny_nz, size));

    size = n_x * n_y * n_z;
    PetscCall(MatCreateAIJ(PETSC_COMM_WORLD, size, size, size, size, 2, NULL, 1, NULL, &Id_nx_ny_nz));
    PetscCall(build_identity_matrix(Id_nx_ny_nz, size));

    // Cy_nx_ny, Cx_nx_ny_nz, Cy_nx_ny_nz, Cz_nx_ny_nz
    PetscCall(MatSeqAIJKron(Id_ny_nz, Cx_nx, MAT_INITIAL_MATRIX, &Cx_nx_ny_nz));
    PetscCall(MatSeqAIJKron(Cx_ny, Id_nx, MAT_INITIAL_MATRIX, &Cy_nx_ny));
    PetscCall(MatSeqAIJKron(Id_nz, Cy_nx_ny, MAT_INITIAL_MATRIX, &Cy_nx_ny_nz));
    PetscCall(MatSeqAIJKron(Cx_nz, Id_nx_ny, MAT_INITIAL_MATRIX, &Cz_nx_ny_nz));

    // Weighted sum
    PetscCall(MatDuplicate(Id_nx_ny_nz, MAT_COPY_VALUES, C));
    PetscCall(MatAXPY(*C, lambda_x, Cx_nx_ny_nz, DIFFERENT_NONZERO_PATTERN));
    PetscCall(MatAXPY(*C, lambda_y, Cy_nx_ny_nz, DIFFERENT_NONZERO_PATTERN));
    PetscCall(MatAXPY(*C, lambda_z, Cz_nx_ny_nz, DIFFERENT_NONZERO_PATTERN));

    PetscCall(MatAssemblyBegin(*C, MAT_FINAL_ASSEMBLY));
    PetscCall(MatAssemblyEnd(*C, MAT_FINAL_ASSEMBLY));

    // Clean up
    PetscCall(MatDestroy(&Cx_nx));
    PetscCall(MatDestroy(&Cx_ny));
    PetscCall(MatDestroy(&Id_nx));
    PetscCall(MatDestroy(&Cy_nx_ny));
    
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode test_3D() {
    PetscFunctionBeginUser;

    // Input parameters
    PetscInt n_x = 4;
    PetscInt n_y = 3;
    PetscInt n_z = 2;
    PetscScalar a_x = 1;
    PetscScalar a_y = 1;
    PetscScalar a_z = 1;
    PetscScalar dt = 1;
    PetscScalar delta_x = 1;
    PetscScalar delta_y = 1;
    PetscScalar delta_z = 1;

    PetscScalar lambda_x = a_x * dt / delta_x;
    PetscScalar lambda_y = a_y * dt / delta_y;
    PetscScalar lambda_z = a_z * dt / delta_z;

    // Variables
    Mat C; // Block-circulant matrix
    Vec X_ref, X, b; // CX=b
    Mat FFT_MAT;

    Vec r;// residual vector r=AX-b
    PetscReal rnorm, enorm;// residual norm and error
    PetscReal X_ref_norm, b_norm;

    // Initialize FFT 3D
    PetscInt ndim = 3;
    PetscInt dims[3] = {n_z, n_y, n_x};
    PetscCall(MatCreateFFT(PETSC_COMM_WORLD, ndim, dims, MATFFTW, &FFT_MAT));
    
    // Initialize vectors
    PetscCall(MatCreateVecsFFTW( FFT_MAT, NULL, NULL, &X_ref));
    PetscCall(MatCreateVecsFFTW( FFT_MAT, &b, NULL, &X));

    // Build the block-circulant matrix C
    PetscCall(MatCreateAIJ(PETSC_COMM_WORLD, n_x * n_y * n_z, n_x * n_y * n_z, n_x * n_y * n_z, n_x * n_y * n_z, 2, NULL, 1, NULL, &C));
    PetscCall(build_C_3D(&C, n_x, n_y, n_z, lambda_x, lambda_y, lambda_z));
    
    // Build the reference vector X_ref and b
    for (PetscInt i = 0; i < n_x * n_y * n_z; i++) 
        VecSetValue( X_ref, i, i*i*i, INSERT_VALUES);
    VecAssemblyBegin(X_ref);
    VecAssemblyEnd(X_ref);

    PetscCall(MatMult(C, X_ref, b));

    // Solve the system
    Fft3DTransportSolver(n_x, n_y, n_z, a_x, a_y, a_z, dt, delta_x, delta_y, delta_z, X, b, FFT_MAT);

    // Compute relative Residual
    VecNorm(b, NORM_2, &b_norm);
    PetscCall(MatCreateVecs( C, NULL, &r));
#if defined(PETSC_USE_COMPLEX)
    MatMult(C, X, r); // r=Ax
    VecAXPY(r, -1, b); // r=Ax-b
#else
    Vec x_short, b_short;
    IS is;
   	ISCreateStride(PETSC_COMM_WORLD, N, 0, 1, &is);
    VecGetSubVector(X, is, &x_short);
    VecGetSubVector(b, is, &b_short);
    MatMult(C, x_short, r); // r=Ax
    VecAXPY(r, -1, b_short); // r=Ax-b
#endif
    VecNorm(r, NORM_2, &rnorm);
    PetscPrintf(PETSC_COMM_WORLD, "Relative residual = %e\n", rnorm / b_norm);

    // Compute relative error
    VecNorm(X_ref, NORM_2, &X_ref_norm);
#if defined(PETSC_USE_COMPLEX)
    VecAXPY(X_ref, -1, X); // X_ref -= X
#else
    Vec x_short;
    IS is;
   	ISCreateStride(PETSC_COMM_WORLD, N, 0, 1, &is);
    VecGetSubVector(X, is, &x_short);
    VecAXPY(X_ref, -1, x_short);
#endif
    VecNorm(X_ref, NORM_2, &enorm);
    PetscPrintf(PETSC_COMM_WORLD, "Relative error = %e\n", enorm / X_ref_norm);

    // Clean up
    PetscCall(VecDestroy(&b));
    PetscCall(VecDestroy(&X));
    PetscCall(VecDestroy(&X_ref));
    PetscCall(VecDestroy(&r));
    PetscCall(MatDestroy(&C));

    PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv) {
    PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));
    test_3D();
    PetscCall(PetscFinalize());
    return 0;
}