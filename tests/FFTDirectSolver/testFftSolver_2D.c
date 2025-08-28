#include <petscmat.h>
#include <petscvec.h>

/* The function VecPointwiseDivideForRealFFT should be compiled only when petsc was buil with real numbers (not complex numbers) */
#if !defined(PETSC_USE_COMPLEX)
PetscErrorCode buildComplexRatio( PetscInt i, Vec v2, Vec v3, PetscReal* z1r, PetscReal* z1i) {
    PetscFunctionBeginUser;
    PetscInt idx;
    PetscReal   z2r, z3r, z2i, z3i, z3_2;

    idx = i;
    VecGetValues(v2,1,&idx,&z2r);
    VecGetValues(v3,1,&idx,&z3r);
    idx = i+1;
    VecGetValues(v2,1,&idx,&z2i);
    VecGetValues(v3,1,&idx,&z3i);

    z3_2 = z3r*z3r + z3i*z3i;
    *z1r = (z2r*z3r + z2i*z3i)/z3_2;
    *z1i = (z2i*z3r - z2r*z3i)/z3_2;
    
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode VecPointwiseDivideForRealFFT( Vec v1, Vec v2, Vec v3) {
    PetscFunctionBeginUser;
    PetscErrorCode ierr = PETSC_ERR_ARG_WRONG;
    PetscLayout map1, map2, map3;
    PetscBool v1v2Compatible, v2v3Compatible;
    PetscInt row_min, row_max, size, idx, istart, iend;
    PetscReal    z1r, z1i;
    
    PetscCall(VecGetLayout(v1,&map1));
    PetscCall(VecGetLayout(v2,&map2));
    PetscCall(VecGetLayout(v3,&map3));
    
    PetscCall(PetscLayoutCompare( map1, map2, &v1v2Compatible));
    PetscCall(PetscLayoutCompare( map2, map3, &v2v3Compatible));
    
    PetscCheck( v2v3Compatible, PETSC_COMM_WORLD, ierr, "!!!!!! Input vectors v2 and v3 should have the same size and parallel distribution !");
    PetscCheck( v1v2Compatible, PETSC_COMM_WORLD, ierr, "!!!!!! Input v1 and output v2 vectors should have the same size and parallel distribution !");

    PetscCall(VecGetSize(v1,&size));
    PetscCheck( size%2==0,   PETSC_COMM_WORLD, ierr, "!!!!!! Vector sizes should be even !");
    
    PetscCall(VecGetOwnershipRange(v2, &row_min, &row_max));
    
    //if irow_min and irow_max are even, no communication is involved
    istart = 2*((row_min+1)/2);//smallest even integer above row_min
    iend   = 2*( row_max  /2);//largest  even integer below row_max

    for( PetscInt i = istart ; i < iend && i < 2*(size/4+1) ; i+=2)//loop with no communication involved, size/4+1 complex numbers are stored in the real fft vector
    {
        buildComplexRatio( i, v2, v3, &z1r, &z1i);//computes z2/z3
        idx = i;
        VecSetValues(v1,1,&idx,&z1r,INSERT_VALUES);
        idx = i+1;
        VecSetValues(v1,1,&idx,&z1i,INSERT_VALUES);
    }
    //Now deal with communications between procs if row_min or row_max is odd
    if( row_min < istart  && row_min-1 < 2*(size/4+1) )//row_min is odd, need real part located at irow_min-1 on another proc
    {
        buildComplexRatio( row_min-1, v2, v3, &z1r, &z1i);//computes z2/z3
        idx = row_min;
        VecSetValues(v1,1,&idx,&z1i,INSERT_VALUES);        
    }
    if( row_max > iend  && row_max-1 < 2*(size/4+1) )//row_max is odd, need imaginary part located at irow_max on another proc
    {
        buildComplexRatio( row_max-1, v2, v3, &z1r, &z1i);//computes z2/z3
        idx = row_max-1;
        VecSetValues(v1,1,&idx,&z1r,INSERT_VALUES);
    }
    
    PetscFunctionReturn(PETSC_SUCCESS);
}
#endif

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

PetscErrorCode build_identity_matrix(Mat Id, int n) {
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
    PetscCall(build_identity_matrix(Id_nx, n_x));

    PetscCall(MatCreateAIJ(PETSC_COMM_WORLD, n_y, n_y, n_y, n_y, 2, NULL, 1, NULL, &Id_ny));
    PetscCall(build_identity_matrix(Id_ny, n_y));

    PetscCall(MatCreateAIJ(PETSC_COMM_WORLD, n_x * n_y, n_x * n_y, n_x * n_y, n_x * n_y, 2, NULL, 1, NULL, &Id_nx_ny));
    PetscCall(build_identity_matrix(Id_nx_ny, n_x * n_y));

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

PetscErrorCode solve_2D(Mat FFT_MAT, Vec X, Vec Diag, Vec b, Vec b_hat) {
    PetscFunctionBeginUser;
    PetscInt size;

    // FFT
    PetscCall(MatMult(FFT_MAT, b, b_hat));

    // b_hat / Diag
#if defined(PETSC_USE_COMPLEX)
    PetscCall(VecPointwiseDivide(b_hat, b_hat, Diag));
#else
    PetscCall(VecPointwiseDivideForRealFFT(b_hat, b_hat, Diag));
#endif

    // IFFT
    PetscCall(MatMultTranspose(FFT_MAT, b_hat, X));

    // Normalize
    PetscCall(VecGetSize(X, &size));
#if defined(PETSC_USE_COMPLEX)
    VecScale(X,1./size);
#else
    VecScale(X,2./size);
#endif

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode test_2D() {
    PetscFunctionBeginUser;

    // Input parameters
    PetscInt n_x = 3;
    PetscInt n_y = 2;
    PetscScalar a_x = 1;
    PetscScalar a_y = 1;
    PetscScalar dt = 1;
    PetscScalar delta_x = 1;
    PetscScalar delta_y = 1;
    PetscScalar lambda_x = a_x * dt / delta_x;
    PetscScalar lambda_y = a_y * dt / delta_y;

    // Variables
    Mat C; // Block-circulant matrix
    Vec X_ref, X, b; // CX=b
    Mat FFT_MAT;
    Vec b_hat; // fft of b
    Vec Diag; // vector of the diagonal matrix

    Vec r;// residual vector r=AX-b
    PetscReal rnorm, enorm;// residual norm and error
    PetscReal X_ref_norm, b_norm;

    // Initialize FFT 2D
    PetscInt ndim = 2;
    PetscInt dims[2] = {n_y, n_x};
    PetscCall(MatCreateFFT(PETSC_COMM_WORLD, ndim, dims, MATFFTW, &FFT_MAT));
    
    // Initialize vectors
    PetscCall(MatCreateVecsFFTW( FFT_MAT, NULL, &Diag, &X_ref));
    PetscCall(MatCreateVecsFFTW( FFT_MAT, &b, &b_hat  , &X));

    // Build the block-circulant matrix C
    PetscCall(MatCreateAIJ(PETSC_COMM_WORLD, n_x * n_y, n_x * n_y, n_x * n_y, n_x * n_y, 2, NULL, 1, NULL, &C));
    PetscCall(build_C_2D(&C, n_x, n_y, lambda_x, lambda_y));
    
    // Build the reference vector X_ref and b
    for (PetscInt i = 0; i < n_x * n_y; i++) 
        VecSetValue( X_ref, i, i*i*i, INSERT_VALUES);
    VecAssemblyBegin(X_ref);
    VecAssemblyEnd(X_ref);

    PetscCall(MatMult(C, X_ref, b));

    // Solve the system
    build_diag_mat_vec_2D(Diag, n_x, n_y, lambda_x, lambda_y);
    solve_2D(FFT_MAT, X, Diag, b, b_hat);

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
    assert( rnorm / b_norm<1);

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
    assert( enorm / X_ref_norm);

    // Clean up
    PetscCall(VecDestroy(&b));
    PetscCall(VecDestroy(&X));
    PetscCall(VecDestroy(&X_ref));
    PetscCall(VecDestroy(&b_hat));
    PetscCall(VecDestroy(&Diag));
    PetscCall(VecDestroy(&r));
    PetscCall(MatDestroy(&FFT_MAT));
    PetscCall(MatDestroy(&C));

    PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv) {
    PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));
    test_2D();
    PetscCall(PetscFinalize());
    return 0;
}
