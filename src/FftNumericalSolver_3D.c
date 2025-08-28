#include <petscmat.h>
#include <petscvec.h>

#include "FftNumericalSolver_3D.h"

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

PetscErrorCode build_transport_col(Vec c, PetscInt size) {
    PetscFunctionBeginUser;
    VecSet(c, 0.0);
    VecSetValue(c, 0, 1.0, INSERT_VALUES);
    VecSetValue(c, 1, -1.0, INSERT_VALUES);
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

PetscErrorCode build_diag_mat_vec_3D(Vec Diag, Vec c_x_hat, Vec c_y_hat, Vec c_z_hat, PetscInt n_x, PetscInt n_y, PetscInt n_z, PetscScalar lambda_x, PetscScalar lambda_y, PetscScalar lambda_z) {
    PetscFunctionBeginUser;
    Vec sum;

    Vec kpi_x, kpi_y, kpi_z, kpi_y_intermediate;
    VecDuplicate(Diag, &kpi_x);
    VecDuplicate(Diag, &kpi_y);
    VecDuplicate(Diag, &kpi_y_intermediate);
    VecDuplicate(Diag, &kpi_z);

    vec_kronecker_product_identity_left(c_x_hat, kpi_x, n_x, n_y * n_z, lambda_x);
    vec_kronecker_product_identity_left(c_y_hat, kpi_y_intermediate, n_y, n_z, lambda_y);
    vec_kronecker_product_identity_right(kpi_y_intermediate, kpi_y, n_y * n_z, n_x, 1);
    vec_kronecker_product_identity_right(c_z_hat, kpi_z, n_z, n_x * n_y, lambda_z);

    PetscCall(VecDuplicate(kpi_x, &sum));
    PetscCall(VecAXPY(sum, 1.0, kpi_x));
    PetscCall(VecAXPY(sum, 1.0, kpi_y));
    PetscCall(VecAXPY(sum, 1.0, kpi_z));
    PetscCall(VecShift(sum, 1.0));

    PetscCall(VecCopy(sum, Diag));

    PetscCall(VecDestroy(&kpi_x));
    PetscCall(VecDestroy(&kpi_y));
    PetscCall(VecDestroy(&kpi_z));
    PetscCall(VecDestroy(&sum));
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode solve_3D(Mat FFT_MAT, Vec X, Vec Diag, Vec b, Vec b_hat) {
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

PetscErrorCode Fft3DSolver(PetscInt n_x, PetscInt n_y, PetscInt n_z,
    PetscScalar a_x, PetscScalar a_y, PetscScalar a_z, PetscScalar dt,
    PetscScalar delta_x, PetscScalar delta_y, PetscScalar delta_z,
    Vec X, Vec b, Mat FFT_MAT, Vec c_x_hat, Vec c_y_hat, Vec c_z_hat) {
    
    PetscFunctionBeginUser;

    // Variables
    Vec b_hat; // fft of b
    Vec Diag; // vector of the diagonal matrix

    // Input parameters
    PetscScalar lambda_x = a_x * dt / delta_x;
    PetscScalar lambda_y = a_y * dt / delta_y;
    PetscScalar lambda_z = a_z * dt / delta_z;
    
    // Initialize vectors
    PetscCall(MatCreateVecsFFTW( FFT_MAT, NULL, &Diag, NULL));
    PetscCall(MatCreateVecsFFTW( FFT_MAT, NULL, &b_hat, NULL));

    // Solve the system
    build_diag_mat_vec_3D(Diag, c_x_hat, c_y_hat, c_z_hat, n_x, n_y, n_z, lambda_x, lambda_y, lambda_z);
    solve_3D(FFT_MAT, X, Diag, b, b_hat);

    // Clean up
    PetscCall(VecDestroy(&Diag));
    PetscCall(VecDestroy(&b_hat));
    PetscCall(MatDestroy(&FFT_MAT));

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Fft3DTransportSolver(PetscInt n_x, PetscInt n_y, PetscInt n_z,
    PetscScalar a_x, PetscScalar a_y, PetscScalar a_z, PetscScalar dt,
    PetscScalar delta_x, PetscScalar delta_y, PetscScalar delta_z,
    Vec X, Vec b, Mat FFT_MAT) {
    
    PetscFunctionBeginUser;

    Vec c_x, c_y, c_z;
    Vec c_x_hat, c_y_hat, c_z_hat;
    Mat FFT_cx, FFT_cy, FFT_cz;
    
    PetscInt ndim_x = 1;
    PetscInt ndim_y = 1;
    PetscInt ndim_z = 1;
    PetscInt dims_x[1] = {n_x};
    PetscInt dims_y[1] = {n_y};
    PetscInt dims_z[1] = {n_z};

    PetscCall(MatCreateFFT(PETSC_COMM_WORLD, ndim_x, dims_x, MATFFTW, &FFT_cx));
    PetscCall(MatCreateFFT(PETSC_COMM_WORLD, ndim_y, dims_y, MATFFTW, &FFT_cy));
    PetscCall(MatCreateFFT(PETSC_COMM_WORLD, ndim_z, dims_z, MATFFTW, &FFT_cz));

    PetscCall(MatCreateVecsFFTW(FFT_cx, &c_x, &c_x_hat, NULL));
    PetscCall(MatCreateVecsFFTW(FFT_cy, &c_y, &c_y_hat, NULL));
    PetscCall(MatCreateVecsFFTW(FFT_cz, &c_z, &c_z_hat, NULL));

    build_transport_col(c_x, n_x);
    build_transport_col(c_y, n_y);
    build_transport_col(c_z, n_z);

    MatMult(FFT_cx, c_x, c_x_hat);
    MatMult(FFT_cy, c_y, c_y_hat);
    MatMult(FFT_cz, c_z, c_z_hat);
    
    PetscCall(Fft3DSolver(n_x, n_y, n_z, a_x, a_y, a_z, dt, delta_x, delta_y, delta_z, X, b, FFT_MAT, c_x_hat, c_y_hat, c_z_hat));
    
    PetscCall(VecDestroy(&c_x));
    PetscCall(VecDestroy(&c_y));
    PetscCall(VecDestroy(&c_z));

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Fft2DTransportSolver(PetscInt n_x, PetscInt n_y,
    PetscScalar a_x, PetscScalar a_y, PetscScalar dt,
    PetscScalar delta_x, PetscScalar delta_y,
    Vec X, Vec b, Mat FFT_MAT) {
    
    PetscFunctionBeginUser;

    PetscCall(Fft3DTransportSolver(n_x, n_y, 1, a_x, a_y, 0, dt, delta_x, delta_y, 1, X, b, FFT_MAT));
    
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode Fft1DTransportSolver(PetscInt n_x, PetscScalar a_x, PetscScalar dt, PetscScalar delta_x, Vec X, Vec b, Mat FFT_MAT) {
    PetscFunctionBeginUser;

    PetscCall(Fft3DTransportSolver(n_x, 1, 1, a_x, 0, 0, dt, delta_x, 1, 1, X, b, FFT_MAT));

    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode PetscFft3DTransportSolver(struct StructuredTransportContext customCtx , Vec b, Vec x) {
    PetscFunctionBeginUser;
    
    PetscCall(Fft3DTransportSolver(customCtx.n_x, customCtx.n_y, customCtx.n_z,
                         customCtx.a_x, customCtx.a_y, customCtx.a_z, customCtx.dt,
                         customCtx.delta_x, customCtx.delta_y, customCtx.delta_z,
                         x, b, customCtx.FFT_MAT));

    PetscFunctionReturn(PETSC_SUCCESS);
}
