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

PetscErrorCode build_circulant_mat_1D( Mat C, PetscInt size, PetscScalar c1, PetscScalar c2) {
    PetscFunctionBeginUser;
    for (PetscInt i = 0; i < size; i++) 
        MatSetValue( C, i, (i+size-1)%size, c2, INSERT_VALUES);
    
    MatAssemblyBegin(C, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(C, MAT_FINAL_ASSEMBLY);

    MatShift(C,c1);
    PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode solve_circulant_system(Mat FFT_MAT, Vec c_col, Vec b, Vec x, Vec lambdas, Vec b_hat) {
    PetscFunctionBeginUser;
    const char *mat_type;
    PetscInt size;
    
    // Check if the input matrix is of type MATFFTW
    PetscCall(MatGetType(FFT_MAT, &mat_type));
    if (strcmp(mat_type, MATFFTW) != 0) {
        SETERRQ(PETSC_COMM_WORLD, PETSC_ERR_ARG_WRONG, "FFT_MAT must be a MATFFTW matrix");
    }

    // Compute the Fourier transform of the circulant column 
    PetscCall(MatMult(FFT_MAT, c_col, lambdas));

    // Compute the Fourier transform of the right-hand side vector
    PetscCall(MatMult(FFT_MAT, b, b_hat));

    PetscPrintf(PETSC_COMM_WORLD, "Premiere colonne de la matrice circulante :\n");
    PetscCall(VecView(c_col, PETSC_VIEWER_STDOUT_WORLD));
    PetscPrintf(PETSC_COMM_WORLD, "\n Valeurs propres lambdas\n");
    PetscCall(VecView( lambdas, PETSC_VIEWER_STDOUT_WORLD));
    PetscPrintf(PETSC_COMM_WORLD, "\n Vecteur second membre b:\n");
    PetscCall(VecView(b, PETSC_VIEWER_STDOUT_WORLD));
    PetscPrintf(PETSC_COMM_WORLD, "\n Vecteur second membre b_hat:\n");
    PetscCall(VecView(b_hat, PETSC_VIEWER_STDOUT_WORLD));

    // Compute the solution in the Fourier domain
#if defined(PETSC_USE_COMPLEX)
        PetscCall(VecPointwiseDivide(b_hat, b_hat, lambdas));
#else
        PetscCall(VecPointwiseDivideForRealFFT(b_hat, b_hat, lambdas));
#endif

    PetscPrintf(PETSC_COMM_WORLD, "\n Vecteur second membre b_hat/lambdas:\n");
    PetscCall(VecView(b_hat, PETSC_VIEWER_STDOUT_WORLD));

    // Compute the inverse Fourier transform to get the solution in the original space
    PetscCall(MatMultTranspose(FFT_MAT, b_hat, x));

    // Suppose that c_col, b, and x have the same size
    PetscCall(VecGetSize(c_col, &size));
#if defined(PETSC_USE_COMPLEX)
    VecScale(x,1./size);
#else
    VecScale(x,2./size);
#endif
    
    PetscPrintf(PETSC_COMM_WORLD, "\n Vecteur solution x:\n");
    PetscCall(VecView(x, PETSC_VIEWER_STDOUT_WORLD));

    PetscFunctionReturn(PETSC_SUCCESS);
}

int main(int argc, char **argv) {
    PetscInt N = 4;
    PetscScalar lambda = 0.5;
    Vec c, b;// input vectors of forward fft
    Vec lambdas, b_hat;// output vectors of forward fft = input vectors of backward fft
    Vec x;// output vectors of backward fft
    Vec r;// residual vector r=AX-b
    PetscReal rnorm;// residual norm
    Mat FFT_MAT, C;
    PetscInt ndim = 1;
    PetscInt dims[1];

    PetscCall(PetscInitialize(&argc, &argv, NULL, NULL));

    dims[0] = N;
    PetscCall(MatCreateFFT(PETSC_COMM_WORLD, ndim, dims, MATFFTW, &FFT_MAT));
    
    PetscCall(MatCreateVecsFFTW( FFT_MAT, &c, &lambdas, &x));
    PetscCall(MatCreateVecsFFTW( FFT_MAT, &b, &b_hat  , NULL));

    PetscCall(VecSet(c, 0.0)); // Initialize c to zero
    PetscCall(VecSetValue( c, 0, 1+lambda, INSERT_VALUES));
    PetscCall(VecSetValue( c, 1, -lambda , INSERT_VALUES));

    VecAssemblyBegin(c);
    VecAssemblyEnd(c);

    //PetscCall(VecSet(b, 1.0); // Set b to a constant vector for testing
    for (PetscInt i = 0; i < N; i++) 
        VecSetValue( b, i, i*i*i, INSERT_VALUES);

    VecAssemblyBegin(b);
    VecAssemblyEnd(b);

    PetscCall(solve_circulant_system(FFT_MAT, c, b, x, lambdas, b_hat));
    
    /* Calcul du résidu r */
    MatCreateAIJ( PETSC_COMM_WORLD,  PETSC_DECIDE ,  PETSC_DECIDE , N, N, 2, NULL, 1, NULL, &C);    //remplacer  PETSC_DECIDE par la taille locale des vecteurs b et/ou c
    build_circulant_mat_1D( C, N, 1+lambda, -lambda);
    PetscCall(MatCreateVecs( C, NULL, &r));

    /* Affichages debug */
    PetscPrintf(PETSC_COMM_WORLD, "\n Matrice circulante C:\n");
    MatView( C, PETSC_VIEWER_STDOUT_WORLD);
    
#if defined(PETSC_USE_COMPLEX)
    MatMult(C, x, r);//r=Ax
    VecAXPY(r, -1, b);//r=Ax-b
#else
    Vec x_short, b_short;
    IS is;
    ISCreateStride(PETSC_COMM_WORLD, N, 0, 1, &is);
    VecGetSubVector(x, is, &x_short);
    VecGetSubVector(b, is, &b_short);
    MatMult(C, x_short, r);//r=Ax
    VecAXPY(r, -1, b_short);//r=Ax-b
#endif

    VecNorm(r, NORM_2, &rnorm);
    PetscPrintf(PETSC_COMM_WORLD, "\n norme du résidu = %e\n", rnorm );
    assert(rnorm<1);
    
 // Clean up
    PetscCall(VecDestroy(&b));
    PetscCall(VecDestroy(&x));
    PetscCall(VecDestroy(&c));
    PetscCall(VecDestroy(&lambdas));
    PetscCall(VecDestroy(&b_hat));
    PetscCall(MatDestroy(&FFT_MAT));
    PetscCall(MatDestroy(&C));
    PetscCall(PetscFinalize());
    return 0;
}
