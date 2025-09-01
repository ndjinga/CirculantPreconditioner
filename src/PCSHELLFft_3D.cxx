#include "PCSHELLFft_3D.hxx"
//#include "MEDCouplingRemapper.hxx"
extern "C" {
    #include "FftLinearSolver_3D.h"
}

    //MatCreateShell(PETSC_COMM_WORLD, PETSC_DECIDE, PETSC_DECIDE, globalNbUnknowns, globalNbUnknowns, &ctx, &A);
    //MatShellSetOperation(A, MATOP_MULT, (void(*)(void))PetscFft3DTransportSolver);
    
PetscErrorCode applyFFT3DPrecTransport(PC pc, Vec b, Vec x)
{
  PetscFunctionBeginUser;

  FFTPrecTransportContext *ctx;
  PetscCall(PCShellGetContext( pc, ctx));

  //transfer b on structured mesh with intersection matrix
  MatMult( ctx->intersectionMatrix, b, ctx->b_cartesien );
  
  //Solve the linear system
  solve_3D( ctx->FFT_MAT, x, ctx->Diag, ctx->b_cartesien, ctx->b_hat, ctx->n_x*ctx->n_y*ctx->n_z );

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode setupFFTPrec3D(PC pc)
{
  PetscFunctionBeginUser;

  FFTPrecTransportContext *ctx;
  PCShellGetContext( pc, ctx);

  //Create FFT matrix, allocate diagonal matrix Lambda and RHS vector
  PetscInt dims[3] = {ctx->n_z, ctx->n_y, ctx->n_x};
  PetscCall(MatCreateFFT(PETSC_COMM_WORLD, ctx->spaceDim, dims, MATFFTW, &(ctx->FFT_MAT)));
  PetscCall(MatCreateVecsFFTW( (ctx->FFT_MAT), NULL               , &(ctx->Diag) , NULL));
  PetscCall(MatCreateVecsFFTW( (ctx->FFT_MAT), &(ctx->b_cartesien), &(ctx->b_hat), NULL));
  
  /* Create the 1D FFTs and Allocate the column vectors c_x, c_y and c_z */
    Vec c_x, c_y, c_z;
    Vec c_x_hat, c_y_hat, c_z_hat;
    Mat FFT_cx, FFT_cy, FFT_cz;
    
    PetscInt ndim_x = 1;
    PetscInt ndim_y = 1;
    PetscInt ndim_z = 1;
    PetscInt dims_x[1] = {ctx->n_x};
    PetscInt dims_y[1] = {ctx->n_y};
    PetscInt dims_z[1] = {ctx->n_z};
    
    PetscCall(MatCreateFFT(PETSC_COMM_WORLD, ndim_x, dims_x, MATFFTW, &FFT_cx));
    PetscCall(MatCreateFFT(PETSC_COMM_WORLD, ndim_y, dims_y, MATFFTW, &FFT_cy));
    PetscCall(MatCreateFFT(PETSC_COMM_WORLD, ndim_z, dims_z, MATFFTW, &FFT_cz));
    
    PetscCall(MatCreateVecsFFTW(FFT_cx, &c_x, &c_x_hat, NULL));
    PetscCall(MatCreateVecsFFTW(FFT_cy, &c_y, &c_y_hat, NULL));
    PetscCall(MatCreateVecsFFTW(FFT_cz, &c_z, &c_z_hat, NULL));
    
  /* Fill the diagonal matrix Lambda from the 1D FFTs and the column vectors c_x, c_y and c_z */
    build_transport_col(c_x, ctx->n_x);
    build_transport_col(c_y, ctx->n_y);
    build_transport_col(c_z, ctx->n_z);
    
    MatMult(FFT_cx, c_x, c_x_hat);
    MatMult(FFT_cy, c_y, c_y_hat);
    MatMult(FFT_cz, c_z, c_z_hat);
    

  build_diag_mat_vec_3D(ctx->Diag, c_x_hat, c_y_hat, c_z_hat, ctx->n_x, ctx->n_y, ctx->n_z, ctx->lambda_x, ctx->lambda_y, ctx->lambda_z);

  //Initialize remapper and extract crude matrix

    PetscCall(VecDestroy(&c_x));
    PetscCall(VecDestroy(&c_y));
    PetscCall(VecDestroy(&c_z));
    PetscCall(VecDestroy(&c_x_hat));
    PetscCall(VecDestroy(&c_y_hat));
    PetscCall(VecDestroy(&c_z_hat));
    PetscCall(MatDestroy(&FFT_cx));
    PetscCall(MatDestroy(&FFT_cy));
    PetscCall(MatDestroy(&FFT_cz));

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode destroyFFTPrec3D(PC pc)
{
  PetscFunctionBeginUser;

  FFTPrecTransportContext *ctx;
  PCShellGetContext( pc, ctx);

  PetscCall(VecDestroy(&ctx->Diag));
  PetscCall(VecDestroy(&ctx->b_cartesien));
  PetscCall(VecDestroy(&ctx->b_hat));
  PetscCall(MatDestroy(&ctx->FFT_MAT));

  PetscFunctionReturn(PETSC_SUCCESS);
}

PetscErrorCode getFFTPrec3DContext(
    PetscInt ndim,
    PetscScalar dt,
    PetscInt nbCells,
    PetscScalar a_x,
    PetscScalar a_y,
    PetscScalar a_z,
    PetscScalar Xmin,
    PetscScalar Ymin,
    PetscScalar Zmin,
    PetscScalar Xmax,
    PetscScalar Ymax,
    PetscScalar Zmax,
    Mesh srcMesh
    )
{
  FFTPrecTransportContext *ctx;
  PetscInt n_x, n_y, n_z;
  
  PetscCheck( ndim>0 && ndim<4, PETSC_COMM_WORLD, PETSC_ERR_ARG_OUTOFRANGE, "Dimnension should be 1, 2 or 3");
  
  if( ndim ==3 )
  {
    n_x = floor( cbrt(nbCells) ) ;
    n_y = n_x;
    n_z = n_x;
  }
  else  if( ndim ==2 )
  {
    n_x = floor( sqrt(nbCells) ) ;
    n_y = n_x;
    n_z = 1;
  }
  else  if( ndim ==1 )
  {
    n_x = nbCells;
    n_y = 1;
    n_z = 1;
  }

  ctx->spaceDim = ndim;
  ctx->n_x  = n_x;
  ctx->n_y  = n_y;
  ctx->n_z  = n_z;
  
  ctx->lambda_x = a_x * dt * (Xmax - Xmin) / n_x;
  ctx->lambda_y = a_y * dt * (Ymax - Ymin) / n_y;
  ctx->lambda_z = a_z * dt * (Zmax - Zmin) / n_z;

  PetscFunctionReturn(PETSC_SUCCESS);
}


