#ifndef PCShellFFT_3D_H
#define PCShellFFT_3D_H

#include <petscksp.h>

#include "Mesh.hxx"

struct FFTPrecTransportContext {
    PetscInt spaceDim;
    PetscInt n_x;
    PetscInt n_y;
    PetscInt n_z;
    PetscScalar lambda_x;
    PetscScalar lambda_y;
    PetscScalar lambda_z;
    Mat FFT_MAT;
    Mat intersectionMatrix;
    Vec Diag;
    Vec b_hat;
    Vec b_cartesien;
};

PetscErrorCode applyFFT3DPrecTransport(PC pc, Vec b, Vec x);
PetscErrorCode setupFFTPrec3D(PC pc);
PetscErrorCode destroyFFTPrec3D(PC pc);

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
    );

PetscErrorCode computeDiagonalMatrixTransport(PetscInt n_x        , PetscInt n_y        , PetscInt n_z,
                                              PetscScalar lambda_x, PetscScalar lambda_y, PetscScalar lambda_z,
                                              Vec X, Vec b, Mat FFT_MAT);

    
#endif // PCShellFFT_3D_H
