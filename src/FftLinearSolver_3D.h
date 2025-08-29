#ifndef FFT_PRECOND_3D_H
#define FFT_PRECOND_3D_H

#include <petscmat.h>
#include <petscvec.h>

struct StructuredTransportContext {
    PetscInt n_x;
    PetscInt n_y;
    PetscInt n_z;
    PetscScalar a_x;
    PetscScalar a_y;
    PetscScalar a_z;
    PetscScalar dt;
    PetscScalar delta_x;
    PetscScalar delta_y;
    PetscScalar delta_z;
    Mat FFT_MAT;
};

PetscErrorCode Fft3DTransportSolver(PetscInt n_x, PetscInt n_y, PetscInt n_z,
    PetscScalar a_x, PetscScalar a_y, PetscScalar a_z, PetscScalar dt,
    PetscScalar delta_x, PetscScalar delta_y, PetscScalar delta_z,
    Vec X, Vec b, Mat FFT_MAT);

PetscErrorCode Fft2DTransportSolver(PetscInt n_x, PetscInt n_y,
    PetscScalar a_x, PetscScalar a_y, PetscScalar dt,
    PetscScalar delta_x, PetscScalar delta_y,
    Vec X, Vec b, Mat FFT_MAT);

PetscErrorCode Fft1DTransportSolver(PetscInt n_x, PetscScalar a_x, PetscScalar dt, PetscScalar delta_x, Vec X, Vec b, Mat FFT_MAT);

PetscErrorCode PetscFft3DTransportSolver(struct StructuredTransportContext customCtx , Vec b, Vec x);

PetscErrorCode FftTransportSolver(PetscInt n_x, PetscInt n_y, PetscInt n_z,
    PetscScalar lambda_x, PetscScalar lambda_y, PetscScalar lambda_z,
    Vec X, Vec b, Mat FFT_MAT);


#endif // FFT_PRECOND_3D_H
