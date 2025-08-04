# CirculantPreconditioner_SRC

This project provides tools and scripts for testing and analyzing simulation on general meshes based on circulant matrix preconditoiners.

The main prerequisites are PETSc for the handling of matrices and MEDCoupling for the handling of meshes (structured and unstructured).
PETSc should be compiled with the external library FFTW in order to allow the efficient manipulation of circulant matrices. Indeed the product of a circulant matrix by a vector can be accelerated with the use of the FFT algorithm.  

For the moment SOLVERLAB is mandatory but we hope to remove this dependency soon  

Python is optional, as well as MPI for parallel execution.

After cloning, the typical installation is done via the commands
'cmake /path/to/SOURCE/DIR -DCMAKE_INSTALL_PREFIX=/path/to/INSTALL/DIR  -DCMAKE_BUILD_TYPE=Release -DSaddlePointLinearSolver_WITH_PYTHON=ON -DSaddlePointLinearSolver_WITH_TESTS=ON -DSaddlePointLinearSolver_WITH_MPI=ON -DPETSC_DIR=/path/to/PETSC/DIR -DPETSC_ARCH=arch-linux-c-opt -DSOLVERLAB_DIR=/volatile/catA/ndjinga/Logiciels/Solverlab/INSTALL_mpi -DMEDCOUPLING_DIR=/volatile/catA/ndjinga/Logiciels/Salome/medCoupling-9.10.0_install_par -DMEDFILE_DIR=/volatile/catA/ndjinga/Logiciels/Salome/med-4.1.0_install_par'

'make'

'make install'

The parameter PETSC_ARCH should correspond to your PETSc installation (usually arch-linux-c-opt or arch-linux-c-opt on linux computers).


