cmake_minimum_required(VERSION 3.12.0)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project("FFTPreconditioner")
set(PROJECT_BRIEF "A preconditioner based on  FFT")

list (APPEND CMAKE_MODULE_PATH "${FFTPreconditioner_SOURCE_DIR}/cmake")

option (FFTPreconditioner_WITH_PYTHON     "Build Python interface" ON)
option (FFTPreconditioner_WITH_TESTS      "Build and run example tests" ON)
option (FFTPreconditioner_WITH_MPI        "Generate a parallel version" OFF)

#Path to installed libraries
set (PYTHON_ROOT_DIR      OFF CACHE STRING "Python library path" )
set (PETSC_DIR            OFF CACHE STRING "PETSc library path" )
set (PETSC_ARCH           OFF CACHE STRING "PETSc architecture" )
set (MPI_HOME             OFF CACHE STRING "MPI library path" )
set (SOLVERLAB_DIR        OFF CACHE STRING "SOLVERLAB library path" )
set (MEDCOUPLING_DIR      OFF CACHE STRING "MEDCOUPLING library path" )
set (MEDFILE_DIR          OFF CACHE STRING "MEDFILE library path" )

set(MY_SOURCES
src/WaveSystem.cxx
src/TransportEquation.cxx
src/FftPrecond_3D.c
)
set(MY_HEADERS
${PROJECT_SOURCE_DIR}/src/
)

set(SOLVERLAB_INCLUDES
${SOLVERLAB_DIR}/include
${MEDCOUPLING_DIR}/include
)
set(SOLVERLAB_LIBS
${SOLVERLAB_DIR}/lib/libsolverlabMesh.so
${SOLVERLAB_DIR}/lib/libbase.so
${MEDCOUPLING_DIR}/lib/libmedloader.so
${MEDCOUPLING_DIR}/lib/libmedcoupling.so
${MEDCOUPLING_DIR}/lib/libinterpkernel.so
${MEDFILE_DIR}/lib/libmed.so
${MEDFILE_DIR}/lib/libmedC.so
)

find_package (PETSc 3.4 REQUIRED)

add_library(FFTPreconditionerLib SHARED ${MY_SOURCES})
message(STATUS ${PETSC_INCLUDES})
target_include_directories(FFTPreconditionerLib PRIVATE ${PETSC_INCLUDES}  ${SOLVERLAB_INCLUDES} ${MY_HEADERS})
target_link_libraries(     FFTPreconditionerLib         ${PETSC_LIBRARIES} ${SOLVERLAB_LIBS})
install(TARGETS FFTPreconditionerLib DESTINATION lib)

if( FFTPreconditioner_WITH_MPI )
  message(STATUS "FFTPreconditioner requested with MPI library. Searching MPI library.")
  if   ( NOT MPI_HOME)
    set( MPI_HOME ${PETSC_DIR}/${PETSC_ARCH} )
  endif( NOT MPI_HOME)
  find_package(MPI REQUIRED)
endif( FFTPreconditioner_WITH_MPI )

if( FFTPreconditioner_WITH_PYTHON )
  message(STATUS "FFTPreconditioner requested with PYTHON library. Searching PYTHON library.")
  find_package(Python ${PYTHON2OR3} REQUIRED COMPONENTS Interpreter Development )
  SET(PYTHON_EXECUTABLE ${Python_EXECUTABLE})
endif( FFTPreconditioner_WITH_PYTHON )

if( FFTPreconditioner_WITH_TESTS )
 enable_testing ()

 add_subdirectory(tests)
endif( FFTPreconditioner_WITH_TESTS )

