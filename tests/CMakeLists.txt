project(testC)

INCLUDE_DIRECTORIES(
  ${PETSC_INCLUDES}	
  ${SOLVERLAB_INCLUDES} 
  ${MY_HEADERS}										    #
)

 set(MY_TEST_SOURCES
 tests/WaveSystem_SphericalExplosion_seq.cxx
 tests/WaveSystem_SphericalExplosion_mpi.cxx
 )
message(STATUS ${MY_HEADERS})

 # build test main exe
 add_executable(WaveSystem_SphericalExplosion_expl_seq                            ./WaveSystem_SphericalExplosion_expl_seq.cxx)
 add_executable(WaveSystem_SphericalExplosion_expl_mpi                            ./WaveSystem_SphericalExplosion_expl_mpi.cxx)
 add_executable(WaveSystem_SphericalExplosion_impl_seq                            ./WaveSystem_SphericalExplosion_impl_seq.cxx)
 add_executable(WaveSystem_SphericalExplosion_impl_mpi                            ./WaveSystem_SphericalExplosion_impl_mpi.cxx)
 target_include_directories(WaveSystem_SphericalExplosion_expl_seq                 PRIVATE ${PETSC_INCLUDES} ${SOLVERLAB_INCLUDES}   ${MY_HEADERS})
 target_include_directories(WaveSystem_SphericalExplosion_expl_mpi                 PRIVATE ${PETSC_INCLUDES} ${SOLVERLAB_INCLUDES}   ${MY_HEADERS})
 target_include_directories(WaveSystem_SphericalExplosion_impl_seq                 PRIVATE ${PETSC_INCLUDES} ${SOLVERLAB_INCLUDES}   ${MY_HEADERS})
 target_include_directories(WaveSystem_SphericalExplosion_impl_mpi                 PRIVATE ${PETSC_INCLUDES} ${SOLVERLAB_INCLUDES}   ${MY_HEADERS})
 target_link_libraries(WaveSystem_SphericalExplosion_expl_seq                       ${PETSC_LIBRARIES} ${SOLVERLAB_LIBS} ${MPI_HOME}/lib/libmpi.so CirculantPreconditionerLib )
 target_link_libraries(WaveSystem_SphericalExplosion_expl_mpi                       ${PETSC_LIBRARIES} ${SOLVERLAB_LIBS} ${MPI_HOME}/lib/libmpi.so CirculantPreconditionerLib )
 target_link_libraries(WaveSystem_SphericalExplosion_impl_seq                       ${PETSC_LIBRARIES} ${SOLVERLAB_LIBS} ${MPI_HOME}/lib/libmpi.so CirculantPreconditionerLib )
 target_link_libraries(WaveSystem_SphericalExplosion_impl_mpi                       ${PETSC_LIBRARIES} ${SOLVERLAB_LIBS} ${MPI_HOME}/lib/libmpi.so CirculantPreconditionerLib )

 add_test(NAME testWaveSystem_SphericalExplosion_expl_seq_2D_SQUARE_squares         COMMAND "./WaveSystem_SphericalExplosion_expl_seq"  )#Default is cartesian mesh 
 add_test(NAME testWaveSystem_SphericalExplosion_expl_seq_3D_CUBE_tetras            COMMAND "./WaveSystem_SphericalExplosion_expl_seq" "${PROJECT_SOURCE_DIR}/../meshes/meshCube.med")     
 add_test(NAME testWaveSystem_SphericalExplosion_expl_mpi_2D_SQUARE_squares         COMMAND "./WaveSystem_SphericalExplosion_expl_mpi"  )#Default is cartesian mesh
 add_test(NAME testWaveSystem_SphericalExplosion_expl_mpi_3D_CUBE_tetras            COMMAND "./WaveSystem_SphericalExplosion_expl_mpi" "${PROJECT_SOURCE_DIR}/../meshes/meshCube.med")     
 add_test(NAME testWaveSystem_SphericalExplosion_impl_seq_2D_SQUARE_squares         COMMAND "./WaveSystem_SphericalExplosion_impl_seq"  )#Default is cartesian mesh 
 add_test(NAME testWaveSystem_SphericalExplosion_impl_seq_3D_CUBE_tetras            COMMAND "./WaveSystem_SphericalExplosion_impl_seq" "${PROJECT_SOURCE_DIR}/../meshes/meshCube.med")     
 add_test(NAME testWaveSystem_SphericalExplosion_impl_mpi_2D_SQUARE_squares         COMMAND "./WaveSystem_SphericalExplosion_impl_mpi"  )#Default is cartesian mesh
 add_test(NAME testWaveSystem_SphericalExplosion_impl_mpi_3D_CUBE_tetras            COMMAND "./WaveSystem_SphericalExplosion_impl_mpi" "${PROJECT_SOURCE_DIR}/../meshes/meshCube.med")     

 set_tests_properties( 
 testWaveSystem_SphericalExplosion_expl_seq_2D_SQUARE_squares 
 testWaveSystem_SphericalExplosion_expl_seq_3D_CUBE_tetras
 testWaveSystem_SphericalExplosion_expl_mpi_2D_SQUARE_squares
 testWaveSystem_SphericalExplosion_expl_mpi_3D_CUBE_tetras
 testWaveSystem_SphericalExplosion_impl_seq_2D_SQUARE_squares
 testWaveSystem_SphericalExplosion_impl_seq_3D_CUBE_tetras
 testWaveSystem_SphericalExplosion_impl_mpi_2D_SQUARE_squares
 testWaveSystem_SphericalExplosion_impl_mpi_3D_CUBE_tetras
 PROPERTIES ENVIRONMENT "LD_LIBRARY_PATH=${SOLVERLAB_DIR}/lib:${MEDCOUPLING_DIR}/lib:${MEDFILE_DIR}/lib"
 )

if( CirculantPreconditioner_WITH_MPI )
 add_test(NAME testWaveSystem_SphericalExplosion_expl_mpi_2D_SQUARE_squares_2procs  COMMAND "${MPIEXEC}" "-n" "2" "./WaveSystem_SphericalExplosion_expl_mpi" )#Default is cartesian mesh
 add_test(NAME testWaveSystem_SphericalExplosion_expl_mpi_2D_SQUARE_squares_4procs  COMMAND "${MPIEXEC}" "-n" "4" "./WaveSystem_SphericalExplosion_expl_mpi" )#Default is cartesian mesh
 add_test(NAME testWaveSystem_SphericalExplosion_expl_mpi_3D_CUBE_tetras_2procs     COMMAND "${MPIEXEC}" "-n" "2" "./WaveSystem_SphericalExplosion_expl_mpi" "${PROJECT_SOURCE_DIR}/../meshes/meshCube.med")     
 add_test(NAME testWaveSystem_SphericalExplosion_expl_mpi_3D_CUBE_tetras_4procs     COMMAND "${MPIEXEC}" "-n" "4" "./WaveSystem_SphericalExplosion_expl_mpi" "${PROJECT_SOURCE_DIR}/../meshes/meshCube.med")     
 add_test(NAME testWaveSystem_SphericalExplosion_impl_mpi_2D_SQUARE_squares_2procs  COMMAND "${MPIEXEC}" "-n" "2" "./WaveSystem_SphericalExplosion_impl_mpi" )#Default is cartesian mesh
 add_test(NAME testWaveSystem_SphericalExplosion_impl_mpi_2D_SQUARE_squares_4procs  COMMAND "${MPIEXEC}" "-n" "4" "./WaveSystem_SphericalExplosion_impl_mpi" )#Default is cartesian mesh
 add_test(NAME testWaveSystem_SphericalExplosion_impl_mpi_3D_CUBE_tetras_2procs     COMMAND "${MPIEXEC}" "-n" "2" "./WaveSystem_SphericalExplosion_impl_mpi" "${PROJECT_SOURCE_DIR}/../meshes/meshCube.med")     
 add_test(NAME testWaveSystem_SphericalExplosion_impl_mpi_3D_CUBE_tetras_4procs     COMMAND "${MPIEXEC}" "-n" "4" "./WaveSystem_SphericalExplosion_impl_mpi" "${PROJECT_SOURCE_DIR}/../meshes/meshCube.med")     

 set_tests_properties( 
 testWaveSystem_SphericalExplosion_expl_mpi_2D_SQUARE_squares_2procs
 testWaveSystem_SphericalExplosion_expl_mpi_3D_CUBE_tetras_2procs
 testWaveSystem_SphericalExplosion_expl_mpi_2D_SQUARE_squares_4procs
 testWaveSystem_SphericalExplosion_expl_mpi_3D_CUBE_tetras_4procs
 testWaveSystem_SphericalExplosion_impl_mpi_2D_SQUARE_squares_2procs
 testWaveSystem_SphericalExplosion_impl_mpi_3D_CUBE_tetras_2procs
 testWaveSystem_SphericalExplosion_impl_mpi_2D_SQUARE_squares_4procs
 testWaveSystem_SphericalExplosion_impl_mpi_3D_CUBE_tetras_4procs
 PROPERTIES ENVIRONMENT "LD_LIBRARY_PATH=${SOLVERLAB_DIR}/lib:${MEDCOUPLING_DIR}/lib:${MEDFILE_DIR}/lib"
 )
endif( CirculantPreconditioner_WITH_MPI )

 install(TARGETS WaveSystem_SphericalExplosion_expl_seq DESTINATION tests)
 install(TARGETS WaveSystem_SphericalExplosion_expl_mpi DESTINATION tests)


