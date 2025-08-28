project(testC)

INCLUDE_DIRECTORIES(
  ${PETSC_INCLUDES}	
  ${SOLVERLAB_INCLUDES} 
  ${MY_HEADERS}										    #
)

 # build test main exe
 add_executable(WaveSystem_SphericalExplosion_expl_seq                            ./WaveSystem_SphericalExplosion_expl_seq.cxx)
 add_executable(WaveSystem_SphericalExplosion_expl_mpi                            ./WaveSystem_SphericalExplosion_expl_mpi.cxx)
 add_executable(WaveSystem_SphericalExplosion_impl_seq                            ./WaveSystem_SphericalExplosion_impl_seq.cxx)
 add_executable(WaveSystem_SphericalExplosion_impl_mpi                            ./WaveSystem_SphericalExplosion_impl_mpi.cxx)
 add_executable(TransportEquation_SphericalExplosion_impl                         ./TransportEquation_SphericalExplosion_impl_mpi.cxx)
 add_executable(TransportEquationFFT_SphericalExplosion_impl                      ./TransportEquationFFT_SphericalExplosion_impl_mpi.cxx)
 target_include_directories(WaveSystem_SphericalExplosion_expl_seq                 PRIVATE ${PETSC_INCLUDES} ${SOLVERLAB_INCLUDES}   ${MY_HEADERS})
 target_include_directories(WaveSystem_SphericalExplosion_expl_mpi                 PRIVATE ${PETSC_INCLUDES} ${SOLVERLAB_INCLUDES}   ${MY_HEADERS})
 target_include_directories(WaveSystem_SphericalExplosion_impl_seq                 PRIVATE ${PETSC_INCLUDES} ${SOLVERLAB_INCLUDES}   ${MY_HEADERS})
 target_include_directories(WaveSystem_SphericalExplosion_impl_mpi                 PRIVATE ${PETSC_INCLUDES} ${SOLVERLAB_INCLUDES}   ${MY_HEADERS})
 target_include_directories(TransportEquation_SphericalExplosion_impl              PRIVATE ${PETSC_INCLUDES} ${SOLVERLAB_INCLUDES}   ${MY_HEADERS})
 target_include_directories(TransportEquationFFT_SphericalExplosion_impl           PRIVATE ${PETSC_INCLUDES} ${SOLVERLAB_INCLUDES}   ${MY_HEADERS})
 target_link_libraries(WaveSystem_SphericalExplosion_expl_seq                       ${PETSC_LIBRARIES} ${SOLVERLAB_LIBS} FFTPreconditionerLib )
 target_link_libraries(WaveSystem_SphericalExplosion_expl_mpi                       ${PETSC_LIBRARIES} ${SOLVERLAB_LIBS} FFTPreconditionerLib )
 target_link_libraries(WaveSystem_SphericalExplosion_impl_seq                       ${PETSC_LIBRARIES} ${SOLVERLAB_LIBS} FFTPreconditionerLib )
 target_link_libraries(WaveSystem_SphericalExplosion_impl_mpi                       ${PETSC_LIBRARIES} ${SOLVERLAB_LIBS} FFTPreconditionerLib )
 target_link_libraries(TransportEquation_SphericalExplosion_impl                    ${PETSC_LIBRARIES} ${SOLVERLAB_LIBS} FFTPreconditionerLib ${MPI_HOME}/lib/libmpi.so)
 target_link_libraries(TransportEquationFFT_SphericalExplosion_impl              ${PETSC_LIBRARIES} ${SOLVERLAB_LIBS} FFTPreconditionerLib ${MPI_HOME}/lib/libmpi.so)

 add_test(NAME testWaveSystem_SphericalExplosion_expl_seq_2D_SQUARE_squares         COMMAND "./WaveSystem_SphericalExplosion_expl_seq"  )#Default is cartesian mesh 
 add_test(NAME testWaveSystem_SphericalExplosion_expl_seq_3D_CUBE_tetras            COMMAND "./WaveSystem_SphericalExplosion_expl_seq" "${PROJECT_SOURCE_DIR}/../meshes/meshCube.med")     
 add_test(NAME testWaveSystem_SphericalExplosion_expl_mpi_2D_SQUARE_squares         COMMAND "./WaveSystem_SphericalExplosion_expl_mpi"  )#Default is cartesian mesh
 add_test(NAME testWaveSystem_SphericalExplosion_expl_mpi_3D_CUBE_tetras            COMMAND "./WaveSystem_SphericalExplosion_expl_mpi" "${PROJECT_SOURCE_DIR}/../meshes/meshCube.med")     
 add_test(NAME testWaveSystem_SphericalExplosion_impl_seq_2D_SQUARE_squares         COMMAND "./WaveSystem_SphericalExplosion_impl_seq"  )#Default is cartesian mesh 
 add_test(NAME testWaveSystem_SphericalExplosion_impl_seq_3D_CUBE_tetras            COMMAND "./WaveSystem_SphericalExplosion_impl_seq" "${PROJECT_SOURCE_DIR}/../meshes/meshCube.med")     
 add_test(NAME testWaveSystem_SphericalExplosion_impl_mpi_2D_SQUARE_squares         COMMAND "./WaveSystem_SphericalExplosion_impl_mpi"  )#Default is cartesian mesh
 add_test(NAME testWaveSystem_SphericalExplosion_impl_mpi_3D_CUBE_tetras            COMMAND "./WaveSystem_SphericalExplosion_impl_mpi" "${PROJECT_SOURCE_DIR}/../meshes/meshCube.med")     

 add_test(NAME testTransportEquation_SphericalExplosion_impl_seq_2D_SQUARE_squares       COMMAND "./TransportEquation_SphericalExplosion_impl"             )#Default is cartesian mesh
 add_test(NAME testTransportEquationFFT_SphericalExplosion_impl_seq_1D_SEGMENT_segments  COMMAND "./TransportEquationFFT_SphericalExplosion_impl" "10"       )
 add_test(NAME testTransportEquationFFT_SphericalExplosion_impl_seq_2D_SQUARE_squares    COMMAND "./TransportEquationFFT_SphericalExplosion_impl" "10" "10"    )
 add_test(NAME testTransportEquationFFT_SphericalExplosion_impl_seq_3D_CUBE_cubes        COMMAND "./TransportEquationFFT_SphericalExplosion_impl" "10" "10" "10" )
 add_test(NAME testTransportEquationFFT_SphericalExplosion_impl_seq_3D_CUBE_cubes_gros        COMMAND "./TransportEquationFFT_SphericalExplosion_impl" "100" "100" "100" )

 set_tests_properties( 
 testWaveSystem_SphericalExplosion_expl_seq_2D_SQUARE_squares 
 testWaveSystem_SphericalExplosion_expl_seq_3D_CUBE_tetras
 testWaveSystem_SphericalExplosion_expl_mpi_2D_SQUARE_squares
 testWaveSystem_SphericalExplosion_expl_mpi_3D_CUBE_tetras
 testWaveSystem_SphericalExplosion_impl_seq_2D_SQUARE_squares
 testWaveSystem_SphericalExplosion_impl_seq_3D_CUBE_tetras
 testWaveSystem_SphericalExplosion_impl_mpi_2D_SQUARE_squares
 testWaveSystem_SphericalExplosion_impl_mpi_3D_CUBE_tetras
 testTransportEquation_SphericalExplosion_impl_seq_2D_SQUARE_squares
 testTransportEquationFFT_SphericalExplosion_impl_seq_1D_SEGMENT_segments
 testTransportEquationFFT_SphericalExplosion_impl_seq_2D_SQUARE_squares
 testTransportEquationFFT_SphericalExplosion_impl_seq_3D_CUBE_cubes
 testTransportEquationFFT_SphericalExplosion_impl_seq_3D_CUBE_cubes_gros
 PROPERTIES ENVIRONMENT "LD_LIBRARY_PATH=${SOLVERLAB_DIR}/lib:${MEDCOUPLING_DIR}/lib:${MEDFILE_DIR}/lib"
 )

if( FFTPreconditioner_WITH_MPI )
 target_link_libraries(WaveSystem_SphericalExplosion_expl_mpi                          ${MPI_HOME}/lib/libmpi.so)
 target_link_libraries(WaveSystem_SphericalExplosion_impl_mpi                          ${MPI_HOME}/lib/libmpi.so)
 target_link_libraries(TransportEquation_SphericalExplosion_impl                       ${MPI_HOME}/lib/libmpi.so)
 target_link_libraries(TransportEquationFFT_SphericalExplosion_impl                    ${MPI_HOME}/lib/libmpi.so)

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
endif( FFTPreconditioner_WITH_MPI )

 install(TARGETS WaveSystem_SphericalExplosion_expl_seq        DESTINATION tests)
 install(TARGETS WaveSystem_SphericalExplosion_expl_mpi        DESTINATION tests)
 install(TARGETS TransportEquation_SphericalExplosion_impl     DESTINATION tests)
 install(TARGETS TransportEquationFFT_SphericalExplosion_impl  DESTINATION tests)

