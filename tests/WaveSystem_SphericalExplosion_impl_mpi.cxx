//============================================================================
// Author      : Michael NDJINGA
// Date        : November 2020
// Description : multiD linear wave system run in parallel with calls to MPI
//               Test used in the MPI version of the Salome platform
//============================================================================

#include "WaveSystem.hxx"

using namespace std;


void WaveSystem_impl_mpi(double tmax, int ntmax, double cfl, int output_freq, const Mesh& my_mesh, const string file, int rank, int size, string resultDirectory)
{
    /* Time iteration variables */
    int it=0;
    bool isStationary=false;
    double time=0.;
    double dt;
    double norm;
        
    /* PETSc variables */
    int globalNbUnknowns;
    int localNbUnknowns;
    int d_nnz, o_nnz;
    Vec Un, dUn, Un_seq;
    Mat A;
    VecScatter scat;
    int idx;//Index where to add the vector values
    double value;//value to add in the vector    
    KSP ksp;
    KSPType ksptype=(char*)&KSPGMRES;
    PC pc;
    PCType pctype=(char*)&PCBJACOBI;
    int maxPetscIts=200;//nombre maximum d'iteration gmres autorisé au cours d'une resolution de système lineaire
    int PetscIts;//the number of iterations performed by the linear solver
    KSPConvergedReason reason;
    double residu;

    
    /* Mesh parameters managed only by proc 0 */
    int nbCells;
    int dim;
    int nbComp;        
    Field pressure_field, velocity_field;
    std::string meshName;

    if(rank == 0)
        globalNbUnknowns=my_mesh.getNumberOfCells()*(my_mesh.getMeshDimension()+1);//nbCells*nbComp
    
    MPI_Bcast(&globalNbUnknowns, 1, MPI_INT, 0, PETSC_COMM_WORLD);
 
    /* iteration vectors */
    VecCreateMPI(PETSC_COMM_WORLD,PETSC_DECIDE    ,globalNbUnknowns,&Un);
    VecDuplicate (Un,&dUn);
    if(rank == 0)
        VecCreateSeq(PETSC_COMM_SELF,globalNbUnknowns,&Un_seq);//For saving results on proc 0

    VecScatterCreateToZero(Un,&scat,&Un_seq);

    /* System matrix */
    VecGetLocalSize(Un, &localNbUnknowns);

    if(rank == 0)
    {
        int nbVoisinsMax = my_mesh.getMaxNbNeighbours(CELLS);
        d_nnz=(nbVoisinsMax+1)*(my_mesh.getMeshDimension()+1);//(nbVoisinsMax+1)*nbComp
        o_nnz= nbVoisinsMax   *(my_mesh.getMeshDimension()+1);//                 nbComp
    }
    MPI_Bcast(&d_nnz, 1, MPI_INT, 0, PETSC_COMM_WORLD);
    MPI_Bcast(&o_nnz, 1, MPI_INT, 0, PETSC_COMM_WORLD);

    MatCreateAIJ(PETSC_COMM_WORLD,localNbUnknowns,localNbUnknowns,globalNbUnknowns,globalNbUnknowns,d_nnz,NULL,o_nnz,NULL,&A);

    if(rank == 0)
        {
        /* Retrieve mesh data */
        nbCells = my_mesh.getNumberOfCells();
        dim=my_mesh.getMeshDimension();
        nbComp=dim+1;        
        meshName=my_mesh.getName();
        double dx_min=my_mesh.minRatioVolSurf();
        dt = cfl * dx_min / c0;
        
        /* Initial conditions */
        cout<<"Building the initial condition on processor 0" << endl;
        
        pressure_field=Field("Pressure",CELLS,my_mesh,1) ;
        velocity_field=Field("Velocity",CELLS,my_mesh,dim) ;
        initial_conditions_shock(my_mesh,pressure_field, velocity_field);
    
        cout << "Saving the solution at time t=" << time <<"  on processor 0"<<endl;
        pressure_field.setTime(time,it);
        pressure_field.writeVTK(resultDirectory+"/WaveSystem"+to_string(dim)+"DUpwind_"+to_string(size)+"Procs_"+meshName+"_pressure");
        velocity_field.setTime(time,it);
        velocity_field.writeVTK(resultDirectory+"/WaveSystem"+to_string(dim)+"DUpwind_"+to_string(size)+"Procs_"+meshName+"_velocity");
        /* --------------------------------------------- */
    

        for(int k =0; k<nbCells; k++)
        {
            idx = k*nbComp;
            value=pressure_field[k];//vale to add in the vector
            VecSetValues(Un,1,&idx,&value,INSERT_VALUES);
            for(int idim =0; idim<dim; idim++)
            {
                idx = k*nbComp+1+idim;
                value =rho0*velocity_field[k,idim];
                VecSetValues(Un,1,&idx,&value,INSERT_VALUES);
            }
        }
        computeDivergenceMatrix(my_mesh,&A,dt);
    }        

    VecAssemblyBegin(Un);
    VecAssemblyEnd(Un);
    
    //VecView(Un, PETSC_VIEWER_STDOUT_WORLD );

    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(  A, MAT_FINAL_ASSEMBLY);

    //MatView(divMat,    PETSC_VIEWER_STDOUT_WORLD );

    MPI_Bcast(&dt, 1, MPI_DOUBLE, 0, PETSC_COMM_WORLD);

    /* Time loop */
    PetscPrintf(PETSC_COMM_WORLD,"Starting computation of the linear wave system on all processors : \n\n");
    while (it<ntmax && time <= tmax && !isStationary)
    {
        VecCopy(Un,dUn);
        KSPSolve(ksp, Un, Un);
        VecAXPY(dUn,-1,Un);
        
        time=time+dt;
        it=it+1;
 
        VecNorm(dUn,NORM_2,&norm);
        isStationary = norm<precision;
        /* Sauvegardes */
        if( it%output_freq==0 or it>=ntmax or isStationary or time >=tmax )
        {
            PetscPrintf(PETSC_COMM_WORLD,"-- Iteration: %d, time: %f, dt: %f, saving results on processor 0 \n", it, time, dt);
            VecScatterBegin(scat,Un,Un_seq,INSERT_VALUES,SCATTER_FORWARD);
            VecScatterEnd(  scat,Un,Un_seq,INSERT_VALUES,SCATTER_FORWARD);

            if(rank == 0)
            {
                for(int k=0; k<nbCells; k++)
                {
                    idx = k*(dim+1)+0;
                    VecGetValues(Un_seq,1,&idx,&value);
                    pressure_field[k]  =value;
                    for(int idim =0; idim<dim; idim++)
                    {
                        idx = k*nbComp+1+idim;
                        VecGetValues(Un_seq,1,&idx,&value);
                        velocity_field[k,idim] = value/rho0;
                    }
                }
                pressure_field.setTime(time,it);
                pressure_field.writeVTK(resultDirectory+"/WaveSystem"+to_string(dim)+"DUpwind_"+to_string(size)+"Procs_"+meshName+"_pressure",false);
                velocity_field.setTime(time,it);
                velocity_field.writeVTK(resultDirectory+"/WaveSystem"+to_string(dim)+"DUpwind_"+to_string(size)+"Procs_"+meshName+"_velocity",false);
            }
            KSPGetConvergedReason( ksp,&reason);
            KSPGetIterationNumber( ksp, &PetscIts);
            KSPGetResidualNorm(ksp,&residu);
            if (reason!=2 and reason!=3)
            {
                    PetscPrintf(PETSC_COMM_WORLD,"!!!!!!!!!!!!! Erreur système linéaire : pas de convergence de Petsc.\n");
                    PetscPrintf(PETSC_COMM_WORLD,"!!!!!!!!!!!!! Itérations maximales %d atteintes, résidu = %1.2e, précision demandée= %1.2e.\n",maxPetscIts,residu,precision);
                    PetscPrintf(PETSC_COMM_WORLD,"Solver used %s, preconditioner %s, Final number of iteration = %d.\n",ksptype,pctype,PetscIts);
            }
            else
                PetscPrintf(PETSC_COMM_WORLD,"## Système linéaire résolu en %d itérations par le solveur %s et le preconditioneur %s, précision demandée = %1.2e, résidu final  = %1.2e\n",PetscIts,ksptype,pctype,precision, residu);        
        }
    }

    PetscPrintf(PETSC_COMM_WORLD,"\n End of calculations at iteration: %d, and time: %f\n", it, time);
    if(it>=ntmax)
        PetscPrintf(PETSC_COMM_WORLD, "Nombre de pas de temps maximum ntmax= %d atteint\n", ntmax);
    else if(isStationary)
        PetscPrintf(PETSC_COMM_WORLD, "Régime stationnaire atteint au pas de temps %d, t= %f\n", it, time);       
    else
        PetscPrintf(PETSC_COMM_WORLD, "Temps maximum tmax= %f atteint\n", tmax);

    VecDestroy(&Un);
    VecDestroy(&Un_seq);
    VecDestroy(&dUn);
    MatDestroy(&A);
}
 
int main(int argc, char *argv[])
{
    /* PETSc initialisation */
    PetscInitialize(&argc, &argv, NULL, NULL);
    PetscMPIInt    size;        /* size of communicator */
    PetscMPIInt    rank;        /* processor rank */
    MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
    MPI_Comm_size(PETSC_COMM_WORLD,&size);
    
    // Problem data
    double tmax=0.05;
    int ntmax=2000000;
    int freqSortie=1;
    string fileOutPut="SphericalWave";
    Mesh myMesh;
    string resultDirectory="./";
    
    if(size>1)
        PetscPrintf(PETSC_COMM_WORLD,"---- More than one processor detected : running a parallel simulation ----\n");
        PetscPrintf(PETSC_COMM_WORLD,"---- Limited parallelism : input and output remain sequential ----\n");
        PetscPrintf(PETSC_COMM_WORLD,"---- Only the linear system is solved in parallel ----\n");
        PetscPrintf(PETSC_COMM_WORLD,"---- Processor 0 is in charge of building the mesh, saving the results, filling and then distributing the matrix to other processors.\n\n");
        
    if(rank == 0)
    {
        cout << "-- Starting the RESOLUTION OF THE 2D WAVE SYSTEM on "<< size <<" processors"<<endl;
        cout << "- Numerical scheme : Upwind implicit scheme" << endl;
        cout << "- Boundary conditions : WALL" << endl;
    
        /* Read or create mesh */
        if(argc<2)
        {
            cout << "- DOMAIN : SQUARE" << endl;
            cout << "- MESH : CARTESIAN, GENERATED INTERNALLY WITH CDMATH" << endl<< endl;
            cout << "Construction of a cartesian mesh on processor 0" << endl;
            double xinf=-0.5;
            double xsup= 0.5;
            double yinf=-0.5;
            double ysup= 0.5;
            int nx=50;
            int ny=50;
            myMesh=Mesh(xinf,xsup,nx,yinf,ysup,ny);
            
            double eps=precision;
            myMesh.setGroupAtPlan(xsup,0,eps,"RightEdge");
            myMesh.setGroupAtPlan(xinf,0,eps,"LeftEdge");
            myMesh.setGroupAtPlan(yinf,1,eps,"BottomEdge");
            myMesh.setGroupAtPlan(ysup,1,eps,"TopEdge");
        }
        else
        {
            cout << "- MESH:  GENERATED EXTERNALLY WITH SALOME" << endl<< endl;
            cout << "Loading of a mesh named "<<argv[1]<<" on processor 0" << endl;
            string filename = argv[1];
            myMesh=Mesh(filename);
        }

        /*Detect directory where to same results*/
        if(argc>2)
            resultDirectory = argv[2];
    }
    
    double cfl=1.e4/myMesh.getSpaceDimension();
    WaveSystem_impl_mpi(tmax,ntmax,cfl,freqSortie,myMesh,fileOutPut, rank, size, resultDirectory);

    if(rank == 0)
        cout << "Simulation complete." << endl;

    PetscFinalize();
    return 0;
}
