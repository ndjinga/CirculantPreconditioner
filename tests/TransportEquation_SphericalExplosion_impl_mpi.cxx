//============================================================================
// Author      : Michael NDJINGA
// Date        : Août 2025
// Description : multiD linear transport equation run in parallel with calls to MPI
//               \partial_t u + \vec a \cdot \vec\nabla u = 0
//============================================================================

#include "TransportEquation2.hxx"

using namespace std;


void TransportEquation_impl_mpi(double tmax, int ntmax, double cfl, int output_freq, const Mesh& my_mesh, const string file, int rank, int size, string resultDirectory, Vector vitesseTransport, double precision)
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
    PetscScalar value;//value to add in the vector    
    KSP ksp;
    KSPType ksptype=(char*)&KSPGMRES;
    PC pc;
    PCType pctype=(char*)&PCBJACOBI;
    int maxPetscIts=1000;//nombre maximum d'iteration gmres autorisé au cours d'une resolution de système lineaire
    int PetscIts;//the number of iterations performed by the linear solver
    KSPConvergedReason reason;
    double residu;

    
    /* Mesh parameters managed only by proc 0 */
    int nbCells;
    int dim;
    Field temperature_field;
    std::string meshName;

    if(rank == 0)
    {
        /* Retrieve mesh data */
        nbCells = my_mesh.getNumberOfCells();
        dim=my_mesh.getMeshDimension();
        meshName=my_mesh.getName();
        double dx_min=my_mesh.minRatioVolSurf();
        dt = cfl * dx_min / vitesseTransport.norm();
        
        globalNbUnknowns=nbCells;
    }
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
        d_nnz= nbVoisinsMax+1;
        o_nnz= nbVoisinsMax   ;
    }
    MPI_Bcast(&d_nnz, 1, MPI_INT, 0, PETSC_COMM_WORLD);
    MPI_Bcast(&o_nnz, 1, MPI_INT, 0, PETSC_COMM_WORLD);

    if(rank == 0)
        MatCreateAIJ(PETSC_COMM_WORLD,localNbUnknowns,localNbUnknowns,globalNbUnknowns,globalNbUnknowns,d_nnz,NULL,o_nnz+(size-1)*d_nnz,NULL,&A);
    else
        MatCreateAIJ(PETSC_COMM_WORLD,localNbUnknowns,localNbUnknowns,globalNbUnknowns,globalNbUnknowns,d_nnz,NULL,o_nnz,NULL,&A);

    if(rank == 0)
    {
        /* Initial conditions */
        cout<<"Building the initial condition on processor 0" << endl;
        
        temperature_field=Field("temperature",CELLS,my_mesh,1) ;
        initial_conditions_shock(my_mesh, temperature_field);
    
        cout << "Saving the solution at time t=" << time <<"  on processor 0"<<endl;
        temperature_field.setTime(time,it);
        temperature_field.writeMED(resultDirectory+"/TransportEquation"+to_string(dim)+"DUpwind_"+to_string(size)+"Procs_"+meshName+"_temperature");
        /* --------------------------------------------- */
    

        for(int k =0; k<nbCells; k++)
        {
            idx = k;
            value=temperature_field[k];//value to add in the vector
            VecSetValues(Un,1,&idx,&value,INSERT_VALUES);
        }
        computeDivergenceMatrix( my_mesh, &A, dt, vitesseTransport);
    }        

    VecAssemblyBegin(Un);
    VecAssemblyEnd(Un);
    
    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(  A, MAT_FINAL_ASSEMBLY);

    MPI_Bcast(&dt, 1, MPI_DOUBLE, 0, PETSC_COMM_WORLD);

    MatShift( A, 1);//Contribution from the time derivative
    
    /* PETSc Linear solver (all procs) */
    KSPCreate(PETSC_COMM_WORLD, &ksp);
    KSPSetType(ksp, ksptype);
    KSPSetTolerances( ksp, precision, precision,PETSC_DEFAULT, maxPetscIts);
    KSPGetPC(ksp, &pc);
    //PETSc preconditioner
    PCSetType( pc, pctype);
    KSPSetOperators(ksp, A, A);

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
                    idx = k+0;
                    VecGetValues(Un_seq,1,&idx,&value);
                    temperature_field[k]  = PetscRealPart(value);
                }
                temperature_field.setTime(time,it);
                temperature_field.writeMED(resultDirectory+"/TransportEquation"+to_string(dim)+"DUpwind_"+to_string(size)+"Procs_"+meshName+"_temperature",false);
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
    double precision=1e-5;
    
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
            
            myMesh.setGroupAtPlan(xsup,0,precision,"RightEdge");
            myMesh.setGroupAtPlan(xinf,0,precision,"LeftEdge");
            myMesh.setGroupAtPlan(yinf,1,precision,"BottomEdge");
            myMesh.setGroupAtPlan(ysup,1,precision,"TopEdge");
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
    
	Vector vitesseTransport=myMesh.getSpaceDimension();
	vitesseTransport[0]=1;

    double cfl=1.e3/myMesh.getSpaceDimension();
    TransportEquation_impl_mpi(tmax,ntmax,cfl,freqSortie,myMesh,fileOutPut, rank, size, resultDirectory, vitesseTransport, precision);

    if(rank == 0)
        cout << "Simulation complete." << endl;

    PetscFinalize();
    return 0;
}
