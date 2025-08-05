//============================================================================
// Author      : Michael NDJINGA
// Date        : November 2020
// Description : multiD linear wave system
//============================================================================

#include "WaveSystem.hxx"

using namespace std;

void WaveSystem_impl_seq(double tmax, int ntmax, double cfl, int output_freq, const Mesh& my_mesh, const string file)
{
    /* Retrieve mesh data */
    int dim=my_mesh.getMeshDimension();
    int nbCells = my_mesh.getNumberOfCells();
    std::string meshName=my_mesh.getName();
    int nbVoisinsMax=my_mesh.getMaxNbNeighbours(CELLS);
    double dx_min=my_mesh.minRatioVolSurf();
    int nbComp=dim+1;        
    double norm;

    /* PETSc variables */
    int globalNbUnknowns;
    int localNbUnknowns;
    int d_nnz, o_nnz;
    Vec Un, dUn;
    Mat A;
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

    globalNbUnknowns=nbCells*nbComp;
    nbVoisinsMax = my_mesh.getMaxNbNeighbours(CELLS);
    d_nnz=(nbVoisinsMax+1)*nbComp;
    o_nnz= nbVoisinsMax   *nbComp;

    MatCreateAIJ(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,globalNbUnknowns,globalNbUnknowns,d_nnz,NULL,o_nnz,NULL,&A);

    /* Initial conditions */
    cout << "Construction of the initial condition" << endl;
    
    Field pressure_field("Pressure",CELLS,my_mesh,1) ;
    Field velocity_field("Velocity",CELLS,my_mesh,dim) ;
    initial_conditions_shock(my_mesh,pressure_field, velocity_field);

    /* iteration vectors */
    VecCreateMPI(PETSC_COMM_WORLD,PETSC_DECIDE    ,globalNbUnknowns,&Un);
    VecDuplicate (Un,&dUn);
    
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
    VecAssemblyBegin(Un);
    VecAssemblyEnd(Un);

    /*
     * MED/VTK output of the initial condition at t=0 and iter = 0
     */
    int it=0;
    bool isStationary=false;
    double time=0.;
    double dt = cfl * dx_min / c0;
    
    cout << "Saving the solution at T=" << time << endl;
    pressure_field.setTime(time,it);
    pressure_field.writeMED("WaveSystem"+to_string(dim)+"DUpwind"+meshName+"_pressure");
    velocity_field.setTime(time,it);
    velocity_field.writeMED("WaveSystem"+to_string(dim)+"DUpwind"+meshName+"_velocity");
    /* --------------------------------------------- */

    /* Fill the linear system matrix */
    computeDivergenceMatrix(my_mesh,&A,dt);

    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(  A, MAT_FINAL_ASSEMBLY);
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
    cout<< "Starting computation of the linear wave system with an implicit UPWIND scheme" << endl;
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
            cout<<"-- Iteration: " << it << ", Time: " << time << ", dt: " << dt<<endl;

            for(int k=0; k<nbCells; k++)
            {
                    idx = k*(dim+1)+0;
                    VecGetValues(Un,1,&idx,&value);
                    pressure_field[k]  =value;
                    for(int idim =0; idim<dim; idim++)
                    {
                        idx = k*nbComp+1+idim;
                        VecGetValues(Un,1,&idx,&value);
                        velocity_field[k,idim] = value/rho0;
                    }
            }
            pressure_field.setTime(time,it);
            pressure_field.writeVTK("WaveSystem"+to_string(dim)+"DUpwind"+meshName+"_pressure",false);
            velocity_field.setTime(time,it);
            velocity_field.writeVTK("WaveSystem"+to_string(dim)+"DUpwind"+meshName+"_velocity",false);
 
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
    cout<<"End of calculation -- Iteration: " << it << ", Time: "<< time<< ", dt: " << dt<<endl;

    if(it>=ntmax)
        cout<< "Nombre de pas de temps maximum ntmax= "<< ntmax<< " atteint"<<endl;
    else if(isStationary)
        cout<< "Régime stationnaire atteint au pas de temps "<< it<< ", t= "<< time<<endl;       
    else
        cout<< "Temps maximum Tmax= "<< tmax<< " atteint"<<endl;

    VecDestroy(&Un);
    VecDestroy(&dUn);
    MatDestroy(&A);
}
 
int main(int argc, char *argv[])
{
    cout << "-- Starting the RESOLUTION OF THE 2D WAVE SYSTEM"<<endl;
    cout << "- Numerical scheme : Upwind explicit scheme" << endl;
    cout << "- Boundary conditions : WALL" << endl;

     PetscInitialize(&argc, &argv, NULL, NULL);

   // Problem data
    double tmax=0.05;
    double ntmax=2000000;
    int freqSortie=100;
    string fileOutPut="SphericalWave";

    Mesh myMesh;
    
    if(argc<2)
    {
            cout << "- DOMAIN : SQUARE" << endl;
            cout << "- MESH : CARTESIAN, GENERATED INTERNALLY WITH CDMATH" << endl<< endl;
            cout << "Construction of a cartesian mesh" << endl;
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
        cout << "- MESH:  GENERATED EXTERNALLY WITH SALOME" << endl;
        cout << "Loading of a mesh named "<<argv[1] << endl;
        string filename = argv[1];
        myMesh=Mesh(filename);
    }

    double cfl=1.e4/myMesh.getSpaceDimension();
    WaveSystem_impl_seq(tmax,ntmax,cfl,freqSortie,myMesh,fileOutPut);
    
    cout << "Simulation complete." << endl;

    PetscFinalize();
    return 0;
}
