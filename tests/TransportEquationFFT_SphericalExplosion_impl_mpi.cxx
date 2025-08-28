//============================================================================
// Author      : Michael NDJINGA, Marwane Kadouci
// Date        : Août 2025
// Description : multiD linear transport equation on cartesian mesh
//               \partial_t u + \vec a \cdot \vec\nabla u = 0
//               linear systems solved with FFT direct solver
//               run in parallel with calls to MPI
//============================================================================

#include "TransportEquation2.hxx"
#include "CdmathException.hxx"

extern "C" {
    #include "FftNumericalSolver_3D.h"
}

using namespace std;


void TransportEquationFFT_impl_mpi(double tmax, int ntmax, double cfl, int output_freq, const Mesh& my_mesh, const string file, int rank, int size, string resultDirectory, Vector vitesseTransport, double precision)
{
    /* Time iteration variables */
    int it=0;
    bool isStationary=false;
    double time=0.;
    double dt;
    double norm;

    /* PETSc variables */
    Vec Un, dUn; // enlever Un
    int idx;//Index where to add the vector values
    PetscScalar value;//value to add in the vector    
    
    /* Mesh parameters managed only by proc 0 */
    int nbCells;
    int dim;
    Field temperature_field;
    std::string meshName;

    
    /* Retrieve mesh data */
    nbCells = my_mesh.getNumberOfCells();
    dim=my_mesh.getMeshDimension();
    meshName=my_mesh.getName();
    double dx_min=my_mesh.minRatioVolSurf();
    dt = cfl * dx_min / vitesseTransport.norm();
    
    
    /* Collect mesh data */
    int nx, ny, nz;
    double delta_x, delta_y, delta_z;
    nx = my_mesh.getNx();
    delta_x = ( my_mesh.getXMax() - my_mesh.getXMin() )/nx;
    if(dim>1)
    {
        ny = my_mesh.getNy();
        delta_y = ( my_mesh.getYMax() - my_mesh.getYMin() )/ny;
        if(dim>2)
        {
            nz = my_mesh.getNz();
            delta_z = ( my_mesh.getZMax() - my_mesh.getZMin() )/nz;
        }
    }    
 
     /* iteration vectors */
    VecCreateMPI(PETSC_COMM_WORLD,PETSC_DECIDE    ,nbCells,&Un);
    VecDuplicate (Un,&dUn);
 
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
    }        

    VecAssemblyBegin(Un);
    VecAssemblyEnd(Un);
    
    MPI_Bcast(&dt, 1, MPI_DOUBLE, 0, PETSC_COMM_WORLD);

    /* Implicit matrix */
    Mat FFT_MAT;
    PetscInt ndim = 3;
    PetscInt dims[3] = {nz, ny, nx};
    MatCreateFFT(PETSC_COMM_WORLD, ndim, dims, MATFFTW, &FFT_MAT);
    StructuredTransportContext ctx = {nx, ny, nz, vitesseTransport[0], vitesseTransport[1], vitesseTransport[2], dt, delta_x, delta_y, delta_z, FFT_MAT};
    
    /* Time loop */
    PetscPrintf(PETSC_COMM_WORLD,"Starting computation of the linear wave system on all processors : \n\n");

    PetscLogDouble v, w;
    while (it<ntmax && time <= tmax && !isStationary)
    {
        VecCopy(Un,dUn);
        PetscTime(&v);
        PetscFft3DTransportSolver(ctx, Un, Un);
        PetscTime(&w);
        VecAXPY(dUn,-1,Un);
        
        time=time+dt;
        it=it+1;
 
        VecNorm(dUn,NORM_2,&norm);
        isStationary = norm<precision;
        /* Sauvegardes */
        if( it%output_freq==0 or it>=ntmax or isStationary or time >=tmax )
        {
            PetscPrintf(PETSC_COMM_WORLD,"-- Pas de temps: %d, time: %f, dt: %f, solve cpu time : %f ,saving results on processor 0 \n", it, time, dt, w - v);

            if(rank == 0)
            {
                for(int k=0; k<nbCells; k++)
                {
                    idx = k+0;
                    VecGetValues(Un,1,&idx,&value);
                    temperature_field[k]  = PetscRealPart(value);
                }
                temperature_field.setTime(time,it);
                temperature_field.writeMED(resultDirectory+"/TransportEquation"+to_string(dim)+"DUpwind_"+to_string(size)+"Procs_"+meshName+"_temperature",false);
            }
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
    VecDestroy(&Un);
    VecDestroy(&dUn);
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
        cout << "-- Starting the FFT RESOLUTION OF THE Transport equation on "<< size <<" processors"<<endl;
        cout << "- Numerical scheme : Upwind implicit scheme" << endl;
        cout << "- Boundary conditions : WALL" << endl;
    
        /* Read or create mesh */
        if(argc==2)
        {
            cout << "- DOMAIN : Interval [-0.5,0.5]" << endl;
            cout << "- MESH : Structured identical cells" << endl<< endl;
            cout << "Construction of a cartesian mesh on processor 0" << endl;
            double xinf=-0.5;
            double xsup= 0.5;
            int nx=atoi(argv[1]);
            myMesh=Mesh(xinf,xsup,nx);
                    }
        else
        if(argc==3)
        {
            cout << "- DOMAIN : SQUARE [-0.5,0.5]x[-0.5,0.5]" << endl;
            cout << "- MESH : Structured identical cells" << endl<< endl;
            cout << "Construction of a cartesian mesh on processor 0" << endl;
            double xinf=-0.5;
            double xsup= 0.5;
            double yinf=-0.5;
            double ysup= 0.5;
            int nx=atoi(argv[1]);
            int ny=atoi(argv[2]);
            myMesh=Mesh(xinf,xsup,nx,yinf,ysup,ny);
        }
        else
        if(argc==4)
        {
            cout << "- DOMAIN : CUBE [-0.5,0.5]x[-0.5,0.5]x[-0.5,0.5]" << endl;
            cout << "- MESH : Structured identical cells" << endl<< endl;
            cout << "Construction of a cartesian mesh on processor 0" << endl;
            double xinf=-0.5;
            double xsup= 0.5;
            double yinf=-0.5;
            double ysup= 0.5;
            double zinf=-0.5;
            double zsup= 0.5;
            int nx=atoi(argv[1]);
            int ny=atoi(argv[2]);
            int nz=atoi(argv[3]);
            myMesh=Mesh(xinf,xsup,nx,yinf,ysup,ny,zinf,zsup,nz);
        }
        else
            throw CdmathException("provide nx, ny, nz in command line");        
    }
    
	Vector vitesseTransport=myMesh.getSpaceDimension();
	vitesseTransport[0]=1;

    double cfl=1.e3/myMesh.getSpaceDimension();
    TransportEquationFFT_impl_mpi(tmax,ntmax,cfl,freqSortie,myMesh,fileOutPut, rank, size, resultDirectory, vitesseTransport, precision);

    if(rank == 0)
        cout << "Simulation complete." << endl;

    PetscFinalize();
    return 0;
}
