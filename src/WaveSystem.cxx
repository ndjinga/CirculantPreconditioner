//============================================================================
// Author      : Michael NDJINGA
// Date        : November 2020
// Description : multiD linear wave system run in parallel with calls to MPI
//               Test used in the MPI version of the Salome platform
//============================================================================

#include <iostream>
#include <string>
#include <cmath>

#include "WaveSystem.hxx"

#include <petscksp.h>

#include "Mesh.hxx"
#include "Cell.hxx"
#include "Face.hxx"
#include "Field.hxx"
#include "CdmathException.hxx"


using namespace std;

void initial_conditions_shock(Mesh my_mesh,Field& pressure_field,Field& velocity_field)
{
    double rayon=0.15;
    double xcentre=0.;
    double ycentre=0;
    double zcentre=0;
    
    double x, y, z;
    double val, valX, valY, valZ;
    
    int dim    =my_mesh.getMeshDimension();
    int nbCells=my_mesh.getNumberOfCells();

    for (int j=0 ; j<nbCells ; j++)
    {
        x = my_mesh.getCell(j).x() ;
        if(dim>1)
        {
            y = my_mesh.getCell(j).y() ;
            if(dim==3)
                z = my_mesh.getCell(j).z() ;
        }

        valX=(x-xcentre)*(x-xcentre);
        if(dim==1)
            val=sqrt(valX);
        else if(dim==2)
        {
            valY=(y-ycentre)*(y-ycentre);
            val=sqrt(valX+valY);        
        }
        else if(dim==3)
        {
            valY=(y-ycentre)*(y-ycentre);
            valZ=(z-zcentre)*(z-zcentre);
            val=sqrt(valX+valY+valZ);        
        }
        
        for(int idim=0; idim<dim; idim++)
            velocity_field[j,idim]=0;
            
        if (val<rayon)
            pressure_field(j) = 155e5;
        else
            pressure_field(j) = 70e5;
    }
}

void addValue( int i, int j, Matrix M, Mat * mat )
{
    int I,J;
    for (int k=0; k<M.getNumberOfRows(); k++)
        for (int l=0; l<M.getNumberOfColumns(); l++)
        {
            I=i+k;
            J=j+l;
            MatSetValues( *mat,1, &I, 1, &J, &M(k,l), ADD_VALUES);
        }
}

Matrix jacobianMatrices(Vector normal, double coeff)
{
    int dim=normal.size();
    Matrix    A(dim+1,dim+1);
    Matrix absA(dim+1,dim+1);

    absA(0,0)=c0*coeff;
    for(int i=0 ; i<dim; i++)
    {
        A(i+1,0)=      normal[i]*coeff;
        A(0,i+1)=c0*c0*normal[i]*coeff;
        for( int j=0 ; j<dim; j++)
            absA(i+1,j+1)=c0*normal[i]*normal[j]*coeff;
    }
    return (A - absA)*(1./2);
}
    
void computeDivergenceMatrix(Mesh my_mesh, Mat * implMat, double dt)
{
    int nbCells = my_mesh.getNumberOfCells();
    int dim=my_mesh.getMeshDimension();
    int nbComp=dim+1;
    Vector normal(dim);

    Matrix idMoinsJacCL(nbComp);
    
    for(int j=0; j<nbCells; j++)//On parcourt les cellules
    {
        Cell Cj = my_mesh.getCell(j);
        int nbFaces = Cj.getNumberOfFaces();

        for(int k=0; k<nbFaces; k++)
        {
            int indexFace = Cj.getFacesId()[k];
            Face Fk = my_mesh.getFace(indexFace);
            for(int i =0; i<dim ; i++)
                normal[i] = Cj.getNormalVector(k, i);//normale sortante

            Matrix Am=jacobianMatrices( normal,dt*Fk.getMeasure()/Cj.getMeasure());

            int cellAutre =-1;
            if ( not Fk.isBorder())
            {
                /* hypothese: La cellule d'index indexC1 est la cellule courante index j */
                if (Fk.getCellsId()[0] == j) 
                    // hypothese verifiée 
                    cellAutre = Fk.getCellsId()[1];
                else if(Fk.getCellsId()[1] == j) 
                    // hypothese non verifiée 
                    cellAutre = Fk.getCellsId()[0];
                else
                    throw CdmathException("computeDivergenceMatrix: problem with mesh, unknown cell number");
                    
                addValue(j*nbComp,cellAutre*nbComp,Am      ,implMat);
                addValue(j*nbComp,        j*nbComp,Am*(-1.),implMat);
            }
            else 
            {    
                if( Fk.getGroupName() != "Periodic" && Fk.getGroupName() != "Neumann")//Wall boundary condition unless Periodic/Neumann specified explicitly
                {
                    Vector v(dim+1);
                    for(int i=0; i<dim; i++)
                        v[i+1]=normal[i];
                    Matrix idMoinsJacCL=v.tensProduct(v)*2;
                    
                    addValue(j*nbComp,j*nbComp,Am*(-1.)*idMoinsJacCL,implMat);
                }
                else if( Fk.getGroupName() == "Periodic")//Periodic boundary condition
                {
                    int indexFP=my_mesh.getIndexFacePeriodic(indexFace);
                    Face Fp = my_mesh.getFace(indexFP);
                    cellAutre = Fp.getCellsId()[0];
                    
                    addValue(j*nbComp,cellAutre*nbComp,Am      ,implMat);
                    addValue(j*nbComp,        j*nbComp,Am*(-1.),implMat);
                }
                else if(Fk.getGroupName() != "Neumann")//Nothing to do for Neumann boundary condition
                {
                    cout<< Fk.getGroupName() <<endl;
                    throw CdmathException("computeDivergenceMatrix: Unknown boundary condition name");
                }
            }
        }   
    }     
}
