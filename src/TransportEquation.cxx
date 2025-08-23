//============================================================================
// Author      : Michael NDJINGA
// Date        : Août 2025
// Description : multiD linear transport equation run in parallel with calls to MPI
//               \partial_t u + \vec a \cdot \vec\nabla u = 0
//============================================================================

#include <iostream>
#include <string>
#include <cmath>

#include "TransportEquation2.hxx"

#include <petscksp.h>

#include "Mesh.hxx"
#include "Cell.hxx"
#include "Face.hxx"
#include "Field.hxx"
#include "CdmathException.hxx"


using namespace std;

void initial_conditions_shock(Mesh my_mesh,Field& Temperature_field)
{
    double x, y, z;
    double val, valX, valY, valZ;
    
    int dim    =my_mesh.getMeshDimension();
    int nbCells=my_mesh.getNumberOfCells();

    double r, r2, rmax=0.3;
    
    double xcentre = ( my_mesh.getXMin()+my_mesh.getXMax() )/2;
    double ycentre;
    double zcentre;

    if(dim>1)
    {
        ycentre = ( my_mesh.getYMin()+my_mesh.getYMax() )/2;
        if(dim==3)
            zcentre = ( my_mesh.getZMin()+my_mesh.getZMax() )/2;
    }
    /*
    cout<< "XMin= "<< my_mesh.getXMin() << ", XMax= "<< my_mesh.getXMax() <<endl;
    cout<< "YMin= "<< my_mesh.getYMin() << ", YMax= "<< my_mesh.getYMax() <<endl;
    cout<< "ZMin= "<< my_mesh.getZMin() << ", ZMax= "<< my_mesh.getZMax() <<endl;
    cout<< "Xcentre= "<< xcentre << "Ycentre= "<< ycentre<< "zcentre= "<< zcentre <<endl;
    */
    for (int j=0 ; j<nbCells ; j++)
    {
        x = my_mesh.getCell(j).x() ;
        r2=(x-xcentre)*(x-xcentre);
        if(dim>1)
        {
            y = my_mesh.getCell(j).y() ;
            r2+=(y-ycentre)*(y-ycentre);
            if(dim==3)
            {
                z = my_mesh.getCell(j).z() ;
                r2+=(z-zcentre)*(z-zcentre);                
            }
        }

        r=sqrt(r2);
        
        if (r<rmax)
            Temperature_field(j) = 650;
        else
            Temperature_field(j) = 600;
    }
}

void computeDivergenceMatrix(Mesh my_mesh, Mat * implMat, double dt, Vector vitesseTransport)
{
    int nbCells = my_mesh.getNumberOfCells();
    int dim=my_mesh.getMeshDimension();
    Vector normal(dim);
    double un;

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

			un=normal*vitesseTransport;

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
                    
				if(un>0)
					MatSetValue(*implMat,j,j        , dt*Fk.getMeasure()/Cj.getMeasure()*un, ADD_VALUES);
				else
					MatSetValue(*implMat,j,cellAutre,-dt*Fk.getMeasure()/Cj.getMeasure()*un, ADD_VALUES);
            }
            else 
            {    
                if( Fk.getGroupName() == "Periodic")//Periodic boundary condition
                {
                    int indexFP=my_mesh.getIndexFacePeriodic(indexFace);
                    Face Fp = my_mesh.getFace(indexFP);
                    cellAutre = Fp.getCellsId()[0];
                    
				if(un>0)
					MatSetValue(*implMat,j,j        , dt*Fk.getMeasure()/Cj.getMeasure()*un, ADD_VALUES);
				else
					MatSetValue(*implMat,j,cellAutre,-dt*Fk.getMeasure()/Cj.getMeasure()*un, ADD_VALUES);
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
