//============================================================================
// Author      : Michael NDJINGA
// Date        : November 2020
// Description : multiD linear wave system run in parallel with calls to MPI
//               Test used in the MPI version of the Salome platform
//============================================================================

#include <string>

#include "Field.hxx"

#include <petscksp.h>

using namespace std;

double p0  =155e5;   //reference pressure in a pressurised nuclear vessel
double c0  =700.;    //reference sound speed for water at 155 bars, 600K
double rho0=p0/c0*c0;//reference density
double precision=1e-5;


void initial_conditions_shock(Mesh my_mesh,Field& pressure_field,Field& velocity_field);

void addValue( int i, int j, Matrix M, Mat * mat );

Matrix jacobianMatrices(Vector normal, double coeff);
    
void computeDivergenceMatrix(Mesh my_mesh, Mat * implMat, double dt);

void WaveSystem(double tmax, int ntmax, double cfl, int output_freq, const Mesh& my_mesh, const string file, int rank, int size, string resultDirectory);

