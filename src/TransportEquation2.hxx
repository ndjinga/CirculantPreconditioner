//============================================================================
// Author      : Michaël NDJINGA
// Date        : Août 2025
// Description : multiD linear transport equation run in parallel with calls to MPI
//               \partial_t u + \vec a \cdot \vec\nabla u = 0
//============================================================================

#include <string>

#include "Field.hxx"

#include <petscksp.h>

using namespace std;


void initial_conditions_shock(Mesh my_mesh, Field& Temperature_field);

void computeDivergenceMatrix(Mesh my_mesh, Mat * implMat, double dt, Vector vitesseTransport);

void TransportEquation(double tmax, int ntmax, double cfl, int output_freq, const Mesh& my_mesh, const string file, int rank, int size, string resultDirectory, Vector vitesseTransport);

