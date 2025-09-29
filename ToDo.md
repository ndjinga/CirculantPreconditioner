- vérifier que tout se branche bien : le test TransportEquation_SphericalExplosion_impl_mpi.cxx crée un préconditioneur implémenté dans PCSHELLFFT.cxx qui fait appel aux structures du solveur direct FFT implémentées dans FFTLinearSolver.c.

- Noms de fonctions : vérifier qu'ils contiennent 'Transport' quand la fonction ne s'applique qu'à l'équation de transport et 1D/2D/3D s'il ne s'applique qu'à une dimension donnée
  cas du fichier FftLinearSolver_3D.* qui peuvent s'appliquer en 1D/2D en complétant avec ny=1, nz=1. Il faut peut être les renommer en FftLinearSolver.*. En effet je crois qu'il n'y a plus intérêt à créer des fichiers FftLinearSolver_1D et FftLinearSolver_2D car ils ne seront pas plus rapide puisqu'il me semble que c'est le produit nx*ny*nz qui détermine le coût de calcul.

- ajouter l'équation de diffusion pour tester la versatilité de la structure : on réutilise la FFT mais il faut créer une structure StructuredDiffusionContext et une structure FFTPrecDiffusionContext

- ajouter des solutions exactes du transport et de la diffusion pour vérifier les solutions issues de la simulation et ainsi détecter les erreurs mathématiques

- ajouter le système des ondes pour voir (enfin) l'effet de notre préconditioneur sur les points selle
  
- brancher la projection d'un maillage sur l'autre avec l'interpolateur. Le truc c'est que la matrice de projection issue de MEDCoupling a un format map<int,vector<double>> (liste des coefficients non nulls de chaque ligne) qu'il faut convertir en matrice PETSc. A moins de récoder la fonction getCrudeMatrix de MEDCoupling pour qu'elle retourne directement une matrice PETSc.


