//                                MFEM Example 6 Extended
//
// Compile with: make ex6_amr
//
// Sample runs:  ./ex6_extended -m ../data/square-disc.mesh -o 1
//               ./ex6_extended -m ../data/square-disc.mesh -o 2
//               ./ex6_extended -m ../data/star.mesh -o 3
//               ./ex6_extended -m ../data/disc-nurbs.mesh -o 2
//               ./ex6_extended -m ../data/square-disc-surf.mesh -o 2
//               ./ex6_extended -m ../data/amr-quad.mesh
//
// Description:  This is a Poisson Equation with a simple adaptive mesh
//               refinement loop. The problem being solved is, as expected,
//               -Delta u = f with homogeneous Dirichlet boundary
//               conditions, where f = -4*(x²+y²-1)/(x²+y²+1)³. The problem
//               is solved on a sequence of meshes which are locally refined
//               in a conforming (triangles, tetrahedrons) or non-conforming
//               (quadrilaterals, hexahedra) manner according to a simple ZZ
//               error estimator. This example requires MFEM installed in the
//               working directory.
//
//               We recommend viewing Example 6 before viewing this example.

#include "mfem.hpp"
#include <fstream>
#include <iostream>
#include <cmath>
#include <chrono>

using namespace std;
using namespace mfem;

double exact_solution(const Vector &x);
double f_function(const Vector &x);

int main(int argc, char *argv[])
{
   // 1. Define .txt file to stream the logs
   std::ofstream fout("ex6_extended_logs.txt");

   // 2. Parse command-line options.
   const char *mesh_file = "../data/star.mesh";
   int order = 1;
   bool pa = false;
   const char *device_config = "cpu";
   bool visualization = true;

   OptionsParser args(argc, argv);
   args.AddOption(&mesh_file, "-m", "--mesh",
                  "Mesh file to use.");
   args.AddOption(&order, "-o", "--order",
                  "Finite element order (polynomial degree).");
   args.AddOption(&pa, "-pa", "--partial-assembly", "-no-pa",
                  "--no-partial-assembly", "Enable Partial Assembly.");
   args.AddOption(&device_config, "-d", "--device",
                  "Device configuration string, see Device::Configure().");
   args.AddOption(&visualization, "-vis", "--visualization", "-no-vis",
                  "--no-visualization",
                  "Enable or disable GLVis visualization.");
   args.Parse();
   if (!args.Good())
   {
      args.PrintUsage(cout);
      return 1;
   }
   args.PrintOptions(cout);

   // 3. Enable hardware devices such as GPUs, and programming models such as
   //    CUDA, OCCA, RAJA and OpenMP based on command line options.
   Device device(device_config);
   device.Print();

   // 4. Read the mesh from the given mesh file. We can handle triangular,
   //    quadrilateral, tetrahedral, hexahedral, surface and volume meshes with
   //    the same code.
   Mesh mesh(mesh_file, 1, 1);
   int dim = mesh.Dimension();
   int sdim = mesh.SpaceDimension();

   // 5. Since a NURBS mesh can currently only be refined uniformly, we need to
   //    convert it to a piecewise-polynomial curved mesh. First we refine the
   //    NURBS mesh a bit more and then project the curvature to quadratic Nodes.
   if (mesh.NURBSext)
   {
      // This is made because FEM method needs certain minimum amount of elements
      // to work
      for (int i = 0; i < 2; i++)
      {
         mesh.UniformRefinement();
      }
      mesh.SetCurvature(2);
   }

   // 6. Define a finite element space on the mesh. The polynomial order is
   //    one (linear) by default, but this can be changed on the command line.
   H1_FECollection fec(order, dim);
   FiniteElementSpace fespace(&mesh, &fec);

   // 7. As in Example 6, we set up bilinear and linear forms corresponding to
   //    the Poisson problem -\Delta u = f. We don't assemble the discrete
   //    problem yet, this will be done in the main loop.
   BilinearForm a(&fespace);
   if (pa) { a.SetAssemblyLevel(AssemblyLevel::PARTIAL); }
   LinearForm b(&fespace);

   ConstantCoefficient one(1.0);
   ConstantCoefficient zero(0.0);

   // Define the function f of the Poisson problem
   FunctionCoefficient func_coef (f_function);

   BilinearFormIntegrator *integ = new DiffusionIntegrator(one);
   a.AddDomainIntegrator(integ);
   b.AddDomainIntegrator(new DomainLFIntegrator(func_coef));

   // 8. The solution vector x and the associated finite element grid function
   //    will be maintained over the AMR iterations. We initialize it to zero.
   GridFunction x(&fespace);
   x = 0.0;

   // Define the FunctionCofficient of the exact solution, this is u = 1/(1+x²+y²)
   // This is made to then compare the solution obtained with the exact one
   FunctionCoefficient solution_coef(exact_solution);

   // 9. All boundary attributes will be used for essential (Dirichlet) BC.
   MFEM_VERIFY(mesh.bdr_attributes.Size() > 0,
               "Boundary attributes required in the mesh.");
   Array<int> ess_bdr(mesh.bdr_attributes.Max());
   ess_bdr = 1;

   // 10. Connect to GLVis.
   char vishost[] = "localhost";
   int  visport   = 19916;
   socketstream sol_sock;
   if (visualization)
   {
      sol_sock.open(vishost, visport);
   }

   // 11. Set up an error estimator. Here we use the Zienkiewicz-Zhu estimator
   //     that uses the ComputeElementFlux method of the DiffusionIntegrator to
   //     recover a smoothed flux (gradient) that is subtracted from the element
   //     flux to get an error indicator. We need to supply the space for the
   //     smoothed flux: an (H1)^sdim (i.e., vector-valued) space is used here.
   FiniteElementSpace flux_fespace(&mesh, &fec, sdim);
   ZienkiewiczZhuEstimator estimator(*integ, x, flux_fespace);
   estimator.SetAnisotropic();

   // 12. A refiner selects and refines elements based on a refinement strategy.
   //     The strategy here is to refine elements with errors larger than a
   //     fraction of the maximum element error. Other strategies are possible.
   //     The refiner will call the given error estimator.
   ThresholdRefiner refiner(estimator);
   refiner.SetTotalErrorFraction(0.7);

   // 13. The main AMR loop. In each iteration we solve the problem on the
   //     current mesh, visualize the solution, and refine the mesh.

   // Start the timer
   auto start = std::chrono::high_resolution_clock::now();
   const int max_dofs = 50000;
   for (int it = 0; ; it++)
   {
      int cdofs = fespace.GetTrueVSize();
      fout << "\nAMR iteration " << it << endl;
      fout << "Number of unknowns: " << cdofs << endl;

      // 14. Assemble the right-hand side.
      b.Assemble();

      // 15. Set Dirichlet boundary values in the GridFunction x.
      //     Determine the list of Dirichlet true DOFs in the linear system.
      Array<int> ess_tdof_list;
      x.ProjectBdrCoefficient(zero, ess_bdr);
      fespace.GetEssentialTrueDofs(ess_bdr, ess_tdof_list);

      // 16. Assemble the stiffness matrix.
      a.Assemble();

      // 17. Create the linear system: eliminate boundary conditions, constrain
      //     hanging nodes and possibly apply other transformations. The system
      //     will be solved for true (unconstrained) DOFs only.
      OperatorPtr A;
      Vector B, X;

      const int copy_interior = 1;
      a.FormLinearSystem(ess_tdof_list, x, b, A, X, B, copy_interior);

      // 18. Solve the linear system A X = B.
      if (!pa)
      {
#ifndef MFEM_USE_SUITESPARSE
         // Use a simple symmetric Gauss-Seidel preconditioner with PCG.
         GSSmoother M((SparseMatrix&)(*A));
         PCG(*A, M, B, X, 3, 200, 1e-12, 0.0);
#else
         // If MFEM was compiled with SuiteSparse, use UMFPACK to solve the system.
         UMFPackSolver umf_solver;
         umf_solver.Control[UMFPACK_ORDERING] = UMFPACK_ORDERING_METIS;
         umf_solver.SetOperator(*A);
         umf_solver.Mult(B, X);
#endif
      }
      else // No preconditioning for now in partial assembly mode.
      {
         CG(*A, B, X, 3, 2000, 1e-12, 0.0);
      }

      // 19. After solving the linear system, reconstruct the solution as a
      //     finite element GridFunction. Constrained nodes are interpolated
      //     from true DOFs (it may therefore happen that x.Size() >= X.Size()).
      a.RecoverFEMSolution(X, b, x);

      // Export the results to Paraview in step 5 and 16
      if (it == 5){
         ParaViewDataCollection paraview_dc("AMR_IT5", &mesh);
         paraview_dc.SetPrefixPath("ParaView");
         paraview_dc.SetLevelsOfDetail(order);
         paraview_dc.SetCycle(0);
         paraview_dc.SetDataFormat(VTKFormat::BINARY);
         paraview_dc.SetHighOrderOutput(true);
         paraview_dc.SetTime(0.0); // set the time
         paraview_dc.RegisterField("solution", &x);
         paraview_dc.Save();
      }else if(it == 16){
         ParaViewDataCollection paraview_dc("AMR_IT16", &mesh);
         paraview_dc.SetPrefixPath("ParaView");
         paraview_dc.SetLevelsOfDetail(order);
         paraview_dc.SetCycle(0);
         paraview_dc.SetDataFormat(VTKFormat::BINARY);
         paraview_dc.SetHighOrderOutput(true);
         paraview_dc.SetTime(0.0); // set the time
         paraview_dc.RegisterField("solution", &x);
         paraview_dc.Save();
      }

      // 20. Send solution by socket to the GLVis server.
      if (visualization && sol_sock.good())
      {
         sol_sock.precision(8);
         sol_sock << "solution\n" << mesh << x << flush;
      }

      // Compute the error with the exact solution defined previously using
      // L_1 and L_2 norms
      fout << std::fixed << std::showpoint;
      fout << std::setprecision(12);
      fout << "L_1 Error: " << x.ComputeL1Error(solution_coef) <<'\n';
      fout << "L_2 Error: " << x.ComputeL2Error(solution_coef) <<'\n';

      if (cdofs > max_dofs)
      {
         fout << "Reached the maximum number of dofs. Stop." << endl;
         break;
      }

      // 21. Call the refiner to modify the mesh. The refiner calls the error
      //     estimator to obtain element errors, then it selects elements to be
      //     refined and finally it modifies the mesh. The Stop() method can be
      //     used to determine if a stopping criterion was met.
      refiner.Apply(mesh);
      if (refiner.Stop())
      {
         fout << "Stopping criterion satisfied. Stop." << endl;
         break;
      }

      // 22. Update the space to reflect the new state of the mesh. Also,
      //     interpolate the solution x so that it lies in the new space but
      //     represents the same function. This saves solver iterations later
      //     since we'll have a good initial guess of x in the next step.
      //     Internally, FiniteElementSpace::Update() calculates an
      //     interpolation matrix which is then used by GridFunction::Update().
      fespace.Update();
      x.Update();

      // 23. Inform also the bilinear and linear forms that the space has
      //     changed.
      a.Update();
      b.Update();
   }

   // Stop the timer and output the duration time
   auto stop = std::chrono::high_resolution_clock::now();
   auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
   fout << duration.count() << endl;

   // Export the solution in its last step to Paraview
   ParaViewDataCollection paraview_dc("AMR_Final", &mesh);
   paraview_dc.SetPrefixPath("ParaView");
   paraview_dc.SetLevelsOfDetail(order);
   paraview_dc.SetCycle(0);
   paraview_dc.SetDataFormat(VTKFormat::BINARY);
   paraview_dc.SetHighOrderOutput(true);
   paraview_dc.SetTime(0.0); // set the time
   paraview_dc.RegisterField("solution",&x);
   paraview_dc.Save();

   return 0;
}

// Exact solution u
double exact_solution(const Vector &x){
   return (1.0) / (1 + pow(x(0), 2) + pow(x(1), 2));
}

// Define the function f of Poisson problem
double f_function(const Vector &x){
   return ((-4.0)*(pow(x(0), 2) + pow(x(1), 2) - 1)) / pow((pow(x(0), 2) + pow(x(1), 2) + 1), 3);
}
