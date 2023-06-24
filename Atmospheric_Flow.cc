/* Author: Giuseppe Orlando, 2023. */

// @sect{Include files}

// We start by including all the necessary deal.II header files
//
#include <deal.II/base/parallel.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/affine_constraints.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/manifold_lib.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>

#include <deal.II/base/timer.h>

#include <deal.II/fe/mapping_q.h>

#include "euler_operator.h"

using namespace Atmospheric_Flow;

// @sect{The <code>EulerSolver</code> class}

// Now for the main class of the program. It implements the solver for the
// Euler equations using the discretization previously implemented.
//
template<int dim>
class EulerSolver {
public:
  EulerSolver(RunTimeParameters::Data_Storage& data); /*--- Class constructor ---*/

  void run(const bool verbose = false, const unsigned int output_interval = 10);
  /*--- The run function which actually runs the simulation ---*/

protected:
  const double t_0;       /*--- Initial time auxiliary variable ----*/
  const double T;         /*--- Final time auxiliary variable ----*/
  unsigned int SSP_stage; /*--- Stage for SSP scheme ---*/
  const double Ma;        /*--- Mach number auxiliary variable ----*/
  double       dt;        /*--- Time step auxiliary variable ---*/

  parallel::distributed::Triangulation<dim> triangulation; /*--- The variable which stores the mesh ---*/

  /*--- Finite element spaces for all the variables ---*/
  FESystem<dim> fe_density;
  FESystem<dim> fe_momentum;
  FESystem<dim> fe_energy;

  /*--- Degrees of freedom handlers for all the variables ---*/
  DoFHandler<dim> dof_handler_density;
  DoFHandler<dim> dof_handler_momentum;
  DoFHandler<dim> dof_handler_energy;

  /*--- Auxiliary quadratures for all the variables ---*/
  QGaussLobatto<dim> quadrature_density;
  QGaussLobatto<dim> quadrature_momentum;
  QGaussLobatto<dim> quadrature_energy;

  /*--- Variables for the density ---*/
  LinearAlgebra::distributed::Vector<double> rho_old;
  LinearAlgebra::distributed::Vector<double> rho_tmp_2;
  LinearAlgebra::distributed::Vector<double> rho_tmp_3;
  LinearAlgebra::distributed::Vector<double> rho_curr;
  LinearAlgebra::distributed::Vector<double> rhs_rho;

  /*--- Variables for the momentum ---*/
  LinearAlgebra::distributed::Vector<double> rhou_old;
  LinearAlgebra::distributed::Vector<double> rhou_tmp_2;
  LinearAlgebra::distributed::Vector<double> rhou_tmp_3;
  LinearAlgebra::distributed::Vector<double> rhou_curr;
  LinearAlgebra::distributed::Vector<double> rhs_rhou;

  /*--- Variables for the energy ---*/
  LinearAlgebra::distributed::Vector<double> rhoE_old;
  LinearAlgebra::distributed::Vector<double> rhoE_tmp_2;
  LinearAlgebra::distributed::Vector<double> rhoE_tmp_3;
  LinearAlgebra::distributed::Vector<double> rhoE_curr;
  LinearAlgebra::distributed::Vector<double> rhs_rhoE;

  DeclException2(ExcInvalidTimeStep,
                 double,
                 double,
                 << " The time step " << arg1 << " is out of range."
                 << std::endl
                 << " The permitted range is (0," << arg2 << "]");

  void create_triangulation(const unsigned int n_refines); /*--- Function to create the grid ---*/

  void setup_dofs(); /*--- Function to set the dofs ---*/

  void initialize(); /*--- Function to initialize the fields ---*/

  void update_density(); /*--- Function to update the density ---*/

  void update_momentum(); /*--- Function to update the momentum ---*/

  void update_energy(); /*--- Function to update the energy ---*/

  void output_results(const unsigned int step); /*--- Function to save the results ---*/

private:
  EquationData::Density<dim>  rho_init;
  EquationData::Momentum<dim> rhou_init;
  EquationData::Energy<dim>   rhoE_init;

  /*--- Auxiliary structures for the matrix-free and for the multigrid ---*/
  std::shared_ptr<MatrixFree<dim, double>> matrix_free_storage;

  EULEROperator<dim, EquationData::degree_rho, EquationData::degree_T, EquationData::degree_u,
                2*EquationData::degree_rho + 1, 2*EquationData::degree_T + 1, 2*EquationData::degree_u + 1,
                LinearAlgebra::distributed::Vector<double>, double> euler_matrix;

  std::vector<const DoFHandler<dim>*> dof_handlers; /*--- Auxiliary container for the matrix-free ---*/

  std::vector<const AffineConstraints<double>*> constraints; /*--- Auxiliary container for the matrix-free ---*/
  AffineConstraints<double> constraints_momentum,
                            constraints_energy,
                            constraints_density;

  std::vector<QGauss<1>> quadratures;  /*--- Auxiliary container for the quadrature in matrix-free ---*/

  unsigned int max_its; /*--- Auxiliary variable for the maximum number of iterations of linear solvers ---*/
  double       eps;     /*--- Auxiliary variable for the tolerance of linear solvers ---*/

  std::string saving_dir; /*--- Auxiliary variable for the directory to save the results ---*/

  /*--- Now we declare a bunch of variables for text output ---*/
  ConditionalOStream pcout;

  std::ofstream      time_out;
  ConditionalOStream ptime_out;
  TimerOutput        time_table;

  std::ofstream output_n_dofs_momentum;
  std::ofstream output_n_dofs_energy;
  std::ofstream output_n_dofs_density;

  double get_minimal_density(); /*--- Get minimal density ---*/

  double get_maximal_density(); /*--- Get maximal density ---*/

  double get_maximal_velocity(); /*--- Get maximal velocity ---*/

  double compute_max_celerity(); /*--- Compute maximum speed of sound so as to compute the acoustic Courant number ---*/
};


// @sect{ <code>EulerSolver::EulerSolver</code> }

// In the constructor, we just read all the data from the
// <code>Data_Storage</code> object that is passed as an argument, verify that
// the data we read are reasonable and, finally, create the triangulation and
// load the initial data.
//
template<int dim>
EulerSolver<dim>::EulerSolver(RunTimeParameters::Data_Storage& data):
  t_0(data.initial_time),
  T(data.final_time),
  SSP_stage(1),
  Ma(data.Mach),
  dt(data.dt),
  triangulation(MPI_COMM_WORLD, parallel::distributed::Triangulation<dim>::limit_level_difference_at_vertices,
                parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy),
  fe_density(FE_DGQ<dim>(EquationData::degree_rho), 1),
  fe_momentum(FE_DGQ<dim>(EquationData::degree_u), dim),
  fe_energy(FE_DGQ<dim>(EquationData::degree_T), 1),
  dof_handler_density(triangulation),
  dof_handler_momentum(triangulation),
  dof_handler_energy(triangulation),
  quadrature_density(EquationData::degree_rho + 1),
  quadrature_momentum(EquationData::degree_u + 1),
  quadrature_energy(EquationData::degree_T + 1),
  rho_init(data.initial_time),
  rhou_init(data.initial_time),
  rhoE_init(data.initial_time),
  euler_matrix(data),
  max_its(data.max_iterations),
  eps(data.eps),
  saving_dir(data.dir),
  pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0),
  time_out("./" + data.dir + "/time_analysis_" +
           Utilities::int_to_string(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)) + "proc.dat"),
  ptime_out(time_out, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0),
  time_table(ptime_out, TimerOutput::summary, TimerOutput::cpu_and_wall_times),
  output_n_dofs_momentum("./" + data.dir + "/n_dofs_momentum.dat", std::ofstream::out),
  output_n_dofs_energy("./" + data.dir + "/n_dofs_energy.dat", std::ofstream::out),
  output_n_dofs_density("./" + data.dir + "/n_dofs_density.dat", std::ofstream::out) {
    AssertThrow(!((dt <= 0.0) || (dt > 0.5*T)), ExcInvalidTimeStep(dt, 0.5*T));

    matrix_free_storage = std::make_shared<MatrixFree<dim, double>>();

    dof_handlers.clear();

    constraints.clear();
    constraints_momentum.clear();
    constraints_energy.clear();
    constraints_density.clear();

    quadratures.clear();

    create_triangulation(data.n_global_refines);
    setup_dofs();
    initialize();
  }


// @sect{<code>EulerSolver::create_triangulation_and_dofs</code>}

// The method that creates the triangulation.
//
template<int dim>
void EulerSolver<dim>::create_triangulation(const unsigned int n_refines) {
  TimerOutput::Scope t(time_table, "Create triangulation");

  GridGenerator::concentric_hyper_shells(triangulation, Point<dim>(), 0.9, 1.0, 1);

  triangulation.refine_global(n_refines);

  pcout << "h_min = " << std::sqrt(dim)/GridTools::minimal_cell_diameter(triangulation) << std::endl;
}


// After creating the triangulation, it creates the mesh dependent
// data, i.e. it distributes degrees of freedom and renumbers them, and
// initializes the matrices and vectors that we will use.
//
template<int dim>
void EulerSolver<dim>::setup_dofs() {
  pcout << "Number of active cells: " << triangulation.n_global_active_cells() << std::endl;
  pcout << "Number of levels: "       << triangulation.n_global_levels()       << std::endl;

  /*--- Set degrees of freedom ---*/
  dof_handler_momentum.distribute_dofs(fe_momentum);
  dof_handler_energy.distribute_dofs(fe_energy);
  dof_handler_density.distribute_dofs(fe_density);

  pcout << "dim (V_h) = " << dof_handler_momentum.n_dofs()
        << std::endl
        << "dim (Q_h) = " << dof_handler_energy.n_dofs()
        << std::endl
        << "dim (X_h) = " << dof_handler_density.n_dofs()
        << std::endl
        << "Ma        = " << Ma << std::endl
        << std::endl;

  /*--- Save the number of degrees of freedom just for info ---*/
  if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0) {
    output_n_dofs_momentum << dof_handler_momentum.n_dofs() << std::endl;
    output_n_dofs_energy   << dof_handler_energy.n_dofs()   << std::endl;
    output_n_dofs_density  << dof_handler_density.n_dofs()  << std::endl;
  }

  /*--- Set additional data to check which variables neeed to be updated ---*/
  typename MatrixFree<dim, double>::AdditionalData additional_data;
  additional_data.mapping_update_flags                = (update_values | update_JxW_values | update_quadrature_points);
  additional_data.mapping_update_flags_inner_faces    = (update_values | update_JxW_values | update_quadrature_points |
                                                         update_normal_vectors);
  additional_data.mapping_update_flags_boundary_faces = (update_values | update_JxW_values | update_quadrature_points |
                                                         update_normal_vectors);
  additional_data.tasks_parallel_scheme               = MatrixFree<dim, double>::AdditionalData::none;

  /*--- Set the container with the dof handlers ---*/
  dof_handlers.push_back(&dof_handler_momentum);
  dof_handlers.push_back(&dof_handler_energy);
  dof_handlers.push_back(&dof_handler_density);

  /*--- Set the container with the constraints. Each entry is empty (no Dirichlet and weak imposition in general)
        and this is necessary only for compatibilty reasons ---*/
  constraints.push_back(&constraints_momentum);
  constraints.push_back(&constraints_energy);
  constraints.push_back(&constraints_density);

  /*--- Set the quadrature formula to compute the integrals for assembling bilinear and linear forms ---*/
  quadratures.push_back(QGauss<1>(2*EquationData::degree_u + 1));

  /*--- Initialize the matrix-free structure with DofHandlers, Constraints, Quadratures and AdditionalData ---*/
  matrix_free_storage->reinit(MappingQ<dim>(EquationData::degree_u), dof_handlers, constraints, quadratures, additional_data);

  /*--- Initialize the variables related to the velocity ---*/
  matrix_free_storage->initialize_dof_vector(rhou_old, 0);
  matrix_free_storage->initialize_dof_vector(rhou_tmp_2, 0);
  matrix_free_storage->initialize_dof_vector(rhou_tmp_3, 0);
  matrix_free_storage->initialize_dof_vector(rhou_curr, 0);
  matrix_free_storage->initialize_dof_vector(rhs_rhou, 0);

  /*--- Initialize the variables related to the pressure ---*/
  matrix_free_storage->initialize_dof_vector(rhoE_old, 1);
  matrix_free_storage->initialize_dof_vector(rhoE_tmp_2, 1);
  matrix_free_storage->initialize_dof_vector(rhoE_tmp_3, 1);
  matrix_free_storage->initialize_dof_vector(rhoE_curr, 1);
  matrix_free_storage->initialize_dof_vector(rhs_rhoE, 1);

  /*--- Initialize the variables related to the density ---*/
  matrix_free_storage->initialize_dof_vector(rho_old, 2);
  matrix_free_storage->initialize_dof_vector(rho_tmp_2, 2);
  matrix_free_storage->initialize_dof_vector(rho_tmp_3, 2);
  matrix_free_storage->initialize_dof_vector(rho_curr, 2);
  matrix_free_storage->initialize_dof_vector(rhs_rho, 2);
}


// @sect{ <code>EulerSolver::initialize</code> }

// This method loads the initial data
//
template<int dim>
void EulerSolver<dim>::initialize() {
  TimerOutput::Scope t(time_table, "Initialize state");

  VectorTools::interpolate(MappingQ<dim>(EquationData::degree_u), dof_handler_density, rho_init, rho_old);
  VectorTools::interpolate(MappingQ<dim>(EquationData::degree_u), dof_handler_momentum, rhou_init, rhou_old);
  VectorTools::interpolate(MappingQ<dim>(EquationData::degree_u), dof_handler_energy, rhoE_init, rhoE_old);
}


// @sect{<code>EulerSolver::update_density</code>}

// This implements the update of the density for the hyperbolic part
//
template<int dim>
void EulerSolver<dim>::update_density() {
  TimerOutput::Scope t(time_table, "Update density");

  const std::vector<unsigned int> tmp = {2};
  euler_matrix.initialize(matrix_free_storage, tmp, tmp);
  euler_matrix.set_NS_stage(1);

  if(SSP_stage == 1) {
    euler_matrix.vmult_rhs_rho_update(rhs_rho, {rho_old, rhou_old, rhoE_old});
  }
  else if(SSP_stage == 2) {
    euler_matrix.vmult_rhs_rho_update(rhs_rho, {rho_tmp_2, rhou_tmp_2, rhoE_tmp_2});
  }
  else {
    euler_matrix.vmult_rhs_rho_update(rhs_rho, {rho_tmp_3, rhou_tmp_3, rhoE_tmp_3});
  }

  SolverControl solver_control(max_its, eps*rhs_rho.l2_norm());
  SolverCG<LinearAlgebra::distributed::Vector<double>> cg(solver_control);

  /*--- Solve the system for the density ---*/
  if(SSP_stage == 1) {
    rho_tmp_2.equ(1.0, rho_old);
    cg.solve(euler_matrix, rho_tmp_2, rhs_rho, PreconditionIdentity());
  }
  else if(SSP_stage == 2) {
    rho_tmp_3.equ(1.0, rho_tmp_2);
    cg.solve(euler_matrix, rho_tmp_3, rhs_rho, PreconditionIdentity());

    rho_tmp_3 *= 0.25;
    rho_tmp_3.add(0.75, rho_old);
  }
  else {
    rho_curr.equ(1.0, rho_tmp_3);
    cg.solve(euler_matrix, rho_curr, rhs_rho, PreconditionIdentity());

    rho_curr *= 2.0/3.0;
    rho_curr.add(1.0/3.0, rho_old);
  }
}


// This implements the update of the momentum for the hyperbolic part
//
template<int dim>
void EulerSolver<dim>::update_momentum() {
  TimerOutput::Scope t(time_table, "Update momentum");

  const std::vector<unsigned int> tmp = {0};

  euler_matrix.initialize(matrix_free_storage, tmp, tmp);
  euler_matrix.set_NS_stage(2);

  if(SSP_stage == 1) {
    euler_matrix.vmult_rhs_momentum_update(rhs_rhou, {rho_old, rhou_old, rhoE_old});
  }
  else if(SSP_stage == 2) {
    euler_matrix.vmult_rhs_momentum_update(rhs_rhou, {rho_tmp_2, rhou_tmp_2, rhoE_tmp_2});
  }
  else {
    euler_matrix.vmult_rhs_momentum_update(rhs_rhou, {rho_tmp_3, rhou_tmp_3, rhoE_tmp_3});
  }

  SolverControl solver_control(max_its, eps*rhs_rhou.l2_norm());
  SolverCG<LinearAlgebra::distributed::Vector<double>> cg(solver_control);

  /*--- Solve the system for the momentum ---*/
  if(SSP_stage == 1) {
    rhou_tmp_2.equ(1.0, rhou_old);
    cg.solve(euler_matrix, rhou_tmp_2, rhs_rhou, PreconditionIdentity());
  }
  else if(SSP_stage == 2) {
    rhou_tmp_3.equ(1.0, rhou_tmp_2);
    cg.solve(euler_matrix, rhou_tmp_3, rhs_rhou, PreconditionIdentity());

    rhou_tmp_3 *= 0.25;
    rhou_tmp_3.add(0.75, rhou_old);
  }
  else {
    rhou_curr.equ(1.0, rhou_tmp_3);
    cg.solve(euler_matrix, rhou_curr, rhs_rhou, PreconditionIdentity());

    rhou_curr *= 2.0/3.0;
    rhou_curr.add(1.0/3.0, rhou_old);
  }
}


// This implements the update of the energy for the hyperbolic part
//
template<int dim>
void EulerSolver<dim>::update_energy() {
  TimerOutput::Scope t(time_table, "Update energy");

  const std::vector<unsigned int> tmp = {1};

  euler_matrix.initialize(matrix_free_storage, tmp, tmp);
  euler_matrix.set_NS_stage(3);

  if(SSP_stage == 1) {
    euler_matrix.vmult_rhs_energy_update(rhs_rhoE, {rho_old, rhou_old, rhoE_old});
  }
  else if(SSP_stage == 2) {
    euler_matrix.vmult_rhs_energy_update(rhs_rhoE, {rho_tmp_2, rhou_tmp_2, rhoE_tmp_2});
  }
  else {
    euler_matrix.vmult_rhs_energy_update(rhs_rhoE, {rho_tmp_3, rhou_tmp_3, rhoE_tmp_3});
  }

  SolverControl solver_control(max_its, eps*rhs_rhoE.l2_norm());
  SolverCG<LinearAlgebra::distributed::Vector<double>> cg(solver_control);

  //--- Solve the system for the pressure
  if(SSP_stage == 1) {
    rhoE_tmp_2.equ(1.0, rhoE_old);
    cg.solve(euler_matrix, rhoE_tmp_2, rhs_rhoE, PreconditionIdentity());
  }
  else if(SSP_stage == 2) {
    rhoE_tmp_3.equ(1.0, rhoE_tmp_2);
    cg.solve(euler_matrix, rhoE_tmp_3, rhs_rhoE, PreconditionIdentity());

    rhoE_tmp_3 *= 0.25;
    rhoE_tmp_3.add(0.75, rhoE_old);
  }
  else {
    rhoE_curr.equ(1.0, rhoE_tmp_3);
    cg.solve(euler_matrix, rhoE_curr, rhs_rhoE, PreconditionIdentity());

    rhoE_curr *= 2.0/3.0;
    rhoE_curr.add(1.0/3.0, rhoE_old);
  }
}


// @sect{ <code>EulerSolver::output_results</code> }

// This method plots the current solution. The main difficulty is that we want
// to create a single output file that contains the data for all velocity
// components and the pressure. On the other hand, velocities and the pressure
// live on separate DoFHandler objects, and
// so can't be written to the same file using a single DataOut object. As a
// consequence, we have to work a bit harder to get the various pieces of data
// into a single DoFHandler object, and then use that to drive graphical
// output.
//
template<int dim>
void EulerSolver<dim>::output_results(const unsigned int step) {
  TimerOutput::Scope t(time_table, "Output results");

  DataOut<dim> data_out;

  rho_old.update_ghost_values();
  data_out.add_data_vector(dof_handler_density, rho_old, "rho", {DataComponentInterpretation::component_is_scalar});

  std::vector<std::string> momentum_names(dim, "rhou");
  std::vector<DataComponentInterpretation::DataComponentInterpretation>
  component_interpretation_momentum(dim, DataComponentInterpretation::component_is_part_of_vector);
  rhou_old.update_ghost_values();
  data_out.add_data_vector(dof_handler_momentum, rhou_old, momentum_names, component_interpretation_momentum);

  rhoE_old.update_ghost_values();
  data_out.add_data_vector(dof_handler_energy, rhoE_old, "rhoE", {DataComponentInterpretation::component_is_scalar});

  data_out.build_patches(MappingQ<dim>(EquationData::degree_u), EquationData::degree_u, DataOut<dim>::curved_inner_cells);

  const std::string output = "./" + saving_dir + "/solution-" + Utilities::int_to_string(step, 5) + ".vtu";
  data_out.write_vtu_in_parallel(output, MPI_COMM_WORLD);
}


// The following function is used in determining the minimal density
//
template<int dim>
double EulerSolver<dim>::get_minimal_density() {
  FEValues<dim> fe_values(fe_density, quadrature_density, update_values);
  std::vector<double> solution_values(quadrature_density.size());

  double min_local_density = std::numeric_limits<double>::max();

  for(const auto& cell: dof_handler_density.active_cell_iterators()) {
    if(cell->is_locally_owned()) {
      fe_values.reinit(cell);
      if(SSP_stage == 1) {
        fe_values.get_function_values(rho_tmp_2, solution_values);
      }
      else if(SSP_stage == 2) {
        fe_values.get_function_values(rho_tmp_3, solution_values);
      }
      else {
        fe_values.get_function_values(rho_curr, solution_values);
      }

      for(unsigned int q = 0; q < quadrature_density.size(); ++q) {
        min_local_density = std::min(min_local_density, solution_values[q]);
      }
    }
  }

  return Utilities::MPI::min(min_local_density, MPI_COMM_WORLD);
}


// The following function is used in determining the maximal density
//
template<int dim>
double EulerSolver<dim>::get_maximal_density() {
  if(SSP_stage == 1) {
    return rho_tmp_2.linfty_norm();
  }
  if(SSP_stage == 2) {
    return rho_tmp_3.linfty_norm();
  }

  return rho_curr.linfty_norm();
}


// The following function is used in determining the maximal velocity
//
template<int dim>
double EulerSolver<dim>::get_maximal_velocity() {
  const int n_q_points = quadrature_momentum.size();

  std::vector<double>         density_values(n_q_points);
  std::vector<Vector<double>> momentum_values(n_q_points, Vector<double>(dim));

  FEValues<dim> fe_values_momentum(fe_momentum, quadrature_momentum, update_values);
  FEValues<dim> fe_values_density(fe_density, quadrature_density, update_values);

  double max_local_velocity = std::numeric_limits<double>::min();

  auto tmp_cell = dof_handler_density.begin_active();
  for(const auto& cell : dof_handler_momentum.active_cell_iterators()) {
    if(cell->is_locally_owned()) {
      fe_values_momentum.reinit(cell);
      fe_values_density.reinit(tmp_cell);

      fe_values_momentum.get_function_values(rhou_old, momentum_values);
      fe_values_density.get_function_values(rho_old, density_values);

      for(int q = 0; q < n_q_points; q++) {
        max_local_velocity = std::max(max_local_velocity, std::sqrt(momentum_values[q]*momentum_values[q])/density_values[q]);
      }
    }
    ++tmp_cell;
  }

  const double max_velocity = Utilities::MPI::max(max_local_velocity, MPI_COMM_WORLD);

  return max_velocity;
}


// The following function is used in determining the maximal celerity
//
template<int dim>
double EulerSolver<dim>::compute_max_celerity() {
  const int n_q_points = quadrature_energy.size();

  std::vector<double> energy_values(n_q_points);
  std::vector<double> density_values(n_q_points);
  std::vector<Vector<double>> momentum_values(n_q_points, Vector<double>(dim));

  FEValues<dim> fe_values_energy(fe_energy, quadrature_energy, update_values);
  FEValues<dim> fe_values_density(fe_density, quadrature_density, update_values);
  FEValues<dim> fe_values_momentum(fe_momentum, quadrature_momentum, update_values);

  double max_local_celerity = std::numeric_limits<double>::min();

  auto tmp_cell     = dof_handler_density.begin_active();
  auto tmp_cell_bis = dof_handler_momentum.begin_active();
  for(const auto& cell: dof_handler_energy.active_cell_iterators()) {
    if(cell->is_locally_owned()) {
      fe_values_energy.reinit(cell);
      fe_values_density.reinit(tmp_cell);
      fe_values_momentum.reinit(tmp_cell_bis);

      fe_values_energy.get_function_values(rhoE_old, energy_values);
      fe_values_density.get_function_values(rho_old, density_values);
      fe_values_momentum.get_function_values(rhou_old, momentum_values);

      for(int q = 0; q < n_q_points; ++q) {
        const double rho   = density_values[q];
        const double pres  = (EquationData::Cp_Cv - 1.0)*
                             (energy_values[q] - 0.5*Ma*Ma*(momentum_values[q]*momentum_values[q])/rho);

        max_local_celerity = std::max(max_local_celerity, std::sqrt(EquationData::Cp_Cv*pres/rho));
      }
    }
    ++tmp_cell;
    ++tmp_cell_bis;
  }

  return Utilities::MPI::max(max_local_celerity, MPI_COMM_WORLD);
}


// @sect{ <code>EulerSolver::run</code> }

// This is the time marching function, which starting at <code>t_0</code>
// advances in time using the projection method with time step <code>dt</code>
// until <code>T</code>.
//
// Its second parameter, <code>verbose</code> indicates whether the function
// should output information what it is doing at any given moment:
// we use the ConditionalOStream class to do that for us.
//
template<int dim>
void EulerSolver<dim>::run(const bool verbose, const unsigned int output_interval) {
  ConditionalOStream verbose_cout(std::cout, verbose && Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);

  output_results(0);
  double time = t_0;
  unsigned int n = 0;
  while(std::abs(T - time) > 1e-10) {
    time += dt;
    n++;
    pcout << "Step = " << n << " Time = " << time << std::endl;

    /*--- First stage of the SSP scheme ---*/
    SSP_stage = 1;

    verbose_cout << "  Update density stage 1" << std::endl;
    update_density();
    pcout << "Minimal density " << get_minimal_density() << std::endl;
    pcout << "Maximal density " << get_maximal_density() << std::endl;

    verbose_cout << "  Update momentum stage 1" << std::endl;
    update_momentum();

    verbose_cout << "  Update energy stage 1" << std::endl;
    update_energy();

    /*--- Second stage of the SSP scheme ---*/
    SSP_stage = 2;

    verbose_cout << "  Update density stage 2" << std::endl;
    update_density();
    pcout << "Minimal density " << get_minimal_density() << std::endl;
    pcout << "Maximal density " << get_maximal_density() << std::endl;

    verbose_cout << "  Update momentum stage 2" << std::endl;
    update_momentum();

    verbose_cout << "  Update energy stage 2" << std::endl;
    update_energy();

    /*--- Second stage of the SSP scheme ---*/
    SSP_stage = 3;

    verbose_cout << "  Update density stage 3" << std::endl;
    update_density();
    pcout << "Minimal density " << get_minimal_density() << std::endl;
    pcout << "Maximal density " << get_maximal_density() << std::endl;

    verbose_cout << "  Update momentum stage 3" << std::endl;
    update_momentum();

    verbose_cout << "  Update energy stage 3" << std::endl;
    update_energy();

    /*--- Update density and velocity before applying the damping layers ---*/
    rho_old.equ(1.0, rho_curr);
    rhou_old.equ(1.0, rhou_curr);
    rhoE_old.equ(1.0, rhoE_curr);

    /*--- Compute Courant numbers ---*/
    const double max_vel = get_maximal_velocity();
    pcout<< "Maximal velocity = " << max_vel << std::endl;
    pcout << "CFL_u = " << dt*max_vel*EquationData::degree_u*
                           std::sqrt(dim)/GridTools::minimal_cell_diameter(triangulation) << std::endl;
    const double max_celerity = compute_max_celerity();
    pcout<< "Maximal celerity = " << 1.0/Ma*max_celerity << std::endl;
    pcout << "CFL_c = " << 1.0/Ma*dt*max_celerity*EquationData::degree_u*
                           std::sqrt(dim)/GridTools::minimal_cell_diameter(triangulation) << std::endl;

    /*--- Save the results each 'output_interval' steps ---*/
    if(n % output_interval == 0) {
      verbose_cout << "Plotting Solution final" << std::endl;
      output_results(n);
    }
    if(T - time < dt && T - time > 1e-10) {
      /*--- Recompute and rest the time if needed towards the end of the simulation to stop at the proper final time ---*/
      dt = T - time;
      euler_matrix.set_dt(dt);
    }
  }
  /*--- Save the final results if not previously done ---*/
  if(n % output_interval != 0) {
    verbose_cout << "Plotting Solution final" << std::endl;
    output_results(n);
  }
}


// @sect{ The main function }

// The main function is quite standard. We just need to declare the EulerSolver
// instance and let the simulation run.
//
int main(int argc, char *argv[]) {
  try {
    using namespace Atmospheric_Flow;

    RunTimeParameters::Data_Storage data;
    data.read_data("parameter-file.prm");

    Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, -1);

    const auto& curr_rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
    deallog.depth_console(data.verbose && curr_rank == 0 ? 2 : 0);

    EulerSolver<3> test(data);
    test.run(data.verbose, data.output_interval);

    if(curr_rank == 0)
      std::cout << "----------------------------------------------------"
                << std::endl
                << "Apparently everything went fine!" << std::endl
                << "Don't forget to brush your teeth :-)" << std::endl
                << std::endl;

    return 0;
  }
  catch(std::exception& exc) {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Exception on processing: " << std::endl
              << exc.what() << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;
    return 1;
  }
  catch(...) {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Unknown exception!" << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;
    return 1;
  }

}
