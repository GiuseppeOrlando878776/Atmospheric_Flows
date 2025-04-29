/* Author: Giuseppe Orlando, 2024. */

// @sect{Include files}

// We start by including all the necessary deal.II header files
//
#include <deal.II/base/parallel.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/affine_constraints.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_tools.h>
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

#include <deal.II/distributed/solution_transfer.h>

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
  const double t0;         /*--- Initial time auxiliary variable ----*/
  const double T;          /*--- Final time auxiliary variable ----*/

  unsigned int IMEX_stage; /*--- Flag to check at which current stage of the IMEX we are ---*/
  double       dt;         /*--- Time step auxiliary variable ---*/

  const double Ma;         /*--- Mach number auxiliary variable ----*/
  const double Fr;         /*--- Froude number auxiliary variable ---*/

  parallel::distributed::Triangulation<dim> triangulation; /*--- The variable which stores the mesh ---*/

  /*--- Finite element spaces for all the variables ---*/
  FESystem<dim> fe_density;
  FESystem<dim> fe_velocity;
  FESystem<dim> fe_pressure;

  /*--- Degrees of freedom handlers for all the variables ---*/
  DoFHandler<dim> dof_handler_density;
  DoFHandler<dim> dof_handler_velocity;
  DoFHandler<dim> dof_handler_pressure;

  /*--- Auxiliary mapping for curved boundary ---*/
  MappingQ<dim> mapping;

  /*--- Auxiliary quadratures for all the variables ---*/
  QGaussLobatto<dim> quadrature_density;
  QGaussLobatto<dim> quadrature_velocity;
  QGaussLobatto<dim> quadrature_pressure;

  /*--- Variables for the density ---*/
  LinearAlgebra::distributed::Vector<double> rho_old;
  LinearAlgebra::distributed::Vector<double> rho_s_2;
  LinearAlgebra::distributed::Vector<double> rho_s_3;
  LinearAlgebra::distributed::Vector<double> rho_curr;
  LinearAlgebra::distributed::Vector<double> rhs_rho;

  LinearAlgebra::distributed::Vector<double> rho_prime_old;
  LinearAlgebra::distributed::Vector<double> rho_prime_s_2;
  LinearAlgebra::distributed::Vector<double> rho_prime_s_3;

  /*--- Variables for the velocity ---*/
  LinearAlgebra::distributed::Vector<double> u_old;
  LinearAlgebra::distributed::Vector<double> u_s_2;
  LinearAlgebra::distributed::Vector<double> u_s_3;
  LinearAlgebra::distributed::Vector<double> u_curr;
  LinearAlgebra::distributed::Vector<double> u_fixed;
  LinearAlgebra::distributed::Vector<double> rhs_u;

  /*--- Variables for the pressure ---*/
  LinearAlgebra::distributed::Vector<double> pres_old;
  LinearAlgebra::distributed::Vector<double> pres_s_2;
  LinearAlgebra::distributed::Vector<double> pres_s_3;
  LinearAlgebra::distributed::Vector<double> pres_fixed;
  LinearAlgebra::distributed::Vector<double> dpres_fixed;
  LinearAlgebra::distributed::Vector<double> rhs_pres;

  LinearAlgebra::distributed::Vector<double> pres_prime_fixed;
  LinearAlgebra::distributed::Vector<double> pres_prime_old;
  LinearAlgebra::distributed::Vector<double> pres_prime_s_2;
  LinearAlgebra::distributed::Vector<double> pres_prime_s_3;

  /*--- Variables for the potential temperature ---*/
  LinearAlgebra::distributed::Vector<double> theta_old;

  LinearAlgebra::distributed::Vector<double> rhs_u_precomputed,
                                             rhs_pres_precomputed,
                                             extra_rhs_u; /*--- Auxiliary vectors for the Schur complement ---*/

  /*--- Background fields ---*/
  LinearAlgebra::distributed::Vector<double> rho_bar;
  LinearAlgebra::distributed::Vector<double> u_bar;
  LinearAlgebra::distributed::Vector<double> pres_bar;

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

  void pressure_fixed_point(); /*--- Function to compute the pressure in the fixed point loop ---*/

  void update_velocity(); /*--- Function to compute the velocity in the fixed point loop ---*/

  void update_pressure(); /*--- Function to compute the pressure for the weighting step of the IMEX ---*/

  void output_results(const unsigned int step); /*--- Function to save the results ---*/

private:
  void precompute_rhs_pressure(); /*--- Auxiliary function to compute the rhs of the pressure equation ---*/

  unsigned int perform_fixed_point_loop(); /*--- Auxiliary function for the fixed point loop ---*/

  /*--- Damping layers functions for all the unknowns ---*/
  LinearAlgebra::distributed::Vector<double> dt_tau_rho;
  LinearAlgebra::distributed::Vector<double> dt_tau_u;
  LinearAlgebra::distributed::Vector<double> dt_tau_pres;
  LinearAlgebra::distributed::Vector<double> dt_tau_rho_aux;
  LinearAlgebra::distributed::Vector<double> dt_tau_u_aux;
  LinearAlgebra::distributed::Vector<double> dt_tau_pres_aux;

  LinearAlgebra::distributed::Vector<double> dt_tau_rho_right;
  LinearAlgebra::distributed::Vector<double> dt_tau_u_right;
  LinearAlgebra::distributed::Vector<double> dt_tau_pres_right;
  LinearAlgebra::distributed::Vector<double> dt_tau_rho_aux_right;
  LinearAlgebra::distributed::Vector<double> dt_tau_u_aux_right;
  LinearAlgebra::distributed::Vector<double> dt_tau_pres_aux_right;

  LinearAlgebra::distributed::Vector<double> dt_tau_rho_left;
  LinearAlgebra::distributed::Vector<double> dt_tau_u_left;
  LinearAlgebra::distributed::Vector<double> dt_tau_pres_left;
  LinearAlgebra::distributed::Vector<double> dt_tau_rho_aux_left;
  LinearAlgebra::distributed::Vector<double> dt_tau_u_aux_left;
  LinearAlgebra::distributed::Vector<double> dt_tau_pres_aux_left;

  /*--- Manifold and fields handling data structures ---*/
  EquationData::TREX_Manifold<dim> manifold;

  EquationData::Density<dim>  rho_init;
  EquationData::Velocity<dim> u_init;
  EquationData::Pressure<dim> pres_init;

  EquationData::Rayleigh<dim, 1>       dt_tau;
  EquationData::Rayleigh_Aux<dim, 1>   dt_tau_aux;
  EquationData::Rayleigh<dim, dim>     dt_tau_vel;
  EquationData::Rayleigh_Aux<dim, dim> dt_tau_vel_aux;

  EquationData::Rayleigh_Right<dim, 1>       dt_tau_right;
  EquationData::Rayleigh_Aux_Right<dim, 1>   dt_tau_aux_right;
  EquationData::Rayleigh_Right<dim, dim>     dt_tau_vel_right;
  EquationData::Rayleigh_Aux_Right<dim, dim> dt_tau_vel_aux_right;

  EquationData::Rayleigh_Left<dim, 1>       dt_tau_left;
  EquationData::Rayleigh_Aux_Left<dim, 1>   dt_tau_aux_left;
  EquationData::Rayleigh_Left<dim, dim>     dt_tau_vel_left;
  EquationData::Rayleigh_Aux_Left<dim, dim> dt_tau_vel_aux_left;

  /*--- Auxiliary structures for the matrix-free and for the multigrid ---*/
  std::shared_ptr<MatrixFree<dim, double>> matrix_free_storage;

  using MatrixType = EULEROperator<dim,
                                   EquationData::degree_u,
                                   EquationData::degree_rho,
                                   EquationData::degree_p,
                                   2*EquationData::degree_u + 1,
                                   2*EquationData::degree_u + 1 + EquationData::extra_quadrature_degree,
                                   LinearAlgebra::distributed::Vector<double>>;
  MatrixType euler_matrix;

  std::vector<const DoFHandler<dim>*> dof_handlers; /*--- Auxiliary container for the matrix-free ---*/

  std::vector<const AffineConstraints<double>*> constraints; /*--- Auxiliary container for the matrix-free ---*/
  AffineConstraints<double> constraints_velocity,
                            constraints_pressure,
                            constraints_density;

  std::vector<QGauss<1>> quadratures; /*--- Auxiliary container for the quadrature in matrix-free ---*/

  unsigned int max_its;         /*--- Auxiliary variable for the maximum number of iterations of linear solvers ---*/
  double       eps;             /*--- Auxiliary variable for the tolerance of linear solvers ---*/
  double       eps_fixed_point; /*--- Auxiliary variable for the tolerance of fixed point loop ---*/

  unsigned int n_refines; /*-- Number of initial global refinements ---*/

  std::string saving_dir; /*--- Auxiliary variable for the directory to save the results ---*/

  bool         restart,
               save_for_restart;
  unsigned int step_restart;
  double       time_restart;
  bool         as_initial_conditions;

  /*--- Now we declare a bunch of variables for text output ---*/
  ConditionalOStream pcout;

  std::ofstream      time_out;
  ConditionalOStream ptime_out;
  TimerOutput        time_table;

  Vector<double> Linfty_error_per_cell_pres; /*--- Auxiliary variable for the end of the fixed point loop ---*/

  double get_max_velocity(); /*--- Get maximum velocity to compute the Courant number ---*/

  double get_min_density(); /*--- Get minimum density ---*/

  double get_max_density(); /*--- Get maximum density ---*/

  double compute_max_celerity(); /*--- Compute maximum celerity for acoustic Courant number ---*/

  std::pair<double, double> compute_max_Cu_x_z(); /*--- Get maximum Courant numbers along both horizontal and vertical directions ---*/

  std::pair<double, double> compute_max_C_x_z(); /*--- Get maximum acoustic Courant numbers along both horizontal and vertical directions ---*/

  void postprocess_integral_quantities(const double time); /*--- Auxiliary function to compute integral quantities ---*/

  /*--- Initial integral quantities to compute relative variations ---*/
  double rho0_integral,
         rhoE0_integral,
         s0_integral,
         eta0_integral;

  /*--- Stream to save integral quantities ---*/
  std::ofstream output_integral_rho,
                output_integral_rhoE,
                output_integral_s,
                output_integral_eta;
};


// @sect{ <code>EulerSolver::EulerSolver</code> }

// In the constructor, we just read all the data from the
// <code>Data_Storage</code> object that is passed as an argument, verify that
// the data we read are reasonable and, finally, create the triangulation and
// load the initial data.
//
template<int dim>
EulerSolver<dim>::EulerSolver(RunTimeParameters::Data_Storage& data):
  t0(data.initial_time),
  T(data.final_time),
  IMEX_stage(2),            /*--- Initialize the flag for the IMEX scheme stage ---*/
  dt(data.dt),
  Ma(data.Mach),
  Fr(data.Froude),
  triangulation(MPI_COMM_WORLD,
                parallel::distributed::Triangulation<dim>::limit_level_difference_at_vertices,
                parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy),
  fe_density(FE_DGQ<dim>(EquationData::degree_rho), 1),
  fe_velocity(FE_DGQ<dim>(EquationData::degree_u), dim),
  fe_pressure(FE_DGQ<dim>(EquationData::degree_p), 1),
  dof_handler_density(triangulation),
  dof_handler_velocity(triangulation),
  dof_handler_pressure(triangulation),
  mapping(EquationData::degree_mapping, true),
  quadrature_density(EquationData::degree_rho + 1),
  quadrature_velocity(EquationData::degree_u + 1),
  quadrature_pressure(EquationData::degree_p + 1),
  manifold("./" + data.dir + "/TREX_profile.dat"),
  rho_init("./" + data.dir + "/theta_profile.dat", data.initial_time),
  u_init("./" + data.dir + "/ux_profile.dat", data.initial_time),
  pres_init("./" + data.dir + "/theta_profile.dat", data.initial_time),
  dt_tau(data.initial_time),
  dt_tau_aux(data.initial_time),
  dt_tau_vel(data.initial_time),
  dt_tau_vel_aux(data.initial_time),
  dt_tau_right(data.initial_time),
  dt_tau_aux_right(data.initial_time),
  dt_tau_vel_right(data.initial_time),
  dt_tau_vel_aux_right(data.initial_time),
  dt_tau_left(data.initial_time),
  dt_tau_aux_left(data.initial_time),
  dt_tau_vel_left(data.initial_time),
  dt_tau_vel_aux_left(data.initial_time),
  euler_matrix(data),
  dof_handlers(EquationData::n_vars),
  constraints(EquationData::n_vars),
  max_its(data.max_iterations),
  eps(data.eps),
  eps_fixed_point(data.eps_fixed_point),
  n_refines(data.n_global_refines),
  saving_dir(data.dir),
  restart(data.restart),
  save_for_restart(data.save_for_restart),
  step_restart(data.step_restart),
  time_restart(data.time_restart),
  as_initial_conditions(data.as_initial_conditions),
  pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0),
  time_out("./" + data.dir + "/time_analysis_" +
           Utilities::int_to_string(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)) + "proc.dat"),
  ptime_out(time_out, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0),
  time_table(ptime_out, TimerOutput::summary, TimerOutput::cpu_and_wall_times),
  output_integral_rho("./" + data.dir + "/analysis_rho.dat", std::ofstream::out),
  output_integral_rhoE("./" + data.dir + "/analysis_rhoE.dat", std::ofstream::out),
  output_integral_s("./" + data.dir + "/analysis_s.dat", std::ofstream::out),
  output_integral_eta("./" + data.dir + "/analysis_eta.dat", std::ofstream::out) {
    AssertThrow(!((dt <= 0.0) || (dt > 0.5*T)), ExcInvalidTimeStep(dt, 0.5*T));

    matrix_free_storage = std::make_shared<MatrixFree<dim, double>>();

    /*--- Clear the containers for safety ---*/
    constraints_velocity.clear();
    constraints_velocity.close();
    constraints_pressure.clear();
    constraints_pressure.close();
    constraints_density.clear();
    constraints_density.close();

    quadratures.clear();

    /*--- Call initializing routines ---*/
    create_triangulation(n_refines);
    setup_dofs();
    initialize();
  }


// @sect{<code>EulerSolver::create_triangulation</code>}

// The method that creates the triangulation.
//
template<int dim>
void EulerSolver<dim>::create_triangulation(const unsigned int n_refines) {
  TimerOutput::Scope t(time_table, "Create triangulation");

  Point<dim> lower_left;
  lower_left[0] = 0.0;
  lower_left[1] = 0.0;
  Point<dim> upper_right;
  upper_right[0] = EquationData::x_max/EquationData::L_ref;
  upper_right[1] = EquationData::z_max/EquationData::L_ref;

  GridGenerator::subdivided_hyper_rectangle(triangulation, {100, 15}, lower_left, upper_right, true);

  /*--- Consider periodic conditions along the horizontal direction ---*/
  std::vector<GridTools::PeriodicFacePair<typename parallel::distributed::Triangulation<dim>::cell_iterator>> periodic_faces;
  GridTools::collect_periodic_faces(triangulation, 0, 1, 0, periodic_faces);
  triangulation.add_periodicity(periodic_faces);

  /*--- Build the proper triangulation ---*/
  if(restart) {
    triangulation.load("./" + saving_dir + "/solution_ser-" + Utilities::int_to_string(step_restart, 5));
  }
  else {
    triangulation.refine_global(n_refines);
  }

  /*--- Apply the mapping to build the physical domain ---*/
  GridTools::transform([this](const Point<dim>& chart_point) {
                                return manifold.push_forward(chart_point);
                              },
                              triangulation);

  triangulation.set_all_manifold_ids_on_boundary(2, 111);
  triangulation.set_manifold(111, manifold);
}

// After creating the triangulation, it creates the mesh dependent
// data, i.e. it distributes degrees of freedom and renumbers them, and
// initializes the matrices and vectors that we will use.
//
template<int dim>
void EulerSolver<dim>::setup_dofs() {
  TimerOutput::Scope t(time_table, "Setup dofs");

  pcout << "Number of active cells: " << triangulation.n_global_active_cells() << std::endl;
  pcout << "Number of levels: "       << triangulation.n_global_levels()       << std::endl;

  /*--- Set degrees of freedom ---*/
  dof_handler_velocity.distribute_dofs(fe_velocity);
  dof_handler_pressure.distribute_dofs(fe_pressure);
  dof_handler_density.distribute_dofs(fe_density);

  pcout << "dim (V_h) = " << dof_handler_velocity.n_dofs()
        << std::endl
        << "dim (Q_h) = " << dof_handler_pressure.n_dofs()
        << std::endl
        << "dim (X_h) = " << dof_handler_density.n_dofs()
        << std::endl
        << "Ma        = " << Ma
        << std::endl
        << "Fr        = " << Fr
        << std::endl
        << std::endl;

  /*--- Set additional data to check which variables neeed to be updated ---*/
  typename MatrixFree<dim, double>::AdditionalData additional_data;
  additional_data.mapping_update_flags                = (update_values | update_gradients |
                                                         update_quadrature_points | update_JxW_values);
  additional_data.mapping_update_flags_inner_faces    = (update_values | update_normal_vectors |
                                                         update_quadrature_points | update_JxW_values);
  additional_data.mapping_update_flags_boundary_faces = (update_values | update_normal_vectors |
                                                         update_quadrature_points | update_JxW_values);
  additional_data.tasks_parallel_scheme               = MatrixFree<dim, double>::AdditionalData::none;

  /*--- Set the container with the dof handlers ---*/
  dof_handlers[EquationData::U_INDEX_DOF]   = &dof_handler_velocity;
  dof_handlers[EquationData::P_INDEX_DOF]   = &dof_handler_pressure;
  dof_handlers[EquationData::RHO_INDEX_DOF] = &dof_handler_density;

  /*--- Set the container with the constraints. Each entry is empty (no Dirichlet and weak imposition in general)
        and this is necessary only for compatibilty reasons ---*/
  constraints[EquationData::U_INDEX_DOF]   = &constraints_velocity;
  constraints[EquationData::P_INDEX_DOF]   = &constraints_pressure;
  constraints[EquationData::RHO_INDEX_DOF] = &constraints_density;

  /*--- Set the quadrature formula to compute the integrals for assembling bilinear and linear forms ---*/
  quadratures.push_back(QGauss<1>(2*EquationData::degree_u + 1));
  quadratures.push_back(QGauss<1>(2*EquationData::degree_u + 1 + EquationData::extra_quadrature_degree));
  quadratures.push_back(QGauss<1>(EquationData::degree_u + 1));

  /*--- Initialize the matrix-free structure with DofHandlers, Constraints, Quadratures and AdditionalData ---*/
  matrix_free_storage->reinit(mapping, dof_handlers, constraints, quadratures, additional_data);

  /*--- Initialize the variables related to the velocity ---*/
  matrix_free_storage->initialize_dof_vector(u_old, EquationData::U_INDEX_DOF);
  matrix_free_storage->initialize_dof_vector(u_s_2, EquationData::U_INDEX_DOF);
  matrix_free_storage->initialize_dof_vector(u_s_3, EquationData::U_INDEX_DOF);
  matrix_free_storage->initialize_dof_vector(u_curr, EquationData::U_INDEX_DOF);
  matrix_free_storage->initialize_dof_vector(u_fixed, EquationData::U_INDEX_DOF);
  matrix_free_storage->initialize_dof_vector(rhs_u, EquationData::U_INDEX_DOF);

  /*--- Initialize the variables related to the pressure ---*/
  matrix_free_storage->initialize_dof_vector(pres_old, EquationData::P_INDEX_DOF);
  matrix_free_storage->initialize_dof_vector(pres_s_2, EquationData::P_INDEX_DOF);
  matrix_free_storage->initialize_dof_vector(pres_s_3, EquationData::P_INDEX_DOF);
  matrix_free_storage->initialize_dof_vector(pres_fixed, EquationData::P_INDEX_DOF);
  matrix_free_storage->initialize_dof_vector(dpres_fixed, EquationData::P_INDEX_DOF);
  matrix_free_storage->initialize_dof_vector(rhs_pres, EquationData::P_INDEX_DOF);

  matrix_free_storage->initialize_dof_vector(pres_prime_fixed, EquationData::P_INDEX_DOF);
  matrix_free_storage->initialize_dof_vector(pres_prime_old, EquationData::P_INDEX_DOF);
  matrix_free_storage->initialize_dof_vector(pres_prime_s_2, EquationData::P_INDEX_DOF);
  matrix_free_storage->initialize_dof_vector(pres_prime_s_3, EquationData::P_INDEX_DOF);

  /*--- Initialize the variables related to the density ---*/
  matrix_free_storage->initialize_dof_vector(rho_old, EquationData::RHO_INDEX_DOF);
  matrix_free_storage->initialize_dof_vector(rho_s_2, EquationData::RHO_INDEX_DOF);
  matrix_free_storage->initialize_dof_vector(rho_s_3, EquationData::RHO_INDEX_DOF);
  matrix_free_storage->initialize_dof_vector(rho_curr, EquationData::RHO_INDEX_DOF);
  matrix_free_storage->initialize_dof_vector(rhs_rho, EquationData::RHO_INDEX_DOF);

  matrix_free_storage->initialize_dof_vector(rho_prime_old, EquationData::RHO_INDEX_DOF);
  matrix_free_storage->initialize_dof_vector(rho_prime_s_2, EquationData::RHO_INDEX_DOF);
  matrix_free_storage->initialize_dof_vector(rho_prime_s_3, EquationData::RHO_INDEX_DOF);

  /*--- Initialize the auxiliary variable for the Schur complement ---*/
  matrix_free_storage->initialize_dof_vector(rhs_u_precomputed, EquationData::U_INDEX_DOF);
  rhs_u_precomputed = 0;
  matrix_free_storage->initialize_dof_vector(rhs_pres_precomputed, EquationData::P_INDEX_DOF);
  rhs_pres_precomputed = 0;
  matrix_free_storage->initialize_dof_vector(extra_rhs_u, EquationData::U_INDEX_DOF);
  extra_rhs_u = 0;

  /*--- Initialize the variables related to the potential temperature ---*/
  matrix_free_storage->initialize_dof_vector(theta_old, EquationData::P_INDEX_DOF);

  /*--- Initialize the variables related to the damping layers ---*/
  matrix_free_storage->initialize_dof_vector(dt_tau_u, EquationData::U_INDEX_DOF);
  matrix_free_storage->initialize_dof_vector(dt_tau_pres, EquationData::P_INDEX_DOF);
  matrix_free_storage->initialize_dof_vector(dt_tau_rho, EquationData::RHO_INDEX_DOF);
  matrix_free_storage->initialize_dof_vector(dt_tau_u_aux, EquationData::U_INDEX_DOF);
  matrix_free_storage->initialize_dof_vector(dt_tau_pres_aux, EquationData::P_INDEX_DOF);
  matrix_free_storage->initialize_dof_vector(dt_tau_rho_aux, EquationData::RHO_INDEX_DOF);
  VectorTools::interpolate(mapping, dof_handler_velocity, dt_tau_vel, dt_tau_u);
  VectorTools::interpolate(mapping, dof_handler_pressure, dt_tau, dt_tau_pres);
  VectorTools::interpolate(mapping, dof_handler_density, dt_tau, dt_tau_rho);
  VectorTools::interpolate(mapping, dof_handler_velocity, dt_tau_vel_aux, dt_tau_u_aux);
  VectorTools::interpolate(mapping, dof_handler_pressure, dt_tau_aux, dt_tau_pres_aux);
  VectorTools::interpolate(mapping, dof_handler_density, dt_tau_aux, dt_tau_rho_aux);

  matrix_free_storage->initialize_dof_vector(dt_tau_u_right, EquationData::U_INDEX_DOF);
  matrix_free_storage->initialize_dof_vector(dt_tau_pres_right, EquationData::P_INDEX_DOF);
  matrix_free_storage->initialize_dof_vector(dt_tau_rho_right, EquationData::RHO_INDEX_DOF);
  matrix_free_storage->initialize_dof_vector(dt_tau_u_aux_right, EquationData::U_INDEX_DOF);
  matrix_free_storage->initialize_dof_vector(dt_tau_pres_aux_right, EquationData::P_INDEX_DOF);
  matrix_free_storage->initialize_dof_vector(dt_tau_rho_aux_right, EquationData::RHO_INDEX_DOF);
  VectorTools::interpolate(mapping, dof_handler_velocity, dt_tau_vel_right, dt_tau_u_right);
  VectorTools::interpolate(mapping, dof_handler_pressure, dt_tau_right, dt_tau_pres_right);
  VectorTools::interpolate(mapping, dof_handler_density, dt_tau_right, dt_tau_rho_right);
  VectorTools::interpolate(mapping, dof_handler_velocity, dt_tau_vel_aux_right, dt_tau_u_aux_right);
  VectorTools::interpolate(mapping, dof_handler_pressure, dt_tau_aux_right, dt_tau_pres_aux_right);
  VectorTools::interpolate(mapping, dof_handler_density, dt_tau_aux_right, dt_tau_rho_aux_right);

  matrix_free_storage->initialize_dof_vector(dt_tau_u_left, EquationData::U_INDEX_DOF);
  matrix_free_storage->initialize_dof_vector(dt_tau_pres_left, EquationData::P_INDEX_DOF);
  matrix_free_storage->initialize_dof_vector(dt_tau_rho_left, EquationData::RHO_INDEX_DOF);
  matrix_free_storage->initialize_dof_vector(dt_tau_u_aux_left, EquationData::U_INDEX_DOF);
  matrix_free_storage->initialize_dof_vector(dt_tau_pres_aux_left, EquationData::P_INDEX_DOF);
  matrix_free_storage->initialize_dof_vector(dt_tau_rho_aux_left, EquationData::RHO_INDEX_DOF);
  VectorTools::interpolate(mapping, dof_handler_velocity, dt_tau_vel_left, dt_tau_u_left);
  VectorTools::interpolate(mapping, dof_handler_pressure, dt_tau_left, dt_tau_pres_left);
  VectorTools::interpolate(mapping, dof_handler_density, dt_tau_left, dt_tau_rho_left);
  VectorTools::interpolate(mapping, dof_handler_velocity, dt_tau_vel_aux_left, dt_tau_u_aux_left);
  VectorTools::interpolate(mapping, dof_handler_pressure, dt_tau_aux_left, dt_tau_pres_aux_left);
  VectorTools::interpolate(mapping, dof_handler_density, dt_tau_aux_left, dt_tau_rho_aux_left);

  matrix_free_storage->initialize_dof_vector(u_bar, EquationData::U_INDEX_DOF);
  matrix_free_storage->initialize_dof_vector(pres_bar, EquationData::P_INDEX_DOF);
  matrix_free_storage->initialize_dof_vector(rho_bar, EquationData::RHO_INDEX_DOF);
  VectorTools::interpolate(mapping, dof_handler_velocity, u_init, u_bar);
  VectorTools::interpolate(mapping, dof_handler_pressure, pres_init, pres_bar);
  VectorTools::interpolate(mapping, dof_handler_density, rho_init, rho_bar);

  dt_tau_u.scale(u_bar);
  dt_tau_pres.scale(pres_bar);
  dt_tau_rho.scale(rho_bar);

  dt_tau_u_right.scale(u_bar);
  dt_tau_pres_right.scale(pres_bar);
  dt_tau_rho_right.scale(rho_bar);

  dt_tau_u_left.scale(u_bar);
  dt_tau_pres_left.scale(pres_bar);
  dt_tau_rho_left.scale(rho_bar);

  /*--- Initialize the auxiliary variable to check the error and stop the fixed point loop ---*/
  Vector<double> error_per_cell_tmp(triangulation.n_active_cells());
  Linfty_error_per_cell_pres.reinit(error_per_cell_tmp);
}

// @sect{ <code>EulerSolver::initialize</code> }

// This method loads the initial data
//
template<int dim>
void EulerSolver<dim>::initialize() {
  TimerOutput::Scope t(time_table, "Initialize state");

  /*--- Initialize the fields in case of restart ---*/
  if(restart) {
    parallel::distributed::SolutionTransfer<dim, LinearAlgebra::distributed::Vector<double>>
    solution_transfer_density(dof_handler_density);
    parallel::distributed::SolutionTransfer<dim, LinearAlgebra::distributed::Vector<double>>
    solution_transfer_velocity(dof_handler_velocity);
    parallel::distributed::SolutionTransfer<dim, LinearAlgebra::distributed::Vector<double>>
    solution_transfer_pressure(dof_handler_pressure);

    rho_old.zero_out_ghost_values();
    u_old.zero_out_ghost_values();
    pres_old.zero_out_ghost_values();

    solution_transfer_density.deserialize(rho_old);
    solution_transfer_velocity.deserialize(u_old);
    solution_transfer_pressure.deserialize(pres_old);
  }
  /*--- Initialize the fields ---*/
  else {
    VectorTools::interpolate(mapping, dof_handler_density, rho_init, rho_old);
    VectorTools::interpolate(mapping, dof_handler_velocity, u_init, u_old);
    VectorTools::interpolate(mapping, dof_handler_pressure, pres_init, pres_old);
  }

  /*--- Initialize density and pressure perturbation ---*/
  rho_prime_old.equ(1.0, rho_old);
  rho_prime_old.add(-1.0, rho_bar);
  pres_prime_old.equ(1.0, pres_old);
  pres_prime_old.add(-1.0, pres_bar);
}


// @sect{<code>EulerSolver::update_density</code>}

// This implements the update of the density
//
template<int dim>
void EulerSolver<dim>::update_density() {
  TimerOutput::Scope t(time_table, "Update density");

  const std::vector<unsigned int> index_dof_handler = {EquationData::RHO_INDEX_DOF};
  euler_matrix.initialize(matrix_free_storage, index_dof_handler, index_dof_handler);
  euler_matrix.set_Euler_stage(EquationData::RHO_INDEX_SYSTEM);

  /*--- Compute the rhs ---*/
  if(IMEX_stage == 2) {
    euler_matrix.vmult_rhs_density(rhs_rho, {rho_old, u_old});
  }
  else if(IMEX_stage == 3) {
    euler_matrix.vmult_rhs_density(rhs_rho, {rho_old, u_old,
                                             rho_s_2, u_s_2});
  }
  else {
    euler_matrix.vmult_rhs_density(rhs_rho, {rho_old, u_old,
                                             rho_s_2, u_s_2,
                                             rho_s_3, u_s_3});
  }

  /*--- Solve the system for the density ---*/
  if(IMEX_stage == 2) {
    euler_matrix.vmult(rho_s_2, rhs_rho);
  }
  else if(IMEX_stage == 3) {
    euler_matrix.vmult(rho_s_3, rhs_rho);
  }
  else {
    euler_matrix.vmult(rho_curr, rhs_rho);
  }
}

// @sect{<code>EulerSolver::pressure_fixed_point</code>}

// This implements a step of the fixed point procedure for the computation of the pressure in the hyperbolic part
//
template<int dim>
void EulerSolver<dim>::pressure_fixed_point() {
  TimerOutput::Scope t(time_table, "Fixed point pressure");

  const std::vector<unsigned int> index_dof_handler = {EquationData::P_INDEX_DOF};
  euler_matrix.initialize(matrix_free_storage, index_dof_handler, index_dof_handler);
  euler_matrix.set_Euler_stage(EquationData::P_INDEX_SYSTEM);

  /*--- Compute the rhs ---*/
  if(IMEX_stage == 2) {
    euler_matrix.vmult_rhs_energy(rhs_pres, {rho_old, u_old, pres_old,
                                             rho_s_2, u_fixed, pres_prime_fixed,
                                             pres_prime_old});

    euler_matrix.vmult_rhs_momentum(rhs_u, {rho_old, u_old, pres_prime_old,
                                            rho_prime_old, rho_prime_s_2}); /*--- This has to be recomputed to avoid issues ---*/
  }
  else if(IMEX_stage == 3) {
    euler_matrix.vmult_rhs_energy(rhs_pres, {rho_old, u_old, pres_old,
                                             rho_s_2, u_s_2, pres_s_2,
                                             rho_s_3, u_fixed, pres_prime_fixed,
                                             pres_prime_old, pres_prime_s_2});

    euler_matrix.vmult_rhs_momentum(rhs_u, {rho_old, u_old, pres_prime_old,
                                            rho_s_2, u_s_2, pres_prime_s_2,
                                            rho_prime_old, rho_prime_s_2, rho_prime_s_3}); /*--- This has to be recomputed to avoid issues ---*/
  }

  // Perform matrix-vector multiplication with enthalpy matrix (which changes over time)
  euler_matrix.set_pres_fixed(pres_fixed); // Set the current pressure for the fixed point loop to the operator
  euler_matrix.vmult_enthalpy(rhs_pres_precomputed, rhs_u_precomputed);

  // Conclude computation of rhs for pressure fixed point
  rhs_pres.add(-1.0, rhs_pres_precomputed);

  /*--- Solve the system for the pressure ---*/
  SolverControl solver_control(max_its, eps*rhs_pres.l2_norm());
  SolverGMRES<LinearAlgebra::distributed::Vector<double>> gmres(solver_control);

  /*--- Jacobi preconditioner for this system ---*/
  PreconditionJacobi<MatrixType> preconditioner_Jacobi;
  euler_matrix.compute_diagonal();
  preconditioner_Jacobi.initialize(euler_matrix);

  /*--- Solve the linear system ---*/
  gmres.solve(euler_matrix, pres_prime_fixed, rhs_pres, preconditioner_Jacobi);
  pres_fixed.equ(1.0, pres_prime_fixed);
  pres_fixed.add(1.0, pres_bar);
}

// Auxiliary routine to compute the rhs of the pressure equation
// (contribution that does not change during the fixed point loop)
//
template<int dim>
void EulerSolver<dim>::precompute_rhs_pressure() {
  if(IMEX_stage == 2) {
    euler_matrix.vmult_rhs_momentum(rhs_u, {rho_old, u_old, pres_prime_old,
                                            rho_prime_old, rho_prime_s_2});
  }
  else if(IMEX_stage == 3) {
    euler_matrix.vmult_rhs_momentum(rhs_u, {rho_old, u_old, pres_prime_old,
                                            rho_s_2, u_s_2, pres_prime_s_2,
                                            rho_prime_old, rho_prime_s_2, rho_prime_s_3});
  }

  const std::vector<unsigned int> index_dof_handler = {EquationData::U_INDEX_DOF};
  euler_matrix.initialize(matrix_free_storage, index_dof_handler, index_dof_handler);
  euler_matrix.set_Euler_stage(EquationData::U_INDEX_SYSTEM);

  /*--- Solve to compute first contribution to rhs --*/
  euler_matrix.vmult(rhs_u_precomputed, rhs_u);
}

// Auxiliary routine for the fixed point loop
//
template<int dim>
unsigned int EulerSolver<dim>::perform_fixed_point_loop() {
  /*--- Compute the contribution to the rhs that never changes ---*/
  precompute_rhs_pressure();

  /*--- Perform the fixed point loop ---*/
  unsigned int iter;
  for(iter = 0; iter < 100; ++iter) {
    dpres_fixed.equ(1.0, pres_fixed),
    pressure_fixed_point();
    update_velocity();

    /*--- Compute the relative error for the pressure ---*/
    VectorTools::integrate_difference(dof_handler_pressure, pres_fixed, Functions::ZeroFunction<dim>(),
                                      Linfty_error_per_cell_pres, quadrature_pressure, VectorTools::Linfty_norm);
    const double den = VectorTools::compute_global_error(triangulation, Linfty_error_per_cell_pres, VectorTools::Linfty_norm);
    dpres_fixed.add(-1.0, pres_fixed);
    VectorTools::integrate_difference(dof_handler_pressure, dpres_fixed, Functions::ZeroFunction<dim>(),
                                      Linfty_error_per_cell_pres, quadrature_pressure, VectorTools::Linfty_norm);
    const double error = VectorTools::compute_global_error(triangulation, Linfty_error_per_cell_pres, VectorTools::Linfty_norm)/den;
    if(error < eps_fixed_point)
      break; /*--- The fixed point loop is stopped whenever the relative error in infinity norm is below the specified tolerance ---*/
  }

  return iter;
}

// @sect{<code>EulerSolver::update_velocity</code>}

// This implements the velocity update in the fixed point procedure
// and in the final update stage of IMEX scheme
//
template<int dim>
void EulerSolver<dim>::update_velocity() {
  TimerOutput::Scope t(time_table, "Update velocity");

  const std::vector<unsigned int> index_dof_handler = {EquationData::U_INDEX_DOF};
  euler_matrix.initialize(matrix_free_storage, index_dof_handler, index_dof_handler);
  euler_matrix.set_Euler_stage(EquationData::U_INDEX_SYSTEM);

  /*--- Compute the rhs (if needed) ---*/
  if(IMEX_stage <= EquationData::n_stages) {
    euler_matrix.vmult_pressure(extra_rhs_u, pres_prime_fixed);
    rhs_u.add(-1.0, extra_rhs_u);
  }
  else {
    euler_matrix.vmult_rhs_momentum(rhs_u, {rho_old, u_old, pres_prime_old,
                                            rho_s_2, u_s_2, pres_prime_s_2,
                                            rho_s_3, u_s_3, pres_prime_s_3,
                                            rho_prime_old, rho_prime_s_2, rho_prime_s_3});
  }

  /*--- Solve the system for the velocity ---*/
  if(IMEX_stage <= EquationData::n_stages) {
    euler_matrix.vmult(u_fixed, rhs_u);
  }
  else {
    euler_matrix.vmult(u_curr, rhs_u);
  }
}

// @sect{<code>EulerSolver::update_pressure</code>}

// This implements the update of the pressure for the hyperbolic part
//
template<int dim>
void EulerSolver<dim>::update_pressure() {
  TimerOutput::Scope t(time_table, "Update pressure");

  const std::vector<unsigned int> index_dof_handler = {EquationData::P_INDEX_DOF};
  euler_matrix.initialize(matrix_free_storage, index_dof_handler, index_dof_handler);
  euler_matrix.set_Euler_stage(EquationData::P_INDEX_SYSTEM);

  /*--- Compute the rhs ---*/
  euler_matrix.vmult_rhs_energy(rhs_pres, {rho_old, u_old, pres_old,
                                           rho_s_2, u_s_2, pres_s_2,
                                           rho_s_3, u_s_3, pres_s_3,
                                           rho_curr, u_curr,
                                           pres_prime_old, pres_prime_s_2, pres_prime_s_3});

  /*--- Solve the system for the pressure ---*/
  euler_matrix.vmult(pres_prime_old, rhs_pres);
  pres_old.equ(1.0, pres_prime_old);
  pres_old.add(1.0, pres_bar);
}


// @sect{ <code>EulerSolver::output_results</code> }

// This method plots the current solution.
//
template<int dim>
void EulerSolver<dim>::output_results(const unsigned int step) {
  TimerOutput::Scope t(time_table, "Output results");

  /*--- Save the fields ---*/
  DataOut<dim> data_out;

  rho_old.update_ghost_values();
  data_out.add_data_vector(dof_handler_density, rho_old, "rho", {DataComponentInterpretation::component_is_scalar});
  std::vector<std::string> velocity_names(dim, "u");
  std::vector<DataComponentInterpretation::DataComponentInterpretation>
  component_interpretation_velocity(dim, DataComponentInterpretation::component_is_part_of_vector);
  u_old.update_ghost_values();
  data_out.add_data_vector(dof_handler_velocity, u_old, velocity_names, component_interpretation_velocity);
  pres_old.update_ghost_values();
  data_out.add_data_vector(dof_handler_pressure, pres_old, "p", {DataComponentInterpretation::component_is_scalar});

  for(const auto& cell: dof_handler_pressure.active_cell_iterators()) {
    if(cell->is_locally_owned()) {
      std::vector<types::global_dof_index> dof_indices(fe_pressure.dofs_per_cell);
      cell->get_dof_indices(dof_indices);
      for(unsigned int idx = 0; idx < dof_indices.size(); ++idx) {
        const double T  = pres_old(dof_indices[idx])/rho_old(dof_indices[idx]);
        const double Pi = std::pow(pres_old(dof_indices[idx]), (EquationData::Cp_Cv - 1.0)/EquationData::Cp_Cv);
        theta_old(dof_indices[idx]) = T/Pi;
      }
    }
  }
  theta_old.update_ghost_values();
  data_out.add_data_vector(dof_handler_pressure, theta_old, "theta", {DataComponentInterpretation::component_is_scalar});

  /*--- Save perturbations ---*/
  rho_prime_old.update_ghost_values();
  data_out.add_data_vector(dof_handler_density, rho_prime_old, "rho_prime", {DataComponentInterpretation::component_is_scalar});
  pres_prime_old.update_ghost_values();
  data_out.add_data_vector(dof_handler_pressure, pres_prime_old, "p_prime", {DataComponentInterpretation::component_is_scalar});

  /*--- Save background state ---*/
  rho_bar.update_ghost_values();
  data_out.add_data_vector(dof_handler_density, rho_bar, "rho_bar", {DataComponentInterpretation::component_is_scalar});
  u_bar.update_ghost_values();
  std::fill(velocity_names.begin(), velocity_names.end(), "u_bar");
  data_out.add_data_vector(dof_handler_velocity, u_bar, velocity_names, component_interpretation_velocity);
  pres_bar.update_ghost_values();
  data_out.add_data_vector(dof_handler_pressure, pres_bar, "p_bar", {DataComponentInterpretation::component_is_scalar});

  /*--- Build patches ---*/
  data_out.build_patches(mapping, EquationData::degree_u, DataOut<dim>::curved_inner_cells);

  DataOutBase::DataOutFilterFlags flags(false, true);
  DataOutBase::DataOutFilter      data_filter(flags);
  data_out.write_filtered_data(data_filter);
  std::string output = "./" + saving_dir + "/solution-" + Utilities::int_to_string(step, 5) + ".h5";
  data_out.write_hdf5_parallel(data_filter, output, MPI_COMM_WORLD);
  std::vector<XDMFEntry> xdmf_entries;
  auto new_xdmf_entry = data_out.create_xdmf_entry(data_filter, output, step, MPI_COMM_WORLD);
  xdmf_entries.push_back(new_xdmf_entry);
  output = "./" + saving_dir + "/solution-" + Utilities::int_to_string(step, 5) + ".xdmf";
  data_out.write_xdmf_file(xdmf_entries, output, MPI_COMM_WORLD);
  output = "./" + saving_dir + "/solution-" + Utilities::int_to_string(step, 5) + ".vtu";
  data_out.write_vtu_in_parallel(output, MPI_COMM_WORLD);

  /*--- Save high order mapping ---*/
  output = "./" + saving_dir + "/solution_high_order-" + Utilities::int_to_string(step, 5) + ".vtu";
  DataOutBase::VtkFlags flags_high_order;
  flags_high_order.write_higher_order_cells = true;
  data_out.set_flags(flags_high_order);
  data_out.build_patches(mapping, EquationData::degree_mapping, DataOut<dim>::curved_inner_cells);
  data_out.write_vtu_in_parallel(output, MPI_COMM_WORLD);

  /*--- Serialization ---*/
  if(save_for_restart) {
    parallel::distributed::SolutionTransfer<dim, LinearAlgebra::distributed::Vector<double>>
    solution_transfer_density(dof_handler_density);
    parallel::distributed::SolutionTransfer<dim, LinearAlgebra::distributed::Vector<double>>
    solution_transfer_velocity(dof_handler_velocity);
    parallel::distributed::SolutionTransfer<dim, LinearAlgebra::distributed::Vector<double>>
    solution_transfer_pressure(dof_handler_pressure);

    rho_old.update_ghost_values();
    u_old.update_ghost_values();
    pres_old.update_ghost_values();

    solution_transfer_density.prepare_for_serialization(rho_old);
    solution_transfer_velocity.prepare_for_serialization(u_old);
    solution_transfer_pressure.prepare_for_serialization(pres_old);

    triangulation.save("./" + saving_dir + "/solution_ser-" + Utilities::int_to_string(step, 5));
  }

  /*--- Call this function to be sure to be able to write again on these fields ---*/
  rho_old.zero_out_ghost_values();
  u_old.zero_out_ghost_values();
  pres_old.zero_out_ghost_values();
  theta_old.zero_out_ghost_values();
  rho_prime_old.zero_out_ghost_values();
  pres_prime_old.zero_out_ghost_values();
}


// The following function is used in determining the maximum velocity
// in order to compute the CFL
//
template<int dim>
double EulerSolver<dim>::get_max_velocity() {
  const int n_q_points = quadrature_velocity.size();

  std::vector<Vector<double>> velocity_values(n_q_points, Vector<double>(dim));

  FEValues<dim> fe_values_velocity(mapping, fe_velocity, quadrature_velocity, update_values);

  double max_local_velocity = 0.0;

  /*--- Loop over all cells ---*/
  for(const auto& cell: dof_handler_velocity.active_cell_iterators()) {
    if(cell->is_locally_owned()) {
      fe_values_velocity.reinit(cell);

      fe_values_velocity.get_function_values(u_old, velocity_values);

      for(int q = 0; q < n_q_points; q++) {
        max_local_velocity = std::max(max_local_velocity, std::sqrt(velocity_values[q][0]*velocity_values[q][0] +
                                                                    velocity_values[q][1]*velocity_values[q][1]));
      }
    }
  }

  const double max_velocity = Utilities::MPI::max(max_local_velocity, MPI_COMM_WORLD);

  return max_velocity;
}

// The following function is used in determining the minimum density
//
template<int dim>
double EulerSolver<dim>::get_min_density() {
  FEValues<dim> fe_values(mapping, fe_density, quadrature_density, update_values);
  std::vector<double> solution_values(quadrature_density.size());

  double min_local_density = std::numeric_limits<double>::max();

  /*--- Loop over all cells ---*/
  for(const auto& cell: dof_handler_density.active_cell_iterators()) {
    if(cell->is_locally_owned()) {
      fe_values.reinit(cell);
      if(IMEX_stage == 2) {
        fe_values.get_function_values(rho_s_2, solution_values);
      }
      else if(IMEX_stage == 3) {
        fe_values.get_function_values(rho_s_3, solution_values);
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

// The following function is used in determining the maximum density
//
template<int dim>
double EulerSolver<dim>::get_max_density() {
  if(IMEX_stage == 2) {
    return rho_s_2.linfty_norm();
  }
  if(IMEX_stage == 3) {
    return rho_s_3.linfty_norm();
  }

  return rho_curr.linfty_norm();
}

// The following function is used in determining the maximum celerity
//
template<int dim>
double EulerSolver<dim>::compute_max_celerity() {
  FEValues<dim> fe_values(mapping, fe_pressure, quadrature_pressure, update_values);
  std::vector<double> solution_values_pressure(quadrature_pressure.size()),
                      solution_values_density(quadrature_pressure.size());

  double max_local_celerity = std::numeric_limits<double>::min();

  /*--- Loop over all cells ---*/
  for(const auto& cell: dof_handler_pressure.active_cell_iterators()) {
    if(cell->is_locally_owned()) {
      fe_values.reinit(cell);
      fe_values.get_function_values(pres_old, solution_values_pressure);
      fe_values.get_function_values(rho_old, solution_values_density);

      for(unsigned int q = 0; q < quadrature_pressure.size(); ++q) {
        max_local_celerity = std::max(max_local_celerity,
                                      std::sqrt(EquationData::Cp_Cv*solution_values_pressure[q]/solution_values_density[q]));
      }
    }
  }

  return Utilities::MPI::max(max_local_celerity, MPI_COMM_WORLD);
}

// The following function is used in determining the maximum advective Courant numbers along the two directions
//
template<int dim>
std::pair<double, double> EulerSolver<dim>::compute_max_Cu_x_z() {
  FEValues<dim>               fe_values(mapping, fe_velocity, quadrature_velocity, update_values);
  std::vector<Vector<double>> solution_values_velocity(quadrature_velocity.size(), Vector<double>(dim));

  double max_Cu_x = std::numeric_limits<double>::min();
  double max_Cu_z = std::numeric_limits<double>::min();

  /*--- Loop over all cells ---*/
  for(const auto& cell: dof_handler_velocity.active_cell_iterators()) {
    if(cell->is_locally_owned()) {
      fe_values.reinit(cell);
      fe_values.get_function_values(u_old, solution_values_velocity);

      for(unsigned int q = 0; q < quadrature_pressure.size(); ++q) {
        max_Cu_x = std::max(max_Cu_x,
                            EquationData::degree_u*std::abs(solution_values_velocity[q](0))*dt/cell->extent_in_direction(0));
        max_Cu_z = std::max(max_Cu_z,
                            EquationData::degree_u*std::abs(solution_values_velocity[q](1))*dt/cell->extent_in_direction(1));
      }
    }
  }

  return std::make_pair(Utilities::MPI::max(max_Cu_x, MPI_COMM_WORLD), Utilities::MPI::max(max_Cu_z, MPI_COMM_WORLD));
}

// The following function is used in determining the maximum Courant number along the two directions
//
template<int dim>
std::pair<double, double> EulerSolver<dim>::compute_max_C_x_z() {
  FEValues<dim> fe_values(mapping, fe_pressure, quadrature_pressure, update_values);
  std::vector<double> solution_values_pressure(quadrature_pressure.size()),
                      solution_values_density(quadrature_pressure.size());

  double max_C_x = std::numeric_limits<double>::min();
  double max_C_z = std::numeric_limits<double>::min();

  /*--- Loop over all cells ---*/
  for(const auto& cell: dof_handler_pressure.active_cell_iterators()) {
    if(cell->is_locally_owned()) {
      fe_values.reinit(cell);
      fe_values.get_function_values(pres_old, solution_values_pressure);
      fe_values.get_function_values(rho_old, solution_values_density);

      for(unsigned int q = 0; q < quadrature_pressure.size(); ++q) {
        double local_celerity = std::sqrt(EquationData::Cp_Cv*solution_values_pressure[q]/solution_values_density[q]);
        max_C_x = std::max(max_C_x, 1.0/Ma*EquationData::degree_u*local_celerity*dt/cell->extent_in_direction(0));
        max_C_z = std::max(max_C_z, 1.0/Ma*EquationData::degree_u*local_celerity*dt/cell->extent_in_direction(1));
      }
    }
  }

  return std::make_pair(Utilities::MPI::max(max_C_x, MPI_COMM_WORLD), Utilities::MPI::max(max_C_z, MPI_COMM_WORLD));
}

// The following function is used to compute some integral quantities
//
template<int dim>
void EulerSolver<dim>::postprocess_integral_quantities(const double time) {
  const int n_q_points = quadrature_density.size();

  FEValues<dim> fe_values_density(mapping, fe_density, quadrature_density,
                                  update_values | update_JxW_values);
  FEValues<dim> fe_values_pressure(mapping, fe_pressure, quadrature_density,
                                   update_values);
  FEValues<dim> fe_values_velocity(mapping, fe_velocity, quadrature_density,
                                   update_values);

  std::vector<double>         density_values(n_q_points);
  std::vector<double>         pressure_values(n_q_points);
  std::vector<Vector<double>> velocity_values(n_q_points, Vector<double>(dim));

  double local_density = 0.0;
  double local_energy  = 0.0;
  double local_s       = 0.0;
  double local_eta     = 0.0;

  auto tmp_cell_pres = dof_handler_pressure.begin_active();
  auto tmp_cell_vel  = dof_handler_velocity.begin_active();
  for(const auto& cell : dof_handler_density.active_cell_iterators()) {
    if(cell->is_locally_owned()) {
      fe_values_density.reinit(cell);
      fe_values_density.get_function_values(rho_old, density_values);

      fe_values_pressure.reinit(tmp_cell_pres);
      fe_values_pressure.get_function_values(pres_old, pressure_values);

      fe_values_velocity.reinit(tmp_cell_vel);
      fe_values_velocity.get_function_values(u_old, velocity_values);

      for(int q = 0; q < n_q_points; q++) {
        local_density += density_values[q]*
                         fe_values_density.JxW(q);

        local_energy += (1.0/(EquationData::Cp_Cv - 1.0)*pressure_values[q] +
                         0.5*density_values[q]*velocity_values[q].norm_sqr())*
                        fe_values_density.JxW(q);

        local_s += std::log(pressure_values[q]*std::pow(density_values[q], EquationData::Cp_Cv))*
                   fe_values_density.JxW(q);

        local_eta += ((-density_values[q]*
                       std::log(pressure_values[q]*std::pow(density_values[q], EquationData::Cp_Cv)))/
                      (EquationData::Cp_Cv - 1.0))*
                     fe_values_density.JxW(q);
      }
    }
    ++tmp_cell_pres;
    ++tmp_cell_vel;
  }

  const auto density_integral = Utilities::MPI::sum(local_density, MPI_COMM_WORLD);
  const auto energy_integral  = Utilities::MPI::sum(local_energy, MPI_COMM_WORLD);
  const auto s_integral       = Utilities::MPI::sum(local_s, MPI_COMM_WORLD);
  const auto eta_integral     = Utilities::MPI::sum(local_eta, MPI_COMM_WORLD);
  if(restart) {
    if(time == time_restart) {
      rho0_integral = density_integral;
      output_integral_rho << std::setprecision(12)
                          << std::setw(35) << std::left << "Time"
                          << std::setw(35) << std::left << "Density integral"
                          << std::setw(35) << std::left << "Abs. diff. density integral"
                          << std::setw(35) << std::left << "Rel. diff. density integral"
                          << std::endl;

      rhoE0_integral = energy_integral;
      output_integral_rhoE << std::setprecision(12)
                           << std::setw(35) << std::left << "Time"
                           << std::setw(35) << std::left << "Tot. energy integral"
                           << std::setw(35) << std::left << "Abs. diff. tot. energy integral"
                           << std::setw(35) << std::left << "Rel. diff. tot. energy integral"
                           << std::endl;

      s0_integral = s_integral;
      output_integral_s << std::setprecision(12)
                        << std::setw(35) << std::left << "Time"
                        << std::setw(35) << std::left << "Physic entropy integral"
                        << std::setw(35) << std::left << "Abs. diff. physic entropy integral"
                        << std::setw(35) << std::left << "Rel. diff. physic entropy integral"
                        << std::endl;

      eta0_integral = eta_integral;
      output_integral_eta << std::setprecision(12)
                          << std::setw(35) << std::left << "Time"
                          << std::setw(35) << std::left << "Math. entropy integral"
                          << std::setw(35) << std::left << "Abs. diff. math. entropy integral"
                          << std::setw(35) << std::left << "Rel. diff. math. entropy integral"
                          << std::endl;
    }
  }
  else {
    if(time == t0) {
      rho0_integral = density_integral;
      output_integral_rho << std::setprecision(12)
                          << std::setw(35) << std::left << "Time"
                          << std::setw(35) << std::left << "Density integral"
                          << std::setw(35) << std::left << "Abs. diff. density integral"
                          << std::setw(35) << std::left << "Rel. diff. density integral"
                          << std::endl;

      rhoE0_integral = energy_integral;
      output_integral_rhoE << std::setprecision(12)
                           << std::setw(35) << std::left << "Time"
                           << std::setw(35) << std::left << "Tot. energy integral"
                           << std::setw(35) << std::left << "Abs. diff. tot. energy integral"
                           << std::setw(35) << std::left << "Rel. diff. tot. energy integral"
                           << std::endl;

      s0_integral = s_integral;
      output_integral_s << std::setprecision(12)
                        << std::setw(35) << std::left << "Time"
                        << std::setw(35) << std::left << "Physic entropy integral"
                        << std::setw(35) << std::left << "Abs. diff. physic entropy integral"
                        << std::setw(35) << std::left << "Rel. diff. physic entropy integral"
                        << std::endl;

      eta0_integral = eta_integral;
      output_integral_eta << std::setprecision(12)
                          << std::setw(35) << std::left << "Time"
                          << std::setw(35) << std::left << "Math. entropy integral"
                          << std::setw(35) << std::left << "Abs. diff. math. entropy integral"
                          << std::setw(35) << std::left << "Rel. diff. math. entropy integral"
                          << std::endl;
    }
  }
  if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0) {
    output_integral_rho << std::setprecision(12)
                        << std::setw(35) << std::left << time
                        << std::setw(35) << std::left << density_integral
                        << std::setw(35) << std::left << std::abs(density_integral - rho0_integral)
                        << std::setw(35) << std::left << std::abs(density_integral - rho0_integral)/std::abs(rho0_integral)
                        << std::endl;

    output_integral_rhoE << std::setprecision(12)
                         << std::setw(35) << std::left << time
                         << std::setw(35) << std::left << energy_integral
                         << std::setw(35) << std::left << std::abs(energy_integral - rhoE0_integral)
                         << std::setw(35) << std::left << std::abs(energy_integral - rhoE0_integral)/std::abs(rhoE0_integral)
                         << std::endl;

    output_integral_s << std::setprecision(12)
                      << std::setw(35) << std::left << time
                      << std::setw(35) << std::left << s_integral
                      << std::setw(35) << std::left << s_integral - s0_integral
                      << std::setw(35) << std::left << (s_integral - s0_integral)/std::abs(s0_integral)
                      << std::endl;

    output_integral_eta << std::setprecision(12)
                        << std::setw(35) << std::left << time
                        << std::setw(35) << std::left << eta_integral
                        << std::setw(35) << std::left << eta_integral - eta0_integral
                        << std::setw(35) << std::left << (eta_integral - eta0_integral)/std::abs(eta0_integral)
                        << std::endl;
  }
}


// @sect{ <code>EulerSolver::run</code> }

// This is the time marching function, which starting at <code>t0</code>
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

  /*--- Initialize and save initial state ---*/
  double time    = t0;
  unsigned int n = 0;

  if(restart && !as_initial_conditions) {
    n    = step_restart;
    time = time_restart;
  }
  else {
    output_results(0);
  }
  postprocess_integral_quantities(time);

  /*--- Time loop ---*/
  double tot_fixed_point_iters = 0.0;
  while(std::abs(T - time) > 1e-10) {
    time += dt;
    n++;
    pcout << "Step = " << n << " Time = " << time << std::endl;

    /*--- Second stage of the IMEX operator ---*/
    IMEX_stage = 2;
    euler_matrix.set_IMEX_stage(IMEX_stage);

    verbose_cout << "  Update density stage " << IMEX_stage << std::endl;
    update_density();
    pcout << "Minimum density " << get_min_density() << std::endl;
    pcout << "Maximum density " << get_max_density() << std::endl;

    verbose_cout << "  Fixed point pressure stage " << IMEX_stage << std::endl;
    // Set the current density to the operator and set the variables for multigrid
    euler_matrix.set_rho_for_fixed(rho_s_2);
    pres_fixed.equ(1.0, pres_old);
    pres_prime_fixed.equ(1.0, pres_prime_old);
    u_fixed.equ(1.0, u_old);
    unsigned int iter = perform_fixed_point_loop();
    tot_fixed_point_iters += (iter + 1);
    // Assign the fields after the fixed point loop
    pres_prime_s_2.equ(1.0, pres_prime_fixed);
    pres_s_2.equ(1.0, pres_fixed);
    u_s_2.equ(1.0, u_fixed);

    /*--- Third stage of IMEX operator ---*/
    IMEX_stage = 3;
    euler_matrix.set_IMEX_stage(IMEX_stage);

    verbose_cout << "  Update density stage " << IMEX_stage << std::endl;
    update_density();
    pcout << "Minimum density " << get_min_density() << std::endl;
    pcout << "Maximum density " << get_max_density() << std::endl;

    verbose_cout << "  Fixed point pressure stage " << IMEX_stage << std::endl;
    // Set the current density to the operator and set the variables for multigrid
    euler_matrix.set_rho_for_fixed(rho_s_3);
    pres_fixed.equ(1.0, pres_s_2);
    pres_prime_fixed.equ(1.0, pres_prime_s_2);
    u_fixed.equ(1.0, u_s_2);
    iter = perform_fixed_point_loop();
    tot_fixed_point_iters += (iter + 1);
    // Assign the fields after the fixed point loop
    pres_prime_s_3.equ(1.0, pres_prime_fixed);
    pres_s_3.equ(1.0, pres_fixed);
    u_s_3.equ(1.0, u_fixed);

    /*--- Final stage of RK scheme to update ---*/
    IMEX_stage = 4;
    euler_matrix.set_IMEX_stage(IMEX_stage);

    verbose_cout << "  Update density" << std::endl;
    update_density();
    pcout << "Minimum density " << get_min_density() << std::endl;
    pcout << "Maximum density " << get_max_density() << std::endl;

    verbose_cout << "  Update velocity" << std::endl;
    // Set the current density to the operator and set the variables for multigrid
    euler_matrix.set_rho_for_fixed(rho_curr);
    update_velocity();

    verbose_cout << "  Update pressure" << std::endl;
    update_pressure();

    /*--- Update before applying damping layer ---*/
    rho_old.equ(1.0, rho_curr);
    u_old.equ(1.0, u_curr);

    /*--- Apply the damping layer for the vertical component ---*/
    rho_old.add(1.0, dt_tau_rho);
    rho_old.scale(dt_tau_rho_aux);
    u_old.add(1.0, dt_tau_u);
    u_old.scale(dt_tau_u_aux);
    pres_old.add(1.0, dt_tau_pres);
    pres_old.scale(dt_tau_pres_aux);

    /*--- Apply the damping layer for the right lateral part ---*/
    rho_old.add(1.0, dt_tau_rho_right);
    rho_old.scale(dt_tau_rho_aux_right);
    u_old.add(1.0, dt_tau_u_right);
    u_old.scale(dt_tau_u_aux_right);
    pres_old.add(1.0, dt_tau_pres_right);
    pres_old.scale(dt_tau_pres_aux_right);

    /*--- Apply the damping layer for the left lateral part ---*/
    rho_old.add(1.0, dt_tau_rho_left);
    rho_old.scale(dt_tau_rho_aux_left);
    u_old.add(1.0, dt_tau_u_left);
    u_old.scale(dt_tau_u_aux_left);
    pres_old.add(1.0, dt_tau_pres_left);
    pres_old.scale(dt_tau_pres_aux_left);

    /*--- Update density and pressure perturbation for the next step ---*/
    rho_prime_old.equ(1.0, rho_old);
    rho_prime_old.add(-1.0, rho_bar);
    pres_prime_old.equ(1.0, pres_old);
    pres_prime_old.add(-1.0, pres_bar);

    /*--- Compute auxiliary post-processing data ---*/
    const double max_celerity = compute_max_celerity();
    pcout<< "Maximum celerity = " << 1.0/Ma*max_celerity << std::endl;
    pcout << "CFL_c = " << 1.0/Ma*dt*max_celerity*EquationData::degree_u*
                           std::sqrt(dim)/GridTools::minimal_cell_diameter(triangulation, mapping) << std::endl;
    const auto max_C_x_z = compute_max_C_x_z();
    pcout << "CFL_c_x = " << max_C_x_z.first  << std::endl;
    pcout << "CFL_c_z = " << max_C_x_z.second << std::endl;
    const double max_velocity = get_max_velocity();
    pcout<< "Maximum velocity = " << max_velocity << std::endl;
    pcout << "CFL_u = " << dt*max_velocity*EquationData::degree_u*
                           std::sqrt(dim)/GridTools::minimal_cell_diameter(triangulation, mapping) << std::endl;
    const auto max_Cu_x_z = compute_max_Cu_x_z();
    pcout << "CFL_u_x = " << max_Cu_x_z.first  << std::endl;
    pcout << "CFL_u_z = " << max_Cu_x_z.second << std::endl;

    postprocess_integral_quantities(time);

    /*--- Save the results each 'output_interval' steps ---*/
    if(n % output_interval == 0) {
      verbose_cout << "Plotting Solution final" << std::endl;
      output_results(n);
      pcout << "average u = " << VectorTools::compute_mean_value(mapping, dof_handler_velocity, quadrature_velocity, u_old, 0) << std::endl;
      pcout << "average w = " << VectorTools::compute_mean_value(mapping, dof_handler_velocity, quadrature_velocity, u_old, 1) << std::endl;
    }

    if(time + dt > T && T - time > 1e-10) {
      /*--- Recompute and reset the time if needed towards the end of the simulation to stop at the proper final time ---*/
      dt = T - time;
      euler_matrix.set_dt(dt);
    }
  }

  /*--- Save the final results if not previously done ---*/
  pcout << "Average fixed point iterations: "
        << static_cast<double>(tot_fixed_point_iters)/((EquationData::n_stages - 1)*n)
        << std::endl;
  if(n % output_interval != 0) {
    verbose_cout << "Plotting Solution final" << std::endl;
    output_results(n);
    pcout << "average u = " << VectorTools::compute_mean_value(mapping, dof_handler_velocity, quadrature_velocity, u_old, 0) << std::endl;
    pcout << "average w = " << VectorTools::compute_mean_value(mapping, dof_handler_velocity, quadrature_velocity, u_old, 1) << std::endl;
  }
}


// @sect{ The main function }

// The main function is quite standard. We just need to declare the EulerSolver
// instance and let the simulation run.
//
int main(int argc, char *argv[]) {
  try {
    RunTimeParameters::Data_Storage data;
    data.read_data("parameter-file.prm");

    Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, -1);

    const auto& curr_rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
    deallog.depth_console(data.verbose && curr_rank == 0 ? 2 : 0);

    EulerSolver<2> test(data);
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
