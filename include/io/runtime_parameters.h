/*--- Author: Giuseppe Orlando, 2025. ---*/

// @sect{Include files}

// We start by including the necessary deal.II header files and some C++
// related ones
//
#include <deal.II/base/parameter_handler.h>

#include <fstream>

// @sect{Run-time parameters}
//
// Since our method has several parameters that can be fine-tuned we put them
// into an external file, so that they can be determined at run-time.
//
namespace RunTimeParameters {
  using namespace dealii;

  template<typename T = double>
  class Data_Storage {
  public:
    Data_Storage(); /*--- Class constructor ---*/

    void read_data(const std::string& filename); /*--- The function that actually reads the parameters ---*/

    /*--- Start with physical parameters ---*/
    T initial_time; /*--- Variable to set the initial time (default equal to 0) ---*/
    T final_time;   /*--- Variable to set the final time ---*/

    // The present code is meant to work using non-dimensional variables and using
    // the non-dimensional equations described in Orlando et al., JCP, 2022.
    // If one wishes to consider a dimensional version, it is sufficient
    // to set the Mach numer equal to 1 and the Froude number equal to 1/sqrt(g),
    // where g is, as usual, the acceleration of gravity.
    //
    T Mach;   /*--- The Mach number ---*/
    T Froude; /*--- The Froude number ---*/

    T L_ref;   /*--- Reference length (not used so far) ---*/
    T u_ref;   /*--- Reference velocity (not used so far) ---*/
    T p_ref;   /*--- Reference pressure (not used so far) ---*/
    T T_ref;   /*--- Reference temperature (not used so far) ---*/
    T rho_ref; /*--- Reference density (not used so far) ---*/

    T h;  /*--- Hill height (not used so far) ---*/
    T xc; /*--- x-Center of the hill (not used so far) ---*/
    T yc; /*--- y-Center of the hill (not used so far) ---*/
    T ac; /*--- Width of the hill (not used so far) ---*/

    /*--- Numerical parameters ---*/
    T dt; /*--- The time-step ---*/

    T atol_fixed_point; /*--- Absolute tolerance for the fixed point loop ---*/
    T rtol_fixed_point; /*--- Relative tolerance for the fixed point loop ---*/

    /*--- Mesh parameters ---*/
    unsigned n_global_refines;    /*--- Number of global refinements for the initial (coarse) mesh ---*/
    unsigned max_loc_refinements; /*--- Maximum number of refinements allowed ---*/
    unsigned min_loc_refinements; /*--- Minimum number of refinements allowed ---*/

    unsigned refinement_iterations; /*--- How often performin mesh adaptation ---*/

    /*--- Parameters related to the linear solver ---*/
    unsigned max_iterations;  /*--- Maximum number of iterations for the linear solver ---*/
    T        atol_iterative;  /*--- Absolute tolerance for the linear solver ---*/
    T        rtol_iterative;  /*--- Relative tolerance for the linear solver ---*/

    /*--- Parameters related to the output ---*/
    bool     verbose;         /*--- Choose if being verboe or not ---*/
    unsigned output_interval; /*--- Set how often save the fields ---*/

    std::string dir; /*--- Directory where the data are saved. This has to be created before launching the code
                           and we assume it is a subfolder of the folder with the executable and the parameter file.
                           This behaviour can be easily changed giving, e.g., the absolute path ---*/

    /*--- Auxiliary parameters related to restart ---*/
    bool     restart;
    bool     save_for_restart;
    unsigned step_restart;
    T        time_restart;
    bool     as_initial_conditions;

  protected:
    ParameterHandler prm; /*--- Auxiliary variable which handles the parameters ---*/
  };

  // In the constructor of this class we declare all the parameters.
  // We employ the 'enter_subsection' to divide into categories and
  // the 'declare_entry' to declare a certain parameter to be setted.
  //
  template<typename T>
  Data_Storage<T>::Data_Storage(): initial_time(0.0),
                                   final_time(1.0),
                                   Mach(1.0),
                                   Froude(0.319275428407050),
                                   L_ref(1.0),
                                   u_ref(1.0),
                                   p_ref(1.0),
                                   T_ref(1.0),
                                   rho_ref(1.0),
                                   h(1.0),
                                   xc(1.0),
                                   yc(1.0),
                                   ac(1.0),
                                   dt(5e-4),
                                   atol_fixed_point(1e-12),
                                   rtol_fixed_point(1e-10),
                                   n_global_refines(0),
                                   max_loc_refinements(0),
                                   min_loc_refinements(0),
                                   refinement_iterations(0),
                                   max_iterations(1000),
                                   atol_iterative(1e-14),
                                   rtol_iterative(1e-12),
                                   verbose(true),
                                   output_interval(15),
                                   restart(false),
                                   save_for_restart(false),
                                   step_restart(0),
                                   time_restart(0.0),
                                   as_initial_conditions(false) {
    /*--- Start declaring entries for the physical parameters ---*/
    prm.enter_subsection("Physical data");
    {
      prm.declare_entry("initial_time",
                        "0.0",
                        Patterns::Double(0.0),
                        "The initial time of the simulation.");
      prm.declare_entry("final_time",
                        "1.0",
                        Patterns::Double(0.0),
                        "The final time of the simulation.");

      prm.declare_entry("Mach",
                        "1.0",
                        Patterns::Double(0.0),
                        " The Mach number.");
      prm.declare_entry("Froude",
                        "0.319275428407050",
                        Patterns::Double(0.0),
                        "The Froude number.");

      prm.declare_entry("L_ref",
                        "1.0",
                        Patterns::Double(0.0),
                        "The reference length.");
      prm.declare_entry("u_ref",
                        "1.0",
                        Patterns::Double(0.0),
                        "The reference velocity.");
      prm.declare_entry("p_ref",
                        "1.0",
                        Patterns::Double(0.0),
                        "The reference pressure.");
      prm.declare_entry("T_ref",
                        "1.0",
                        Patterns::Double(0.0),
                        "The reference temperature.");
      prm.declare_entry("rho_ref",
                        "1.0",
                        Patterns::Double(0.0),
                        "The reference density.");

      prm.declare_entry("h",
                        "1.0",
                        Patterns::Double(0.0),
                        "The hill height.");
      prm.declare_entry("xc",
                        "1.0",
                        Patterns::Double(0.0),
                        "The x-Center of the hill.");
      prm.declare_entry("yc",
                        "1.0",
                        Patterns::Double(0.0),
                        "The y-Center of the hill.");
      prm.declare_entry("ac",
                        "1.0",
                        Patterns::Double(0.0),
                        "The width of the hill.");
    }
    prm.leave_subsection();

    /*--- Focus now on some numerical parameters ---*/
    prm.enter_subsection("Numerical data");
    {
      prm.declare_entry("dt",
                        "5e-4",
                        Patterns::Double(0.0),
                        "The time step size.");

      prm.declare_entry("atol_fixed_point",
                        "1e-10",
                        Patterns::Double(0.0),
                        "Absolute tolerance for the fixed point loop.");
      prm.declare_entry("rtol_fixed_point",
                        "1e-10",
                        Patterns::Double(0.0),
                        "Relative tolerance for the fixed point loop.");
    }
    prm.leave_subsection();

    /*--- Focus now on some mesh parameters ---*/
    prm.enter_subsection("Mesh parameters");
    {
      prm.declare_entry("n_of_refines",
                        "3",
                        Patterns::Integer(0, 15),
                        "The number of global refinements we want for the mesh.");
      prm.declare_entry("max_loc_refinements",
                        "4",
                         Patterns::Integer(1, 10),
                         " The number of maximum local refinements in case of adaptive mesh.");
      prm.declare_entry("min_loc_refinements",
                        "2",
                         Patterns::Integer(0, 10),
                         " The number of minimum local refinements in case of adaptive mesh.");
      prm.declare_entry("refinement_iterations",
                        "0",
                         Patterns::Integer(0, 100000000),
                         "How ofter performing mesh adaptation if desired.");
    }
    prm.leave_subsection();

    /*--- Focus now on the data of the linear solvers ---*/
    prm.enter_subsection("Data linear solvers");
    {
      prm.declare_entry("max_iterations",
                        "1000",
                        Patterns::Integer(1, 30000),
                        "The maximal number of iterations GMRES must make.");
      prm.declare_entry("atol_iterative",
                        "1e-12",
                        Patterns::Double(0.0),
                        "Absolute tolerance for the linear solver.");
      prm.declare_entry("rtol_iterative",
                        "1e-12",
                        Patterns::Double(0.0),
                        "Relative tolerance for the linear solver.");
    }
    prm.leave_subsection();

    /*--- Focus now on the restart parameters ---*/
    prm.enter_subsection("Restart data");
    {
      prm.declare_entry("time_restart",
                        "5e-4",
                        Patterns::Double(0.0),
                        "The time of restart.");
      prm.declare_entry("step_restart",
                        "0",
                         Patterns::Integer(0, 100000000),
                         "The step at which restart occurs.");
      prm.declare_entry("restart",
                        "false",
                        Patterns::Bool(),
                        "This indicates whether we are in presence of a "
                        "restart or not.");
      prm.declare_entry("save_for_restart",
                        "false",
                        Patterns::Bool(),
                        "This indicates whether we want to save for possible "
                        "restart or not.");
      prm.declare_entry("as_initial_conditions",
                        "false",
                        Patterns::Bool(),
                        "This indicates whether restart is used as initial condition "
                        "or to continue the simulation.");
    }
    prm.leave_subsection();

    /*--- Output related parameters ---*/
    prm.declare_entry("verbose",
                      "true",
                      Patterns::Bool(),
                      "This indicates whether the output of the solution "
                      "process should be verbose.");

    prm.declare_entry("output_interval",
                      "1",
                      Patterns::Integer(1),
                      "This indicates between how many time steps we print "
                      "the solution.");

    prm.declare_entry("saving directory", "SimTest");
  }

  // Function to read all declared parameters in the constructor
  //
  template<typename T>
  void Data_Storage<T>::read_data(const std::string& filename) {
    std::ifstream file(filename);
    AssertThrow(file, ExcFileNotOpen(filename));

    prm.parse_input(file);

    /*--- Start with physical related parameters ---*/
    prm.enter_subsection("Physical data");
    {
      initial_time = prm.get_double("initial_time");
      final_time   = prm.get_double("final_time");

      Mach   = prm.get_double("Mach");
      Froude = prm.get_double("Froude");

      L_ref   = prm.get_double("L_ref");
      u_ref   = prm.get_double("u_ref");
      p_ref   = prm.get_double("p_ref");
      T_ref   = prm.get_double("T_ref");
      rho_ref = prm.get_double("rho_ref");

      h  = prm.get_double("h");
      xc = prm.get_double("xc");
      yc = prm.get_double("yc");
      ac = prm.get_double("ac");
    }
    prm.leave_subsection();

    /*--- Focus now on some numerical parameters ---*/
    prm.enter_subsection("Numerical data");
    {
      dt = prm.get_double("dt");

      atol_fixed_point = prm.get_double("atol_fixed_point");
      rtol_fixed_point = prm.get_double("rtol_fixed_point");
    }
    prm.leave_subsection();

    /*--- Focus now on some mesh parameters ---*/
    prm.enter_subsection("Mesh parameters");
    {
      n_global_refines      = prm.get_integer("n_of_refines");
      max_loc_refinements   = prm.get_integer("max_loc_refinements");
      min_loc_refinements   = prm.get_integer("min_loc_refinements");
      refinement_iterations = prm.get_integer("refinement_iterations");
    }
    prm.leave_subsection();

    /*--- Focus now on the data of the linear solvers ---*/
    prm.enter_subsection("Data linear solvers");
    {
      max_iterations = prm.get_integer("max_iterations");
      atol_iterative = prm.get_double("atol_iterative");
      rtol_iterative = prm.get_double("rtol_iterative");
    }
    prm.leave_subsection();

    /*--- Read parameters related to restart ---*/
    prm.enter_subsection("Restart data");
    {
      time_restart          = prm.get_double("time_restart");
      step_restart          = prm.get_integer("step_restart");
      restart               = prm.get_bool("restart");
      save_for_restart      = prm.get_bool("save_for_restart");
      as_initial_conditions = prm.get_bool("as_initial_conditions");
    }
    prm.leave_subsection();

    /*--- Output related data ---*/
    verbose = prm.get_bool("verbose");

    output_interval = prm.get_integer("output_interval");

    dir = prm.get("saving directory");
  }

} // namespace RunTimeParameters
