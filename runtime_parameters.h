/* This file is part of the Atmospheric_Flows/tree/Mesoscale repository and subject to the
   LGPL license. See the LICENSE file in the top level directory of this
   project for details. */

/*--- Author: Giuseppe Orlando, 2023. ---*/

// @sect{Include files}

// We start by including the necessary deal.II header files and some C++
// related ones.
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

  class Data_Storage {
  public:
    Data_Storage(); /*--- Class constructor ---*/

    void read_data(const std::string& filename); /*--- The function that actually reads the parameters ---*/

    double initial_time; /*--- Variable to set the initial time (default equal to 0) ---*/
    double final_time;   /*--- Variable to set the final time ---*/

    // The present code is meant to work using non-dimensional variables and using
    // the non-dimensional equations described in Orlando et al., JCP, 2022.
    // If one wishes to consider a dimensional version, it is sufficient
    // to set the Mach numer equal to 1 and the Froude number equal to 1/sqrt(g),
    // where g is, as usual, the acceleration of gravity.
    //
    double Mach;   /*--- The Mach number ---*/
    double Froude; /*--- The Froude number ---*/
    double dt;     /*--- The time-step ---*/

    unsigned int n_global_refines; /*--- Number of global refinements for the initial coarse mesh ---*/

    unsigned int max_iterations; /*--- Maximum number of iterations for the linear solver ---*/
    double       eps;            /*--- Tolerance for the linear solver ---*/

    bool         verbose;          /*--- Choose if being verboe or not ---*/
    unsigned int output_interval; /*--- Set how often save the fields ---*/

    std::string  dir; /*--- Directory where the data are saved. This has to be created before launching the code
                            and we assume it is a subfolder of the folder with the executable and the parameter file.
                            This behaviour can be easily changed giving the absolute path ---*/

    /*--- Auxiliary parameters related to restart ---*/
    bool         restart;
    bool         save_for_restart;
    unsigned int step_restart;
    double       time_restart;
    bool         as_initial_conditions;

  protected:
    ParameterHandler prm; /*--- Auxiliary variable which handles the parameters ---*/
  };

  // In the constructor of this class we declare all the parameters.
  // We employ the 'enter_subsection' to divide into categories and
  // the 'declare_entry' to declare a certain parameter to be setted.
  //
  Data_Storage::Data_Storage(): initial_time(0.0),
                                final_time(1.0),
                                Mach(1.0),
                                Froude(1.0),
                                dt(5e-4),
                                n_global_refines(0),
                                max_iterations(1000),
                                eps(1e-12),
                                verbose(true),
                                output_interval(15),
                                restart(false),
                                save_for_restart(false),
                                step_restart(0),
                                time_restart(0.0),
                                as_initial_conditions(false) {
    prm.enter_subsection("Physical data");
    {
      prm.declare_entry("initial_time",
                        "0.0",
                        Patterns::Double(0.0),
                        " The initial time of the simulation. ");
      prm.declare_entry("final_time",
                        "1.0",
                        Patterns::Double(0.0),
                        " The final time of the simulation. ");
      prm.declare_entry("Mach",
                        "1.0",
                        Patterns::Double(0.0),
                        " The Mach number. ");
      prm.declare_entry("Froude",
                        "1.0",
                        Patterns::Double(0.0),
                        " The Froude number. ");
    }
    prm.leave_subsection();

    prm.enter_subsection("Time step data");
    {
      prm.declare_entry("dt",
                        "5e-4",
                        Patterns::Double(0.0),
                        " The time step size. ");
      prm.declare_entry("time_restart",
                        "5e-4",
                        Patterns::Double(0.0),
                        " The time of restart. ");
    }
    prm.leave_subsection();

    prm.enter_subsection("Space discretization");
    {
      prm.declare_entry("n_of_refines",
                        "3",
                        Patterns::Integer(0, 15),
                        " The number of global refinements we want for the mesh. ");
    }
    prm.leave_subsection();

    prm.enter_subsection("Data solve");
    {
      prm.declare_entry("max_iterations",
                        "1000",
                        Patterns::Integer(1, 30000),
                        " The maximal number of iterations GMRES must make. ");
      prm.declare_entry("eps",
                        "1e-12",
                        Patterns::Double(0.0),
                        " The stopping criterion. ");
      prm.declare_entry("step_restart",
                        "0",
                         Patterns::Integer(0, 100000000),
                         " The step at which restart occurs");
    }
    prm.leave_subsection();

    prm.declare_entry("verbose",
                      "true",
                      Patterns::Bool(),
                      " This indicates whether the output of the solution "
                      "process should be verbose. ");

    prm.declare_entry("output_interval",
                      "1",
                      Patterns::Integer(1),
                      " This indicates between how many time steps we print "
                      "the solution. ");

    prm.declare_entry("saving directory", "SimTest");

    prm.declare_entry("restart",
                      "false",
                      Patterns::Bool(),
                      " This indicates whether we are in presence of a "
                      "restart or not. ");
    prm.declare_entry("save_for_restart",
                      "false",
                      Patterns::Bool(),
                      " This indicates whether we want to save for possible "
                      "restart or not. ");
    prm.declare_entry("as_initial_conditions",
                      "false",
                      Patterns::Bool(),
                      " This indicates whether restart is used as initial condition "
                      "or to continue the simulation. ");
  }

  // Function to read all declared parameters in the constructor
  //
  void Data_Storage::read_data(const std::string& filename) {
    std::ifstream file(filename);
    AssertThrow(file, ExcFileNotOpen(filename));

    prm.parse_input(file);

    prm.enter_subsection("Physical data");
    {
      initial_time = prm.get_double("initial_time");
      final_time   = prm.get_double("final_time");
      Mach         = prm.get_double("Mach");
      Froude       = prm.get_double("Froude");
    }
    prm.leave_subsection();

    prm.enter_subsection("Time step data");
    {
      dt           = prm.get_double("dt");
      time_restart = prm.get_double("time_restart");
    }
    prm.leave_subsection();

    prm.enter_subsection("Space discretization");
    {
      n_global_refines = prm.get_integer("n_of_refines");
    }
    prm.leave_subsection();

    prm.enter_subsection("Data solve");
    {
      max_iterations = prm.get_integer("max_iterations");
      eps            = prm.get_double("eps");
      step_restart   = prm.get_integer("step_restart");
    }
    prm.leave_subsection();

    verbose = prm.get_bool("verbose");

    output_interval = prm.get_integer("output_interval");

    dir = prm.get("saving directory");

    /*--- Read parameters related to restart ---*/
    restart               = prm.get_bool("restart");
    save_for_restart      = prm.get_bool("save_for_restart");
    as_initial_conditions = prm.get_bool("as_initial_conditions");
  }

} // namespace RunTimeParameters
