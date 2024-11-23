/*--- Author: Giuseppe Orlando, 2024. ---*/

// @sect{Include files}

// We start by including the necessary deal.II header files and some C++
// related ones.
#include <deal.II/base/point.h>
#include <deal.II/base/function.h>

#include <cmath>

constexpr int my_ceil(double num) {
  return (static_cast<float>(static_cast<int>(num)) == num) ?
          static_cast<int>(num) :
          static_cast<int>(num) + ((num > 0) ? 1 : 0);
}

// @sect{Equation data}

// In this namespace, we declare the initial background conditions,
// the Rayleigh dumping profiles and the mapping between reference and physical
// elements using the Gal-Chen mapping.
//
namespace EquationData {
  using namespace dealii;

  static const unsigned int n_stages = 3; /*--- Number of stages of the IMEX scheme ---*/

  static const unsigned int n_vars = 3; /*--- Number of variables for which we solve a linear system ---*/

  /*--- Define axuliary indices related to the dof hadlers order and to linear systems under consideration ---*/
  static const unsigned int RHO_INDEX_SYSTEM = 1;
  static const unsigned int P_INDEX_SYSTEM = 2;
  static const unsigned int U_INDEX_SYSTEM = 3;

  static const unsigned int U_INDEX_DOF = 0;
  static const unsigned int P_INDEX_DOF = 1;
  static const unsigned int RHO_INDEX_DOF = 2;

  /*--- Polynomial degrees. We typically consider the same polynomial degree for all the variables ---*/
  static const unsigned int degree_p   = 4;
  static const unsigned int degree_rho = 4;
  static const unsigned int degree_u   = 4;

  static const double Cp_Cv = 1.4;   /*--- Specific heats ratio ---*/
  static const double R     = 287.0; /*--- Specific gas constant ---*/

  static const double g = 9.81; /*--- Acceleration of gravity ---*/
  static const double N = 0.01; /*--- Buoyancy frequency ---*/

  static const double h  = 400.0;   /*--- Hill height ---*/
  static const double xc = 30000.0; /*--- x-Center of the hill ---*/
  static const double yc = 20000.0; /*--- x-Center of the hill ---*/
  static const double ac = 1000.0;  /*--- Width of the hill ---*/

  static const double x_max = 60000.0; /*--- Extension along horizontal direction ---*/
  static const double y_max = 40000.0; /*--- Extension along y direction ---*/
  static const double z_max = 16000.0; /*--- Extension along vertical direction ---*/

  static const double z_start       = 10000.0; /*--- Start of Rayleigh damping for top boundary ---*/
  static const double x_start_left  = 20000.0; /*--- Start of Rayleigh damping for left boundary ---*/
  static const double x_start_right = 40000.0; /*--- Start of Rayleigh damping for right boundary ---*/
  static const double y_start_left  = 10000.0; /*--- Start of Rayleigh damping for left boundary ---*/
  static const double y_start_right = 30000.0; /*--- Start of Rayleigh damping for right boundary ---*/

  static const double L_ref   = 1000.0;          /*--- Reference length ---*/
  static const double u_ref   = 10.0;            /*--- Reference velocity ---*/
  static const double p_ref   = 100000.0;        /*--- Reference pressure ---*/
  static const double T_ref   = 293.15;          /*--- Reference temperature ---*/
  static const double rho_ref = p_ref/(R*T_ref); /*--- Reference density ---*/

  static const unsigned int degree_mapping          = 2;                                                             /*--- Mapping degree ---*/
  static const unsigned int extra_quadrature_degree = (degree_mapping == 1) ? 0 : my_ceil(0.5*(degree_mapping - 2)); /*--- Extra accuracy
                                                                                                                           for quadratures ---*/

  /* We declare now the class that describes the initial condition for the velocity.
  */
  template<int dim>
  class Velocity: public Function<dim> {
  public:
    Velocity(const double initial_time = 0.0); /*--- Class constructor ---*/

    virtual double value(const Point<dim>&  p,
                         const unsigned int component = 0) const override; /*--- Evaluation for each component ---*/

    virtual void vector_value(const Point<dim>& p,
                              Vector<double>&   values) const override; /*--- Vector evaluation of the velocity ---*/
  };

  // Constructor which relies on the 'Function' constructor.
  //
  template<int dim>
  Velocity<dim>::Velocity(const double initial_time): Function<dim>(dim, initial_time) {}

  // Specify the value for each spatial component. This function is overriden.
  //
  template<int dim>
  double Velocity<dim>::value(const Point<dim>& p, const unsigned int component) const {
    AssertIndexRange(component, dim);

    if(component == 0) {
      return 1.0;
    }
    else {
      return 0.0;
    }
  }

  // Put together for a vector evalutation of the velocity.
  //
  template<int dim>
  void Velocity<dim>::vector_value(const Point<dim>& p, Vector<double>& values) const {
    Assert(values.size() == dim, ExcDimensionMismatch(values.size(), dim));

    for(unsigned int i = 0; i < dim; ++i) {
      values[i] = value(p, i);
    }
  }


  /* We do the same for the pressure. Notice that in order to
     get a dimensional version one should multiply the result by p_ref
  */
  template<int dim>
  class Pressure: public Function<dim> {
  public:
    Pressure(const double initial_time = 0.0); /*--- Class constructor ---*/

    virtual double value(const Point<dim>&  p,
                         const unsigned int component = 0) const override; /*--- Evalution of the pressure ---*/
  };

  // Constructor which again relies on the 'Function' constructor.
  //
  template<int dim>
  Pressure<dim>::Pressure(const double initial_time): Function<dim>(1, initial_time) {}

  // Evaluation depending on the spatial coordinates. The input argument 'component'
  // will be unused but it has to be kept to override
  //
  template<int dim>
  double Pressure<dim>::value(const Point<dim>& p, const unsigned int component) const {
    (void)component;
    AssertIndexRange(component, 1);

    const double Gamma  = (EquationData::Cp_Cv - 1.0)/EquationData::Cp_Cv;

    const double pi_bar = 1.0 - EquationData::g*EquationData::g/(EquationData::N*EquationData::N)*Gamma*EquationData::rho_ref/EquationData::p_ref*
                                (1.0 - std::exp(-EquationData::N*EquationData::N/EquationData::g*p[2]*EquationData::L_ref));

    return std::pow(pi_bar, 1.0/Gamma);
  }


  /* We do the same for the density. Notice that in order to
     get a dimensional version one should multiply the result by rho_ref
  */
  template<int dim>
  class Density: public Function<dim> {
  public:
    Density(const double initial_time = 0.0); /*--- Class constructor ---*/

    virtual double value(const Point<dim>&  p,
                         const unsigned int component = 0) const override; /*--- Evaluation of the density ---*/
  };

  // Constructor which again relies on the 'Function' constructor.
  //
  template<int dim>
  Density<dim>::Density(const double initial_time): Function<dim>(1, initial_time) {}

  // Evaluation depending on the spatial coordinates. The input argument 'component'
  // will be unused but it has to be kept to override
  //
  template<int dim>
  double Density<dim>::value(const Point<dim>& p, const unsigned int component) const {
    (void)component;
    AssertIndexRange(component, 1);

    const double Gamma     = (EquationData::Cp_Cv - 1.0)/EquationData::Cp_Cv;

    const double pi_bar    = 1.0
                           - EquationData::g*EquationData::g/(EquationData::N*EquationData::N)*Gamma*EquationData::rho_ref/EquationData::p_ref*
                             (1.0 - std::exp(-EquationData::N*EquationData::N/EquationData::g*p[2]*EquationData::L_ref));

    const double theta_bar = EquationData::T_ref*std::exp(EquationData::N*EquationData::N/EquationData::g*p[2]*EquationData::L_ref);

    return EquationData::T_ref/theta_bar*std::pow(pi_bar, 1.0/(EquationData::Cp_Cv - 1.0));
  }


  /* We focus now on the Rayleigh damping profile along the vertical direction.
     We create a suitable function for that. This function will be either scalar
     or vectorial (for the velocity). That's why the auxiliary template parameter
     n_comp is present.
  */
  template<int dim, unsigned int n_comp>
  class Rayleigh: public Function<dim> {
  public:
    Rayleigh(const double initial_time = 0.0); /*--- Class constructor ---*/

    virtual double value(const Point<dim>&  p,
                         const unsigned int component = 0) const override; /*--- Damping profile evaluation ---*/

    virtual void vector_value(const Point<dim>& p,
                              Vector<double>&   values) const override; /*--- Damping profile vector evaluation for the velocity ---*/

  private:
    const double z_start; /*--- Starting coordinate of the damping layer ---*/
    const double z_max;   /*--- Ending coordinate of the damping layer ---*/
  };

  // Class constructor, which simply calls the parent class constructor
  // and then initialize some data
  //
  template<int dim, unsigned int n_comp>
  Rayleigh<dim, n_comp>::Rayleigh(const double initial_time): Function<dim>(n_comp, initial_time),
                                                              z_start(EquationData::z_start/EquationData::L_ref),
                                                              z_max(EquationData::z_max/EquationData::L_ref) {}

  // Evaluation of Rayleigh damping profile
  //
  template<int dim, unsigned int n_comp>
  double Rayleigh<dim, n_comp>::value(const Point<dim>& p, const unsigned int component) const {
    (void)component;
    AssertIndexRange(component, n_comp);

    if(p[2] < z_start) {
      return 0.0;
    }

    return 1.2*std::sin(0.5*numbers::PI*(p[2] - z_start)/(z_max - z_start))*
               std::sin(0.5*numbers::PI*(p[2] - z_start)/(z_max - z_start)); /*--- Rayleigh profile expression ---*/
  }

  // We need a vector value instance to deal with the velocity or, more in general,
  // if n_comp > 1.
  //
  template<int dim, unsigned int n_comp>
  void Rayleigh<dim, n_comp>::vector_value(const Point<dim>& p, Vector<double>& values) const {
    Assert(values.size() == n_comp, ExcDimensionMismatch(values.size(), dim));

    for(unsigned int i = 0; i < n_comp; ++i) {
      values[i] = value(p, i);
    }
  }


  /* We create an auxiliary class for the term (1/(1 + dt*tau)) in order to avoid loop.
     The template parameter n_comp has the same meaning of the previous class.
  */
  template<int dim, unsigned int n_comp>
  class Rayleigh_Aux: public Function<dim> {
  public:
    Rayleigh_Aux(const double initial_time = 0.0); /*--- Class constructor ---*/

    virtual double value(const Point<dim>&  p,
                         const unsigned int component = 0) const override; /*--- Damping profile evaluation ---*/

    virtual void vector_value(const Point<dim>& p,
                              Vector<double>&   values) const override; /*--- Damping profile vector evaluation for the velocity ---*/

  private:
    const double z_start; /*--- Starting coordinate of the damping layer ---*/
    const double z_max;   /*--- Ending coordinate of the damping layer ---*/
  };

  // Class constructor, which simply calls the parent class constructor
  // and then initialize some data
  //
  template<int dim, unsigned int n_comp>
  Rayleigh_Aux<dim, n_comp>::Rayleigh_Aux(const double initial_time): Function<dim>(n_comp, initial_time),
                                                                      z_start(EquationData::z_start/EquationData::L_ref),
                                                                      z_max(EquationData::z_max/EquationData::L_ref) {}

  // Evaluation of Rayleigh damping profile
  //
  template<int dim, unsigned int n_comp>
  double Rayleigh_Aux<dim, n_comp>::value(const Point<dim>& p, const unsigned int component) const {
    (void)component;
    AssertIndexRange(component, n_comp);

    if(p[2] < z_start) {
      return 1.0;
    }

    return 1.0/(1.0 + 1.2*std::sin(0.5*numbers::PI*(p[2] - z_start)/(z_max - z_start))*
                          std::sin(0.5*numbers::PI*(p[2] - z_start)/(z_max - z_start)));
  }

  // We need a vector value instance to deal with the velocity or, more in general,
  // if n_comp > 1.
  //
  template<int dim, unsigned int n_comp>
  void Rayleigh_Aux<dim, n_comp>::vector_value(const Point<dim>& p, Vector<double>& values) const {
    Assert(values.size() == n_comp, ExcDimensionMismatch(values.size(), dim));

    for(unsigned int i = 0; i < n_comp; ++i) {
      values[i] = value(p, i);
    }
  }


  /* We do the same for the Rayleigh damping profile along the right lateral boundary.
  */
  template<int dim, unsigned int n_comp>
  class Rayleigh_Right: public Function<dim> {
  public:
    Rayleigh_Right(const double initial_time = 0.0); /*--- Class constructor ---*/

    virtual double value(const Point<dim>&  p,
                         const unsigned int component = 0) const override; /*--- Damping profile evaluation ---*/

    virtual void vector_value(const Point<dim>& p,
                              Vector<double>&   values) const override; /*--- Damping profile vector evaluation for the velocity ---*/

  private:
    const double x_start; /*--- Starting coordinate of the damping layer ---*/
    const double x_max;   /*--- Ending coordinate of the damping layer ---*/
  };

  // Class constructor, which simply calls the parent class constructor
  // and then initialize some data
  //
  template<int dim, unsigned int n_comp>
  Rayleigh_Right<dim, n_comp>::Rayleigh_Right(const double initial_time): Function<dim>(n_comp, initial_time),
                                                                          x_start(EquationData::x_start_right/EquationData::L_ref),
                                                                          x_max(EquationData::x_max/EquationData::L_ref) {}

  // Evaluation of Rayleigh damping profile
  //
  template<int dim, unsigned int n_comp>
  double Rayleigh_Right<dim, n_comp>::value(const Point<dim>& p, const unsigned int component) const {
    (void)component;
    AssertIndexRange(component, n_comp);

    if(p[0] < x_start) {
      return 0.0;
    }

    return 1.2*std::sin(0.5*numbers::PI*(p[0] - x_start)/(x_max - x_start))*
               std::sin(0.5*numbers::PI*(p[0] - x_start)/(x_max - x_start));
  }

  // We need a vector value instance to deal with the velocity or, more in general,
  // if n_comp > 1.
  //
  template<int dim, unsigned int n_comp>
  void Rayleigh_Right<dim, n_comp>::vector_value(const Point<dim>& p, Vector<double>& values) const {
    Assert(values.size() == n_comp, ExcDimensionMismatch(values.size(), dim));
    for(unsigned int i = 0; i < n_comp; ++i)
      values[i] = value(p, i);
  }


  /* We create an auxiliary class for the term (1/(1 + dt*tau)) in order to avoid loop.
  */
  template<int dim, unsigned int n_comp>
  class Rayleigh_Aux_Right: public Function<dim> {
  public:
    Rayleigh_Aux_Right(const double initial_time = 0.0); /*--- Class constructor ---*/

    virtual double value(const Point<dim>&  p,
                         const unsigned int component = 0) const override; /*--- Damping profile evaluation ---*/

    virtual void vector_value(const Point<dim>& p,
                              Vector<double>&   values) const override; /*--- Damping profile vector evaluation for the velocity ---*/

  private:
    const double x_start; /*--- Starting coordinate of the damping layer ---*/
    const double x_max;   /*--- Ending coordinate of the damping layer ---*/
  };

  // Class constructor, which simply calls the parent class constructor
  // and then initialize some data
  //
  template<int dim, unsigned int n_comp>
  Rayleigh_Aux_Right<dim, n_comp>::Rayleigh_Aux_Right(const double initial_time): Function<dim>(n_comp, initial_time),
                                                                                  x_start(EquationData::x_start_right/EquationData::L_ref),
                                                                                  x_max(EquationData::x_max/EquationData::L_ref) {}

  // Evaluation of Rayleigh damping profile
  //
  template<int dim, unsigned int n_comp>
  double Rayleigh_Aux_Right<dim, n_comp>::value(const Point<dim>& p, const unsigned int component) const {
    (void)component;
    AssertIndexRange(component, n_comp);

    if(p[0] < x_start) {
      return 1.0;
    }

    return 1.0/(1.0 + 1.2*std::sin(0.5*numbers::PI*(p[0] - x_start)/(x_max - x_start))*
                          std::sin(0.5*numbers::PI*(p[0] - x_start)/(x_max - x_start)));
  }

  // We need a vector value instance to deal with the velocity or, more in general,
  // if n_comp > 1.
  //
  template<int dim, unsigned int n_comp>
  void Rayleigh_Aux_Right<dim, n_comp>::vector_value(const Point<dim>& p, Vector<double>& values) const {
    Assert(values.size() == n_comp, ExcDimensionMismatch(values.size(), dim));

    for(unsigned int i = 0; i < n_comp; ++i) {
      values[i] = value(p, i);
    }
  }


  /* We do the same for the Rayleigh damping profile along the left lateral boundary
  */
  template<int dim, unsigned int n_comp>
  class Rayleigh_Left: public Function<dim> {
  public:
    Rayleigh_Left(const double initial_time = 0.0); /*--- Class constructor ---*/

    virtual double value(const Point<dim>&  p,
                         const unsigned int component = 0) const override; /*--- Damping profile evaluation ---*/

    virtual void vector_value(const Point<dim>& p,
                              Vector<double>&   values) const override; /*--- Damping profile vector evaluation for the velocity ---*/

  private:
    const double x_start; /*--- Starting coordinate of the damping layer ---*/
    const double x_min;   /*--- Ending coordinate of the damping layer ---*/
  };

  // Class constructor, which simply calls the parent class constructor
  // and then initialize some data
  //
  template<int dim, unsigned int n_comp>
  Rayleigh_Left<dim, n_comp>::Rayleigh_Left(const double initial_time): Function<dim>(n_comp, initial_time),
                                                                        x_start(EquationData::x_start_left/EquationData::L_ref),
                                                                        x_min(0.0) {}

  // Evaluation of Rayleigh damping profile
  //
  template<int dim, unsigned int n_comp>
  double Rayleigh_Left<dim, n_comp>::value(const Point<dim>& p, const unsigned int component) const {
    (void)component;
    AssertIndexRange(component, n_comp);

    if(p[0] > x_start) {
      return 0.0;
    }

    return 1.2*std::sin(0.5*numbers::PI*(p[0] - x_start)/(x_min - x_start))*
               std::sin(0.5*numbers::PI*(p[0] - x_start)/(x_min - x_start));
  }

  // We need a vector value instance to deal with the velocity or, more in general,
  // if n_comp > 1.
  //
  template<int dim, unsigned int n_comp>
  void Rayleigh_Left<dim, n_comp>::vector_value(const Point<dim>& p, Vector<double>& values) const {
    Assert(values.size() == n_comp, ExcDimensionMismatch(values.size(), dim));

    for(unsigned int i = 0; i < n_comp; ++i) {
      values[i] = value(p, i);
    }
  }


  /* We create an auxiliary class for the term (1/(1 + dt*tau)) in order to avoid loop.
  */
  template<int dim, unsigned int n_comp>
  class Rayleigh_Aux_Left: public Function<dim> {
  public:
    Rayleigh_Aux_Left(const double initial_time = 0.0); /*--- Class constructor ---*/

    virtual double value(const Point<dim>&  p,
                         const unsigned int component = 0) const override; /*--- Damping profile evaluation ---*/

    virtual void vector_value(const Point<dim>& p,
                              Vector<double>&   values) const override; /*--- Damping profile vector evaluation for the velocity ---*/

  private:
    const double x_start; /*--- Starting coordinate of the damping layer ---*/
    const double x_min;   /*--- Ending coordinate of the damping layer ---*/
  };

  // Class constructor, which simply calls the parent class constructor
  // and then initialize some data
  //
  template<int dim, unsigned int n_comp>
  Rayleigh_Aux_Left<dim, n_comp>::Rayleigh_Aux_Left(const double initial_time): Function<dim>(n_comp, initial_time),
                                                                                x_start(EquationData::x_start_left/EquationData::L_ref),
                                                                                x_min(0.0) {}

  // Evaluation of Rayleigh damping profile
  //
  template<int dim, unsigned int n_comp>
  double Rayleigh_Aux_Left<dim, n_comp>::value(const Point<dim>& p, const unsigned int component) const {
    (void)component;
    AssertIndexRange(component, n_comp);

    if(p[0] > x_start) {
      return 1.0;
    }

    return 1.0/(1.0 + 1.2*std::sin(0.5*numbers::PI*(p[0] - x_start)/(x_min - x_start))*
                          std::sin(0.5*numbers::PI*(p[0] - x_start)/(x_min - x_start)));
  }

  // We need a vector value instance to deal with the velocity or, more in general,
  // if n_comp > 1.
  //
  template<int dim, unsigned int n_comp>
  void Rayleigh_Aux_Left<dim, n_comp>::vector_value(const Point<dim>& p, Vector<double>& values) const {
    Assert(values.size() == n_comp, ExcDimensionMismatch(values.size(), dim));

    for(unsigned int i = 0; i < n_comp; ++i) {
      values[i] = value(p, i);
    }
  }


  /* We do the same for the Rayleigh damping profile along the right y lateral boundary.
  */
  template<int dim, unsigned int n_comp>
  class Rayleigh_RightY: public Function<dim> {
  public:
    Rayleigh_RightY(const double initial_time = 0.0); /*--- Class constructor ---*/

    virtual double value(const Point<dim>&  p,
                         const unsigned int component = 0) const override; /*--- Damping profile evaluation ---*/

    virtual void vector_value(const Point<dim>& p,
                              Vector<double>&   values) const override; /*--- Damping profile vector evaluation for the velocity ---*/

  private:
    const double y_start; /*--- Starting coordinate of the damping layer ---*/
    const double y_max;   /*--- Ending coordinate of the damping layer ---*/
  };

  // Class constructor, which simply calls the parent class constructor
  // and then initialize some data
  //
  template<int dim, unsigned int n_comp>
  Rayleigh_RightY<dim, n_comp>::Rayleigh_RightY(const double initial_time): Function<dim>(n_comp, initial_time),
                                                                            y_start(EquationData::y_start_right/EquationData::L_ref),
                                                                            y_max(EquationData::y_max/EquationData::L_ref) {}

  // Evaluation of Rayleigh damping profile
  //
  template<int dim, unsigned int n_comp>
  double Rayleigh_RightY<dim, n_comp>::value(const Point<dim>& p, const unsigned int component) const {
    (void)component;
    AssertIndexRange(component, n_comp);

    if(p[1] < y_start) {
      return 0.0;
    }

    return 1.2*std::sin(0.5*numbers::PI*(p[1] - y_start)/(y_max - y_start))*
               std::sin(0.5*numbers::PI*(p[1] - y_start)/(y_max - y_start));
  }

  // We need a vector value instance to deal with the velocity or, more in general,
  // if n_comp > 1.
  //
  template<int dim, unsigned int n_comp>
  void Rayleigh_RightY<dim, n_comp>::vector_value(const Point<dim>& p, Vector<double>& values) const {
    Assert(values.size() == n_comp, ExcDimensionMismatch(values.size(), dim));
    for(unsigned int i = 0; i < n_comp; ++i)
      values[i] = value(p, i);
  }


  /* We create an auxiliary class for the term (1/(1 + dt*tau)) in order to avoid loop.
  */
  template<int dim, unsigned int n_comp>
  class Rayleigh_Aux_RightY: public Function<dim> {
  public:
    Rayleigh_Aux_RightY(const double initial_time = 0.0); /*--- Class constructor ---*/

    virtual double value(const Point<dim>&  p,
                         const unsigned int component = 0) const override; /*--- Damping profile evaluation ---*/

    virtual void vector_value(const Point<dim>& p,
                              Vector<double>&   values) const override; /*--- Damping profile vector evaluation for the velocity ---*/

  private:
    const double y_start; /*--- Starting coordinate of the damping layer ---*/
    const double y_max;   /*--- Ending coordinate of the damping layer ---*/
  };

  // Class constructor, which simply calls the parent class constructor
  // and then initialize some data
  //
  template<int dim, unsigned int n_comp>
  Rayleigh_Aux_RightY<dim, n_comp>::Rayleigh_Aux_RightY(const double initial_time): Function<dim>(n_comp, initial_time),
                                                                                    y_start(EquationData::y_start_right/EquationData::L_ref),
                                                                                    y_max(EquationData::y_max/EquationData::L_ref) {}

  // Evaluation of Rayleigh damping profile
  //
  template<int dim, unsigned int n_comp>
  double Rayleigh_Aux_RightY<dim, n_comp>::value(const Point<dim>& p, const unsigned int component) const {
    (void)component;
    AssertIndexRange(component, n_comp);

    if(p[1] < y_start) {
      return 1.0;
    }

    return 1.0/(1.0 + 1.2*std::sin(0.5*numbers::PI*(p[1] - y_start)/(y_max - y_start))*
                          std::sin(0.5*numbers::PI*(p[1] - y_start)/(y_max - y_start)));
  }

  // We need a vector value instance to deal with the velocity or, more in general,
  // if n_comp > 1.
  //
  template<int dim, unsigned int n_comp>
  void Rayleigh_Aux_RightY<dim, n_comp>::vector_value(const Point<dim>& p, Vector<double>& values) const {
    Assert(values.size() == n_comp, ExcDimensionMismatch(values.size(), dim));

    for(unsigned int i = 0; i < n_comp; ++i) {
      values[i] = value(p, i);
    }
  }


  /* We do the same for the Rayleigh damping profile along the left y lateral boundary
  */
  template<int dim, unsigned int n_comp>
  class Rayleigh_LeftY: public Function<dim> {
  public:
    Rayleigh_LeftY(const double initial_time = 0.0); /*--- Class constructor ---*/

    virtual double value(const Point<dim>&  p,
                         const unsigned int component = 0) const override; /*--- Damping profile evaluation ---*/

    virtual void vector_value(const Point<dim>& p,
                              Vector<double>&   values) const override; /*--- Damping profile vector evaluation for the velocity ---*/

  private:
    const double y_start; /*--- Starting coordinate of the damping layer ---*/
    const double y_min;   /*--- Ending coordinate of the damping layer ---*/
  };

  // Class constructor, which simply calls the parent class constructor
  // and then initialize some data
  //
  template<int dim, unsigned int n_comp>
  Rayleigh_LeftY<dim, n_comp>::Rayleigh_LeftY(const double initial_time): Function<dim>(n_comp, initial_time),
                                                                          y_start(EquationData::y_start_left/EquationData::L_ref),
                                                                          y_min(0.0) {}

  // Evaluation of Rayleigh damping profile
  //
  template<int dim, unsigned int n_comp>
  double Rayleigh_LeftY<dim, n_comp>::value(const Point<dim>& p, const unsigned int component) const {
    (void)component;
    AssertIndexRange(component, n_comp);

    if(p[1] > y_start) {
      return 0.0;
    }

    return 1.2*std::sin(0.5*numbers::PI*(p[1] - y_start)/(y_min - y_start))*
               std::sin(0.5*numbers::PI*(p[1] - y_start)/(y_min - y_start));
  }

  // We need a vector value instance to deal with the velocity or, more in general,
  // if n_comp > 1.
  //
  template<int dim, unsigned int n_comp>
  void Rayleigh_LeftY<dim, n_comp>::vector_value(const Point<dim>& p, Vector<double>& values) const {
    Assert(values.size() == n_comp, ExcDimensionMismatch(values.size(), dim));

    for(unsigned int i = 0; i < n_comp; ++i) {
      values[i] = value(p, i);
    }
  }


  /* We create an auxiliary class for the term (1/(1 + dt*tau)) in order to avoid loop.
  */
  template<int dim, unsigned int n_comp>
  class Rayleigh_Aux_LeftY: public Function<dim> {
  public:
    Rayleigh_Aux_LeftY(const double initial_time = 0.0); /*--- Class constructor ---*/

    virtual double value(const Point<dim>&  p,
                         const unsigned int component = 0) const override; /*--- Damping profile evaluation ---*/

    virtual void vector_value(const Point<dim>& p,
                              Vector<double>&   values) const override; /*--- Damping profile vector evaluation for the velocity ---*/

  private:
    const double y_start; /*--- Starting coordinate of the damping layer ---*/
    const double y_min;   /*--- Ending coordinate of the damping layer ---*/
  };

  // Class constructor, which simply calls the parent class constructor
  // and then initialize some data
  //
  template<int dim, unsigned int n_comp>
  Rayleigh_Aux_LeftY<dim, n_comp>::Rayleigh_Aux_LeftY(const double initial_time): Function<dim>(n_comp, initial_time),
                                                                                  y_start(EquationData::y_start_left/EquationData::L_ref),
                                                                                  y_min(0.0) {}

  // Evaluation of Rayleigh damping profile
  //
  template<int dim, unsigned int n_comp>
  double Rayleigh_Aux_LeftY<dim, n_comp>::value(const Point<dim>& p, const unsigned int component) const {
    (void)component;
    AssertIndexRange(component, n_comp);

    if(p[1] > y_start) {
      return 1.0;
    }

    return 1.0/(1.0 + 1.2*std::sin(0.5*numbers::PI*(p[1] - y_start)/(y_min - y_start))*
                          std::sin(0.5*numbers::PI*(p[1] - y_start)/(y_min - y_start)));
  }

  // We need a vector value instance to deal with the velocity or, more in general,
  // if n_comp > 1.
  //
  template<int dim, unsigned int n_comp>
  void Rayleigh_Aux_LeftY<dim, n_comp>::vector_value(const Point<dim>& p, Vector<double>& values) const {
    Assert(values.size() == n_comp, ExcDimensionMismatch(values.size(), dim));

    for(unsigned int i = 0; i < n_comp; ++i) {
      values[i] = value(p, i);
    }
  }


  /* Now we can focus on mappings from reference element to the physical one
     using the Gal-Chen. Notice that lenghts are in kilometers because of
     the non-dimensional version (the characteristic length is assumed 1 km).
  */
  template <int dim>
  class PushForward : public Function<dim> {
  public:
    PushForward() : Function<dim>(dim, 0.0), z_max(EquationData::z_max/EquationData::L_ref) {}

    virtual ~PushForward() {};

    virtual double value(const Point<dim>& p, const unsigned int component = 0) const;

  private:
    const double z_max;
  };

  // Mapping from reference to physical
  //
  template <int dim>
  double PushForward<dim>::value(const Point<dim>& p, const unsigned int component) const {
    // x component
    if(component == 0) {
      return p[0];
    }
    // y component
    else if(component == 1) {
      return p[1];
    }
    // z component
    else if(component == 2) {
      double hX = EquationData::h/std::pow(1.0 +
                                           (p[0]*EquationData::L_ref - EquationData::xc)/EquationData::ac*
                                           (p[0]*EquationData::L_ref - EquationData::xc)/EquationData::ac +
                                           (p[1]*EquationData::L_ref - EquationData::yc)/EquationData::ac*
                                           (p[1]*EquationData::L_ref - EquationData::yc)/EquationData::ac, 1.5);
      hX /= EquationData::L_ref;

      return p[2] + ((z_max - p[2])/z_max)*hX;
    }
  }


  /* We compute now the inverse mapping (from physical to reference).
     Notice that lenghts are in kilometers becasue of the non-dimensional version.
  */
  template <int dim>
  class PullBack : public Function<dim> {
  public:
    PullBack() : Function<dim>(dim, 0.0), z_max(EquationData::z_max/EquationData::L_ref) {}

    virtual ~PullBack() {};

    virtual double value(const Point<dim>& p, const unsigned int component = 0) const;

  private:
    const double z_max;
  };

  // Mapping from physical to reference
  //
  template <int dim>
  double PullBack<dim>::value(const Point<dim>& p, const unsigned int component) const {
    // x component
    if(component == 0) {
      return p[0];
    }
    // y component
    else if(component == 1) {
      return p[1];
    }
    // z component
    else if(component == 2) {
      double hx = EquationData::h/std::pow(1.0 +
                                           (p[0]*EquationData::L_ref - EquationData::xc)/EquationData::ac*
                                           (p[0]*EquationData::L_ref - EquationData::xc)/EquationData::ac +
                                           (p[1]*EquationData::L_ref - EquationData::yc)/EquationData::ac*
                                           (p[1]*EquationData::L_ref - EquationData::yc)/EquationData::ac, 1.5);
      hx /= EquationData::L_ref;

      return z_max*(p[2] - hx)/(z_max - hx);
    }
  }

} // namespace EquationData
