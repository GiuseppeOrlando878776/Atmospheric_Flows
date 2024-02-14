/* This file is part of the Atmospheric_Flows/tree/Mesoscale repository and subject to the
   LGPL license. See the LICENSE file in the top level directory of this
   project for details. */

/*--- Author: Giuseppe Orlando, 2023. ---*/

// @sect{Include files}

// We start by including the necessary deal.II header files and some C++
// related ones.
//
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

  /*--- Polynomial degrees. We typically consider the same polynomial degree for all the variables ---*/
  static const unsigned int degree_T   = 4;
  static const unsigned int degree_rho = 4;
  static const unsigned int degree_u   = 4;

  static const double Cp_Cv = 1.4;   /*--- Specific heats ratio ---*/
  static const double R     = 287.0; /*--- Specific gas constant ---*/

  static const double g = 9.81; /*--- Acceleration of gravity ---*/
  static const double N = 0.02; /*--- Buyoancy frequency ---*/

  static const double h  = 450.0;   /*--- Hill height ---*/
  static const double xc = 20000.0; /*--- Center of the hill ---*/
  static const double ac = 1000.0;  /*--- Width of the hill ---*/

  static const double x_max = 40000.0; /*--- Extension along horizontal direction ---*/
  static const double z_max = 20000.0; /*--- Extension along vertical direction ---*/

  static const double z_start       = 9000.0;  /*--- Start of Rayleigh damping for top boundary ---*/
  static const double x_start_left  = 10000.0; /*--- Start of Rayleigh damping for left boundary ---*/
  static const double x_start_right = 30000.0; /*--- Start of Rayleigh damping for right boundary ---*/

  static const double L_ref   = 1000.0;          /*--- Reference length ---*/
  static const double u_ref   = 13.28;           /*--- Reference velocity ---*/
  static const double p_ref   = 100000.0;        /*--- Reference pressure ---*/
  static const double T_ref   = 273.0;
  static const double rho_ref = p_ref/(R*T_ref); /*--- Reference density ---*/

  static const unsigned int degree_mapping          = 2;                                                             /*--- Mapping degree ---*/
  static const unsigned int extra_quadrature_degree = (degree_mapping == 1) ? 0 : my_ceil(0.5*(degree_mapping - 2)); /*--- Extra accuracy
                                                                                                                           for quadratures ---*/

  // We declare now the class that describes the initial condition for the velocity.
  //
  template<int dim>
  class Velocity: public Function<dim> {
  public:
    Velocity(const double initial_time = 0.0); /*--- Class constructor ---*/

    virtual double value(const Point<dim>&  p,
                         const unsigned int component = 0) const override; /*--- Point value for a single component ---*/

    virtual void vector_value(const Point<dim>& p,
                              Vector<double>&   values) const override; /*--- Point value for the whole vector ---*/
  };

  // Constructor which simply relies on the 'Function' constructor.
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


  // We do the same for the pressure (since it is a scalar field) we can derive
  // directly from the deal.II built-in class Function. Notice that in order to
  // get a dimensional version one should multiply the result by p_ref.
  //
  template<int dim>
  class Pressure: public Function<dim> {
  public:
    Pressure(const double initial_time = 0.0); /*--- Class constructor ---*/

    virtual double value(const Point<dim>&  p,
                         const unsigned int component = 0) const override; /*--- Point value ---*/
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
                                (1.0 - std::exp(-EquationData::N*EquationData::N/EquationData::g*p[1]*EquationData::L_ref));

    return std::pow(pi_bar, 1.0/Gamma);
  }


  // We do the same for the density (since it is a scalar field) we can derive
  // directly from the deal.II built-in class Function. Notice that in order to
  // get a dimensional version one should multiply the result by rho_ref.
  //
  template<int dim>
  class Density: public Function<dim> {
  public:
    Density(const double initial_time = 0.0); /*--- Class constructor ---*/

    virtual double value(const Point<dim>&  p,
                         const unsigned int component = 0) const override; /*--- Point value ---*/
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
                             (1.0 - std::exp(-EquationData::N*EquationData::N/EquationData::g*p[1]*EquationData::L_ref));

    const double theta_bar = EquationData::T_ref*std::exp(EquationData::N*EquationData::N/EquationData::g*p[1]*EquationData::L_ref);

    return EquationData::T_ref/theta_bar*std::pow(pi_bar, 1.0/(EquationData::Cp_Cv - 1.0));
  }


  // We focus now on the Rayleigh damping profile along the vertical direction.
  // We create a suitable function for that. This function will be either scalar
  // or vectorial (for the velocity). That's why the auxiliary template parameter
  // n_comp is present.
  //
  template<int dim, unsigned int n_comp>
  class Rayleigh: public Function<dim> {
  public:
    Rayleigh(const double initial_time = 0.0); /*--- Class constructor ---*/

    virtual double value(const Point<dim>&  p,
                         const unsigned int component = 0) const override; /*--- Point value for each component ---*/

    virtual void vector_value(const Point<dim>& p,
                              Vector<double>&   values) const override; /*--- Point value for the whole vector ---*/

  private:
    const double z_start; /*--- Starting coordinate of the damping layer ---*/
    const double z_max;   /*--- Ending coordinate of the damping layer ---*/
  };

  // Constructor which again relies on the 'Function' constructor.
  //
  template<int dim, unsigned int n_comp>
  Rayleigh<dim, n_comp>::Rayleigh(const double initial_time): Function<dim>(n_comp, initial_time),
                                                              z_start(EquationData::z_start/EquationData::L_ref),
                                                              z_max(EquationData::z_max/EquationData::L_ref) {}

  // Evaluation depending on the spatial coordinates. The input argument 'component'
  // will be unused but it has to be kept to override
  //
  template<int dim, unsigned int n_comp>
  double Rayleigh<dim, n_comp>::value(const Point<dim>& p, const unsigned int component) const {
    (void)component;
    AssertIndexRange(component, n_comp);

    if(p[1] < z_start) {
      return 0.0;
    }

    return 0.15*std::sin(0.5*numbers::PI*(p[1] - z_start)/(z_max - z_start))*
                std::sin(0.5*numbers::PI*(p[1] - z_start)/(z_max - z_start)); /*--- Rayleigh profile expression ---*/
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


  // We create an auxiliary class for the term (1/(1 + dt*tau)) in order to avoid loop.
  // The template parameter n_comp has the same meaning of the previous class.
  //
  template<int dim, unsigned int n_comp>
  class Rayleigh_Aux: public Function<dim> {
  public:
    Rayleigh_Aux(const double initial_time = 0.0); /*--- Class constructor ---*/

    virtual double value(const Point<dim>&  p,
                         const unsigned int component = 0) const override; /*--- Point value ---*/

    virtual void vector_value(const Point<dim>& p,
                              Vector<double>&   values) const override; /*--- Point value for the whole vector ---*/

  private:
    const double z_start; /*--- Starting coordinate of the damping layer ---*/
    const double z_max;   /*--- Ending coordinate of the damping layer ---*/
  };

  // Constructor which again relies on the 'Function' constructor.
  //
  template<int dim, unsigned int n_comp>
  Rayleigh_Aux<dim, n_comp>::Rayleigh_Aux(const double initial_time): Function<dim>(n_comp, initial_time),
                                                                      z_start(EquationData::z_start/EquationData::L_ref),
                                                                      z_max(EquationData::z_max/EquationData::L_ref) {}

  // Evaluation depending on the spatial coordinates. The input argument 'component'
  // will be unused but it has to be kept to override
  //
  template<int dim, unsigned int n_comp>
  double Rayleigh_Aux<dim, n_comp>::value(const Point<dim>& p, const unsigned int component) const {
    (void)component;
    AssertIndexRange(component, n_comp);

    if(p[1] < z_start) {
      return 1.0;
    }

    return 1.0/(1.0 + 0.15*std::sin(0.5*numbers::PI*(p[1] - z_start)/(z_max - z_start))*
                           std::sin(0.5*numbers::PI*(p[1] - z_start)/(z_max - z_start)));
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


  // We do the same for the Rayleigh damping profile along the right lateral boundary.
  //
  template<int dim, unsigned int n_comp>
  class Rayleigh_Right: public Function<dim> {
  public:
    Rayleigh_Right(const double initial_time = 0.0); /*--- Class constructor ---*/

    virtual double value(const Point<dim>&  p,
                         const unsigned int component = 0) const override; /*--- Point value ---*/

    virtual void vector_value(const Point<dim>& p,
                              Vector<double>&   values) const override; /*--- Point value for the whole vector ---*/

  private:
    const double x_start; /*--- Starting coordinate of the damping layer ---*/
    const double x_max;   /*--- Ending coordinate of the damping layer ---*/
  };

  // Constructor which again relies on the 'Function' constructor.
  //
  template<int dim, unsigned int n_comp>
  Rayleigh_Right<dim, n_comp>::Rayleigh_Right(const double initial_time): Function<dim>(n_comp, initial_time),
                                                                          x_start(EquationData::x_start_right/EquationData::L_ref),
                                                                          x_max(EquationData::x_max/EquationData::L_ref) {}

  // Evaluation depending on the spatial coordinates. The input argument 'component'
  // will be unused but it has to be kept to override
  //
  template<int dim, unsigned int n_comp>
  double Rayleigh_Right<dim, n_comp>::value(const Point<dim>& p, const unsigned int component) const {
    (void)component;
    AssertIndexRange(component, n_comp);

    if(p[0] < x_start) {
      return 0.0;
    }

    return 0.15*std::sin(0.5*numbers::PI*(p[0] - x_start)/(x_max - x_start))*
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


  // We create an auxiliary class for the term (1/(1 + dt*tau)) in order to avoid loop.
  //
  template<int dim, unsigned int n_comp>
  class Rayleigh_Aux_Right: public Function<dim> {
  public:
    Rayleigh_Aux_Right(const double initial_time = 0.0); /*--- Class constructor ---*/

    virtual double value(const Point<dim>&  p,
                         const unsigned int component = 0) const override; /*--- Point value ---*/

    virtual void vector_value(const Point<dim>& p,
                              Vector<double>&   values) const override; /*--- Point value for the whole vector ---*/

  private:
    const double x_start; /*--- Starting coordinate of the damping layer ---*/
    const double x_max;   /*--- Ending coordinate of the damping layer ---*/
  };

  // Constructor which again relies on the 'Function' constructor.
  //
  template<int dim, unsigned int n_comp>
  Rayleigh_Aux_Right<dim, n_comp>::Rayleigh_Aux_Right(const double initial_time): Function<dim>(n_comp, initial_time),
                                                                                  x_start(EquationData::x_start_right/EquationData::L_ref),
                                                                                  x_max(EquationData::x_max/EquationData::L_ref) {}

  // Evaluation depending on the spatial coordinates. The input argument 'component'
  // will be unused but it has to be kept to override
  //
  template<int dim, unsigned int n_comp>
  double Rayleigh_Aux_Right<dim, n_comp>::value(const Point<dim>& p, const unsigned int component) const {
    (void)component;
    AssertIndexRange(component, n_comp);

    if(p[0] < x_start) {
      return 1.0;
    }

    return 1.0/(1.0 + 0.15*std::sin(0.5*numbers::PI*(p[0] - x_start)/(x_max - x_start))*
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


  // We do the same for the Rayleigh damping profile along the left lateral boundary
  //
  template<int dim, unsigned int n_comp>
  class Rayleigh_Left: public Function<dim> {
  public:
    Rayleigh_Left(const double initial_time = 0.0); /*--- Class constructor ---*/

    virtual double value(const Point<dim>&  p,
                         const unsigned int component = 0) const override; /*--- Point value ---*/

    virtual void vector_value(const Point<dim>& p,
                              Vector<double>&   values) const override; /*--- Point value for the whole vector ---*/

  private:
    const double x_start; /*--- Starting coordinate of the damping layer ---*/
    const double x_min;   /*--- Ending coordinate of the damping layer ---*/
  };

  // Constructor which again relies on the 'Function' constructor.
  //
  template<int dim, unsigned int n_comp>
  Rayleigh_Left<dim, n_comp>::Rayleigh_Left(const double initial_time): Function<dim>(n_comp, initial_time),
                                                                        x_start(EquationData::x_start_left/EquationData::L_ref),
                                                                        x_min(0.0) {}

  // Evaluation depending on the spatial coordinates. The input argument 'component'
  // will be unused but it has to be kept to override
  //
  template<int dim, unsigned int n_comp>
  double Rayleigh_Left<dim, n_comp>::value(const Point<dim>& p, const unsigned int component) const {
    (void)component;
    AssertIndexRange(component, n_comp);

    if(p[0] > x_start) {
      return 0.0;
    }

    return 0.15*std::sin(0.5*numbers::PI*(p[0] - x_start)/(x_min - x_start))*
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


  // We create an auxiliary class for the term (1/(1 + dt*tau)) in order to avoid loop.
  //
  template<int dim, unsigned int n_comp>
  class Rayleigh_Aux_Left: public Function<dim> {
  public:
    Rayleigh_Aux_Left(const double initial_time = 0.0); /*--- Class constructor ---*/

    virtual double value(const Point<dim>&  p,
                         const unsigned int component = 0) const override; /*--- Point value ---*/

    virtual void vector_value(const Point<dim>& p,
                              Vector<double>&   values) const override; /*--- Point value for the whole vector ---*/

  private:
    const double x_start; /*--- Starting coordinate of the damping layer ---*/
    const double x_min;   /*--- Ending coordinate of the damping layer ---*/
  };

  // Constructor which again relies on the 'Function' constructor.
  //
  template<int dim, unsigned int n_comp>
  Rayleigh_Aux_Left<dim, n_comp>::Rayleigh_Aux_Left(const double initial_time): Function<dim>(n_comp, initial_time),
                                                                                x_start(EquationData::x_start_left/EquationData::L_ref),
                                                                                x_min(0.0) {}

  // Evaluation depending on the spatial coordinates. The input argument 'component'
  // will be unused but it has to be kept to override
  //
  template<int dim, unsigned int n_comp>
  double Rayleigh_Aux_Left<dim, n_comp>::value(const Point<dim>& p, const unsigned int component) const {
    (void)component;
    AssertIndexRange(component, n_comp);

    if(p[0] > x_start) {
      return 1.0;
    }

    return 1.0/(1.0 + 0.15*std::sin(0.5*numbers::PI*(p[0] - x_start)/(x_min - x_start))*
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


  // Now we can focus on mappings from reference element to the physical one
  // using the Gal-Chen. Notice that lenghts are in kilometers becasue of
  // the non-dimensional version (the characteristic length is assumed 1 km).
  //
  template <int dim>
  class PushForward : public Function<dim> {
  public:
    /*--- Default constructor ---*/
    PushForward() : Function<dim>(dim, 0.0), z_max(EquationData::z_max/EquationData::L_ref) {}

    virtual ~PushForward() {}; /*--- Default destructor ---*/

    virtual double value(const Point<dim>& p, const unsigned int component = 0) const; /*--- Point evaluation for each component ---*/

  private:
    const double z_max; /*--- Height of the domain ---*/
  };

  // Evaluate for each component
  //
  template <int dim>
  double PushForward<dim>::value(const Point<dim>& p, const unsigned int component) const {
    // x component
    if(component == 0) {
      return p[0];
    }
    // z component
    else if(component == 1) {
      double hX = EquationData::h/(1.0 + (p[0]*EquationData::L_ref - EquationData::xc)/EquationData::ac*
                                         (p[0]*EquationData::L_ref - EquationData::xc)/EquationData::ac);
      hX /= EquationData::L_ref;

      return p[1] + ((z_max - p[1])/z_max)*hX;
    }
  }


  // We compute now the inverse mapping (from physical to reference).
  // Notice that lenghts are in kilometers becasue of the non-dimensional version.
  //
  template <int dim>
  class PullBack : public Function<dim> {
  public:
    /*--- Default constructor ---*/
    PullBack() : Function<dim>(dim, 0.0), z_max(EquationData::z_max/EquationData::L_ref) {}

    virtual ~PullBack() {}; /*--- Default destructor ---*/

    virtual double value(const Point<dim>& p, const unsigned int component = 0) const; /*--- Point evaluation for each component ---*/

  private:
    const double z_max; /*--- Height of the domain ---*/
  };

  // Evaluate for each component
  //
  template <int dim>
  double PullBack<dim>::value(const Point<dim>& p, const unsigned int component) const {
    // x component
    if(component == 0) {
      return p[0];
    }
    // z component
    else if(component == 1) {
      double hx = EquationData::h/(1.0 + (p[0]*EquationData::L_ref - EquationData::xc)/EquationData::ac*
                                         (p[0]*EquationData::L_ref - EquationData::xc)/EquationData::ac);
      hx /= EquationData::L_ref;

      return z_max*(p[1] - hx)/(z_max - hx);
    }
  }

} // namespace EquationData
