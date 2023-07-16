/*--- Author: Giuseppe Orlando, 2023. ---*/

// @sect{Include files}

// We start by including the necessary deal.II header files and some C++
// related ones.
#include <deal.II/base/point.h>
#include <deal.II/base/function.h>

#include <cmath>

// @sect{Equation data}

// In this namespace, we declare the initial background conditions.
//
namespace EquationData {
  using namespace dealii;

  /*--- Polynomial degrees. We typically consider the same polynomial degree for all the variables ---*/
  static const unsigned int degree_T   = 1;
  static const unsigned int degree_rho = 1;
  static const unsigned int degree_u   = 1;

  static const double Cp_Cv = 1.4;   /*--- Specific heats ratio ---*/
  static const double R     = 287.0; /*--- Specific gas constant ---*/

  static const unsigned int degree_mapping = 2;

  // We start with the density (since it is a scalar field) we can derive
  // directly from the deal.II built-in class Function. Notice that in order to
  // get a dimensional version one should multiply the result by rho_ref.
  //
  template<int dim>
  class Density: public Function<dim> {
  public:
    Density(const double initial_time = 0.0);

    virtual double value(const Point<dim>&  p,
                         const unsigned int component = 0) const override;
  };


  template<int dim>
  Density<dim>::Density(const double initial_time): Function<dim>(1, initial_time) {}

  template<int dim>
  double Density<dim>::value(const Point<dim>& p, const unsigned int component) const {
    (void)component;
    AssertIndexRange(component, 1);

    const double radius   = std::sqrt(p[0]*p[0] + p[1]*p[1] + p[2]*p[2]);
    const double theta    = std::asin(p[2]/radius); // latitude (asin returns range -pi/2 to pi/2, which is ok for geographic applications)

    const double T_bar    = 273.0;

    const double a        = 1.0;                     /*--- Radius of the earth ---*/
    const double s        = radius*std::cos(theta); /*--- Radial distance from the polar axis ---*/
    const double u0_tilde = 1.0;
    const double g        = 9.81;
    const double q        = 1.0/(EquationData::R*T_bar)*
                            (0.5*u0_tilde*u0_tilde*s*s - a*g*(1.0 - a/radius));

    return std::exp(q);
  }


  // We declare now the class that describes the initial condition for the velocity.
  //
  template<int dim>
  class Velocity: public Function<dim> {
  public:
    Velocity(const double initial_time = 0.0);

    virtual double value(const Point<dim>&  p,
                         const unsigned int component = 0) const override;

    virtual void vector_value(const Point<dim>& p,
                              Vector<double>&   values) const override;
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

    const double radius = std::sqrt(p[0]*p[0] + p[1]*p[1] + p[2]*p[2]);
    const double theta  = std::asin(p[2]/radius); // latitude (asin returns range -pi/2 to pi/2, which is ok for geographic applications)
    const double lambda = std::atan2(p[1], p[0]); // longitude (atan2 returns range -pi to pi, which is ok for geographic applications)

    const double U      = radius/1.0;

    const double alpha  = 0.0;
    const double u      = U*(std::cos(alpha)*std::cos(theta) +
                             std::sin(theta)*std::cos(lambda)*std::sin(alpha)); //longitudinal
    const double v      = -U*std::sin(alpha)*std::sin(lambda); //latitudinal

    if(component == 0) {
      return -u*std::sin(lambda) -v*std::cos(lambda)*std::sin(theta);
    }
    else if(component == 1) {
      return u*std::cos(lambda) - v*std::sin(lambda)*std::sin(theta);
    }
    else {
      return v*std::cos(theta);
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
    Pressure(const double initial_time = 0.0);

    virtual double value(const Point<dim>&  p,
                         const unsigned int component = 0) const override;
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

    const double radius   = std::sqrt(p[0]*p[0] + p[1]*p[1] + p[2]*p[2]);
    const double theta    = std::asin(p[2]/radius); // latitude (asin returns range -pi/2 to pi/2, which is ok for geographic applications)

    const double T_bar    = 273.0;

    const double a        = 1.0;                     /*--- Radius of the earth ---*/
    const double s        = radius*std::cos(theta); /*--- Radial distance from the polar axis ---*/
    const double u0_tilde = 1.0;
    const double g        = 9.81;
    const double q        = 1.0/(EquationData::R*T_bar)*
                            (0.5*u0_tilde*u0_tilde*s*s - a*g*(1.0 - a/radius));

    return std::exp(q);
  }

} // namespace EquationData
