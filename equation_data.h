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
  static const unsigned int degree_T   = 2;
  static const unsigned int degree_rho = 2;
  static const unsigned int degree_u   = 2;

  static const double Cp_Cv = 1.4;   /*--- Specific heats ratio ---*/
  static const double R     = 287.0; /*--- Specific gas constant ---*/

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

    return 0.0;
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

    /*const double radius   = std::sqrt(p[0]*p[0] + p[1]*p[1] + p[2]*p[2]);
    const double theta    = std::asin(p[2]/radius); // latitude (asin returns range -pi/2 to pi/2, which is ok for geographic applications)
    const double lambda   = std::atan2(p[1], p[0]); // longitude (atan2 returns range -pi to pi, which is ok for geographic applications)

    const double lambda_c = 1.5*numbers::PI;
    const double theta_c  = 0.0;
    const double r        = radius*std::acos(std::sin(theta_c)*std::sin(theta) +
                                             std::cos(theta_c)*std::cos(theta)*std::cos(lambda - lambda_c));

    const double R        = 1.0/3.0;
    const double p_prime  = 0.5*(1.0 + std::cos(numbers::PI*r/R))*(r < R);

    return 1.0 + 0.000*p_prime;*/
    return 1.0;
  }


  // We do the same for the density (since it is a scalar field) we can derive
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

    /*const double radius   = std::sqrt(p[0]*p[0] + p[1]*p[1] + p[2]*p[2]);
    const double theta    = std::asin(p[2]/radius); // latitude (asin returns range -pi/2 to pi/2, which is ok for geographic applications)
    const double lambda   = std::atan2(p[1], p[0]); // longitude (atan2 returns range -pi to pi, which is ok for geographic applications)

    const double lambda_c = 1.5*numbers::PI;
    const double theta_c  = 0.0;
    const double r        = radius*std::acos(std::sin(theta_c)*std::sin(theta) +
                                             std::cos(theta_c)*std::cos(theta)*std::cos(lambda - lambda_c));

    const double R        = 1.0/3.0;
    const double p_prime  = 0.5*(1.0 + std::cos(numbers::PI*r/R))*(r < R);

    const double pres = (1.0 + 0.000*p_prime)*1e5;*/

    const double pres = 1e5;

    const double T = 300.0;

    const double rho = pres/(EquationData::R*T);
    const double rho_ref = 1e5/(EquationData::R*300.0);

    return rho/rho_ref;
  }

} // namespace EquationData
