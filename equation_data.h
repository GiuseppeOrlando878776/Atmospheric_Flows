/*--- Author: Giuseppe Orlando, 2023. ---*/

// @sect{Include files}

// We start by including the necessary deal.II header files and some C++
// related ones.
#include <deal.II/base/point.h>
#include <deal.II/base/function.h>

#include <cmath>

// @sect{Equation data}

// In this namespace, we declare the initial conditions,
// the velocity field
//
namespace EquationData {
  using namespace dealii;

  static const unsigned int degree = 2; /*--- Polynomial degree ---*/

  static const unsigned int degree_mapping = 2; /*--- Polynomial degree for mapping ---*/


  // We declare now the class that describes the velocity.
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

    const double U      = radius;

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


  // We do the same for the density (since it is a scalar field) we can derive
  // directly from the deal.II built-in class Function.
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
    const double lambda   = std::atan2(p[1], p[0]); // longitude (atan2 returns range -pi to pi, which is ok for geographic applications)

    const double lambda_c = 1.5*numbers::PI;
    const double theta_c  = 0.0;
    const double r        = radius*std::acos(std::sin(theta_c)*std::sin(theta) +
                                             std::cos(theta_c)*std::cos(theta)*std::cos(lambda - lambda_c));

    const double R        = 1.0/3.0;
    const double value    = 0.5*(1.0 + std::cos(numbers::PI*r/R))*(r < R);

    return value;
  }

} // namespace EquationData
