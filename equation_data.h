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

  static const double Cp_Cv = 1.4; /*--- Specific heats ratio ---*/


  // We do the start with the density (since it is a scalar field) we can derive
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

    return 1.0;
  }

  // We declare now the class that describes the initial condition for the momentum.
  //
  template<int dim>
  class Momentum: public Function<dim> {
  public:
    Momentum(const double initial_time = 0.0);

    virtual double value(const Point<dim>&  p,
                         const unsigned int component = 0) const override;

    virtual void vector_value(const Point<dim>& p,
                              Vector<double>&   values) const override;
  };


  // Constructor which simply relies on the 'Function' constructor.
  //
  template<int dim>
  Momentum<dim>::Momentum(const double initial_time): Function<dim>(dim, initial_time) {}

  // Specify the value for each spatial component. This function is overriden.
  //
  template<int dim>
  double Momentum<dim>::value(const Point<dim>& p, const unsigned int component) const {
    AssertIndexRange(component, dim);

    return 0.0;
  }

  // Put together for a vector evalutation of the momentum.
  //
  template<int dim>
  void Momentum<dim>::vector_value(const Point<dim>& p, Vector<double>& values) const {
    Assert(values.size() == dim, ExcDimensionMismatch(values.size(), dim));

    for(unsigned int i = 0; i < dim; ++i) {
      values[i] = value(p, i);
    }
  }


  // We do the same for the energy (since it is a scalar field) we can derive
  // directly from the deal.II built-in class Function.
  //
  template<int dim>
  class Energy: public Function<dim> {
  public:
    Energy(const double initial_time = 0.0);

    virtual double value(const Point<dim>&  p,
                         const unsigned int component = 0) const override;
  };


  // Constructor which again relies on the 'Function' constructor.
  //
  template<int dim>
  Energy<dim>::Energy(const double initial_time): Function<dim>(1, initial_time) {}

  // Evaluation depending on the spatial coordinates. The input argument 'component'
  // will be unused but it has to be kept to override
  //
  template<int dim>
  double Energy<dim>::value(const Point<dim>& p, const unsigned int component) const {
    (void)component;
    AssertIndexRange(component, 1);

    return 1.0/(EquationData::Cp_Cv - 1.0);
  }

} // namespace EquationData
