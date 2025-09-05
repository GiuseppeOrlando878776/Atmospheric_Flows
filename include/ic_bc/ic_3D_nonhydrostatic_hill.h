/*--- Author: Giuseppe Orlando, 2025. ---*/

// @sect{Include files}

// We start by including the necessary deal.II header files and some C++
// related ones
//
#include <deal.II/base/function.h>

#include "../equation_data.h"

#include <cmath>

// @sect{Initial conditions}

// In this namespace, we declare the initial background conditions.
// Some parameters could be read at run-time, but this would be very
// configuration dependent and the parameter file would become unreadable
//
namespace ICBC {
  using namespace dealii;

  /*--- Parameter of the initial conditions for the test case under consideration
        (in this case, the 3D non-hydrostatic hill) ---*/
  static const double N = 0.01; /*--- Buoyancy frequency ---*/

  /**
   * We declare now the class that describes the initial condition for the velocity.
   */
  template<unsigned dim, typename T = double>
  class Velocity: public Function<dim, T> {
  public:
    Velocity(const T initial_time = static_cast<T>(0.0)); /*--- Class constructor ---*/

    virtual T value(const Point<dim, T>& p,
                    const unsigned       component = 0) const override; /*--- Evaluation for each component ---*/

    virtual void vector_value(const Point<dim, T>& p,
                              Vector<T>&           values) const override; /*--- Vector evaluation of the velocity ---*/
  };

  // Constructor which relies on the 'Function' constructor.
  //
  template<unsigned dim, typename T>
  Velocity<dim, T>::Velocity(const T initial_time):
    Function<dim, T>(dim, initial_time) {}

  // Specify the value for each spatial component. This function is overriden.
  //
  template<unsigned dim, typename T>
  T Velocity<dim, T>::value(const Point<dim, T>& p,
                            const unsigned       component) const {
    AssertIndexRange(component, dim);

    if(component == 0) {
      return static_cast<T>(1.0);
    }
    else {
      return static_cast<T>(0.0);
    }
  }

  // Put together for a vector evalutation of the velocity.
  //
  template<unsigned dim, typename T>
  void Velocity<dim, T>::vector_value(const Point<dim, T>& p,
                                      Vector<T>&           values) const {
    Assert(values.size() == dim, ExcDimensionMismatch(values.size(), dim));

    for(unsigned i = 0; i < dim; ++i) {
      values[i] = value(p, i);
    }
  }


  /**
   * We do the same for the pressure. Notice that in order to
     get a dimensional version one should multiply the result by p_ref
   */
  template<unsigned dim, typename T = double>
  class Pressure: public Function<dim, T> {
  public:
    Pressure(const T initial_time = static_cast<T>(0.0)); /*--- Class constructor ---*/

    virtual T value(const Point<dim, T>& p,
                    const unsigned       component = 0) const override; /*--- Evalution of the pressure ---*/
  };

  // Constructor which again relies on the 'Function' constructor.
  //
  template<unsigned dim, typename T>
  Pressure<dim, T>::Pressure(const T initial_time):
    Function<dim, T>(1, initial_time) {}

  // Evaluation depending on the spatial coordinates. The input argument 'component'
  // will be unused but it has to be kept to override
  //
  template<unsigned dim, typename T>
  T Pressure<dim, T>::value(const Point<dim, T>& p,
                            const unsigned       component) const {
    (void)component;
    AssertIndexRange(component, 1);

    const auto Gamma  = (static_cast<T>(EquationData::Cp_Cv) - static_cast<T>(1.0))/
                        static_cast<T>(EquationData::Cp_Cv);

    const auto pi_bar = static_cast<T>(1.0)
                      - static_cast<T>(EquationData::g)*static_cast<T>(EquationData::g)/
                        (static_cast<T>(ICBC::N)*static_cast<T>(ICBC::N))*
                        Gamma*static_cast<T>(EquationData::rho_ref)/static_cast<T>(EquationData::p_ref)*
                        (static_cast<T>(1.0) -
                         std::exp(-static_cast<T>(ICBC::N)*static_cast<T>(ICBC::N)/static_cast<T>(EquationData::g)*
                         p[2]*static_cast<T>(EquationData::L_ref)));

    return std::pow(pi_bar, static_cast<T>(1.0)/Gamma);
  }


  /**
   * We do the same for the density. Notice that in order to
     get a dimensional version one should multiply the result by rho_ref
   */
  template<unsigned dim, typename T = double>
  class Density: public Function<dim, T> {
  public:
    Density(const T initial_time = static_cast<T>(0.0)); /*--- Class constructor ---*/

    virtual T value(const Point<dim, T>& p,
                    const unsigned       component = 0) const override; /*--- Evaluation of the density ---*/
  };

  // Constructor which again relies on the 'Function' constructor.
  //
  template<unsigned dim, typename T>
  Density<dim, T>::Density(const T initial_time):
    Function<dim, T>(1, initial_time) {}

  // Evaluation depending on the spatial coordinates. The input argument 'component'
  // will be unused but it has to be kept to override
  //
  template<unsigned dim, typename T>
  T Density<dim, T>::value(const Point<dim, T>& p,
                           const unsigned       component) const {
    (void)component;
    AssertIndexRange(component, 1);

    const auto Gamma  = (static_cast<T>(EquationData::Cp_Cv) - static_cast<T>(1.0))/
                        static_cast<T>(EquationData::Cp_Cv);

    const auto pi_bar = static_cast<T>(1.0)
                      - static_cast<T>(EquationData::g)*static_cast<T>(EquationData::g)/
                        (static_cast<T>(ICBC::N)*static_cast<T>(ICBC::N))*
                        Gamma*static_cast<T>(EquationData::rho_ref)/static_cast<T>(EquationData::p_ref)*
                        (static_cast<T>(1.0) -
                         std::exp(-static_cast<T>(ICBC::N)*static_cast<T>(ICBC::N)/static_cast<T>(EquationData::g)*
                                  p[2]*static_cast<T>(EquationData::L_ref)));

    const auto theta_bar = static_cast<T>(EquationData::T_ref)*
                           std::exp(static_cast<T>(ICBC::N)*static_cast<T>(ICBC::N)/static_cast<T>(EquationData::g)*
                                    p[2]*static_cast<T>(EquationData::L_ref));

    return static_cast<T>(EquationData::T_ref)/theta_bar*
           std::pow(pi_bar, static_cast<T>(1.0)/
                            (static_cast<T>(EquationData::Cp_Cv) - static_cast<T>(1.0)));
  }

} // namespace EquationData
