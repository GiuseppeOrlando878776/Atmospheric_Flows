/*--- Author: Giuseppe Orlando, 2025. ---*/
#pragma once

// @sect{Include files}

// We start by including the necessary header file
//
#include "../equation_data.h"

#include <type_traits>

// @sect{Physical flux}

// In this namespace, we implement signature functions for the numerical flux
//
namespace Physics {
  using namespace dealii;

  /**
   * We declare now the class for a the physical flux for the Euler equations
   */
  template<unsigned dim, typename Number>
  class PhysicalFluxEuler {
  public:
    using value_type = typename std::conditional<std::is_floating_point<Number>::value,
                                                 Number,
                                                 typename Number::value_type>::type; /*--- Define the arythmetic type for this class ---*/

    PhysicalFluxEuler(); /*--- Default class constructor ---*/

    PhysicalFluxEuler(const value_type Ma_); /*--- Class constructor ---*/

    inline DEAL_II_ALWAYS_INLINE
    value_type get_Mach() const;

    inline DEAL_II_ALWAYS_INLINE
    Tensor<1, dim, Number> physical_flux_continuity(const Number& rho,
                                                    const Tensor<1, dim, Number>& u) const; /*--- Physical flux continuity equation ---*/

    Tensor<2, dim, Number> physical_flux_momentum(const Number& rho,
                                                  const Tensor<1, dim, Number>& u,
                                                  const Number& p) const; /*--- Physical flux momentum equation ---*/

    Tensor<1, dim, Number> physical_flux_energy(const Number& rho,
                                                const Tensor<1, dim, Number>& u,
                                                const Number& p) const; /*--- Physical flux energy equation ---*/

  protected:
    const value_type Ma; /*--- Mach number ---*/
  };

  // Default class constructor
  //
  template<unsigned dim, typename Number>
  PhysicalFluxEuler<dim, Number>::PhysicalFluxEuler():
    Ma() {}

  // Class constructor
  //
  template<unsigned dim, typename Number>
  PhysicalFluxEuler<dim, Number>::PhysicalFluxEuler(const value_type Ma_):
    Ma(Ma_) {}

  // Getter of the Mach number
  //
  template<unsigned dim, typename Number>
  inline DEAL_II_ALWAYS_INLINE
  typename PhysicalFluxEuler<dim, Number>::value_type
  PhysicalFluxEuler<dim, Number>::get_Mach() const {
    return Ma;
  }

  // Physical flux of the continuity equation
  //
  template<unsigned dim, typename Number>
  inline DEAL_II_ALWAYS_INLINE
  Tensor<1, dim, Number> PhysicalFluxEuler<dim, Number>::
                         physical_flux_continuity(const Number& rho,
                                                  const Tensor<1, dim, Number>& u) const {
    return rho*u;
  }

  // Physical flux of the momentum equation
  //
  template<unsigned dim, typename Number>
  Tensor<2, dim, Number> PhysicalFluxEuler<dim, Number>::
                         physical_flux_momentum(const Number& rho,
                                                const Tensor<1, dim, Number>& u,
                                                const Number& p) const {

    Tensor<2, dim, Number> identity;
    for(unsigned d = 0; d < dim; ++d) {
      identity[d][d] = Number(1.0);
    }

    /*--- Return the momentum flux ---*/
    return outer_product(rho*u, u) +
           static_cast<value_type>(1.0)/(Ma*Ma)*(p*identity);
  }

  // Physical flux of the energy equation
  //
  template<unsigned dim, typename Number>
  Tensor<1, dim, Number> PhysicalFluxEuler<dim, Number>::
                         physical_flux_energy(const Number& rho,
                                              const Tensor<1, dim, Number>& u,
                                              const Number& p) const {
    /*--- Compute internal enrgy ---*/
    const auto& e = static_cast<value_type>(1.0)/
                    (static_cast<value_type>(EquationData::Cp_Cv) - static_cast<value_type>(1.0))*
                    (p/rho);

    /*--- Compute kinetic energy ---*/
    const auto& k = static_cast<value_type>(0.5)*scalar_product(u, u);

    /*--- Return the energy flux ---*/
    return ((rho*e + p) + (Ma*Ma)*(rho*k))*u;
  }

} // namespace PhysicalFlux
