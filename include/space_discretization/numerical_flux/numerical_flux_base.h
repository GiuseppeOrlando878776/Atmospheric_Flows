/*--- Author: Giuseppe Orlando, 2025. ---*/
#pragma once

// @sect{Include files}

// We start by including the necessary header file
//
#include "../../physics/physical_flux.h"

// @sect{Numerical flux}

// In this namespace, we implement (virtual) signature functions for the numerical flux
//
namespace NumericalFlux {
  using namespace dealii;

  /**
   * We declare now the class for a generic flux for the Euler equations
   */
  template<unsigned dim, typename Number>
  class NumericalFluxEuler: public Physics::PhysicalFluxEuler<dim, Number> {
  public:
    using value_type = typename Physics::PhysicalFluxEuler<dim, Number>::value_type; /*--- Arithmetic type for this class ---*/

    NumericalFluxEuler() = default;

    NumericalFluxEuler(const value_type Ma_); /*--- Class constructor ---*/

    // Start with the numerical flux for the continuity equation
    virtual Number numerical_flux_continuity(const Number& rho_m,
                                             const Tensor<1, dim, Number>& u_m,
                                             const Number& rho_p,
                                             const Tensor<1, dim, Number>& u_p,
                                             const Tensor<1, dim, Number>& n_minus) const = 0; /*--- Numerical flux continuity equation ---*/

    // Focus now on the functions for the momentum equation
    virtual Tensor<1, dim, Number> numerical_flux_momentum_explicit(const Number& rho_m,
                                                                    const Tensor<1, dim, Number>& u_m,
                                                                    const Number& rho_p,
                                                                    const Tensor<1, dim, Number>& u_p,
                                                                    const Tensor<1, dim, Number>& n_minus) const = 0; /*--- Numerical flux momentum equation
                                                                                                                            for the explicit part ---*/

    virtual Tensor<1, dim, Number> numerical_flux_momentum_implicit(const Number& pres_m,
                                                                    const Number& pres_p,
                                                                    const Tensor<1, dim, Number>& n_minus) const = 0; /*--- Numerical flux momentum equation
                                                                                                                            for the implicit part ---*/

    // Focus now on the functions for the energy equation
    virtual Number numerical_flux_energy_explicit(const Number& rho_m,
                                                  const Tensor<1, dim, Number>& u_m,
                                                  const Number& rho_p,
                                                  const Tensor<1, dim, Number>& u_p,
                                                  const Tensor<1, dim, Number>& n_minus) const = 0; /*--- Numerical flux energy equation
                                                                                                          for the explicit part ---*/

    virtual Number numerical_flux_energy_implicit(const Tensor<1, dim, Number>& u_m,
                                                  const Number& pres_m,
                                                  const Tensor<1, dim, Number>& u_p,
                                                  const Number& pres_p,
                                                  const Tensor<1, dim, Number>& n_minus) const = 0; /*--- Numerical flux energy equation
                                                                                                          for the explicit part ---*/
  };

  // Class constructor
  //
  template<unsigned dim, typename Number>
  NumericalFluxEuler<dim, Number>::NumericalFluxEuler(const value_type Ma_):
    Physics::PhysicalFluxEuler<dim, Number>(Ma_) {}

} // namespace NumericalFlux
