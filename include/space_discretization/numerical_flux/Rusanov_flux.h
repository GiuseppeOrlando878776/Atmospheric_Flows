/* ------------------------------------------------------------------------
 *
 * SPDX-License-Identifier: LGPL-2.1-or-later
 * Copyright (C) 2022-2026 Giuseppe Orlando
 *
 * This code is free software; you can use it, redistribute it,
 * and/or modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * ------------------------------------------------------------------------
 *
 * Author: Giuseppe Orlando, 2026
 */
#pragma once

// @sect{Include files}

// We start by including the necessary header file
//
#include "numerical_flux_base.h"

// @sect{Numerical flux}

// In this namespace, we implement signature functions for the numerical flux
//
namespace NumericalFlux {
  using namespace dealii;

  /**
   * We declare now the class for a Rusanov (local Lax Friedrichs) flux for the Euler equations
   */
  template<unsigned dim, typename Number>
  class RusanovFluxEuler: public NumericalFluxEuler<dim, Number> {
  public:
    // Arithmetic type for this class
    using value_type = typename Physics::PhysicalFluxEuler<dim, Number>::value_type;

    /**
     * Default class constructor
     */
    RusanovFluxEuler() = default;

    /**
     * Class constructor
     * @param Ma_ Mach number
     */
    RusanovFluxEuler(const value_type Ma_);

    /**
     * Stabilization parameter of the Rusanov flux
     * @param u_m velocity 'interior' side
     * @param u_p velocity 'exterior' side
     * @param n_minus Unit normal from 'interior' to 'exterior'
     */
    inline DEAL_II_ALWAYS_INLINE
    Number compute_lambda(const Tensor<1, dim, Number>& u_m,
                          const Tensor<1, dim, Number>& u_p,
                          const Tensor<1, dim, Number>& n_minus) const;

    /**
     * Numerical flux for the continuity equation
     * @param rho_m density 'interior' side
     * @param u_m velocity 'interior' side
     * @param rho_p density 'exterior' side
     * @param u_p velocity 'exterior' side
     * @param n_minus Unit normal from 'interior' to 'exterior'
     */
    virtual Number numerical_flux_continuity(const Number& rho_m,
                                             const Tensor<1, dim, Number>& u_m,
                                             const Number& rho_p,
                                             const Tensor<1, dim, Number>& u_p,
                                             const Tensor<1, dim, Number>& n_minus) const override;

    /**
     * Numerical flux for the momentum equation (explicit part)
     * @param rho_m density 'interior' side
     * @param u_m velocity 'interior' side
     * @param rho_p density 'exterior' side
     * @param u_p velocity 'exterior' side
     * @param n_minus Unit normal from 'interior' to 'exterior'
     */
    virtual Tensor<1, dim, Number> numerical_flux_momentum_explicit(const Number& rho_m,
                                                                    const Tensor<1, dim, Number>& u_m,
                                                                    const Number& rho_p,
                                                                    const Tensor<1, dim, Number>& u_p,
                                                                    const Tensor<1, dim, Number>& n_minus) const override;

    /**
     * Numerical flux for the momentum equation (implicit part)
     * @param pres_m pressure 'interior' side
     * @param pres_p pressure 'exterior' side
     * @param n_minus Unit normal from 'interior' to 'exterior'
     */
    virtual Tensor<1, dim, Number> numerical_flux_momentum_implicit(const Number& pres_m,
                                                                    const Number& pres_p,
                                                                    const Tensor<1, dim, Number>& n_minus) const override;

    /**
     * Numerical flux for the energy equation (explicit part)
     * @param rho_m density 'interior' side
     * @param u_m velocity 'interior' side
     * @param rho_p density 'exterior' side
     * @param u_p velocity 'exterior' side
     * @param n_minus Unit normal from 'interior' to 'exterior'
     */
    virtual Number numerical_flux_energy_explicit(const Number& rho_m,
                                                  const Tensor<1, dim, Number>& u_m,
                                                  const Number& rho_p,
                                                  const Tensor<1, dim, Number>& u_p,
                                                  const Tensor<1, dim, Number>& n_minus) const override;

    /**
     * Numerical flux for the energy equation (implicit part)
     * @param u_m velocity 'interior' side
     * @param pres_m pressure 'interior' side
     * @param u_p velocity 'exterior' side
     * @param pres_p pressure 'exterior' side
     * @param n_minus Unit normal from 'interior' to 'exterior'
     */
    virtual Number numerical_flux_energy_implicit(const Tensor<1, dim, Number>& u_m,
                                                  const Number& pres_m,
                                                  const Tensor<1, dim, Number>& u_p,
                                                  const Number& pres_p,
                                                  const Tensor<1, dim, Number>& n_minus) const override;

  private:
    Number inv_Gamma; // gamma/(gamma - 1)
  };

  // Class constructor
  //
  template<unsigned dim, typename Number>
  RusanovFluxEuler<dim, Number>::RusanovFluxEuler(const value_type Ma_):
    NumericalFluxEuler<dim, Number>(Ma_),
    inv_Gamma(static_cast<value_type>(EquationData::Cp_Cv)*this->inv_gamma_m1) {}

  // Stabilization parameter of the Rusanov flux
  //
  template<unsigned dim, typename Number>
  inline DEAL_II_ALWAYS_INLINE
  Number RusanovFluxEuler<dim, Number>::
         compute_lambda(const Tensor<1, dim, Number>& u_m,
                        const Tensor<1, dim, Number>& u_p,
                        const Tensor<1, dim, Number>& n_minus) const {
    return std::max(std::abs(scalar_product(u_m, n_minus)),
                    std::abs(scalar_product(u_p, n_minus)));
  }

  // Numerical flux of the continuity equation
  //
  template<unsigned dim, typename Number>
  Number RusanovFluxEuler<dim, Number>::
         numerical_flux_continuity(const Number& rho_m,
                                   const Tensor<1, dim, Number>& u_m,
                                   const Number& rho_p,
                                   const Tensor<1, dim, Number>& u_p,
                                   const Tensor<1, dim, Number>& n_minus) const {
    // Start with centered contribution
    const auto avg_flux = static_cast<value_type>(0.5)*
                          (this->physical_flux_continuity(rho_m, u_m) +
                           this->physical_flux_continuity(rho_p, u_p));

    // Focus on stabilization term
    const auto& lambda   = compute_lambda(u_m, u_p, n_minus);
    const auto& jump_rho = rho_m - rho_p;

    // Return the Rusanov flux
    return scalar_product(avg_flux, n_minus) +
           static_cast<value_type>(0.5)*lambda*jump_rho;
  }

  // Numerical flux explicit part momentum equation
  //
  template<unsigned dim, typename Number>
  Tensor<1, dim, Number> RusanovFluxEuler<dim, Number>::
                         numerical_flux_momentum_explicit(const Number& rho_m,
                                                          const Tensor<1, dim, Number>& u_m,
                                                          const Number& rho_p,
                                                          const Tensor<1, dim, Number>& u_p,
                                                          const Tensor<1, dim, Number>& n_minus) const {
    // Start with centered contribution
    const auto& avg_tensor_product_u = static_cast<value_type>(0.5)*
                                       (outer_product(rho_m*u_m, u_m) +
                                        outer_product(rho_p*u_p, u_p));

    // Focus on stabilization term
    const auto& lambda    = compute_lambda(u_m, u_p, n_minus);
    const auto& jump_rhou = rho_m*u_m - rho_p*u_p;

    // Return the Rusanov flux (explicit part momentum)
    return avg_tensor_product_u*n_minus +
           static_cast<value_type>(0.5)*lambda*jump_rhou;
  }

  // Numerical flux implicit part momentum equation
  //
  template<unsigned dim, typename Number>
  Tensor<1, dim, Number> RusanovFluxEuler<dim, Number>::
                         numerical_flux_momentum_implicit(const Number& pres_m,
                                                          const Number& pres_p,
                                                          const Tensor<1, dim, Number>& n_minus) const {
    return (static_cast<value_type>(0.5)*(pres_m + pres_p))*(this->inv_Ma2)*n_minus;
  }

  // Numerical flux explicit part energy equation
  //
  template<unsigned dim, typename Number>
  Number RusanovFluxEuler<dim, Number>::
         numerical_flux_energy_explicit(const Number& rho_m,
                                        const Tensor<1, dim, Number>& u_m,
                                        const Number& rho_p,
                                        const Tensor<1, dim, Number>& u_p,
                                        const Tensor<1, dim, Number>& n_minus) const {
    // Start with centered contribution
    const auto& avg_kinetic = static_cast<value_type>(0.5)*
                              (static_cast<value_type>(0.5)*scalar_product(u_m, u_m)*rho_m*u_m +
                               static_cast<value_type>(0.5)*scalar_product(u_p, u_p)*rho_p*u_p);

    // Focus on stabilization term
    const auto& lambda       = compute_lambda(u_m, u_p, n_minus);
    const auto& jump_rho_kin = rho_m*(static_cast<value_type>(0.5)*scalar_product(u_m, u_m)) -
                               rho_p*(static_cast<value_type>(0.5)*scalar_product(u_p, u_p));

    // Return the Rusanov flux (explicit part energy)
    return (this->Ma2)*(scalar_product(avg_kinetic, n_minus) +
                        static_cast<value_type>(0.5)*lambda*jump_rho_kin);
  }

  // Numerical flux implicit part energy equation
  //
  template<unsigned dim, typename Number>
  Number RusanovFluxEuler<dim, Number>::
         numerical_flux_energy_implicit(const Tensor<1, dim, Number>& u_m,
                                        const Number& pres_m,
                                        const Tensor<1, dim, Number>& u_p,
                                        const Number& pres_p,
                                        const Tensor<1, dim, Number>& n_minus) const {
    // Start with centered contribution
    const auto& avg_enthalpy = static_cast<value_type>(0.5)*inv_Gamma*
                               (pres_m*u_m + pres_p*u_p);

    // Focus on stabilization term
    const auto& lambda     = compute_lambda(u_m, u_p, n_minus);
    const auto& jump_rho_e = this->inv_gamma_m1*(pres_m - pres_p);

    // Return the Rusanov flux (implicit part energy)
    return scalar_product(avg_enthalpy, n_minus) +
           static_cast<value_type>(0.5)*lambda*jump_rho_e;
  }

} // namespace NumericalFlux
