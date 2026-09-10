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
#include "../../physics/physical_flux.h"

// @sect{Numerical flux}

// In this namespace, we implement (virtual) signature functions for the numerical flux
//
namespace NumericalFlux {
  using namespace dealii;

  /**
   * We declare now the class for a generic numerical flux for the Euler equations
   */
  template<unsigned dim, typename Number>
  class NumericalFluxEuler: public Physics::PhysicalFluxEuler<dim, Number> {
  public:
    // Define the arithmetic type for this class
    using value_type = typename Physics::PhysicalFluxEuler<dim, Number>::value_type;

    /**
     * Default class constructor
     */
    NumericalFluxEuler() = default;

    /**
     * Class constructor
     * @param Ma_ Mach number
     */
    NumericalFluxEuler(const value_type Ma_);

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
                                             const Tensor<1, dim, Number>& n_minus) const = 0;

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
                                                                    const Tensor<1, dim, Number>& n_minus) const = 0;

    /**
     * Numerical flux for the momentum equation (implicit part)
     * @param pres_m pressure 'interior' side
     * @param pres_p pressure 'exterior' side
     * @param n_minus Unit normal from 'interior' to 'exterior'
     */
    virtual Tensor<1, dim, Number> numerical_flux_momentum_implicit(const Number& pres_m,
                                                                    const Number& pres_p,
                                                                    const Tensor<1, dim, Number>& n_minus) const = 0;

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
                                                  const Tensor<1, dim, Number>& n_minus) const = 0;

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
                                                  const Tensor<1, dim, Number>& n_minus) const = 0;
  };

  // Class constructor
  //
  template<unsigned dim, typename Number>
  NumericalFluxEuler<dim, Number>::NumericalFluxEuler(const value_type Ma_):
    Physics::PhysicalFluxEuler<dim, Number>(Ma_) {}

} // namespace NumericalFlux
