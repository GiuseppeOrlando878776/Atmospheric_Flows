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
#include <vector>

// @sect{Time stepping routines}

// In this namespace, we declare a general interface for Runge-Kutta method
//
namespace TimeStepping {
  /**
   * We declare now the class for a generic Runge-Kutta method
   */
  template<typename T = double>
  class RungeKutta {
  public:
    /**
     * Default class constructor. This should never be used
     */
    RungeKutta() = default;

    /**
     * Class constructor
     * @param a coefficients of the method (Butcher tableau)
     * @param b weights of the method (Nutcher tableau)
     */
    RungeKutta(const std::vector<std::vector<T>>& a_,
               const std::vector<T>& b_);

    /**
     * Get the number of stages
     * @return n_stages number of stages
     */
    inline unsigned get_n_stages() const;

    /**
     * Get the coefficients in the Butcher tableau representation
     * @return a coefficient of the method
     * @return b weights of the method
     */
    void get_coefficients(std::vector<std::vector<T>>& a_,
                          std::vector<T>&              b_) const;

    /**
     * Get the coefficients in the Butcher tableau representation
     * @return a coefficient of the method
     * @return b weights of the method
     * @return c nodes of the method
     */
    void get_coefficients(std::vector<std::vector<T>>& a_,
                          std::vector<T>&              b_,
                          std::vector<T>&              c_) const;

  protected:
    const unsigned n_stages; /*!< Number of stages */

    const std::vector<std::vector<T>> a; /*!< Coefficients of the method (Butcher tableau) */

    const std::vector<T> b; /*!< Weigths of the Runge-Kutta method */

    std::vector<T> c; /*!< Nodes of the Runge-Kutta method */
  };

  // Class constructor
  //
  template<typename T>
  RungeKutta<T>::RungeKutta(const std::vector<std::vector<T>>& a_,
                            const std::vector<T>&              b_):
    n_stages(b_.size()), a(a_), b(b_)
    {
      // Initialize the nodes using the classical rule $c_{i} = \sum_{j}a_{ij}$.
      // This is not mandatory in principle, but ALL the methods obey to it
      c.resize(n_stages);
      std::fill(c.begin(), c.end(), static_cast<T>(0.0));
      for(unsigned i = 0; i < n_stages; ++i) {
        for(unsigned j = 0; j < n_stages; ++j) {
          c[i] += a[i][j];
        }
      }
    }

  // Get the number of stages
  //
  template<typename T>
  inline unsigned RungeKutta<T>::get_n_stages() const {
    return n_stages;
  }

  // Get the coefficients of the Runge-Kutta scheme
  //
  template<typename T>
  void RungeKutta<T>::get_coefficients(std::vector<std::vector<T>>& a_,
                                       std::vector<T>&              b_) const {
    a_ = this->a;
    b_ = this->b;
  }

  // Get the coefficients of the Runge-Kutta scheme
  //
  template<typename T>
  void RungeKutta<T>::get_coefficients(std::vector<std::vector<T>>& a_,
                                       std::vector<T>&              b_,
                                       std::vector<T>&              c_) const {
    a_ = this->a;
    b_ = this->b;
    c_ = this->c;
  }

} // namespace TimeStepping
