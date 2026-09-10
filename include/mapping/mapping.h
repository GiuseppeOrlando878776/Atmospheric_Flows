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

// We start by including the necessary deal.II header file and a related
// header file with some constant values
//
#include <deal.II/base/function.h>

#include "../equation_data.h"

// @sect{Mapping reference-physical domain}

// In this namespace, we declare the mapping between reference and physical
// elements using the Gal-Chen mapping
//
namespace GalChenMapping {
  using namespace dealii;

  static const unsigned degree_mapping          = 2; /*!< Mapping degree */
  static const unsigned extra_quadrature_degree = (degree_mapping == 1) ?
                                                  0 : my_ceil(0.5*(degree_mapping - 2)); /*!< Extra accuracy
                                                                                              for quadratures */

  /**
   * Now we can focus on mappings from reference element to the physical one
     using the Gal-Chen transformation. Notice that this is specific for the
     3D versiera of Agnesi (it requires a user-defined mountain profile in general)
   */
  template<unsigned dim, typename T = double>
  class PushForward: public Function<dim, T> {
  public:
    /**
     * Class constructor
     * @param z_max_ height of the domain
     * @param h_ height of the mountain
     * @param xc_ x-center of the mountain
     * @param yc_ y-center of the mountain
     * @param ac_ semi-width of the mountain
     * @param L_ref_ Reference length
     */
    PushForward(const T z_max_,
                const T h_, const T xc_, const T yc_, const T ac_,
                const T L_ref_ = static_cast<T>(1.0));

    /**
     * Class destructor
     */
    virtual ~PushForward() {};

    /**
     * Evaluate Gal-Chen transformation
     * @param p point coordinates
     * @param component component to be evaluated (for the sake of compatibility)
     */
    virtual T value(const Point<dim, T>& p,
                    const unsigned       component = 0) const override;

  private:
    const T L_ref; /*!< Reference length */

    const T z_max; /*!< Height of the domain */

    const T h;  /*!< Height of the mountain */
    const T xc; /*!< x-center of the mountain */
    const T yc; /*!< y-center of the mountain */
    const T ac; /*!< semi-width of the mountain */
  };

  // Class constructor
  //
  template<unsigned dim, typename T>
  PushForward<dim, T>::PushForward(const T z_max_,
                                   const T h_, const T xc_, const T yc_, const T ac_,
                                   const T L_ref_):
    Function<dim, T>(dim),
    L_ref(L_ref_), z_max(z_max_/L_ref),
    h(h_), xc(xc_), yc(yc_), ac(ac_) {}

  // Mapping from reference to physical
  //
  template<unsigned dim, typename T>
  T PushForward<dim, T>::value(const Point<dim, T>& p,
                               const unsigned       component) const {
    // x component
    if(component == 0) {
      return p[0];
    }
    // y component
    else if(component == 1) {
      return p[1];
    }
    // z component
    else if(component == 2) {
      auto hX = h/std::pow(static_cast<T>(1.0) +
                           ((p[0]*L_ref - xc)/ac)*((p[0]*L_ref - xc)/ac) +
                           ((p[1]*L_ref - yc)/ac)*((p[1]*L_ref - yc)/ac), static_cast<T>(1.5));
      hX /= L_ref;

      return p[2] + ((z_max - p[2])/z_max)*hX;
    }
  }

  /**
   * We compute now the inverse mapping (from physical to reference).
     Notice again that this is specific for the 3D versiera of Agnesi
     (it requires a user-defined mountain profile in general)
   */
  template<unsigned dim, typename T = double>
  class PullBack: public Function<dim, T> {
  public:
    /**
     * Class constructor
     * @param z_max_ height of the domain
     * @param h_ height of the mountain
     * @param xc_ x-center of the mountain
     * @param yc_ y-center of the mountain
     * @param ac_ semi-width of the mountain
     * @param L_ref_ Reference length
     */
    PullBack(const T z_max_,
             const T h_, const T xc_, const T yc_, const T ac_,
             const T L_ref_ = static_cast<T>(1.0));

    /**
     * Class destructor
     */
    virtual ~PullBack() {};

    /**
     * Evaluate inverse of Gal-Chen transformation
     * @param p point coordinates
     * @param component component to be evaluated (for the sake of compatibility)
     */
    virtual T value(const Point<dim, T>& p,
                    const unsigned       component = 0) const override;

  private:
    const T L_ref; /*!< Reference length */

    const T z_max; /*!< Height of the domain */

    const T h;  /*!< Height of the mountain */
    const T xc; /*!< x-center of the mountain */
    const T yc; /*!< y-center of the mountain */
    const T ac; /*!< semi-width of the mountain */
  };

  // Class constructor
  //
  template<unsigned dim, typename T>
  PullBack<dim, T>::PullBack(const T z_max_,
                             const T h_, const T xc_, const T yc_, const T ac_,
                             const T L_ref_):
    Function<dim, T>(dim),
    L_ref(L_ref_), z_max(z_max_/L_ref),
    h(h_), xc(xc_), yc(yc_), ac(ac_) {}

  // Mapping from physical to reference
  //
  template<unsigned dim, typename T>
  T PullBack<dim, T>::value(const Point<dim, T>& p,
                            const unsigned       component) const {
    // x component
    if(component == 0) {
      return p[0];
    }
    // y component
    else if(component == 1) {
      return p[1];
    }
    // z component
    else if(component == 2) {
      auto hx = h/std::pow(static_cast<T>(1.0) +
                           ((p[0]*L_ref - xc)/ac)*((p[0]*L_ref - xc)/ac) +
                           ((p[1]*L_ref - yc)/ac)*((p[1]*L_ref - yc)/ac), static_cast<T>(1.5));
      hx /= L_ref;

      return z_max*(p[2] - hx)/(z_max - hx);
    }
  }

} // namespace Mapping
