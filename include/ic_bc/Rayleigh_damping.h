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

// @sect{Rayleigh damping}

// In this namespace, we declare the Rayleigh dumping profiles
//
namespace RayleighDamping {
  using namespace dealii;

  /**
   * We focus now on the Rayleigh damping profile along the vertical direction.
     We create a suitable function for that. This function will be either scalar
     or vectorial (for the velocity). That's why the auxiliary template parameter
     n_comp is present.
   */
  template<unsigned dim, unsigned n_comp, typename T = double>
  class Rayleigh: public Function<dim, T> {
  public:
    /**
     * Class constructor
     * @param z_start_ starting coordinate of the damping layer
     * @param z_max_ ending coordinate of the damping layer
     * @param lambda_z_ intensity of the damping layer
     * @param L_ref_ reference length
     */
    Rayleigh(const T z_start_,
             const T z_max_,
             const T lambda_z_,
             const T L_ref_ = static_cast<T>(1.0));

    /**
     * Evaluation of the damping layer profile for each component
     * @param p point coordinate
     * @param component spatial component to be evaluated
     */
    virtual T value(const Point<dim, T>& p,
                    const unsigned       component = 0) const override;

    /**
     * Vector evaluation of the damping layer
     * @param p point coordinate
     * @return values vector with all components
     */
    virtual void vector_value(const Point<dim, T>& p,
                              Vector<T>&           values) const override;

  private:
    const T L_ref; /*!< Reference length */

    const T z_start;  /*!< Starting coordinate of the damping layer */
    const T z_max;    /*!< Ending coordinate of the damping layer */
    const T lambda_z; /*!< Intensity of the damping layer */
  };

  // Class constructor, which simply calls the parent class constructor
  // and then initializes some data
  //
  template<unsigned dim, unsigned n_comp, typename T>
  Rayleigh<dim, n_comp, T>::Rayleigh(const T z_start_,
                                     const T z_max_,
                                     const T lambda_z_,
                                     const T L_ref_):
    Function<dim, T>(n_comp), L_ref(L_ref_),
    z_start(z_start_/L_ref), z_max(z_max_/L_ref), lambda_z(lambda_z_) {}

  // Evaluation of Rayleigh damping profile
  //
  template<unsigned dim, unsigned n_comp, typename T>
  T Rayleigh<dim, n_comp, T>::value(const Point<dim, T>& p,
                                    const unsigned       component) const {
    (void)component;
    AssertIndexRange(component, n_comp);

    if(p[dim - 1] < z_start) {
      return static_cast<T>(0.0);
    }

    // Rayleigh profile expression
    return lambda_z*
           std::sin(static_cast<T>(0.5)*static_cast<T>(numbers::PI)*(p[dim - 1] - z_start)/(z_max - z_start))*
           std::sin(static_cast<T>(0.5)*static_cast<T>(numbers::PI)*(p[dim - 1] - z_start)/(z_max - z_start));
  }

  // We need a vector value instance to deal with the velocity or, more in general,
  // if n_comp > 1.
  //
  template<unsigned dim, unsigned n_comp, typename T>
  void Rayleigh<dim, n_comp, T>::vector_value(const Point<dim, T>& p,
                                              Vector<T>&           values) const {
    Assert(values.size() == n_comp, ExcDimensionMismatch(values.size(), n_comp));

    for(unsigned i = 0; i < n_comp; ++i) {
      values[i] = value(p, i);
    }
  }


  /**
   * We create an auxiliary class for the term (1/(1 + dt*tau)) in order to avoid loop.
     The template parameter n_comp has the same meaning of the previous class.
   */
  template<unsigned dim, unsigned n_comp, typename T = double>
  class Rayleigh_Aux: public Function<dim, T> {
  public:
    /**
     * Class constructor
     * @param z_start_ starting coordinate of the damping layer
     * @param z_max_ ending coordinate of the damping layer
     * @param lambda_z_ intensity of the damping layer
     * @param L_ref_ reference length
     */
    Rayleigh_Aux(const T z_start_,
                 const T z_max_,
                 const T lambda_z_,
                 const T L_ref_ = static_cast<T>(1.0));

    /**
     * Evaluation for each component
     * @param p point coordinate
     * @param component spatial component to be evaluated
     */
    virtual T value(const Point<dim, T>& p,
                    const unsigned       component = 0) const override;

    /**
     * Vector evaluation
     * @param p point coordinate
     * @return values vector with all components
     */
    virtual void vector_value(const Point<dim, T>& p,
                              Vector<T>&           values) const override;

  private:
    const T L_ref; /*!< Reference length */

    const T z_start;  /*!< Starting coordinate of the damping layer */
    const T z_max;    /*!< Ending coordinate of the damping layer */
    const T lambda_z; /*!< Intensity of the damping layer */
  };

  // Class constructor, which simply calls the parent class constructor
  // and then initialize some data
  //
  template<unsigned dim, unsigned n_comp, typename T>
  Rayleigh_Aux<dim, n_comp, T>::Rayleigh_Aux(const T z_start_, const T z_max_, const T lambda_z_,
                                             const T L_ref_):
    Function<dim, T>(n_comp), L_ref(L_ref_),
    z_start(z_start_/L_ref), z_max(z_max_/L_ref), lambda_z(lambda_z_) {}

  // Evaluation of Rayleigh damping profile
  //
  template<unsigned dim, unsigned n_comp, typename T>
  T Rayleigh_Aux<dim, n_comp, T>::value(const Point<dim, T>& p,
                                        const unsigned       component) const {
    (void)component;
    AssertIndexRange(component, n_comp);

    if(p[dim - 1] < z_start) {
      return static_cast<T>(1.0);
    }

    return static_cast<T>(1.0)/
           (static_cast<T>(1.0) +
            lambda_z*std::sin(static_cast<T>(0.5)*static_cast<T>(numbers::PI)*(p[dim - 1] - z_start)/(z_max - z_start))*
                     std::sin(static_cast<T>(0.5)*static_cast<T>(numbers::PI)*(p[dim - 1] - z_start)/(z_max - z_start)));
  }

  // We need a vector value instance to deal with the velocity or, more in general,
  // if n_comp > 1.
  //
  template<unsigned dim, unsigned n_comp, typename T>
  void Rayleigh_Aux<dim, n_comp, T>::vector_value(const Point<dim, T>& p,
                                                  Vector<T>&           values) const {
    Assert(values.size() == n_comp, ExcDimensionMismatch(values.size(), n_comp));

    for(unsigned i = 0; i < n_comp; ++i) {
      values[i] = value(p, i);
    }
  }


  /**
   * We do the same for the Rayleigh damping profile along the right lateral boundary.
   */
  template<unsigned dim, unsigned n_comp, typename T = double>
  class Rayleigh_Right: public Function<dim, T> {
  public:
    /**
     * Class constructor
     * @param x_start_ starting coordinate of the damping layer
     * @param x_max_ ending coordinate of the damping layer
     * @param lambda_x_ intensity of the damping layer
     * @param L_ref_ reference length
     */
    Rayleigh_Right(const T x_start_,
                   const T x_max_,
                   const T lambda_x_,
                   const T L_ref_ = static_cast<T>(1.0));

    /**
     * Evaluation of the damping layer profile for each component
     * @param p point coordinate
     * @param component spatial component to be evaluated
     */
    virtual T value(const Point<dim, T>& p,
                    const unsigned       component = 0) const override;

    /**
     * Vector evaluation of the damping layer
     * @param p point coordinate
     * @return values vector with all components
     */
    virtual void vector_value(const Point<dim, T>& p,
                              Vector<T>&           values) const override;

  private:
    const T L_ref; /*!< Reference length */

    const T x_start;  /*!< Starting coordinate of the damping layer */
    const T x_max;    /*!< Ending coordinate of the damping layer */
    const T lambda_x; /*!< Intensity of the damping layer */
  };

  // Class constructor, which simply calls the parent class constructor
  // and then initialize some data
  //
  template<unsigned dim, unsigned n_comp, typename T>
  Rayleigh_Right<dim, n_comp, T>::Rayleigh_Right(const T x_start_, const T x_max_, const T lambda_x_,
                                                 const T L_ref_):
    Function<dim, T>(n_comp, 0.0), L_ref(L_ref_),
    x_start(x_start_/L_ref), x_max(x_max_/L_ref), lambda_x(lambda_x_) {}

  // Evaluation of Rayleigh damping profile
  //
  template<unsigned dim, unsigned n_comp, typename T>
  T Rayleigh_Right<dim, n_comp, T>::value(const Point<dim, T>& p,
                                          const unsigned       component) const {
    (void)component;
    AssertIndexRange(component, n_comp);

    if(p[0] < x_start) {
      return static_cast<T>(0.0);
    }

    return lambda_x*
           std::sin(static_cast<T>(0.5)*static_cast<T>(numbers::PI)*(p[0] - x_start)/(x_max - x_start))*
           std::sin(static_cast<T>(0.5)*static_cast<T>(numbers::PI)*(p[0] - x_start)/(x_max - x_start));
  }

  // We need a vector value instance to deal with the velocity or, more in general,
  // if n_comp > 1.
  //
  template<unsigned dim, unsigned n_comp, typename T>
  void Rayleigh_Right<dim, n_comp, T>::vector_value(const Point<dim, T>& p,
                                                    Vector<T>&           values) const {
    Assert(values.size() == n_comp, ExcDimensionMismatch(values.size(), n_comp));

    for(unsigned i = 0; i < n_comp; ++i) {
      values[i] = value(p, i);
    }
  }


  /**
   * We create an auxiliary class for the term (1/(1 + dt*tau)) in order to avoid loop.
   */
  template<unsigned dim, unsigned n_comp, typename T = double>
  class Rayleigh_Aux_Right: public Function<dim, T> {
  public:
    /**
     * Class constructor
     * @param x_start_ starting coordinate of the damping layer
     * @param x_max_ ending coordinate of the damping layer
     * @param lambda_x_ intensity of the damping layer
     * @param L_ref_ reference length
     */
    Rayleigh_Aux_Right(const T x_start_,
                       const T x_max_,
                       const T lambda_x_,
                       const T L_ref_ = static_cast<T>(1.0));

    /**
     * Evaluation for each component
     * @param p point coordinate
     * @param component spatial component to be evaluated
     */
    virtual T value(const Point<dim, T>& p,
                    const unsigned       component = 0) const override;

    /**
     * Vector evaluation
     * @param p point coordinate
     * @return values vector with all components
     */
    virtual void vector_value(const Point<dim, T>& p,
                              Vector<T>&           values) const override;

  private:
    const T L_ref; /*!< Reference length */

    const T x_start;  /*!< Starting coordinate of the damping layer */
    const T x_max;    /*!< Ending coordinate of the damping layer */
    const T lambda_x; /*!< Intensity of the damping layer */
  };

  // Class constructor, which simply calls the parent class constructor
  // and then initialize some data
  //
  template<unsigned dim, unsigned n_comp, typename T>
  Rayleigh_Aux_Right<dim, n_comp, T>::Rayleigh_Aux_Right(const T x_start_,
                                                         const T x_max_,
                                                         const T lambda_x_,
                                                         const T L_ref_):
    Function<dim>(n_comp), L_ref(L_ref_),
    x_start(x_start_/L_ref), x_max(x_max_/L_ref), lambda_x(lambda_x_) {}

  // Evaluation of Rayleigh damping profile
  //
  template<unsigned dim, unsigned n_comp, typename T>
  T Rayleigh_Aux_Right<dim, n_comp, T>::value(const Point<dim, T>& p,
                                              const unsigned       component) const {
    (void)component;
    AssertIndexRange(component, n_comp);

    if(p[0] < x_start) {
      return static_cast<T>(1.0);
    }

    return static_cast<T>(1.0)/
           (static_cast<T>(1.0) +
            lambda_x*std::sin(static_cast<T>(0.5)*static_cast<T>(numbers::PI)*(p[0] - x_start)/(x_max - x_start))*
                     std::sin(static_cast<T>(0.5)*static_cast<T>(numbers::PI)*(p[0] - x_start)/(x_max - x_start)));
  }

  // We need a vector value instance to deal with the velocity or, more in general,
  // if n_comp > 1.
  //
  template<unsigned dim, unsigned n_comp, typename T>
  void Rayleigh_Aux_Right<dim, n_comp, T>::vector_value(const Point<dim, T>& p,
                                                        Vector<T>&           values) const {
    Assert(values.size() == n_comp, ExcDimensionMismatch(values.size(), n_comp));

    for(unsigned i = 0; i < n_comp; ++i) {
      values[i] = value(p, i);
    }
  }


  /**
   * We do the same for the Rayleigh damping profile along the left lateral boundary
   */
  template<unsigned dim, unsigned n_comp, typename T = double>
  class Rayleigh_Left: public Function<dim, T> {
  public:
    /**
     * Class constructor
     * @param x_start_ starting coordinate of the damping layer
     * @param x_min_ ending coordinate of the damping layer
     * @param lambda_x_ intensity of the damping layer
     * @param L_ref_ reference length
     */
    Rayleigh_Left(const T x_start_,
                  const T x_min_,
                  const T lambda_x_,
                  const T L_ref_ = static_cast<T>(1.0));

    /**
     * Evaluation of the damping layer profile for each component
     * @param p point coordinate
     * @param component spatial component to be evaluated
     */
    virtual T value(const Point<dim, T>& p,
                    const unsigned       component = 0) const override;

    /**
     * Vector evaluation of the damping layer
     * @param p point coordinate
     * @return values vector with all components
     */
    virtual void vector_value(const Point<dim, T>& p,
                              Vector<T>&           values) const override;

  private:
    const T L_ref; /*!< Reference length */

    const T x_start;  /*!< Starting coordinate of the damping layer */
    const T x_min;    /*!< Ending coordinate of the damping layer */
    const T lambda_x; /*!< Intensity of the damping layer */
  };

  // Class constructor, which simply calls the parent class constructor
  // and then initialize some data
  //
  template<unsigned dim, unsigned n_comp, typename T>
  Rayleigh_Left<dim, n_comp, T>::Rayleigh_Left(const T x_start_,
                                               const T x_min_,
                                               const T lambda_x_,
                                               const T L_ref_):
    Function<dim, T>(n_comp), L_ref(L_ref_),
    x_start(x_start_/L_ref), x_min(x_min_/L_ref), lambda_x(lambda_x_) {}

  // Evaluation of Rayleigh damping profile
  //
  template<unsigned dim, unsigned n_comp, typename T>
  T Rayleigh_Left<dim, n_comp, T>::value(const Point<dim, T>& p,
                                         const unsigned component) const {
    (void)component;
    AssertIndexRange(component, n_comp);

    if(p[0] > x_start) {
      return static_cast<T>(0.0);
    }

    return lambda_x*
           std::sin(static_cast<T>(0.5)*static_cast<T>(numbers::PI)*(p[0] - x_start)/(x_min - x_start))*
           std::sin(static_cast<T>(0.5)*static_cast<T>(numbers::PI)*(p[0] - x_start)/(x_min - x_start));
  }

  // We need a vector value instance to deal with the velocity or, more in general,
  // if n_comp > 1.
  //
  template<unsigned dim, unsigned n_comp, typename T>
  void Rayleigh_Left<dim, n_comp, T>::vector_value(const Point<dim, T>& p,
                                                   Vector<T>&           values) const {
    Assert(values.size() == n_comp, ExcDimensionMismatch(values.size(), n_comp));

    for(unsigned i = 0; i < n_comp; ++i) {
      values[i] = value(p, i);
    }
  }


  /**
   * We create an auxiliary class for the term (1/(1 + dt*tau)) in order to avoid loop.
   */
  template<unsigned dim, unsigned n_comp, typename T = double>
  class Rayleigh_Aux_Left: public Function<dim, T> {
  public:
    /**
     * Class constructor
     * @param x_start_ starting coordinate of the damping layer
     * @param x_min_ ending coordinate of the damping layer
     * @param lambda_x_ intensity of the damping layer
     * @param L_ref_ reference length
     */
    Rayleigh_Aux_Left(const T x_start_, const T x_min_, const T lambda_x_,
                      const T L_ref_);

    /**
     * Evaluation for each component
     * @param p point coordinate
     * @param component spatial component to be evaluated
     */
    virtual T value(const Point<dim, T>& p,
                    const unsigned       component = 0) const override;

    /**
     * Vector evaluation
     * @param p point coordinate
     * @return values vector with all components
     */
    virtual void vector_value(const Point<dim, T>& p,
                              Vector<T>&           values) const override;

  private:
    const T L_ref; /*!< Reference length */

    const T x_start;  /*!< Starting coordinate of the damping layer */
    const T x_min;    /*!< Ending coordinate of the damping layer */
    const T lambda_x; /*!< Intensity of the damping layer */
  };

  // Class constructor, which simply calls the parent class constructor
  // and then initialize some data
  //
  template<unsigned dim, unsigned n_comp, typename T>
  Rayleigh_Aux_Left<dim, n_comp, T>::Rayleigh_Aux_Left(const T x_start_,
                                                       const T x_min_,
                                                       const T lambda_x_,
                                                       const T L_ref_):
    Function<dim>(n_comp), L_ref(L_ref_),
    x_start(x_start_/L_ref), x_min(x_min_/L_ref), lambda_x(lambda_x_) {}

  // Evaluation of Rayleigh damping profile
  //
  template<unsigned dim, unsigned n_comp, typename T>
  T Rayleigh_Aux_Left<dim, n_comp, T>::value(const Point<dim, T>& p,
                                             const unsigned component) const {
    (void)component;
    AssertIndexRange(component, n_comp);

    if(p[0] > x_start) {
      return static_cast<T>(1.0);
    }

    return static_cast<T>(1.0)/
           (static_cast<T>(1.0) +
            lambda_x*
            std::sin(static_cast<T>(0.5)*static_cast<T>(numbers::PI)*(p[0] - x_start)/(x_min - x_start))*
            std::sin(static_cast<T>(0.5)*static_cast<T>(numbers::PI)*(p[0] - x_start)/(x_min - x_start)));
  }

  // We need a vector value instance to deal with the velocity or, more in general,
  // if n_comp > 1.
  //
  template<unsigned dim, unsigned n_comp, typename T>
  void Rayleigh_Aux_Left<dim, n_comp, T>::vector_value(const Point<dim, T>& p,
                                                       Vector<T>&           values) const {
    Assert(values.size() == n_comp, ExcDimensionMismatch(values.size(), n_comp));

    for(unsigned i = 0; i < n_comp; ++i) {
      values[i] = value(p, i);
    }
  }


  /**
   * We do the same for the Rayleigh damping profile along the right y lateral boundary.
   */
  template<unsigned dim, unsigned n_comp, typename T = double>
  class Rayleigh_RightY: public Function<dim, T> {
  public:
    /**
     * Class constructor
     * @param y_start_ starting coordinate of the damping layer
     * @param y_max_ ending coordinate of the damping layer
     * @param lambda_y_ intensity of the damping layer
     * @param L_ref_ reference length
     */
    Rayleigh_RightY(const T y_start_,
                    const T y_max_,
                    const T lambda_y_,
                    const T L_ref_ = static_cast<T>(1.0));

    /**
     * Evaluation of the damping layer profile for each component
     * @param p point coordinate
     * @param component spatial component to be evaluated
     */
    virtual T value(const Point<dim, T>& p,
                    const unsigned       component = 0) const override;

    /**
     * Vector evaluation of the damping layer
     * @param p point coordinate
     * @return values vector with all components
     */
    virtual void vector_value(const Point<dim, T>& p,
                              Vector<T>&           values) const override;

  private:
    const T L_ref; /*!< Reference length */

    const T y_start;  /*!< Starting coordinate of the damping layer */
    const T y_max;    /*!< Ending coordinate of the damping layer */
    const T lambda_y; /*!< Intensity of the damping layer */
  };

  // Class constructor, which simply calls the parent class constructor
  // and then initialize some data
  //
  template<unsigned dim, unsigned n_comp, typename T>
  Rayleigh_RightY<dim, n_comp, T>::Rayleigh_RightY(const T y_start_,
                                                   const T y_max_,
                                                   const T lambda_y_,
                                                   const T L_ref_):
    Function<dim>(n_comp), L_ref(L_ref_),
    y_start(y_start_/L_ref), y_max(y_max_/L_ref), lambda_y(lambda_y_) {}

  // Evaluation of Rayleigh damping profile
  //
  template<unsigned dim, unsigned n_comp, typename T>
  T Rayleigh_RightY<dim, n_comp, T>::value(const Point<dim, T>& p,
                                           const unsigned       component) const {
    (void)component;
    AssertIndexRange(component, n_comp);

    if(p[1] < y_start) {
      return static_cast<T>(0.0);
    }

    return lambda_y*
           std::sin(static_cast<T>(0.5)*static_cast<T>(numbers::PI)*(p[1] - y_start)/(y_max - y_start))*
           std::sin(static_cast<T>(0.5)*static_cast<T>(numbers::PI)*(p[1] - y_start)/(y_max - y_start));
  }

  // We need a vector value instance to deal with the velocity or, more in general,
  // if n_comp > 1.
  //
  template<unsigned dim, unsigned n_comp, typename T>
  void Rayleigh_RightY<dim, n_comp, T>::vector_value(const Point<dim, T>& p,
                                                     Vector<T>& values) const {
    Assert(values.size() == n_comp, ExcDimensionMismatch(values.size(), n_comp));

    for(unsigned i = 0; i < n_comp; ++i) {
      values[i] = value(p, i);
    }
  }


  /**
   * We create an auxiliary class for the term (1/(1 + dt*tau)) in order to avoid loop.
   */
  template<unsigned dim, unsigned n_comp, typename T = double>
  class Rayleigh_Aux_RightY: public Function<dim, T> {
  public:
    /**
     * Class constructor
     * @param y_start_ starting coordinate of the damping layer
     * @param y_max_ ending coordinate of the damping layer
     * @param lambda_y_ intensity of the damping layer
     * @param L_ref_ reference length
     */
    Rayleigh_Aux_RightY(const T y_start_, const T y_max_, const T lambda_y_,
                        const T L_ref_ = static_cast<T>(1.0));

    /**
     * Evaluation for each component
     * @param p point coordinate
     * @param component spatial component to be evaluated
     */
    virtual T value(const Point<dim, T>& p,
                    const unsigned       component = 0) const override;

    /**
     * Vector evaluation
     * @param p point coordinate
     * @return values vector with all components
     */
    virtual void vector_value(const Point<dim, T>& p,
                              Vector<T>&           values) const override;

  private:
    const T L_ref; /*!< Reference length */

    const T y_start;  /*!< Starting coordinate of the damping layer */
    const T y_max;    /*!< Ending coordinate of the damping layer */
    const T lambda_y; /*!< Intensity of the damping layer */
  };

  // Class constructor, which simply calls the parent class constructor
  // and then initialize some data
  //
  template<unsigned dim, unsigned n_comp, typename T>
  Rayleigh_Aux_RightY<dim, n_comp, T>::Rayleigh_Aux_RightY(const T y_start_, const T y_max_, const T lambda_y_,
                                                           const T L_ref_):
    Function<dim>(n_comp, 0.0), L_ref(L_ref_),
    y_start(y_start_/L_ref), y_max(y_max_/L_ref), lambda_y(lambda_y_) {}

  // Evaluation of Rayleigh damping profile
  //
  template<unsigned dim, unsigned n_comp, typename T>
  T Rayleigh_Aux_RightY<dim, n_comp, T>::value(const Point<dim, T>& p,
                                               const unsigned       component) const {
    (void)component;
    AssertIndexRange(component, n_comp);

    if(p[1] < y_start) {
      return static_cast<T>(1.0);
    }

    return static_cast<T>(1.0)/
           (static_cast<T>(1.0) +
            lambda_y*
            std::sin(static_cast<T>(0.5)*static_cast<T>(numbers::PI)*(p[1] - y_start)/(y_max - y_start))*
            std::sin(static_cast<T>(0.5)*static_cast<T>(numbers::PI)*(p[1] - y_start)/(y_max - y_start)));
  }

  // We need a vector value instance to deal with the velocity or, more in general,
  // if n_comp > 1.
  //
  template<unsigned dim, unsigned n_comp, typename T>
  void Rayleigh_Aux_RightY<dim, n_comp, T>::vector_value(const Point<dim, T>& p,
                                                         Vector<T>&           values) const {
    Assert(values.size() == n_comp, ExcDimensionMismatch(values.size(), n_comp));

    for(unsigned i = 0; i < n_comp; ++i) {
      values[i] = value(p, i);
    }
  }


  /* We do the same for the Rayleigh damping profile along the left y lateral boundary
  */
  template<unsigned dim, unsigned n_comp, typename T = double>
  class Rayleigh_LeftY: public Function<dim, T> {
  public:
    /**
     * Class constructor
     * @param y_start_ starting coordinate of the damping layer
     * @param y_min_ ending coordinate of the damping layer
     * @param lambda_y_ intensity of the damping layer
     * @param L_ref_ reference length
     */
    Rayleigh_LeftY(const T y_start_,
                   const T y_min_,
                   const T lambda_y_,
                   const T L_ref_ = static_cast<T>(1.0));

    /**
     * Evaluation of the damping layer profile for each component
     * @param p point coordinate
     * @param component spatial component to be evaluated
     */
    virtual T value(const Point<dim, T>& p,
                    const unsigned       component = 0) const override;

    /**
     * Vector evaluation of the damping layer
     * @param p point coordinate
     * @return values vector with all components
     */
    virtual void vector_value(const Point<dim, T>& p,
                              Vector<T>&           values) const override;

  private:
    const T L_ref; /*!< Reference length */

    const T y_start;  /*!< Starting coordinate of the damping layer */
    const T y_min;    /*!< Ending coordinate of the damping layer */
    const T lambda_y; /*!< Intensity of the damping layer */
  };

  // Class constructor, which simply calls the parent class constructor
  // and then initialize some data
  //
  template<unsigned dim, unsigned n_comp, typename T>
  Rayleigh_LeftY<dim, n_comp, T>::Rayleigh_LeftY(const T y_start_,
                                                 const T y_min_,
                                                 const T lambda_y_,
                                                 const T L_ref_):
    Function<dim>(n_comp), L_ref(L_ref_),
    y_start(y_start_/L_ref), y_min(y_min_/L_ref), lambda_y(lambda_y_) {}

  // Evaluation of Rayleigh damping profile
  //
  template<unsigned dim, unsigned n_comp, typename T>
  T Rayleigh_LeftY<dim, n_comp, T>::value(const Point<dim, T>& p,
                                          const unsigned       component) const {
    (void)component;
    AssertIndexRange(component, n_comp);

    if(p[1] > y_start) {
      return static_cast<T>(0.0);
    }

    return lambda_y*
           std::sin(static_cast<T>(0.5)*static_cast<T>(numbers::PI)*(p[1] - y_start)/(y_min - y_start))*
           std::sin(static_cast<T>(0.5)*static_cast<T>(numbers::PI)*(p[1] - y_start)/(y_min - y_start));
  }

  // We need a vector value instance to deal with the velocity or, more in general,
  // if n_comp > 1.
  //
  template<unsigned dim, unsigned n_comp, typename T>
  void Rayleigh_LeftY<dim, n_comp, T>::vector_value(const Point<dim, T>& p,
                                                    Vector<T>&           values) const {
    Assert(values.size() == n_comp, ExcDimensionMismatch(values.size(), n_comp));

    for(unsigned i = 0; i < n_comp; ++i) {
      values[i] = value(p, i);
    }
  }


  /**
   * We create an auxiliary class for the term (1/(1 + dt*tau)) in order to avoid loop.
   */
  template<unsigned dim, unsigned n_comp, typename T = double>
  class Rayleigh_Aux_LeftY: public Function<dim, T> {
  public:
    /**
     * Class constructor
     * @param y_start_ starting coordinate of the damping layer
     * @param y_min_ ending coordinate of the damping layer
     * @param lambda_y_ intensity of the damping layer
     * @param L_ref_ reference length
     */
    Rayleigh_Aux_LeftY(const T y_start_, const T y_min_, const T lambda_y_,
                       const T L_ref_ = static_cast<T>(1.0));

    /**
     * Evaluation for each component
     * @param p point coordinate
     * @param component spatial component to be evaluated
     */
    virtual T value(const Point<dim, T>& p,
                    const unsigned       component = 0) const override;

    /**
     * Vector evaluation
     * @param p point coordinate
     * @return values vector with all components
     */
    virtual void vector_value(const Point<dim, T>& p,
                              Vector<T>&           values) const override;

  private:
    const T L_ref; /*!< Reference length */

    const T y_start;  /*!< Starting coordinate of the damping layer */
    const T y_min;    /*!< Ending coordinate of the damping layer */
    const T lambda_y; /*!< Intensity of the damping layer */
  };

  // Class constructor, which simply calls the parent class constructor
  // and then initialize some data
  //
  template<unsigned dim, unsigned n_comp, typename T>
  Rayleigh_Aux_LeftY<dim, n_comp, T>::Rayleigh_Aux_LeftY(const T y_start_,
                                                         const T y_min_,
                                                         const T lambda_y_,
                                                         const T L_ref_):
    Function<dim>(n_comp), L_ref(L_ref_),
    y_start(y_start_/L_ref), y_min(y_min_/L_ref), lambda_y(lambda_y_) {}

  // Evaluation of Rayleigh damping profile
  //
  template<unsigned dim, unsigned n_comp, typename T>
  T Rayleigh_Aux_LeftY<dim, n_comp, T>::value(const Point<dim, T>& p,
                                              const unsigned       component) const {
    (void)component;
    AssertIndexRange(component, n_comp);

    if(p[1] > y_start) {
      return static_cast<T>(1.0);
    }

    return static_cast<T>(1.0)/
           (static_cast<T>(1.0) +
            lambda_y*
            std::sin(static_cast<T>(0.5)*static_cast<T>(numbers::PI)*(p[1] - y_start)/(y_min - y_start))*
            std::sin(static_cast<T>(0.5)*static_cast<T>(numbers::PI)*(p[1] - y_start)/(y_min - y_start)));
  }

  // We need a vector value instance to deal with the velocity or, more in general,
  // if n_comp > 1.
  //
  template<unsigned dim, unsigned n_comp, typename T>
  void Rayleigh_Aux_LeftY<dim, n_comp, T>::vector_value(const Point<dim, T>& p,
                                                        Vector<T>&           values) const {
    Assert(values.size() == n_comp, ExcDimensionMismatch(values.size(), n_comp));

    for(unsigned i = 0; i < n_comp; ++i) {
      values[i] = value(p, i);
    }
  }

} // namespace RayleighDamping
