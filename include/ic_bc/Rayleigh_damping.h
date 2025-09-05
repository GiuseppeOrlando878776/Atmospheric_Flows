/*--- Author: Giuseppe Orlando, 2025. ---*/

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

  static const double z_start       = 10000.0; /*--- Start of Rayleigh damping for top boundary ---*/
  static const double x_start_left  = 20000.0; /*--- Start of Rayleigh damping for left boundary ---*/
  static const double x_start_right = 40000.0; /*--- Start of Rayleigh damping for right boundary ---*/
  static const double y_start_left  = 10000.0; /*--- Start of Rayleigh damping for y left boundary ---*/
  static const double y_start_right = 30000.0; /*--- Start of Rayleigh damping for y right boundary ---*/

  /**
   * We focus now on the Rayleigh damping profile along the vertical direction.
     We create a suitable function for that. This function will be either scalar
     or vectorial (for the velocity). That's why the auxiliary template parameter
     n_comp is present.
   */
  template<unsigned dim, unsigned n_comp, typename T = double>
  class Rayleigh: public Function<dim, T> {
  public:
    Rayleigh(const T initial_time = static_cast<T>(0.0)); /*--- Class constructor ---*/

    virtual T value(const Point<dim, T>& p,
                    const unsigned       component = 0) const override; /*--- Damping profile evaluation ---*/

    virtual void vector_value(const Point<dim, T>& p,
                              Vector<T>&           values) const override; /*--- Damping profile vector evaluation for the velocity ---*/

  private:
    const T z_start; /*--- Starting coordinate of the damping layer ---*/
    const T z_max;   /*--- Ending coordinate of the damping layer ---*/
  };

  // Class constructor, which simply calls the parent class constructor
  // and then initialize some data
  //
  template<unsigned dim, unsigned n_comp, typename T>
  Rayleigh<dim, n_comp, T>::Rayleigh(const T initial_time):
    Function<dim, T>(n_comp, initial_time),
    z_start(static_cast<T>(RayleighDamping::z_start)/static_cast<T>(EquationData::L_ref)),
    z_max(static_cast<T>(EquationData::z_max)/static_cast<T>(EquationData::L_ref)) {}

  // Evaluation of Rayleigh damping profile
  //
  template<unsigned dim, unsigned n_comp, typename T>
  T Rayleigh<dim, n_comp, T>::value(const Point<dim, T>& p,
                                    const unsigned       component) const {
    (void)component;
    AssertIndexRange(component, n_comp);

    if(p[2] < z_start) {
      return static_cast<T>(0.0);
    }

    return static_cast<T>(1.2)*
           std::sin(static_cast<T>(0.5)*static_cast<T>(numbers::PI)*(p[2] - z_start)/(z_max - z_start))*
           std::sin(static_cast<T>(0.5)*static_cast<T>(numbers::PI)*(p[2] - z_start)/(z_max - z_start)); /*--- Rayleigh profile expression ---*/
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
    Rayleigh_Aux(const T initial_time = static_cast<T>(0.0)); /*--- Class constructor ---*/

    virtual T value(const Point<dim, T>& p,
                    const unsigned       component = 0) const override; /*--- Damping profile evaluation ---*/

    virtual void vector_value(const Point<dim, T>& p,
                              Vector<T>&           values) const override; /*--- Damping profile vector evaluation for the velocity ---*/

  private:
    const T z_start; /*--- Starting coordinate of the damping layer ---*/
    const T z_max;   /*--- Ending coordinate of the damping layer ---*/
  };

  // Class constructor, which simply calls the parent class constructor
  // and then initialize some data
  //
  template<unsigned dim, unsigned n_comp, typename T>
  Rayleigh_Aux<dim, n_comp, T>::Rayleigh_Aux(const T initial_time):
    Function<dim, T>(n_comp, initial_time),
    z_start(static_cast<T>(RayleighDamping::z_start)/static_cast<T>(EquationData::L_ref)),
    z_max(static_cast<T>(EquationData::z_max)/static_cast<T>(EquationData::L_ref)) {}

  // Evaluation of Rayleigh damping profile
  //
  template<unsigned dim, unsigned n_comp, typename T>
  T Rayleigh_Aux<dim, n_comp, T>::value(const Point<dim, T>& p,
                                        const unsigned       component) const {
    (void)component;
    AssertIndexRange(component, n_comp);

    if(p[2] < z_start) {
      return static_cast<T>(1.0);
    }

    return static_cast<T>(1.0)/
           (static_cast<T>(1.0) +
            static_cast<T>(1.2)*std::sin(static_cast<T>(0.5)*static_cast<T>(numbers::PI)*(p[2] - z_start)/(z_max - z_start))*
                                std::sin(static_cast<T>(0.5)*static_cast<T>(numbers::PI)*(p[2] - z_start)/(z_max - z_start)));
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
    Rayleigh_Right(const T initial_time = static_cast<T>(0.0)); /*--- Class constructor ---*/

    virtual T value(const Point<dim, T>& p,
                    const unsigned       component = 0) const override; /*--- Damping profile evaluation ---*/

    virtual void vector_value(const Point<dim, T>& p,
                              Vector<T>&           values) const override; /*--- Damping profile vector evaluation for the velocity ---*/

  private:
    const T x_start; /*--- Starting coordinate of the damping layer ---*/
    const T x_max;   /*--- Ending coordinate of the damping layer ---*/
  };

  // Class constructor, which simply calls the parent class constructor
  // and then initialize some data
  //
  template<unsigned dim, unsigned n_comp, typename T>
  Rayleigh_Right<dim, n_comp, T>::Rayleigh_Right(const T initial_time):
    Function<dim, T>(n_comp, initial_time),
    x_start(static_cast<T>(RayleighDamping::x_start_right)/static_cast<T>(EquationData::L_ref)),
    x_max(static_cast<T>(EquationData::x_max)/static_cast<T>(EquationData::L_ref)) {}

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

    return static_cast<T>(1.2)*
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
    for(unsigned i = 0; i < n_comp; ++i)
      values[i] = value(p, i);
  }


  /**
   * We create an auxiliary class for the term (1/(1 + dt*tau)) in order to avoid loop.
   */
  template<unsigned dim, unsigned n_comp, typename T = double>
  class Rayleigh_Aux_Right: public Function<dim, T> {
  public:
    Rayleigh_Aux_Right(const T initial_time = static_cast<T>(0.0)); /*--- Class constructor ---*/

    virtual T value(const Point<dim, T>& p,
                    const unsigned       component = 0) const override; /*--- Damping profile evaluation ---*/

    virtual void vector_value(const Point<dim, T>& p,
                              Vector<T>&           values) const override; /*--- Damping profile vector evaluation for the velocity ---*/

  private:
    const T x_start; /*--- Starting coordinate of the damping layer ---*/
    const T x_max;   /*--- Ending coordinate of the damping layer ---*/
  };

  // Class constructor, which simply calls the parent class constructor
  // and then initialize some data
  //
  template<unsigned dim, unsigned n_comp, typename T>
  Rayleigh_Aux_Right<dim, n_comp, T>::Rayleigh_Aux_Right(const T initial_time):
    Function<dim>(n_comp, initial_time),
    x_start(static_cast<T>(RayleighDamping::x_start_right)/static_cast<T>(EquationData::L_ref)),
    x_max(static_cast<T>(EquationData::x_max)/static_cast<T>(EquationData::L_ref)) {}

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
            static_cast<T>(1.2)*std::sin(static_cast<T>(0.5)*static_cast<T>(numbers::PI)*(p[0] - x_start)/(x_max - x_start))*
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
    Rayleigh_Left(const T initial_time = static_cast<T>(0.0)); /*--- Class constructor ---*/

    virtual T value(const Point<dim, T>& p,
                    const unsigned       component = 0) const override; /*--- Damping profile evaluation ---*/

    virtual void vector_value(const Point<dim, T>& p,
                              Vector<T>&           values) const override; /*--- Damping profile vector evaluation for the velocity ---*/

  private:
    const T x_start; /*--- Starting coordinate of the damping layer ---*/
    const T x_min;   /*--- Ending coordinate of the damping layer ---*/
  };

  // Class constructor, which simply calls the parent class constructor
  // and then initialize some data
  //
  template<unsigned dim, unsigned n_comp, typename T>
  Rayleigh_Left<dim, n_comp, T>::Rayleigh_Left(const T initial_time):
    Function<dim, T>(n_comp, initial_time),
    x_start(RayleighDamping::x_start_left/static_cast<T>(EquationData::L_ref)),
    x_min(static_cast<T>(0.0)) {}

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

    return static_cast<T>(1.2)*
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
    Rayleigh_Aux_Left(const T initial_time = static_cast<T>(0.0)); /*--- Class constructor ---*/

    virtual T value(const Point<dim, T>& p,
                    const unsigned       component = 0) const override; /*--- Damping profile evaluation ---*/

    virtual void vector_value(const Point<dim, T>& p,
                              Vector<T>&           values) const override; /*--- Damping profile vector evaluation for the velocity ---*/

  private:
    const T x_start; /*--- Starting coordinate of the damping layer ---*/
    const T x_min;   /*--- Ending coordinate of the damping layer ---*/
  };

  // Class constructor, which simply calls the parent class constructor
  // and then initialize some data
  //
  template<unsigned dim, unsigned n_comp, typename T>
  Rayleigh_Aux_Left<dim, n_comp, T>::Rayleigh_Aux_Left(const T initial_time):
    Function<dim>(n_comp, initial_time),
    x_start(static_cast<T>(RayleighDamping::x_start_left)/static_cast<T>(EquationData::L_ref)),
    x_min(static_cast<T>(0.0)) {}

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
            static_cast<T>(1.2)*
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
    Rayleigh_RightY(const T initial_time = static_cast<T>(0.0)); /*--- Class constructor ---*/

    virtual T value(const Point<dim, T>&  p,
                    const unsigned component = 0) const override; /*--- Damping profile evaluation ---*/

    virtual void vector_value(const Point<dim, T>& p,
                              Vector<T>&           values) const override; /*--- Damping profile vector evaluation for the velocity ---*/

  private:
    const T y_start; /*--- Starting coordinate of the damping layer ---*/
    const T y_max;   /*--- Ending coordinate of the damping layer ---*/
  };

  // Class constructor, which simply calls the parent class constructor
  // and then initialize some data
  //
  template<unsigned dim, unsigned n_comp, typename T>
  Rayleigh_RightY<dim, n_comp, T>::Rayleigh_RightY(const T initial_time):
    Function<dim>(n_comp, initial_time),
    y_start(static_cast<T>(RayleighDamping::y_start_right)/static_cast<T>(EquationData::L_ref)),
    y_max(static_cast<T>(EquationData::y_max)/static_cast<T>(EquationData::L_ref)) {}

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

    return static_cast<T>(1.2)*
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

    for(unsigned i = 0; i < n_comp; ++i)
      values[i] = value(p, i);
  }


  /**
   * We create an auxiliary class for the term (1/(1 + dt*tau)) in order to avoid loop.
   */
  template<unsigned dim, unsigned n_comp, typename T = double>
  class Rayleigh_Aux_RightY: public Function<dim, T> {
  public:
    Rayleigh_Aux_RightY(const T initial_time = static_cast<T>(0.0)); /*--- Class constructor ---*/

    virtual T value(const Point<dim, T>& p,
                    const unsigned       component = 0) const override; /*--- Damping profile evaluation ---*/

    virtual void vector_value(const Point<dim, T>& p,
                              Vector<T>&           values) const override; /*--- Damping profile vector evaluation for the velocity ---*/

  private:
    const T y_start; /*--- Starting coordinate of the damping layer ---*/
    const T y_max;   /*--- Ending coordinate of the damping layer ---*/
  };

  // Class constructor, which simply calls the parent class constructor
  // and then initialize some data
  //
  template<unsigned dim, unsigned n_comp, typename T>
  Rayleigh_Aux_RightY<dim, n_comp, T>::Rayleigh_Aux_RightY(const T initial_time):
    Function<dim>(n_comp, initial_time),
    y_start(static_cast<T>(RayleighDamping::y_start_right)/static_cast<T>(EquationData::L_ref)),
    y_max(static_cast<T>(EquationData::y_max)/static_cast<T>(EquationData::L_ref)) {}

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
            static_cast<T>(1.2)*
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
    Rayleigh_LeftY(const T initial_time = static_cast<T>(0.0)); /*--- Class constructor ---*/

    virtual T value(const Point<dim, T>& p,
                    const unsigned       component = 0) const override; /*--- Damping profile evaluation ---*/

    virtual void vector_value(const Point<dim, T>& p,
                              Vector<T>&           values) const override; /*--- Damping profile vector evaluation for the velocity ---*/

  private:
    const T y_start; /*--- Starting coordinate of the damping layer ---*/
    const T y_min;   /*--- Ending coordinate of the damping layer ---*/
  };

  // Class constructor, which simply calls the parent class constructor
  // and then initialize some data
  //
  template<unsigned dim, unsigned n_comp, typename T>
  Rayleigh_LeftY<dim, n_comp, T>::Rayleigh_LeftY(const T initial_time):
    Function<dim>(n_comp, initial_time),
    y_start(static_cast<T>(RayleighDamping::y_start_left)/static_cast<T>(EquationData::L_ref)),
    y_min(static_cast<T>(0.0)) {}

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

    return static_cast<T>(1.2)*
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
    Rayleigh_Aux_LeftY(const T initial_time = static_cast<T>(0.0)); /*--- Class constructor ---*/

    virtual T value(const Point<dim, T>& p,
                    const unsigned       component = 0) const override; /*--- Damping profile evaluation ---*/

    virtual void vector_value(const Point<dim, T>& p,
                              Vector<T>&           values) const override; /*--- Damping profile vector evaluation for the velocity ---*/

  private:
    const T y_start; /*--- Starting coordinate of the damping layer ---*/
    const T y_min;   /*--- Ending coordinate of the damping layer ---*/
  };

  // Class constructor, which simply calls the parent class constructor
  // and then initialize some data
  //
  template<unsigned dim, unsigned n_comp, typename T>
  Rayleigh_Aux_LeftY<dim, n_comp, T>::Rayleigh_Aux_LeftY(const T initial_time):
    Function<dim>(n_comp, initial_time),
    y_start(static_cast<T>(RayleighDamping::y_start_left)/static_cast<T>(EquationData::L_ref)),
    y_min(static_cast<T>(0.0)) {}

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
            static_cast<T>(1.2)*
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
