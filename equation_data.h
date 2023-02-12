/*--- Author: Giuseppe Orlando, 2023. ---*/

// @sect{Include files}

// We start by including the necessary deal.II header files and some C++
// related ones.
#include <deal.II/base/point.h>
#include <deal.II/base/function.h>

#include <cmath>

// @sect{Equation data}

// In this namespace, we declare the initial background conditions,
// the Raylegh dumping profiles and the mapping between reference and physical
// elements using the Gal-Chen mapping
//
namespace EquationData {
  using namespace dealii;

  /*--- Polynomial degrees. We typically consider the same poylnomial degree for all the variables ---*/
  static const unsigned int degree_T   = 4;
  static const unsigned int degree_rho = 4;
  static const unsigned int degree_u   = 4;

  static const double Cp_Cv = 1.4;   /*--- Specific heats ratio ---*/
  static const double R     = 287.0; /*--- Specific gas constant ---*/

  // We declare now the class that describes the initial condition for the velocity.
  //
  template<int dim>
  class Velocity: public Function<dim> {
  public:
    Velocity(const double initial_time = 0.0);

    virtual double value(const Point<dim>&  p,
                         const unsigned int component = 0) const override;

    virtual void vector_value(const Point<dim>& p,
                              Vector<double>&   values) const override;
  };


  template<int dim>
  Velocity<dim>::Velocity(const double initial_time): Function<dim>(dim, initial_time) {}


  template<int dim>
  double Velocity<dim>::value(const Point<dim>& p, const unsigned int component) const {
    AssertIndexRange(component, dim);

    if(component == 0)
      return 1.0;
    else
      return 0.0;
  }


  template<int dim>
  void Velocity<dim>::vector_value(const Point<dim>& p, Vector<double>& values) const {
    Assert(values.size() == dim, ExcDimensionMismatch(values.size(), dim));
    for(unsigned int i = 0; i < dim; ++i)
      values[i] = value(p, i);
  }


  // We do the same for the pressure (since it is a scalar field) we can derive
  // directly from the deal.II built-in class Function. Notice that in order to
  // get a dimensional version one should multiply the result by p_ref
  //
  template<int dim>
  class Pressure: public Function<dim> {
  public:
    Pressure(const double initial_time = 0.0);

    virtual double value(const Point<dim>&  p,
                         const unsigned int component = 0) const override;
  };


  template<int dim>
  Pressure<dim>::Pressure(const double initial_time): Function<dim>(1, initial_time) {}


  template<int dim>
  double Pressure<dim>::value(const Point<dim>& p, const unsigned int component) const {
    (void)component;
    AssertIndexRange(component, 1);

    const double N       = 0.02; /*--- Brunt-Väisälä frequency ---*/
    const double g       = 9.81; /*--- Accelaeration of gravity ---*/

    const double T_ref   = 273.0;    /*--- Reference temperature ---*/
    const double p_ref   = 100000.0; /*--- Reference pressure ---*/
    const double rho_ref = p_ref/(EquationData::R*T_ref); /*--- Reference density ---*/
    const double Gamma   = (EquationData::Cp_Cv - 1.0)/EquationData::Cp_Cv;
    const double pi_bar  = 1.0 - g*g/(N*N)*Gamma*rho_ref/p_ref*(1.0 - std::exp(-N*N/g*p[1]*1000.0)); /*--- Background Exner pressure ---*/

    return std::pow(pi_bar, 1.0/Gamma);
  }


  // We do the same for the density (since it is a scalar field) we can derive
  // directly from the deal.II built-in class Function. Notice that in order to
  // get a dimensional version one should multiply the result by rho_ref
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

    const double N         = 0.02; /*--- Brunt-Väisälä frequency ---*/
    const double g         = 9.81; /*--- Accelaeration of gravity ---*/

    const double T_ref     = 273.0; /*--- Reference temperature ---*/
    const double theta_bar = T_ref*std::exp(N*N/g*p[1]*1000.0); /*--- Background potential temperature ---*/
    const double p_ref     = 100000.0; /*--- Reference pressure ---*/
    const double rho_ref   = p_ref/(EquationData::R*T_ref); /*--- Reference density ---*/
    const double Gamma     = (EquationData::Cp_Cv - 1.0)/EquationData::Cp_Cv;
    const double pi_bar    = 1.0 - g*g/(N*N)*Gamma*rho_ref/p_ref*(1.0 - std::exp(-N*N/g*p[1]*1000.0));

    return T_ref/theta_bar*std::pow(pi_bar, 1.0/(EquationData::Cp_Cv - 1.0)); /*--- Background Exner pressure ---*/
  }


  // We focus now on the Raylegh damping profile along the vertical direction.
  // We create a suitable function for that.
  //
  template<int dim, unsigned int n_comp>
  class Raylegh: public Function<dim> {
  public:
    Raylegh(const double initial_time = 0.0);

    virtual double value(const Point<dim>&  p,
                         const unsigned int component = 0) const override;

    virtual void vector_value(const Point<dim>& p,
                              Vector<double>&   values) const override;

  private:
    const double z_start; /*--- Starting coordinate of the damping layer ---*/
    const double z_max;   /*--- Ending coordinate of the damping layer ---*/
  };


  template<int dim, unsigned int n_comp>
  Raylegh<dim, n_comp>::Raylegh(const double initial_time): Function<dim>(n_comp, initial_time),
                                                            z_start(9.0), z_max(20.0) {}


  template<int dim, unsigned int n_comp>
  double Raylegh<dim, n_comp>::value(const Point<dim>& p, const unsigned int component) const {
    (void)component;
    AssertIndexRange(component, n_comp);

    if(p[1] < z_start)
      return 0.0;

    return 0.15*std::sin(0.5*numbers::PI*(p[1] - z_start)/(z_max - z_start))*
                std::sin(0.5*numbers::PI*(p[1] - z_start)/(z_max - z_start));
  }


  template<int dim, unsigned int n_comp>
  void Raylegh<dim, n_comp>::vector_value(const Point<dim>& p, Vector<double>& values) const {
    Assert(values.size() == n_comp, ExcDimensionMismatch(values.size(), dim));
    for(unsigned int i = 0; i < n_comp; ++i)
      values[i] = value(p, i);
  }


  // We create an auxiliary class for the term (1/(1 + dt*tau)) in order to avoid loop
  //
  template<int dim, unsigned int n_comp>
  class Raylegh_Aux: public Function<dim> {
  public:
    Raylegh_Aux(const double initial_time = 0.0);

    virtual double value(const Point<dim>&  p,
                         const unsigned int component = 0) const override;

    virtual void vector_value(const Point<dim>& p,
                              Vector<double>&   values) const override;

  private:
    const double z_start; /*--- Starting coordinate of the damping layer ---*/
    const double z_max;   /*--- Ending coordinate of the damping layer ---*/
  };


  template<int dim, unsigned int n_comp>
  Raylegh_Aux<dim, n_comp>::Raylegh_Aux(const double initial_time): Function<dim>(n_comp, initial_time),
                                                                    z_start(9.0), z_max(20.0) {}


  template<int dim, unsigned int n_comp>
  double Raylegh_Aux<dim, n_comp>::value(const Point<dim>& p, const unsigned int component) const {
    (void)component;
    AssertIndexRange(component, n_comp);

    if(p[1] < z_start)
      return 1.0;

    return 1.0/(1.0 + 0.15*std::sin(0.5*numbers::PI*(p[1] - z_start)/(z_max - z_start))*
                           std::sin(0.5*numbers::PI*(p[1] - z_start)/(z_max - z_start)));
  }


  template<int dim, unsigned int n_comp>
  void Raylegh_Aux<dim, n_comp>::vector_value(const Point<dim>& p, Vector<double>& values) const {
    Assert(values.size() == n_comp, ExcDimensionMismatch(values.size(), dim));
    for(unsigned int i = 0; i < n_comp; ++i)
      values[i] = value(p, i);
  }


  // We do the same for the Raylegh damping profile along the right lateral boundary
  //
  template<int dim, unsigned int n_comp>
  class Raylegh_Right: public Function<dim> {
  public:
    Raylegh_Right(const double initial_time = 0.0);

    virtual double value(const Point<dim>&  p,
                         const unsigned int component = 0) const override;

    virtual void vector_value(const Point<dim>& p,
                              Vector<double>&   values) const override;

  private:
    const double x_start; /*--- Starting coordinate of the damping layer ---*/
    const double x_max;   /*--- Ending coordinate of the damping layer ---*/
  };


  template<int dim, unsigned int n_comp>
  Raylegh_Right<dim, n_comp>::Raylegh_Right(const double initial_time): Function<dim>(n_comp, initial_time),
                                                                        x_start(30.0), x_max(40.0) {}


  template<int dim, unsigned int n_comp>
  double Raylegh_Right<dim, n_comp>::value(const Point<dim>& p, const unsigned int component) const {
    (void)component;
    AssertIndexRange(component, n_comp);

    if(p[0] < x_start)
      return 0.0;

    return 0.15*std::sin(0.5*numbers::PI*(p[0] - x_start)/(x_max - x_start))*
                std::sin(0.5*numbers::PI*(p[0] - x_start)/(x_max - x_start));
  }


  template<int dim, unsigned int n_comp>
  void Raylegh_Right<dim, n_comp>::vector_value(const Point<dim>& p, Vector<double>& values) const {
    Assert(values.size() == n_comp, ExcDimensionMismatch(values.size(), dim));
    for(unsigned int i = 0; i < n_comp; ++i)
      values[i] = value(p, i);
  }


  // We create an auxiliary class for the term (1/(1 + dt*tau)) in order to avoid loop
  //
  template<int dim, unsigned int n_comp>
  class Raylegh_Aux_Right: public Function<dim> {
  public:
    Raylegh_Aux_Right(const double initial_time = 0.0);

    virtual double value(const Point<dim>&  p,
                         const unsigned int component = 0) const override;

    virtual void vector_value(const Point<dim>& p,
                              Vector<double>&   values) const override;

  private:
    const double x_start; /*--- Starting coordinate of the damping layer ---*/
    const double x_max;   /*--- Ending coordinate of the damping layer ---*/
  };


  template<int dim, unsigned int n_comp>
  Raylegh_Aux_Right<dim, n_comp>::Raylegh_Aux_Right(const double initial_time): Function<dim>(n_comp, initial_time),
                                                                                x_start(30.0), x_max(40.0) {}


  template<int dim, unsigned int n_comp>
  double Raylegh_Aux_Right<dim, n_comp>::value(const Point<dim>& p, const unsigned int component) const {
    (void)component;
    AssertIndexRange(component, n_comp);

    if(p[0] < x_start)
      return 1.0;

    return 1.0/(1.0 + 0.15*std::sin(0.5*numbers::PI*(p[0] - x_start)/(x_max - x_start))*
                           std::sin(0.5*numbers::PI*(p[0] - x_start)/(x_max - x_start)));
  }


  template<int dim, unsigned int n_comp>
  void Raylegh_Aux_Right<dim, n_comp>::vector_value(const Point<dim>& p, Vector<double>& values) const {
    Assert(values.size() == n_comp, ExcDimensionMismatch(values.size(), dim));
    for(unsigned int i = 0; i < n_comp; ++i)
      values[i] = value(p, i);
  }


  // We do the same for the Raylegh damping profile along the left lateral boundary
  //
  template<int dim, unsigned int n_comp>
  class Raylegh_Left: public Function<dim> {
  public:
    Raylegh_Left(const double initial_time = 0.0);

    virtual double value(const Point<dim>&  p,
                         const unsigned int component = 0) const override;

    virtual void vector_value(const Point<dim>& p,
                              Vector<double>&   values) const override;

  private:
    const double x_start; /*--- Starting coordinate of the damping layer ---*/
    const double x_min;   /*--- Ending coordinate of the damping layer ---*/
  };


  template<int dim, unsigned int n_comp>
  Raylegh_Left<dim, n_comp>::Raylegh_Left(const double initial_time): Function<dim>(n_comp, initial_time),
                                                                      x_start(10.0), x_min(0.0) {}


  template<int dim, unsigned int n_comp>
  double Raylegh_Left<dim, n_comp>::value(const Point<dim>& p, const unsigned int component) const {
    (void)component;
    AssertIndexRange(component, n_comp);

    if(p[0] > x_start)
      return 0.0;

    return 0.15*std::sin(0.5*numbers::PI*(p[0] - x_start)/(x_min - x_start))*
                std::sin(0.5*numbers::PI*(p[0] - x_start)/(x_min - x_start));
  }


  template<int dim, unsigned int n_comp>
  void Raylegh_Left<dim, n_comp>::vector_value(const Point<dim>& p, Vector<double>& values) const {
    Assert(values.size() == n_comp, ExcDimensionMismatch(values.size(), dim));
    for(unsigned int i = 0; i < n_comp; ++i)
      values[i] = value(p, i);
  }


  // We create an auxiliary class for the term (1/(1 + dt*tau)) in order to avoid loop
  //
  template<int dim, unsigned int n_comp>
  class Raylegh_Aux_Left: public Function<dim> {
  public:
    Raylegh_Aux_Left(const double initial_time = 0.0);

    virtual double value(const Point<dim>&  p,
                         const unsigned int component = 0) const override;

    virtual void vector_value(const Point<dim>& p,
                              Vector<double>&   values) const override;

  private:
    const double x_start; /*--- Starting coordinate of the damping layer ---*/
    const double x_min;   /*--- Ending coordinate of the damping layer ---*/
  };


  template<int dim, unsigned int n_comp>
  Raylegh_Aux_Left<dim, n_comp>::Raylegh_Aux_Left(const double initial_time): Function<dim>(n_comp, initial_time),
                                                                              x_start(10.0), x_min(0.0) {}


  template<int dim, unsigned int n_comp>
  double Raylegh_Aux_Left<dim, n_comp>::value(const Point<dim>& p, const unsigned int component) const {
    (void)component;
    AssertIndexRange(component, n_comp);

    if(p[0] > x_start)
      return 1.0;

    return 1.0/(1.0 + 0.15*std::sin(0.5*numbers::PI*(p[0] - x_start)/(x_min - x_start))*
                           std::sin(0.5*numbers::PI*(p[0] - x_start)/(x_min - x_start)));
  }


  template<int dim, unsigned int n_comp>
  void Raylegh_Aux_Left<dim, n_comp>::vector_value(const Point<dim>& p, Vector<double>& values) const {
    Assert(values.size() == n_comp, ExcDimensionMismatch(values.size(), dim));
    for(unsigned int i = 0; i < n_comp; ++i)
      values[i] = value(p, i);
  }


  // Now we can focus on mappings from reference element to the physical one
  // using the Gal-Chen. Notice that lenghts are in kilometers becasue of the non-dimensional version.
  //
  template <int dim>
  class PushForward : public Function<dim> {
  public:
    PushForward() : Function<dim>(dim, 0.0), z_max(20.0) {}

    virtual ~PushForward() {};

    virtual double value(const Point<dim>& p, const unsigned int component = 0) const;

  private:
    const double z_max;
  };


  template <int dim>
  double
  PushForward<dim>::value(const Point<dim>& p, const unsigned int component) const {
    // x component
    if(component == 0)
      return p[0];

    // y component
    else if(component == 1) {
      const double hX = 0.45/(1.0 + ((p[0] - 20.0)/1.0)*((p[0] - 20.0)/1.0));
      return p[1] + ((z_max - p[1])/z_max)*hX;
    }
  }


  // We compute now the inverse mapping (from physical to reference).
  // Notice that lenghts are in kilometers becasue of the non-dimensional version.
  //
  template <int dim>
  class PullBack : public Function<dim> {
  public:
    PullBack() : Function<dim>(dim, 0.0), z_max(20.0) {}

    virtual ~PullBack() {};

    virtual double value(const Point<dim>& p, const unsigned int component = 0) const;

  private:
    const double z_max;
  };

  template <int dim>
  double
  PullBack<dim>::value(const Point<dim>& p, const unsigned int component) const {
    // x component
    if(component == 0)
      return p[0];

    // y component
    else if(component == 1) {
      const double hx = 0.45/(1.0 + ((p[0] - 20.0)/1.0)*((p[0] - 20.0)/1.0));
      return z_max*(p[1] - hx)/(z_max - hx);
    }
  }

} // namespace EquationData
