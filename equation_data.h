/*--- Author: Giuseppe Orlando, 2023. ---*/

// @sect{Include files}

// We start by including the necessary deal.II header files and some C++
// related ones.
#include <deal.II/base/point.h>
#include <deal.II/base/function.h>

#include <cmath>

// @sect{Equation data}

// In this namespace, we declare the initial background conditions.
//
namespace EquationData {
  using namespace dealii;

  /*--- Polynomial degrees. We typically consider the same polynomial degree for all the variables ---*/
  static const unsigned int degree_T   = 1;
  static const unsigned int degree_rho = 1;
  static const unsigned int degree_u   = 1;

  static const double Cp_Cv = 1.4;   /*--- Specific heats ratio ---*/
  static const double R     = 287.0; /*--- Specific gas constant ---*/

  static const double a     = 6.37122e6; /*--- Radius of Earth ---*/

  static const unsigned int degree_mapping = 2;

  // We start with the density (since it is a scalar field) we can derive
  // directly from the deal.II built-in class Function. Notice that in order to
  // get a dimensional version one should multiply the result by rho_ref.
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

    const double N           = 0.01;    /*--- Buyoancy frequency ---*/
    const double g           = 9.80616; /*--- Acceleration of gravity ---*/

    const double T_ref       = 300.0;    /*--- Reference temperature ---*/
    const double p_ref       = 100000.0; /*--- Reference pressure ---*/
    const double rho_ref     = p_ref/(EquationData::R*T_ref); /*--- Reference density ---*/
    const double Gamma       = (EquationData::Cp_Cv - 1.0)/EquationData::Cp_Cv;

    const double radius      = std::sqrt(p[0]*p[0] + p[1]*p[1] + p[2]*p[2]);
    const double z           = radius - EquationData::a;
    const double pi_bar      = 1.0 - g*g/(N*N)*Gamma*rho_ref/p_ref*(1.0 - std::exp(-N*N/g*z));

    const double theta_bar   = T_ref*std::exp(N*N/g*z);

    const double phi         = std::asin(p[2]/radius); // latitude (asin returns range -pi/2 to pi/2, which is ok for geographic applications)
    const double lambda      = std::atan2(p[1], p[0]) + numbers::PI; // longitude (atan2 returns range -pi to pi,
                                                                     //            which is not ok for geographic applications)
    const double lambda_c    = numbers::PI;
    const double phi_c       = 0.0;
    const double r           = radius*std::acos(std::sin(phi_c)*std::sin(phi) +
                                                std::cos(phi_c)*std::cos(phi)*std::cos(lambda - lambda_c)); // great circle distance
    const double R           = EquationData::a/3.0;
    const double s           = 0.5*(1.0 + std::cos(numbers::PI*r/R))*(r < R);
    const double dtheta      = 10.0;
    const double Lz          = 20000.0;
    const double theta_prime = dtheta*s*std::sin(2.0*numbers::PI*z/Lz);

    const double theta       = theta_bar + theta_prime;

    return rho_ref*T_ref/theta*std::pow(pi_bar, 1.0/(EquationData::Cp_Cv - 1.0));
  }


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


  // Constructor which simply relies on the 'Function' constructor.
  //
  template<int dim>
  Velocity<dim>::Velocity(const double initial_time): Function<dim>(dim, initial_time) {}

  // Specify the value for each spatial component. This function is overriden.
  //
  template<int dim>
  double Velocity<dim>::value(const Point<dim>& p, const unsigned int component) const {
    AssertIndexRange(component, dim);

    return 0.0;
  }

  // Put together for a vector evalutation of the velocity.
  //
  template<int dim>
  void Velocity<dim>::vector_value(const Point<dim>& p, Vector<double>& values) const {
    Assert(values.size() == dim, ExcDimensionMismatch(values.size(), dim));

    for(unsigned int i = 0; i < dim; ++i) {
      values[i] = value(p, i);
    }
  }


  // We do the same for the pressure (since it is a scalar field) we can derive
  // directly from the deal.II built-in class Function. Notice that in order to
  // get a dimensional version one should multiply the result by p_ref.
  //
  template<int dim>
  class Pressure: public Function<dim> {
  public:
    Pressure(const double initial_time = 0.0);

    virtual double value(const Point<dim>&  p,
                         const unsigned int component = 0) const override;
  };


  // Constructor which again relies on the 'Function' constructor.
  //
  template<int dim>
  Pressure<dim>::Pressure(const double initial_time): Function<dim>(1, initial_time) {}

  // Evaluation depending on the spatial coordinates. The input argument 'component'
  // will be unused but it has to be kept to override
  //
  template<int dim>
  double Pressure<dim>::value(const Point<dim>& p, const unsigned int component) const {
    (void)component;
    AssertIndexRange(component, 1);

    const double N       = 0.01;    /*--- Buyoancy frequency ---*/
    const double g       = 9.80616; /*--- Acceleration of gravity ---*/

    const double T_ref   = 300.0;    /*--- Reference temperature ---*/
    const double p_ref   = 100000.0; /*--- Reference pressure ---*/
    const double rho_ref = p_ref/(EquationData::R*T_ref); /*--- Reference density ---*/
    const double Gamma   = (EquationData::Cp_Cv - 1.0)/EquationData::Cp_Cv;

    const double radius  = std::sqrt(p[0]*p[0] + p[1]*p[1] + p[2]*p[2]);
    const double z       = radius - EquationData::a;

    const double pi_bar  = 1.0 - g*g/(N*N)*Gamma*rho_ref/p_ref*(1.0 - std::exp(-N*N/g*z));

    return p_ref*std::pow(pi_bar, 1.0/Gamma);
  }

} // namespace EquationData
