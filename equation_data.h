/*--- Author: Giuseppe Orlando, 2024. ---*/

// @sect{Include files}

// We start by including the necessary deal.II header files and some C++
// related ones.
#include <deal.II/base/point.h>
#include <deal.II/base/function.h>

#include <cmath>

constexpr int my_ceil(double num) {
  return (static_cast<float>(static_cast<int>(num)) == num) ?
          static_cast<int>(num) :
          static_cast<int>(num) + ((num > 0) ? 1 : 0);
}

// @sect{Equation data}

// In this namespace, we declare the initial background conditions,
// the Rayleigh dumping profiles and the mapping between reference and physical
// elements using the Gal-Chen mapping.
//
namespace EquationData {
  using namespace dealii;

  static const unsigned int n_stages = 3; /*--- Number of stages of the IMEX scheme ---*/

  static const unsigned int n_vars = 3; /*--- Number of variables for which we solve a linear system ---*/

  /*--- Define axuliary indices related to the dof hadlers order and to linear systems under consideration ---*/
  static const unsigned int RHO_INDEX_SYSTEM = 1;
  static const unsigned int P_INDEX_SYSTEM = 2;
  static const unsigned int U_INDEX_SYSTEM = 3;

  static const unsigned int U_INDEX_DOF = 0;
  static const unsigned int P_INDEX_DOF = 1;
  static const unsigned int RHO_INDEX_DOF = 2;

  /*--- Polynomial degrees. We typically consider the same polynomial degree for all the variables ---*/
  static const unsigned int degree_p   = 4;
  static const unsigned int degree_rho = 4;
  static const unsigned int degree_u   = 4;

  static const double Cp_Cv = 1.4;   /*--- Specific heats ratio ---*/
  static const double R     = 287.0; /*--- Specific gas constant ---*/

  static const double g = 9.81; /*--- Acceleration of gravity ---*/
  static const double N = 0.01; /*--- Buoyancy frequency ---*/

  static const double xc = 100000.0; /*--- Center of the perturbation ---*/
  static const double ac = 5000.0;   /*--- Width of the pertrubation ---*/
  static const double H  = 10000.0;  /*--- Height of the pertrubaation ---*/

  static const double x_max = 300000.0; /*--- Extension along horizontal direction ---*/
  static const double z_max = 10000.0; /*--- Extension along vertical direction ---*/

  static const double L_ref   = 1000.0;          /*--- Reference length ---*/
  static const double u_ref   = 20.0;            /*--- Reference velocity ---*/
  static const double p_ref   = 100000.0;        /*--- Reference pressure ---*/
  static const double T_ref   = 300.0;
  static const double rho_ref = p_ref/(R*T_ref); /*--- Reference density ---*/

  static const unsigned int degree_mapping          = 1;                                                             /*--- Mapping degree ---*/
  static const unsigned int extra_quadrature_degree = (degree_mapping == 1) ? 0 : my_ceil(0.5*(degree_mapping - 2)); /*--- Extra accuracy
                                                                                                                           for quadratures ---*/

  /* We declare now the class that describes the initial condition for the velocity.
  */
  template<int dim>
  class Velocity: public Function<dim> {
  public:
    Velocity(const double initial_time = 0.0); /*--- Class constructor ---*/

    virtual double value(const Point<dim>&  p,
                         const unsigned int component = 0) const override; /*--- Evaluation for each component ---*/

    virtual void vector_value(const Point<dim>& p,
                              Vector<double>&   values) const override; /*--- Vector evaluation of the velocity ---*/
  };

  // Constructor which relies on the 'Function' constructor.
  //
  template<int dim>
  Velocity<dim>::Velocity(const double initial_time): Function<dim>(dim, initial_time) {}

  // Specify the value for each spatial component. This function is overriden.
  //
  template<int dim>
  double Velocity<dim>::value(const Point<dim>& p, const unsigned int component) const {
    AssertIndexRange(component, dim);

    if(component == 0) {
      return 1.0;
    }
    else {
      return 0.0;
    }
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


  /* We do the same for the pressure. Notice that in order to
     get a dimensional version one should multiply the result by p_ref
  */
  template<int dim>
  class Pressure: public Function<dim> {
  public:
    Pressure(const double initial_time = 0.0); /*--- Class constructor ---*/

    virtual double value(const Point<dim>&  p,
                         const unsigned int component = 0) const override; /*--- Evalution of the pressure ---*/
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

    const double Gamma  = (EquationData::Cp_Cv - 1.0)/EquationData::Cp_Cv;

    const double pi_bar = 1.0 - EquationData::g*EquationData::g/(EquationData::N*EquationData::N)*Gamma*EquationData::rho_ref/EquationData::p_ref*
                                (1.0 - std::exp(-EquationData::N*EquationData::N/EquationData::g*p[1]*EquationData::L_ref));

    return std::pow(pi_bar, 1.0/Gamma);
  }


  /* We do the same for the density. Notice that in order to
     get a dimensional version one should multiply the result by rho_ref
  */
  template<int dim>
  class Density: public Function<dim> {
  public:
    Density(const double initial_time = 0.0); /*--- Class constructor ---*/

    virtual double value(const Point<dim>&  p,
                         const unsigned int component = 0) const override; /*--- Evaluation of the density ---*/
  };

  // Constructor which again relies on the 'Function' constructor.
  //
  template<int dim>
  Density<dim>::Density(const double initial_time): Function<dim>(1, initial_time) {}

  // Evaluation depending on the spatial coordinates. The input argument 'component'
  // will be unused but it has to be kept to override
  //
  template<int dim>
  double Density<dim>::value(const Point<dim>& p, const unsigned int component) const {
    (void)component;
    AssertIndexRange(component, 1);

    const double Gamma     = (EquationData::Cp_Cv - 1.0)/EquationData::Cp_Cv;

    const double pi_bar    = 1.0
                           - EquationData::g*EquationData::g/(EquationData::N*EquationData::N)*Gamma*EquationData::rho_ref/EquationData::p_ref*
                             (1.0 - std::exp(-EquationData::N*EquationData::N/EquationData::g*p[1]*EquationData::L_ref));

    const double theta_bar   = EquationData::T_ref*std::exp(EquationData::N*EquationData::N/EquationData::g*p[1]*EquationData::L_ref);
    const double theta_prime = 0.01*std::sin(numbers::PI*p[1]*EquationData::L_ref/EquationData::H)/
                                    (1.0 + ((p[0]*EquationData::L_ref - EquationData::xc)/EquationData::ac)*
                                           ((p[0]*EquationData::L_ref - EquationData::xc)/EquationData::ac));
    const double theta       = theta_bar + theta_prime;

    return EquationData::T_ref/theta*std::pow(pi_bar, 1.0/(EquationData::Cp_Cv - 1.0));
  }

  /* We do the same for the backgroudn density. Notice that in order to
     get a dimensional version one should multiply the result by rho_ref
  */
  template<int dim>
  class Density_Bar: public Function<dim> {
  public:
    Density_Bar(const double initial_time = 0.0); /*--- Class constructor ---*/

    virtual double value(const Point<dim>&  p,
                         const unsigned int component = 0) const override; /*--- Evaluation of the density ---*/
  };

  // Constructor which again relies on the 'Function' constructor.
  //
  template<int dim>
  Density_Bar<dim>::Density_Bar(const double initial_time): Function<dim>(1, initial_time) {}

  // Evaluation depending on the spatial coordinates. The input argument 'component'
  // will be unused but it has to be kept to override
  //
  template<int dim>
  double Density_Bar<dim>::value(const Point<dim>& p, const unsigned int component) const {
    (void)component;
    AssertIndexRange(component, 1);

    const double Gamma     = (EquationData::Cp_Cv - 1.0)/EquationData::Cp_Cv;

    const double pi_bar    = 1.0
                           - EquationData::g*EquationData::g/(EquationData::N*EquationData::N)*Gamma*EquationData::rho_ref/EquationData::p_ref*
                             (1.0 - std::exp(-EquationData::N*EquationData::N/EquationData::g*p[1]*EquationData::L_ref));

    const double theta_bar   = EquationData::T_ref*std::exp(EquationData::N*EquationData::N/EquationData::g*p[1]*EquationData::L_ref);

    return EquationData::T_ref/theta_bar*std::pow(pi_bar, 1.0/(EquationData::Cp_Cv - 1.0));
  }

} // namespace EquationData
