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
  static const unsigned int P_INDEX_SYSTEM   = 2;
  static const unsigned int U_INDEX_SYSTEM   = 3;

  static const unsigned int U_INDEX_SYSTEM_TURB = 1;
  static const unsigned int THETA_INDEX_SYSTEM  = 2;

  static const unsigned int U_INDEX_DOF   = 0;
  static const unsigned int P_INDEX_DOF   = 1;
  static const unsigned int RHO_INDEX_DOF = 2;

  static const unsigned int THETA_INDEX_DOF = P_INDEX_DOF;

  /*--- Polynomial degrees. We typically consider the same polynomial degree for all the variables ---*/
  static const unsigned int degree_p   = 2;
  static const unsigned int degree_rho = 2;
  static const unsigned int degree_u   = 2;

  static const double Cp_Cv = 1.4;   /*--- Specific heats ratio ---*/
  static const double R     = 287.0; /*--- Specific gas constant ---*/

  static const double g = 9.81; /*--- Acceleration of gravity ---*/

  static const double x_max = 400000.0; /*--- Extension along horizontal direction ---*/
  static const double z_max = 26000.0;  /*--- Extension along vertical direction ---*/

  static const double z_start       = 20000.0;  /*--- Start of Rayleigh damping for top boundary ---*/
  static const double x_start_left  = 50000.0;  /*--- Start of Rayleigh damping for left boundary ---*/
  static const double x_start_right = 350000.0; /*--- Start of Rayleigh damping for right boundary ---*/

  static const double L_ref = 1000.0;   /*--- Reference length ---*/
  static const double u_ref = 20.0;     /*--- Reference velocity ---*/
  static const double p_ref = 100000.0; /*--- Reference pressure ---*/
  static const double T_ref = 273.0;    /*--- Reference temperature ---*/

  static const double l_mixing = 100.0/L_ref;           /*--- non-dimensional mixing length ---*/
  static const double Fr2      = u_ref*u_ref/(g*L_ref); /*--- squared Froude number ---*/

  static const unsigned int degree_mapping          = 1;                                                             /*--- Mapping degree ---*/
  static const unsigned int extra_quadrature_degree = (degree_mapping == 1) ? 0 : my_ceil(0.5*(degree_mapping - 2)); /*--- Extra accuracy
                                                                                                                           for quadratures ---*/

  /* We declare now the class that describes the initial condition for the velocity.
  */
  template<int dim>
  class Velocity: public Function<dim> {
  public:
    Velocity(const std::string& velocity_profile, const double initial_time = 0.0); /*--- Class constructor ---*/

    virtual double value(const Point<dim>&  p,
                         const unsigned int component = 0) const override; /*--- Evaluation for each component ---*/

    virtual void vector_value(const Point<dim>& p,
                              Vector<double>&   values) const override; /*--- Vector evaluation of the velocity ---*/

  private:
    /*--- Auxiliary vectors to store the velocity profile ---*/
    std::vector<double> z_coords;
    std::vector<double> u;
  };

  // Constructor which relies on the 'Function' constructor. Moreover it reads
  // the data for the T-REX profile
  //
  template<int dim>
  Velocity<dim>::Velocity(const std::string& velocity_profile, const double initial_time): Function<dim>(dim, initial_time)
  {
    /*--- Auxiliary variables to read each line, the words as a vector of strings and each value ---*/
    std::vector<std::string> row;
    std::string line, word;

    /*--- Open the file and read data ---*/
    std::ifstream input_data;

    input_data.open(velocity_profile, std::ios::in);

    /*--- Read line by line ---*/
    while(std::getline(input_data, line)) {
      row.clear();

      if(!line.empty()) {
        std::istringstream iss(line);
        /*--- Read each value, which is separated by a comma ---*/
        while(std::getline(iss, word, ',')) {
          row.push_back(word);
        }

        /*--- Convert strings to double and save data ---*/
        z_coords.push_back(std::stod(row[0]));
        u.push_back(std::stod(row[1]));
      }
    }

    /*--- Close the file ---*/
    input_data.close();
  }

  // Specify the value for each spatial component. This function is overriden.
  //
  template<int dim>
  double Velocity<dim>::value(const Point<dim>& p, const unsigned int component) const {
    AssertIndexRange(component, dim);

    if(component == 0) {
      /*--- Perform a binary search and the linear interpolation ---*/
      double u_interpolated = 0.0;
      if(p[1]*EquationData::L_ref <= z_coords[0]) {
        u_interpolated = u[0];
      }
      else if(p[1]*EquationData::L_ref >= z_coords.back()) {
        u_interpolated = u.back();
      }
      else {
        /*--- Perform binary search to find out the interval of our coordinate ---*/
        unsigned int low  = 0;
        unsigned int high = u.size();
        unsigned int mid  = static_cast<unsigned int>((low + high)/2.0);
        while(p[1]*EquationData::L_ref < z_coords[mid] ||
              p[1]*EquationData::L_ref >= z_coords[mid + 1]) {
          if(p[1]*EquationData::L_ref < z_coords[mid]) {
            high = mid;
          }
          else {
            low = mid;
          }
          mid = static_cast<unsigned int>((low + high)/2.0);
        }

        /*--- Apply linear interpolation ---*/
        u_interpolated = u[mid]
                       + (u[mid + 1] - u[mid])/(z_coords[mid + 1] - z_coords[mid])*
                         (p[1]*EquationData::L_ref - z_coords[mid]);
      }

      return u_interpolated/EquationData::u_ref;
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
    Pressure(const std::string& theta_profile, const double initial_time = 0.0); /*--- Class constructor ---*/

    virtual double value(const Point<dim>&  p,
                         const unsigned int component = 0) const override; /*--- Evalution of the pressure ---*/

  private:
    /*--- Auxiliary vectors to store the potential temperature profile to compute the pressure ---*/
    std::vector<double> z_coords;
    std::vector<double> theta;
  };

  // Constructor which again relies on the 'Function' constructor. Moreover it reads
  // the data for the T-REX profile
  //
  template<int dim>
  Pressure<dim>::Pressure(const std::string& theta_profile, const double initial_time): Function<dim>(1, initial_time)
  {
    /*--- Auxiliary variables to read each line, the words as a vector of strings and each value ---*/
    std::vector<std::string> row;
    std::string line, word;

    /*--- Open the file and read data ---*/
    std::ifstream input_data;

    input_data.open(theta_profile, std::ios::in);

    /*--- Read line by line ---*/
    while(std::getline(input_data, line)) {
      row.clear();

      if(!line.empty()) {
        std::istringstream iss(line);
        /*--- Read each value, which is separated by a comma ---*/
        while(std::getline(iss, word, ',')) {
          row.push_back(word);
        }

        /*--- Convert strings to double and save data ---*/
        z_coords.push_back(std::stod(row[0]));
        theta.push_back(std::stod(row[1]));
      }
    }

    /*--- Close the file ---*/
    input_data.close();
  }

  // Evaluation depending on the spatial coordinates. The input argument 'component'
  // will be unused but it has to be kept to override
  //
  template<int dim>
  double Pressure<dim>::value(const Point<dim>& p, const unsigned int component) const {
    (void)component;
    AssertIndexRange(component, 1);

    const double Gamma = (EquationData::Cp_Cv - 1.0)/EquationData::Cp_Cv;
    const double Cp    = EquationData::R/Gamma;

    /*--- Apply binary search and hydrostatic balance p(z) = p0 (1 - \frac{g}{Cp} \int_0^z \frac{ds}{\theta(s)})^(gamma/(gamma - 1)}
          on the piecewise linear profile to compute the pressure ---*/
    double pres_interpolated = 0.0;
    if(p[1]*EquationData::L_ref <= z_coords[0]) {
      pres_interpolated = 1.0;
    }
    else if(p[1]*EquationData::L_ref >= z_coords.back()) {
      /*--- Compute \int_{0}^{L} 1/theta(s)ds ---*/
      double int_theta_m1 = 0.0;
      for(unsigned int j = 0; j < theta.size() - 1; ++j) {
        const double dz     = z_coords[j + 1] - z_coords[j];
        const double dtheta = theta[j + 1] - theta[j];

        int_theta_m1 += dz/dtheta*std::log(dtheta/theta[j] + 1.0);
      }

      pres_interpolated = std::pow(1.0 - EquationData::g/Cp*int_theta_m1, 1.0/Gamma);
    }
    else {
      /*--- Perform binary search to find out the interval of our coordinate ---*/
      unsigned int low  = 0;
      unsigned int high = theta.size();
      unsigned int mid  = static_cast<unsigned int>((low + high)/2.0);
      while(p[1]*EquationData::L_ref < z_coords[mid] || p[1]*EquationData::L_ref >= z_coords[mid + 1]) {
        if(p[1]*EquationData::L_ref < z_coords[mid]) {
          high = mid;
        }
        else {
          low = mid;
        }
        mid = static_cast<unsigned int>((low + high)/2.0);
      }

      /*--- Compute \int_{0}^{z} 1/theta(s)ds. Since this is a global contribution,
            we first need to some the contributions of the previous intervals ---*/
      double int_theta_m1 = 0.0;
      for(unsigned int j = 0; j < mid; ++j) {
        const double dz     = z_coords[j + 1] - z_coords[j];
        const double dtheta = theta[j + 1] - theta[j];

        const double curr_int_theta_m1 = dz/dtheta*std::log(dtheta/theta[j] + 1.0);

        int_theta_m1 += curr_int_theta_m1;
      }
      /*--- Add contribution of the found interval ---*/
      const double dz     = z_coords[mid + 1] - z_coords[mid];
      const double dtheta = theta[mid + 1] - theta[mid];
      int_theta_m1 += dz/dtheta*std::log(dtheta/(theta[mid]*dz)*(p[1]*EquationData::L_ref - z_coords[mid]) + 1.0);

      /*--- Compute the pressure ---*/
      pres_interpolated = std::pow(1.0 - EquationData::g/Cp*int_theta_m1, 1.0/Gamma);
    }

    return pres_interpolated;
  }


  /* We do the same for the density. Notice that in order to
     get a dimensional version one should multiply the result by p_ref
  */
  template<int dim>
  class Density: public Function<dim> {
  public:
    Density(const std::string& theta_profile, const double initial_time = 0.0); /*--- Class constructor ---*/

    virtual double value(const Point<dim>&  p,
                         const unsigned int component = 0) const override; /*--- Evaluation of the density ---*/

  private:
    /*--- Auxiliary vectors to store the potential temperature profile to compute the density ---*/
    std::vector<double> z_coords;
    std::vector<double> theta;
  };

  // Constructor which again relies on the 'Function' constructor. Moreover it reads
  // the data for the T-REX profile
  //
  template<int dim>
  Density<dim>::Density(const std::string& theta_profile, const double initial_time): Function<dim>(1, initial_time)
  {
    /*--- Auxiliary variables to read each line, the words as a vector of strings and each value ---*/
    std::vector<std::string> row;
    std::string line, word;

    /*--- Open the file and read data ---*/
    std::ifstream input_data;

    input_data.open(theta_profile, std::ios::in);

    /*--- Read line by line ---*/
    while(std::getline(input_data, line)) {
      row.clear();

      if(!line.empty()) {
        std::istringstream iss(line);
        /*--- Read each value, which is separated by a comma ---*/
        while(std::getline(iss, word, ',')) {
          row.push_back(word);
        }

        /*--- Convert strings to double and save data ---*/
        z_coords.push_back(std::stod(row[0]));
        theta.push_back(std::stod(row[1]));
      }
    }

    /*--- Close the file ---*/
    input_data.close();
  }

  // Evaluation depending on the spatial coordinates. The input argument 'component'
  // will be unused but it has to be kept to override
  //
  template<int dim>
  double Density<dim>::value(const Point<dim>& p, const unsigned int component) const {
    (void)component;
    AssertIndexRange(component, 1);

    const double Gamma = (EquationData::Cp_Cv - 1.0)/EquationData::Cp_Cv;
    const double Cp    = EquationData::R/Gamma;

    /*--- Perform a binary search and the linear interpolation ---*/
    double rho_interpolated = 0.0;
    if(p[1]*EquationData::L_ref <= z_coords[0]) {
      rho_interpolated = EquationData::T_ref/theta[0];
    }
    else if(p[1]*EquationData::L_ref >= z_coords.back()) {
      /*--- Compute \int_{0}^{L} 1/theta(s)ds ---*/
      double int_theta_m1 = 0.0;
      for(unsigned int j = 0; j < theta.size() - 1; ++j) {
        const double dz     = z_coords[j + 1] - z_coords[j];
        const double dtheta = theta[j + 1] - theta[j];

        int_theta_m1 += dz/dtheta*std::log(dtheta/theta[j] + 1.0);
      }

      rho_interpolated = EquationData::T_ref/theta.back()*std::pow(1.0 - EquationData::g/Cp*int_theta_m1, 1.0/(EquationData::Cp_Cv - 1.0));
    }
    else {
      /*--- Perform binary search to find out the interval of our coordinate ---*/
      unsigned int low  = 0;
      unsigned int high = theta.size();
      unsigned int mid  = static_cast<unsigned int>((low + high)/2.0);
      while(p[1]*EquationData::L_ref < z_coords[mid] ||
            p[1]*EquationData::L_ref >= z_coords[mid + 1]) {
        if(p[1]*EquationData::L_ref < z_coords[mid]) {
          high = mid;
        }
        else {
          low = mid;
        }
        mid = static_cast<unsigned int>((low + high)/2.0);
      }

      /*--- Apply linear interpolation for the potential temperature ---*/
      const double theta_interpolated = theta[mid]
                                      + (theta[mid + 1] - theta[mid])/(z_coords[mid + 1] - z_coords[mid])*
                                        (p[1]*EquationData::L_ref - z_coords[mid]);

      /*--- Compute \int_{0}^{z} 1/theta(s)ds. Since this is a global contribution,
            we first need to some the contributions of the previous intervals ---*/
      double int_theta_m1 = 0.0;
      for(unsigned int j = 0; j < mid; ++j) {
        const double dz     = z_coords[j + 1] - z_coords[j];
        const double dtheta = theta[j + 1] - theta[j];

        const double curr_int_theta_m1 = dz/dtheta*std::log(dtheta/theta[j] + 1.0);

        int_theta_m1 += curr_int_theta_m1;
      }
      /*--- Add contribution of the found interval ---*/
      const double dz     = z_coords[mid + 1] - z_coords[mid];
      const double dtheta = theta[mid + 1] - theta[mid];
      int_theta_m1 += dz/dtheta*std::log(dtheta/(theta[mid]*dz)*(p[1]*EquationData::L_ref - z_coords[mid]) + 1.0);

      /*--- Compute the density ---*/
      rho_interpolated = EquationData::T_ref/theta_interpolated*
                         std::pow(1.0 - EquationData::g/Cp*int_theta_m1, 1.0/(EquationData::Cp_Cv - 1.0));
    }

    return rho_interpolated;
  }


  /* We focus now on the Rayleigh damping profile along the vertical direction.
     We create a suitable function for that. This function will be either scalar
     or vectorial (for the velocity). That's why the auxiliary template parameter
     n_comp is present.
  */
  template<int dim, unsigned int n_comp>
  class Rayleigh: public Function<dim> {
  public:
    Rayleigh(const double initial_time = 0.0); /*--- Class constructor ---*/

    virtual double value(const Point<dim>&  p,
                         const unsigned int component = 0) const override; /*--- Damping profile evaluation ---*/

    virtual void vector_value(const Point<dim>& p,
                              Vector<double>&   values) const override; /*--- Damping profile vector evaluation for the velocity ---*/

  private:
    const double z_start; /*--- Starting coordinate of the damping layer ---*/
    const double z_max;   /*--- Ending coordinate of the damping layer ---*/
  };

  // Class constructor, which simply calls the parent class constructor
  // and then initialize some data
  //
  template<int dim, unsigned int n_comp>
  Rayleigh<dim, n_comp>::Rayleigh(const double initial_time): Function<dim>(n_comp, initial_time),
                                                              z_start(EquationData::z_start/EquationData::L_ref),
                                                              z_max(EquationData::z_max/EquationData::L_ref) {}

  // Evaluation of Rayleigh damping profile
  //
  template<int dim, unsigned int n_comp>
  double Rayleigh<dim, n_comp>::value(const Point<dim>& p, const unsigned int component) const {
    (void)component;
    AssertIndexRange(component, n_comp);

    if(p[1] < z_start) {
      return 0.0;
    }

    return 0.15*std::sin(0.5*numbers::PI*(p[1] - z_start)/(z_max - z_start))*
                std::sin(0.5*numbers::PI*(p[1] - z_start)/(z_max - z_start)); /*--- Rayleigh profile expression ---*/
  }

  // We need a vector value instance to deal with the velocity or, more in general,
  // if n_comp > 1.
  //
  template<int dim, unsigned int n_comp>
  void Rayleigh<dim, n_comp>::vector_value(const Point<dim>& p, Vector<double>& values) const {
    Assert(values.size() == n_comp, ExcDimensionMismatch(values.size(), dim));

    for(unsigned int i = 0; i < n_comp; ++i) {
      values[i] = value(p, i);
    }
  }


  /* We create an auxiliary class for the term (1/(1 + dt*tau)) in order to avoid loop.
     The template parameter n_comp has the same meaning of the previous class.
  */
  template<int dim, unsigned int n_comp>
  class Rayleigh_Aux: public Function<dim> {
  public:
    Rayleigh_Aux(const double initial_time = 0.0); /*--- Class constructor ---*/

    virtual double value(const Point<dim>&  p,
                         const unsigned int component = 0) const override; /*--- Damping profile evaluation ---*/

    virtual void vector_value(const Point<dim>& p,
                              Vector<double>&   values) const override; /*--- Damping profile vector evaluation for the velocity ---*/

  private:
    const double z_start; /*--- Starting coordinate of the damping layer ---*/
    const double z_max;   /*--- Ending coordinate of the damping layer ---*/
  };

  // Class constructor, which simply calls the parent class constructor
  // and then initialize some data
  //
  template<int dim, unsigned int n_comp>
  Rayleigh_Aux<dim, n_comp>::Rayleigh_Aux(const double initial_time): Function<dim>(n_comp, initial_time),
                                                                      z_start(EquationData::z_start/EquationData::L_ref),
                                                                      z_max(EquationData::z_max/EquationData::L_ref) {}

  // Evaluation of Rayleigh damping profile
  //
  template<int dim, unsigned int n_comp>
  double Rayleigh_Aux<dim, n_comp>::value(const Point<dim>& p, const unsigned int component) const {
    (void)component;
    AssertIndexRange(component, n_comp);

    if(p[1] < z_start) {
      return 1.0;
    }

    return 1.0/(1.0 + 0.15*std::sin(0.5*numbers::PI*(p[1] - z_start)/(z_max - z_start))*
                           std::sin(0.5*numbers::PI*(p[1] - z_start)/(z_max - z_start)));
  }

  // We need a vector value instance to deal with the velocity or, more in general,
  // if n_comp > 1.
  //
  template<int dim, unsigned int n_comp>
  void Rayleigh_Aux<dim, n_comp>::vector_value(const Point<dim>& p, Vector<double>& values) const {
    Assert(values.size() == n_comp, ExcDimensionMismatch(values.size(), dim));

    for(unsigned int i = 0; i < n_comp; ++i) {
      values[i] = value(p, i);
    }
  }


  /* We do the same for the Rayleigh damping profile along the right lateral boundary.
  */
  template<int dim, unsigned int n_comp>
  class Rayleigh_Right: public Function<dim> {
  public:
    Rayleigh_Right(const double initial_time = 0.0); /*--- Class constructor ---*/

    virtual double value(const Point<dim>&  p,
                         const unsigned int component = 0) const override; /*--- Damping profile evaluation ---*/

    virtual void vector_value(const Point<dim>& p,
                              Vector<double>&   values) const override; /*--- Damping profile vector evaluation for the velocity ---*/

  private:
    const double x_start; /*--- Starting coordinate of the damping layer ---*/
    const double x_max;   /*--- Ending coordinate of the damping layer ---*/
  };

  // Class constructor, which simply calls the parent class constructor
  // and then initialize some data
  //
  template<int dim, unsigned int n_comp>
  Rayleigh_Right<dim, n_comp>::Rayleigh_Right(const double initial_time): Function<dim>(n_comp, initial_time),
                                                                          x_start(EquationData::x_start_right/EquationData::L_ref),
                                                                          x_max(EquationData::x_max/EquationData::L_ref) {}

  // Evaluation of Rayleigh damping profile
  //
  template<int dim, unsigned int n_comp>
  double Rayleigh_Right<dim, n_comp>::value(const Point<dim>& p, const unsigned int component) const {
    (void)component;
    AssertIndexRange(component, n_comp);

    if(p[0] < x_start) {
      return 0.0;
    }

    return 0.15*std::sin(0.5*numbers::PI*(p[0] - x_start)/(x_max - x_start))*
                std::sin(0.5*numbers::PI*(p[0] - x_start)/(x_max - x_start));
  }

  // We need a vector value instance to deal with the velocity or, more in general,
  // if n_comp > 1.
  //
  template<int dim, unsigned int n_comp>
  void Rayleigh_Right<dim, n_comp>::vector_value(const Point<dim>& p, Vector<double>& values) const {
    Assert(values.size() == n_comp, ExcDimensionMismatch(values.size(), dim));
    for(unsigned int i = 0; i < n_comp; ++i)
      values[i] = value(p, i);
  }


  /* We create an auxiliary class for the term (1/(1 + dt*tau)) in order to avoid loop.
  */
  template<int dim, unsigned int n_comp>
  class Rayleigh_Aux_Right: public Function<dim> {
  public:
    Rayleigh_Aux_Right(const double initial_time = 0.0); /*--- Class constructor ---*/

    virtual double value(const Point<dim>&  p,
                         const unsigned int component = 0) const override; /*--- Damping profile evaluation ---*/

    virtual void vector_value(const Point<dim>& p,
                              Vector<double>&   values) const override; /*--- Damping profile vector evaluation for the velocity ---*/

  private:
    const double x_start; /*--- Starting coordinate of the damping layer ---*/
    const double x_max;   /*--- Ending coordinate of the damping layer ---*/
  };

  // Class constructor, which simply calls the parent class constructor
  // and then initialize some data
  //
  template<int dim, unsigned int n_comp>
  Rayleigh_Aux_Right<dim, n_comp>::Rayleigh_Aux_Right(const double initial_time): Function<dim>(n_comp, initial_time),
                                                                                  x_start(EquationData::x_start_right/EquationData::L_ref),
                                                                                  x_max(EquationData::x_max/EquationData::L_ref) {}

  // Evaluation of Rayleigh damping profile
  //
  template<int dim, unsigned int n_comp>
  double Rayleigh_Aux_Right<dim, n_comp>::value(const Point<dim>& p, const unsigned int component) const {
    (void)component;
    AssertIndexRange(component, n_comp);

    if(p[0] < x_start) {
      return 1.0;
    }

    return 1.0/(1.0 + 0.15*std::sin(0.5*numbers::PI*(p[0] - x_start)/(x_max - x_start))*
                           std::sin(0.5*numbers::PI*(p[0] - x_start)/(x_max - x_start)));
  }

  // We need a vector value instance to deal with the velocity or, more in general,
  // if n_comp > 1.
  //
  template<int dim, unsigned int n_comp>
  void Rayleigh_Aux_Right<dim, n_comp>::vector_value(const Point<dim>& p, Vector<double>& values) const {
    Assert(values.size() == n_comp, ExcDimensionMismatch(values.size(), dim));

    for(unsigned int i = 0; i < n_comp; ++i) {
      values[i] = value(p, i);
    }
  }


  /* We do the same for the Rayleigh damping profile along the left lateral boundary
  */
  template<int dim, unsigned int n_comp>
  class Rayleigh_Left: public Function<dim> {
  public:
    Rayleigh_Left(const double initial_time = 0.0); /*--- Class constructor ---*/

    virtual double value(const Point<dim>&  p,
                         const unsigned int component = 0) const override; /*--- Damping profile evaluation ---*/

    virtual void vector_value(const Point<dim>& p,
                              Vector<double>&   values) const override; /*--- Damping profile vector evaluation for the velocity ---*/

  private:
    const double x_start; /*--- Starting coordinate of the damping layer ---*/
    const double x_min;   /*--- Ending coordinate of the damping layer ---*/
  };

  // Class constructor, which simply calls the parent class constructor
  // and then initialize some data
  //
  template<int dim, unsigned int n_comp>
  Rayleigh_Left<dim, n_comp>::Rayleigh_Left(const double initial_time): Function<dim>(n_comp, initial_time),
                                                                        x_start(EquationData::x_start_left/EquationData::L_ref),
                                                                        x_min(0.0) {}

  // Evaluation of Rayleigh damping profile
  //
  template<int dim, unsigned int n_comp>
  double Rayleigh_Left<dim, n_comp>::value(const Point<dim>& p, const unsigned int component) const {
    (void)component;
    AssertIndexRange(component, n_comp);

    if(p[0] > x_start) {
      return 0.0;
    }

    return 0.15*std::sin(0.5*numbers::PI*(p[0] - x_start)/(x_min - x_start))*
                std::sin(0.5*numbers::PI*(p[0] - x_start)/(x_min - x_start));
  }

  // We need a vector value instance to deal with the velocity or, more in general,
  // if n_comp > 1.
  //
  template<int dim, unsigned int n_comp>
  void Rayleigh_Left<dim, n_comp>::vector_value(const Point<dim>& p, Vector<double>& values) const {
    Assert(values.size() == n_comp, ExcDimensionMismatch(values.size(), dim));

    for(unsigned int i = 0; i < n_comp; ++i) {
      values[i] = value(p, i);
    }
  }


  /* We create an auxiliary class for the term (1/(1 + dt*tau)) in order to avoid loop.
  */
  template<int dim, unsigned int n_comp>
  class Rayleigh_Aux_Left: public Function<dim> {
  public:
    Rayleigh_Aux_Left(const double initial_time = 0.0); /*--- Class constructor ---*/

    virtual double value(const Point<dim>&  p,
                         const unsigned int component = 0) const override; /*--- Damping profile evaluation ---*/

    virtual void vector_value(const Point<dim>& p,
                              Vector<double>&   values) const override; /*--- Damping profile vector evaluation for the velocity ---*/

  private:
    const double x_start; /*--- Starting coordinate of the damping layer ---*/
    const double x_min;   /*--- Ending coordinate of the damping layer ---*/
  };

  // Class constructor, which simply calls the parent class constructor
  // and then initialize some data
  //
  template<int dim, unsigned int n_comp>
  Rayleigh_Aux_Left<dim, n_comp>::Rayleigh_Aux_Left(const double initial_time): Function<dim>(n_comp, initial_time),
                                                                                x_start(EquationData::x_start_left/EquationData::L_ref),
                                                                                x_min(0.0) {}

  // Evaluation of Rayleigh damping profile
  //
  template<int dim, unsigned int n_comp>
  double Rayleigh_Aux_Left<dim, n_comp>::value(const Point<dim>& p, const unsigned int component) const {
    (void)component;
    AssertIndexRange(component, n_comp);

    if(p[0] > x_start) {
      return 1.0;
    }

    return 1.0/(1.0 + 0.15*std::sin(0.5*numbers::PI*(p[0] - x_start)/(x_min - x_start))*
                           std::sin(0.5*numbers::PI*(p[0] - x_start)/(x_min - x_start)));
  }

  // We need a vector value instance to deal with the velocity or, more in general,
  // if n_comp > 1.
  //
  template<int dim, unsigned int n_comp>
  void Rayleigh_Aux_Left<dim, n_comp>::vector_value(const Point<dim>& p, Vector<double>& values) const {
    Assert(values.size() == n_comp, ExcDimensionMismatch(values.size(), dim));

    for(unsigned int i = 0; i < n_comp; ++i) {
      values[i] = value(p, i);
    }
  }


  /* Now we can focus on mappings from reference element to the physical one
     using the Gal-Chen. Notice that lenghts are in kilometers becasue of
     the non-dimensional version (the characteristic length is assumed 1 km).
     For this purpose, since we are not using anymore an analytical function,
     I prefer creating my own manifold starting from ChartManifold
  */
  template<int dim, int spacedim = dim, int chartdim = dim>
  class TREX_Manifold: public ChartManifold<dim, spacedim, chartdim> {
  public:
    TREX_Manifold(const std::string& profile_input_file); /*--- Class constructor ---*/

    virtual Point<spacedim> push_forward(const Point<chartdim>& chart_point) const override; /*--- Map between reference and physical ---*/

    virtual Point<chartdim> pull_back(const Point<spacedim>& space_point) const override; /*--- Inverse map ---*/

    virtual std::unique_ptr<Manifold<dim, spacedim>> clone() const override; /*--- Pure virtual function to be overriden ---*/

  private:
    const double z_max; /*--- Non-dimensional height of the domain ---*/

    /*--- Auxiliary vectors to store the profile data ---*/
    std::vector<double> x_coords;
    std::vector<double> heights_coords;
  };

  // Class constructor. We also read the (x,h(x)) coordinates
  //
  template<int dim, int spacedim, int chartdim>
  TREX_Manifold<dim, spacedim, chartdim>::TREX_Manifold(const std::string& profile_input_file) : ChartManifold<dim, spacedim, chartdim>(),
                                                                                                 z_max(EquationData::z_max/EquationData::L_ref)
  {
    /*--- Auxiliary variables to read each line, the words as a vector of strings and each value ---*/
    std::vector<std::string> row;
    std::string line, word;

    /*--- Open the file and read data ---*/
    std::ifstream input_data;

    input_data.open(profile_input_file, std::ios::in);

    /*--- Read line by line ---*/
    while(std::getline(input_data, line)) {
      row.clear();

      if(!line.empty()) {
        std::istringstream iss(line);
        /*--- Read each value, which is separated by a comma ---*/
        while(std::getline(iss, word, ',')) {
          row.push_back(word);
        }

        /*--- Convert strings to double and save data ---*/
        x_coords.push_back(std::stod(row[0]));
        heights_coords.push_back(std::stod(row[1]));
      }
    }

    /*--- Close the file ---*/
    input_data.close();
  }

  // Mapping between reference and physical element
  //
  template<int dim, int spacedim, int chartdim>
  Point<spacedim> TREX_Manifold<dim, spacedim, chartdim>::push_forward(const Point<chartdim>& chart_point) const {
    /*--- Compute interpolated value with a binary search ---*/
    double hX;
    if(chart_point[0] <= x_coords[0]) {
      hX = heights_coords[0];
    }
    else if(chart_point[0] >= x_coords.back()) {
      hX = heights_coords.back();
    }
    else {
      /*--- Perform a binary search to verify in which interval we are ---*/
      unsigned int low  = 0;
      unsigned int high = heights_coords.size();
      unsigned int mid  = static_cast<unsigned int>((low + high)/2.0);
      while(chart_point[0] < x_coords[mid] || chart_point[0] >= x_coords[mid + 1]) {
        if(chart_point[0] < x_coords[mid]) {
          high = mid;
        }
        else {
          low = mid;
        }
        mid = static_cast<unsigned int>((low + high)/2.0);
      }

      /*--- Apply spline interpolation ---*/
      hX = heights_coords[mid]
         + (heights_coords[mid + 1] - heights_coords[mid])/(x_coords[mid + 1] - x_coords[mid])*
           (chart_point[0] - x_coords[mid]);
    }

    Point<spacedim> res;

    res[0] = chart_point[0];
    res[1] = chart_point[1] + ((z_max - chart_point[1])/z_max)*hX;

    return res;
  }

  // Inverse mapping (from physical to reference)
  //
  template<int dim, int spacedim, int chartdim>
  Point<chartdim> TREX_Manifold<dim, spacedim, chartdim>::pull_back(const Point<spacedim>& space_point) const {
    /*--- Compute interpolated value with a binary search ---*/
    double hx;
    if(space_point[0] <= x_coords[0]) {
      hx = heights_coords[0];
    }
    else if(space_point[0] >= x_coords.back()) {
      hx = heights_coords.back();
    }
    else {
      /*--- Perform a binary search to verify in which interval we are ---*/
      unsigned int low  = 0;
      unsigned int high = heights_coords.size();
      unsigned int mid  = static_cast<unsigned int>((low + high)/2.0);
      while(space_point[0] < x_coords[mid] || space_point[0] >= x_coords[mid + 1]) {
        if(space_point[0] < x_coords[mid]) {
          high = mid;
        }
        else {
          low = mid;
        }
        mid = static_cast<unsigned int>((low + high)/2.0);
      }

      /*--- Apply spline interpolation ---*/
      hx = heights_coords[mid]
         + (heights_coords[mid + 1] - heights_coords[mid])/(x_coords[mid + 1] - x_coords[mid])*
           (space_point[0] - x_coords[mid]);
    }

    Point<chartdim> res;

    res[0] = space_point[0];
    res[1] = z_max*(space_point[1] - hx)/(z_max - hx);

    return res;
  }

  // Clone function necessary because pure virtual function
  //
  template<int dim, int spacedim, int chartdim>
  std::unique_ptr<Manifold<dim, spacedim>> TREX_Manifold<dim, spacedim, chartdim>::clone() const {
    return std::make_unique<TREX_Manifold<dim, spacedim, chartdim>>(*this);
  }

} // namespace EquationData
