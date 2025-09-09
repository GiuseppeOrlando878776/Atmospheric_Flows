/*--- Author: Giuseppe Orlando, 2025. ---*/
#pragma once

// @sect{Include files}

// We start by including the necessary header files
//
#include <vector>

// @sect{Time stepping routines}

// In this namespace, we declare a general interface for Runge-Kutta method
//
namespace TimeStepping {
  using namespace dealii;

  /**
   * We declare now the class for a generic Runge-Kutta method
   */
  template<typename T = double>
  class RungeKutta {
  public:
    RungeKutta() = default; /*--- Default class constructor. This should never be used ---*/

    RungeKutta(const std::vector<std::vector<T>>& a_,
               const std::vector<T>& b_); /*--- Class constructor to set the coefficients of the method following the Butcher tableau representation ---*/

    inline unsigned get_n_stages() const; /*--- Get the number of stages of the method ---*/

    void get_coefficients(std::vector<std::vector<T>>& a_,
                          std::vector<T>&              b_) const; /*--- Get the coefficents in the Butcher tableau representation ---*/

    void get_coefficients(std::vector<std::vector<T>>& a_,
                          std::vector<T>&              b_,
                          std::vector<T>&              c_) const; /*--- Get the coefficents in the Butcher tableau representation ---*/

  protected:
    const unsigned int n_stages; /*--- Number of stages ---*/

    const std::vector<std::vector<T>> a; /*--- Coefficients of the method (Butcher tableau) ---*/

    const std::vector<T> b; /*--- Weigths of the Runge-Kutta method ---*/

    std::vector<T> c; /*--- Nodes of the Runge-Kutta method ---*/
  };

  // Class constructor
  //
  template<typename T>
  RungeKutta<T>::RungeKutta(const std::vector<std::vector<T>>& a_,
                            const std::vector<T>&              b_):
    n_stages(b_.size()), a(a_), b(b_)
    {
      /*--- Initialize the nodes using the classical $rule c_{i} = \sum_{j}a_{ij}$.
            This is not mandatory, but ALL the methods obey to it ---*/
      c.resize(n_stages);
      std::fill(c.begin(), c.end(), static_cast<T>(0.0));
      for(std::size_t i = 0; i < n_stages; ++i) {
        for(std::size_t j = 0; j < n_stages; ++j) {
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
