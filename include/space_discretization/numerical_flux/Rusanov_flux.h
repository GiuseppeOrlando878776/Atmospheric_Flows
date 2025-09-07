/*--- Author: Giuseppe Orlando, 2025. ---*/

// @sect{Include files}

// We start by including the necessary header file
//
#include "numerical_flux_base.h"

// @sect{Numerical flux}

// In this namespace, we implement (virtual) signature functions for the numerical flux
//
namespace NumericalFlux {
  using namespace dealii;

  /**
   * We declare now the class for a generic flux for the Euler equations
   */
  template<unsigned dim, typename Number>
  class RusanovFluxEuler: public NumericalFluxEuler<dim, Number> {
  public:
    using value_type = typename Physics::PhysicalFluxEuler<dim, Number>::value_type; /*--- Arythmetic type for this class ---*/

    RusanovFluxEuler() = default;

    RusanovFluxEuler(const value_type Ma_); /*--- Class constructor ---*/

    inline DEAL_II_ALWAYS_INLINE
    Number compute_lambda(const Tensor<1, dim, Number>& u_m,
                          const Tensor<1, dim, Number>& u_p,
                          const Tensor<1, dim, Number>& n_minus) const; /*--- Stabilization parameter of the Rusanov flux ---*/

    // Start with the functions (physical and numerical flux)
    // for the continuity equation
    virtual Number numerical_flux_continuity(const Number& rho_m,
                                             const Tensor<1, dim, Number>& u_m,
                                             const Number& rho_p,
                                             const Tensor<1, dim, Number>& u_p,
                                             const Tensor<1, dim, Number>& n_minus) const override; /*--- Numerical flux continuity equation ---*/

    // Focus now on the functions (physical and numerical flux)
    // for the momentum equation
    virtual Tensor<1, dim, Number> numerical_flux_momentum_explicit(const Number& rho_m,
                                                                    const Tensor<1, dim, Number>& u_m,
                                                                    const Number& rho_p,
                                                                    const Tensor<1, dim, Number>& u_p,
                                                                    const Tensor<1, dim, Number>& n_minus) const override; /*--- Numerical flux momentum equation
                                                                                                                                 for the explicit part ---*/

    virtual Tensor<1, dim, Number> numerical_flux_momentum_implicit(const Number& pres_m,
                                                                    const Number& pres_p,
                                                                    const Tensor<1, dim, Number>& n_minus) const override; /*--- Numerical flux momentum equation
                                                                                                                                 for the implicit part ---*/

    // Focus now on the functions (physical and numerical flux)
    // for the energy equation
    virtual Number numerical_flux_energy_explicit(const Number& rho_m,
                                                  const Tensor<1, dim, Number>& u_m,
                                                  const Number& rho_p,
                                                  const Tensor<1, dim, Number>& u_p,
                                                  const Tensor<1, dim, Number>& n_minus) const override; /*--- Numerical flux energy equation
                                                                                                               for the explicit part ---*/

    virtual Number numerical_flux_energy_implicit(const Tensor<1, dim, Number>& u_m,
                                                  const Number& pres_m,
                                                  const Tensor<1, dim, Number>& u_p,
                                                  const Number& pres_p,
                                                  const Tensor<1, dim, Number>& n_minus) const override; /*--- Numerical flux energy equation
                                                                                                               for the implicit part ---*/

  private:
  };

  // Class constructor
  //
  template<unsigned dim, typename Number>
  RusanovFluxEuler<dim, Number>::RusanovFluxEuler(const value_type Ma_):
    NumericalFluxEuler<dim, Number>(Ma_) {}

  // Stabilization parameter of the Rusanov flux
  //
  template<unsigned dim, typename Number>
  inline DEAL_II_ALWAYS_INLINE
  Number RusanovFluxEuler<dim, Number>::
         compute_lambda(const Tensor<1, dim, Number>& u_m,
                        const Tensor<1, dim, Number>& u_p,
                        const Tensor<1, dim, Number>& n_minus) const {
    return std::max(std::abs(scalar_product(u_m, n_minus)),
                    std::abs(scalar_product(u_p, n_minus)));
  }

  // Numerical flux of the continuity equation
  //
  template<unsigned dim, typename Number>
  Number RusanovFluxEuler<dim, Number>::
         numerical_flux_continuity(const Number& rho_m,
                                   const Tensor<1, dim, Number>& u_m,
                                   const Number& rho_p,
                                   const Tensor<1, dim, Number>& u_p,
                                   const Tensor<1, dim, Number>& n_minus) const {
    /*--- Start with centered contribution ---*/
    const auto avg_flux = static_cast<value_type>(0.5)*
                          (this->physical_flux_continuity(rho_m, u_m) +
                           this->physical_flux_continuity(rho_p, u_p));

    /*--- Focus on stabilization term ---*/
    const auto& lambda   = compute_lambda(u_m, u_p, n_minus);
    const auto& jump_rho = rho_m - rho_p;

    /*--- Return the numerical flux ---*/
    return scalar_product(avg_flux, n_minus) +
           static_cast<value_type>(0.5)*lambda*jump_rho;
  }

  // Numerical flux explicit part momentum equation
  //
  template<unsigned dim, typename Number>
  Tensor<1, dim, Number> RusanovFluxEuler<dim, Number>::
                         numerical_flux_momentum_explicit(const Number& rho_m,
                                                          const Tensor<1, dim, Number>& u_m,
                                                          const Number& rho_p,
                                                          const Tensor<1, dim, Number>& u_p,
                                                          const Tensor<1, dim, Number>& n_minus) const {
    /*--- Start with centered contribution ---*/
    const auto& avg_tensor_product_u = static_cast<value_type>(0.5)*
                                       (outer_product(rho_m*u_m, u_m) +
                                        outer_product(rho_p*u_p, u_p));

    /*--- Focus on stabilization term ---*/
    const auto& lambda    = compute_lambda(u_m, u_p, n_minus);
    const auto& jump_rhou = rho_m*u_m - rho_p*u_p;

    /*--- Return the numerical flux ---*/
    return avg_tensor_product_u*n_minus +
           static_cast<value_type>(0.5)*lambda*jump_rhou;
  }

  // Numerical flux implicit part momentum equation
  //
  template<unsigned dim, typename Number>
  Tensor<1, dim, Number> RusanovFluxEuler<dim, Number>::
                         numerical_flux_momentum_implicit(const Number& pres_m,
                                                          const Number& pres_p,
                                                          const Tensor<1, dim, Number>& n_minus) const {
    return (static_cast<value_type>(0.5)*(pres_m + pres_p))/(this->Ma*this->Ma)*n_minus;
  }

  // Numerical flux explicit part energy equation
  //
  template<unsigned dim, typename Number>
  Number RusanovFluxEuler<dim, Number>::
         numerical_flux_energy_explicit(const Number& rho_m,
                                        const Tensor<1, dim, Number>& u_m,
                                        const Number& rho_p,
                                        const Tensor<1, dim, Number>& u_p,
                                        const Tensor<1, dim, Number>& n_minus) const {
    /*--- Start with centered contribution ---*/
    const auto& avg_kinetic = static_cast<value_type>(0.5)*
                              (static_cast<value_type>(0.5)*scalar_product(u_m, u_m)*rho_m*u_m +
                               static_cast<value_type>(0.5)*scalar_product(u_p, u_p)*rho_p*u_p);

    /*--- Focus on stabilization term ---*/
    const auto& lambda       = compute_lambda(u_m, u_p, n_minus);
    const auto& jump_rho_kin = rho_m*(static_cast<value_type>(0.5)*scalar_product(u_m, u_m)) -
                               rho_p*(static_cast<value_type>(0.5)*scalar_product(u_p, u_p));

    /*--- Return the numerical flux ---*/
    return (this->Ma*this->Ma)*(scalar_product(avg_kinetic, n_minus) +
                                static_cast<value_type>(0.5)*lambda*jump_rho_kin);
  }

  // Numerical flux explicit part energy equation
  //
  template<unsigned dim, typename Number>
  Number RusanovFluxEuler<dim, Number>::
         numerical_flux_energy_implicit(const Tensor<1, dim, Number>& u_m,
                                        const Number& pres_m,
                                        const Tensor<1, dim, Number>& u_p,
                                        const Number& pres_p,
                                        const Tensor<1, dim, Number>& n_minus) const {
    /*--- Start with centered contribution ---*/
    const auto& avg_enthalpy = static_cast<value_type>(0.5)*
                               (static_cast<value_type>(EquationData::Cp_Cv)/
                                (static_cast<value_type>(EquationData::Cp_Cv) - static_cast<value_type>(1.0)))*
                               (pres_m*u_m + pres_p*u_p);

    /*--- Focus on stabilization term ---*/
    const auto& lambda     = compute_lambda(u_m, u_p, n_minus);
    const auto& jump_rho_e = static_cast<value_type>(1.0)/
                             (static_cast<value_type>(EquationData::Cp_Cv) - static_cast<value_type>(1.0))*
                             (pres_m - pres_p);

    /*--- Return the numerical flux ---*/
    return scalar_product(avg_enthalpy, n_minus) +
           static_cast<value_type>(0.5)*lambda*jump_rho_e;
  }

} // namespace NumericalFlux
