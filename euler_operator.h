/* Author: Giuseppe Orlando, 2025. */

// @sect{Include files}

// We start by including all the necessary deal.II header files
//
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>
#include <deal.II/matrix_free/fe_evaluation.h>

#include <deal.II/meshworker/mesh_loop.h>

/*--- Include headers related to the problem of interest ---*/
#include "include/io/runtime_parameters.h"
#include "include/equation_data.h"
#include "include/space_discretization/numerical_flux/Rusanov_flux.h"

// This is the class that implements the discretization
//
namespace Atmospheric_Flow {
  using namespace dealii;

  // @sect{ <code>EULEROperator::EULEROperator</code> }
  //
  template<unsigned dim,
           unsigned fe_degree_u, unsigned fe_degree_rho, unsigned fe_degree_p,
           unsigned n_q_points_1d, unsigned n_q_points_1d_boundary,
           typename Vec>
  class EULEROperator: public MatrixFreeOperators::Base<dim, Vec> {
  public:
    using Number = typename Vec::value_type;

    EULEROperator(); /*--- Default constructor ---*/

    EULEROperator(RunTimeParameters::Data_Storage<Number>& data); /*--- Constructor with some input related data ---*/

    void set_dt(const Number time_step); /*--- Setter of the time-step. This is useful both for multigrid purposes and also
                                               in case of modifications of the time step. ---*/

    inline DEAL_II_ALWAYS_INLINE
    Number get_Mach() const; /*--- Getter of the Mach number. This is useful for debugging purpose. ---*/

    inline DEAL_II_ALWAYS_INLINE
    Number get_Froude() const; /*--- Getter of the Froude number. This is useful for debugging purpose. ---*/

    inline DEAL_II_ALWAYS_INLINE
    void set_IMEX_stage(const unsigned stage); /*--- Setter of the IMEX stage. ---*/

    inline DEAL_II_ALWAYS_INLINE
    void set_Euler_stage(const unsigned stage); /*--- Setter of the equation currently under solution. ---*/

    inline DEAL_II_ALWAYS_INLINE
    unsigned get_Euler_stage() const; /*--- Getter of the equation currently under solution. ---*/

    void set_rho_for_fixed(const Vec& src); /*--- Setter of the current density. This is for the assembling of the bilinear forms
                                                  where only one source vector can be passed in input. ---*/

    void set_pres_fixed(const Vec& src); /*--- Setter of the current pressure. This is for the assembling of the bilinear forms
                                               where only one source vector can be passed in input. ---*/

    void vmult_rhs_density(Vec& dst, const std::vector<Vec>& src) const; /*--- Auxiliary function to assemble the rhs
                                                                               of the continuity equation. ---*/

    void vmult_rhs_momentum(Vec& dst, const std::vector<Vec>& src) const;  /*--- Auxiliary function to assemble the rhs
                                                                                 of the momentum equation. ---*/

    void vmult_rhs_energy(Vec& dst, const std::vector<Vec>& src) const;  /*--- Auxiliary function to assemble the rhs
                                                                               of the energy equation. ---*/

    void vmult_pressure(Vec& dst, const Vec& src) const; /*--- Action of matrix 'B'. ---*/

    void vmult_enthalpy(Vec& dst, const Vec& src) const; /*--- Action of matrix 'C'. ---*/

    virtual void compute_diagonal() override; /*--- Overriden function to compute the diagonal. ---*/

  protected:
    /*--- Define typedef for sake of readability and convenience ----*/
    using FEEvaluation_rho  = FEEvaluation<dim, fe_degree_rho, n_q_points_1d, 1, Number>;
    using FEEvaluation_u    = FEEvaluation<dim, fe_degree_u, n_q_points_1d, dim, Number>;
    using FEEvaluation_pres = FEEvaluation<dim, fe_degree_p, n_q_points_1d, 1, Number>;

    using FEFaceEvaluation_rho  = FEFaceEvaluation<dim, fe_degree_rho, n_q_points_1d, 1, Number>;
    using FEFaceEvaluation_u    = FEFaceEvaluation<dim, fe_degree_u, n_q_points_1d, dim, Number>;
    using FEFaceEvaluation_pres = FEFaceEvaluation<dim, fe_degree_p, n_q_points_1d, 1, Number>;

    using FEFaceEvaluation_rho_boundary  = FEFaceEvaluation<dim, fe_degree_rho, n_q_points_1d_boundary, 1, Number>;
    using FEFaceEvaluation_u_boundary    = FEFaceEvaluation<dim, fe_degree_u, n_q_points_1d_boundary, dim, Number>;
    using FEFaceEvaluation_pres_boundary = FEFaceEvaluation<dim, fe_degree_p, n_q_points_1d_boundary, 1, Number>;

    Number Ma; /*--- Mach number. ---*/
    Number Fr; /*--- Froude number. ---*/

    Number dt; /*--- Time step. ---*/

    const Number gamma; /*--- TR-BDF2 (i.e. implicit part) parameter. ---*/
    /*--- The following variables follow the classical Butcher tableaux notation ---*/
    const Number a21;
    const Number a31;
    const Number a32;

    const Number a21_tilde;
    const Number a22_tilde;
    const Number a31_tilde;
    const Number a32_tilde;
    const Number a33_tilde;

    const Number b1;
    const Number b2;
    const Number b3;

    unsigned IMEX_stage;          /*--- Flag for the IMEX stage ---*/
    mutable unsigned Euler_stage; /*--- Flag for the equation actually considered ---*/

    virtual void apply_add(Vec& dst, const Vec& src) const override; /*--- Overriden function which actually assembles the
                                                                           bilinear forms ---*/

  private:
    Vec rho_for_fixed,
        pres_fixed;

    /*--- Auxiliary function for the numerical flux ---*/
    NumericalFlux::RusanovFluxEuler<dim, VectorizedArray<Number>> num_flux;

    /*--- Assembler functions for the rhs related to the continuity equation. Here, and also in the following,
          we distinguish between the contribution for cells, faces and boundary. ---*/
    void assemble_rhs_cell_term_density(const MatrixFree<dim, Number>&       data,
                                        Vec&                                 dst,
                                        const std::vector<Vec>&              src,
                                        const std::pair<unsigned, unsigned>& cell_range) const;
    void assemble_rhs_face_term_density(const MatrixFree<dim, Number>&       data,
                                        Vec&                                 dst,
                                        const std::vector<Vec>&              src,
                                        const std::pair<unsigned, unsigned>& face_range) const;
    void assemble_rhs_boundary_term_density(const MatrixFree<dim, Number>&       data,
                                            Vec&                                 dst,
                                            const std::vector<Vec>&              src,
                                            const std::pair<unsigned, unsigned>& face_range) const {}
                                               /*-- No flux, so no contribution from this function ---*/

    /*--- Assembler function related to the bilinear form of the continuity equation. Only cell contribution is present,
          since, basically, we end up with a mass matrix. ---*/
    void assemble_cell_term_density(const MatrixFree<dim, Number>&       data,
                                    Vec&                                 dst,
                                    const Vec&                           src,
                                    const std::pair<unsigned, unsigned>& cell_range) const;

    /*--- Assembler functions for the rhs related to the momentum equation. ---*/
    void assemble_rhs_cell_term_momentum(const MatrixFree<dim, Number>&       data,
                                         Vec&                                 dst,
                                         const std::vector<Vec>&              src,
                                         const std::pair<unsigned, unsigned>& cell_range) const;
    void assemble_rhs_face_term_momentum(const MatrixFree<dim, Number>&       data,
                                         Vec&                                 dst,
                                         const std::vector<Vec>&              src,
                                         const std::pair<unsigned, unsigned>& face_range) const;
    void assemble_rhs_boundary_term_momentum(const MatrixFree<dim, Number>&       data,
                                             Vec&                                 dst,
                                             const std::vector<Vec>&              src,
                                             const std::pair<unsigned, unsigned>& face_range) const;

    /*--- Assembler function for the 'A' matrix. ---*/
    void assemble_cell_term_velocity(const MatrixFree<dim, Number>&       data,
                                     Vec&                                 dst,
                                     const Vec&                           src,
                                     const std::pair<unsigned, unsigned>& cell_range) const;

    /*--- Assembler functions for the 'B' matrix. ---*/
    void assemble_cell_term_pressure(const MatrixFree<dim, Number>&       data,
                                     Vec&                                 dst,
                                     const Vec&                           src,
                                     const std::pair<unsigned, unsigned>& cell_range) const;
    void assemble_face_term_pressure(const MatrixFree<dim, Number>&       data,
                                     Vec&                                 dst,
                                     const Vec&                           src,
                                     const std::pair<unsigned, unsigned>& face_range) const;
    void assemble_boundary_term_pressure(const MatrixFree<dim, Number>&       data,
                                         Vec&                                 dst,
                                         const Vec&                           src,
                                         const std::pair<unsigned, unsigned>& face_range) const;

    /*--- Assembler functions for the rhs of the energy equation. ---*/
    void assemble_rhs_cell_term_energy(const MatrixFree<dim, Number>&       data,
                                       Vec&                                 dst,
                                       const std::vector<Vec>&              src,
                                       const std::pair<unsigned, unsigned>& cell_range) const;
    void assemble_rhs_face_term_energy(const MatrixFree<dim, Number>&       data,
                                       Vec&                                 dst,
                                       const std::vector<Vec>&              src,
                                       const std::pair<unsigned, unsigned>& face_range) const;
    void assemble_rhs_boundary_term_energy(const MatrixFree<dim, Number>&       data,
                                           Vec&                                 dst,
                                           const std::vector<Vec>&              src,
                                           const std::pair<unsigned, unsigned>& face_range) const {}
                                           /*-- No flux, so no contribution from this function ---*/

    /*--- Assembler function for the 'D' matrix. ---*/
    void assemble_cell_term_internal_energy(const MatrixFree<dim, Number>&       data,
                                            Vec&                                 dst,
                                            const Vec&                           src,
                                            const std::pair<unsigned, unsigned>& cell_range) const;

    void assemble_inverse_cell_term_internal_energy(const MatrixFree<dim, Number>&       data,
                                                    Vec&                                 dst,
                                                    const Vec&                           src,
                                                    const std::pair<unsigned, unsigned>& cell_range) const;

    /*--- Assembler function for the 'C' matrix. ---*/
    void assemble_cell_term_enthalpy(const MatrixFree<dim, Number>&       data,
                                     Vec&                                 dst,
                                     const Vec&                           src,
                                     const std::pair<unsigned, unsigned>& cell_range) const;
    void assemble_face_term_enthalpy(const MatrixFree<dim, Number>&       data,
                                     Vec&                                 dst,
                                     const Vec&                           src,
                                     const std::pair<unsigned, unsigned>& face_range) const;
    void assemble_boundary_term_enthalpy(const MatrixFree<dim, Number>&       data,
                                         Vec&                                 dst,
                                         const Vec&                           src,
                                         const std::pair<unsigned, unsigned>& face_range) const {}
                                         /*-- No flux, so no contribution from this function ---*/

    /*--- Assembler functions for the diagonal part of the matrix for the continuity equation. For compatibilty conditions,
          also face and boundary contributions have to be defined, even though they are empty. ---*/
    void assemble_diagonal_cell_term_density(const MatrixFree<dim, Number>&       data,
                                             Vec&                                 dst,
                                             const unsigned&                      src,
                                             const std::pair<unsigned, unsigned>& cell_range) const;

    /*--- Assembler functions for the diagonal part of 'A' matrix. ---*/
    void assemble_diagonal_cell_term_velocity(const MatrixFree<dim, Number>&       data,
                                              Vec&                                 dst,
                                              const unsigned&                      src,
                                              const std::pair<unsigned, unsigned>& cell_range) const;

    /*--- Assembler functions for the diagonal part of the ellptic operator associated to the Schur complement for the pressure. ---*/
    void assemble_diagonal_cell_term_pressure(const MatrixFree<dim, Number>&       data,
                                              Vec&                                 dst,
                                              const unsigned&                      src,
                                              const std::pair<unsigned, unsigned>& cell_range) const;

    /*--- Assembler functions for the diagonal part of 'D' matrix. ---*/
    void assemble_diagonal_cell_term_internal_energy(const MatrixFree<dim, Number>&       data,
                                                     Vec&                                 dst,
                                                     const unsigned&                      src,
                                                     const std::pair<unsigned, unsigned>& cell_range) const;
  };

  //////////////////////////////////////////////////////////////
  /*---- START WITH CLASS CONSTRUCTORS ---*/
  /////////////////////////////////////////////////////////////


  // Default constructor
  //
  template<unsigned dim,
           unsigned fe_degree_u, unsigned fe_degree_rho, unsigned fe_degree_p,
           unsigned n_q_points_1d, unsigned n_q_points_1d_boundary,
           typename Vec>
  EULEROperator<dim,
                fe_degree_u, fe_degree_rho, fe_degree_p,
                n_q_points_1d, n_q_points_1d_boundary,
                Vec>::
  EULEROperator():
    MatrixFreeOperators::Base<dim, Vec>(), Ma(), Fr(), dt(),
    gamma(static_cast<Number>(2.0) - static_cast<Number>(std::sqrt(2.0))), a21(gamma),
    a31(static_cast<Number>(0.5)), a32(static_cast<Number>(0.5)),
    a21_tilde(static_cast<Number>(0.5)*gamma), a22_tilde(static_cast<Number>(0.5)*gamma),
    a31_tilde(static_cast<Number>(0.5) - static_cast<Number>(0.25)*gamma),
    a32_tilde(static_cast<Number>(0.5) - static_cast<Number>(0.25)*gamma),
    a33_tilde(static_cast<Number>(0.5)*gamma),
    b1(static_cast<Number>(0.5) - static_cast<Number>(0.25)*gamma),
    b2(static_cast<Number>(0.5) - static_cast<Number>(0.25)*gamma),
    b3(static_cast<Number>(0.5)*gamma),
    IMEX_stage(1), Euler_stage(1), num_flux() {}

  // Constructor with runtime parameters storage
  //
  template<unsigned dim,
           unsigned fe_degree_u, unsigned fe_degree_rho, unsigned fe_degree_p,
           unsigned n_q_points_1d, unsigned n_q_points_1d_boundary,
           typename Vec>
  EULEROperator<dim,
                fe_degree_u, fe_degree_rho, fe_degree_p,
                n_q_points_1d, n_q_points_1d_boundary,
                Vec>::
  EULEROperator(RunTimeParameters::Data_Storage<Number>& data):
    MatrixFreeOperators::Base<dim, Vec>(),
    Ma(data.Mach), Fr(data.Froude), dt(data.dt),
    gamma(static_cast<Number>(2.0) - static_cast<Number>(std::sqrt(2.0))), a21(gamma),
    a31(static_cast<Number>(0.5)), a32(static_cast<Number>(0.5)),
    a21_tilde(static_cast<Number>(0.5)*gamma), a22_tilde(static_cast<Number>(0.5)*gamma),
    a31_tilde(static_cast<Number>(0.5) - static_cast<Number>(0.25)*gamma),
    a32_tilde(static_cast<Number>(0.5) - static_cast<Number>(0.25)*gamma),
    a33_tilde(static_cast<Number>(0.5)*gamma),
    b1(static_cast<Number>(0.5) - static_cast<Number>(0.25)*gamma),
    b2(static_cast<Number>(0.5) - static_cast<Number>(0.25)*gamma),
    b3(static_cast<Number>(0.5)*gamma),
    IMEX_stage(1), Euler_stage(1), num_flux(Ma) {}


  //////////////////////////////////////////////////////////////
  /*---- FOCUS NOW ON SOME AUXILIARY GETTERS AND SETTERS ---*/
  /////////////////////////////////////////////////////////////

  // Setter of time-step
  //
  template<unsigned dim,
           unsigned fe_degree_u, unsigned fe_degree_rho, unsigned fe_degree_p,
           unsigned n_q_points_1d, unsigned n_q_points_1d_boundary,
           typename Vec>
  inline DEAL_II_ALWAYS_INLINE
  void EULEROperator<dim,
                     fe_degree_u, fe_degree_rho, fe_degree_p,
                     n_q_points_1d, n_q_points_1d_boundary,
                     Vec>::
  set_dt(const Number time_step) {
    dt = time_step;
  }

  // Getter of Mach number
  //
  template<unsigned dim,
           unsigned fe_degree_u, unsigned fe_degree_rho, unsigned fe_degree_p,
           unsigned n_q_points_1d, unsigned n_q_points_1d_boundary,
           typename Vec>
  inline DEAL_II_ALWAYS_INLINE
  typename EULEROperator<dim,
                     fe_degree_u, fe_degree_rho, fe_degree_p,
                     n_q_points_1d, n_q_points_1d_boundary,
                     Vec>::Number
  EULEROperator<dim,
                     fe_degree_u, fe_degree_rho, fe_degree_p,
                     n_q_points_1d, n_q_points_1d_boundary,
                     Vec>::
  get_Mach() const {
    return Ma;
  }

  // Getter of Froude number
  //
  template<unsigned dim,
           unsigned fe_degree_u, unsigned fe_degree_rho, unsigned fe_degree_p,
           unsigned n_q_points_1d, unsigned n_q_points_1d_boundary,
           typename Vec>
  inline DEAL_II_ALWAYS_INLINE
  typename EULEROperator<dim,
                     fe_degree_u, fe_degree_rho, fe_degree_p,
                     n_q_points_1d, n_q_points_1d_boundary,
                     Vec>::Number
  EULEROperator<dim,
                     fe_degree_u, fe_degree_rho, fe_degree_p,
                     n_q_points_1d, n_q_points_1d_boundary,
                     Vec>::
  get_Froude() const {
    return Fr;
  }

  // Setter of IMEX stage (this can be known only during the effective execution
  // and so it has to be demanded to the class that really solves the problem)
  //
  template<unsigned dim,
           unsigned fe_degree_u, unsigned fe_degree_rho, unsigned fe_degree_p,
           unsigned n_q_points_1d, unsigned n_q_points_1d_boundary,
           typename Vec>
  inline DEAL_II_ALWAYS_INLINE
  void EULEROperator<dim,
                     fe_degree_u, fe_degree_rho, fe_degree_p,
                     n_q_points_1d, n_q_points_1d_boundary,
                     Vec>::
  set_IMEX_stage(const unsigned stage) {
    AssertIndexRange(stage, EquationData::n_stages + 2);
    Assert(stage > 0, ExcInternalError());

    IMEX_stage = stage;
  }

  // Setter of Euler stage (this can be known only during the effective execution
  // and so it has to be demanded to the class that really solves the problem)
  //
  template<unsigned dim,
           unsigned fe_degree_u, unsigned fe_degree_rho, unsigned fe_degree_p,
           unsigned n_q_points_1d, unsigned n_q_points_1d_boundary,
           typename Vec>
  inline DEAL_II_ALWAYS_INLINE
  void EULEROperator<dim,
                     fe_degree_u, fe_degree_rho, fe_degree_p,
                     n_q_points_1d, n_q_points_1d_boundary,
                     Vec>::
  set_Euler_stage(const unsigned stage) {
    AssertIndexRange(stage, EquationData::n_vars + 1);
    Assert(stage > 0, ExcInternalError());

    Euler_stage = stage;
  }

  // Getter of Euler stage (this can be known only during the effective execution
  // and so it has to be demanded to the class that really solves the problem)
  //
  template<unsigned dim,
           unsigned fe_degree_u, unsigned fe_degree_rho, unsigned fe_degree_p,
           unsigned n_q_points_1d, unsigned n_q_points_1d_boundary,
           typename Vec>
  inline DEAL_II_ALWAYS_INLINE
  unsigned EULEROperator<dim,
                         fe_degree_u, fe_degree_rho, fe_degree_p,
                         n_q_points_1d, n_q_points_1d_boundary,
                         Vec>::
  get_Euler_stage() const {
    return Euler_stage;
  }


  // Setter of density for fixed point
  //
  template<unsigned dim,
           unsigned fe_degree_u, unsigned fe_degree_rho, unsigned fe_degree_p,
           unsigned n_q_points_1d, unsigned n_q_points_1d_boundary,
           typename Vec>
  void EULEROperator<dim,
                     fe_degree_u, fe_degree_rho, fe_degree_p,
                     n_q_points_1d, n_q_points_1d_boundary,
                     Vec>::
  set_rho_for_fixed(const Vec& src) {
    rho_for_fixed = src;
    rho_for_fixed.update_ghost_values();
  }

  // Setter of pressure for fixed point
  //
  template<unsigned dim,
           unsigned fe_degree_u, unsigned fe_degree_rho, unsigned fe_degree_p,
           unsigned n_q_points_1d, unsigned n_q_points_1d_boundary,
           typename Vec>
  void EULEROperator<dim,
                     fe_degree_u, fe_degree_rho, fe_degree_p,
                     n_q_points_1d, n_q_points_1d_boundary,
                     Vec>::
  set_pres_fixed(const Vec& src) {
    pres_fixed = src;
    pres_fixed.update_ghost_values();
  }


  //////////////////////////////////////////////////////////////
  /*---- ASSEMBLING LINEAR AND BILINEAR FORMS FOR THE CONTINUITY EQUATION ---*/
  /////////////////////////////////////////////////////////////

  // Assemble rhs cell term for the density update
  //
  template<unsigned dim,
           unsigned fe_degree_u, unsigned fe_degree_rho, unsigned fe_degree_p,
           unsigned n_q_points_1d, unsigned n_q_points_1d_boundary,
           typename Vec>
  void EULEROperator<dim,
                     fe_degree_u, fe_degree_rho, fe_degree_p,
                     n_q_points_1d, n_q_points_1d_boundary,
                     Vec>::
  assemble_rhs_cell_term_density(const MatrixFree<dim, Number>&       data,
                                 Vec&                                 dst,
                                 const std::vector<Vec>&              src,
                                 const std::pair<unsigned, unsigned>& cell_range) const {
    if(IMEX_stage == 2) {
      /*--- We first start by declaring the suitable instances to read the old density and
      the old velocity. 'phi' will be used only to 'submit' the result.
      The second argument specifies which dof handler has to be used. ---*/
      FEEvaluation_rho phi(data, EquationData::RHO_INDEX_DOF),
                       phi_rho_old(data, EquationData::RHO_INDEX_DOF);
      FEEvaluation_u   phi_u_old(data, EquationData::U_INDEX_DOF);

      /*--- Loop over all cells ---*/
      for(unsigned cell = cell_range.first; cell < cell_range.second; ++cell) {
        /*--- Now we need to assign the current cell to each FEEvaluation object and then to specify which src vector
        it has to read (the proper order is clearly delegated to the user, which has to pay attention in the function
        call to be coherent). All these considerations are valid also for the other assembler functions. ---*/
        phi_rho_old.reinit(cell);
        phi_rho_old.gather_evaluate(src[0], EvaluationFlags::values);
        phi_u_old.reinit(cell);
        phi_u_old.gather_evaluate(src[1], EvaluationFlags::values);

        phi.reinit(cell);

        /*--- Loop over quadrature points of each cell ---*/
        for(unsigned q = 0; q < phi.n_q_points; ++q) {
          /*--- Compute the quantities at the previous step. ---*/
          const auto& rho_old = phi_rho_old.get_value(q);
          const auto& u_old   = phi_u_old.get_value(q);

          phi.submit_value(rho_old, q);
          /*--- submit_value is used for quantities to be tested against test functions ---*/
          phi.submit_gradient(a21*dt*(rho_old*u_old), q);
          /*--- submit_gradient is used for quantities to be tested against gradient of test functions ---*/
        }

        phi.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
        /*--- 'integrate_scatter' is the responsible of distributing into dst.
              The flag parameter specifies if we are testing against the test function and/or its gradient ---*/
      }
    }
    else if(IMEX_stage == 3) {
      /*--- We first start by declaring the suitable instances to read the available quantities. ---*/
      FEEvaluation_rho phi(data, EquationData::RHO_INDEX_DOF),
                       phi_rho_old(data, EquationData::RHO_INDEX_DOF),
                       phi_rho_s_2(data, EquationData::RHO_INDEX_DOF);
      FEEvaluation_u   phi_u_old(data, EquationData::U_INDEX_DOF),
                       phi_u_s_2(data, EquationData::U_INDEX_DOF);

      /*--- Loop over all cells ---*/
      for(unsigned cell = cell_range.first; cell < cell_range.second; ++cell) {
        phi_rho_old.reinit(cell);
        phi_rho_old.gather_evaluate(src[0], EvaluationFlags::values);
        phi_u_old.reinit(cell);
        phi_u_old.gather_evaluate(src[1], EvaluationFlags::values);

        phi_rho_s_2.reinit(cell);
        phi_rho_s_2.gather_evaluate(src[2], EvaluationFlags::values);
        phi_u_s_2.reinit(cell);
        phi_u_s_2.gather_evaluate(src[3], EvaluationFlags::values);

        phi.reinit(cell);

        /*--- Loop over quadrature points of each cell ---*/
        for(unsigned q = 0; q < phi.n_q_points; ++q) {
          /*--- Compute the quantities at the previous step ---*/
          const auto& rho_old  = phi_rho_old.get_value(q);
          const auto& u_old    = phi_u_old.get_value(q);

          /*--- Compute the quantities at the previous stage ---*/
          const auto& rho_s_2 = phi_rho_s_2.get_value(q);
          const auto& u_s_2   = phi_u_s_2.get_value(q);

          phi.submit_value(rho_old, q);
          phi.submit_gradient(a31*dt*(rho_old*u_old) +
                              a32*dt*(rho_s_2*u_s_2), q);
        }

        phi.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
      }
    }
    else {
      /*--- We first start by declaring the suitable instances to read the available quantities. ---*/
      FEEvaluation_rho phi(data, EquationData::RHO_INDEX_DOF),
                       phi_rho_old(data, EquationData::RHO_INDEX_DOF),
                       phi_rho_s_2(data, EquationData::RHO_INDEX_DOF),
                       phi_rho_s_3(data, EquationData::RHO_INDEX_DOF);
      FEEvaluation_u   phi_u_old(data, EquationData::U_INDEX_DOF),
                       phi_u_s_2(data, EquationData::U_INDEX_DOF),
                       phi_u_s_3(data, EquationData::U_INDEX_DOF);

      /*--- Loop over all cells ---*/
      for(unsigned cell = cell_range.first; cell < cell_range.second; ++cell) {
        phi_rho_old.reinit(cell);
        phi_rho_old.gather_evaluate(src[0], EvaluationFlags::values);
        phi_u_old.reinit(cell);
        phi_u_old.gather_evaluate(src[1], EvaluationFlags::values);

        phi_rho_s_2.reinit(cell);
        phi_rho_s_2.gather_evaluate(src[2], EvaluationFlags::values);
        phi_u_s_2.reinit(cell);
        phi_u_s_2.gather_evaluate(src[3], EvaluationFlags::values);

        phi_rho_s_3.reinit(cell);
        phi_rho_s_3.gather_evaluate(src[4], EvaluationFlags::values);
        phi_u_s_3.reinit(cell);
        phi_u_s_3.gather_evaluate(src[5], EvaluationFlags::values);

        phi.reinit(cell);

        /*--- Loop over quadrature points of each cell ---*/
        for(unsigned q = 0; q < phi.n_q_points; ++q) {
          /*--- Compute the quantities at the previous step ---*/
          const auto& rho_old = phi_rho_old.get_value(q);
          const auto& u_old   = phi_u_old.get_value(q);

          /*--- Compute the quantities at the second stage ---*/
          const auto& rho_s_2 = phi_rho_s_2.get_value(q);
          const auto& u_s_2   = phi_u_s_2.get_value(q);

          /*--- Compute the quantities at the final stage ---*/
          const auto& rho_s_3 = phi_rho_s_3.get_value(q);
          const auto& u_s_3   = phi_u_s_3.get_value(q);

          phi.submit_value(rho_old, q);
          phi.submit_gradient(b1*dt*(rho_old*u_old) +
                              b2*dt*(rho_s_2*u_s_2) +
                              b3*dt*(rho_s_3*u_s_3), q);
        }

        phi.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
      }
    }
  }

  // Assemble rhs face term for the density update
  //
  template<unsigned dim,
           unsigned fe_degree_u, unsigned fe_degree_rho, unsigned fe_degree_p,
           unsigned n_q_points_1d, unsigned n_q_points_1d_boundary,
           typename Vec>
  void EULEROperator<dim,
                     fe_degree_u, fe_degree_rho, fe_degree_p,
                     n_q_points_1d, n_q_points_1d_boundary,
                     Vec>::
  assemble_rhs_face_term_density(const MatrixFree<dim, Number>&       data,
                                 Vec&                                 dst,
                                 const std::vector<Vec>&              src,
                                 const std::pair<unsigned, unsigned>& face_range) const {
    if(IMEX_stage == 2) {
      /*--- We first start by declaring the suitable instances to read the available quantities.
            'true' means that we are reading the information from 'inside', whereas 'false' from 'outside' ---*/
      FEFaceEvaluation_rho phi_m(data, true, EquationData::RHO_INDEX_DOF),
                           phi_p(data, false, EquationData::RHO_INDEX_DOF),
                           phi_rho_old_m(data, true, EquationData::RHO_INDEX_DOF),
                           phi_rho_old_p(data, false, EquationData::RHO_INDEX_DOF);
      FEFaceEvaluation_u   phi_u_old_m(data, true, EquationData::U_INDEX_DOF),
                           phi_u_old_p(data, false, EquationData::U_INDEX_DOF);

      /*--- Loop over all internal faces ---*/
      for(unsigned face = face_range.first; face < face_range.second; ++face) {
        phi_rho_old_m.reinit(face);
        phi_rho_old_m.gather_evaluate(src[0], EvaluationFlags::values);
        phi_rho_old_p.reinit(face);
        phi_rho_old_p.gather_evaluate(src[0], EvaluationFlags::values);
        phi_u_old_m.reinit(face);
        phi_u_old_m.gather_evaluate(src[1], EvaluationFlags::values);
        phi_u_old_p.reinit(face);
        phi_u_old_p.gather_evaluate(src[1], EvaluationFlags::values);

        phi_m.reinit(face);
        phi_p.reinit(face);

        /*--- Loop over quadrature points of each internal face ---*/
        for(unsigned q = 0; q < phi_m.n_q_points; ++q) {
          const auto& n_minus = phi_m.get_normal_vector(q); /*--- Notice that the unit normal vector is the same from
                                                                  'both sides'. ---*/

          /*--- Compute the quantities at the previous step ---*/
          const auto& rho_old_m = phi_rho_old_m.get_value(q);
          const auto& rho_old_p = phi_rho_old_p.get_value(q);
          const auto& u_old_m   = phi_u_old_m.get_value(q);
          const auto& u_old_p   = phi_u_old_p.get_value(q);

          /*--- Compute the numerical flux ---*/
          const auto& flux_num = a21*dt*num_flux.numerical_flux_continuity(rho_old_m, u_old_m,
                                                                           rho_old_p, u_old_p,
                                                                           n_minus);

          phi_m.submit_value(-flux_num, q);
          phi_p.submit_value(flux_num, q);
        }

        phi_m.integrate_scatter(EvaluationFlags::values, dst);
        phi_p.integrate_scatter(EvaluationFlags::values, dst);
      }
    }
    else if(IMEX_stage == 3) {
      /*--- We first start by declaring the suitable instances to read the available quantities. ---*/
      FEFaceEvaluation_rho phi_m(data, true, EquationData::RHO_INDEX_DOF),
                           phi_p(data, false, EquationData::RHO_INDEX_DOF),
                           phi_rho_old_m(data, true, EquationData::RHO_INDEX_DOF),
                           phi_rho_old_p(data, false, EquationData::RHO_INDEX_DOF),
                           phi_rho_s_2_m(data, true, EquationData::RHO_INDEX_DOF),
                           phi_rho_s_2_p(data, false, EquationData::RHO_INDEX_DOF);
      FEFaceEvaluation_u   phi_u_old_m(data, true, EquationData::U_INDEX_DOF),
                           phi_u_old_p(data, false, EquationData::U_INDEX_DOF),
                           phi_u_s_2_m(data, true, EquationData::U_INDEX_DOF),
                           phi_u_s_2_p(data, false, EquationData::U_INDEX_DOF);

      /*--- Loop over all internal faces ---*/
      for(unsigned face = face_range.first; face < face_range.second; ++face) {
        phi_rho_old_m.reinit(face);
        phi_rho_old_m.gather_evaluate(src[0], EvaluationFlags::values);
        phi_rho_old_p.reinit(face);
        phi_rho_old_p.gather_evaluate(src[0], EvaluationFlags::values);
        phi_u_old_m.reinit(face);
        phi_u_old_m.gather_evaluate(src[1], EvaluationFlags::values);
        phi_u_old_p.reinit(face);
        phi_u_old_p.gather_evaluate(src[1], EvaluationFlags::values);

        phi_rho_s_2_m.reinit(face);
        phi_rho_s_2_m.gather_evaluate(src[2], EvaluationFlags::values);
        phi_rho_s_2_p.reinit(face);
        phi_rho_s_2_p.gather_evaluate(src[2], EvaluationFlags::values);
        phi_u_s_2_m.reinit(face);
        phi_u_s_2_m.gather_evaluate(src[3], EvaluationFlags::values);
        phi_u_s_2_p.reinit(face);
        phi_u_s_2_p.gather_evaluate(src[3], EvaluationFlags::values);

        phi_m.reinit(face);
        phi_p.reinit(face);

        /*--- Loop over all quadrature points ---*/
        for(unsigned q = 0; q < phi_m.n_q_points; ++q) {
          const auto& n_minus = phi_m.get_normal_vector(q);

          /*--- Compute the quantities at the previous step ---*/
          const auto& rho_old_m = phi_rho_old_m.get_value(q);
          const auto& rho_old_p = phi_rho_old_p.get_value(q);
          const auto& u_old_m   = phi_u_old_m.get_value(q);
          const auto& u_old_p   = phi_u_old_p.get_value(q);

          /*--- Compute the quantities at the previous stage ---*/
          const auto& rho_s_2_m = phi_rho_s_2_m.get_value(q);
          const auto& rho_s_2_p = phi_rho_s_2_p.get_value(q);
          const auto& u_s_2_m   = phi_u_s_2_m.get_value(q);
          const auto& u_s_2_p   = phi_u_s_2_p.get_value(q);

          /*--- Compute the numerical flux ---*/
          const auto& flux_num = a31*dt*num_flux.numerical_flux_continuity(rho_old_m, u_old_m,
                                                                           rho_old_p, u_old_p,
                                                                           n_minus)
                               + a32*dt*num_flux.numerical_flux_continuity(rho_s_2_m, u_s_2_m,
                                                                           rho_s_2_p, u_s_2_p,
                                                                           n_minus);

          phi_m.submit_value(-flux_num, q);
          phi_p.submit_value(flux_num, q);
        }

        phi_m.integrate_scatter(EvaluationFlags::values, dst);
        phi_p.integrate_scatter(EvaluationFlags::values, dst);
      }
    }
    else {
      /*--- We first start by declaring the suitable instances to read the available quantities. ---*/
      FEFaceEvaluation_rho phi_m(data, true, EquationData::RHO_INDEX_DOF),
                           phi_p(data, false, EquationData::RHO_INDEX_DOF),
                           phi_rho_old_m(data, true, EquationData::RHO_INDEX_DOF),
                           phi_rho_old_p(data, false, EquationData::RHO_INDEX_DOF),
                           phi_rho_s_2_m(data, true, EquationData::RHO_INDEX_DOF),
                           phi_rho_s_2_p(data, false, EquationData::RHO_INDEX_DOF),
                           phi_rho_s_3_m(data, true, EquationData::RHO_INDEX_DOF),
                           phi_rho_s_3_p(data, false, EquationData::RHO_INDEX_DOF);
      FEFaceEvaluation_u   phi_u_old_m(data, true, EquationData::U_INDEX_DOF),
                           phi_u_old_p(data, false, EquationData::U_INDEX_DOF),
                           phi_u_s_2_m(data, true, EquationData::U_INDEX_DOF),
                           phi_u_s_2_p(data, false, EquationData::U_INDEX_DOF),
                           phi_u_s_3_m(data, true, EquationData::U_INDEX_DOF),
                           phi_u_s_3_p(data, false, EquationData::U_INDEX_DOF);

      /*--- Loop over all internal faces ---*/
      for(unsigned face = face_range.first; face < face_range.second; ++face) {
        phi_rho_old_m.reinit(face);
        phi_rho_old_m.gather_evaluate(src[0], EvaluationFlags::values);
        phi_rho_old_p.reinit(face);
        phi_rho_old_p.gather_evaluate(src[0], EvaluationFlags::values);
        phi_u_old_m.reinit(face);
        phi_u_old_m.gather_evaluate(src[1], EvaluationFlags::values);
        phi_u_old_p.reinit(face);
        phi_u_old_p.gather_evaluate(src[1], EvaluationFlags::values);

        phi_rho_s_2_m.reinit(face);
        phi_rho_s_2_m.gather_evaluate(src[2], EvaluationFlags::values);
        phi_rho_s_2_p.reinit(face);
        phi_rho_s_2_p.gather_evaluate(src[2], EvaluationFlags::values);
        phi_u_s_2_m.reinit(face);
        phi_u_s_2_m.gather_evaluate(src[3], EvaluationFlags::values);
        phi_u_s_2_p.reinit(face);
        phi_u_s_2_p.gather_evaluate(src[3], EvaluationFlags::values);

        phi_rho_s_3_m.reinit(face);
        phi_rho_s_3_m.gather_evaluate(src[4], EvaluationFlags::values);
        phi_rho_s_3_p.reinit(face);
        phi_rho_s_3_p.gather_evaluate(src[4], EvaluationFlags::values);
        phi_u_s_3_m.reinit(face);
        phi_u_s_3_m.gather_evaluate(src[5], EvaluationFlags::values);
        phi_u_s_3_p.reinit(face);
        phi_u_s_3_p.gather_evaluate(src[5], EvaluationFlags::values);

        phi_m.reinit(face);
        phi_p.reinit(face);

        /*--- Loop over all quadrature points. ---*/
        for(unsigned q = 0; q < phi_m.n_q_points; ++q) {
          const auto& n_minus = phi_m.get_normal_vector(q);

          /*--- Compute the quantities at the previous step ---*/
          const auto& rho_old_m = phi_rho_old_m.get_value(q);
          const auto& rho_old_p = phi_rho_old_p.get_value(q);
          const auto& u_old_m   = phi_u_old_m.get_value(q);
          const auto& u_old_p   = phi_u_old_p.get_value(q);

          /*--- Compute the quantities at the second stage ---*/
          const auto& rho_s_2_m = phi_rho_s_2_m.get_value(q);
          const auto& rho_s_2_p = phi_rho_s_2_p.get_value(q);
          const auto& u_s_2_m   = phi_u_s_2_m.get_value(q);
          const auto& u_s_2_p   = phi_u_s_2_p.get_value(q);

          /*--- Compute the quantities at the final stage ---*/
          const auto& rho_s_3_m = phi_rho_s_3_m.get_value(q);
          const auto& rho_s_3_p = phi_rho_s_3_p.get_value(q);
          const auto& u_s_3_m   = phi_u_s_3_m.get_value(q);
          const auto& u_s_3_p   = phi_u_s_3_p.get_value(q);

          /*--- Compute the numerical flux ---*/
          const auto& flux_num = b1*dt*num_flux.numerical_flux_continuity(rho_old_m, u_old_m,
                                                                          rho_old_p, u_old_p,
                                                                          n_minus)
                               + b2*dt*num_flux.numerical_flux_continuity(rho_s_2_m, u_s_2_m,
                                                                          rho_s_2_p, u_s_2_p,
                                                                          n_minus)
                               + b3*dt*num_flux.numerical_flux_continuity(rho_s_3_m, u_s_3_m,
                                                                          rho_s_3_p, u_s_3_p,
                                                                          n_minus);

          phi_m.submit_value(-flux_num, q);
          phi_p.submit_value(flux_num, q);
        }

        phi_m.integrate_scatter(EvaluationFlags::values, dst);
        phi_p.integrate_scatter(EvaluationFlags::values, dst);
      }
    }
  }

  // Put together all the previous steps for density update
  //
  template<unsigned dim,
           unsigned fe_degree_u, unsigned fe_degree_rho, unsigned fe_degree_p,
           unsigned n_q_points_1d, unsigned n_q_points_1d_boundary,
           typename Vec>
  void EULEROperator<dim,
                     fe_degree_u, fe_degree_rho, fe_degree_p,
                     n_q_points_1d, n_q_points_1d_boundary,
                     Vec>::
  vmult_rhs_density(Vec& dst, const std::vector<Vec>& src) const {
    for(unsigned d = 0; d < src.size(); ++d) {
      src[d].update_ghost_values();
    }

    this->data->loop(&EULEROperator::assemble_rhs_cell_term_density,
                     &EULEROperator::assemble_rhs_face_term_density,
                     &EULEROperator::assemble_rhs_boundary_term_density,
                     this, dst, src, true,
                     MatrixFree<dim, Number>::DataAccessOnFaces::values,
                     MatrixFree<dim, Number>::DataAccessOnFaces::values);
  }

  // Assemble cell term for the density update
  //
  template<unsigned dim,
           unsigned fe_degree_u, unsigned fe_degree_rho, unsigned fe_degree_p,
           unsigned n_q_points_1d, unsigned n_q_points_1d_boundary,
           typename Vec>
  void EULEROperator<dim,
                     fe_degree_u, fe_degree_rho, fe_degree_p,
                     n_q_points_1d, n_q_points_1d_boundary,
                     Vec>::
  assemble_cell_term_density(const MatrixFree<dim, Number>&               data,
                             Vec&                                         dst,
                             const Vec&                                   src,
                             const std::pair<unsigned, unsigned>& cell_range) const {
    FEEvaluation<dim, fe_degree_rho, fe_degree_rho + 1, 1, Number> phi(data, EquationData::RHO_INDEX_DOF, 2);

    MatrixFreeOperators::CellwiseInverseMassMatrix<dim, fe_degree_rho, 1, Number> inverse(phi);

    /*--- Loop over all cells ---*/
    for(unsigned cell = cell_range.first; cell < cell_range.second; ++cell) {
      phi.reinit(cell);
      phi.read_dof_values(src);

      inverse.apply(phi.begin_dof_values(),
                    phi.begin_dof_values());

      phi.set_dof_values(dst);
    }
  }


  //////////////////////////////////////////////////////////////
  /*---- ASSEMBLING LINEAR AND BILINEAR FORMS FOR THE MOMENTUM EQUATION ---*/
  /////////////////////////////////////////////////////////////

  // Assemble rhs cell term of the momentum equation
  //
  template<unsigned dim,
           unsigned fe_degree_u, unsigned fe_degree_rho, unsigned fe_degree_p,
           unsigned n_q_points_1d, unsigned n_q_points_1d_boundary,
           typename Vec>
  void EULEROperator<dim,
                     fe_degree_u, fe_degree_rho, fe_degree_p,
                     n_q_points_1d, n_q_points_1d_boundary,
                     Vec>::
  assemble_rhs_cell_term_momentum(const MatrixFree<dim, Number>&       data,
                                  Vec&                                 dst,
                                  const std::vector<Vec>&              src,
                                  const std::pair<unsigned, unsigned>& cell_range) const {
    /*--- We create an auxiliary vector for the unit vector along vertical direction. This will never change
          independently on the stage, so we declare it once and for all. ---*/
    Tensor<1, dim, VectorizedArray<Number>> e_k;
    for(unsigned d = 0; d < dim - 1; ++d) {
      e_k[d] = make_vectorized_array<Number>(0.0);
    }
    e_k[dim - 1] = make_vectorized_array<Number>(1.0);

    if(IMEX_stage == 2) {
      /*--- We first start by declaring the suitable instances to read the available quantities. ---*/
      FEEvaluation_u    phi(data, EquationData::U_INDEX_DOF),
                        phi_u_old(data, EquationData::U_INDEX_DOF);
      FEEvaluation_pres phi_pres_old(data, EquationData::P_INDEX_DOF);
      FEEvaluation_rho  phi_rho_old(data, EquationData::RHO_INDEX_DOF),
                        phi_rho_s_2(data, EquationData::RHO_INDEX_DOF);

      /*--- Loop over all cells ---*/
      for(unsigned cell = cell_range.first; cell < cell_range.second; ++cell) {
        phi_rho_old.reinit(cell);
        phi_rho_old.gather_evaluate(src[0], EvaluationFlags::values);
        phi_u_old.reinit(cell);
        phi_u_old.gather_evaluate(src[1], EvaluationFlags::values);
        phi_pres_old.reinit(cell);
        phi_pres_old.gather_evaluate(src[2], EvaluationFlags::values);

        phi_rho_s_2.reinit(cell);
        phi_rho_s_2.gather_evaluate(src[3], EvaluationFlags::values);

        phi.reinit(cell);

        /*--- Loop over all quadrature points ---*/
        for(unsigned q = 0; q < phi.n_q_points; ++q) {
          /*--- Compute the quantities at the previous step ---*/
          const auto& rho_old            = phi_rho_old.get_value(q);
          const auto& u_old              = phi_u_old.get_value(q);
          const auto& pres_old           = phi_pres_old.get_value(q);

          const auto& tensor_product_u_n = outer_product(u_old, u_old);
          /*--- For the sake of compatibility, since after integration by parts, the pressure gradient
                would be tested against the divergence of the test function. This is equaivalent to test a diagonal matrix
                with diagonal entries equal to the pressure itself against the gradient of the test function. ---*/
          Tensor<2, dim, VectorizedArray<Number>> p_n_times_identity;
          for(unsigned d = 0; d < dim; ++d) {
            p_n_times_identity[d][d] = pres_old;
          }

          const auto& rho_s_2 = phi_rho_s_2.get_value(q);

          phi.submit_value(rho_old*u_old -
                           a21_tilde*dt*(rho_old*e_k/(Fr*Fr)) -
                           a22_tilde*dt*(rho_s_2*e_k/(Fr*Fr)), q);
          phi.submit_gradient(a21*dt*(rho_old*tensor_product_u_n) +
                              a21_tilde*dt*(p_n_times_identity/(Ma*Ma)), q);
        }

        phi.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
      }
    }
    else if(IMEX_stage == 3) {
      /*--- We first start by declaring the suitable instances to read the available quantities. ---*/
      FEEvaluation_u    phi(data, EquationData::U_INDEX_DOF),
                        phi_u_old(data, EquationData::U_INDEX_DOF),
                        phi_u_s_2(data, EquationData::U_INDEX_DOF);
      FEEvaluation_pres phi_pres_old(data, EquationData::P_INDEX_DOF),
                        phi_pres_s_2(data, EquationData::P_INDEX_DOF);
      FEEvaluation_rho  phi_rho_old(data, EquationData::RHO_INDEX_DOF),
                        phi_rho_s_2(data, EquationData::RHO_INDEX_DOF),
                        phi_rho_curr(data, EquationData::RHO_INDEX_DOF);

      /*--- Loop over all cells ---*/
      for(unsigned cell = cell_range.first; cell < cell_range.second; ++cell) {
        phi_rho_old.reinit(cell);
        phi_rho_old.gather_evaluate(src[0], EvaluationFlags::values);
        phi_u_old.reinit(cell);
        phi_u_old.gather_evaluate(src[1], EvaluationFlags::values);
        phi_pres_old.reinit(cell);
        phi_pres_old.gather_evaluate(src[2], EvaluationFlags::values);

        phi_rho_s_2.reinit(cell);
        phi_rho_s_2.gather_evaluate(src[3], EvaluationFlags::values);
        phi_u_s_2.reinit(cell);
        phi_u_s_2.gather_evaluate(src[4], EvaluationFlags::values);
        phi_pres_s_2.reinit(cell);
        phi_pres_s_2.gather_evaluate(src[5], EvaluationFlags::values);

        phi_rho_curr.reinit(cell);
        phi_rho_curr.gather_evaluate(src[6], EvaluationFlags::values);

        phi.reinit(cell);

        /*--- Loop over all quadrature points ---*/
        for(unsigned q = 0; q < phi.n_q_points; ++q) {
          /*--- Compute the quantities at the previous step ---*/
          const auto& rho_old            = phi_rho_old.get_value(q);
          const auto& u_old              = phi_u_old.get_value(q);
          const auto& pres_old           = phi_pres_old.get_value(q);

          const auto& tensor_product_u_n = outer_product(u_old, u_old);
          Tensor<2, dim, VectorizedArray<Number>> p_n_times_identity;
          for(unsigned d = 0; d < dim; ++d) {
            p_n_times_identity[d][d] = pres_old;
          }

          /*--- Compute the quantities at the previous stage ---*/
          const auto& rho_s_2              = phi_rho_s_2.get_value(q);
          const auto& u_s_2                = phi_u_s_2.get_value(q);
          const auto& pres_s_2             = phi_pres_s_2.get_value(q);

          const auto& tensor_product_u_s_2 = outer_product(u_s_2, u_s_2);
          Tensor<2, dim, VectorizedArray<Number>> p_s_2_times_identity;
          for(unsigned d = 0; d < dim; ++d) {
            p_s_2_times_identity[d][d] = pres_s_2;
          }

          const auto& rho_curr = phi_rho_curr.get_value(q);

          phi.submit_value(rho_old*u_old -
                           a31_tilde*dt*(rho_old*e_k/(Fr*Fr)) -
                           a32_tilde*dt*(rho_s_2*e_k/(Fr*Fr)) -
                           a33_tilde*dt*(rho_curr*e_k/(Fr*Fr)), q);
          phi.submit_gradient(a31*dt*(rho_old*tensor_product_u_n) +
                              a31_tilde*dt*(p_n_times_identity/(Ma*Ma)) +
                              a32*dt*(rho_s_2*tensor_product_u_s_2) +
                              a32_tilde*dt*(p_s_2_times_identity/(Ma*Ma)), q);
        }

        phi.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
      }
    }
    else {
      /*--- We first start by declaring the suitable instances to read the available quantities. ---*/
      FEEvaluation_u    phi(data, EquationData::U_INDEX_DOF),
                        phi_u_old(data, EquationData::U_INDEX_DOF),
                        phi_u_s_2(data, EquationData::U_INDEX_DOF),
                        phi_u_s_3(data, EquationData::U_INDEX_DOF);
      FEEvaluation_pres phi_pres_old(data, EquationData::P_INDEX_DOF),
                        phi_pres_s_2(data, EquationData::P_INDEX_DOF),
                        phi_pres_s_3(data, EquationData::P_INDEX_DOF);
      FEEvaluation_rho  phi_rho_old(data, EquationData::RHO_INDEX_DOF),
                        phi_rho_s_2(data, EquationData::RHO_INDEX_DOF),
                        phi_rho_s_3(data, EquationData::RHO_INDEX_DOF);

      /*--- Loop over all cells ---*/
      for(unsigned cell = cell_range.first; cell < cell_range.second; ++cell) {
        phi_rho_old.reinit(cell);
        phi_rho_old.gather_evaluate(src[0], EvaluationFlags::values);
        phi_u_old.reinit(cell);
        phi_u_old.gather_evaluate(src[1], EvaluationFlags::values);
        phi_pres_old.reinit(cell);
        phi_pres_old.gather_evaluate(src[2], EvaluationFlags::values);

        phi_rho_s_2.reinit(cell);
        phi_rho_s_2.gather_evaluate(src[3], EvaluationFlags::values);
        phi_u_s_2.reinit(cell);
        phi_u_s_2.gather_evaluate(src[4], EvaluationFlags::values);
        phi_pres_s_2.reinit(cell);
        phi_pres_s_2.gather_evaluate(src[5], EvaluationFlags::values);

        phi_rho_s_3.reinit(cell);
        phi_rho_s_3.gather_evaluate(src[6], EvaluationFlags::values);
        phi_u_s_3.reinit(cell);
        phi_u_s_3.gather_evaluate(src[7], EvaluationFlags::values);
        phi_pres_s_3.reinit(cell);
        phi_pres_s_3.gather_evaluate(src[8], EvaluationFlags::values);

        phi.reinit(cell);

        /*--- Loop over all quadrature points ---*/
        for(unsigned q = 0; q < phi.n_q_points; ++q) {
          /*--- Compute the quantities at the previous step ---*/
          const auto& rho_old            = phi_rho_old.get_value(q);
          const auto& u_old              = phi_u_old.get_value(q);
          const auto& pres_old           = phi_pres_old.get_value(q);

          const auto& tensor_product_u_n = outer_product(u_old, u_old);
          Tensor<2, dim, VectorizedArray<Number>> p_n_times_identity;
          for(unsigned d = 0; d < dim; ++d) {
            p_n_times_identity[d][d] = pres_old;
          }

          /*--- Compute the quantities at the second stage ---*/
          const auto& rho_s_2              = phi_rho_s_2.get_value(q);
          const auto& u_s_2                = phi_u_s_2.get_value(q);
          const auto& pres_s_2             = phi_pres_s_2.get_value(q);

          const auto& tensor_product_u_s_2 = outer_product(u_s_2, u_s_2);
          Tensor<2, dim, VectorizedArray<Number>> p_s_2_times_identity;
          for(unsigned d = 0; d < dim; ++d) {
            p_s_2_times_identity[d][d] = pres_s_2;
          }

          /*--- Compute the quantities at the final stage ---*/
          const auto& rho_s_3              = phi_rho_s_3.get_value(q);
          const auto& u_s_3                = phi_u_s_3.get_value(q);
          const auto& pres_s_3             = phi_pres_s_3.get_value(q);

          const auto& tensor_product_u_s_3 = outer_product(u_s_3, u_s_3);
          Tensor<2, dim, VectorizedArray<Number>> p_s_3_times_identity;
          for(unsigned d = 0; d < dim; ++d) {
            p_s_3_times_identity[d][d] = pres_s_3;
          }

          phi.submit_value(rho_old*u_old -
                           b1*dt*(rho_old*e_k/(Fr*Fr)) -
                           b2*dt*(rho_s_2*e_k/(Fr*Fr)) -
                           b3*dt*(rho_s_3*e_k/(Fr*Fr)), q);
          phi.submit_gradient(b1*dt*(rho_old*tensor_product_u_n) +
                              b1*dt*(p_n_times_identity/(Ma*Ma)) +
                              b2*dt*(rho_s_2*tensor_product_u_s_2) +
                              b2*dt*(p_s_2_times_identity/(Ma*Ma)) +
                              b3*dt*(rho_s_3*tensor_product_u_s_3) +
                              b3*dt*(p_s_3_times_identity/(Ma*Ma)), q);
        }

        phi.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
      }
    }
  }

  // Assemble rhs face term of the momentum equation
  //
  template<unsigned dim,
           unsigned fe_degree_u, unsigned fe_degree_rho, unsigned fe_degree_p,
           unsigned n_q_points_1d, unsigned n_q_points_1d_boundary, typename Vec>
  void EULEROperator<dim,
                     fe_degree_u, fe_degree_rho, fe_degree_p,
                     n_q_points_1d, n_q_points_1d_boundary,
                     Vec>::
  assemble_rhs_face_term_momentum(const MatrixFree<dim, Number>&       data,
                                  Vec&                                 dst,
                                  const std::vector<Vec>&              src,
                                  const std::pair<unsigned, unsigned>& face_range) const {
    if(IMEX_stage == 2) {
      /*--- We first start by declaring the suitable instances to read the available quantities. ---*/
      FEFaceEvaluation_u    phi_m(data, true, EquationData::U_INDEX_DOF),
                            phi_p(data, false, EquationData::U_INDEX_DOF),
                            phi_u_old_m(data, true, EquationData::U_INDEX_DOF),
                            phi_u_old_p(data, false, EquationData::U_INDEX_DOF);
      FEFaceEvaluation_pres phi_pres_old_m(data, true, EquationData::P_INDEX_DOF),
                            phi_pres_old_p(data, false, EquationData::P_INDEX_DOF);
      FEFaceEvaluation_rho  phi_rho_old_m(data, true, EquationData::RHO_INDEX_DOF),
                            phi_rho_old_p(data, false, EquationData::RHO_INDEX_DOF);

      /*--- Loop over all internal faces ---*/
      for(unsigned face = face_range.first; face < face_range.second; ++face) {
        phi_rho_old_m.reinit(face);
        phi_rho_old_m.gather_evaluate(src[0], EvaluationFlags::values);
        phi_rho_old_p.reinit(face);
        phi_rho_old_p.gather_evaluate(src[0], EvaluationFlags::values);
        phi_u_old_m.reinit(face);
        phi_u_old_m.gather_evaluate(src[1], EvaluationFlags::values);
        phi_u_old_p.reinit(face);
        phi_u_old_p.gather_evaluate(src[1], EvaluationFlags::values);
        phi_pres_old_m.reinit(face);
        phi_pres_old_m.gather_evaluate(src[2], EvaluationFlags::values);
        phi_pres_old_p.reinit(face);
        phi_pres_old_p.gather_evaluate(src[2], EvaluationFlags::values);

        phi_m.reinit(face);
        phi_p.reinit(face);

        /*--- Loop over all quadrature points ---*/
        for(unsigned q = 0; q < phi_m.n_q_points; ++q) {
          const auto& n_minus = phi_m.get_normal_vector(q);

          /*--- Compute the quantities at the previous step ---*/
          const auto& rho_old_m  = phi_rho_old_m.get_value(q);
          const auto& rho_old_p  = phi_rho_old_p.get_value(q);
          const auto& u_old_m    = phi_u_old_m.get_value(q);
          const auto& u_old_p    = phi_u_old_p.get_value(q);
          const auto& pres_old_m = phi_pres_old_m.get_value(q);
          const auto& pres_old_p = phi_pres_old_p.get_value(q);

          /*--- Compute the numerical flux ---*/
          const auto& flux_num_explicit = a21*dt*num_flux.numerical_flux_momentum_explicit(rho_old_m, u_old_m,
                                                                                           rho_old_p, u_old_p,
                                                                                           n_minus);
          const auto& flux_num_implicit = a21_tilde*dt*num_flux.numerical_flux_momentum_implicit(pres_old_m,
                                                                                                 pres_old_p,
                                                                                                 n_minus);
          const auto& flux_num          = flux_num_explicit + flux_num_implicit;

          phi_m.submit_value(-flux_num, q);
          phi_p.submit_value(flux_num, q);
        }

        phi_m.integrate_scatter(EvaluationFlags::values, dst);
        phi_p.integrate_scatter(EvaluationFlags::values, dst);
      }
    }
    else if(IMEX_stage == 3) {
      /*--- We first start by declaring the suitable instances to read the available quantities. ---*/
      FEFaceEvaluation_u    phi_m(data, true, EquationData::U_INDEX_DOF),
                            phi_p(data, false, EquationData::U_INDEX_DOF),
                            phi_u_old_m(data, true, EquationData::U_INDEX_DOF),
                            phi_u_old_p(data, false, EquationData::U_INDEX_DOF),
                            phi_u_s_2_m(data, true, EquationData::U_INDEX_DOF),
                            phi_u_s_2_p(data, false, EquationData::U_INDEX_DOF);
      FEFaceEvaluation_pres phi_pres_old_m(data, true, EquationData::P_INDEX_DOF),
                            phi_pres_old_p(data, false, EquationData::P_INDEX_DOF),
                            phi_pres_s_2_m(data, true, EquationData::P_INDEX_DOF),
                            phi_pres_s_2_p(data, false, EquationData::P_INDEX_DOF);
      FEFaceEvaluation_rho  phi_rho_old_m(data, true, EquationData::RHO_INDEX_DOF),
                            phi_rho_old_p(data, false, EquationData::RHO_INDEX_DOF),
                            phi_rho_s_2_m(data, true, EquationData::RHO_INDEX_DOF),
                            phi_rho_s_2_p(data, false, EquationData::RHO_INDEX_DOF);

      /*---Loop over all internal faces ---*/
      for(unsigned face = face_range.first; face < face_range.second; ++face) {
        phi_rho_old_m.reinit(face);
        phi_rho_old_m.gather_evaluate(src[0], EvaluationFlags::values);
        phi_rho_old_p.reinit(face);
        phi_rho_old_p.gather_evaluate(src[0], EvaluationFlags::values);
        phi_u_old_m.reinit(face);
        phi_u_old_m.gather_evaluate(src[1], EvaluationFlags::values);
        phi_u_old_p.reinit(face);
        phi_u_old_p.gather_evaluate(src[1], EvaluationFlags::values);
        phi_pres_old_m.reinit(face);
        phi_pres_old_m.gather_evaluate(src[2], EvaluationFlags::values);
        phi_pres_old_p.reinit(face);
        phi_pres_old_p.gather_evaluate(src[2], EvaluationFlags::values);

        phi_rho_s_2_m.reinit(face);
        phi_rho_s_2_m.gather_evaluate(src[3], EvaluationFlags::values);
        phi_rho_s_2_p.reinit(face);
        phi_rho_s_2_p.gather_evaluate(src[3], EvaluationFlags::values);
        phi_u_s_2_m.reinit(face);
        phi_u_s_2_m.gather_evaluate(src[4], EvaluationFlags::values);
        phi_u_s_2_p.reinit(face);
        phi_u_s_2_p.gather_evaluate(src[4], EvaluationFlags::values);
        phi_pres_s_2_m.reinit(face);
        phi_pres_s_2_m.gather_evaluate(src[5], EvaluationFlags::values);
        phi_pres_s_2_p.reinit(face);
        phi_pres_s_2_p.gather_evaluate(src[5], EvaluationFlags::values);

        phi_m.reinit(face);
        phi_p.reinit(face);

        /*--- Loop over all quadrature points ---*/
        for(unsigned q = 0; q < phi_m.n_q_points; ++q) {
          const auto& n_minus = phi_m.get_normal_vector(q);

          /*--- Compute the quantities at the previous step ---*/
          const auto& rho_old_m  = phi_rho_old_m.get_value(q);
          const auto& rho_old_p  = phi_rho_old_p.get_value(q);
          const auto& u_old_m    = phi_u_old_m.get_value(q);
          const auto& u_old_p    = phi_u_old_p.get_value(q);
          const auto& pres_old_m = phi_pres_old_m.get_value(q);
          const auto& pres_old_p = phi_pres_old_p.get_value(q);

          /*--- Compute the quantities at the previous stage ---*/
          const auto& rho_s_2_m  = phi_rho_s_2_m.get_value(q);
          const auto& rho_s_2_p  = phi_rho_s_2_p.get_value(q);
          const auto& u_s_2_m    = phi_u_s_2_m.get_value(q);
          const auto& u_s_2_p    = phi_u_s_2_p.get_value(q);
          const auto& pres_s_2_m = phi_pres_s_2_m.get_value(q);
          const auto& pres_s_2_p = phi_pres_s_2_p.get_value(q);

          /*--- Compute the numerical flux ---*/
          const auto& flux_num_explicit = a31*dt*num_flux.numerical_flux_momentum_explicit(rho_old_m, u_old_m,
                                                                                           rho_old_p, u_old_p,
                                                                                           n_minus)
                                        + a32*dt*num_flux.numerical_flux_momentum_explicit(rho_s_2_m, u_s_2_m,
                                                                                           rho_s_2_p, u_s_2_p,
                                                                                           n_minus);
          const auto& flux_num_implicit = a31_tilde*dt*num_flux.numerical_flux_momentum_implicit(pres_old_m,
                                                                                                 pres_old_p,
                                                                                                 n_minus)
                                        + a32_tilde*dt*num_flux.numerical_flux_momentum_implicit(pres_s_2_m,
                                                                                                 pres_s_2_p,
                                                                                                 n_minus);
          const auto& flux_num          = flux_num_explicit + flux_num_implicit;

          phi_m.submit_value(-flux_num, q);
          phi_p.submit_value(flux_num, q);
        }

        phi_m.integrate_scatter(EvaluationFlags::values, dst);
        phi_p.integrate_scatter(EvaluationFlags::values, dst);
      }
    }
    else {
      /*--- We first start by declaring the suitable instances to read the available quantities. ---*/
      FEFaceEvaluation_u    phi_m(data, true, EquationData::U_INDEX_DOF),
                            phi_p(data, false, EquationData::U_INDEX_DOF),
                            phi_u_old_m(data, true, EquationData::U_INDEX_DOF),
                            phi_u_old_p(data, false, EquationData::U_INDEX_DOF),
                            phi_u_s_2_m(data, true, EquationData::U_INDEX_DOF),
                            phi_u_s_2_p(data, false, EquationData::U_INDEX_DOF),
                            phi_u_s_3_m(data, true, EquationData::U_INDEX_DOF),
                            phi_u_s_3_p(data, false, EquationData::U_INDEX_DOF);
      FEFaceEvaluation_pres phi_pres_old_m(data, true, EquationData::P_INDEX_DOF),
                            phi_pres_old_p(data, false, EquationData::P_INDEX_DOF),
                            phi_pres_s_2_m(data, true, EquationData::P_INDEX_DOF),
                            phi_pres_s_2_p(data, false, EquationData::P_INDEX_DOF),
                            phi_pres_s_3_m(data, true, EquationData::P_INDEX_DOF),
                            phi_pres_s_3_p(data, false, EquationData::P_INDEX_DOF);
      FEFaceEvaluation_rho  phi_rho_old_m(data, true, EquationData::RHO_INDEX_DOF),
                            phi_rho_old_p(data, false, EquationData::RHO_INDEX_DOF),
                            phi_rho_s_2_m(data, true, EquationData::RHO_INDEX_DOF),
                            phi_rho_s_2_p(data, false, EquationData::RHO_INDEX_DOF),
                            phi_rho_s_3_m(data, true, EquationData::RHO_INDEX_DOF),
                            phi_rho_s_3_p(data, false, EquationData::RHO_INDEX_DOF);

      /*--- Loop over all internal faces ---*/
      for(unsigned face = face_range.first; face < face_range.second; ++face) {
        phi_rho_old_m.reinit(face);
        phi_rho_old_m.gather_evaluate(src[0], EvaluationFlags::values);
        phi_rho_old_p.reinit(face);
        phi_rho_old_p.gather_evaluate(src[0], EvaluationFlags::values);
        phi_u_old_m.reinit(face);
        phi_u_old_m.gather_evaluate(src[1], EvaluationFlags::values);
        phi_u_old_p.reinit(face);
        phi_u_old_p.gather_evaluate(src[1], EvaluationFlags::values);
        phi_pres_old_m.reinit(face);
        phi_pres_old_m.gather_evaluate(src[2], EvaluationFlags::values);
        phi_pres_old_p.reinit(face);
        phi_pres_old_p.gather_evaluate(src[2], EvaluationFlags::values);

        phi_rho_s_2_m.reinit(face);
        phi_rho_s_2_m.gather_evaluate(src[3], EvaluationFlags::values);
        phi_rho_s_2_p.reinit(face);
        phi_rho_s_2_p.gather_evaluate(src[3], EvaluationFlags::values);
        phi_u_s_2_m.reinit(face);
        phi_u_s_2_m.gather_evaluate(src[4], EvaluationFlags::values);
        phi_u_s_2_p.reinit(face);
        phi_u_s_2_p.gather_evaluate(src[4], EvaluationFlags::values);
        phi_pres_s_2_m.reinit(face);
        phi_pres_s_2_m.gather_evaluate(src[5], EvaluationFlags::values);
        phi_pres_s_2_p.reinit(face);
        phi_pres_s_2_p.gather_evaluate(src[5], EvaluationFlags::values);

        phi_rho_s_3_m.reinit(face);
        phi_rho_s_3_m.gather_evaluate(src[6], EvaluationFlags::values);
        phi_rho_s_3_p.reinit(face);
        phi_rho_s_3_p.gather_evaluate(src[6], EvaluationFlags::values);
        phi_u_s_3_m.reinit(face);
        phi_u_s_3_m.gather_evaluate(src[7], EvaluationFlags::values);
        phi_u_s_3_p.reinit(face);
        phi_u_s_3_p.gather_evaluate(src[7], EvaluationFlags::values);
        phi_pres_s_3_m.reinit(face);
        phi_pres_s_3_m.gather_evaluate(src[8], EvaluationFlags::values);
        phi_pres_s_3_p.reinit(face);
        phi_pres_s_3_p.gather_evaluate(src[8], EvaluationFlags::values);

        phi_m.reinit(face);
        phi_p.reinit(face);

        /*--- Loop over all quadrature points ---*/
        for(unsigned q = 0; q < phi_m.n_q_points; ++q) {
          const auto& n_minus = phi_m.get_normal_vector(q);

          /*--- Compute the quantities at the previous step ---*/
          const auto& rho_old_m  = phi_rho_old_m.get_value(q);
          const auto& rho_old_p  = phi_rho_old_p.get_value(q);
          const auto& u_old_m    = phi_u_old_m.get_value(q);
          const auto& u_old_p    = phi_u_old_p.get_value(q);
          const auto& pres_old_m = phi_pres_old_m.get_value(q);
          const auto& pres_old_p = phi_pres_old_p.get_value(q);

          /*--- Compute the quantities at the previous stage ---*/
          const auto& rho_s_2_m  = phi_rho_s_2_m.get_value(q);
          const auto& rho_s_2_p  = phi_rho_s_2_p.get_value(q);
          const auto& u_s_2_m    = phi_u_s_2_m.get_value(q);
          const auto& u_s_2_p    = phi_u_s_2_p.get_value(q);
          const auto& pres_s_2_m = phi_pres_s_2_m.get_value(q);
          const auto& pres_s_2_p = phi_pres_s_2_p.get_value(q);

          /*--- Compute the quantities at the final stage ---*/
          const auto& rho_s_3_m  = phi_rho_s_3_m.get_value(q);
          const auto& rho_s_3_p  = phi_rho_s_3_p.get_value(q);
          const auto& u_s_3_m    = phi_u_s_3_m.get_value(q);
          const auto& u_s_3_p    = phi_u_s_3_p.get_value(q);
          const auto& pres_s_3_m = phi_pres_s_3_m.get_value(q);
          const auto& pres_s_3_p = phi_pres_s_3_p.get_value(q);

          /*--- Compute the numerical flux ---*/
          const auto& flux_num_explicit = b1*dt*num_flux.numerical_flux_momentum_explicit(rho_old_m, u_old_m,
                                                                                          rho_old_p, u_old_p,
                                                                                          n_minus) +
                                          b2*dt*num_flux.numerical_flux_momentum_explicit(rho_s_2_m, u_s_2_m,
                                                                                          rho_s_2_p, u_s_2_p,
                                                                                          n_minus) +
                                          b3*dt*num_flux.numerical_flux_momentum_explicit(rho_s_3_m, u_s_3_m,
                                                                                          rho_s_3_p, u_s_3_p,
                                                                                          n_minus);
          const auto& flux_num_implicit = b1*dt*num_flux.numerical_flux_momentum_implicit(pres_old_m,
                                                                                          pres_old_p,
                                                                                          n_minus)
                                        + b2*dt*num_flux.numerical_flux_momentum_implicit(pres_s_2_m,
                                                                                          pres_s_2_p,
                                                                                          n_minus)
                                        + b3*dt*num_flux.numerical_flux_momentum_implicit(pres_s_3_m,
                                                                                          pres_s_3_p,
                                                                                          n_minus);
          const auto& flux_num          = flux_num_explicit + flux_num_implicit;

          phi_m.submit_value(-flux_num, q);
          phi_p.submit_value(flux_num, q);
        }

        phi_m.integrate_scatter(EvaluationFlags::values, dst);
        phi_p.integrate_scatter(EvaluationFlags::values, dst);
      }
    }
  }

  // Assemble rhs boundary term of the momentum equation
  //
  template<unsigned dim,
           unsigned fe_degree_u, unsigned fe_degree_rho, unsigned fe_degree_p,
           unsigned n_q_points_1d, unsigned n_q_points_1d_boundary,
           typename Vec>
  void EULEROperator<dim,
                     fe_degree_u, fe_degree_rho, fe_degree_p,
                     n_q_points_1d, n_q_points_1d_boundary,
                     Vec>::
  assemble_rhs_boundary_term_momentum(const MatrixFree<dim, Number>&       data,
                                      Vec&                                 dst,
                                      const std::vector<Vec>&              src,
                                      const std::pair<unsigned, unsigned>& face_range) const {
    if(IMEX_stage == 2) {
      /*--- We first start by declaring the suitable instances to read the available quantities. ---*/
      FEFaceEvaluation_u_boundary    phi(data, true, EquationData::U_INDEX_DOF, 1);
      FEFaceEvaluation_pres_boundary phi_pres_old(data, true, EquationData::P_INDEX_DOF, 1);

      /*--- Loop over all boundary faces ---*/
      for(unsigned face = face_range.first; face < face_range.second; ++face) {
        phi_pres_old.reinit(face);
        phi_pres_old.gather_evaluate(src[2], EvaluationFlags::values);

        phi.reinit(face);

        /*--- Loop over all quadrature points ---*/
        for(unsigned q = 0; q < phi.n_q_points; ++q) {
          const auto& n_minus = phi.get_normal_vector(q);

          /*--- Compute the quantities at the previous step ---*/
          const auto& pres_old   = phi_pres_old.get_value(q);
          const auto& pres_old_D = pres_old;

          phi.submit_value(-a21_tilde*dt*
                            num_flux.numerical_flux_momentum_implicit(pres_old, pres_old_D, n_minus), q);
        }

        phi.integrate_scatter(EvaluationFlags::values, dst);
      }
    }
    else if(IMEX_stage == 3) {
      /*--- We first start by declaring the suitable instances to read the available quantities. ---*/
      FEFaceEvaluation_u_boundary    phi(data, true, EquationData::U_INDEX_DOF, 1);
      FEFaceEvaluation_pres_boundary phi_pres_old(data, true, EquationData::P_INDEX_DOF, 1),
                                     phi_pres_s_2(data, true, EquationData::P_INDEX_DOF, 1);

      /*--- Loop over all boundary faces ---*/
      for(unsigned face = face_range.first; face < face_range.second; ++face) {
        phi_pres_old.reinit(face);
        phi_pres_old.gather_evaluate(src[2], EvaluationFlags::values);

        phi_pres_s_2.reinit(face);
        phi_pres_s_2.gather_evaluate(src[5], EvaluationFlags::values);

        phi.reinit(face);

        /*--- Loop over all quadrature points ---*/
        for(unsigned q = 0; q < phi.n_q_points; ++q) {
          const auto& n_minus = phi.get_normal_vector(q);

          /*--- Compute the quantities at the previous step ---*/
          const auto& pres_old   = phi_pres_old.get_value(q);
          const auto& pres_old_D = pres_old;

          /*--- Compute the quantities at the previous stage ---*/
          const auto& pres_s_2   = phi_pres_s_2.get_value(q);
          const auto& pres_s_2_D = pres_s_2;

          phi.submit_value(-a31_tilde*dt*
                            num_flux.numerical_flux_momentum_implicit(pres_old, pres_old_D, n_minus)
                           -a32_tilde*dt*
                            num_flux.numerical_flux_momentum_implicit(pres_s_2, pres_s_2_D, n_minus), q);
        }

        phi.integrate_scatter(EvaluationFlags::values, dst);
      }
    }
    else {
      /*--- We first start by declaring the suitable instances to read the available quantities. ---*/
      FEFaceEvaluation_u_boundary    phi(data, true, EquationData::U_INDEX_DOF, 1);
      FEFaceEvaluation_pres_boundary phi_pres_old(data, true, EquationData::P_INDEX_DOF, 1),
                                     phi_pres_s_2(data, true, EquationData::P_INDEX_DOF, 1),
                                     phi_pres_s_3(data, true, EquationData::P_INDEX_DOF, 1);

      /*--- Loop over all boundary faces ---*/
      for(unsigned face = face_range.first; face < face_range.second; ++face) {
        phi_pres_old.reinit(face);
        phi_pres_old.gather_evaluate(src[2], EvaluationFlags::values);

        phi_pres_s_2.reinit(face);
        phi_pres_s_2.gather_evaluate(src[5], EvaluationFlags::values);

        phi_pres_s_3.reinit(face);
        phi_pres_s_3.gather_evaluate(src[8], EvaluationFlags::values);

        phi.reinit(face);

        /*--- Loop over all quadrature points ---*/
        for(unsigned q = 0; q < phi.n_q_points; ++q) {
          const auto& n_minus = phi.get_normal_vector(q);

          /*--- Compute the quantities at the previous step ---*/
          const auto& pres_old   = phi_pres_old.get_value(q);
          const auto& pres_old_D = pres_old;

          /*--- Compute the quantities at the previous stage ---*/
          const auto& pres_s_2   = phi_pres_s_2.get_value(q);
          const auto& pres_s_2_D = pres_s_2;

          /*--- Compute the quantities at the final steage---*/
          const auto& pres_s_3   = phi_pres_s_3.get_value(q);
          const auto& pres_s_3_D = pres_s_3;

          phi.submit_value(-b1*dt*
                            num_flux.numerical_flux_momentum_implicit(pres_old, pres_old_D, n_minus)
                           -b2*dt*
                            num_flux.numerical_flux_momentum_implicit(pres_s_2, pres_s_2_D, n_minus)
                           -b3*dt*
                            num_flux.numerical_flux_momentum_implicit(pres_s_3, pres_s_3_D, n_minus), q);
        }

        phi.integrate_scatter(EvaluationFlags::values, dst);
      }
    }
  }

  // Put together all the previous steps for the momentum equation
  //
  template<unsigned dim,
           unsigned fe_degree_u, unsigned fe_degree_rho, unsigned fe_degree_p,
           unsigned n_q_points_1d, unsigned n_q_points_1d_boundary,
           typename Vec>
  void EULEROperator<dim,
                     fe_degree_u, fe_degree_rho, fe_degree_p,
                     n_q_points_1d, n_q_points_1d_boundary,
                     Vec>::
  vmult_rhs_momentum(Vec& dst, const std::vector<Vec>& src) const {
    for(unsigned d = 0; d < src.size(); ++d) {
      src[d].update_ghost_values();
    }

    this->data->loop(&EULEROperator::assemble_rhs_cell_term_momentum,
                     &EULEROperator::assemble_rhs_face_term_momentum,
                     &EULEROperator::assemble_rhs_boundary_term_momentum,
                     this, dst, src, true,
                     MatrixFree<dim, Number>::DataAccessOnFaces::values,
                     MatrixFree<dim, Number>::DataAccessOnFaces::values);
  }

  // Assemble cell term for the velocity update
  //
  template<unsigned dim,
           unsigned fe_degree_u, unsigned fe_degree_rho, unsigned fe_degree_p,
           unsigned n_q_points_1d, unsigned n_q_points_1d_boundary,
           typename Vec>
  void EULEROperator<dim,
                     fe_degree_u, fe_degree_rho, fe_degree_p,
                     n_q_points_1d, n_q_points_1d_boundary,
                     Vec>::
  assemble_cell_term_velocity(const MatrixFree<dim, Number>&       data,
                              Vec&                                 dst,
                              const Vec&                           src,
                              const std::pair<unsigned, unsigned>& cell_range) const {
    /*--- We first start by declaring the suitable instances to read also available quantities.
          Since here we have just one 'src' vector, but we also need to deal with the current density,
          we employ the auxiliary vector 'rho_for_fixed' where we setted this information ---*/
    FEEvaluation<dim, fe_degree_u, fe_degree_u + 1, dim, Number> phi(data, EquationData::U_INDEX_DOF, 2);
    FEEvaluation<dim, fe_degree_rho, fe_degree_u + 1, 1, Number> phi_rho_for_fixed(data, EquationData::RHO_INDEX_DOF, 2);

    MatrixFreeOperators::CellwiseInverseMassMatrix<dim, fe_degree_u, dim, Number> inverse(phi);

    /*--- Loop over all cells ---*/
    for(unsigned cell = cell_range.first; cell < cell_range.second; ++cell) {
      phi_rho_for_fixed.reinit(cell);
      phi_rho_for_fixed.gather_evaluate(rho_for_fixed, EvaluationFlags::values);

      phi.reinit(cell);
      phi.read_dof_values(src);

      AlignedVector<VectorizedArray<Number>> inverse_jxw(phi.n_q_points);
      inverse.fill_inverse_JxW_values(inverse_jxw);

      /*--- Loop over all quadrature points to fill the inverse of the coefficient ---*/
      for(unsigned q = 0; q < phi.n_q_points; ++q) {
        inverse_jxw[q] *= 1.0/phi_rho_for_fixed.get_value(q);
      }

      inverse.apply(inverse_jxw, dim,
                    phi.begin_dof_values(),
                    phi.begin_dof_values());

      phi.set_dof_values(dst);
    }
  }

  // Assemble cell term for the pressure
  //
  template<unsigned dim,
           unsigned fe_degree_u, unsigned fe_degree_rho, unsigned fe_degree_p,
           unsigned n_q_points_1d, unsigned n_q_points_1d_boundary,
           typename Vec>
  void EULEROperator<dim,
                     fe_degree_u, fe_degree_rho, fe_degree_p,
                     n_q_points_1d, n_q_points_1d_boundary,
                     Vec>::
  assemble_cell_term_pressure(const MatrixFree<dim, Number>&       data,
                              Vec&                                 dst,
                              const Vec&                           src,
                              const std::pair<unsigned, unsigned>& cell_range) const {
    /*--- We first start by declaring the suitable instances to read quantities. This operator we are going to implement
          represents a rectangular matrix (we start from the pressure FE space and we end up with the velocity FE space).
          This is the reason of the distinction between 'phi' and 'phi_src'. ---*/
    FEEvaluation_u    phi(data, EquationData::U_INDEX_DOF);
    FEEvaluation_pres phi_src(data, EquationData::P_INDEX_DOF);

    /*--- This term changes between second and third stage of the IMEX scheme, but its structure not, so we do not need
          to explicitly distinguish the two cases as done for the rhs. ---*/
    const auto coeff = (IMEX_stage == 2) ? a22_tilde : a33_tilde;

    for(unsigned cell = cell_range.first; cell < cell_range.second; ++cell) {
      phi_src.reinit(cell);
      phi_src.gather_evaluate(src, EvaluationFlags::values);

      phi.reinit(cell);

      for(unsigned q = 0; q < phi.n_q_points; ++q) {
        /*--- Here we are testing against the divergence of the test function and, therefore, we employ 'submit_divergence'. ---*/
        phi.submit_divergence(-coeff*dt*(phi_src.get_value(q)/(Ma*Ma)), q);
      }

      phi.integrate_scatter(EvaluationFlags::gradients, dst);
    }
  }

  // Assemble face term for the pressure
  //
  template<unsigned dim,
           unsigned fe_degree_u, unsigned fe_degree_rho, unsigned fe_degree_p,
           unsigned n_q_points_1d, unsigned n_q_points_1d_boundary,
           typename Vec>
  void EULEROperator<dim,
                     fe_degree_u, fe_degree_rho, fe_degree_p,
                     n_q_points_1d, n_q_points_1d_boundary,
                     Vec>::
  assemble_face_term_pressure(const MatrixFree<dim, Number>&       data,
                              Vec&                                 dst,
                              const Vec&                           src,
                              const std::pair<unsigned, unsigned>& face_range) const {
    FEFaceEvaluation_u    phi_m(data, true, EquationData::U_INDEX_DOF),
                          phi_p(data, false, EquationData::U_INDEX_DOF);
    FEFaceEvaluation_pres phi_src_m(data, true, EquationData::P_INDEX_DOF),
                          phi_src_p(data, false, EquationData::P_INDEX_DOF);

    /*--- This term changes between second and third stage of the IMEX scheme, but its structure not, so we do not need
          to explicitly distinguish the two cases as done for the rhs. ---*/
    const auto coeff = (IMEX_stage == 2) ? a22_tilde : a33_tilde;

    /*--- Loop over all internal faces ---*/
    for(unsigned face = face_range.first; face < face_range.second; ++face) {
      phi_src_m.reinit(face);
      phi_src_m.gather_evaluate(src, EvaluationFlags::values);
      phi_src_p.reinit(face);
      phi_src_p.gather_evaluate(src, EvaluationFlags::values);

      phi_m.reinit(face);
      phi_p.reinit(face);

      /*--- Loop over all quadrature points ---*/
      for(unsigned q = 0; q < phi_m.n_q_points; ++q) {
        const auto& n_minus  = phi_m.get_normal_vector(q);

        const auto& avg_term = 0.5*(phi_src_m.get_value(q) +
                                    phi_src_p.get_value(q));

        phi_m.submit_value(coeff*dt*(avg_term/(Ma*Ma)*n_minus), q);
        phi_p.submit_value(-coeff*dt*(avg_term/(Ma*Ma)*n_minus), q);
      }

      phi_m.integrate_scatter(EvaluationFlags::values, dst);
      phi_p.integrate_scatter(EvaluationFlags::values, dst);
    }
  }

  // Assemble boundary term for the pressure
  //
  template<unsigned dim,
           unsigned fe_degree_u, unsigned fe_degree_rho, unsigned fe_degree_p,
           unsigned n_q_points_1d, unsigned n_q_points_1d_boundary,
           typename Vec>
  void EULEROperator<dim,
                     fe_degree_u, fe_degree_rho, fe_degree_p,
                     n_q_points_1d, n_q_points_1d_boundary,
                     Vec>::
  assemble_boundary_term_pressure(const MatrixFree<dim, Number>&               data,
                                  Vec&                                         dst,
                                  const Vec&                                   src,
                                  const std::pair<unsigned, unsigned>& face_range) const {
    FEFaceEvaluation_u_boundary    phi(data, true, EquationData::U_INDEX_DOF, 1);
    FEFaceEvaluation_pres_boundary phi_src(data, true, EquationData::P_INDEX_DOF, 1);

    /*--- This term changes between second and third stage of the IMEX scheme, but its structure not, so we do not need
          to explicitly distinguish the two cases as done for the rhs. ---*/
    const auto coeff = (IMEX_stage == 2) ? a22_tilde : a33_tilde;

    /*--- Loop over all boundary faces ---*/
    for(unsigned face = face_range.first; face < face_range.second; ++face) {
      phi_src.reinit(face);
      phi_src.gather_evaluate(src, EvaluationFlags::values);

      phi.reinit(face);

      /*--- Loop over all quadrature points ---*/
      for(unsigned q = 0; q < phi.n_q_points; ++q) {
        const auto& n_minus      = phi.get_normal_vector(q);

        const auto& pres_fixed_D = phi_src.get_value(q);

        const auto& avg_term     = 0.5*(phi_src.get_value(q) +
                                        pres_fixed_D);

        phi.submit_value(coeff*dt*(avg_term/(Ma*Ma)*n_minus), q);
      }

      phi.integrate_scatter(EvaluationFlags::values, dst);
    }
  }


  //////////////////////////////////////////////////////////////
  /*---- ASSEMBLING LINEAR AND BILINEAR FORMS FOR THE ENERGY EQUATION ---*/
  /////////////////////////////////////////////////////////////

  // Assemble rhs cell term of the energy equation
  //
  template<unsigned dim,
           unsigned fe_degree_u, unsigned fe_degree_rho, unsigned fe_degree_p,
           unsigned n_q_points_1d, unsigned n_q_points_1d_boundary,
           typename Vec>
  void EULEROperator<dim,
                     fe_degree_u, fe_degree_rho, fe_degree_p,
                     n_q_points_1d, n_q_points_1d_boundary,
                     Vec>::
  assemble_rhs_cell_term_energy(const MatrixFree<dim, Number>&       data,
                                Vec&                                 dst,
                                const std::vector<Vec>&              src,
                                const std::pair<unsigned, unsigned>& cell_range) const {
    if(IMEX_stage == 2) {
      /*--- We first start by declaring the suitable instances to read the available quantities. ---*/
      FEEvaluation_pres phi(data, EquationData::P_INDEX_DOF),
                        phi_pres_old(data, EquationData::P_INDEX_DOF);
      FEEvaluation_u    phi_u_old(data, EquationData::U_INDEX_DOF),
                        phi_u_fixed(data, EquationData::U_INDEX_DOF);
      FEEvaluation_rho  phi_rho_old(data, EquationData::RHO_INDEX_DOF),
                        phi_rho_s_2(data, EquationData::RHO_INDEX_DOF);

      /*--- Loop over all cells ---*/
      for(unsigned cell = cell_range.first; cell < cell_range.second; ++cell) {
        phi_rho_old.reinit(cell);
        phi_rho_old.gather_evaluate(src[0], EvaluationFlags::values);
        phi_u_old.reinit(cell);
        phi_u_old.gather_evaluate(src[1], EvaluationFlags::values);
        phi_pres_old.reinit(cell);
        phi_pres_old.gather_evaluate(src[2], EvaluationFlags::values);

        phi_rho_s_2.reinit(cell);
        phi_rho_s_2.gather_evaluate(src[3], EvaluationFlags::values);
        phi_u_fixed.reinit(cell);
        phi_u_fixed.gather_evaluate(src[4], EvaluationFlags::values);

        phi.reinit(cell);

        /*--- Loop over all quadrature points ---*/
        for(unsigned q = 0; q < phi.n_q_points; ++q) {
          /*--- Compute the quantities at the previous step ---*/
          const auto& rho_old  = phi_rho_old.get_value(q);
          const auto& u_old    = phi_u_old.get_value(q);
          const auto& pres_old = phi_pres_old.get_value(q);

          /*--- We assign to the rhs the contribution due to kinetic energy in the fixed point loop ---*/
          const auto& rho_s_2 = phi_rho_s_2.get_value(q);
          const auto& u_fixed = phi_u_fixed.get_value(q);

          phi.submit_value(1.0/(EquationData::Cp_Cv - 1.0)*pres_old +
                           rho_old*(0.5*(Ma*Ma)*scalar_product(u_old, u_old)) -
                           rho_s_2*(0.5*(Ma*Ma)*scalar_product(u_fixed, u_fixed)) -
                           a21_tilde*dt*((Ma*Ma)/(Fr*Fr)*rho_old*u_old[dim - 1]) -
                           a22_tilde*dt*((Ma*Ma)/(Fr*Fr)*rho_s_2*u_fixed[dim - 1]), q);
          phi.submit_gradient(a21*dt*(0.5*(Ma*Ma)*scalar_product(u_old, u_old)*rho_old*u_old) +
                              a21_tilde*dt*(EquationData::Cp_Cv/(EquationData::Cp_Cv - 1.0)*
                                            pres_old*u_old), q);
          /*--- The specific enthalpy is computed with the generic relation e + p/rho ---*/
        }

        phi.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
      }
    }
    else if(IMEX_stage == 3) {
      /*--- We first start by declaring the suitable instances to read the available quantities. ---*/
      FEEvaluation_pres phi(data, EquationData::P_INDEX_DOF),
                        phi_pres_old(data, EquationData::P_INDEX_DOF),
                        phi_pres_s_2(data, EquationData::P_INDEX_DOF);
      FEEvaluation_u    phi_u_old(data, EquationData::U_INDEX_DOF),
                        phi_u_s_2(data, EquationData::U_INDEX_DOF),
                        phi_u_fixed(data, EquationData::U_INDEX_DOF);
      FEEvaluation_rho  phi_rho_old(data, EquationData::RHO_INDEX_DOF),
                        phi_rho_s_2(data, EquationData::RHO_INDEX_DOF),
                        phi_rho_s_3(data, EquationData::RHO_INDEX_DOF);

      /*--- Loop over all cells ---*/
      for(unsigned cell = cell_range.first; cell < cell_range.second; ++cell) {
        phi_rho_old.reinit(cell);
        phi_rho_old.gather_evaluate(src[0], EvaluationFlags::values);
        phi_u_old.reinit(cell);
        phi_u_old.gather_evaluate(src[1], EvaluationFlags::values);
        phi_pres_old.reinit(cell);
        phi_pres_old.gather_evaluate(src[2], EvaluationFlags::values);

        phi_rho_s_2.reinit(cell);
        phi_rho_s_2.gather_evaluate(src[3], EvaluationFlags::values);
        phi_u_s_2.reinit(cell);
        phi_u_s_2.gather_evaluate(src[4], EvaluationFlags::values);
        phi_pres_s_2.reinit(cell);
        phi_pres_s_2.gather_evaluate(src[5], EvaluationFlags::values);

        phi_rho_s_3.reinit(cell);
        phi_rho_s_3.gather_evaluate(src[6], EvaluationFlags::values);
        phi_u_fixed.reinit(cell);
        phi_u_fixed.gather_evaluate(src[7], EvaluationFlags::values);

        phi.reinit(cell);

        /*--- Loop over all quadrature points ---*/
        for(unsigned q = 0; q < phi.n_q_points; ++q) {
          /*--- Compute the quantities at the previous step ---*/
          const auto& rho_old  = phi_rho_old.get_value(q);
          const auto& u_old    = phi_u_old.get_value(q);
          const auto& pres_old = phi_pres_old.get_value(q);

          /*--- Compute the quantities at the previous stage ---*/
          const auto& rho_s_2  = phi_rho_s_2.get_value(q);
          const auto& u_s_2    = phi_u_s_2.get_value(q);
          const auto& pres_s_2 = phi_pres_s_2.get_value(q);

          /*--- We assign to the rhs the contribution due to kinetic energy in the fixed point loop ---*/
          const auto& rho_s_3  = phi_rho_s_3.get_value(q);
          const auto& u_fixed  = phi_u_fixed.get_value(q);

          phi.submit_value(1.0/(EquationData::Cp_Cv - 1.0)*pres_old +
                           rho_old*(0.5*(Ma*Ma)*scalar_product(u_old, u_old)) -
                           rho_s_3*(0.5*(Ma*Ma)*scalar_product(u_fixed, u_fixed)) -
                           a31_tilde*dt*((Ma*Ma)/(Fr*Fr)*rho_old*u_old[dim - 1]) -
                           a32_tilde*dt*((Ma*Ma)/(Fr*Fr)*rho_s_2*u_s_2[dim - 1]) -
                           a33_tilde*dt*((Ma*Ma)/(Fr*Fr)*rho_s_3*u_fixed[dim - 1]), q);
          phi.submit_gradient(a31*dt*(0.5*(Ma*Ma)*scalar_product(u_old, u_old)*rho_old*u_old) +
                              a31_tilde*dt*(EquationData::Cp_Cv/(EquationData::Cp_Cv - 1.0)*
                                            pres_old*u_old) +
                              a32*dt*(0.5*(Ma*Ma)*scalar_product(u_s_2, u_s_2)*rho_s_2*u_s_2) +
                              a32_tilde*dt*(EquationData::Cp_Cv/(EquationData::Cp_Cv - 1.0)*
                                            pres_s_2*u_s_2), q);
        }

        phi.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
      }
    }
    else {
      /*--- We first start by declaring the suitable instances to read the available quantities. ---*/
      FEEvaluation_pres phi(data, EquationData::P_INDEX_DOF),
                        phi_pres_old(data, EquationData::P_INDEX_DOF),
                        phi_pres_s_2(data, EquationData::P_INDEX_DOF),
                        phi_pres_s_3(data, EquationData::P_INDEX_DOF);
      FEEvaluation_u    phi_u_old(data, EquationData::U_INDEX_DOF),
                        phi_u_s_2(data, EquationData::U_INDEX_DOF),
                        phi_u_s_3(data, EquationData::U_INDEX_DOF),
                        phi_u_curr(data, EquationData::U_INDEX_DOF);
      FEEvaluation_rho  phi_rho_old(data, EquationData::RHO_INDEX_DOF),
                        phi_rho_s_2(data, EquationData::RHO_INDEX_DOF),
                        phi_rho_s_3(data, EquationData::RHO_INDEX_DOF),
                        phi_rho_curr(data, EquationData::RHO_INDEX_DOF);

      /*--- Loop over all cells ---*/
      for(unsigned cell = cell_range.first; cell < cell_range.second; ++cell) {
        phi_rho_old.reinit(cell);
        phi_rho_old.gather_evaluate(src[0], EvaluationFlags::values);
        phi_u_old.reinit(cell);
        phi_u_old.gather_evaluate(src[1], EvaluationFlags::values);
        phi_pres_old.reinit(cell);
        phi_pres_old.gather_evaluate(src[2], EvaluationFlags::values);

        phi_rho_s_2.reinit(cell);
        phi_rho_s_2.gather_evaluate(src[3], EvaluationFlags::values);
        phi_u_s_2.reinit(cell);
        phi_u_s_2.gather_evaluate(src[4], EvaluationFlags::values);
        phi_pres_s_2.reinit(cell);
        phi_pres_s_2.gather_evaluate(src[5], EvaluationFlags::values);

        phi_rho_s_3.reinit(cell);
        phi_rho_s_3.gather_evaluate(src[6], EvaluationFlags::values);
        phi_u_s_3.reinit(cell);
        phi_u_s_3.gather_evaluate(src[7], EvaluationFlags::values);
        phi_pres_s_3.reinit(cell);
        phi_pres_s_3.gather_evaluate(src[8], EvaluationFlags::values);

        phi_rho_curr.reinit(cell);
        phi_rho_curr.gather_evaluate(src[9], EvaluationFlags::values);
        phi_u_curr.reinit(cell);
        phi_u_curr.gather_evaluate(src[10], EvaluationFlags::values);

        phi.reinit(cell);

        /*--- Loop over all quadrature points ---*/
        for(unsigned q = 0; q < phi.n_q_points; ++q) {
          /*--- Compute the quantities at the previous step ---*/
          const auto& rho_old  = phi_rho_old.get_value(q);
          const auto& u_old    = phi_u_old.get_value(q);
          const auto& pres_old = phi_pres_old.get_value(q);

          /*--- Compute the quantities at the second stage ---*/
          const auto& rho_s_2  = phi_rho_s_2.get_value(q);
          const auto& u_s_2    = phi_u_s_2.get_value(q);
          const auto& pres_s_2 = phi_pres_s_2.get_value(q);

          /*--- Compute the quantities at the final stage ---*/
          const auto& rho_s_3  = phi_rho_s_3.get_value(q);
          const auto& u_s_3    = phi_u_s_3.get_value(q);
          const auto& pres_s_3 = phi_pres_s_3.get_value(q);

          /*--- Assign to rhs the contribution of the (already updated) kinetic energy ---*/
          const auto& rho_curr = phi_rho_curr.get_value(q);
          const auto& u_curr   = phi_u_curr.get_value(q);

          phi.submit_value(1.0/(EquationData::Cp_Cv - 1.0)*pres_old +
                           rho_old*(0.5*(Ma*Ma)*scalar_product(u_old, u_old)) -
                           rho_curr*(0.5*(Ma*Ma)*scalar_product(u_curr, u_curr)) -
                           b1*dt*((Ma*Ma)/(Fr*Fr)*rho_old*u_old[dim - 1]) -
                           b2*dt*((Ma*Ma)/(Fr*Fr)*rho_s_2*u_s_2[dim - 1]) -
                           b3*dt*((Ma*Ma)/(Fr*Fr)*rho_s_3*u_s_3[dim - 1]), q);
          phi.submit_gradient(b1*dt*(0.5*(Ma*Ma)*scalar_product(u_old, u_old)*rho_old*u_old) +
                              b1*dt*(EquationData::Cp_Cv/(EquationData::Cp_Cv - 1.0)*
                                     pres_old*u_old) +
                              b2*dt*(0.5*(Ma*Ma)*scalar_product(u_s_2, u_s_2)*rho_s_2*u_s_2) +
                              b2*dt*(EquationData::Cp_Cv/(EquationData::Cp_Cv - 1.0)*
                                     pres_s_2*u_s_2) +
                              b3*dt*(0.5*(Ma*Ma)*scalar_product(u_s_3, u_s_3)*rho_s_3*u_s_3) +
                              b3*dt*(EquationData::Cp_Cv/(EquationData::Cp_Cv - 1.0)*
                                     pres_s_3*u_s_3), q);
        }

        phi.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
      }
    }
  }

  // Assemble rhs face term of the energy equation
  //
  template<unsigned dim,
           unsigned fe_degree_u, unsigned fe_degree_rho, unsigned fe_degree_p,
           unsigned n_q_points_1d, unsigned n_q_points_1d_boundary,
           typename Vec>
  void EULEROperator<dim,
                     fe_degree_u, fe_degree_rho, fe_degree_p,
                     n_q_points_1d, n_q_points_1d_boundary,
                     Vec>::
  assemble_rhs_face_term_energy(const MatrixFree<dim, Number>&       data,
                                Vec&                                 dst,
                                const std::vector<Vec>&              src,
                                const std::pair<unsigned, unsigned>& face_range) const {
    if(IMEX_stage == 2) {
      /*--- We first start by declaring the suitable instances to read the available quantities. ---*/
      FEFaceEvaluation_pres phi_m(data, true, EquationData::P_INDEX_DOF),
                            phi_p(data, false, EquationData::P_INDEX_DOF),
                            phi_pres_old_m(data, true, EquationData::P_INDEX_DOF),
                            phi_pres_old_p(data, false, EquationData::P_INDEX_DOF),
                            phi_pres_fixed_m(data, true, EquationData::P_INDEX_DOF),
                            phi_pres_fixed_p(data, false, EquationData::P_INDEX_DOF);
      FEFaceEvaluation_u    phi_u_old_m(data, true, EquationData::U_INDEX_DOF),
                            phi_u_old_p(data, false, EquationData::U_INDEX_DOF),
                            phi_u_fixed_m(data, true, EquationData::U_INDEX_DOF),
                            phi_u_fixed_p(data, false, EquationData::U_INDEX_DOF);
      FEFaceEvaluation_rho  phi_rho_old_m(data, true, EquationData::RHO_INDEX_DOF),
                            phi_rho_old_p(data, false, EquationData::RHO_INDEX_DOF);

      /*--- Loop over all internal faces ---*/
      for(unsigned face = face_range.first; face < face_range.second; ++face) {
        phi_rho_old_m.reinit(face);
        phi_rho_old_m.gather_evaluate(src[0], EvaluationFlags::values);
        phi_rho_old_p.reinit(face);
        phi_rho_old_p.gather_evaluate(src[0], EvaluationFlags::values);
        phi_u_old_m.reinit(face);
        phi_u_old_m.gather_evaluate(src[1], EvaluationFlags::values);
        phi_u_old_p.reinit(face);
        phi_u_old_p.gather_evaluate(src[1], EvaluationFlags::values);
        phi_pres_old_m.reinit(face);
        phi_pres_old_m.gather_evaluate(src[2], EvaluationFlags::values);
        phi_pres_old_p.reinit(face);
        phi_pres_old_p.gather_evaluate(src[2], EvaluationFlags::values);

        phi_u_fixed_m.reinit(face);
        phi_u_fixed_m.gather_evaluate(src[4], EvaluationFlags::values);
        phi_u_fixed_p.reinit(face);
        phi_u_fixed_p.gather_evaluate(src[4], EvaluationFlags::values);
        phi_pres_fixed_m.reinit(face);
        phi_pres_fixed_m.gather_evaluate(src[5], EvaluationFlags::values);
        phi_pres_fixed_p.reinit(face);
        phi_pres_fixed_p.gather_evaluate(src[5], EvaluationFlags::values);

        phi_m.reinit(face);
        phi_p.reinit(face);

        /*--- Loop over quadrature points ---*/
        for(unsigned q = 0; q < phi_m.n_q_points; ++q) {
          const auto& n_minus = phi_m.get_normal_vector(q);

          /*--- Compute the quantities at the previous step ---*/
          const auto& rho_old_m  = phi_rho_old_m.get_value(q);
          const auto& rho_old_p  = phi_rho_old_p.get_value(q);
          const auto& u_old_m    = phi_u_old_m.get_value(q);
          const auto& u_old_p    = phi_u_old_p.get_value(q);
          const auto& pres_old_m = phi_pres_old_m.get_value(q);
          const auto& pres_old_p = phi_pres_old_p.get_value(q);

          /*--- Compute the quantities at the current stage ---*/
          const auto& u_fixed_m        = phi_u_fixed_m.get_value(q);
          const auto& u_fixed_p        = phi_u_fixed_p.get_value(q);
          const auto& pres_fixed_m     = phi_pres_fixed_m.get_value(q);
          const auto& pres_fixed_p     = phi_pres_fixed_p.get_value(q);

          const auto& lambda_fixed     = num_flux.compute_lambda(u_fixed_m, u_fixed_p, n_minus);
          const auto& jump_rho_e_fixed = 1.0/(EquationData::Cp_Cv - 1.0)*
                                         (pres_fixed_m - pres_fixed_p);

          /*--- Compute the numerical flux ---*/
          const auto& flux_num_explicit = a21*dt*num_flux.numerical_flux_energy_explicit(rho_old_m, u_old_m,
                                                                                         rho_old_p, u_old_p,
                                                                                         n_minus);
          const auto& flux_num_implicit = a21_tilde*dt*num_flux.numerical_flux_energy_implicit(u_old_m, pres_old_m,
                                                                                               u_old_p, pres_old_p,
                                                                                               n_minus)
                                        + a22_tilde*dt*(0.5*lambda_fixed*jump_rho_e_fixed);
          const auto& flux_num          = flux_num_explicit + flux_num_implicit;

          phi_m.submit_value(-flux_num, q);
          phi_p.submit_value(flux_num, q);
        }

        phi_m.integrate_scatter(EvaluationFlags::values, dst);
        phi_p.integrate_scatter(EvaluationFlags::values, dst);
      }
    }
    else if(IMEX_stage == 3) {
      /*--- We first start by declaring the suitable instances to read the available quantities. ---*/
      FEFaceEvaluation_pres phi_m(data, true, EquationData::P_INDEX_DOF),
                            phi_p(data, false, EquationData::P_INDEX_DOF),
                            phi_pres_old_m(data, true, EquationData::P_INDEX_DOF),
                            phi_pres_old_p(data, false, EquationData::P_INDEX_DOF),
                            phi_pres_s_2_m(data, true, EquationData::P_INDEX_DOF),
                            phi_pres_s_2_p(data, false, EquationData::P_INDEX_DOF),
                            phi_pres_fixed_m(data, true, EquationData::P_INDEX_DOF),
                            phi_pres_fixed_p(data, false, EquationData::P_INDEX_DOF);
      FEFaceEvaluation_u    phi_u_old_m(data, true, EquationData::U_INDEX_DOF),
                            phi_u_old_p(data, false, EquationData::U_INDEX_DOF),
                            phi_u_s_2_m(data, true, EquationData::U_INDEX_DOF),
                            phi_u_s_2_p(data, false, EquationData::U_INDEX_DOF),
                            phi_u_fixed_m(data, true, EquationData::U_INDEX_DOF),
                            phi_u_fixed_p(data, false, EquationData::U_INDEX_DOF);
      FEFaceEvaluation_rho  phi_rho_old_m(data, true, EquationData::RHO_INDEX_DOF),
                            phi_rho_old_p(data, false, EquationData::RHO_INDEX_DOF),
                            phi_rho_s_2_m(data, true, EquationData::RHO_INDEX_DOF),
                            phi_rho_s_2_p(data, false, EquationData::RHO_INDEX_DOF);

      /*--- Loop over all internal faces ---*/
      for(unsigned face = face_range.first; face < face_range.second; ++face) {
        phi_rho_old_m.reinit(face);
        phi_rho_old_m.gather_evaluate(src[0], EvaluationFlags::values);
        phi_rho_old_p.reinit(face);
        phi_rho_old_p.gather_evaluate(src[0], EvaluationFlags::values);
        phi_u_old_m.reinit(face);
        phi_u_old_m.gather_evaluate(src[1], EvaluationFlags::values);
        phi_u_old_p.reinit(face);
        phi_u_old_p.gather_evaluate(src[1], EvaluationFlags::values);
        phi_pres_old_m.reinit(face);
        phi_pres_old_m.gather_evaluate(src[2], EvaluationFlags::values);
        phi_pres_old_p.reinit(face);
        phi_pres_old_p.gather_evaluate(src[2], EvaluationFlags::values);

        phi_rho_s_2_m.reinit(face);
        phi_rho_s_2_m.gather_evaluate(src[3], EvaluationFlags::values);
        phi_rho_s_2_p.reinit(face);
        phi_rho_s_2_p.gather_evaluate(src[3], EvaluationFlags::values);
        phi_u_s_2_m.reinit(face);
        phi_u_s_2_m.gather_evaluate(src[4], EvaluationFlags::values);
        phi_u_s_2_p.reinit(face);
        phi_u_s_2_p.gather_evaluate(src[4], EvaluationFlags::values);
        phi_pres_s_2_m.reinit(face);
        phi_pres_s_2_m.gather_evaluate(src[5], EvaluationFlags::values);
        phi_pres_s_2_p.reinit(face);
        phi_pres_s_2_p.gather_evaluate(src[5], EvaluationFlags::values);

        phi_u_fixed_m.reinit(face);
        phi_u_fixed_m.gather_evaluate(src[7], EvaluationFlags::values);
        phi_u_fixed_p.reinit(face);
        phi_u_fixed_p.gather_evaluate(src[7], EvaluationFlags::values);
        phi_pres_fixed_m.reinit(face);
        phi_pres_fixed_m.gather_evaluate(src[8], EvaluationFlags::values);
        phi_pres_fixed_p.reinit(face);
        phi_pres_fixed_p.gather_evaluate(src[8], EvaluationFlags::values);

        phi_m.reinit(face);
        phi_p.reinit(face);

        /*--- Loop over all quadrature points ---*/
        for(unsigned q = 0; q < phi_m.n_q_points; ++q) {
          const auto& n_minus = phi_m.get_normal_vector(q);

          /*--- Compute the quantities at the previous step ---*/
          const auto& rho_old_m  = phi_rho_old_m.get_value(q);
          const auto& rho_old_p  = phi_rho_old_p.get_value(q);
          const auto& u_old_m    = phi_u_old_m.get_value(q);
          const auto& u_old_p    = phi_u_old_p.get_value(q);
          const auto& pres_old_m = phi_pres_old_m.get_value(q);
          const auto& pres_old_p = phi_pres_old_p.get_value(q);

          /*--- Compute the quantities at the previous stage ---*/
          const auto& rho_s_2_m  = phi_rho_s_2_m.get_value(q);
          const auto& rho_s_2_p  = phi_rho_s_2_p.get_value(q);
          const auto& u_s_2_m    = phi_u_s_2_m.get_value(q);
          const auto& u_s_2_p    = phi_u_s_2_p.get_value(q);
          const auto& pres_s_2_m = phi_pres_s_2_m.get_value(q);
          const auto& pres_s_2_p = phi_pres_s_2_p.get_value(q);

          /*--- Compute the quantities at the current stage ---*/
          const auto& u_fixed_m        = phi_u_fixed_m.get_value(q);
          const auto& u_fixed_p        = phi_u_fixed_p.get_value(q);
          const auto& pres_fixed_m     = phi_pres_fixed_m.get_value(q);
          const auto& pres_fixed_p     = phi_pres_fixed_p.get_value(q);

          const auto& lambda_fixed     = num_flux.compute_lambda(u_fixed_m, u_fixed_p, n_minus);
          const auto& jump_rho_e_fixed = 1.0/(EquationData::Cp_Cv - 1.0)*
                                         (pres_fixed_m - pres_fixed_p);

          /*--- Compute the numerical flux ---*/
          const auto& flux_num_explicit = a31*dt*num_flux.numerical_flux_energy_explicit(rho_old_m, u_old_m,
                                                                                         rho_old_p, u_old_p,
                                                                                         n_minus)
                                        + a32*dt*num_flux.numerical_flux_energy_explicit(rho_s_2_m, u_s_2_m,
                                                                                         rho_s_2_p, u_s_2_p,
                                                                                         n_minus);
          const auto& flux_num_implicit = a31_tilde*dt*num_flux.numerical_flux_energy_implicit(u_old_m, pres_old_m,
                                                                                               u_old_p, pres_old_p,
                                                                                               n_minus)
                                        + a32_tilde*dt*num_flux.numerical_flux_energy_implicit(u_s_2_m, pres_s_2_m,
                                                                                               u_s_2_p, pres_s_2_p,
                                                                                               n_minus)
                                        + a33_tilde*dt*(0.5*lambda_fixed*jump_rho_e_fixed);
          const auto& flux_num          = flux_num_explicit + flux_num_implicit;

          phi_m.submit_value(-flux_num, q);
          phi_p.submit_value(flux_num, q);
        }

        phi_m.integrate_scatter(EvaluationFlags::values, dst);
        phi_p.integrate_scatter(EvaluationFlags::values, dst);
      }
    }
    else {
      /*--- We first start by declaring the suitable instances to read the available quantities. ---*/
      FEFaceEvaluation_pres phi_m(data, true, EquationData::P_INDEX_DOF),
                            phi_p(data, false, EquationData::P_INDEX_DOF),
                            phi_pres_old_m(data, true, EquationData::P_INDEX_DOF),
                            phi_pres_old_p(data, false, EquationData::P_INDEX_DOF),
                            phi_pres_s_2_m(data, true, EquationData::P_INDEX_DOF),
                            phi_pres_s_2_p(data, false, EquationData::P_INDEX_DOF),
                            phi_pres_s_3_m(data, true, EquationData::P_INDEX_DOF),
                            phi_pres_s_3_p(data, false, EquationData::P_INDEX_DOF);
      FEFaceEvaluation_u    phi_u_old_m(data, true, EquationData::U_INDEX_DOF),
                            phi_u_old_p(data, false, EquationData::U_INDEX_DOF),
                            phi_u_s_2_m(data, true, EquationData::U_INDEX_DOF),
                            phi_u_s_2_p(data, false, EquationData::U_INDEX_DOF),
                            phi_u_s_3_m(data, true, EquationData::U_INDEX_DOF),
                            phi_u_s_3_p(data, false, EquationData::U_INDEX_DOF);
      FEFaceEvaluation_rho  phi_rho_old_m(data, true, EquationData::RHO_INDEX_DOF),
                            phi_rho_old_p(data, false, EquationData::RHO_INDEX_DOF),
                            phi_rho_s_2_m(data, true, EquationData::RHO_INDEX_DOF),
                            phi_rho_s_2_p(data, false, EquationData::RHO_INDEX_DOF),
                            phi_rho_s_3_m(data, true, EquationData::RHO_INDEX_DOF),
                            phi_rho_s_3_p(data, false, EquationData::RHO_INDEX_DOF);

      /*--- loop over all internal faces ---*/
      for(unsigned face = face_range.first; face < face_range.second; ++face) {
        phi_rho_old_m.reinit(face);
        phi_rho_old_m.gather_evaluate(src[0], EvaluationFlags::values);
        phi_rho_old_p.reinit(face);
        phi_rho_old_p.gather_evaluate(src[0], EvaluationFlags::values);
        phi_u_old_m.reinit(face);
        phi_u_old_m.gather_evaluate(src[1], EvaluationFlags::values);
        phi_u_old_p.reinit(face);
        phi_u_old_p.gather_evaluate(src[1], EvaluationFlags::values);
        phi_pres_old_m.reinit(face);
        phi_pres_old_m.gather_evaluate(src[2], EvaluationFlags::values);
        phi_pres_old_p.reinit(face);
        phi_pres_old_p.gather_evaluate(src[2], EvaluationFlags::values);

        phi_rho_s_2_m.reinit(face);
        phi_rho_s_2_m.gather_evaluate(src[3], EvaluationFlags::values);
        phi_rho_s_2_p.reinit(face);
        phi_rho_s_2_p.gather_evaluate(src[3], EvaluationFlags::values);
        phi_u_s_2_m.reinit(face);
        phi_u_s_2_m.gather_evaluate(src[4], EvaluationFlags::values);
        phi_u_s_2_p.reinit(face);
        phi_u_s_2_p.gather_evaluate(src[4], EvaluationFlags::values);
        phi_pres_s_2_m.reinit(face);
        phi_pres_s_2_m.gather_evaluate(src[5], EvaluationFlags::values);
        phi_pres_s_2_p.reinit(face);
        phi_pres_s_2_p.gather_evaluate(src[5], EvaluationFlags::values);

        phi_rho_s_3_m.reinit(face);
        phi_rho_s_3_m.gather_evaluate(src[6], EvaluationFlags::values);
        phi_rho_s_3_p.reinit(face);
        phi_rho_s_3_p.gather_evaluate(src[6], EvaluationFlags::values);
        phi_u_s_3_m.reinit(face);
        phi_u_s_3_m.gather_evaluate(src[7], EvaluationFlags::values);
        phi_u_s_3_p.reinit(face);
        phi_u_s_3_p.gather_evaluate(src[7], EvaluationFlags::values);
        phi_pres_s_3_m.reinit(face);
        phi_pres_s_3_m.gather_evaluate(src[8], EvaluationFlags::values);
        phi_pres_s_3_p.reinit(face);
        phi_pres_s_3_p.gather_evaluate(src[8], EvaluationFlags::values);

        phi_m.reinit(face);
        phi_p.reinit(face);

        /*--- Loop over all quadrature points ---*/
        for(unsigned q = 0; q < phi_m.n_q_points; ++q) {
          const auto& n_minus = phi_m.get_normal_vector(q);

          /*--- Compute the quantities at the previous step ---*/
          const auto& rho_old_m  = phi_rho_old_m.get_value(q);
          const auto& rho_old_p  = phi_rho_old_p.get_value(q);
          const auto& u_old_m    = phi_u_old_m.get_value(q);
          const auto& u_old_p    = phi_u_old_p.get_value(q);
          const auto& pres_old_m = phi_pres_old_m.get_value(q);
          const auto& pres_old_p = phi_pres_old_p.get_value(q);

          /*--- Compute the quantities at the second stage ---*/
          const auto& rho_s_2_m  = phi_rho_s_2_m.get_value(q);
          const auto& rho_s_2_p  = phi_rho_s_2_p.get_value(q);
          const auto& u_s_2_m    = phi_u_s_2_m.get_value(q);
          const auto& u_s_2_p    = phi_u_s_2_p.get_value(q);
          const auto& pres_s_2_m = phi_pres_s_2_m.get_value(q);
          const auto& pres_s_2_p = phi_pres_s_2_p.get_value(q);

          /*--- Compute the quantities at the final stage ---*/
          const auto& rho_s_3_m  = phi_rho_s_3_m.get_value(q);
          const auto& rho_s_3_p  = phi_rho_s_3_p.get_value(q);
          const auto& u_s_3_m    = phi_u_s_3_m.get_value(q);
          const auto& u_s_3_p    = phi_u_s_3_p.get_value(q);
          const auto& pres_s_3_m = phi_pres_s_3_m.get_value(q);
          const auto& pres_s_3_p = phi_pres_s_3_p.get_value(q);

          /*--- Compute the numerical flux ---*/
          const auto& flux_num = b1*dt*num_flux.numerical_flux_energy_explicit(rho_old_m, u_old_m,
                                                                               rho_old_p, u_old_p,
                                                                               n_minus)
                               + b1*dt*num_flux.numerical_flux_energy_implicit(u_old_m, pres_old_m,
                                                                               u_old_p, pres_old_p,
                                                                               n_minus)
                               + b2*dt*num_flux.numerical_flux_energy_explicit(rho_s_2_m, u_s_2_m,
                                                                               rho_s_2_p, u_s_2_p,
                                                                               n_minus)
                               + b2*dt*num_flux.numerical_flux_energy_implicit(u_s_2_m, pres_s_2_m,
                                                                               u_s_2_p, pres_s_2_p,
                                                                               n_minus)
                               + b3*dt*num_flux.numerical_flux_energy_explicit(rho_s_3_m, u_s_3_m,
                                                                               rho_s_3_p, u_s_3_p,
                                                                               n_minus)
                               + b3*dt*num_flux.numerical_flux_energy_implicit(u_s_3_m, pres_s_3_m,
                                                                               u_s_3_p, pres_s_3_p,
                                                                               n_minus);

          phi_m.submit_value(-flux_num, q);
          phi_p.submit_value(flux_num, q);
        }

        phi_m.integrate_scatter(EvaluationFlags::values, dst);
        phi_p.integrate_scatter(EvaluationFlags::values, dst);
      }
    }
  }

  // Put together all the previous steps for the energy equation
  //
  template<unsigned dim,
           unsigned fe_degree_u, unsigned fe_degree_rho, unsigned fe_degree_p,
           unsigned n_q_points_1d, unsigned n_q_points_1d_boundary,
           typename Vec>
  void EULEROperator<dim,
                     fe_degree_u, fe_degree_rho, fe_degree_p,
                     n_q_points_1d, n_q_points_1d_boundary,
                     Vec>::
  vmult_rhs_energy(Vec& dst, const std::vector<Vec>& src) const {
    for(unsigned d = 0; d < src.size(); ++d) {
      src[d].update_ghost_values();
    }

    this->data->loop(&EULEROperator::assemble_rhs_cell_term_energy,
                     &EULEROperator::assemble_rhs_face_term_energy,
                     &EULEROperator::assemble_rhs_boundary_term_energy,
                     this, dst, src, true,
                     MatrixFree<dim, Number>::DataAccessOnFaces::values,
                     MatrixFree<dim, Number>::DataAccessOnFaces::values);
  }

  // Assemble cell term for the contribution due to internal energy
  //
  template<unsigned dim,
           unsigned fe_degree_u, unsigned fe_degree_rho, unsigned fe_degree_p,
           unsigned n_q_points_1d, unsigned n_q_points_1d_boundary,
           typename Vec>
  void EULEROperator<dim,
                     fe_degree_u, fe_degree_rho, fe_degree_p,
                     n_q_points_1d, n_q_points_1d_boundary,
                     Vec>::
  assemble_inverse_cell_term_internal_energy(const MatrixFree<dim, Number>&       data,
                                             Vec&                                 dst,
                                             const Vec&                           src,
                                             const std::pair<unsigned, unsigned>& cell_range) const {
    FEEvaluation<dim, fe_degree_p, fe_degree_p + 1, 1, Number> phi(data, EquationData::P_INDEX_DOF, 2);

    MatrixFreeOperators::CellwiseInverseMassMatrix<dim, fe_degree_p, 1, Number> inverse(phi);

    /*--- Loop over all cells ---*/
    for(unsigned cell = cell_range.first; cell < cell_range.second; ++cell) {
      phi.reinit(cell);
      phi.read_dof_values(src);

      AlignedVector<VectorizedArray<Number>> inverse_jxw(phi.n_q_points);
      inverse.fill_inverse_JxW_values(inverse_jxw);

      /*--- Loop over all quadrature points to fill the inverse of the coefficient ---*/
      for(unsigned q = 0; q < phi.n_q_points; ++q) {
        inverse_jxw[q] *= (EquationData::Cp_Cv - 1.0);
      }

      inverse.apply(inverse_jxw, 1,
                    phi.begin_dof_values(),
                    phi.begin_dof_values());

      phi.set_dof_values(dst);
    }
  }

  // Assemble cell term for the contribution due to internal energy
  //
  template<unsigned dim,
           unsigned fe_degree_u, unsigned fe_degree_rho, unsigned fe_degree_p,
           unsigned n_q_points_1d, unsigned n_q_points_1d_boundary,
           typename Vec>
  void EULEROperator<dim,
                     fe_degree_u, fe_degree_rho, fe_degree_p,
                     n_q_points_1d, n_q_points_1d_boundary,
                     Vec>::
  assemble_cell_term_internal_energy(const MatrixFree<dim, Number>&               data,
                                     Vec&                                         dst,
                                     const Vec&                                   src,
                                     const std::pair<unsigned, unsigned>& cell_range) const {
    FEEvaluation<dim, fe_degree_p, fe_degree_p + 1, 1, Number> phi(data, EquationData::P_INDEX_DOF, 2);

    for(unsigned cell = cell_range.first; cell < cell_range.second; ++cell) {
      phi.reinit(cell);
      phi.gather_evaluate(src, EvaluationFlags::values);

      for(unsigned q = 0; q < phi.n_q_points; ++q) {
        /*--- For an ideal gas the part associated to the internal energy for a pressure based
              is just a modification of the mass matrix ---*/
        phi.submit_value(1.0/(EquationData::Cp_Cv - 1.0)*phi.get_value(q), q);
      }

      phi.integrate_scatter(EvaluationFlags::values, dst);
    }
  }

  // Assemble cell term for the contribution due to enthalpy
  //
  template<unsigned dim,
           unsigned fe_degree_u, unsigned fe_degree_rho, unsigned fe_degree_p,
           unsigned n_q_points_1d, unsigned n_q_points_1d_boundary,
           typename Vec>
  void EULEROperator<dim,
                     fe_degree_u, fe_degree_rho, fe_degree_p,
                     n_q_points_1d, n_q_points_1d_boundary,
                     Vec>::
  assemble_cell_term_enthalpy(const MatrixFree<dim, Number>&               data,
                              Vec&                                         dst,
                              const Vec&                                   src,
                              const std::pair<unsigned, unsigned>& cell_range) const {
    /*--- We first start by declaring the suitable instances to read also available quantities.
          Since here we have just one 'src' vector, but we also need to deal with the current pressure
          in the fixed point loop, we employ the auxiliary vector 'pres_fixed' where we setted this information ---*/
    FEEvaluation_pres phi(data, EquationData::P_INDEX_DOF),
                      phi_pres_fixed(data, EquationData::P_INDEX_DOF);
    FEEvaluation_u    phi_src(data, EquationData::U_INDEX_DOF);

    /*--- This term changes between second and third stage of the IMEX scheme, but its structure not, so we do not need
          to explicitly distinguish the two cases as done for the rhs. ---*/
    const auto coeff = (IMEX_stage == 2) ? a22_tilde : a33_tilde;

    /*--- Loop over all cells ---*/
    for(unsigned cell = cell_range.first; cell < cell_range.second; ++cell) {
      phi_pres_fixed.reinit(cell);
      phi_pres_fixed.gather_evaluate(pres_fixed, EvaluationFlags::values);

      phi_src.reinit(cell);
      phi_src.gather_evaluate(src, EvaluationFlags::values);

      phi.reinit(cell);

      /*--- loop over all quadrature points ---*/
      for(unsigned q = 0; q < phi.n_q_points; ++q) {
        const auto& pres_fixed = phi_pres_fixed.get_value(q);

        phi.submit_gradient(-coeff*dt*(EquationData::Cp_Cv/(EquationData::Cp_Cv - 1.0)*
                                       pres_fixed*phi_src.get_value(q)), q);
      }

      phi.integrate_scatter(EvaluationFlags::gradients, dst);
    }
  }

  // Assemble face term for the contribution due to enthalpy
  //
  template<unsigned dim,
           unsigned fe_degree_u, unsigned fe_degree_rho, unsigned fe_degree_p,
           unsigned n_q_points_1d, unsigned n_q_points_1d_boundary,
           typename Vec>
  void EULEROperator<dim,
                     fe_degree_u, fe_degree_rho, fe_degree_p,
                     n_q_points_1d, n_q_points_1d_boundary,
                     Vec>::
  assemble_face_term_enthalpy(const MatrixFree<dim, Number>&       data,
                              Vec&                                 dst,
                              const Vec&                           src,
                              const std::pair<unsigned, unsigned>& face_range) const {
    FEFaceEvaluation_pres phi_m(data, true, EquationData::P_INDEX_DOF),
                          phi_p(data, false, EquationData::P_INDEX_DOF),
                          phi_pres_fixed_m(data, true, EquationData::P_INDEX_DOF),
                          phi_pres_fixed_p(data, false, EquationData::P_INDEX_DOF);
    FEFaceEvaluation_u    phi_src_m(data, true, EquationData::U_INDEX_DOF),
                          phi_src_p(data, false, EquationData::U_INDEX_DOF);

    /*--- This term changes between second and third stage of the IMEX scheme, but its structure not, so we do not need
          to explicitly distinguish the two cases as done for the rhs. ---*/
    const auto coeff = (IMEX_stage == 2) ? a22_tilde : a33_tilde;

    /*--- Loop over all faces ---*/
    for(unsigned face = face_range.first; face < face_range.second; ++face) {
      phi_pres_fixed_m.reinit(face);
      phi_pres_fixed_m.gather_evaluate(pres_fixed, EvaluationFlags::values);
      phi_pres_fixed_p.reinit(face);
      phi_pres_fixed_p.gather_evaluate(pres_fixed, EvaluationFlags::values);

      phi_src_m.reinit(face);
      phi_src_m.gather_evaluate(src, EvaluationFlags::values);
      phi_src_p.reinit(face);
      phi_src_p.gather_evaluate(src, EvaluationFlags::values);

      phi_m.reinit(face);
      phi_p.reinit(face);

      /*--- Loop over all quadrature points ---*/
      for(unsigned q = 0; q < phi_m.n_q_points; ++q) {
        const auto& n_minus           = phi_m.get_normal_vector(q);

        const auto& pres_fixed_m      = phi_pres_fixed_m.get_value(q);
        const auto& pres_fixed_p      = phi_pres_fixed_p.get_value(q);

        const auto& avg_flux_enthalpy = 0.5*(EquationData::Cp_Cv/(EquationData::Cp_Cv - 1.0))*
                                        (pres_fixed_m*phi_src_m.get_value(q) +
                                         pres_fixed_p*phi_src_p.get_value(q));

        phi_m.submit_value(coeff*dt*scalar_product(avg_flux_enthalpy, n_minus), q);
        phi_p.submit_value(-coeff*dt*scalar_product(avg_flux_enthalpy, n_minus), q);
      }

      phi_m.integrate_scatter(EvaluationFlags::values, dst);
      phi_p.integrate_scatter(EvaluationFlags::values, dst);
    }
  }


  //////////////////////////////////////////////////////////////
  /*---- APPLICATION OF THE DIFFERENT LINEAR OPERATORS ---*/
  /////////////////////////////////////////////////////////////

  // Put together all previous steps
  //
  template<unsigned dim,
           unsigned fe_degree_u, unsigned fe_degree_rho, unsigned fe_degree_p,
           unsigned n_q_points_1d, unsigned n_q_points_1d_boundary,
           typename Vec>
  void EULEROperator<dim,
                     fe_degree_u, fe_degree_rho, fe_degree_p,
                     n_q_points_1d, n_q_points_1d_boundary,
                     Vec>::
  apply_add(Vec& dst, const Vec& src) const {
    AssertIndexRange(Euler_stage, EquationData::n_vars + 1);
    Assert(Euler_stage > 0, ExcInternalError());

    if(Euler_stage == EquationData::RHO_INDEX_SYSTEM) {
      this->data->cell_loop(&EULEROperator::assemble_cell_term_density,
                            this, dst, src, false);
    }
    else if(Euler_stage == EquationData::P_INDEX_SYSTEM) {
      if(IMEX_stage <= EquationData::n_stages) {
        this->data->cell_loop(&EULEROperator::assemble_cell_term_internal_energy,
                              this, dst, src, false);

        /*--- Implementation of the Schur complement operations ---*/
        Vec tmp_1;
        this->data->initialize_dof_vector(tmp_1, EquationData::U_INDEX_DOF);
        this->vmult_pressure(tmp_1, src);

        Euler_stage = EquationData::U_INDEX_SYSTEM;
        const std::vector<unsigned> index_dof_handler_reinit = {EquationData::U_INDEX_DOF};
        auto* tmp_matrix = const_cast<EULEROperator*>(this);
        Vec tmp_2;
        this->data->initialize_dof_vector(tmp_2, EquationData::U_INDEX_DOF);
        tmp_matrix->initialize(tmp_matrix->get_matrix_free(), index_dof_handler_reinit, index_dof_handler_reinit);
        this->vmult(tmp_2, tmp_1);

        Vec tmp_3;
        this->data->initialize_dof_vector(tmp_3, EquationData::P_INDEX_DOF);
        this->vmult_enthalpy(tmp_3, tmp_2);

        dst.add(static_cast<Number>(-1.0), tmp_3);
        Euler_stage = EquationData::P_INDEX_SYSTEM;
        const std::vector<unsigned> index_dof_handler = {EquationData::P_INDEX_DOF};
        tmp_matrix->initialize(tmp_matrix->get_matrix_free(), index_dof_handler, index_dof_handler);
        tmp_matrix->compute_diagonal();
      }
      else {
        this->data->cell_loop(&EULEROperator::assemble_inverse_cell_term_internal_energy,
                              this, dst, src, false);
      }
    }
    else if(Euler_stage == EquationData::U_INDEX_SYSTEM) {
      this->data->cell_loop(&EULEROperator::assemble_cell_term_velocity,
                            this, dst, src, false);
    }
    else {
      Assert(false, ExcInternalError());
    }
  }


  // Application of pressure matrix
  //
  template<unsigned dim,
           unsigned fe_degree_u, unsigned fe_degree_rho, unsigned fe_degree_p,
           unsigned n_q_points_1d, unsigned n_q_points_1d_boundary,
           typename Vec>
  void EULEROperator<dim,
                     fe_degree_u, fe_degree_rho, fe_degree_p,
                     n_q_points_1d, n_q_points_1d_boundary,
                     Vec>::
  vmult_pressure(Vec& dst, const Vec& src) const {
    src.update_ghost_values();

    this->data->loop(&EULEROperator::assemble_cell_term_pressure,
                     &EULEROperator::assemble_face_term_pressure,
                     &EULEROperator::assemble_boundary_term_pressure,
                     this, dst, src, true,
                     MatrixFree<dim, Number>::DataAccessOnFaces::values,
                     MatrixFree<dim, Number>::DataAccessOnFaces::values);
  }


  // Application of enthalpy matrix
  //
  template<unsigned dim,
           unsigned fe_degree_u, unsigned fe_degree_rho, unsigned fe_degree_p,
           unsigned n_q_points_1d, unsigned n_q_points_1d_boundary,
           typename Vec>
  void EULEROperator<dim,
                     fe_degree_u, fe_degree_rho, fe_degree_p,
                     n_q_points_1d, n_q_points_1d_boundary,
                     Vec>::
  vmult_enthalpy(Vec& dst, const Vec& src) const {
    src.update_ghost_values();

    this->data->loop(&EULEROperator::assemble_cell_term_enthalpy,
                     &EULEROperator::assemble_face_term_enthalpy,
                     &EULEROperator::assemble_boundary_term_enthalpy,
                     this, dst, src, true,
                     MatrixFree<dim, Number>::DataAccessOnFaces::values,
                     MatrixFree<dim, Number>::DataAccessOnFaces::values);
  }


  //////////////////////////////////////////////////////////////
  /*---- COMPUTE DIAGONALS---*/
  /////////////////////////////////////////////////////////////

  // Assemble diagonal cell term for the density update
  //
  template<unsigned dim,
           unsigned fe_degree_u, unsigned fe_degree_rho, unsigned fe_degree_p,
           unsigned n_q_points_1d, unsigned n_q_points_1d_boundary,
           typename Vec>
  void EULEROperator<dim,
                     fe_degree_u, fe_degree_rho, fe_degree_p,
                     n_q_points_1d, n_q_points_1d_boundary,
                     Vec>::
  assemble_diagonal_cell_term_density(const MatrixFree<dim, Number>&               data,
                                      Vec&                                         dst,
                                      const unsigned&                          ,
                                      const std::pair<unsigned, unsigned>& cell_range) const {
    FEEvaluation<dim, fe_degree_rho, fe_degree_rho + 1, 1, Number> phi(data, EquationData::RHO_INDEX_DOF, 2);

    AlignedVector<VectorizedArray<Number>> diagonal(phi.dofs_per_component);

    /*--- Loop over all cells ---*/
    for(unsigned cell = cell_range.first; cell < cell_range.second; ++cell) {
      phi.reinit(cell);

      /*--- Loop over all dofs ---*/
      for(unsigned i = 0; i < phi.dofs_per_component; ++i) {
        for(unsigned j = 0; j < phi.dofs_per_component; ++j) {
          phi.submit_dof_value(VectorizedArray<Number>(), j);
        }
        phi.submit_dof_value(make_vectorized_array<Number>(1.0), i);
        /*--- We are in a matrix-free framework. Hence, in order to compute the diagonal, we need to test the operator against
              a vector which is 1 for the node of interest and 0 elsewhere.---*/
        phi.evaluate(EvaluationFlags::values);

        /*--- Loop over all quadrature points ---*/
        for(unsigned q = 0; q < phi.n_q_points; ++q) {
          phi.submit_value(phi.get_value(q), q);
        }

        phi.integrate(EvaluationFlags::values);
        diagonal[i] = phi.get_dof_value(i);
      }

      for(unsigned i = 0; i < phi.dofs_per_component; ++i) {
        phi.submit_dof_value(diagonal[i], i);
      }
      phi.distribute_local_to_global(dst);
    }
  }


  // Assemble diagonal cell term for the velocity update
  //
  template<unsigned dim,
           unsigned fe_degree_u, unsigned fe_degree_rho, unsigned fe_degree_p,
           unsigned n_q_points_1d, unsigned n_q_points_1d_boundary,
           typename Vec>
  void EULEROperator<dim,
                     fe_degree_u, fe_degree_rho, fe_degree_p,
                     n_q_points_1d, n_q_points_1d_boundary,
                     Vec>::
  assemble_diagonal_cell_term_velocity(const MatrixFree<dim, Number>&       data,
                                       Vec&                                 dst,
                                       const unsigned&                      ,
                                       const std::pair<unsigned, unsigned>& cell_range) const {
    FEEvaluation_u   phi(data, EquationData::U_INDEX_DOF);
    FEEvaluation_rho phi_rho_for_fixed(data, EquationData::RHO_INDEX_DOF);

    /*--- We are in a matrix-free framework. Hence, in order to compute the diagonal, we need to test the operator against
          a vector which is 1 for the node of interest and 0 elsewhere. This is what 'tmp' does.
          Moreover, since here we have just one 'src' vector, but we also need to deal with the current density,
          we employ the auxiliary vector 'rho_for_fixed' where we setted this information ---*/
    AlignedVector<Tensor<1, dim, VectorizedArray<Number>>> diagonal(phi.dofs_per_component);
    Tensor<1, dim, VectorizedArray<Number>> tmp;
    for(unsigned d = 0; d < dim; ++d) {
      tmp[d] = make_vectorized_array<Number>(1.0);
    }

    /*--- Loop over all cells ---*/
    for(unsigned cell = cell_range.first; cell < cell_range.second; ++cell) {
      phi_rho_for_fixed.reinit(cell);
      phi_rho_for_fixed.gather_evaluate(rho_for_fixed, EvaluationFlags::values);

      phi.reinit(cell);

      /*--- Loop over all dofs ---*/
      for(unsigned i = 0; i < phi.dofs_per_component; ++i) {
        for(unsigned j = 0; j < phi.dofs_per_component; ++j) {
          phi.submit_dof_value(Tensor<1, dim, VectorizedArray<Number>>(), j);
        }
        phi.submit_dof_value(tmp, i);
        phi.evaluate(EvaluationFlags::values);

        /*--- Loop over all quadrature points ---*/
        for(unsigned q = 0; q < phi.n_q_points; ++q) {
          phi.submit_value(phi_rho_for_fixed.get_value(q)*phi.get_value(q), q);
        }

        phi.integrate(EvaluationFlags::values);
        diagonal[i] = phi.get_dof_value(i);
      }

      for(unsigned i = 0; i < phi.dofs_per_component; ++i) {
        phi.submit_dof_value(diagonal[i], i);
      }
      phi.distribute_local_to_global(dst);
    }
  }


  // Assemble diagonal cell term for the pressure updated with Schur complement
  //
  template<unsigned dim,
           unsigned fe_degree_u, unsigned fe_degree_rho, unsigned fe_degree_p,
           unsigned n_q_points_1d, unsigned n_q_points_1d_boundary,
           typename Vec>
  void EULEROperator<dim,
                     fe_degree_u, fe_degree_rho, fe_degree_p,
                     n_q_points_1d, n_q_points_1d_boundary,
                     Vec>::
  assemble_diagonal_cell_term_pressure(const MatrixFree<dim, Number>&       data,
                                       Vec&                                 dst,
                                       const unsigned&                      ,
                                       const std::pair<unsigned, unsigned>& cell_range) const {
    FEEvaluation_pres phi(data, EquationData::P_INDEX_DOF),
                      phi_pres_fixed(data, EquationData::P_INDEX_DOF);
    FEEvaluation_rho  phi_rho_for_fixed(data, EquationData::RHO_INDEX_DOF);

    AlignedVector<VectorizedArray<Number>> diagonal(phi.dofs_per_component);

    /*--- This term changes between second and third stage of the IMEX scheme, but its structure not, so we do not need
          to explicitly distinguish the two cases as done for the rhs. ---*/
    const auto coeff = (IMEX_stage == 2) ? a22_tilde : a33_tilde;

    for(unsigned cell = cell_range.first; cell < cell_range.second; ++cell) {
      phi_pres_fixed.reinit(cell);
      phi_pres_fixed.gather_evaluate(pres_fixed, EvaluationFlags::values);

      phi_rho_for_fixed.reinit(cell);
      phi_rho_for_fixed.gather_evaluate(rho_for_fixed, EvaluationFlags::values);

      phi.reinit(cell);

      /*--- Loop over all dofs ---*/
      for(unsigned i = 0; i < phi.dofs_per_component; ++i) {
        for(unsigned j = 0; j < phi.dofs_per_component; ++j) {
          phi.submit_dof_value(VectorizedArray<Number>(), j);
        }
        phi.submit_dof_value(make_vectorized_array<Number>(1.0), i);
        /*--- We are in a matrix-free framework. Hence, in order to compute the diagonal, we need to test the operator against
              a vector which is 1 for the node of interest and 0 elsewhere.---*/
        phi.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);

        /*--- Loop over all quadrature points ---*/
        for(unsigned q = 0; q < phi.n_q_points; ++q) {
          const auto& pres_fixed    = phi_pres_fixed.get_value(q);

          const auto& rho_for_fixed = phi_rho_for_fixed.get_value(q);

          phi.submit_value(1.0/(EquationData::Cp_Cv - 1.0)*phi.get_value(q), q);
          phi.submit_gradient((coeff*dt/Ma)*(coeff*dt/Ma)*
                              (EquationData::Cp_Cv/(EquationData::Cp_Cv - 1.0)*(pres_fixed/rho_for_fixed)*phi.get_gradient(q)), q);
        }

        phi.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
        diagonal[i] = phi.get_dof_value(i);
      }

      for(unsigned i = 0; i < phi.dofs_per_component; ++i) {
        phi.submit_dof_value(diagonal[i], i);
      }
      phi.distribute_local_to_global(dst);
    }
  }


  // Assemble diagonal cell term for the contribution due to internal energy
  //
  template<unsigned dim,
           unsigned fe_degree_u, unsigned fe_degree_rho, unsigned fe_degree_p,
           unsigned n_q_points_1d, unsigned n_q_points_1d_boundary,
           typename Vec>
  void EULEROperator<dim,
                     fe_degree_u, fe_degree_rho, fe_degree_p,
                     n_q_points_1d, n_q_points_1d_boundary,
                     Vec>::
  assemble_diagonal_cell_term_internal_energy(const MatrixFree<dim, Number>&       data,
                                              Vec&                                 dst,
                                              const unsigned&                      ,
                                              const std::pair<unsigned, unsigned>& cell_range) const {
    FEEvaluation<dim, fe_degree_p, fe_degree_p + 1, 1, Number> phi(data, EquationData::P_INDEX_DOF, 2);

    AlignedVector<VectorizedArray<Number>> diagonal(phi.dofs_per_component);

    for(unsigned cell = cell_range.first; cell < cell_range.second; ++cell) {
      phi.reinit(cell);

      /*--- Loop over all dofs ---*/
      for(unsigned i = 0; i < phi.dofs_per_component; ++i) {
        for(unsigned j = 0; j < phi.dofs_per_component; ++j) {
          phi.submit_dof_value(VectorizedArray<Number>(), j);
        }
        phi.submit_dof_value(make_vectorized_array<Number>(1.0), i);
        /*--- We are in a matrix-free framework. Hence, in order to compute the diagonal, we need to test the operator against
              a vector which is 1 for the node of interest and 0 elsewhere.---*/
        phi.evaluate(EvaluationFlags::values);

        for(unsigned q = 0; q < phi.n_q_points; ++q) {
          phi.submit_value(1.0/(EquationData::Cp_Cv - 1.0)*phi.get_value(q), q);
        }

        phi.integrate(EvaluationFlags::values);
        diagonal[i] = phi.get_dof_value(i);
      }

      for(unsigned i = 0; i < phi.dofs_per_component; ++i) {
        phi.submit_dof_value(diagonal[i], i);
      }
      phi.distribute_local_to_global(dst);
    }
  }


  // Compute diagonal of various steps
  //
  template<unsigned dim,
           unsigned fe_degree_u, unsigned fe_degree_rho, unsigned fe_degree_p,
           unsigned n_q_points_1d, unsigned n_q_points_1d_boundary,
           typename Vec>
  void EULEROperator<dim,
                     fe_degree_u, fe_degree_rho, fe_degree_p,
                     n_q_points_1d, n_q_points_1d_boundary,
                     Vec>::
  compute_diagonal() {
    AssertIndexRange(Euler_stage, EquationData::n_vars + 1);
    Assert(Euler_stage > 0, ExcInternalError());

    this->inverse_diagonal_entries.reset(new DiagonalMatrix<Vec>());
    auto& inverse_diagonal = this->inverse_diagonal_entries->get_vector();

    const unsigned dummy = 0;

    if(Euler_stage == EquationData::RHO_INDEX_SYSTEM) {
      this->data->initialize_dof_vector(inverse_diagonal, EquationData::RHO_INDEX_DOF);

      this->data->cell_loop(&EULEROperator::assemble_diagonal_cell_term_density,
                            this, inverse_diagonal, dummy, false);
    }
    else if(Euler_stage == EquationData::P_INDEX_SYSTEM) {
      this->data->initialize_dof_vector(inverse_diagonal, EquationData::P_INDEX_DOF);

      if(IMEX_stage <= EquationData::n_stages) {
        this->data->cell_loop(&EULEROperator::assemble_diagonal_cell_term_pressure,
                              this, inverse_diagonal, dummy, false);
      }
      else {
        this->data->cell_loop(&EULEROperator::assemble_diagonal_cell_term_internal_energy,
                              this, inverse_diagonal, dummy, false);
      }
    }
    else if(Euler_stage == EquationData::U_INDEX_SYSTEM) {
      this->data->initialize_dof_vector(inverse_diagonal, EquationData::U_INDEX_DOF);

      this->data->cell_loop(&EULEROperator::assemble_diagonal_cell_term_velocity,
                            this, inverse_diagonal, dummy, false);
    }
    else {
      Assert(false, ExcInternalError());
    }

    /*--- For the preconditioner, we actually need the inverse of the diagonal ---*/
    for(unsigned i = 0; i < inverse_diagonal.locally_owned_size(); ++i) {
      Assert(inverse_diagonal.local_element(i) != static_cast<Number>(0.0),
             ExcMessage("No diagonal entry in a definite operator should be zero"));
      inverse_diagonal.local_element(i) = static_cast<Number>(1.0)/inverse_diagonal.local_element(i);
    }
  }
}
