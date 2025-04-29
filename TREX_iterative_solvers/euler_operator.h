/* Author: Giuseppe Orlando, 2024. */

// @sect{Include files}

// We start by including all the necessary deal.II header files and some C++
// related ones.
//
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>
#include <deal.II/matrix_free/fe_evaluation.h>

#include <deal.II/meshworker/mesh_loop.h>

#include "runtime_parameters.h"
#include "equation_data.h"

// This is the class that implements the discretization
//
namespace Atmospheric_Flow {
  using namespace dealii;

  // @sect{ <code>EULEROperator::EULEROperator</code> }
  //
  template<int dim, int fe_degree_u, int fe_degree_rho, int fe_degree_p,
           int n_q_points_1d, int n_q_points_1d_boundary, typename Vec>
  class EULEROperator: public MatrixFreeOperators::Base<dim, Vec> {
  public:
    using Number = typename Vec::value_type;

    EULEROperator(); /*--- Default constructor ---*/

    EULEROperator(RunTimeParameters::Data_Storage& data); /*--- Constructor with some input related data ---*/

    void set_dt(const double time_step); /*--- Setter of the time-step. This is useful both for multigrid purposes and also
                                               in case of modifications of the time step. ---*/

    void set_Mach(const double Ma_); /*--- Setter of the Mach number. This is useful for multigrid purpose. ---*/

    void set_Froude(const double Fr_); /*--- Setter of the Froude number. This is useful for multigrid purpose. ---*/

    void set_IMEX_stage(const unsigned int stage); /*--- Setter of the IMEX stage. ---*/

    void set_Euler_stage(const unsigned int stage); /*--- Setter of the equation currently under solution. ---*/

    unsigned int get_Euler_stage() const; /*--- Getter of the equation currently under solution. ---*/

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
    double       Ma;  /*--- Mach number. ---*/
    double       Fr;  /*--- Froude number. ---*/
    double       dt;  /*--- Time step. ---*/

    const double gamma; /*--- TR-BDF2 (i.e. implicit part) parameter. ---*/
    /*--- The following variables follow the classical Butcher tableaux notation ---*/
    const double a21;
    const double a31;
    const double a32;

    const double a21_tilde;
    const double a22_tilde;
    const double a31_tilde;
    const double a32_tilde;
    const double a33_tilde;

    const double b1;
    const double b2;
    const double b3;

    unsigned int IMEX_stage;          /*--- Flag for the IMEX stage ---*/
    mutable unsigned int Euler_stage; /*--- Flag for the equation actually considered ---*/

    virtual void apply_add(Vec& dst, const Vec& src) const override; /*--- Overriden function which actually assembles the
                                                                           bilinear forms ---*/

  private:
    Vec rho_for_fixed,
        pres_fixed;

    /*--- Auxiliary function to compute the upwind penalization constant ---*/
    inline VectorizedArray<Number> compute_lambda(const Tensor<1, dim, VectorizedArray<Number>>& u_m,
                                                  const Tensor<1, dim, VectorizedArray<Number>>& u_p,
                                                  const Tensor<1, dim, VectorizedArray<Number>>& n_minus) const;

    /*--- Assembler functions for the rhs related to the continuity equation. Here, and also in the following,
          we distinguish between the contribution for cells, faces and boundary. ---*/
    void assemble_rhs_cell_term_density(const MatrixFree<dim, Number>&               data,
                                        Vec&                                         dst,
                                        const std::vector<Vec>&                      src,
                                        const std::pair<unsigned int, unsigned int>& cell_range) const;
    void assemble_rhs_face_term_density(const MatrixFree<dim, Number>&               data,
                                        Vec&                                         dst,
                                        const std::vector<Vec>&                      src,
                                        const std::pair<unsigned int, unsigned int>& face_range) const;
    void assemble_rhs_boundary_term_density(const MatrixFree<dim, Number>&               data,
                                            Vec&                                         dst,
                                            const std::vector<Vec>&                      src,
                                            const std::pair<unsigned int, unsigned int>& face_range) const {}
                                               /*-- No flux, so no contribution from this function ---*/

    /*--- Assembler function related to the bilinear form of the continuity equation. Only cell contribution is present,
          since, basically, we end up with a mass matrix. ---*/
    void assemble_cell_term_density(const MatrixFree<dim, Number>&               data,
                                    Vec&                                         dst,
                                    const Vec&                                   src,
                                    const std::pair<unsigned int, unsigned int>& cell_range) const;

    /*--- Assembler functions for the rhs related to the momentum equation. ---*/
    void assemble_rhs_cell_term_momentum(const MatrixFree<dim, Number>&               data,
                                         Vec&                                         dst,
                                         const std::vector<Vec>&                      src,
                                         const std::pair<unsigned int, unsigned int>& cell_range) const;
    void assemble_rhs_face_term_momentum(const MatrixFree<dim, Number>&               data,
                                         Vec&                                         dst,
                                         const std::vector<Vec>&                      src,
                                         const std::pair<unsigned int, unsigned int>& face_range) const;
    void assemble_rhs_boundary_term_momentum(const MatrixFree<dim, Number>&               data,
                                             Vec&                                         dst,
                                             const std::vector<Vec>&                      src,
                                             const std::pair<unsigned int, unsigned int>& face_range) const;

    /*--- Assembler function for the 'A' matrix. ---*/
    void assemble_cell_term_velocity(const MatrixFree<dim, Number>&               data,
                                     Vec&                                         dst,
                                     const Vec&                                   src,
                                     const std::pair<unsigned int, unsigned int>& cell_range) const;

    /*--- Assembler functions for the 'B' matrix. ---*/
    void assemble_cell_term_pressure(const MatrixFree<dim, Number>&               data,
                                     Vec&                                         dst,
                                     const Vec&                                   src,
                                     const std::pair<unsigned int, unsigned int>& cell_range) const;
    void assemble_face_term_pressure(const MatrixFree<dim, Number>&               data,
                                     Vec&                                         dst,
                                     const Vec&                                   src,
                                     const std::pair<unsigned int, unsigned int>& face_range) const;
    void assemble_boundary_term_pressure(const MatrixFree<dim, Number>&               data,
                                         Vec&                                         dst,
                                         const Vec&                                   src,
                                         const std::pair<unsigned int, unsigned int>& face_range) const;

    /*--- Assembler functions for the rhs of the energy equation. ---*/
    void assemble_rhs_cell_term_energy(const MatrixFree<dim, Number>&               data,
                                       Vec&                                         dst,
                                       const std::vector<Vec>&                      src,
                                       const std::pair<unsigned int, unsigned int>& cell_range) const;
    void assemble_rhs_face_term_energy(const MatrixFree<dim, Number>&               data,
                                       Vec&                                         dst,
                                       const std::vector<Vec>&                      src,
                                       const std::pair<unsigned int, unsigned int>& face_range) const;
    void assemble_rhs_boundary_term_energy(const MatrixFree<dim, Number>&               data,
                                           Vec&                                         dst,
                                           const std::vector<Vec>&                      src,
                                           const std::pair<unsigned int, unsigned int>& face_range) const {}
                                             /*-- No flux, so no contribution from this function ---*/

    /*--- Assembler function for the 'D' matrix. ---*/
    void assemble_cell_term_internal_energy(const MatrixFree<dim, Number>&               data,
                                            Vec&                                         dst,
                                            const Vec&                                   src,
                                            const std::pair<unsigned int, unsigned int>& cell_range) const;

    /*--- Assembler function for the 'C' matrix. ---*/
    void assemble_cell_term_enthalpy(const MatrixFree<dim, Number>&               data,
                                     Vec&                                         dst,
                                     const Vec&                                   src,
                                     const std::pair<unsigned int, unsigned int>& cell_range) const;
    void assemble_face_term_enthalpy(const MatrixFree<dim, Number>&               data,
                                     Vec&                                         dst,
                                     const Vec&                                   src,
                                     const std::pair<unsigned int, unsigned int>& face_range) const;
    void assemble_boundary_term_enthalpy(const MatrixFree<dim, Number>&               data,
                                         Vec&                                         dst,
                                         const Vec&                                   src,
                                         const std::pair<unsigned int, unsigned int>& face_range) const {}
                                         /*-- No flux, so no contribution from this function ---*/

    /*--- Assembler functions for the diagonal part of the matrix for the continuity equation. For compatibilty conditions,
          also face and boundary contributions have to be defined, even though they are empty. ---*/
    void assemble_diagonal_cell_term_density(const MatrixFree<dim, Number>&               data,
                                             Vec&                                         dst,
                                             const unsigned int&                          src,
                                             const std::pair<unsigned int, unsigned int>& cell_range) const;

    /*--- Assembler functions for the diagonal part of 'A' matrix. ---*/
    void assemble_diagonal_cell_term_velocity(const MatrixFree<dim, Number>&               data,
                                              Vec&                                         dst,
                                              const unsigned int&                          src,
                                              const std::pair<unsigned int, unsigned int>& cell_range) const;

    /*--- Assembler functions for the diagonal part of the ellptic operator associated to the Schur complement for the pressure. ---*/
    void assemble_diagonal_cell_term_pressure(const MatrixFree<dim, Number>&               data,
                                              Vec&                                         dst,
                                              const unsigned int&                          src,
                                              const std::pair<unsigned int, unsigned int>& cell_range) const;

    /*--- Assembler functions for the diagonal part of 'D' matrix. ---*/
    void assemble_diagonal_cell_term_internal_energy(const MatrixFree<dim, Number>&               data,
                                                     Vec&                                         dst,
                                                     const unsigned int&                          src,
                                                     const std::pair<unsigned int, unsigned int>& cell_range) const;
  };


  // Default constructor
  //
  template<int dim, int fe_degree_u, int fe_degree_rho, int fe_degree_p,
           int n_q_points_1d, int n_q_points_1d_boundary, typename Vec>
  EULEROperator<dim, fe_degree_u, fe_degree_rho, fe_degree_p,
                n_q_points_1d, n_q_points_1d_boundary, Vec>::
  EULEROperator(): MatrixFreeOperators::Base<dim, Vec>(), Ma(), Fr(), dt(),
                   gamma(2.0 - std::sqrt(2.0)), a21(gamma),
                   a31(0.5), a32(0.5),
                   a21_tilde(0.5*gamma), a22_tilde(0.5*gamma),
                   a31_tilde(0.5 - 0.25*gamma), a32_tilde(0.5 - 0.25*gamma), a33_tilde(0.5*gamma),
                   b1(0.5 - 0.25*gamma), b2(0.5 - 0.25*gamma), b3(0.5*gamma),
                   IMEX_stage(1), Euler_stage(1) {}

  // Constructor with runtime parameters storage
  //
  template<int dim, int fe_degree_u, int fe_degree_rho, int fe_degree_p,
           int n_q_points_1d, int n_q_points_1d_boundary, typename Vec>
  EULEROperator<dim, fe_degree_u, fe_degree_rho, fe_degree_p,
                n_q_points_1d, n_q_points_1d_boundary, Vec>::
  EULEROperator(RunTimeParameters::Data_Storage& data): MatrixFreeOperators::Base<dim, Vec>(),
                                                        Ma(data.Mach), Fr(data.Froude), dt(data.dt),
                                                        gamma(2.0 - std::sqrt(2.0)), a21(gamma),
                                                        a31(0.5), a32(0.5),
                                                        a21_tilde(0.5*gamma), a22_tilde(0.5*gamma),
                                                        a31_tilde(0.5 - 0.25*gamma), a32_tilde(0.5 - 0.25*gamma), a33_tilde(0.5*gamma),
                                                        b1(0.5 - 0.25*gamma), b2(0.5 - 0.25*gamma), b3(0.5*gamma),
                                                        IMEX_stage(1), Euler_stage(1) {}


  // Setter of time-step
  //
  template<int dim, int fe_degree_u, int fe_degree_rho, int fe_degree_p,
           int n_q_points_1d, int n_q_points_1d_boundary, typename Vec>
  void EULEROperator<dim, fe_degree_u, fe_degree_rho, fe_degree_p,
                     n_q_points_1d, n_q_points_1d_boundary, Vec>::
  set_dt(const double time_step) {
    dt = time_step;
  }

  // Setter of Mach number
  //
  template<int dim, int fe_degree_u, int fe_degree_rho, int fe_degree_p,
           int n_q_points_1d, int n_q_points_1d_boundary, typename Vec>
  void EULEROperator<dim, fe_degree_u, fe_degree_rho, fe_degree_p,
                     n_q_points_1d, n_q_points_1d_boundary, Vec>::
  set_Mach(const double Ma_) {
    Ma = Ma_;
  }

  // Setter of Froude number
  //
  template<int dim, int fe_degree_u, int fe_degree_rho, int fe_degree_p,
           int n_q_points_1d, int n_q_points_1d_boundary, typename Vec>
  void EULEROperator<dim, fe_degree_u, fe_degree_rho, fe_degree_p,
                     n_q_points_1d, n_q_points_1d_boundary, Vec>::
  set_Froude(const double Fr_) {
    Fr = Fr_;
  }

  // Setter of IMEX stage (this can be known only during the effective execution
  // and so it has to be demanded to the class that really solves the problem)
  //
  template<int dim, int fe_degree_u, int fe_degree_rho, int fe_degree_p,
           int n_q_points_1d, int n_q_points_1d_boundary, typename Vec>
  void EULEROperator<dim, fe_degree_u, fe_degree_rho, fe_degree_p,
                     n_q_points_1d, n_q_points_1d_boundary, Vec>::
  set_IMEX_stage(const unsigned int stage) {
    AssertIndexRange(stage, EquationData::n_stages + 2);
    Assert(stage > 0, ExcInternalError());

    IMEX_stage = stage;
  }

  // Setter of Euler stage (this can be known only during the effective execution
  // and so it has to be demanded to the class that really solves the problem)
  //
  template<int dim, int fe_degree_u, int fe_degree_rho, int fe_degree_p,
           int n_q_points_1d, int n_q_points_1d_boundary, typename Vec>
  void EULEROperator<dim, fe_degree_u, fe_degree_rho, fe_degree_p,
                     n_q_points_1d, n_q_points_1d_boundary, Vec>::
  set_Euler_stage(const unsigned int stage) {
    AssertIndexRange(stage, EquationData::n_vars + 1);
    Assert(stage > 0, ExcInternalError());

    Euler_stage = stage;
  }

  // Getter of Euler stage (this can be known only during the effective execution
  // and so it has to be demanded to the class that really solves the problem)
  //
  template<int dim, int fe_degree_u, int fe_degree_rho, int fe_degree_p,
           int n_q_points_1d, int n_q_points_1d_boundary, typename Vec>
  unsigned int EULEROperator<dim, fe_degree_u, fe_degree_rho, fe_degree_p,
                             n_q_points_1d, n_q_points_1d_boundary, Vec>::
  get_Euler_stage() const {
    return Euler_stage;
  }


  // Setter of density for fixed point
  //
  template<int dim, int fe_degree_u, int fe_degree_rho, int fe_degree_p,
           int n_q_points_1d, int n_q_points_1d_boundary, typename Vec>
  void EULEROperator<dim, fe_degree_u, fe_degree_rho, fe_degree_p,
                     n_q_points_1d, n_q_points_1d_boundary, Vec>::
  set_rho_for_fixed(const Vec& src) {
    rho_for_fixed = src;
    rho_for_fixed.update_ghost_values();
  }

  // Setter of pressure for fixed point
  //
  template<int dim, int fe_degree_u, int fe_degree_rho, int fe_degree_p,
           int n_q_points_1d, int n_q_points_1d_boundary, typename Vec>
  void EULEROperator<dim, fe_degree_u, fe_degree_rho, fe_degree_p,
                     n_q_points_1d, n_q_points_1d_boundary, Vec>::
  set_pres_fixed(const Vec& src) {
    pres_fixed = src;
    pres_fixed.update_ghost_values();
  }

  // Auxiliary function to compute the stabilization term
  //
  template<int dim, int fe_degree_u, int fe_degree_rho, int fe_degree_p,
           int n_q_points_1d, int n_q_points_1d_boundary, typename Vec>
  inline VectorizedArray<typename Vec::value_type>
  EULEROperator<dim, fe_degree_u, fe_degree_rho, fe_degree_p,
                n_q_points_1d, n_q_points_1d_boundary, Vec>::
  compute_lambda(const Tensor<1, dim, VectorizedArray<Number>>& u_m,
                 const Tensor<1, dim, VectorizedArray<Number>>& u_p,
                 const Tensor<1, dim, VectorizedArray<Number>>& n_minus) const {
    return std::max(std::abs(scalar_product(u_m, n_minus)),
                    std::abs(scalar_product(u_p, n_minus)));
  }


  // Assemble rhs cell term for the density update
  //
  template<int dim, int fe_degree_u, int fe_degree_rho, int fe_degree_p,
           int n_q_points_1d, int n_q_points_1d_boundary, typename Vec>
  void EULEROperator<dim, fe_degree_u, fe_degree_rho, fe_degree_p,
                     n_q_points_1d, n_q_points_1d_boundary, Vec>::
  assemble_rhs_cell_term_density(const MatrixFree<dim, Number>&               data,
                                 Vec&                                         dst,
                                 const std::vector<Vec>&                      src,
                                 const std::pair<unsigned int, unsigned int>& cell_range) const {
    if(IMEX_stage == 2) {
      /*--- We first start by declaring the suitable instances to read the old density and
      the old velocity. 'phi' will be used only to 'submit' the result.
      The second argument specifies which dof handler has to be used (in this implementation 0 stands for
      velocity, 1 for pressure and 2 for density). ---*/
      FEEvaluation<dim, fe_degree_rho, n_q_points_1d, 1, Number> phi(data, EquationData::RHO_INDEX_DOF),
                                                                 phi_rho_old(data, EquationData::RHO_INDEX_DOF);
      FEEvaluation<dim, fe_degree_u, n_q_points_1d, dim, Number> phi_u_old(data, EquationData::U_INDEX_DOF);

      /*--- Loop over all cells ---*/
      for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
        /*--- Now we need to assign the current cell to each FEEvaluation object and then to specify which src vector
        it has to read (the proper order is clearly delegated to the user, which has to pay attention in the function
        call to be coherent). All these considerations are valid also for the other assembler functions. ---*/
        phi_rho_old.reinit(cell);
        phi_rho_old.gather_evaluate(src[0], EvaluationFlags::values);
        phi_u_old.reinit(cell);
        phi_u_old.gather_evaluate(src[1], EvaluationFlags::values);

        phi.reinit(cell);

        /*--- Loop over quadrature points of each cell ---*/
        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
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
      FEEvaluation<dim, fe_degree_rho, n_q_points_1d, 1, Number> phi(data, EquationData::RHO_INDEX_DOF),
                                                                 phi_rho_old(data, EquationData::RHO_INDEX_DOF),
                                                                 phi_rho_s_2(data, EquationData::RHO_INDEX_DOF);
      FEEvaluation<dim, fe_degree_u, n_q_points_1d, dim, Number> phi_u_old(data, EquationData::U_INDEX_DOF),
                                                                 phi_u_s_2(data, EquationData::U_INDEX_DOF);

      /*--- Loop over all cells ---*/
      for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
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
        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          /*--- Compute the quantities at the previous step ---*/
          const auto& rho_old = phi_rho_old.get_value(q);
          const auto& u_old   = phi_u_old.get_value(q);

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
      FEEvaluation<dim, fe_degree_rho, n_q_points_1d, 1, Number> phi(data, EquationData::RHO_INDEX_DOF),
                                                                 phi_rho_old(data, EquationData::RHO_INDEX_DOF),
                                                                 phi_rho_s_2(data, EquationData::RHO_INDEX_DOF),
                                                                 phi_rho_s_3(data, EquationData::RHO_INDEX_DOF);
      FEEvaluation<dim, fe_degree_u, n_q_points_1d, dim, Number> phi_u_old(data, EquationData::U_INDEX_DOF),
                                                                 phi_u_s_2(data, EquationData::U_INDEX_DOF),
                                                                 phi_u_s_3(data, EquationData::U_INDEX_DOF);

      /*--- Loop over all cells ---*/
      for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
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
        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
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
  template<int dim, int fe_degree_u, int fe_degree_rho, int fe_degree_p,
           int n_q_points_1d, int n_q_points_1d_boundary, typename Vec>
  void EULEROperator<dim, fe_degree_u, fe_degree_rho, fe_degree_p,
                     n_q_points_1d, n_q_points_1d_boundary, Vec>::
  assemble_rhs_face_term_density(const MatrixFree<dim, Number>&               data,
                                 Vec&                                         dst,
                                 const std::vector<Vec>&                      src,
                                 const std::pair<unsigned int, unsigned int>& face_range) const {
    if(IMEX_stage == 2) {
      /*--- We first start by declaring the suitable instances to read the available quantities.
            'true' means that we are reading the information from 'inside', whereas 'false' from 'outside' ---*/
      FEFaceEvaluation<dim, fe_degree_rho, n_q_points_1d, 1, Number> phi_m(data, true, EquationData::RHO_INDEX_DOF),
                                                                     phi_p(data, false, EquationData::RHO_INDEX_DOF),
                                                                     phi_rho_old_m(data, true, EquationData::RHO_INDEX_DOF),
                                                                     phi_rho_old_p(data, false, EquationData::RHO_INDEX_DOF);
      FEFaceEvaluation<dim, fe_degree_u, n_q_points_1d, dim, Number> phi_u_old_m(data, true, EquationData::U_INDEX_DOF),
                                                                     phi_u_old_p(data, false, EquationData::U_INDEX_DOF);

      /*--- Loop over all internal faces ---*/
      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
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
        for(unsigned int q = 0; q < phi_m.n_q_points; ++q) {
          const auto& n_minus      = phi_m.get_normal_vector(q); /*--- Notice that the unit normal vector is the same from
                                                                       'both sides'. ---*/

          /*--- Compute the quantities at the previous step ---*/
          const auto& rho_old_m    = phi_rho_old_m.get_value(q);
          const auto& rho_old_p    = phi_rho_old_p.get_value(q);
          const auto& u_old_m      = phi_u_old_m.get_value(q);
          const auto& u_old_p      = phi_u_old_p.get_value(q);

          const auto& avg_flux_old = 0.5*(rho_old_m*u_old_m + rho_old_p*u_old_p);
          const auto& lambda_old   = compute_lambda(u_old_m, u_old_p, n_minus);
          const auto& jump_rho_old = rho_old_m - rho_old_p;

          /*--- Using an upwind flux ---*/
          phi_m.submit_value(-a21*dt*(scalar_product(avg_flux_old, n_minus) + 0.5*lambda_old*jump_rho_old), q);
          phi_p.submit_value(a21*dt*(scalar_product(avg_flux_old, n_minus) + 0.5*lambda_old*jump_rho_old), q);
        }

        phi_m.integrate_scatter(EvaluationFlags::values, dst);
        phi_p.integrate_scatter(EvaluationFlags::values, dst);
      }
    }
    else if(IMEX_stage == 3) {
      /*--- We first start by declaring the suitable instances to read the available quantities. ---*/
      FEFaceEvaluation<dim, fe_degree_rho, n_q_points_1d, 1, Number> phi_m(data, true, EquationData::RHO_INDEX_DOF),
                                                                     phi_p(data, false, EquationData::RHO_INDEX_DOF),
                                                                     phi_rho_old_m(data, true, EquationData::RHO_INDEX_DOF),
                                                                     phi_rho_old_p(data, false, EquationData::RHO_INDEX_DOF),
                                                                     phi_rho_s_2_m(data, true, EquationData::RHO_INDEX_DOF),
                                                                     phi_rho_s_2_p(data, false, EquationData::RHO_INDEX_DOF);
      FEFaceEvaluation<dim, fe_degree_u, n_q_points_1d, dim, Number> phi_u_old_m(data, true, EquationData::U_INDEX_DOF),
                                                                     phi_u_old_p(data, false, EquationData::U_INDEX_DOF),
                                                                     phi_u_s_2_m(data, true, EquationData::U_INDEX_DOF),
                                                                     phi_u_s_2_p(data, false, EquationData::U_INDEX_DOF);

      /*--- Loop over all internal faces ---*/
      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
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
        for(unsigned int q = 0; q < phi_m.n_q_points; ++q) {
          const auto& n_minus      = phi_m.get_normal_vector(q);

          /*--- Compute the quantities at the previous step ---*/
          const auto& rho_old_m    = phi_rho_old_m.get_value(q);
          const auto& rho_old_p    = phi_rho_old_p.get_value(q);
          const auto& u_old_m      = phi_u_old_m.get_value(q);
          const auto& u_old_p      = phi_u_old_p.get_value(q);

          const auto& avg_flux_old = 0.5*(rho_old_m*u_old_m + rho_old_p*u_old_p);
          const auto& lambda_old   = compute_lambda(u_old_m, u_old_p, n_minus);
          const auto& jump_rho_old = rho_old_m - rho_old_p;

          /*--- Compute the quantities at the previous stage ---*/
          const auto& rho_s_2_m    = phi_rho_s_2_m.get_value(q);
          const auto& rho_s_2_p    = phi_rho_s_2_p.get_value(q);
          const auto& u_s_2_m      = phi_u_s_2_m.get_value(q);
          const auto& u_s_2_p      = phi_u_s_2_p.get_value(q);

          const auto& avg_flux_s_2 = 0.5*(rho_s_2_m*u_s_2_m + rho_s_2_p*u_s_2_p);
          const auto& lambda_s_2   = compute_lambda(u_s_2_m, u_s_2_p, n_minus);
          const auto& jump_rho_s_2 = rho_s_2_m - rho_s_2_p;

          /*--- Using an upwind flux ---*/
          phi_m.submit_value(-a31*dt*(scalar_product(avg_flux_old, n_minus) + 0.5*lambda_old*jump_rho_old)
                             -a32*dt*(scalar_product(avg_flux_s_2, n_minus) + 0.5*lambda_s_2*jump_rho_s_2), q);
          phi_p.submit_value(a31*dt*(scalar_product(avg_flux_old, n_minus) + 0.5*lambda_old*jump_rho_old) +
                             a32*dt*(scalar_product(avg_flux_s_2, n_minus) + 0.5*lambda_s_2*jump_rho_s_2), q);
        }

        phi_m.integrate_scatter(EvaluationFlags::values, dst);
        phi_p.integrate_scatter(EvaluationFlags::values, dst);
      }
    }
    else {
      /*--- We first start by declaring the suitable instances to read the available quantities. ---*/
      FEFaceEvaluation<dim, fe_degree_rho, n_q_points_1d, 1, Number> phi_m(data, true, EquationData::RHO_INDEX_DOF),
                                                                     phi_p(data, false, EquationData::RHO_INDEX_DOF),
                                                                     phi_rho_old_m(data, true, EquationData::RHO_INDEX_DOF),
                                                                     phi_rho_old_p(data, false, EquationData::RHO_INDEX_DOF),
                                                                     phi_rho_s_2_m(data, true, EquationData::RHO_INDEX_DOF),
                                                                     phi_rho_s_2_p(data, false, EquationData::RHO_INDEX_DOF),
                                                                     phi_rho_s_3_m(data, true, EquationData::RHO_INDEX_DOF),
                                                                     phi_rho_s_3_p(data, false, EquationData::RHO_INDEX_DOF);
      FEFaceEvaluation<dim, fe_degree_u, n_q_points_1d, dim, Number> phi_u_old_m(data, true, EquationData::U_INDEX_DOF),
                                                                     phi_u_old_p(data, false, EquationData::U_INDEX_DOF),
                                                                     phi_u_s_2_m(data, true, EquationData::U_INDEX_DOF),
                                                                     phi_u_s_2_p(data, false, EquationData::U_INDEX_DOF),
                                                                     phi_u_s_3_m(data, true, EquationData::U_INDEX_DOF),
                                                                     phi_u_s_3_p(data, false, EquationData::U_INDEX_DOF);

      /*--- Loop over all internal faces ---*/
      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
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
        for(unsigned int q = 0; q < phi_m.n_q_points; ++q) {
          const auto& n_minus      = phi_m.get_normal_vector(q);

          /*--- Compute the quantities at the previous step ---*/
          const auto& rho_old_m    = phi_rho_old_m.get_value(q);
          const auto& rho_old_p    = phi_rho_old_p.get_value(q);
          const auto& u_old_m      = phi_u_old_m.get_value(q);
          const auto& u_old_p      = phi_u_old_p.get_value(q);

          const auto& avg_flux_old = 0.5*(rho_old_m*u_old_m + rho_old_p*u_old_p);
          const auto& lambda_old   = compute_lambda(u_old_m, u_old_p, n_minus);
          const auto& jump_rho_old = rho_old_m - rho_old_p;

          /*--- Compute the quantities at the second stage ---*/
          const auto& rho_s_2_m    = phi_rho_s_2_m.get_value(q);
          const auto& rho_s_2_p    = phi_rho_s_2_p.get_value(q);
          const auto& u_s_2_m      = phi_u_s_2_m.get_value(q);
          const auto& u_s_2_p      = phi_u_s_2_p.get_value(q);

          const auto& avg_flux_s_2 = 0.5*(rho_s_2_m*u_s_2_m + rho_s_2_p*u_s_2_p);
          const auto& lambda_s_2   = compute_lambda(u_s_2_m, u_s_2_p, n_minus);
          const auto& jump_rho_s_2 = rho_s_2_m - rho_s_2_p;

          /*--- Compute the quantities at the final stage ---*/
          const auto& rho_s_3_m    = phi_rho_s_3_m.get_value(q);
          const auto& rho_s_3_p    = phi_rho_s_3_p.get_value(q);
          const auto& u_s_3_m      = phi_u_s_3_m.get_value(q);
          const auto& u_s_3_p      = phi_u_s_3_p.get_value(q);

          const auto& avg_flux_s_3 = 0.5*(rho_s_3_m*u_s_3_m + rho_s_3_p*u_s_3_p);
          const auto& lambda_s_3   = compute_lambda(u_s_3_m, u_s_3_p, n_minus);
          const auto& jump_rho_s_3 = rho_s_3_m - rho_s_3_p;

          /*--- Using an upwind flux ---*/
          phi_m.submit_value(-b1*dt*(scalar_product(avg_flux_old, n_minus) + 0.5*lambda_old*jump_rho_old)
                             -b2*dt*(scalar_product(avg_flux_s_2, n_minus) + 0.5*lambda_s_2*jump_rho_s_2)
                             -b3*dt*(scalar_product(avg_flux_s_3, n_minus) + 0.5*lambda_s_3*jump_rho_s_3), q);
          phi_p.submit_value(b1*dt*(scalar_product(avg_flux_old, n_minus) + 0.5*lambda_old*jump_rho_old) +
                             b2*dt*(scalar_product(avg_flux_s_2, n_minus) + 0.5*lambda_s_2*jump_rho_s_2) +
                             b3*dt*(scalar_product(avg_flux_s_3, n_minus) + 0.5*lambda_s_3*jump_rho_s_3), q);
        }

        phi_m.integrate_scatter(EvaluationFlags::values, dst);
        phi_p.integrate_scatter(EvaluationFlags::values, dst);
      }
    }
  }

  // Put together all the previous steps for density update
  //
  template<int dim, int fe_degree_u, int fe_degree_rho, int fe_degree_p,
           int n_q_points_1d, int n_q_points_1d_boundary, typename Vec>
  void EULEROperator<dim, fe_degree_u, fe_degree_rho, fe_degree_p,
                     n_q_points_1d, n_q_points_1d_boundary, Vec>::
  vmult_rhs_density(Vec& dst, const std::vector<Vec>& src) const {
    for(unsigned int d = 0; d < src.size(); ++d) {
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
  template<int dim, int fe_degree_u, int fe_degree_rho, int fe_degree_p,
           int n_q_points_1d, int n_q_points_1d_boundary, typename Vec>
  void EULEROperator<dim, fe_degree_u, fe_degree_rho, fe_degree_p,
                     n_q_points_1d, n_q_points_1d_boundary, Vec>::
  assemble_cell_term_density(const MatrixFree<dim, Number>&               data,
                             Vec&                                         dst,
                             const Vec&                                   src,
                             const std::pair<unsigned int, unsigned int>& cell_range) const {
    FEEvaluation<dim, fe_degree_rho, n_q_points_1d, 1, Number> phi(data, EquationData::RHO_INDEX_DOF);

    /*--- Loop over all cells ---*/
    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
      phi.reinit(cell);
      phi.gather_evaluate(src, EvaluationFlags::values);

      /*--- Loop over all quadrature points ---*/
      for(unsigned int q = 0; q < phi.n_q_points; ++q) {
        phi.submit_value(phi.get_value(q), q); /*--- Here we need to assemble just a mass matrix,
                                                     so we simply test against the test fuction, the 'src' vector ---*/
      }

      phi.integrate_scatter(EvaluationFlags::values, dst);
    }
  }


  // Assemble rhs cell term of the momentum equation
  //
  template<int dim, int fe_degree_u, int fe_degree_rho, int fe_degree_p,
           int n_q_points_1d, int n_q_points_1d_boundary, typename Vec>
  void EULEROperator<dim, fe_degree_u, fe_degree_rho, fe_degree_p,
                     n_q_points_1d, n_q_points_1d_boundary, Vec>::
  assemble_rhs_cell_term_momentum(const MatrixFree<dim, Number>&               data,
                                  Vec&                                         dst,
                                  const std::vector<Vec>&                      src,
                                  const std::pair<unsigned int, unsigned int>& cell_range) const {
    /*--- We create an auxiliary vector for the unit vector along vertical direction. This will never change
          independently on the stage, so we declare it once and for all. ---*/
    Tensor<1, dim, VectorizedArray<Number>> e_k;
    for(unsigned int d = 0; d < dim - 1; ++d) {
      e_k[d] = make_vectorized_array<Number>(0.0);
    }
    e_k[dim - 1] = make_vectorized_array<Number>(1.0);

    if(IMEX_stage == 2) {
      /*--- We first start by declaring the suitable instances to read the available quantities. ---*/
      FEEvaluation<dim, fe_degree_u, n_q_points_1d, dim, Number> phi(data, EquationData::U_INDEX_DOF),
                                                                 phi_u_old(data, EquationData::U_INDEX_DOF);
      FEEvaluation<dim, fe_degree_p, n_q_points_1d, 1, Number>   phi_pres_old(data, EquationData::P_INDEX_DOF);
      FEEvaluation<dim, fe_degree_rho, n_q_points_1d, 1, Number> phi_rho_old(data, EquationData::RHO_INDEX_DOF),
                                                                 phi_rho_s_2(data, EquationData::RHO_INDEX_DOF);

      /*--- Loop over all cells ---*/
      for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
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
        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          /*--- Compute the quantities at the previous step ---*/
          const auto& rho_old            = phi_rho_old.get_value(q);
          const auto& u_old              = phi_u_old.get_value(q);
          const auto& pres_old           = phi_pres_old.get_value(q);

          const auto& tensor_product_u_n = outer_product(u_old, u_old);
          /*--- For the sake of compatibility, since after integration by parts, the pressure gradient
                would be tested against the divergence of the test function. This is equaivalent to test a diagonal matrix
                with diagonal entries equal to the pressure itself against the gradient of the test function. ---*/
          Tensor<2, dim, VectorizedArray<Number>> p_n_times_identity;
          for(unsigned int d = 0; d < dim; ++d) {
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
      FEEvaluation<dim, fe_degree_u, n_q_points_1d, dim, Number> phi(data, EquationData::U_INDEX_DOF),
                                                                 phi_u_old(data, EquationData::U_INDEX_DOF),
                                                                 phi_u_s_2(data, EquationData::U_INDEX_DOF);
      FEEvaluation<dim, fe_degree_p, n_q_points_1d, 1, Number>   phi_pres_old(data, EquationData::P_INDEX_DOF),
                                                                 phi_pres_s_2(data, EquationData::P_INDEX_DOF);
      FEEvaluation<dim, fe_degree_rho, n_q_points_1d, 1, Number> phi_rho_old(data, EquationData::RHO_INDEX_DOF),
                                                                 phi_rho_s_2(data, EquationData::RHO_INDEX_DOF),
                                                                 phi_rho_curr(data, EquationData::RHO_INDEX_DOF);

      /*--- Loop over all cells ---*/
      for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
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
        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          /*--- Compute the quantities at the previous step ---*/
          const auto& rho_old            = phi_rho_old.get_value(q);
          const auto& u_old              = phi_u_old.get_value(q);
          const auto& pres_old           = phi_pres_old.get_value(q);

          const auto& tensor_product_u_n = outer_product(u_old, u_old);
          Tensor<2, dim, VectorizedArray<Number>> p_n_times_identity;
          for(unsigned int d = 0; d < dim; ++d) {
            p_n_times_identity[d][d] = pres_old;
          }

          /*--- Compute the quantities at the previous stage ---*/
          const auto& rho_s_2              = phi_rho_s_2.get_value(q);
          const auto& u_s_2                = phi_u_s_2.get_value(q);
          const auto& pres_s_2             = phi_pres_s_2.get_value(q);

          const auto& tensor_product_u_s_2 = outer_product(u_s_2, u_s_2);
          Tensor<2, dim, VectorizedArray<Number>> p_s_2_times_identity;
          for(unsigned int d = 0; d < dim; ++d) {
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
      FEEvaluation<dim, fe_degree_u, n_q_points_1d, dim, Number> phi(data, EquationData::U_INDEX_DOF),
                                                                 phi_u_old(data, EquationData::U_INDEX_DOF),
                                                                 phi_u_s_2(data, EquationData::U_INDEX_DOF),
                                                                 phi_u_s_3(data, EquationData::U_INDEX_DOF);
      FEEvaluation<dim, fe_degree_p, n_q_points_1d, 1, Number>   phi_pres_old(data, EquationData::P_INDEX_DOF),
                                                                 phi_pres_s_2(data, EquationData::P_INDEX_DOF),
                                                                 phi_pres_s_3(data, EquationData::P_INDEX_DOF);
      FEEvaluation<dim, fe_degree_rho, n_q_points_1d, 1, Number> phi_rho_old(data, EquationData::RHO_INDEX_DOF),
                                                                 phi_rho_s_2(data, EquationData::RHO_INDEX_DOF),
                                                                 phi_rho_s_3(data, EquationData::RHO_INDEX_DOF);

      /*--- Loop over all cells ---*/
      for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
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
        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          /*--- Compute the quantities at the previous step ---*/
          const auto& rho_old            = phi_rho_old.get_value(q);
          const auto& u_old              = phi_u_old.get_value(q);
          const auto& pres_old           = phi_pres_old.get_value(q);

          const auto& tensor_product_u_n = outer_product(u_old, u_old);
          Tensor<2, dim, VectorizedArray<Number>> p_n_times_identity;
          for(unsigned int d = 0; d < dim; ++d) {
            p_n_times_identity[d][d] = pres_old;
          }

          /*--- Compute the quantities at the second stage ---*/
          const auto& rho_s_2              = phi_rho_s_2.get_value(q);
          const auto& u_s_2                = phi_u_s_2.get_value(q);
          const auto& pres_s_2             = phi_pres_s_2.get_value(q);

          const auto& tensor_product_u_s_2 = outer_product(u_s_2, u_s_2);
          Tensor<2, dim, VectorizedArray<Number>> p_s_2_times_identity;
          for(unsigned int d = 0; d < dim; ++d) {
            p_s_2_times_identity[d][d] = pres_s_2;
          }

          /*--- Compute the quantities at the final stage ---*/
          const auto& rho_s_3              = phi_rho_s_3.get_value(q);
          const auto& u_s_3                = phi_u_s_3.get_value(q);
          const auto& pres_s_3             = phi_pres_s_3.get_value(q);

          const auto& tensor_product_u_s_3 = outer_product(u_s_3, u_s_3);
          Tensor<2, dim, VectorizedArray<Number>> p_s_3_times_identity;
          for(unsigned int d = 0; d < dim; ++d) {
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
  template<int dim, int fe_degree_u, int fe_degree_rho, int fe_degree_p,
           int n_q_points_1d, int n_q_points_1d_boundary, typename Vec>
  void EULEROperator<dim, fe_degree_u, fe_degree_rho, fe_degree_p,
                     n_q_points_1d, n_q_points_1d_boundary, Vec>::
  assemble_rhs_face_term_momentum(const MatrixFree<dim, Number>&               data,
                                  Vec&                                         dst,
                                  const std::vector<Vec>&                      src,
                                  const std::pair<unsigned int, unsigned int>& face_range) const {
    if(IMEX_stage == 2) {
      /*--- We first start by declaring the suitable instances to read the available quantities. ---*/
      FEFaceEvaluation<dim, fe_degree_u, n_q_points_1d, dim, Number> phi_m(data, true, EquationData::U_INDEX_DOF),
                                                                     phi_p(data, false, EquationData::U_INDEX_DOF),
                                                                     phi_u_old_m(data, true, EquationData::U_INDEX_DOF),
                                                                     phi_u_old_p(data, false, EquationData::U_INDEX_DOF);
      FEFaceEvaluation<dim, fe_degree_p, n_q_points_1d, 1, Number>   phi_pres_old_m(data, true, EquationData::P_INDEX_DOF),
                                                                     phi_pres_old_p(data, false, EquationData::P_INDEX_DOF);
      FEFaceEvaluation<dim, fe_degree_rho, n_q_points_1d, 1, Number> phi_rho_old_m(data, true, EquationData::RHO_INDEX_DOF),
                                                                     phi_rho_old_p(data, false, EquationData::RHO_INDEX_DOF);

      /*--- Loop over all internal faces ---*/
      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
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
        for(unsigned int q = 0; q < phi_m.n_q_points; ++q) {
          const auto& n_minus                = phi_m.get_normal_vector(q);

          /*--- Compute the quantities at the previous step ---*/
          const auto& rho_old_m              = phi_rho_old_m.get_value(q);
          const auto& rho_old_p              = phi_rho_old_p.get_value(q);
          const auto& u_old_m                = phi_u_old_m.get_value(q);
          const auto& u_old_p                = phi_u_old_p.get_value(q);
          const auto& pres_old_m             = phi_pres_old_m.get_value(q);
          const auto& pres_old_p             = phi_pres_old_p.get_value(q);

          const auto& avg_tensor_product_u_n = 0.5*(outer_product(rho_old_m*u_old_m, u_old_m) +
                                                    outer_product(rho_old_p*u_old_p, u_old_p));
          const auto& avg_pres_old           = 0.5*(pres_old_m + pres_old_p);

          const auto& jump_rhou_old          = rho_old_m*u_old_m - rho_old_p*u_old_p;
          const auto& lambda_old             = compute_lambda(u_old_m, u_old_p, n_minus);

          phi_m.submit_value(-a21*dt*(avg_tensor_product_u_n*n_minus +
                                      0.5*lambda_old*jump_rhou_old)
                             -a21_tilde*dt*(avg_pres_old/(Ma*Ma)*n_minus), q);
          phi_p.submit_value(a21*dt*(avg_tensor_product_u_n*n_minus +
                                     0.5*lambda_old*jump_rhou_old) +
                             a21_tilde*dt*(avg_pres_old/(Ma*Ma)*n_minus), q);
        }

        phi_m.integrate_scatter(EvaluationFlags::values, dst);
        phi_p.integrate_scatter(EvaluationFlags::values, dst);
      }
    }
    else if(IMEX_stage == 3) {
      /*--- We first start by declaring the suitable instances to read the available quantities. ---*/
      FEFaceEvaluation<dim, fe_degree_u, n_q_points_1d, dim, Number> phi_m(data, true, EquationData::U_INDEX_DOF),
                                                                     phi_p(data, false, EquationData::U_INDEX_DOF),
                                                                     phi_u_old_m(data, true, EquationData::U_INDEX_DOF),
                                                                     phi_u_old_p(data, false, EquationData::U_INDEX_DOF),
                                                                     phi_u_s_2_m(data, true, EquationData::U_INDEX_DOF),
                                                                     phi_u_s_2_p(data, false, EquationData::U_INDEX_DOF);
      FEFaceEvaluation<dim, fe_degree_p, n_q_points_1d, 1, Number>   phi_pres_old_m(data, true, EquationData::P_INDEX_DOF),
                                                                     phi_pres_old_p(data, false, EquationData::P_INDEX_DOF),
                                                                     phi_pres_s_2_m(data, true, EquationData::P_INDEX_DOF),
                                                                     phi_pres_s_2_p(data, false, EquationData::P_INDEX_DOF);
      FEFaceEvaluation<dim, fe_degree_rho, n_q_points_1d, 1, Number> phi_rho_old_m(data, true, EquationData::RHO_INDEX_DOF),
                                                                     phi_rho_old_p(data, false, EquationData::RHO_INDEX_DOF),
                                                                     phi_rho_s_2_m(data, true, EquationData::RHO_INDEX_DOF),
                                                                     phi_rho_s_2_p(data, false, EquationData::RHO_INDEX_DOF);

      /*---Loop over all internal faces ---*/
      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
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
        for(unsigned int q = 0; q < phi_m.n_q_points; ++q) {
          const auto& n_minus                  = phi_m.get_normal_vector(q);

          /*--- Compute the quantities at the previous step ---*/
          const auto& rho_old_m                = phi_rho_old_m.get_value(q);
          const auto& rho_old_p                = phi_rho_old_p.get_value(q);
          const auto& u_old_m                  = phi_u_old_m.get_value(q);
          const auto& u_old_p                  = phi_u_old_p.get_value(q);
          const auto& pres_old_m               = phi_pres_old_m.get_value(q);
          const auto& pres_old_p               = phi_pres_old_p.get_value(q);

          const auto& avg_tensor_product_u_n   = 0.5*(outer_product(rho_old_m*u_old_m, u_old_m) +
                                                      outer_product(rho_old_p*u_old_p, u_old_p));
          const auto& avg_pres_old             = 0.5*(pres_old_m + pres_old_p);

          const auto& jump_rhou_old            = rho_old_m*u_old_m - rho_old_p*u_old_p;
          const auto& lambda_old               = compute_lambda(u_old_m, u_old_p, n_minus);

          /*--- Compute the quantities at the previous stage ---*/
          const auto& rho_s_2_m                = phi_rho_s_2_m.get_value(q);
          const auto& rho_s_2_p                = phi_rho_s_2_p.get_value(q);
          const auto& u_s_2_m                  = phi_u_s_2_m.get_value(q);
          const auto& u_s_2_p                  = phi_u_s_2_p.get_value(q);
          const auto& pres_s_2_m               = phi_pres_s_2_m.get_value(q);
          const auto& pres_s_2_p               = phi_pres_s_2_p.get_value(q);

          const auto& avg_tensor_product_u_s_2 = 0.5*(outer_product(rho_s_2_m*u_s_2_m, u_s_2_m) +
                                                      outer_product(rho_s_2_p*u_s_2_p, u_s_2_p));
          const auto& avg_pres_s_2             = 0.5*(pres_s_2_m + pres_s_2_p);

          const auto& jump_rhou_s_2            = rho_s_2_m*u_s_2_m - rho_s_2_p*u_s_2_p;
          const auto& lambda_s_2               = compute_lambda(u_s_2_m, u_s_2_p, n_minus);

          phi_m.submit_value(-a31*dt*(avg_tensor_product_u_n*n_minus +
                                      0.5*lambda_old*jump_rhou_old)
                             -a31_tilde*dt*(avg_pres_old/(Ma*Ma)*n_minus)
                             -a32*dt*(avg_tensor_product_u_s_2*n_minus +
                                      0.5*lambda_s_2*jump_rhou_s_2)
                             -a32_tilde*dt*(avg_pres_s_2/(Ma*Ma)*n_minus), q);
          phi_p.submit_value(a31*dt*(avg_tensor_product_u_n*n_minus +
                                     0.5*lambda_old*jump_rhou_old) +
                             a31_tilde*dt*(avg_pres_old/(Ma*Ma)*n_minus) +
                             a32*dt*(avg_tensor_product_u_s_2*n_minus +
                                     0.5*lambda_s_2*jump_rhou_s_2) +
                             a32_tilde*dt*(avg_pres_s_2/(Ma*Ma)*n_minus), q);
        }

        phi_m.integrate_scatter(EvaluationFlags::values, dst);
        phi_p.integrate_scatter(EvaluationFlags::values, dst);
      }
    }
    else {
      /*--- We first start by declaring the suitable instances to read the available quantities. ---*/
      FEFaceEvaluation<dim, fe_degree_u, n_q_points_1d, dim, Number> phi_m(data, true, EquationData::U_INDEX_DOF),
                                                                     phi_p(data, false, EquationData::U_INDEX_DOF),
                                                                     phi_u_old_m(data, true, EquationData::U_INDEX_DOF),
                                                                     phi_u_old_p(data, false, EquationData::U_INDEX_DOF),
                                                                     phi_u_s_2_m(data, true, EquationData::U_INDEX_DOF),
                                                                     phi_u_s_2_p(data, false, EquationData::U_INDEX_DOF),
                                                                     phi_u_s_3_m(data, true, EquationData::U_INDEX_DOF),
                                                                     phi_u_s_3_p(data, false, EquationData::U_INDEX_DOF);
      FEFaceEvaluation<dim, fe_degree_p, n_q_points_1d, 1, Number>   phi_pres_old_m(data, true, EquationData::P_INDEX_DOF),
                                                                     phi_pres_old_p(data, false, EquationData::P_INDEX_DOF),
                                                                     phi_pres_s_2_m(data, true, EquationData::P_INDEX_DOF),
                                                                     phi_pres_s_2_p(data, false, EquationData::P_INDEX_DOF),
                                                                     phi_pres_s_3_m(data, true, EquationData::P_INDEX_DOF),
                                                                     phi_pres_s_3_p(data, false, EquationData::P_INDEX_DOF);
      FEFaceEvaluation<dim, fe_degree_rho, n_q_points_1d, 1, Number> phi_rho_old_m(data, true, EquationData::RHO_INDEX_DOF),
                                                                     phi_rho_old_p(data, false, EquationData::RHO_INDEX_DOF),
                                                                     phi_rho_s_2_m(data, true, EquationData::RHO_INDEX_DOF),
                                                                     phi_rho_s_2_p(data, false, EquationData::RHO_INDEX_DOF),
                                                                     phi_rho_s_3_m(data, true, EquationData::RHO_INDEX_DOF),
                                                                     phi_rho_s_3_p(data, false, EquationData::RHO_INDEX_DOF);

      /*--- Loop over all internal faces ---*/
      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
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
        for(unsigned int q = 0; q < phi_m.n_q_points; ++q) {
          const auto& n_minus                  = phi_m.get_normal_vector(q);

          /*--- Compute the quantities at the previous step ---*/
          const auto& rho_old_m                = phi_rho_old_m.get_value(q);
          const auto& rho_old_p                = phi_rho_old_p.get_value(q);
          const auto& u_old_m                  = phi_u_old_m.get_value(q);
          const auto& u_old_p                  = phi_u_old_p.get_value(q);
          const auto& pres_old_m               = phi_pres_old_m.get_value(q);
          const auto& pres_old_p               = phi_pres_old_p.get_value(q);

          const auto& avg_tensor_product_u_n   = 0.5*(outer_product(rho_old_m*u_old_m, u_old_m) +
                                                      outer_product(rho_old_p*u_old_p, u_old_p));
          const auto& avg_pres_old             = 0.5*(pres_old_m + pres_old_p);

          const auto& jump_rhou_old            = rho_old_m*u_old_m - rho_old_p*u_old_p;
          const auto& lambda_old               = compute_lambda(u_old_m, u_old_p, n_minus);

          /*--- Compute the quantities at the previous stage ---*/
          const auto& rho_s_2_m                = phi_rho_s_2_m.get_value(q);
          const auto& rho_s_2_p                = phi_rho_s_2_p.get_value(q);
          const auto& u_s_2_m                  = phi_u_s_2_m.get_value(q);
          const auto& u_s_2_p                  = phi_u_s_2_p.get_value(q);
          const auto& pres_s_2_m               = phi_pres_s_2_m.get_value(q);
          const auto& pres_s_2_p               = phi_pres_s_2_p.get_value(q);

          const auto& avg_tensor_product_u_s_2 = 0.5*(outer_product(rho_s_2_m*u_s_2_m, u_s_2_m) +
                                                      outer_product(rho_s_2_p*u_s_2_p, u_s_2_p));
          const auto& avg_pres_s_2             = 0.5*(pres_s_2_m + pres_s_2_p);

          const auto& jump_rhou_s_2            = rho_s_2_m*u_s_2_m - rho_s_2_p*u_s_2_p;
          const auto& lambda_s_2               = compute_lambda(u_s_2_m, u_s_2_p, n_minus);

          /*--- Compute the quantities at the final stage ---*/
          const auto& rho_s_3_m                = phi_rho_s_3_m.get_value(q);
          const auto& rho_s_3_p                = phi_rho_s_3_p.get_value(q);
          const auto& u_s_3_m                  = phi_u_s_3_m.get_value(q);
          const auto& u_s_3_p                  = phi_u_s_3_p.get_value(q);
          const auto& pres_s_3_m               = phi_pres_s_3_m.get_value(q);
          const auto& pres_s_3_p               = phi_pres_s_3_p.get_value(q);

          const auto& avg_tensor_product_u_s_3 = 0.5*(outer_product(rho_s_3_m*u_s_3_m, u_s_3_m) +
                                                      outer_product(rho_s_3_p*u_s_3_p, u_s_3_p));
          const auto& avg_pres_s_3             = 0.5*(pres_s_3_m + pres_s_3_p);

          const auto& jump_rhou_s_3            = rho_s_3_m*u_s_3_m - rho_s_3_p*u_s_3_p;
          const auto& lambda_s_3               = compute_lambda(u_s_3_m, u_s_3_p, n_minus);

          phi_m.submit_value(-b1*dt*(avg_tensor_product_u_n*n_minus +
                                     0.5*lambda_old*jump_rhou_old)
                             -b1*dt*(avg_pres_old/(Ma*Ma)*n_minus)
                             -b2*dt*(avg_tensor_product_u_s_2*n_minus +
                                     0.5*lambda_s_2*jump_rhou_s_2)
                             -b2*dt*(avg_pres_s_2/(Ma*Ma)*n_minus)
                             -b3*dt*(avg_tensor_product_u_s_3*n_minus +
                                     0.5*lambda_s_3*jump_rhou_s_3)
                             -b3*dt*(avg_pres_s_3/(Ma*Ma)*n_minus), q);
          phi_p.submit_value(b1*dt*(avg_tensor_product_u_n*n_minus +
                                    0.5*lambda_old*jump_rhou_old) +
                             b1*dt*(avg_pres_old/(Ma*Ma)*n_minus) +
                             b2*dt*(avg_tensor_product_u_s_2*n_minus +
                                    0.5*lambda_s_2*jump_rhou_s_2) +
                             b2*dt*(avg_pres_s_2/(Ma*Ma)*n_minus) +
                             b3*dt*(avg_tensor_product_u_s_3*n_minus +
                                    0.5*lambda_s_3*jump_rhou_s_3) +
                             b3*dt*(avg_pres_s_3/(Ma*Ma)*n_minus), q);
        }

        phi_m.integrate_scatter(EvaluationFlags::values, dst);
        phi_p.integrate_scatter(EvaluationFlags::values, dst);
      }
    }
  }

  // Assemble rhs boundary term of the momentum equation
  //
  template<int dim, int fe_degree_u, int fe_degree_rho, int fe_degree_p,
           int n_q_points_1d, int n_q_points_1d_boundary, typename Vec>
  void EULEROperator<dim, fe_degree_u, fe_degree_rho, fe_degree_p,
                     n_q_points_1d, n_q_points_1d_boundary, Vec>::
  assemble_rhs_boundary_term_momentum(const MatrixFree<dim, Number>&               data,
                                      Vec&                                         dst,
                                      const std::vector<Vec>&                      src,
                                      const std::pair<unsigned int, unsigned int>& face_range) const {
    if(IMEX_stage == 2) {
      /*--- We first start by declaring the suitable instances to read the available quantities. ---*/
      FEFaceEvaluation<dim, fe_degree_u, n_q_points_1d_boundary, dim, Number> phi(data, true, EquationData::U_INDEX_DOF, 1);
      FEFaceEvaluation<dim, fe_degree_p, n_q_points_1d_boundary, 1, Number>   phi_pres_old(data, true, EquationData::P_INDEX_DOF, 1);

      /*--- Loop over all boundary faces ---*/
      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        phi_pres_old.reinit(face);
        phi_pres_old.gather_evaluate(src[2], EvaluationFlags::values);

        phi.reinit(face);

        /*--- Loop over all quadrature points ---*/
        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          const auto& n_minus      = phi.get_normal_vector(q);

          /*--- Compute the quantities at the previous step ---*/
          const auto& pres_old     = phi_pres_old.get_value(q);
          const auto& pres_old_D   = pres_old;

          const auto& avg_pres_old = 0.5*(pres_old + pres_old_D);

          phi.submit_value(-a21_tilde*dt*(avg_pres_old/(Ma*Ma)*n_minus), q);
        }

        phi.integrate_scatter(EvaluationFlags::values, dst);
      }
    }
    else if(IMEX_stage == 3) {
      /*--- We first start by declaring the suitable instances to read the available quantities. ---*/
      FEFaceEvaluation<dim, fe_degree_u, n_q_points_1d_boundary, dim, Number> phi(data, true, EquationData::U_INDEX_DOF, 1);
      FEFaceEvaluation<dim, fe_degree_p, n_q_points_1d_boundary, 1, Number>   phi_pres_old(data, true, EquationData::P_INDEX_DOF, 1),
                                                                              phi_pres_s_2(data, true, EquationData::P_INDEX_DOF, 1);

      /*--- Loop over all boundary faces ---*/
      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        phi_pres_old.reinit(face);
        phi_pres_old.gather_evaluate(src[2], EvaluationFlags::values);

        phi_pres_s_2.reinit(face);
        phi_pres_s_2.gather_evaluate(src[5], EvaluationFlags::values);

        phi.reinit(face);

        /*--- Loop over all quadrature points ---*/
        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          const auto& n_minus      = phi.get_normal_vector(q);

          /*--- Compute the quantities at the previous step ---*/
          const auto& pres_old     = phi_pres_old.get_value(q);
          const auto& pres_old_D   = pres_old;

          const auto& avg_pres_old = 0.5*(pres_old + pres_old_D);

          /*--- Compute the quantities at the previous stage ---*/
          const auto& pres_s_2     = phi_pres_s_2.get_value(q);
          const auto& pres_s_2_D   = pres_s_2;

          const auto& avg_pres_s_2 = 0.5*(pres_s_2 + pres_s_2_D);

          phi.submit_value(-a31_tilde*dt*(avg_pres_old/(Ma*Ma)*n_minus)
                           -a32_tilde*dt*(avg_pres_s_2/(Ma*Ma)*n_minus), q);
        }

        phi.integrate_scatter(EvaluationFlags::values, dst);
      }
    }
    else {
      /*--- We first start by declaring the suitable instances to read the available quantities. ---*/
      FEFaceEvaluation<dim, fe_degree_u, n_q_points_1d_boundary, dim, Number> phi(data, true, EquationData::U_INDEX_DOF, 1);
      FEFaceEvaluation<dim, fe_degree_p, n_q_points_1d_boundary, 1, Number>   phi_pres_old(data, true, EquationData::P_INDEX_DOF, 1),
                                                                              phi_pres_s_2(data, true, EquationData::P_INDEX_DOF, 1),
                                                                              phi_pres_s_3(data, true, EquationData::P_INDEX_DOF, 1);

      /*--- Loop over all boundary faces ---*/
      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        phi_pres_old.reinit(face);
        phi_pres_old.gather_evaluate(src[2], EvaluationFlags::values);

        phi_pres_s_2.reinit(face);
        phi_pres_s_2.gather_evaluate(src[5], EvaluationFlags::values);

        phi_pres_s_3.reinit(face);
        phi_pres_s_3.gather_evaluate(src[8], EvaluationFlags::values);

        phi.reinit(face);

        /*--- Loop over all quadrature points ---*/
        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          const auto& n_minus      = phi.get_normal_vector(q);

          /*--- Compute the quantities at the previous step ---*/
          const auto& pres_old     = phi_pres_old.get_value(q);
          const auto& pres_old_D   = pres_old;

          const auto& avg_pres_old = 0.5*(pres_old + pres_old_D);

          /*--- Compute the quantities at the previous stage ---*/
          const auto& pres_s_2     = phi_pres_s_2.get_value(q);
          const auto& pres_s_2_D   = pres_s_2;

          const auto& avg_pres_s_2 = 0.5*(pres_s_2 + pres_s_2_D);

          /*--- Compute the quantities at the final steage---*/
          const auto& pres_s_3     = phi_pres_s_3.get_value(q);
          const auto& pres_s_3_D   = pres_s_3;

          const auto& avg_pres_s_3 = 0.5*(pres_s_3 + pres_s_3_D);

          phi.submit_value(-b1*dt*(avg_pres_old/(Ma*Ma)*n_minus)
                           -b2*dt*(avg_pres_s_2/(Ma*Ma)*n_minus)
                           -b3*dt*(avg_pres_s_3/(Ma*Ma)*n_minus), q);
        }

        phi.integrate_scatter(EvaluationFlags::values, dst);
      }
    }
  }

  // Put together all the previous steps for the momentum equation
  //
  template<int dim, int fe_degree_u, int fe_degree_rho, int fe_degree_p,
           int n_q_points_1d, int n_q_points_1d_boundary, typename Vec>
  void EULEROperator<dim, fe_degree_u, fe_degree_rho, fe_degree_p,
                     n_q_points_1d, n_q_points_1d_boundary, Vec>::
  vmult_rhs_momentum(Vec& dst, const std::vector<Vec>& src) const {
    for(unsigned int d = 0; d < src.size(); ++d) {
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
  template<int dim, int fe_degree_u, int fe_degree_rho, int fe_degree_p,
           int n_q_points_1d, int n_q_points_1d_boundary, typename Vec>
  void EULEROperator<dim, fe_degree_u, fe_degree_rho, fe_degree_p,
                     n_q_points_1d, n_q_points_1d_boundary, Vec>::
  assemble_cell_term_velocity(const MatrixFree<dim, Number>&               data,
                              Vec&                                         dst,
                              const Vec&                                   src,
                              const std::pair<unsigned int, unsigned int>& cell_range) const {
    /*--- We first start by declaring the suitable instances to read also available quantities.
          Since here we have just one 'src' vector, but we also need to deal with the current density,
          we employ the auxiliary vector 'rho_for_fixed' where we setted this information ---*/
    FEEvaluation<dim, fe_degree_u, n_q_points_1d, dim, Number> phi(data, EquationData::U_INDEX_DOF);
    FEEvaluation<dim, fe_degree_rho, n_q_points_1d, 1, Number> phi_rho_for_fixed(data, EquationData::RHO_INDEX_DOF);

    /*--- Loop over all cells ---*/
    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
      phi_rho_for_fixed.reinit(cell);
      phi_rho_for_fixed.gather_evaluate(rho_for_fixed, EvaluationFlags::values);

      phi.reinit(cell);
      phi.gather_evaluate(src, EvaluationFlags::values);

      /*--- Loop over all quadrature points ---*/
      for(unsigned int q = 0; q < phi.n_q_points; ++q) {
        phi.submit_value(phi_rho_for_fixed.get_value(q)*phi.get_value(q), q);
      }

      phi.integrate_scatter(EvaluationFlags::values, dst);
    }
  }

  // Assemble cell term for the pressure
  //
  template<int dim, int fe_degree_u, int fe_degree_rho, int fe_degree_p,
           int n_q_points_1d, int n_q_points_1d_boundary, typename Vec>
  void EULEROperator<dim, fe_degree_u, fe_degree_rho, fe_degree_p,
                     n_q_points_1d, n_q_points_1d_boundary, Vec>::
  assemble_cell_term_pressure(const MatrixFree<dim, Number>&               data,
                              Vec&                                         dst,
                              const Vec&                                   src,
                              const std::pair<unsigned int, unsigned int>& cell_range) const {
    /*--- We first start by declaring the suitable instances to read quantities. This operator we are going to implement
          represents a rectangular matrix (we start from the pressure FE space and we end up with the velocity FE space).
          This is the reason of the distinction between 'phi' and 'phi_src'. ---*/
    FEEvaluation<dim, fe_degree_u, n_q_points_1d, dim, Number> phi(data, EquationData::U_INDEX_DOF);
    FEEvaluation<dim, fe_degree_p, n_q_points_1d, 1, Number>   phi_src(data, EquationData::P_INDEX_DOF);

    const double coeff = (IMEX_stage == 2) ? a22_tilde : a33_tilde;

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
      phi_src.reinit(cell);
      phi_src.gather_evaluate(src, EvaluationFlags::values);

      phi.reinit(cell);

      for(unsigned int q = 0; q < phi.n_q_points; ++q) {
        /*--- Here we are testing against the divergence of the test function and, therefore, we employ 'submit_divergence'. ---*/
        phi.submit_divergence(-coeff*dt*(phi_src.get_value(q)/(Ma*Ma)), q);
      }

      phi.integrate_scatter(EvaluationFlags::gradients, dst);
    }
  }

  // Assemble face term for the pressure
  //
  template<int dim, int fe_degree_u, int fe_degree_rho, int fe_degree_p,
           int n_q_points_1d, int n_q_points_1d_boundary, typename Vec>
  void EULEROperator<dim, fe_degree_u, fe_degree_rho, fe_degree_p,
                     n_q_points_1d, n_q_points_1d_boundary, Vec>::
  assemble_face_term_pressure(const MatrixFree<dim, Number>&               data,
                              Vec&                                         dst,
                              const Vec&                                   src,
                              const std::pair<unsigned int, unsigned int>& face_range) const {
    FEFaceEvaluation<dim, fe_degree_u, n_q_points_1d, dim, Number> phi_m(data, true, EquationData::U_INDEX_DOF),
                                                                   phi_p(data, false, EquationData::U_INDEX_DOF);
    FEFaceEvaluation<dim, fe_degree_p, n_q_points_1d, 1, Number>   phi_src_m(data, true, EquationData::P_INDEX_DOF),
                                                                   phi_src_p(data, false, EquationData::P_INDEX_DOF);

    const double coeff = (IMEX_stage == 2) ? a22_tilde : a33_tilde;

    /*--- Loop over all internal faces ---*/
    for(unsigned int face = face_range.first; face < face_range.second; ++face) {
      phi_src_m.reinit(face);
      phi_src_m.gather_evaluate(src, EvaluationFlags::values);
      phi_src_p.reinit(face);
      phi_src_p.gather_evaluate(src, EvaluationFlags::values);

      phi_m.reinit(face);
      phi_p.reinit(face);

      /*--- Loop over all quadrature points ---*/
      for(unsigned int q = 0; q < phi_m.n_q_points; ++q) {
        const auto& n_minus  = phi_m.get_normal_vector(q);

        const auto& avg_term = 0.5*(phi_src_m.get_value(q) + phi_src_p.get_value(q));

        phi_m.submit_value(coeff*dt*(avg_term/(Ma*Ma)*n_minus), q);
        phi_p.submit_value(-coeff*dt*(avg_term/(Ma*Ma)*n_minus), q);
      }

      phi_m.integrate_scatter(EvaluationFlags::values, dst);
      phi_p.integrate_scatter(EvaluationFlags::values, dst);
    }
  }

  // Assemble boundary term for the pressure
  //
  template<int dim, int fe_degree_u, int fe_degree_rho, int fe_degree_p,
           int n_q_points_1d, int n_q_points_1d_boundary, typename Vec>
  void EULEROperator<dim, fe_degree_u, fe_degree_rho, fe_degree_p,
                     n_q_points_1d, n_q_points_1d_boundary, Vec>::
  assemble_boundary_term_pressure(const MatrixFree<dim, Number>&               data,
                                  Vec&                                         dst,
                                  const Vec&                                   src,
                                  const std::pair<unsigned int, unsigned int>& face_range) const {
    FEFaceEvaluation<dim, fe_degree_u, n_q_points_1d_boundary, dim, Number> phi(data, true, EquationData::U_INDEX_DOF, 1);
    FEFaceEvaluation<dim, fe_degree_p, n_q_points_1d_boundary, 1, Number>   phi_src(data, true, EquationData::P_INDEX_DOF, 1);

    const double coeff = (IMEX_stage == 2) ? a22_tilde : a33_tilde;

    /*--- Loop over all boundary faces ---*/
    for(unsigned int face = face_range.first; face < face_range.second; ++face) {
      phi_src.reinit(face);
      phi_src.gather_evaluate(src, EvaluationFlags::values);

      phi.reinit(face);

      /*--- Loop over all quadrature points ---*/
      for(unsigned int q = 0; q < phi.n_q_points; ++q) {
        const auto& n_minus      = phi.get_normal_vector(q);

        const auto& pres_fixed_D = phi_src.get_value(q);

        phi.submit_value(coeff*dt*((0.5*(phi_src.get_value(q) + pres_fixed_D))/(Ma*Ma)*n_minus), q);
      }

      phi.integrate_scatter(EvaluationFlags::values, dst);
    }
  }


  // Assemble rhs cell term of the energy equation
  //
  template<int dim, int fe_degree_u, int fe_degree_rho, int fe_degree_p,
           int n_q_points_1d, int n_q_points_1d_boundary, typename Vec>
  void EULEROperator<dim, fe_degree_u, fe_degree_rho, fe_degree_p,
                     n_q_points_1d, n_q_points_1d_boundary, Vec>::
  assemble_rhs_cell_term_energy(const MatrixFree<dim, Number>&               data,
                                Vec&                                         dst,
                                const std::vector<Vec>&                      src,
                                const std::pair<unsigned int, unsigned int>& cell_range) const {
    if(IMEX_stage == 2) {
      /*--- We first start by declaring the suitable instances to read the available quantities. ---*/
      FEEvaluation<dim, fe_degree_p, n_q_points_1d, 1, Number>   phi(data, EquationData::P_INDEX_DOF),
                                                                 phi_pres_old(data, EquationData::P_INDEX_DOF);
      FEEvaluation<dim, fe_degree_u, n_q_points_1d, dim, Number> phi_u_old(data, EquationData::U_INDEX_DOF),
                                                                 phi_u_fixed(data, EquationData::U_INDEX_DOF);
      FEEvaluation<dim, fe_degree_rho, n_q_points_1d, 1, Number> phi_rho_old(data, EquationData::RHO_INDEX_DOF),
                                                                 phi_rho_s_2(data, EquationData::RHO_INDEX_DOF);

      /*--- Loop over all cells ---*/
      for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
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
        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          /*--- Compute the quantities at the previous step ---*/
          const auto& rho_old  = phi_rho_old.get_value(q);
          const auto& u_old    = phi_u_old.get_value(q);
          const auto& pres_old = phi_pres_old.get_value(q);

          /*--- We assign to the rhs the contribution due to kinetic energy in the fixed point loop ---*/
          const auto& rho_s_2 = phi_rho_s_2.get_value(q);
          const auto& u_fixed = phi_u_fixed.get_value(q);

          phi.submit_value(1.0/(EquationData::Cp_Cv - 1.0)*pres_old +
                           rho_old*(0.5*Ma*Ma*scalar_product(u_old, u_old)) -
                           rho_s_2*(0.5*Ma*Ma*scalar_product(u_fixed, u_fixed)) -
                           a21_tilde*dt*(Ma*Ma/(Fr*Fr)*rho_old*u_old[dim - 1]) -
                           a22_tilde*dt*(Ma*Ma/(Fr*Fr)*rho_s_2*u_fixed[dim - 1]), q);
          phi.submit_gradient(a21*dt*(0.5*Ma*Ma*scalar_product(u_old, u_old)*rho_old*u_old) +
                              a21_tilde*dt*(EquationData::Cp_Cv/(EquationData::Cp_Cv - 1.0)*pres_old*u_old), q);
          /*--- The specific enthalpy is computed with the generic relation e + p/rho ---*/
        }

        phi.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
      }
    }
    else if(IMEX_stage == 3) {
      /*--- We first start by declaring the suitable instances to read the available quantities. ---*/
      FEEvaluation<dim, fe_degree_p, n_q_points_1d, 1, Number>   phi(data, EquationData::P_INDEX_DOF),
                                                                 phi_pres_old(data, EquationData::P_INDEX_DOF),
                                                                 phi_pres_s_2(data, EquationData::P_INDEX_DOF);
      FEEvaluation<dim, fe_degree_u, n_q_points_1d, dim, Number> phi_u_old(data, EquationData::U_INDEX_DOF),
                                                                 phi_u_s_2(data, EquationData::U_INDEX_DOF),
                                                                 phi_u_fixed(data, EquationData::U_INDEX_DOF);
      FEEvaluation<dim, fe_degree_rho, n_q_points_1d, 1, Number> phi_rho_old(data, EquationData::RHO_INDEX_DOF),
                                                                 phi_rho_s_2(data, EquationData::RHO_INDEX_DOF),
                                                                 phi_rho_s_3(data, EquationData::RHO_INDEX_DOF);

      /*--- Loop over all cells ---*/
      for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
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
        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
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
                           rho_old*(0.5*Ma*Ma*scalar_product(u_old, u_old)) -
                           rho_s_3*(0.5*Ma*Ma*scalar_product(u_fixed, u_fixed)) -
                           a31_tilde*dt*(Ma*Ma/(Fr*Fr)*rho_old*u_old[dim - 1]) -
                           a32_tilde*dt*(Ma*Ma/(Fr*Fr)*rho_s_2*u_s_2[dim - 1]) -
                           a33_tilde*dt*(Ma*Ma/(Fr*Fr)*rho_s_3*u_fixed[dim - 1]), q);
          phi.submit_gradient(a31*dt*(0.5*Ma*Ma*scalar_product(u_old, u_old)*rho_old*u_old) +
                              a31_tilde*dt*(EquationData::Cp_Cv/(EquationData::Cp_Cv - 1.0)*pres_old*u_old) +
                              a32*dt*(0.5*Ma*Ma*scalar_product(u_s_2, u_s_2)*rho_s_2*u_s_2) +
                              a32_tilde*dt*(EquationData::Cp_Cv/(EquationData::Cp_Cv - 1.0)*pres_s_2*u_s_2), q);
        }

        phi.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
      }
    }
    else {
      /*--- We first start by declaring the suitable instances to read the available quantities. ---*/
      FEEvaluation<dim, fe_degree_p, n_q_points_1d, 1, Number>   phi(data, EquationData::P_INDEX_DOF),
                                                                 phi_pres_old(data, EquationData::P_INDEX_DOF),
                                                                 phi_pres_s_2(data, EquationData::P_INDEX_DOF),
                                                                 phi_pres_s_3(data, EquationData::P_INDEX_DOF);
      FEEvaluation<dim, fe_degree_u, n_q_points_1d, dim, Number> phi_u_old(data, EquationData::U_INDEX_DOF),
                                                                 phi_u_s_2(data, EquationData::U_INDEX_DOF),
                                                                 phi_u_s_3(data, EquationData::U_INDEX_DOF),
                                                                 phi_u_curr(data, EquationData::U_INDEX_DOF);
      FEEvaluation<dim, fe_degree_rho, n_q_points_1d, 1, Number> phi_rho_old(data, EquationData::RHO_INDEX_DOF),
                                                                 phi_rho_s_2(data, EquationData::RHO_INDEX_DOF),
                                                                 phi_rho_s_3(data, EquationData::RHO_INDEX_DOF),
                                                                 phi_rho_curr(data, EquationData::RHO_INDEX_DOF);

      /*--- Loop over all cells ---*/
      for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
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
        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
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
                           rho_old*(0.5*Ma*Ma*scalar_product(u_old, u_old)) -
                           rho_curr*(0.5*Ma*Ma*scalar_product(u_curr, u_curr)) -
                           b1*dt*(Ma*Ma/(Fr*Fr)*rho_old*u_old[dim - 1]) -
                           b2*dt*(Ma*Ma/(Fr*Fr)*rho_s_2*u_s_2[dim - 1]) -
                           b3*dt*(Ma*Ma/(Fr*Fr)*rho_s_3*u_s_3[dim - 1]), q);
          phi.submit_gradient(b1*dt*(0.5*Ma*Ma*scalar_product(u_old, u_old)*rho_old*u_old) +
                              b1*dt*(EquationData::Cp_Cv/(EquationData::Cp_Cv - 1.0)*pres_old*u_old) +
                              b2*dt*(0.5*Ma*Ma*scalar_product(u_s_2, u_s_2)*rho_s_2*u_s_2) +
                              b2*dt*(EquationData::Cp_Cv/(EquationData::Cp_Cv - 1.0)*pres_s_2*u_s_2) +
                              b3*dt*(0.5*Ma*Ma*scalar_product(u_s_3, u_s_3)*rho_s_3*u_s_3) +
                              b3*dt*(EquationData::Cp_Cv/(EquationData::Cp_Cv - 1.0)*pres_s_3*u_s_3), q);
        }

        phi.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
      }
    }
  }

  // Assemble rhs face term of the energy equation
  //
  template<int dim, int fe_degree_u, int fe_degree_rho, int fe_degree_p,
           int n_q_points_1d, int n_q_points_1d_boundary, typename Vec>
  void EULEROperator<dim, fe_degree_u, fe_degree_rho, fe_degree_p,
                     n_q_points_1d, n_q_points_1d_boundary, Vec>::
  assemble_rhs_face_term_energy(const MatrixFree<dim, Number>&               data,
                                Vec&                                         dst,
                                const std::vector<Vec>&                      src,
                                const std::pair<unsigned int, unsigned int>& face_range) const {
    if(IMEX_stage == 2) {
      /*--- We first start by declaring the suitable instances to read the available quantities. ---*/
      FEFaceEvaluation<dim, fe_degree_p, n_q_points_1d, 1, Number>   phi_m(data, true, EquationData::P_INDEX_DOF),
                                                                     phi_p(data, false, EquationData::P_INDEX_DOF),
                                                                     phi_pres_old_m(data, true, EquationData::P_INDEX_DOF),
                                                                     phi_pres_old_p(data, false, EquationData::P_INDEX_DOF),
                                                                     phi_pres_fixed_m(data, true, EquationData::P_INDEX_DOF),
                                                                     phi_pres_fixed_p(data, false, EquationData::P_INDEX_DOF);
      FEFaceEvaluation<dim, fe_degree_u, n_q_points_1d, dim, Number> phi_u_old_m(data, true, EquationData::U_INDEX_DOF),
                                                                     phi_u_old_p(data, false, EquationData::U_INDEX_DOF),
                                                                     phi_u_fixed_m(data, true, EquationData::U_INDEX_DOF),
                                                                     phi_u_fixed_p(data, false, EquationData::U_INDEX_DOF);
      FEFaceEvaluation<dim, fe_degree_rho, n_q_points_1d, 1, Number> phi_rho_old_m(data, true, EquationData::RHO_INDEX_DOF),
                                                                     phi_rho_old_p(data, false, EquationData::RHO_INDEX_DOF);

      /*--- Loop over all internal faces ---*/
      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
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
        for(unsigned int q = 0; q < phi_m.n_q_points; ++q) {
          const auto& n_minus          = phi_m.get_normal_vector(q);

          /*--- Compute the quantities at the previous step ---*/
          const auto& rho_old_m        = phi_rho_old_m.get_value(q);
          const auto& rho_old_p        = phi_rho_old_p.get_value(q);
          const auto& u_old_m          = phi_u_old_m.get_value(q);
          const auto& u_old_p          = phi_u_old_p.get_value(q);
          const auto& avg_kinetic_old  = 0.5*(0.5*scalar_product(u_old_m, u_old_m)*rho_old_m*u_old_m +
                                              0.5*scalar_product(u_old_p, u_old_p)*rho_old_p*u_old_p);

          const auto& pres_old_m       = phi_pres_old_m.get_value(q);
          const auto& pres_old_p       = phi_pres_old_p.get_value(q);
          const auto& avg_enthalpy_old = 0.5*EquationData::Cp_Cv/(EquationData::Cp_Cv - 1.0)*
                                         (pres_old_m*u_old_m + pres_old_p*u_old_p);

          const auto& lambda_old       = compute_lambda(u_old_m, u_old_p, n_minus);
          const auto& jump_rho_kin_old = rho_old_m*(0.5*scalar_product(u_old_m, u_old_m)) -
                                         rho_old_p*(0.5*scalar_product(u_old_p, u_old_p));
          const auto& jump_rho_e_old   = 1.0/(EquationData::Cp_Cv - 1.0)*(pres_old_m - pres_old_p);

          /*--- Compute the quantities at the current stage ---*/
          const auto& u_fixed_m        = phi_u_fixed_m.get_value(q);
          const auto& u_fixed_p        = phi_u_fixed_p.get_value(q);
          const auto& pres_fixed_m     = phi_pres_fixed_m.get_value(q);
          const auto& pres_fixed_p     = phi_pres_fixed_p.get_value(q);

          const auto& lambda_fixed     = compute_lambda(u_fixed_m, u_fixed_p, n_minus);
          const auto& jump_rho_e_fixed = 1.0/(EquationData::Cp_Cv - 1.0)*(pres_fixed_m - pres_fixed_p);

          phi_m.submit_value(-a21*dt*(Ma*Ma*(scalar_product(avg_kinetic_old, n_minus) + 0.5*lambda_old*jump_rho_kin_old))
                             -a21_tilde*dt*(scalar_product(avg_enthalpy_old, n_minus) + 0.5*lambda_old*jump_rho_e_old)
                             -a22_tilde*dt*(0.5*lambda_fixed*jump_rho_e_fixed), q);
          phi_p.submit_value(a21*dt*(Ma*Ma*(scalar_product(avg_kinetic_old, n_minus) + 0.5*lambda_old*jump_rho_kin_old)) +
                             a21_tilde*dt*(scalar_product(avg_enthalpy_old, n_minus) + 0.5*lambda_old*jump_rho_e_old) +
                             a22_tilde*dt*(0.5*lambda_fixed*jump_rho_e_fixed), q);
        }

        phi_m.integrate_scatter(EvaluationFlags::values, dst);
        phi_p.integrate_scatter(EvaluationFlags::values, dst);
      }
    }
    else if(IMEX_stage == 3) {
      /*--- We first start by declaring the suitable instances to read the available quantities. ---*/
      FEFaceEvaluation<dim, fe_degree_p, n_q_points_1d, 1, Number>   phi_m(data, true, EquationData::P_INDEX_DOF),
                                                                     phi_p(data, false, EquationData::P_INDEX_DOF),
                                                                     phi_pres_old_m(data, true, EquationData::P_INDEX_DOF),
                                                                     phi_pres_old_p(data, false, EquationData::P_INDEX_DOF),
                                                                     phi_pres_s_2_m(data, true, EquationData::P_INDEX_DOF),
                                                                     phi_pres_s_2_p(data, false, EquationData::P_INDEX_DOF),
                                                                     phi_pres_fixed_m(data, true, EquationData::P_INDEX_DOF),
                                                                     phi_pres_fixed_p(data, false, EquationData::P_INDEX_DOF);
      FEFaceEvaluation<dim, fe_degree_u, n_q_points_1d, dim, Number> phi_u_old_m(data, true, EquationData::U_INDEX_DOF),
                                                                     phi_u_old_p(data, false, EquationData::U_INDEX_DOF),
                                                                     phi_u_s_2_m(data, true, EquationData::U_INDEX_DOF),
                                                                     phi_u_s_2_p(data, false, EquationData::U_INDEX_DOF),
                                                                     phi_u_fixed_m(data, true, EquationData::U_INDEX_DOF),
                                                                     phi_u_fixed_p(data, false, EquationData::U_INDEX_DOF);
      FEFaceEvaluation<dim, fe_degree_rho, n_q_points_1d, 1, Number> phi_rho_old_m(data, true, EquationData::RHO_INDEX_DOF),
                                                                     phi_rho_old_p(data, false, EquationData::RHO_INDEX_DOF),
                                                                     phi_rho_s_2_m(data, true, EquationData::RHO_INDEX_DOF),
                                                                     phi_rho_s_2_p(data, false, EquationData::RHO_INDEX_DOF);

      /*--- Loop over all internal faces ---*/
      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
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
        for(unsigned int q = 0; q < phi_m.n_q_points; ++q) {
          const auto& n_minus          = phi_m.get_normal_vector(q);

          /*--- Compute the quantities at the previous step ---*/
          const auto& rho_old_m        = phi_rho_old_m.get_value(q);
          const auto& rho_old_p        = phi_rho_old_p.get_value(q);
          const auto& u_old_m          = phi_u_old_m.get_value(q);
          const auto& u_old_p          = phi_u_old_p.get_value(q);
          const auto& avg_kinetic_old  = 0.5*(0.5*scalar_product(u_old_m, u_old_m)*rho_old_m*u_old_m +
                                              0.5*scalar_product(u_old_p, u_old_p)*rho_old_p*u_old_p);

          const auto& pres_old_m       = phi_pres_old_m.get_value(q);
          const auto& pres_old_p       = phi_pres_old_p.get_value(q);
          const auto& avg_enthalpy_old = 0.5*EquationData::Cp_Cv/(EquationData::Cp_Cv - 1.0)*
                                         (pres_old_m*u_old_m + pres_old_p*u_old_p);

          const auto& lambda_old       = compute_lambda(u_old_m, u_old_p, n_minus);
          const auto& jump_rho_kin_old = rho_old_m*(0.5*scalar_product(u_old_m, u_old_m)) -
                                         rho_old_p*(0.5*scalar_product(u_old_p, u_old_p));
          const auto& jump_rho_e_old   = 1.0/(EquationData::Cp_Cv - 1.0)*(pres_old_m - pres_old_p);

          /*--- Compute the quantities at the previous stage ---*/
          const auto& rho_s_2_m        = phi_rho_s_2_m.get_value(q);
          const auto& rho_s_2_p        = phi_rho_s_2_p.get_value(q);
          const auto& u_s_2_m          = phi_u_s_2_m.get_value(q);
          const auto& u_s_2_p          = phi_u_s_2_p.get_value(q);
          const auto& avg_kinetic_s_2  = 0.5*(0.5*scalar_product(u_s_2_m, u_s_2_m)*rho_s_2_m*u_s_2_m +
                                              0.5*scalar_product(u_s_2_p, u_s_2_p)*rho_s_2_p*u_s_2_p);

          const auto& pres_s_2_m       = phi_pres_s_2_m.get_value(q);
          const auto& pres_s_2_p       = phi_pres_s_2_p.get_value(q);
          const auto& avg_enthalpy_s_2 = 0.5*EquationData::Cp_Cv/(EquationData::Cp_Cv - 1.0)*
                                         (pres_s_2_m*u_s_2_m + pres_s_2_p*u_s_2_p);

          const auto& lambda_s_2       = compute_lambda(u_s_2_m, u_s_2_p, n_minus);
          const auto& jump_rho_kin_s_2 = rho_s_2_m*(0.5*scalar_product(u_s_2_m, u_s_2_m)) -
                                         rho_s_2_p*(0.5*scalar_product(u_s_2_p, u_s_2_p));
          const auto& jump_rho_e_s_2   = 1.0/(EquationData::Cp_Cv - 1.0)*(pres_s_2_m - pres_s_2_p);

          /*--- Compute the quantities at the current stage ---*/
          const auto& u_fixed_m        = phi_u_fixed_m.get_value(q);
          const auto& u_fixed_p        = phi_u_fixed_p.get_value(q);
          const auto& pres_fixed_m     = phi_pres_fixed_m.get_value(q);
          const auto& pres_fixed_p     = phi_pres_fixed_p.get_value(q);

          const auto& lambda_fixed     = compute_lambda(u_fixed_m, u_fixed_p, n_minus);
          const auto& jump_rho_e_fixed = 1.0/(EquationData::Cp_Cv - 1.0)*(pres_fixed_m - pres_fixed_p);

          phi_m.submit_value(-a31*dt*(Ma*Ma*(scalar_product(avg_kinetic_old, n_minus) + 0.5*lambda_old*jump_rho_kin_old))
                             -a31_tilde*dt*(scalar_product(avg_enthalpy_old, n_minus) + 0.5*lambda_old*jump_rho_e_old)
                             -a32*dt*(Ma*Ma*(scalar_product(avg_kinetic_s_2, n_minus) + 0.5*lambda_s_2*jump_rho_kin_s_2))
                             -a32_tilde*dt*(scalar_product(avg_enthalpy_s_2, n_minus) + 0.5*lambda_s_2*jump_rho_e_s_2)
                             -a33_tilde*dt*(0.5*lambda_fixed*jump_rho_e_fixed), q);
          phi_p.submit_value(a31*dt*(Ma*Ma*(scalar_product(avg_kinetic_old, n_minus) + 0.5*lambda_old*jump_rho_kin_old)) +
                             a31_tilde*dt*(scalar_product(avg_enthalpy_old, n_minus) + 0.5*lambda_old*jump_rho_e_old) +
                             a32*dt*(Ma*Ma*(scalar_product(avg_kinetic_s_2, n_minus) + 0.5*lambda_s_2*jump_rho_kin_s_2)) +
                             a32_tilde*dt*(scalar_product(avg_enthalpy_s_2, n_minus) + 0.5*lambda_s_2*jump_rho_e_s_2) +
                             a33_tilde*dt*(0.5*lambda_fixed*jump_rho_e_fixed), q);
        }

        phi_m.integrate_scatter(EvaluationFlags::values, dst);
        phi_p.integrate_scatter(EvaluationFlags::values, dst);
      }
    }
    else {
      /*--- We first start by declaring the suitable instances to read the available quantities. ---*/
      FEFaceEvaluation<dim, fe_degree_p, n_q_points_1d, 1, Number>   phi_m(data, true, EquationData::P_INDEX_DOF),
                                                                     phi_p(data, false, EquationData::P_INDEX_DOF),
                                                                     phi_pres_old_m(data, true, EquationData::P_INDEX_DOF),
                                                                     phi_pres_old_p(data, false, EquationData::P_INDEX_DOF),
                                                                     phi_pres_s_2_m(data, true, EquationData::P_INDEX_DOF),
                                                                     phi_pres_s_2_p(data, false, EquationData::P_INDEX_DOF),
                                                                     phi_pres_s_3_m(data, true, EquationData::P_INDEX_DOF),
                                                                     phi_pres_s_3_p(data, false, EquationData::P_INDEX_DOF);
      FEFaceEvaluation<dim, fe_degree_u, n_q_points_1d, dim, Number> phi_u_old_m(data, true, EquationData::U_INDEX_DOF),
                                                                     phi_u_old_p(data, false, EquationData::U_INDEX_DOF),
                                                                     phi_u_s_2_m(data, true, EquationData::U_INDEX_DOF),
                                                                     phi_u_s_2_p(data, false, EquationData::U_INDEX_DOF),
                                                                     phi_u_s_3_m(data, true, EquationData::U_INDEX_DOF),
                                                                     phi_u_s_3_p(data, false, EquationData::U_INDEX_DOF);
      FEFaceEvaluation<dim, fe_degree_rho, n_q_points_1d, 1, Number> phi_rho_old_m(data, true, EquationData::RHO_INDEX_DOF),
                                                                     phi_rho_old_p(data, false, EquationData::RHO_INDEX_DOF),
                                                                     phi_rho_s_2_m(data, true, EquationData::RHO_INDEX_DOF),
                                                                     phi_rho_s_2_p(data, false, EquationData::RHO_INDEX_DOF),
                                                                     phi_rho_s_3_m(data, true, EquationData::RHO_INDEX_DOF),
                                                                     phi_rho_s_3_p(data, false, EquationData::RHO_INDEX_DOF);

      /*--- loop over all internal faces ---*/
      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
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
        for(unsigned int q = 0; q < phi_m.n_q_points; ++q) {
          const auto& n_minus          = phi_m.get_normal_vector(q);

          /*--- Compute the quantities at the previous step ---*/
          const auto& rho_old_m        = phi_rho_old_m.get_value(q);
          const auto& rho_old_p        = phi_rho_old_p.get_value(q);
          const auto& u_old_m          = phi_u_old_m.get_value(q);
          const auto& u_old_p          = phi_u_old_p.get_value(q);
          const auto& avg_kinetic_old  = 0.5*(0.5*scalar_product(u_old_m, u_old_m)*rho_old_m*u_old_m +
                                              0.5*scalar_product(u_old_p, u_old_p)*rho_old_p*u_old_p);

          const auto& pres_old_m       = phi_pres_old_m.get_value(q);
          const auto& pres_old_p       = phi_pres_old_p.get_value(q);
          const auto& avg_enthalpy_old = 0.5*EquationData::Cp_Cv/(EquationData::Cp_Cv - 1.0)*
                                         (pres_old_m*u_old_m + pres_old_p*u_old_p);

          const auto& lambda_old       = compute_lambda(u_old_m, u_old_p, n_minus);
          const auto& rhoE_old_m       = 1.0/(EquationData::Cp_Cv - 1.0)*pres_old_m
                                       + rho_old_m*(0.5*Ma*Ma*scalar_product(u_old_m, u_old_m));
          const auto& rhoE_old_p       = 1.0/(EquationData::Cp_Cv - 1.0)*pres_old_p
                                       + rho_old_p*(0.5*Ma*Ma*scalar_product(u_old_p, u_old_p));
          const auto& jump_rhoE_old    = rhoE_old_m - rhoE_old_p;

          /*--- Compute the quantities at the second stage ---*/
          const auto& rho_s_2_m        = phi_rho_s_2_m.get_value(q);
          const auto& rho_s_2_p        = phi_rho_s_2_p.get_value(q);
          const auto& u_s_2_m          = phi_u_s_2_m.get_value(q);
          const auto& u_s_2_p          = phi_u_s_2_p.get_value(q);
          const auto& avg_kinetic_s_2  = 0.5*(0.5*scalar_product(u_s_2_m, u_s_2_m)*rho_s_2_m*u_s_2_m +
                                                0.5*scalar_product(u_s_2_p, u_s_2_p)*rho_s_2_p*u_s_2_p);

          const auto& pres_s_2_m       = phi_pres_s_2_m.get_value(q);
          const auto& pres_s_2_p       = phi_pres_s_2_p.get_value(q);
          const auto& avg_enthalpy_s_2 = 0.5*EquationData::Cp_Cv/(EquationData::Cp_Cv - 1.0)*
                                           (pres_s_2_m*u_s_2_m + pres_s_2_p*u_s_2_p);

          const auto& lambda_s_2       = compute_lambda(u_s_2_m, u_s_2_p, n_minus);
          const auto& rhoE_s_2_m       = 1.0/(EquationData::Cp_Cv - 1.0)*pres_s_2_m
                                       + rho_s_2_m*(0.5*Ma*Ma*scalar_product(u_s_2_m, u_s_2_m));
          const auto& rhoE_s_2_p       = 1.0/(EquationData::Cp_Cv - 1.0)*pres_s_2_p
                                       + rho_s_2_p*(0.5*Ma*Ma*scalar_product(u_s_2_p, u_s_2_p));
          const auto& jump_rhoE_s_2    = rhoE_s_2_m - rhoE_s_2_p;

          /*--- Compute the quantities at the final stage ---*/
          const auto& rho_s_3_m        = phi_rho_s_3_m.get_value(q);
          const auto& rho_s_3_p        = phi_rho_s_3_p.get_value(q);
          const auto& u_s_3_m          = phi_u_s_3_m.get_value(q);
          const auto& u_s_3_p          = phi_u_s_3_p.get_value(q);
          const auto& avg_kinetic_s_3  = 0.5*(0.5*scalar_product(u_s_3_m, u_s_3_m)*rho_s_3_m*u_s_3_m +
                                              0.5*scalar_product(u_s_3_p, u_s_3_p)*rho_s_3_p*u_s_3_p);

          const auto& pres_s_3_m       = phi_pres_s_3_m.get_value(q);
          const auto& pres_s_3_p       = phi_pres_s_3_p.get_value(q);
          const auto& avg_enthalpy_s_3 = 0.5*EquationData::Cp_Cv/(EquationData::Cp_Cv - 1.0)*
                                         (pres_s_3_m*u_s_3_m + pres_s_3_p*u_s_3_p);

          const auto& lambda_s_3       = compute_lambda(u_s_3_m, u_s_3_p, n_minus);
          const auto& rhoE_s_3_m       = 1.0/(EquationData::Cp_Cv - 1.0)*pres_s_3_m
                                       + rho_s_3_m*(0.5*Ma*Ma*scalar_product(u_s_3_m, u_s_3_m));
          const auto& rhoE_s_3_p       = 1.0/(EquationData::Cp_Cv - 1.0)*pres_s_3_p
                                       + rho_s_3_p*(0.5*Ma*Ma*scalar_product(u_s_3_p, u_s_3_p));
          const auto& jump_rhoE_s_3    = rhoE_s_3_m - rhoE_s_3_p;

          phi_m.submit_value(-b1*dt*(Ma*Ma*scalar_product(avg_kinetic_old, n_minus))
                             -b1*dt*scalar_product(avg_enthalpy_old, n_minus)
                             -b1*dt*(0.5*lambda_old*jump_rhoE_old)
                             -b2*dt*(Ma*Ma*scalar_product(avg_kinetic_s_2, n_minus))
                             -b2*dt*scalar_product(avg_enthalpy_s_2, n_minus)
                             -b2*dt*(0.5*lambda_s_2*jump_rhoE_s_2)
                             -b3*dt*(Ma*Ma*scalar_product(avg_kinetic_s_3, n_minus))
                             -b3*dt*scalar_product(avg_enthalpy_s_3, n_minus)
                             -b3*dt*(0.5*lambda_s_3*jump_rhoE_s_3), q);
          phi_p.submit_value(b1*dt*(Ma*Ma*scalar_product(avg_kinetic_old, n_minus)) +
                             b1*dt*scalar_product(avg_enthalpy_old, n_minus) +
                             b1*dt*(0.5*lambda_old*jump_rhoE_old) +
                             b2*dt*(Ma*Ma*scalar_product(avg_kinetic_s_2, n_minus)) +
                             b2*dt*scalar_product(avg_enthalpy_s_2, n_minus) +
                             b2*dt*(0.5*lambda_s_2*jump_rhoE_s_2) +
                             b3*dt*(Ma*Ma*scalar_product(avg_kinetic_s_3, n_minus)) +
                             b3*dt*scalar_product(avg_enthalpy_s_3, n_minus) +
                             b3*dt*(0.5*lambda_s_3*jump_rhoE_s_3), q);
        }

        phi_m.integrate_scatter(EvaluationFlags::values, dst);
        phi_p.integrate_scatter(EvaluationFlags::values, dst);
      }
    }
  }

  // Put together all the previous steps for the energy equation
  //
  template<int dim, int fe_degree_u, int fe_degree_rho, int fe_degree_p,
           int n_q_points_1d, int n_q_points_1d_boundary, typename Vec>
  void EULEROperator<dim, fe_degree_u, fe_degree_rho, fe_degree_p,
                     n_q_points_1d, n_q_points_1d_boundary, Vec>::
  vmult_rhs_energy(Vec& dst, const std::vector<Vec>& src) const {
    for(unsigned int d = 0; d < src.size(); ++d) {
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
  template<int dim, int fe_degree_u, int fe_degree_rho, int fe_degree_p,
           int n_q_points_1d, int n_q_points_1d_boundary, typename Vec>
  void EULEROperator<dim, fe_degree_u, fe_degree_rho, fe_degree_p,
                     n_q_points_1d, n_q_points_1d_boundary, Vec>::
  assemble_cell_term_internal_energy(const MatrixFree<dim, Number>&               data,
                                     Vec&                                         dst,
                                     const Vec&                                   src,
                                     const std::pair<unsigned int, unsigned int>& cell_range) const {
    FEEvaluation<dim, fe_degree_p, n_q_points_1d, 1, Number> phi(data, EquationData::P_INDEX_DOF);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
      phi.reinit(cell);
      phi.gather_evaluate(src, EvaluationFlags::values);

      for(unsigned int q = 0; q < phi.n_q_points; ++q) {
        /*--- For an ideal gas the part associated to the internal energy for a pressure based
              is just a modification of the mass matrix ---*/
        phi.submit_value(1.0/(EquationData::Cp_Cv - 1.0)*phi.get_value(q), q);
      }

      phi.integrate_scatter(EvaluationFlags::values, dst);
    }
  }

  // Assemble cell term for the contribution due to enthalpy
  //
  template<int dim, int fe_degree_u, int fe_degree_rho, int fe_degree_p,
           int n_q_points_1d, int n_q_points_1d_boundary, typename Vec>
  void EULEROperator<dim, fe_degree_u, fe_degree_rho, fe_degree_p,
                     n_q_points_1d, n_q_points_1d_boundary, Vec>::
  assemble_cell_term_enthalpy(const MatrixFree<dim, Number>&               data,
                              Vec&                                         dst,
                              const Vec&                                   src,
                              const std::pair<unsigned int, unsigned int>& cell_range) const {
    /*--- We first start by declaring the suitable instances to read also available quantities.
          Since here we have just one 'src' vector, but we also need to deal with the current pressure
          in the fixed point loop, we employ the auxiliary vector 'pres_fixed' where we setted this information ---*/
    FEEvaluation<dim, fe_degree_p, n_q_points_1d, 1, Number>   phi(data, EquationData::P_INDEX_DOF),
                                                               phi_pres_fixed(data, EquationData::P_INDEX_DOF);
    FEEvaluation<dim, fe_degree_u, n_q_points_1d, dim, Number> phi_src(data, EquationData::U_INDEX_DOF);

    const double coeff = (IMEX_stage == 2) ? a22_tilde : a33_tilde;

    /*--- Loop over all cells ---*/
    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
      phi_pres_fixed.reinit(cell);
      phi_pres_fixed.gather_evaluate(pres_fixed, EvaluationFlags::values);

      phi_src.reinit(cell);
      phi_src.gather_evaluate(src, EvaluationFlags::values);

      phi.reinit(cell);

      /*--- loop over all quadrature points ---*/
      for(unsigned int q = 0; q < phi.n_q_points; ++q) {
        const auto& pres_fixed = phi_pres_fixed.get_value(q);

        phi.submit_gradient(-coeff*dt*(EquationData::Cp_Cv/(EquationData::Cp_Cv - 1.0)*pres_fixed*phi_src.get_value(q)), q);
      }

      phi.integrate_scatter(EvaluationFlags::gradients, dst);
    }
  }

  // Assemble face term for the contribution due to enthalpy
  //
  template<int dim, int fe_degree_u, int fe_degree_rho, int fe_degree_p,
           int n_q_points_1d, int n_q_points_1d_boundary, typename Vec>
  void EULEROperator<dim, fe_degree_u, fe_degree_rho, fe_degree_p,
                     n_q_points_1d, n_q_points_1d_boundary, Vec>::
  assemble_face_term_enthalpy(const MatrixFree<dim, Number>&               data,
                              Vec&                                         dst,
                              const Vec&                                   src,
                              const std::pair<unsigned int, unsigned int>& face_range) const {
    FEFaceEvaluation<dim, fe_degree_p, n_q_points_1d, 1, Number>   phi_m(data, true, EquationData::P_INDEX_DOF),
                                                                   phi_p(data, false, EquationData::P_INDEX_DOF),
                                                                   phi_pres_fixed_m(data, true, EquationData::P_INDEX_DOF),
                                                                   phi_pres_fixed_p(data, false, EquationData::P_INDEX_DOF);
    FEFaceEvaluation<dim, fe_degree_u, n_q_points_1d, dim, Number> phi_src_m(data, true, EquationData::U_INDEX_DOF),
                                                                   phi_src_p(data, false, EquationData::U_INDEX_DOF);

    /*--- This term changes between second and third stage of the IMEX scheme, but its structure not, so we do not need
          to explicitly distinguish the two cases as done for the rhs. ---*/
    const double coeff = (IMEX_stage == 2) ? a22_tilde : a33_tilde;

    /*--- Loop over all faces ---*/
    for(unsigned int face = face_range.first; face < face_range.second; ++face) {
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
      for(unsigned int q = 0; q < phi_m.n_q_points; ++q) {
        const auto& n_minus           = phi_m.get_normal_vector(q);

        const auto& pres_fixed_m      = phi_pres_fixed_m.get_value(q);
        const auto& pres_fixed_p      = phi_pres_fixed_p.get_value(q);

        const auto& avg_flux_enthalpy = 0.5*EquationData::Cp_Cv/(EquationData::Cp_Cv - 1.0)*
                                        (pres_fixed_m*phi_src_m.get_value(q) + pres_fixed_p*phi_src_p.get_value(q));

        phi_m.submit_value(coeff*dt*scalar_product(avg_flux_enthalpy, n_minus), q);
        phi_p.submit_value(-coeff*dt*scalar_product(avg_flux_enthalpy, n_minus), q);
      }

      phi_m.integrate_scatter(EvaluationFlags::values, dst);
      phi_p.integrate_scatter(EvaluationFlags::values, dst);
    }
  }


  // Put together all previous steps
  //
  template<int dim, int fe_degree_u, int fe_degree_rho, int fe_degree_p,
           int n_q_points_1d, int n_q_points_1d_boundary, typename Vec>
  void EULEROperator<dim, fe_degree_u, fe_degree_rho, fe_degree_p,
                     n_q_points_1d, n_q_points_1d_boundary, Vec>::
  apply_add(Vec& dst, const Vec& src) const {
    AssertIndexRange(Euler_stage, EquationData::n_vars + 1);
    Assert(Euler_stage > 0, ExcInternalError());

    if(Euler_stage == EquationData::RHO_INDEX_SYSTEM) {
      this->data->cell_loop(&EULEROperator::assemble_cell_term_density,
                            this, dst, src, false);
    }
    else if(Euler_stage == EquationData::P_INDEX_SYSTEM) {
      this->data->cell_loop(&EULEROperator::assemble_cell_term_internal_energy,
                            this, dst, src, false);

      if(IMEX_stage <= EquationData::n_stages) {
        /*--- Implementation of the Schur complement operations ---*/
        Vec tmp_1;
        this->data->initialize_dof_vector(tmp_1, EquationData::U_INDEX_DOF);
        this->vmult_pressure(tmp_1, src);

        Euler_stage = EquationData::U_INDEX_SYSTEM;
        const std::vector<unsigned int> index_dof_handler_reinit = {EquationData::U_INDEX_DOF};
        auto* tmp_matrix = const_cast<EULEROperator*>(this);
        Vec tmp_2;
        this->data->initialize_dof_vector(tmp_2, EquationData::U_INDEX_DOF);
        tmp_2 = 0;
        tmp_matrix->initialize(tmp_matrix->get_matrix_free(), index_dof_handler_reinit, index_dof_handler_reinit);

        SolverControl solver_control(10000, 1e-12*tmp_1.l2_norm());
        SolverCG<Vec> cg(solver_control);
        PreconditionJacobi<EULEROperator> preconditioner_Jacobi;
        tmp_matrix->compute_diagonal();
        preconditioner_Jacobi.initialize(*tmp_matrix);

        cg.solve(*tmp_matrix, tmp_2, tmp_1, preconditioner_Jacobi);

        Vec tmp_3;
        this->data->initialize_dof_vector(tmp_3, EquationData::P_INDEX_DOF);
        this->vmult_enthalpy(tmp_3, tmp_2);

        dst.add(-1.0, tmp_3);
        Euler_stage = EquationData::P_INDEX_SYSTEM;
        const std::vector<unsigned int> index_dof_handler = {EquationData::P_INDEX_DOF};
        tmp_matrix->initialize(tmp_matrix->get_matrix_free(), index_dof_handler, index_dof_handler);
        tmp_matrix->compute_diagonal();
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
  template<int dim, int fe_degree_u, int fe_degree_rho, int fe_degree_p,
           int n_q_points_1d, int n_q_points_1d_boundary, typename Vec>
  void EULEROperator<dim, fe_degree_u, fe_degree_rho, fe_degree_p,
                     n_q_points_1d, n_q_points_1d_boundary, Vec>::
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
  template<int dim, int fe_degree_u, int fe_degree_rho, int fe_degree_p,
           int n_q_points_1d, int n_q_points_1d_boundary, typename Vec>
  void EULEROperator<dim, fe_degree_u, fe_degree_rho, fe_degree_p,
                     n_q_points_1d, n_q_points_1d_boundary, Vec>::
  vmult_enthalpy(Vec& dst, const Vec& src) const {
    src.update_ghost_values();

    this->data->loop(&EULEROperator::assemble_cell_term_enthalpy,
                     &EULEROperator::assemble_face_term_enthalpy,
                     &EULEROperator::assemble_boundary_term_enthalpy,
                     this, dst, src, true,
                     MatrixFree<dim, Number>::DataAccessOnFaces::values,
                     MatrixFree<dim, Number>::DataAccessOnFaces::values);
  }


  // Assemble diagonal cell term for the density update
  //
  template<int dim, int fe_degree_u, int fe_degree_rho, int fe_degree_p,
           int n_q_points_1d, int n_q_points_1d_boundary, typename Vec>
  void EULEROperator<dim, fe_degree_u, fe_degree_rho, fe_degree_p,
                     n_q_points_1d, n_q_points_1d_boundary, Vec>::
  assemble_diagonal_cell_term_density(const MatrixFree<dim, Number>&               data,
                                      Vec&                                         dst,
                                      const unsigned int&                          ,
                                      const std::pair<unsigned int, unsigned int>& cell_range) const {
    FEEvaluation<dim, fe_degree_rho, n_q_points_1d, 1, Number> phi(data, EquationData::RHO_INDEX_DOF);

    AlignedVector<VectorizedArray<Number>> diagonal(phi.dofs_per_component);

    /*--- Loop over all cells ---*/
    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
      phi.reinit(cell);

      /*--- Loop over all dofs ---*/
      for(unsigned int i = 0; i < phi.dofs_per_component; ++i) {
        for(unsigned int j = 0; j < phi.dofs_per_component; ++j) {
          phi.submit_dof_value(VectorizedArray<Number>(), j);
        }
        phi.submit_dof_value(make_vectorized_array<Number>(1.0), i);
        /*--- We are in a matrix-free framework. Hence, in order to compute the diagonal, we need to test the operator against
              a vector which is 1 for the node of interest and 0 elsewhere.---*/
        phi.evaluate(EvaluationFlags::values);

        /*--- Loop over all quadrature points ---*/
        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          phi.submit_value(phi.get_value(q), q);
        }

        phi.integrate(EvaluationFlags::values);
        diagonal[i] = phi.get_dof_value(i);
      }

      for(unsigned int i = 0; i < phi.dofs_per_component; ++i) {
        phi.submit_dof_value(diagonal[i], i);
      }
      phi.distribute_local_to_global(dst);
    }
  }


  // Assemble diagonal cell term for the velocity update
  //
  template<int dim, int fe_degree_u, int fe_degree_rho, int fe_degree_p,
           int n_q_points_1d, int n_q_points_1d_boundary, typename Vec>
  void EULEROperator<dim, fe_degree_u, fe_degree_rho, fe_degree_p,
                     n_q_points_1d, n_q_points_1d_boundary, Vec>::
  assemble_diagonal_cell_term_velocity(const MatrixFree<dim, Number>&               data,
                                       Vec&                                         dst,
                                       const unsigned int&                          ,
                                       const std::pair<unsigned int, unsigned int>& cell_range) const {
    FEEvaluation<dim, fe_degree_u, n_q_points_1d, dim, Number> phi(data, EquationData::U_INDEX_DOF);
    FEEvaluation<dim, fe_degree_rho, n_q_points_1d, 1, Number> phi_rho_for_fixed(data, EquationData::RHO_INDEX_DOF);

    /*--- We are in a matrix-free framework. Hence, in order to compute the diagonal, we need to test the operator against
          a vector which is 1 for the node of interest and 0 elsewhere. This is what 'tmp' does.
          Moreover, since here we have just one 'src' vector, but we also need to deal with the current density,
          we employ the auxiliary vector 'rho_for_fixed' where we setted this information ---*/
    AlignedVector<Tensor<1, dim, VectorizedArray<Number>>> diagonal(phi.dofs_per_component);
    Tensor<1, dim, VectorizedArray<Number>> tmp;
    for(unsigned int d = 0; d < dim; ++d) {
      tmp[d] = make_vectorized_array<Number>(1.0);
    }

    /*--- Loop over all cells ---*/
    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
      phi_rho_for_fixed.reinit(cell);
      phi_rho_for_fixed.gather_evaluate(rho_for_fixed, EvaluationFlags::values);

      phi.reinit(cell);

      /*--- Loop over all dofs ---*/
      for(unsigned int i = 0; i < phi.dofs_per_component; ++i) {
        for(unsigned int j = 0; j < phi.dofs_per_component; ++j) {
          phi.submit_dof_value(Tensor<1, dim, VectorizedArray<Number>>(), j);
        }
        phi.submit_dof_value(tmp, i);
        phi.evaluate(EvaluationFlags::values);

        /*--- Loop over all quadrature points ---*/
        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          phi.submit_value(phi_rho_for_fixed.get_value(q)*phi.get_value(q), q);
        }

        phi.integrate(EvaluationFlags::values);
        diagonal[i] = phi.get_dof_value(i);
      }

      for(unsigned int i = 0; i < phi.dofs_per_component; ++i) {
        phi.submit_dof_value(diagonal[i], i);
      }
      phi.distribute_local_to_global(dst);
    }
  }


  // Assemble diagonal cell term for the pressure updated with Schur complement
  //
  template<int dim, int fe_degree_u, int fe_degree_rho, int fe_degree_p,
           int n_q_points_1d, int n_q_points_1d_boundary, typename Vec>
  void EULEROperator<dim, fe_degree_u, fe_degree_rho, fe_degree_p,
                     n_q_points_1d, n_q_points_1d_boundary, Vec>::
  assemble_diagonal_cell_term_pressure(const MatrixFree<dim, Number>&               data,
                                       Vec&                                         dst,
                                       const unsigned int&                          ,
                                       const std::pair<unsigned int, unsigned int>& cell_range) const {
    FEEvaluation<dim, fe_degree_p, n_q_points_1d, 1, Number>   phi(data, EquationData::P_INDEX_DOF),
                                                               phi_pres_fixed(data, EquationData::P_INDEX_DOF);
    FEEvaluation<dim, fe_degree_rho, n_q_points_1d, 1, Number> phi_rho_for_fixed(data, EquationData::RHO_INDEX_DOF);

    AlignedVector<VectorizedArray<Number>> diagonal(phi.dofs_per_component);

    /*--- This term changes between second and third stage of the IMEX scheme, but its structure not, so we do not need
          to explicitly distinguish the two cases as done for the rhs. ---*/
    const double coeff = (IMEX_stage == 2) ? a22_tilde : a33_tilde;

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
      phi_pres_fixed.reinit(cell);
      phi_pres_fixed.gather_evaluate(pres_fixed, EvaluationFlags::values);

      phi_rho_for_fixed.reinit(cell);
      phi_rho_for_fixed.gather_evaluate(rho_for_fixed, EvaluationFlags::values);

      phi.reinit(cell);

      /*--- Loop over all dofs ---*/
      for(unsigned int i = 0; i < phi.dofs_per_component; ++i) {
        for(unsigned int j = 0; j < phi.dofs_per_component; ++j) {
          phi.submit_dof_value(VectorizedArray<Number>(), j);
        }
        phi.submit_dof_value(make_vectorized_array<Number>(1.0), i);
        /*--- We are in a matrix-free framework. Hence, in order to compute the diagonal, we need to test the operator against
              a vector which is 1 for the node of interest and 0 elsewhere.---*/
        phi.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);

        /*--- Loop over all quadrature points ---*/
        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          const auto& pres_fixed    = phi_pres_fixed.get_value(q);

          const auto& rho_for_fixed = phi_rho_for_fixed.get_value(q);

          phi.submit_value(1.0/(EquationData::Cp_Cv - 1.0)*phi.get_value(q), q);
          phi.submit_gradient((coeff*dt/Ma)*(coeff*dt/Ma)*
                              (EquationData::Cp_Cv/(EquationData::Cp_Cv - 1.0)*(pres_fixed/rho_for_fixed)*phi.get_gradient(q)), q);
        }

        phi.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
        diagonal[i] = phi.get_dof_value(i);
      }

      for(unsigned int i = 0; i < phi.dofs_per_component; ++i) {
        phi.submit_dof_value(diagonal[i], i);
      }
      phi.distribute_local_to_global(dst);
    }
  }


  // Assemble diagonal cell term for the contribution due to internal energy
  //
  template<int dim, int fe_degree_u, int fe_degree_rho, int fe_degree_p,
           int n_q_points_1d, int n_q_points_1d_boundary, typename Vec>
  void EULEROperator<dim, fe_degree_u, fe_degree_rho, fe_degree_p,
                     n_q_points_1d, n_q_points_1d_boundary, Vec>::
  assemble_diagonal_cell_term_internal_energy(const MatrixFree<dim, Number>&               data,
                                              Vec&                                         dst,
                                              const unsigned int&                          ,
                                              const std::pair<unsigned int, unsigned int>& cell_range) const {
    FEEvaluation<dim, fe_degree_p, n_q_points_1d, 1, Number> phi(data, EquationData::P_INDEX_DOF);

    AlignedVector<VectorizedArray<Number>> diagonal(phi.dofs_per_component);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
      phi.reinit(cell);

      /*--- Loop over all dofs ---*/
      for(unsigned int i = 0; i < phi.dofs_per_component; ++i) {
        for(unsigned int j = 0; j < phi.dofs_per_component; ++j) {
          phi.submit_dof_value(VectorizedArray<Number>(), j);
        }
        phi.submit_dof_value(make_vectorized_array<Number>(1.0), i);
        /*--- We are in a matrix-free framework. Hence, in order to compute the diagonal, we need to test the operator against
              a vector which is 1 for the node of interest and 0 elsewhere.---*/
        phi.evaluate(EvaluationFlags::values);

        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          phi.submit_value(1.0/(EquationData::Cp_Cv - 1.0)*phi.get_value(q), q);
        }

        phi.integrate(EvaluationFlags::values);
        diagonal[i] = phi.get_dof_value(i);
      }

      for(unsigned int i = 0; i < phi.dofs_per_component; ++i) {
        phi.submit_dof_value(diagonal[i], i);
      }
      phi.distribute_local_to_global(dst);
    }
  }


  // Compute diagonal of various steps
  //
  template<int dim, int fe_degree_u, int fe_degree_rho, int fe_degree_p,
           int n_q_points_1d, int n_q_points_1d_boundary, typename Vec>
  void EULEROperator<dim, fe_degree_u, fe_degree_rho, fe_degree_p,
                     n_q_points_1d, n_q_points_1d_boundary, Vec>::
  compute_diagonal() {
    AssertIndexRange(Euler_stage, EquationData::n_vars + 1);
    Assert(Euler_stage > 0, ExcInternalError());

    this->inverse_diagonal_entries.reset(new DiagonalMatrix<Vec>());
    auto& inverse_diagonal = this->inverse_diagonal_entries->get_vector();

    const unsigned int dummy = 0;

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
    for(unsigned int i = 0; i < inverse_diagonal.locally_owned_size(); ++i) {
      Assert(inverse_diagonal.local_element(i) != 0.0,
             ExcMessage("No diagonal entry in a definite operator should be zero"));
      inverse_diagonal.local_element(i) = 1.0/inverse_diagonal.local_element(i);
    }
  }
  
}
