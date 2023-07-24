/* Author: Giuseppe Orlando, 2023. */

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
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  class EULEROperator: public MatrixFreeOperators::Base<dim, Vec> {
  public:
    EULEROperator(); /*--- Default constructor ---*/

    EULEROperator(RunTimeParameters::Data_Storage& data); /*--- Constructor with some input related data ---*/

    void set_dt(const double time_step); /*--- Setter of the time-step. This is useful both for multigrid purposes and also
                                               in case of modifications of the time step. ---*/

    void set_Mach(const double Ma_); /*--- Setter of the Mach number. This is useful for multigrid purpose. ---*/

    void set_Froude(const double Fr_); /*--- Setter of the Froude number. This is useful for multigrid purpose. ---*/

    void set_HYPERBOLIC_stage(const unsigned int stage); /*--- Setter of the IMEX stage. ---*/

    void set_NS_stage(const unsigned int stage); /*--- Setter of the equation currently under solution. ---*/

    void set_rho_for_fixed(const Vec& src); /*--- Setter of the current density. This is for the assembling of the bilinear forms
                                                  where only one source vector can be passed in input. ---*/

    void set_pres_fixed(const Vec& src); /*--- Setter of the current pressure. This is for the assembling of the bilinear forms
                                               where only one source vector can be passed in input. ---*/

    void set_u_fixed(const Vec& src); /*--- Setter of the current velocity. This is for the assembling of the bilinear forms
                                            where only one source vector can be passed in input. ---*/

    void vmult_rhs_rho_update(Vec& dst, const std::vector<Vec>& src) const; /*--- Auxiliary function to assemble the rhs
                                                                                  for the density. ---*/

    void vmult_rhs_pressure(Vec& dst, const std::vector<Vec>& src) const;  /*--- Auxiliary function to assemble the rhs
                                                                                 for the pressure. ---*/

    void vmult_rhs_velocity_fixed(Vec& dst, const std::vector<Vec>& src) const;  /*--- Auxiliary function to assemble the rhs
                                                                                       for the velocity. ---*/

    void vmult_pressure(Vec& dst, const Vec& src) const; /*--- Action of matrix 'B'. ---*/

    void vmult_enthalpy(Vec& dst, const Vec& src) const; /*--- Action of matrix 'C'. ---*/

    virtual void compute_diagonal() override; /*--- Overriden function to compute the diagonal. ---*/

  protected:
    double       Ma;  /*--- Mach number. ---*/
    double       Fr;  /*--- Froude number. ---*/
    double       dt;  /*--- Time step. ---*/

    double       gamma; /*--- TR-BDF2 (i.e. implicit part) parameter. ---*/
    /*--- The following variables follow the classical Butcher tableaux notation ---*/
    double       a21;
    double       a22;
    double       a31;
    double       a32;
    double       a33;
    double       a21_tilde;
    double       a22_tilde;
    double       a31_tilde;
    double       a32_tilde;
    double       a33_tilde;
    double       b1;
    double       b2;
    double       b3;

    unsigned int HYPERBOLIC_stage; /*--- Flag for the IMEX stage ---*/
    mutable unsigned int NS_stage; /*--- Flag for the IMEX stage ---*/

    virtual void apply_add(Vec& dst, const Vec& src) const override; /*--- Overriden function which actually assembles the
                                                                           bilinear forms ---*/

  private:
    Vec rho_for_fixed,
        pres_fixed,
        u_fixed;

    EquationData::Density<dim>  rho_exact;
    EquationData::Velocity<dim> u_exact;
    EquationData::Pressure<dim> pres_exact;

    /*--- Assembler functions for the rhs related to the continuity equation. Here, and also in the following,
          we distinguish between the contribution for cells, faces and boundary. ---*/
    void assemble_rhs_cell_term_rho_update(const MatrixFree<dim, Number>&               data,
                                           Vec&                                         dst,
                                           const std::vector<Vec>&                      src,
                                           const std::pair<unsigned int, unsigned int>& cell_range) const;
    void assemble_rhs_face_term_rho_update(const MatrixFree<dim, Number>&               data,
                                           Vec&                                         dst,
                                           const std::vector<Vec>&                      src,
                                           const std::pair<unsigned int, unsigned int>& face_range) const;
    void assemble_rhs_boundary_term_rho_update(const MatrixFree<dim, Number>&               data,
                                               Vec&                                         dst,
                                               const std::vector<Vec>&                      src,
                                               const std::pair<unsigned int, unsigned int>& face_range) const;

    /*--- Assembler function related to the bilinear form of the continuity equation. Only cell contribution is present,
          since, basically, we end up with a mass matrix. ---*/
    void assemble_cell_term_rho_update(const MatrixFree<dim, Number>&               data,
                                       Vec&                                         dst,
                                       const Vec&                                   src,
                                       const std::pair<unsigned int, unsigned int>& cell_range) const;

    /*--- Assembler functions for the rhs related to the velocity equation. ---*/
    void assemble_rhs_cell_term_velocity_fixed(const MatrixFree<dim, Number>&               data,
                                               Vec&                                         dst,
                                               const std::vector<Vec>&                      src,
                                               const std::pair<unsigned int, unsigned int>& cell_range) const;
    void assemble_rhs_face_term_velocity_fixed(const MatrixFree<dim, Number>&               data,
                                               Vec&                                         dst,
                                               const std::vector<Vec>&                      src,
                                               const std::pair<unsigned int, unsigned int>& face_range) const;
    void assemble_rhs_boundary_term_velocity_fixed(const MatrixFree<dim, Number>&               data,
                                                   Vec&                                         dst,
                                                   const std::vector<Vec>&                      src,
                                                   const std::pair<unsigned int, unsigned int>& face_range) const;

    /*--- Assembler function for the 'A' matrix. ---*/
    void assemble_cell_term_velocity_fixed(const MatrixFree<dim, Number>&               data,
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

    /*--- Assembler functions for the rhs of the pressure equation. ---*/
    void assemble_rhs_cell_term_pressure(const MatrixFree<dim, Number>&               data,
                                         Vec&                                         dst,
                                         const std::vector<Vec>&                      src,
                                         const std::pair<unsigned int, unsigned int>& cell_range) const;
    void assemble_rhs_face_term_pressure(const MatrixFree<dim, Number>&               data,
                                         Vec&                                         dst,
                                         const std::vector<Vec>&                      src,
                                         const std::pair<unsigned int, unsigned int>& face_range) const;
    void assemble_rhs_boundary_term_pressure(const MatrixFree<dim, Number>&               data,
                                             Vec&                                         dst,
                                             const std::vector<Vec>&                      src,
                                             const std::pair<unsigned int, unsigned int>& face_range) const;

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
                                         const std::pair<unsigned int, unsigned int>& face_range) const;

    /*--- Assembler functions for the diagonal part of the matrix for the continuity equation. ---*/
    void assemble_diagonal_cell_term_rho_update(const MatrixFree<dim, Number>&               data,
                                                Vec&                                         dst,
                                                const unsigned int&                          src,
                                                const std::pair<unsigned int, unsigned int>& cell_range) const;

    /*--- Assembler functions for the diagonal part of 'A' matrix. ---*/
    void assemble_diagonal_cell_term_velocity_fixed(const MatrixFree<dim, Number>&               data,
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
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  EULEROperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                     n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  EULEROperator(): MatrixFreeOperators::Base<dim, Vec>(), Ma(), Fr(), dt(), gamma(2.0 - std::sqrt(2.0)),
                   a21(gamma), a22(0.0), a31(0.5), a32(0.5), a33(0.0),
                   a21_tilde(0.5*gamma), a22_tilde(0.5*gamma), a31_tilde(std::sqrt(2)/4.0), a32_tilde(std::sqrt(2)/4.0),
                   a33_tilde(1.0 - std::sqrt(2)/2.0), b1(a31_tilde), b2(a32_tilde), b3(a33_tilde),
                   HYPERBOLIC_stage(1), NS_stage(1), rho_exact(), u_exact(), pres_exact() {}


  // Constructor with runtime parameters storage
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  EULEROperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                     n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  EULEROperator(RunTimeParameters::Data_Storage& data): MatrixFreeOperators::Base<dim, Vec>(),
                                                        Ma(data.Mach), Fr(data.Froude), dt(data.dt), gamma(2.0 - std::sqrt(2.0)),
                                                        a21(gamma), a22(0.0), a31(0.5),
                                                        a32(0.5), a33(0.0),
                                                        a21_tilde(0.5*gamma), a22_tilde(0.5*gamma),
                                                        a31_tilde(std::sqrt(2)/4.0), a32_tilde(std::sqrt(2)/4.0),
                                                        a33_tilde(1.0 - std::sqrt(2)/2.0), b1(a31_tilde),
                                                        b2(a32_tilde), b3(a33_tilde), HYPERBOLIC_stage(1), NS_stage(1),
                                                        rho_exact(data.initial_time), u_exact(data.initial_time),
                                                        pres_exact(data.initial_time) {}


  // Setter of time-step
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void EULEROperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  set_dt(const double time_step) {
    dt = time_step;
  }


  // Setter of Mach number
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void EULEROperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  set_Mach(const double Ma_) {
    Ma = Ma_;
  }


  // Setter of Froude number
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void EULEROperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  set_Froude(const double Fr_) {
    Fr = Fr_;
  }


  // Setter of HYPERBOLIC stage (this can be known only during the effective execution
  // and so it has to be demanded to the class that really solves the problem)
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void EULEROperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  set_HYPERBOLIC_stage(const unsigned int stage) {
    AssertIndexRange(stage, 4);
    Assert(stage > 0, ExcInternalError());

    HYPERBOLIC_stage = stage;
  }


  // Setter of NS stage (this can be known only during the effective execution
  // and so it has to be demanded to the class that really solves the problem)
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void EULEROperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  set_NS_stage(const unsigned int stage) {
    AssertIndexRange(stage, 7);
    Assert(stage > 0, ExcInternalError());

    NS_stage = stage;
  }


  // Setter of density for fixed point
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void EULEROperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  set_rho_for_fixed(const Vec& src) {
    rho_for_fixed = src;
    rho_for_fixed.update_ghost_values();
  }


  // Setter of pressure for fixed point
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void EULEROperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  set_pres_fixed(const Vec& src) {
    pres_fixed = src;
    pres_fixed.update_ghost_values();
  }


  // Setter of velocity for fixed point
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void EULEROperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  set_u_fixed(const Vec& src) {
    u_fixed = src;
    u_fixed.update_ghost_values();
  }


  // Assemble rhs cell term for the density update
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void EULEROperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  assemble_rhs_cell_term_rho_update(const MatrixFree<dim, Number>&               data,
                                    Vec&                                         dst,
                                    const std::vector<Vec>&                      src,
                                    const std::pair<unsigned int, unsigned int>& cell_range) const {
    if(HYPERBOLIC_stage == 1) {
      /*--- We first start by declaring the suitable instances to read the old density and
      the old velocity. 'phi' will be used only to 'submit' the result.
      The second argument specifies which dof handler has to be used (in this implementation 0 stands for
      velocity, 1 for pressure and 2 for density). ---*/
      FEEvaluation<dim, fe_degree_rho, n_q_points_1d_u, 1, Number> phi(data, 2),
                                                                   phi_rho_old(data, 2);
      FEEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number> phi_u_old(data, 0);

      /*--- Loop over all cells ---*/
      for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
        /*--- Now we need to assign the current cell to each FEEvaluation object and then to specify which src vector
        it has to read (the proper order is clearly delegated to the user, which has to pay attention in the function
        call to be coherent). All these considerations are valid also for the other assembler functions ---*/
        phi_rho_old.reinit(cell);
        phi_rho_old.gather_evaluate(src[0], EvaluationFlags::values);
        phi_u_old.reinit(cell);
        phi_u_old.gather_evaluate(src[1], EvaluationFlags::values);

        phi.reinit(cell);

        /*--- Loop over quadrature points of each cell ---*/
        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          const auto& rho_old = phi_rho_old.get_value(q);
          const auto& u_old   = phi_u_old.get_value(q);

          phi.submit_value(rho_old, q);
          /*--- submit_value is used for quantities to be tested against test functions ---*/
          phi.submit_gradient(a21*dt*rho_old*u_old, q);
          /*--- submit_gradient is used for quantities to be tested against gradient of test functions ---*/
        }
        phi.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
        /*--- 'integrate_scatter' is the responsible of distributing into dst.
              The flag parameter specifies if we are testing against the test function and/or its gradient ---*/
      }
    }
    else if(HYPERBOLIC_stage == 2) {
      /*--- We first start by declaring the suitable instances to read the available quantities. ---*/
      FEEvaluation<dim, fe_degree_rho, n_q_points_1d_u, 1, Number> phi(data, 2),
                                                                   phi_rho_old(data, 2),
                                                                   phi_rho_tmp_2(data, 2);
      FEEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number> phi_u_old(data, 0),
                                                                   phi_u_tmp_2(data, 0);

      /*--- Loop over all cells ---*/
      for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
        phi_rho_old.reinit(cell);
        phi_rho_old.gather_evaluate(src[0], EvaluationFlags::values);
        phi_u_old.reinit(cell);
        phi_u_old.gather_evaluate(src[1], EvaluationFlags::values);

        phi_rho_tmp_2.reinit(cell);
        phi_rho_tmp_2.gather_evaluate(src[2], EvaluationFlags::values);
        phi_u_tmp_2.reinit(cell);
        phi_u_tmp_2.gather_evaluate(src[3], EvaluationFlags::values);

        phi.reinit(cell);

        /*--- Loop over quadrature points of each cell ---*/
        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          const auto& rho_old   = phi_rho_old.get_value(q);
          const auto& u_old     = phi_u_old.get_value(q);

          const auto& rho_tmp_2 = phi_rho_tmp_2.get_value(q);
          const auto& u_tmp_2   = phi_u_tmp_2.get_value(q);

          phi.submit_value(rho_old, q);
          phi.submit_gradient(a31*dt*rho_old*u_old +
                              a32*dt*rho_tmp_2*u_tmp_2, q);
        }
        phi.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
      }
    }
    else {
      /*--- We first start by declaring the suitable instances to read the available quantities. ---*/
      FEEvaluation<dim, fe_degree_rho, n_q_points_1d_u, 1, Number> phi(data, 2),
                                                                   phi_rho_old(data, 2),
                                                                   phi_rho_tmp_2(data, 2),
                                                                   phi_rho_tmp_3(data, 2);
      FEEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number> phi_u_old(data, 0),
                                                                   phi_u_tmp_2(data, 0),
                                                                   phi_u_tmp_3(data, 0);

      /*--- Loop over all cells ---*/
      for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
        phi_rho_old.reinit(cell);
        phi_rho_old.gather_evaluate(src[0], EvaluationFlags::values);
        phi_u_old.reinit(cell);
        phi_u_old.gather_evaluate(src[1], EvaluationFlags::values);

        phi_rho_tmp_2.reinit(cell);
        phi_rho_tmp_2.gather_evaluate(src[2], EvaluationFlags::values);
        phi_u_tmp_2.reinit(cell);
        phi_u_tmp_2.gather_evaluate(src[3], EvaluationFlags::values);

        phi_rho_tmp_3.reinit(cell);
        phi_rho_tmp_3.gather_evaluate(src[4], EvaluationFlags::values);
        phi_u_tmp_3.reinit(cell);
        phi_u_tmp_3.gather_evaluate(src[5], EvaluationFlags::values);

        phi.reinit(cell);

        /*--- Loop over quadrature points of each cell ---*/
        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          const auto& rho_old   = phi_rho_old.get_value(q);
          const auto& u_old     = phi_u_old.get_value(q);

          const auto& rho_tmp_2 = phi_rho_tmp_2.get_value(q);
          const auto& u_tmp_2   = phi_u_tmp_2.get_value(q);

          const auto& rho_tmp_3 = phi_rho_tmp_3.get_value(q);
          const auto& u_tmp_3   = phi_u_tmp_3.get_value(q);

          phi.submit_value(rho_old, q);
          phi.submit_gradient(b1*dt*rho_old*u_old +
                              b2*dt*rho_tmp_2*u_tmp_2 +
                              b3*dt*rho_tmp_3*u_tmp_3, q);
        }
        phi.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
      }
    }
  }


  // Assemble rhs face term for the density update
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void EULEROperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  assemble_rhs_face_term_rho_update(const MatrixFree<dim, Number>&               data,
                                    Vec&                                         dst,
                                    const std::vector<Vec>&                      src,
                                    const std::pair<unsigned int, unsigned int>& face_range) const {
    if(HYPERBOLIC_stage == 1) {
      /*--- We first start by declaring the suitable instances to read the available quantities.
            'true' means that we are reading the information from 'inside', whereas 'false' from 'outside' ---*/
      FEFaceEvaluation<dim, fe_degree_rho, n_q_points_1d_u, 1, Number> phi_p(data, true, 2),
                                                                       phi_m(data, false, 2),
                                                                       phi_rho_old_p(data, true, 2),
                                                                       phi_rho_old_m(data, false, 2);
      FEFaceEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number> phi_u_old_p(data, true, 0),
                                                                       phi_u_old_m(data, false, 0);

      /*--- Loop over all internal faces ---*/
      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        phi_rho_old_p.reinit(face);
        phi_rho_old_p.gather_evaluate(src[0], EvaluationFlags::values);
        phi_rho_old_m.reinit(face);
        phi_rho_old_m.gather_evaluate(src[0], EvaluationFlags::values);
        phi_u_old_p.reinit(face);
        phi_u_old_p.gather_evaluate(src[1], EvaluationFlags::values);
        phi_u_old_m.reinit(face);
        phi_u_old_m.gather_evaluate(src[1], EvaluationFlags::values);

        phi_p.reinit(face);
        phi_m.reinit(face);

        /*--- Loop over quadrature points of each internal face ---*/
        for(unsigned int q = 0; q < phi_p.n_q_points; ++q) {
          const auto& n_plus       = phi_p.get_normal_vector(q); /*--- Notice that the unit normal vector is the same from
                                                                       'both sides'. ---*/

          const auto& rho_old_p    = phi_rho_old_p.get_value(q);
          const auto& rho_old_m    = phi_rho_old_m.get_value(q);
          const auto& u_old_p      = phi_u_old_p.get_value(q);
          const auto& u_old_m      = phi_u_old_m.get_value(q);
          const auto& avg_flux_old = 0.5*(rho_old_p*u_old_p + rho_old_m*u_old_m);
          const auto& lambda_old   = std::max(std::abs(scalar_product(u_old_p, n_plus)),
                                              std::abs(scalar_product(u_old_m, n_plus)));
          const auto& jump_rho_old = rho_old_p - rho_old_m;

          /*--- Using an upwind flux ---*/
          phi_p.submit_value(-a21*dt*(scalar_product(avg_flux_old, n_plus) + 0.5*lambda_old*jump_rho_old), q);
          phi_m.submit_value(a21*dt*(scalar_product(avg_flux_old, n_plus) + 0.5*lambda_old*jump_rho_old), q);
        }
        phi_p.integrate_scatter(EvaluationFlags::values, dst);
        phi_m.integrate_scatter(EvaluationFlags::values, dst);
      }
    }
    else if(HYPERBOLIC_stage == 2) {
      /*--- We first start by declaring the suitable instances to read the available quantities. ---*/
      FEFaceEvaluation<dim, fe_degree_rho, n_q_points_1d_u, 1, Number> phi_p(data, true, 2),
                                                                       phi_m(data, false, 2),
                                                                       phi_rho_old_p(data, true, 2),
                                                                       phi_rho_old_m(data, false, 2),
                                                                       phi_rho_tmp_2_p(data, true, 2),
                                                                       phi_rho_tmp_2_m(data, false, 2);
      FEFaceEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number> phi_u_old_p(data, true, 0),
                                                                       phi_u_old_m(data, false, 0),
                                                                       phi_u_tmp_2_p(data, true, 0),
                                                                       phi_u_tmp_2_m(data, false, 0);

      /*--- Loop over all internal faces ---*/
      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        phi_rho_old_p.reinit(face);
        phi_rho_old_p.gather_evaluate(src[0], EvaluationFlags::values);
        phi_rho_old_m.reinit(face);
        phi_rho_old_m.gather_evaluate(src[0], EvaluationFlags::values);
        phi_u_old_p.reinit(face);
        phi_u_old_p.gather_evaluate(src[1], EvaluationFlags::values);
        phi_u_old_m.reinit(face);
        phi_u_old_m.gather_evaluate(src[1], EvaluationFlags::values);

        phi_rho_tmp_2_p.reinit(face);
        phi_rho_tmp_2_p.gather_evaluate(src[2], EvaluationFlags::values);
        phi_rho_tmp_2_m.reinit(face);
        phi_rho_tmp_2_m.gather_evaluate(src[2], EvaluationFlags::values);
        phi_u_tmp_2_p.reinit(face);
        phi_u_tmp_2_p.gather_evaluate(src[3], EvaluationFlags::values);
        phi_u_tmp_2_m.reinit(face);
        phi_u_tmp_2_m.gather_evaluate(src[3], EvaluationFlags::values);

        phi_p.reinit(face);
        phi_m.reinit(face);

        /*--- Loop over all quadrature points ---*/
        for(unsigned int q = 0; q < phi_p.n_q_points; ++q) {
          const auto& n_plus         = phi_p.get_normal_vector(q);

          const auto& rho_old_p      = phi_rho_old_p.get_value(q);
          const auto& rho_old_m      = phi_rho_old_m.get_value(q);
          const auto& u_old_p        = phi_u_old_p.get_value(q);
          const auto& u_old_m        = phi_u_old_m.get_value(q);
          const auto& avg_flux_old   = 0.5*(rho_old_p*u_old_p + rho_old_m*u_old_m);
          const auto& lambda_old     = std::max(std::abs(scalar_product(u_old_p, n_plus)),
                                              std::abs(scalar_product(u_old_m, n_plus)));
          const auto& jump_rho_old   = rho_old_p - rho_old_m;

          const auto& rho_tmp_2_p    = phi_rho_tmp_2_p.get_value(q);
          const auto& rho_tmp_2_m    = phi_rho_tmp_2_m.get_value(q);
          const auto& u_tmp_2_p      = phi_u_tmp_2_p.get_value(q);
          const auto& u_tmp_2_m      = phi_u_tmp_2_m.get_value(q);
          const auto& avg_flux_tmp_2 = 0.5*(rho_tmp_2_p*u_tmp_2_p + rho_tmp_2_m*u_tmp_2_m);
          const auto& lambda_tmp_2   = std::max(std::abs(scalar_product(u_tmp_2_p, n_plus)),
                                                std::abs(scalar_product(u_tmp_2_m, n_plus)));
          const auto& jump_rho_tmp_2 = rho_tmp_2_p - rho_tmp_2_m;

          /*--- Using an upwind flux ---*/
          phi_p.submit_value(-a31*dt*(scalar_product(avg_flux_old, n_plus) + 0.5*lambda_old*jump_rho_old)
                             -a32*dt*(scalar_product(avg_flux_tmp_2, n_plus) + 0.5*lambda_tmp_2*jump_rho_tmp_2), q);
          phi_m.submit_value(a31*dt*(scalar_product(avg_flux_old, n_plus) + 0.5*lambda_old*jump_rho_old) +
                             a32*dt*(scalar_product(avg_flux_tmp_2, n_plus) + 0.5*lambda_tmp_2*jump_rho_tmp_2), q);
        }
        phi_p.integrate_scatter(EvaluationFlags::values, dst);
        phi_m.integrate_scatter(EvaluationFlags::values, dst);
      }
    }
    else {
      /*--- We first start by declaring the suitable instances to read the available quantities. ---*/
      FEFaceEvaluation<dim, fe_degree_rho, n_q_points_1d_u, 1, Number> phi_p(data, true, 2),
                                                                       phi_m(data, false, 2),
                                                                       phi_rho_old_p(data, true, 2),
                                                                       phi_rho_old_m(data, false, 2),
                                                                       phi_rho_tmp_2_p(data, true, 2),
                                                                       phi_rho_tmp_2_m(data, false, 2),
                                                                       phi_rho_tmp_3_p(data, true, 2),
                                                                       phi_rho_tmp_3_m(data, false, 2);
      FEFaceEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number> phi_u_old_p(data, true, 0),
                                                                       phi_u_old_m(data, false, 0),
                                                                       phi_u_tmp_2_p(data, true, 0),
                                                                       phi_u_tmp_2_m(data, false, 0),
                                                                       phi_u_tmp_3_p(data, true, 0),
                                                                       phi_u_tmp_3_m(data, false, 0);

      /*--- Loop over all internal faces ---*/
      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        phi_rho_old_p.reinit(face);
        phi_rho_old_p.gather_evaluate(src[0], EvaluationFlags::values);
        phi_rho_old_m.reinit(face);
        phi_rho_old_m.gather_evaluate(src[0], EvaluationFlags::values);
        phi_u_old_p.reinit(face);
        phi_u_old_p.gather_evaluate(src[1], EvaluationFlags::values);
        phi_u_old_m.reinit(face);
        phi_u_old_m.gather_evaluate(src[1], EvaluationFlags::values);

        phi_rho_tmp_2_p.reinit(face);
        phi_rho_tmp_2_p.gather_evaluate(src[2], EvaluationFlags::values);
        phi_rho_tmp_2_m.reinit(face);
        phi_rho_tmp_2_m.gather_evaluate(src[2], EvaluationFlags::values);
        phi_u_tmp_2_p.reinit(face);
        phi_u_tmp_2_p.gather_evaluate(src[3], EvaluationFlags::values);
        phi_u_tmp_2_m.reinit(face);
        phi_u_tmp_2_m.gather_evaluate(src[3], EvaluationFlags::values);

        phi_rho_tmp_3_p.reinit(face);
        phi_rho_tmp_3_p.gather_evaluate(src[4], EvaluationFlags::values);
        phi_rho_tmp_3_m.reinit(face);
        phi_rho_tmp_3_m.gather_evaluate(src[4], EvaluationFlags::values);
        phi_u_tmp_3_p.reinit(face);
        phi_u_tmp_3_p.gather_evaluate(src[5], EvaluationFlags::values);
        phi_u_tmp_3_m.reinit(face);
        phi_u_tmp_3_m.gather_evaluate(src[5], EvaluationFlags::values);

        phi_p.reinit(face);
        phi_m.reinit(face);

        /*--- Loop over all quadrature points. ---*/
        for(unsigned int q = 0; q < phi_p.n_q_points; ++q) {
          const auto& n_plus         = phi_p.get_normal_vector(q);

          const auto& rho_old_p      = phi_rho_old_p.get_value(q);
          const auto& rho_old_m      = phi_rho_old_m.get_value(q);
          const auto& u_old_p        = phi_u_old_p.get_value(q);
          const auto& u_old_m        = phi_u_old_m.get_value(q);
          const auto& avg_flux_old   = 0.5*(rho_old_p*u_old_p + rho_old_m*u_old_m);
          const auto& lambda_old     = std::max(std::abs(scalar_product(u_old_p, n_plus)),
                                              std::abs(scalar_product(u_old_m, n_plus)));
          const auto& jump_rho_old   = rho_old_p - rho_old_m;

          const auto& rho_tmp_2_p    = phi_rho_tmp_2_p.get_value(q);
          const auto& rho_tmp_2_m    = phi_rho_tmp_2_m.get_value(q);
          const auto& u_tmp_2_p      = phi_u_tmp_2_p.get_value(q);
          const auto& u_tmp_2_m      = phi_u_tmp_2_m.get_value(q);
          const auto& avg_flux_tmp_2 = 0.5*(rho_tmp_2_p*u_tmp_2_p + rho_tmp_2_m*u_tmp_2_m);
          const auto& lambda_tmp_2   = std::max(std::abs(scalar_product(u_tmp_2_p, n_plus)),
                                                std::abs(scalar_product(u_tmp_2_m, n_plus)));
          const auto& jump_rho_tmp_2 = rho_tmp_2_p - rho_tmp_2_m;

          const auto& rho_tmp_3_p    = phi_rho_tmp_3_p.get_value(q);
          const auto& rho_tmp_3_m    = phi_rho_tmp_3_m.get_value(q);
          const auto& u_tmp_3_p      = phi_u_tmp_3_p.get_value(q);
          const auto& u_tmp_3_m      = phi_u_tmp_3_m.get_value(q);
          const auto& avg_flux_tmp_3 = 0.5*(rho_tmp_3_p*u_tmp_3_p + rho_tmp_3_m*u_tmp_3_m);
          const auto& lambda_tmp_3   = std::max(std::abs(scalar_product(u_tmp_3_p, n_plus)),
                                                std::abs(scalar_product(u_tmp_3_m, n_plus)));
          const auto& jump_rho_tmp_3 = rho_tmp_3_p - rho_tmp_3_m;

          /*--- Using an upwind flux ---*/
          phi_p.submit_value(-b1*dt*(scalar_product(avg_flux_old, n_plus) + 0.5*lambda_old*jump_rho_old)
                             -b2*dt*(scalar_product(avg_flux_tmp_2, n_plus) + 0.5*lambda_tmp_2*jump_rho_tmp_2)
                             -b3*dt*(scalar_product(avg_flux_tmp_3, n_plus) + 0.5*lambda_tmp_3*jump_rho_tmp_3), q);
          phi_m.submit_value(b1*dt*(scalar_product(avg_flux_old, n_plus) + 0.5*lambda_old*jump_rho_old) +
                             b2*dt*(scalar_product(avg_flux_tmp_2, n_plus) + 0.5*lambda_tmp_2*jump_rho_tmp_2) +
                             b3*dt*(scalar_product(avg_flux_tmp_3, n_plus) + 0.5*lambda_tmp_3*jump_rho_tmp_3), q);
        }
        phi_p.integrate_scatter(EvaluationFlags::values, dst);
        phi_m.integrate_scatter(EvaluationFlags::values, dst);
      }
    }
  }


  // Assemble rhs boundary term for the density update
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void EULEROperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  assemble_rhs_boundary_term_rho_update(const MatrixFree<dim, Number>&               data,
                                        Vec&                                         dst,
                                        const std::vector<Vec>&                      src,
                                        const std::pair<unsigned int, unsigned int>& face_range) const {
    if(HYPERBOLIC_stage == 1) {
      /*--- We first start by declaring the suitable instances to read the available quantities.
            'true' means that we are reading the information from 'inside'. ---*/
      FEFaceEvaluation<dim, fe_degree_rho, n_q_points_1d_u, 1, Number> phi(data, true, 2),
                                                                       phi_rho_old(data, true, 2);
      FEFaceEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number> phi_u_old(data, true, 0);

      /*--- Loop over all internal faces ---*/
      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        phi_rho_old.reinit(face);
        phi_rho_old.gather_evaluate(src[0], EvaluationFlags::values);
        phi_u_old.reinit(face);
        phi_u_old.gather_evaluate(src[1], EvaluationFlags::values);

        phi.reinit(face);

        /*--- Loop over quadrature points of each internal face ---*/
        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          const auto& n_plus           = phi.get_normal_vector(q);

          const auto& rho_old          = phi_rho_old.get_value(q);
          const auto& u_old            = phi_u_old.get_value(q);
          auto rho_old_D               = VectorizedArray<Number>();
          auto u_old_D                 = Tensor<1, dim, VectorizedArray<Number>>();
          const auto& point_vectorized = phi.quadrature_point(q);
          for(unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v) {
            Point<dim> point; /*--- The point returned by the 'quadrature_point' function is not an instance of Point
                                    and so it is not ready to be directly used. We need to pay attention to the
                                    vectorization ---*/
            for(unsigned int d = 0; d < dim; ++d) {
              point[d] = point_vectorized[d][v];
            }
            rho_old_D = rho_exact.value(point);
            for(unsigned int d = 0; d < dim; ++d) {
              u_old_D[d][v] = u_exact.value(point, d);
            }
          }
          const auto& avg_flux_old    = 0.5*(rho_old*u_old + rho_old_D*u_old_D);
          const auto& lambda_old      = std::max(std::abs(scalar_product(u_old, n_plus)),
                                                 std::abs(scalar_product(u_old_D, n_plus)));
          const auto& jump_rho_old    = rho_old - rho_old_D;

          /*--- Using an upwind flux ---*/
          phi.submit_value(-a21*dt*(scalar_product(avg_flux_old, n_plus) + 0.5*lambda_old*jump_rho_old), q);
        }
        phi.integrate_scatter(EvaluationFlags::values, dst);
      }
    }
    else if(HYPERBOLIC_stage == 2) {
      /*--- We first start by declaring the suitable instances to read the available quantities. ---*/
      FEFaceEvaluation<dim, fe_degree_rho, n_q_points_1d_u, 1, Number> phi(data, true, 2),
                                                                       phi_rho_old(data, true, 2),
                                                                       phi_rho_tmp_2(data, true, 2);
      FEFaceEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number> phi_u_old(data, true, 0),
                                                                       phi_u_tmp_2(data, true, 0);

      /*--- Loop over all internal faces ---*/
      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        phi_rho_old.reinit(face);
        phi_rho_old.gather_evaluate(src[0], EvaluationFlags::values);
        phi_u_old.reinit(face);
        phi_u_old.gather_evaluate(src[1], EvaluationFlags::values);

        phi_rho_tmp_2.reinit(face);
        phi_rho_tmp_2.gather_evaluate(src[2], EvaluationFlags::values);
        phi_u_tmp_2.reinit(face);
        phi_u_tmp_2.gather_evaluate(src[3], EvaluationFlags::values);

        phi.reinit(face);

        /*--- Loop over all quadrature points ---*/
        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          const auto& n_plus           = phi.get_normal_vector(q);

          const auto& rho_old          = phi_rho_old.get_value(q);
          const auto& u_old            = phi_u_old.get_value(q);
          auto rho_old_D               = VectorizedArray<Number>();
          auto u_old_D                 = Tensor<1, dim, VectorizedArray<Number>>();
          const auto& point_vectorized = phi.quadrature_point(q);
          for(unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v) {
            Point<dim> point;
            for(unsigned int d = 0; d < dim; ++d) {
              point[d] = point_vectorized[d][v];
            }
            rho_old_D = rho_exact.value(point);
            for(unsigned int d = 0; d < dim; ++d) {
              u_old_D[d][v] = u_exact.value(point, d);
            }
          }
          const auto& avg_flux_old    = 0.5*(rho_old*u_old + rho_old_D*u_old_D);
          const auto& lambda_old      = std::max(std::abs(scalar_product(u_old, n_plus)),
                                                 std::abs(scalar_product(u_old_D, n_plus)));
          const auto& jump_rho_old    = rho_old - rho_old_D;

          const auto& rho_tmp_2       = phi_rho_tmp_2.get_value(q);
          const auto& u_tmp_2         = phi_u_tmp_2.get_value(q);
          auto rho_tmp_2_D            = VectorizedArray<Number>();
          auto u_tmp_2_D              = Tensor<1, dim, VectorizedArray<Number>>();
          for(unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v) {
            Point<dim> point;
            for(unsigned int d = 0; d < dim; ++d) {
              point[d] = point_vectorized[d][v];
            }
            rho_tmp_2_D = rho_exact.value(point);
            for(unsigned int d = 0; d < dim; ++d) {
              u_tmp_2_D[d][v] = u_exact.value(point, d);
            }
          }
          const auto& avg_flux_tmp_2  = 0.5*(rho_tmp_2*u_tmp_2 + rho_tmp_2_D*u_tmp_2_D);
          const auto& lambda_tmp_2    = std::max(std::abs(scalar_product(u_tmp_2, n_plus)),
                                                 std::abs(scalar_product(u_tmp_2_D, n_plus)));
          const auto& jump_rho_tmp_2  = rho_tmp_2 - rho_tmp_2_D;

          /*--- Using an upwind flux ---*/
          phi.submit_value(-a31*dt*(scalar_product(avg_flux_old, n_plus) + 0.5*lambda_old*jump_rho_old)
                           -a32*dt*(scalar_product(avg_flux_tmp_2, n_plus) + 0.5*lambda_tmp_2*jump_rho_tmp_2), q);
        }
        phi.integrate_scatter(EvaluationFlags::values, dst);
      }
    }
    else {
      /*--- We first start by declaring the suitable instances to read the available quantities. ---*/
      FEFaceEvaluation<dim, fe_degree_rho, n_q_points_1d_u, 1, Number> phi(data, true, 2),
                                                                       phi_rho_old(data, true, 2),
                                                                       phi_rho_tmp_2(data, true, 2),
                                                                       phi_rho_tmp_3(data, true, 2);
      FEFaceEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number> phi_u_old(data, true, 0),
                                                                       phi_u_tmp_2(data, true, 0),
                                                                       phi_u_tmp_3(data, true, 0);

      /*--- Loop over all internal faces ---*/
      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        phi_rho_old.reinit(face);
        phi_rho_old.gather_evaluate(src[0], EvaluationFlags::values);
        phi_u_old.reinit(face);
        phi_u_old.gather_evaluate(src[1], EvaluationFlags::values);

        phi_rho_tmp_2.reinit(face);
        phi_rho_tmp_2.gather_evaluate(src[2], EvaluationFlags::values);
        phi_u_tmp_2.reinit(face);
        phi_u_tmp_2.gather_evaluate(src[3], EvaluationFlags::values);

        phi_rho_tmp_3.reinit(face);
        phi_rho_tmp_3.gather_evaluate(src[4], EvaluationFlags::values);
        phi_u_tmp_3.reinit(face);
        phi_u_tmp_3.gather_evaluate(src[5], EvaluationFlags::values);

        phi.reinit(face);

        /*--- Loop over all quadrature points. ---*/
        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          const auto& n_plus         = phi.get_normal_vector(q);

          const auto& rho_old          = phi_rho_old.get_value(q);
          const auto& u_old            = phi_u_old.get_value(q);
          auto rho_old_D               = VectorizedArray<Number>();
          auto u_old_D                 = Tensor<1, dim, VectorizedArray<Number>>();
          const auto& point_vectorized = phi.quadrature_point(q);
          for(unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v) {
            Point<dim> point;
            for(unsigned int d = 0; d < dim; ++d) {
              point[d] = point_vectorized[d][v];
            }
            rho_old_D = rho_exact.value(point);
            for(unsigned int d = 0; d < dim; ++d) {
              u_old_D[d][v] = u_exact.value(point, d);
            }
          }
          const auto& avg_flux_old    = 0.5*(rho_old*u_old + rho_old_D*u_old_D);
          const auto& lambda_old      = std::max(std::abs(scalar_product(u_old, n_plus)),
                                                 std::abs(scalar_product(u_old_D, n_plus)));
          const auto& jump_rho_old    = rho_old - rho_old_D;

          const auto& rho_tmp_2       = phi_rho_tmp_2.get_value(q);
          const auto& u_tmp_2         = phi_u_tmp_2.get_value(q);
          auto rho_tmp_2_D            = VectorizedArray<Number>();
          auto u_tmp_2_D              = Tensor<1, dim, VectorizedArray<Number>>();
          for(unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v) {
            Point<dim> point;
            for(unsigned int d = 0; d < dim; ++d) {
              point[d] = point_vectorized[d][v];
            }
            rho_tmp_2_D = rho_exact.value(point);
            for(unsigned int d = 0; d < dim; ++d) {
              u_tmp_2_D[d][v] = u_exact.value(point, d);
            }
          }
          const auto& avg_flux_tmp_2  = 0.5*(rho_tmp_2*u_tmp_2 + rho_tmp_2_D*u_tmp_2_D);
          const auto& lambda_tmp_2    = std::max(std::abs(scalar_product(u_tmp_2, n_plus)),
                                                 std::abs(scalar_product(u_tmp_2_D, n_plus)));
          const auto& jump_rho_tmp_2  = rho_tmp_2 - rho_tmp_2_D;

          const auto& rho_tmp_3       = phi_rho_tmp_3.get_value(q);
          const auto& u_tmp_3         = phi_u_tmp_3.get_value(q);
          auto rho_tmp_3_D            = VectorizedArray<Number>();
          auto u_tmp_3_D              = Tensor<1, dim, VectorizedArray<Number>>();
          for(unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v) {
            Point<dim> point;
            for(unsigned int d = 0; d < dim; ++d) {
              point[d] = point_vectorized[d][v];
            }
            rho_tmp_3_D = rho_exact.value(point);
            for(unsigned int d = 0; d < dim; ++d) {
              u_tmp_3_D[d][v] = u_exact.value(point, d);
            }
          }
          const auto& avg_flux_tmp_3  = 0.5*(rho_tmp_3*u_tmp_3 + rho_tmp_3_D*u_tmp_3_D);
          const auto& lambda_tmp_3    = std::max(std::abs(scalar_product(u_tmp_3, n_plus)),
                                                 std::abs(scalar_product(u_tmp_3_D, n_plus)));
          const auto& jump_rho_tmp_3  = rho_tmp_3 - rho_tmp_3_D;

          /*--- Using an upwind flux ---*/
          phi.submit_value(-b1*dt*(scalar_product(avg_flux_old, n_plus) + 0.5*lambda_old*jump_rho_old)
                           -b2*dt*(scalar_product(avg_flux_tmp_2, n_plus) + 0.5*lambda_tmp_2*jump_rho_tmp_2)
                           -b3*dt*(scalar_product(avg_flux_tmp_3, n_plus) + 0.5*lambda_tmp_3*jump_rho_tmp_3), q);
        }
        phi.integrate_scatter(EvaluationFlags::values, dst);
      }
    }
  }


  // Put together all the previous steps for density update
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void EULEROperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  vmult_rhs_rho_update(Vec& dst, const std::vector<Vec>& src) const {
    for(unsigned int d = 0; d < src.size(); ++d) {
      src[d].update_ghost_values();
    }

    this->data->loop(&EULEROperator::assemble_rhs_cell_term_rho_update,
                     &EULEROperator::assemble_rhs_face_term_rho_update,
                     &EULEROperator::assemble_rhs_boundary_term_rho_update,
                     this, dst, src, true,
                     MatrixFree<dim, Number>::DataAccessOnFaces::unspecified,
                     MatrixFree<dim, Number>::DataAccessOnFaces::unspecified);
  }


  // Assemble cell term for the density update
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void EULEROperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  assemble_cell_term_rho_update(const MatrixFree<dim, Number>&               data,
                                    Vec&                                         dst,
                                    const Vec&                                   src,
                                    const std::pair<unsigned int, unsigned int>& cell_range) const {
    FEEvaluation<dim, fe_degree_rho, n_q_points_1d_u, 1, Number> phi(data, 2);

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


  // Assemble rhs cell term for the pressure
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void EULEROperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  assemble_rhs_cell_term_pressure(const MatrixFree<dim, Number>&               data,
                                  Vec&                                         dst,
                                  const std::vector<Vec>&                      src,
                                  const std::pair<unsigned int, unsigned int>& cell_range) const {
    if(HYPERBOLIC_stage == 1) {
      /*--- We first start by declaring the suitable instances to read the available quantities. ---*/
      FEEvaluation<dim, fe_degree_T, n_q_points_1d_u, 1, Number>   phi(data, 1),
                                                                   phi_pres_old(data, 1);
      FEEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number> phi_u_old(data, 0),
                                                                   phi_u_fixed(data, 0);
      FEEvaluation<dim, fe_degree_rho, n_q_points_1d_u, 1, Number> phi_rho_old(data, 2),
                                                                   phi_rho_tmp_2(data, 2);

      /*--- Loop over all cells ---*/
      for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
        phi_rho_old.reinit(cell);
        phi_rho_old.gather_evaluate(src[0], EvaluationFlags::values);
        phi_u_old.reinit(cell);
        phi_u_old.gather_evaluate(src[1], EvaluationFlags::values);
        phi_pres_old.reinit(cell);
        phi_pres_old.gather_evaluate(src[2], EvaluationFlags::values);

        phi_rho_tmp_2.reinit(cell);
        phi_rho_tmp_2.gather_evaluate(src[3], EvaluationFlags::values);
        phi_u_fixed.reinit(cell);
        phi_u_fixed.gather_evaluate(src[4], EvaluationFlags::values);

        phi.reinit(cell);

        /*--- Loop over all quadrature points ---*/
        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          const auto& rho_old   = phi_rho_old.get_value(q);
          const auto& u_old     = phi_u_old.get_value(q);
          const auto& pres_old  = phi_pres_old.get_value(q);
          const auto& E_old     = 1.0/(EquationData::Cp_Cv - 1.0)*(pres_old/rho_old)
                                + 0.5*Ma*Ma*scalar_product(u_old, u_old);

          /*--- We assign to the rhs the contribution due to kinetic energy in the fixed point loop ---*/
          const auto& rho_tmp_2 = phi_rho_tmp_2.get_value(q);
          const auto& u_fixed   = phi_u_fixed.get_value(q);

          phi.submit_value(rho_old*E_old -
                           0.5*rho_tmp_2*Ma*Ma*scalar_product(u_fixed, u_fixed) -
                           a21_tilde*dt*Ma*Ma/(Fr*Fr)*rho_old*u_old[dim - 1] -
                           a22_tilde*dt*Ma*Ma/(Fr*Fr)*rho_tmp_2*u_fixed[dim - 1], q);
          phi.submit_gradient(0.5*a21*dt*Ma*Ma*scalar_product(u_old, u_old)*rho_old*u_old +
                              a21_tilde*dt*(rho_old*(E_old - 0.5*Ma*Ma*scalar_product(u_old, u_old)) + pres_old)*u_old, q);
          /*--- The specific enthalpy is computed with the generic relation e + p/rho ---*/
        }
        phi.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
      }
    }
    else if(HYPERBOLIC_stage == 2) {
      /*--- We first start by declaring the suitable instances to read the available quantities. ---*/
      FEEvaluation<dim, fe_degree_T, n_q_points_1d_u, 1, Number>   phi(data, 1),
                                                                   phi_pres_old(data, 1),
                                                                   phi_pres_tmp_2(data, 1);
      FEEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number> phi_u_old(data, 0),
                                                                   phi_u_tmp_2(data, 0),
                                                                   phi_u_fixed(data, 0);
      FEEvaluation<dim, fe_degree_rho, n_q_points_1d_u, 1, Number> phi_rho_old(data, 2),
                                                                   phi_rho_tmp_2(data, 2),
                                                                   phi_rho_tmp_3(data, 2);

      /*--- Loop over all cells ---*/
      for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
        phi_rho_old.reinit(cell);
        phi_rho_old.gather_evaluate(src[0], EvaluationFlags::values);
        phi_u_old.reinit(cell);
        phi_u_old.gather_evaluate(src[1], EvaluationFlags::values);
        phi_pres_old.reinit(cell);
        phi_pres_old.gather_evaluate(src[2], EvaluationFlags::values);

        phi_rho_tmp_2.reinit(cell);
        phi_rho_tmp_2.gather_evaluate(src[3], EvaluationFlags::values);
        phi_u_tmp_2.reinit(cell);
        phi_u_tmp_2.gather_evaluate(src[4], EvaluationFlags::values);
        phi_pres_tmp_2.reinit(cell);
        phi_pres_tmp_2.gather_evaluate(src[5], EvaluationFlags::values);

        phi_rho_tmp_3.reinit(cell);
        phi_rho_tmp_3.gather_evaluate(src[6], EvaluationFlags::values);
        phi_u_fixed.reinit(cell);
        phi_u_fixed.gather_evaluate(src[7], EvaluationFlags::values);

        phi.reinit(cell);

        /*--- Loop over all quadrature points ---*/
        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          const auto& rho_old    = phi_rho_old.get_value(q);
          const auto& u_old      = phi_u_old.get_value(q);
          const auto& pres_old   = phi_pres_old.get_value(q);
          const auto& E_old      = 1.0/(EquationData::Cp_Cv - 1.0)*(pres_old/rho_old)
                                 + 0.5*Ma*Ma*scalar_product(u_old, u_old);

          const auto& rho_tmp_2  = phi_rho_tmp_2.get_value(q);
          const auto& u_tmp_2    = phi_u_tmp_2.get_value(q);
          const auto& pres_tmp_2 = phi_pres_tmp_2.get_value(q);
          const auto& E_tmp_2    = 1.0/(EquationData::Cp_Cv - 1.0)*(pres_tmp_2/rho_tmp_2)
                                 + 0.5*Ma*Ma*scalar_product(u_tmp_2, u_tmp_2);

          const auto& rho_tmp_3  = phi_rho_tmp_3.get_value(q);
          const auto& u_fixed    = phi_u_fixed.get_value(q);

          phi.submit_value(rho_old*E_old -
                           0.5*rho_tmp_3*Ma*Ma*scalar_product(u_fixed, u_fixed) -
                           a31_tilde*dt*Ma*Ma/(Fr*Fr)*rho_old*u_old[dim - 1] -
                           a32_tilde*dt*Ma*Ma/(Fr*Fr)*rho_tmp_2*u_tmp_2[dim - 1] -
                           a33_tilde*dt*Ma*Ma/(Fr*Fr)*rho_tmp_3*u_fixed[dim - 1], q);
          phi.submit_gradient(0.5*a31*dt*Ma*Ma*scalar_product(u_old, u_old)*rho_old*u_old +
                              a31_tilde*dt*
                              (rho_old*(E_old - 0.5*Ma*Ma*scalar_product(u_old, u_old)) + pres_old)*u_old +
                              0.5*a32*dt*Ma*Ma*scalar_product(u_tmp_2, u_tmp_2)*rho_tmp_2*u_tmp_2 +
                              a32_tilde*dt*
                              (rho_tmp_2*(E_tmp_2 - 0.5*Ma*Ma*scalar_product(u_tmp_2, u_tmp_2)) + pres_tmp_2)*u_tmp_2, q);
        }
        phi.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
      }
    }
    else {
      /*--- We first start by declaring the suitable instances to read the available quantities. ---*/
      FEEvaluation<dim, fe_degree_T, n_q_points_1d_u, 1, Number>   phi(data, 1),
                                                                   phi_pres_old(data, 1),
                                                                   phi_pres_tmp_2(data, 1),
                                                                   phi_pres_tmp_3(data, 1);
      FEEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number> phi_u_old(data, 0),
                                                                   phi_u_tmp_2(data, 0),
                                                                   phi_u_tmp_3(data, 0),
                                                                   phi_u_fixed(data, 0);
      FEEvaluation<dim, fe_degree_rho, n_q_points_1d_u, 1, Number> phi_rho_old(data, 2),
                                                                   phi_rho_tmp_2(data, 2),
                                                                   phi_rho_tmp_3(data, 2),
                                                                   phi_rho_curr(data, 2);

      /*--- Loop over all cells ---*/
      for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
        phi_rho_old.reinit(cell);
        phi_rho_old.gather_evaluate(src[0], EvaluationFlags::values);
        phi_u_old.reinit(cell);
        phi_u_old.gather_evaluate(src[1], EvaluationFlags::values);
        phi_pres_old.reinit(cell);
        phi_pres_old.gather_evaluate(src[2], EvaluationFlags::values);

        phi_rho_tmp_2.reinit(cell);
        phi_rho_tmp_2.gather_evaluate(src[3], EvaluationFlags::values);
        phi_u_tmp_2.reinit(cell);
        phi_u_tmp_2.gather_evaluate(src[4], EvaluationFlags::values);
        phi_pres_tmp_2.reinit(cell);
        phi_pres_tmp_2.gather_evaluate(src[5], EvaluationFlags::values);

        phi_rho_tmp_3.reinit(cell);
        phi_rho_tmp_3.gather_evaluate(src[6], EvaluationFlags::values);
        phi_u_tmp_3.reinit(cell);
        phi_u_tmp_3.gather_evaluate(src[7], EvaluationFlags::values);
        phi_pres_tmp_3.reinit(cell);
        phi_pres_tmp_3.gather_evaluate(src[8], EvaluationFlags::values);

        phi_rho_curr.reinit(cell);
        phi_rho_curr.gather_evaluate(src[9], EvaluationFlags::values);
        phi_u_fixed.reinit(cell);
        phi_u_fixed.gather_evaluate(src[10], EvaluationFlags::values);

        phi.reinit(cell);

        /*--- Loop over all quadrature points ---*/
        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          const auto& rho_old    = phi_rho_old.get_value(q);
          const auto& u_old      = phi_u_old.get_value(q);
          const auto& pres_old   = phi_pres_old.get_value(q);
          const auto& E_old      = 1.0/(EquationData::Cp_Cv - 1.0)*(pres_old/rho_old)
                                 + 0.5*Ma*Ma*scalar_product(u_old, u_old);

          const auto& rho_tmp_2  = phi_rho_tmp_2.get_value(q);
          const auto& u_tmp_2    = phi_u_tmp_2.get_value(q);
          const auto& pres_tmp_2 = phi_pres_tmp_2.get_value(q);
          const auto& E_tmp_2    = 1.0/(EquationData::Cp_Cv - 1.0)*(pres_tmp_2/rho_tmp_2)
                                 + 0.5*Ma*Ma*scalar_product(u_tmp_2, u_tmp_2);

          const auto& rho_tmp_3  = phi_rho_tmp_3.get_value(q);
          const auto& u_tmp_3    = phi_u_tmp_3.get_value(q);
          const auto& pres_tmp_3 = phi_pres_tmp_3.get_value(q);
          const auto& E_tmp_3    = 1.0/(EquationData::Cp_Cv - 1.0)*(pres_tmp_3/rho_tmp_3)
                                 + 0.5*Ma*Ma*scalar_product(u_tmp_3, u_tmp_3);

          const auto& rho_curr   = phi_rho_curr.get_value(q);
          const auto& u_fixed    = phi_u_fixed.get_value(q);

          phi.submit_value(rho_old*E_old -
                           0.5*rho_curr*Ma*Ma*scalar_product(u_fixed, u_fixed) -
                           b1*dt*Ma*Ma/(Fr*Fr)*rho_old*u_old[dim - 1] -
                           b2*dt*Ma*Ma/(Fr*Fr)*rho_tmp_2*u_tmp_2[dim - 1] -
                           b3*dt*Ma*Ma/(Fr*Fr)*rho_tmp_3*u_tmp_3[dim - 1], q);
          phi.submit_gradient(0.5*b1*dt*Ma*Ma*scalar_product(u_old, u_old)*rho_old*u_old +
                              b1*dt*(rho_old*(E_old - 0.5*Ma*Ma*scalar_product(u_old, u_old)) + pres_old)*u_old +
                              0.5*b2*dt*Ma*Ma*scalar_product(u_tmp_2, u_tmp_2)*rho_tmp_2*u_tmp_2 +
                              b2*dt*(rho_tmp_2*(E_tmp_2 - 0.5*Ma*Ma*scalar_product(u_tmp_2, u_tmp_2)) + pres_tmp_2)*u_tmp_2 +
                              0.5*b3*dt*Ma*Ma*scalar_product(u_tmp_3, u_tmp_3)*rho_tmp_3*u_tmp_3 +
                              b3*dt*(rho_tmp_3*(E_tmp_3 - 0.5*Ma*Ma*scalar_product(u_tmp_3, u_tmp_3)) + pres_tmp_3)*u_tmp_3, q);
        }
        phi.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
      }
    }
  }


  // Assemble rhs face term for the pressure
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void EULEROperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  assemble_rhs_face_term_pressure(const MatrixFree<dim, Number>&               data,
                                  Vec&                                         dst,
                                  const std::vector<Vec>&                      src,
                                  const std::pair<unsigned int, unsigned int>& face_range) const {
    if(HYPERBOLIC_stage == 1) {
      /*--- We first start by declaring the suitable instances to read the available quantities. ---*/
      FEFaceEvaluation<dim, fe_degree_T, n_q_points_1d_u, 1, Number>   phi_p(data, true, 1),
                                                                       phi_m(data, false, 1),
                                                                       phi_pres_old_p(data, true, 1),
                                                                       phi_pres_old_m(data, false, 1);
      FEFaceEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number> phi_u_old_p(data, true, 0),
                                                                       phi_u_old_m(data, false, 0);
      FEFaceEvaluation<dim, fe_degree_rho, n_q_points_1d_u, 1, Number> phi_rho_old_p(data, true, 2),
                                                                       phi_rho_old_m(data, false, 2);

      /*--- Loop over all internal faces ---*/
      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        phi_rho_old_p.reinit(face);
        phi_rho_old_p.gather_evaluate(src[0], EvaluationFlags::values);
        phi_rho_old_m.reinit(face);
        phi_rho_old_m.gather_evaluate(src[0], EvaluationFlags::values);
        phi_u_old_p.reinit(face);
        phi_u_old_p.gather_evaluate(src[1], EvaluationFlags::values);
        phi_u_old_m.reinit(face);
        phi_u_old_m.gather_evaluate(src[1], EvaluationFlags::values);
        phi_pres_old_p.reinit(face);
        phi_pres_old_p.gather_evaluate(src[2], EvaluationFlags::values);
        phi_pres_old_m.reinit(face);
        phi_pres_old_m.gather_evaluate(src[2], EvaluationFlags::values);

        phi_p.reinit(face);
        phi_m.reinit(face);

        /*--- Loop over quadrature points ---*/
        for(unsigned int q = 0; q < phi_p.n_q_points; ++q) {
          const auto& n_plus           = phi_p.get_normal_vector(q);

          const auto& rho_old_p        = phi_rho_old_p.get_value(q);
          const auto& rho_old_m        = phi_rho_old_m.get_value(q);
          const auto& u_old_p          = phi_u_old_p.get_value(q);
          const auto& u_old_m          = phi_u_old_m.get_value(q);
          const auto& avg_kinetic_old  = 0.5*(0.5*scalar_product(u_old_p, u_old_p)*rho_old_p*u_old_p +
                                              0.5*scalar_product(u_old_m, u_old_m)*rho_old_m*u_old_m);

          const auto& pres_old_p       = phi_pres_old_p.get_value(q);
          const auto& pres_old_m       = phi_pres_old_m.get_value(q);
          const auto& E_old_p          = 1.0/(EquationData::Cp_Cv - 1.0)*pres_old_p/rho_old_p
                                       + 0.5*Ma*Ma*scalar_product(u_old_p, u_old_p);
          const auto& E_old_m          = 1.0/(EquationData::Cp_Cv - 1.0)*pres_old_m/rho_old_m
                                       + 0.5*Ma*Ma*scalar_product(u_old_m, u_old_m);
          const auto& avg_enthalpy_old = 0.5*(((E_old_p - 0.5*Ma*Ma*scalar_product(u_old_p, u_old_p))*rho_old_p + pres_old_p)*u_old_p +
                                              ((E_old_m - 0.5*Ma*Ma*scalar_product(u_old_m, u_old_m))*rho_old_m + pres_old_m)*u_old_m);

          const auto& lambda_old       = std::max(std::abs(scalar_product(u_old_p, n_plus)),
                                                  std::abs(scalar_product(u_old_m, n_plus)));
          const auto& jump_rho_kin_old = rho_old_p*0.5*scalar_product(u_old_p, u_old_p) -
                                         rho_old_m*0.5*scalar_product(u_old_m, u_old_m);
          const auto& jump_rho_e_old   = rho_old_p*(E_old_p - 0.5*Ma*Ma*scalar_product(u_old_p, u_old_p)) -
                                         rho_old_m*(E_old_m - 0.5*Ma*Ma*scalar_product(u_old_m, u_old_m));

          phi_p.submit_value(-a21*dt*Ma*Ma*(scalar_product(avg_kinetic_old, n_plus) + 0.5*lambda_old*jump_rho_kin_old)
                             -a21_tilde*dt*(scalar_product(avg_enthalpy_old, n_plus) + 0.5*lambda_old*jump_rho_e_old), q);
          phi_m.submit_value(a21*dt*Ma*Ma*(scalar_product(avg_kinetic_old, n_plus) + 0.5*lambda_old*jump_rho_kin_old) +
                             a21_tilde*dt*(scalar_product(avg_enthalpy_old, n_plus) + 0.5*lambda_old*jump_rho_e_old), q);
        }
        phi_p.integrate_scatter(EvaluationFlags::values, dst);
        phi_m.integrate_scatter(EvaluationFlags::values, dst);
      }
    }
    else if(HYPERBOLIC_stage == 2) {
      /*--- We first start by declaring the suitable instances to read the available quantities. ---*/
      FEFaceEvaluation<dim, fe_degree_T, n_q_points_1d_u, 1, Number>   phi_p(data, true, 1),
                                                                       phi_m(data, false, 1),
                                                                       phi_pres_old_p(data, true, 1),
                                                                       phi_pres_old_m(data, false, 1),
                                                                       phi_pres_tmp_2_p(data, true, 1),
                                                                       phi_pres_tmp_2_m(data, false, 1);
      FEFaceEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number> phi_u_old_p(data, true, 0),
                                                                       phi_u_old_m(data, false, 0),
                                                                       phi_u_tmp_2_p(data, true, 0),
                                                                       phi_u_tmp_2_m(data, false, 0);
      FEFaceEvaluation<dim, fe_degree_rho, n_q_points_1d_u, 1, Number> phi_rho_old_p(data, true, 2),
                                                                       phi_rho_old_m(data, false, 2),
                                                                       phi_rho_tmp_2_p(data, true, 2),
                                                                       phi_rho_tmp_2_m(data, false, 2);

      /*--- Loop over all internal faces ---*/
      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        phi_rho_old_p.reinit(face);
        phi_rho_old_p.gather_evaluate(src[0], EvaluationFlags::values);
        phi_rho_old_m.reinit(face);
        phi_rho_old_m.gather_evaluate(src[0], EvaluationFlags::values);
        phi_u_old_p.reinit(face);
        phi_u_old_p.gather_evaluate(src[1], EvaluationFlags::values);
        phi_u_old_m.reinit(face);
        phi_u_old_m.gather_evaluate(src[1], EvaluationFlags::values);
        phi_pres_old_p.reinit(face);
        phi_pres_old_p.gather_evaluate(src[2], EvaluationFlags::values);
        phi_pres_old_m.reinit(face);
        phi_pres_old_m.gather_evaluate(src[2], EvaluationFlags::values);

        phi_rho_tmp_2_p.reinit(face);
        phi_rho_tmp_2_p.gather_evaluate(src[3], EvaluationFlags::values);
        phi_rho_tmp_2_m.reinit(face);
        phi_rho_tmp_2_m.gather_evaluate(src[3], EvaluationFlags::values);
        phi_u_tmp_2_p.reinit(face);
        phi_u_tmp_2_p.gather_evaluate(src[4], EvaluationFlags::values);
        phi_u_tmp_2_m.reinit(face);
        phi_u_tmp_2_m.gather_evaluate(src[4], EvaluationFlags::values);
        phi_pres_tmp_2_p.reinit(face);
        phi_pres_tmp_2_p.gather_evaluate(src[5], EvaluationFlags::values);
        phi_pres_tmp_2_m.reinit(face);
        phi_pres_tmp_2_m.gather_evaluate(src[5], EvaluationFlags::values);

        phi_p.reinit(face);
        phi_m.reinit(face);

        /*--- Loop over all quadrature points ---*/
        for(unsigned int q = 0; q < phi_p.n_q_points; ++q) {
          const auto& n_plus             = phi_p.get_normal_vector(q);

          const auto& rho_old_p          = phi_rho_old_p.get_value(q);
          const auto& rho_old_m          = phi_rho_old_m.get_value(q);
          const auto& u_old_p            = phi_u_old_p.get_value(q);
          const auto& u_old_m            = phi_u_old_m.get_value(q);
          const auto& avg_kinetic_old    = 0.5*(0.5*scalar_product(u_old_p, u_old_p)*rho_old_p*u_old_p +
                                                0.5*scalar_product(u_old_m, u_old_m)*rho_old_m*u_old_m);

          const auto& pres_old_p         = phi_pres_old_p.get_value(q);
          const auto& pres_old_m         = phi_pres_old_m.get_value(q);
          const auto& E_old_p            = 1.0/(EquationData::Cp_Cv - 1.0)*pres_old_p/rho_old_p
                                         + 0.5*Ma*Ma*scalar_product(u_old_p, u_old_p);
          const auto& E_old_m            = 1.0/(EquationData::Cp_Cv - 1.0)*pres_old_m/rho_old_m
                                         + 0.5*Ma*Ma*scalar_product(u_old_m, u_old_m);
          const auto& avg_enthalpy_old   = 0.5*
                                           (((E_old_p - 0.5*Ma*Ma*scalar_product(u_old_p, u_old_p))*rho_old_p + pres_old_p)*u_old_p +
                                            ((E_old_m - 0.5*Ma*Ma*scalar_product(u_old_m, u_old_m))*rho_old_m + pres_old_m)*u_old_m);

          const auto& lambda_old         = std::max(std::abs(scalar_product(u_old_p, n_plus)),
                                                    std::abs(scalar_product(u_old_m, n_plus)));
          const auto& jump_rho_kin_old   = rho_old_p*0.5*scalar_product(u_old_p, u_old_p) -
                                           rho_old_m*0.5*scalar_product(u_old_m, u_old_m);
          const auto& jump_rho_e_old     = rho_old_p*(E_old_p - 0.5*Ma*Ma*scalar_product(u_old_p, u_old_p)) -
                                           rho_old_m*(E_old_m - 0.5*Ma*Ma*scalar_product(u_old_m, u_old_m));

          const auto& rho_tmp_2_p        = phi_rho_tmp_2_p.get_value(q);
          const auto& rho_tmp_2_m        = phi_rho_tmp_2_m.get_value(q);
          const auto& u_tmp_2_p          = phi_u_tmp_2_p.get_value(q);
          const auto& u_tmp_2_m          = phi_u_tmp_2_m.get_value(q);
          const auto& avg_kinetic_tmp_2  = 0.5*(0.5*scalar_product(u_tmp_2_p, u_tmp_2_p)*rho_tmp_2_p*u_tmp_2_p +
                                                0.5*scalar_product(u_tmp_2_m, u_tmp_2_m)*rho_tmp_2_m*u_tmp_2_m);

          const auto& pres_tmp_2_p       = phi_pres_tmp_2_p.get_value(q);
          const auto& pres_tmp_2_m       = phi_pres_tmp_2_m.get_value(q);
          const auto& E_tmp_2_p          = 1.0/(EquationData::Cp_Cv - 1.0)*pres_tmp_2_p/rho_tmp_2_p
                                         + 0.5*Ma*Ma*scalar_product(u_tmp_2_p, u_tmp_2_p);
          const auto& E_tmp_2_m          = 1.0/(EquationData::Cp_Cv - 1.0)*pres_tmp_2_m/rho_tmp_2_m
                                         + 0.5*Ma*Ma*scalar_product(u_tmp_2_m, u_tmp_2_m);
          const auto& avg_enthalpy_tmp_2 = 0.5*
                                           (((E_tmp_2_p - 0.5*Ma*Ma*scalar_product(u_tmp_2_p, u_tmp_2_p))*rho_tmp_2_p +
                                              pres_tmp_2_p)*u_tmp_2_p +
                                            ((E_tmp_2_m - 0.5*Ma*Ma*scalar_product(u_tmp_2_m, u_tmp_2_m))*rho_tmp_2_m +
                                              pres_tmp_2_m)*u_tmp_2_m);

          const auto& lambda_tmp_2       = std::max(std::abs(scalar_product(u_tmp_2_p, n_plus)),
                                                    std::abs(scalar_product(u_tmp_2_m, n_plus)));
          const auto& jump_rho_kin_tmp_2 = rho_tmp_2_p*0.5*scalar_product(u_tmp_2_p, u_old_p) -
                                           rho_tmp_2_m*0.5*scalar_product(u_tmp_2_m, u_old_m);
          const auto& jump_rho_e_tmp_2   = rho_tmp_2_p*(E_tmp_2_p - 0.5*Ma*Ma*scalar_product(u_tmp_2_p, u_tmp_2_p)) -
                                           rho_tmp_2_m*(E_tmp_2_m - 0.5*Ma*Ma*scalar_product(u_tmp_2_m, u_tmp_2_m));

          phi_p.submit_value(-a31*dt*Ma*Ma*(scalar_product(avg_kinetic_old, n_plus) + 0.5*lambda_old*jump_rho_kin_old)
                             -a31_tilde*dt*(scalar_product(avg_enthalpy_old, n_plus) + 0.5*lambda_old*jump_rho_e_old)
                             -a32*dt*Ma*Ma*(scalar_product(avg_kinetic_tmp_2, n_plus) + 0.5*lambda_tmp_2*jump_rho_kin_tmp_2)
                             -a32_tilde*dt*(scalar_product(avg_enthalpy_tmp_2, n_plus) + 0.5*lambda_tmp_2*jump_rho_e_tmp_2), q);
          phi_m.submit_value(a31*dt*Ma*Ma*(scalar_product(avg_kinetic_old, n_plus) + 0.5*lambda_old*jump_rho_kin_old) +
                             a31_tilde*dt*(scalar_product(avg_enthalpy_old, n_plus) + 0.5*lambda_old*jump_rho_e_old) +
                             a32*dt*Ma*Ma*(scalar_product(avg_kinetic_tmp_2, n_plus) + 0.5*lambda_tmp_2*jump_rho_kin_tmp_2) +
                             a32_tilde*dt*(scalar_product(avg_enthalpy_tmp_2, n_plus) + 0.5*lambda_tmp_2*jump_rho_e_tmp_2), q);
        }
        phi_p.integrate_scatter(EvaluationFlags::values, dst);
        phi_m.integrate_scatter(EvaluationFlags::values, dst);
      }
    }
    else {
      /*--- We first start by declaring the suitable instances to read the available quantities. ---*/
      FEFaceEvaluation<dim, fe_degree_T, n_q_points_1d_u, 1, Number>   phi_p(data, true, 1),
                                                                       phi_m(data, false, 1),
                                                                       phi_pres_old_p(data, true, 1),
                                                                       phi_pres_old_m(data, false, 1),
                                                                       phi_pres_tmp_2_p(data, true, 1),
                                                                       phi_pres_tmp_2_m(data, false, 1),
                                                                       phi_pres_tmp_3_p(data, true, 1),
                                                                       phi_pres_tmp_3_m(data, false, 1);
      FEFaceEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number> phi_u_old_p(data, true, 0),
                                                                       phi_u_old_m(data, false, 0),
                                                                       phi_u_tmp_2_p(data, true, 0),
                                                                       phi_u_tmp_2_m(data, false, 0),
                                                                       phi_u_tmp_3_p(data, true, 0),
                                                                       phi_u_tmp_3_m(data, false, 0);
      FEFaceEvaluation<dim, fe_degree_rho, n_q_points_1d_u, 1, Number> phi_rho_old_p(data, true, 2),
                                                                       phi_rho_old_m(data, false, 2),
                                                                       phi_rho_tmp_2_p(data, true, 2),
                                                                       phi_rho_tmp_2_m(data, false, 2),
                                                                       phi_rho_tmp_3_p(data, true, 2),
                                                                       phi_rho_tmp_3_m(data, false, 2);

      /*--- loop over all internal faces ---*/
      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        phi_rho_old_p.reinit(face);
        phi_rho_old_p.gather_evaluate(src[0], EvaluationFlags::values);
        phi_rho_old_m.reinit(face);
        phi_rho_old_m.gather_evaluate(src[0], EvaluationFlags::values);
        phi_u_old_p.reinit(face);
        phi_u_old_p.gather_evaluate(src[1], EvaluationFlags::values);
        phi_u_old_m.reinit(face);
        phi_u_old_m.gather_evaluate(src[1], EvaluationFlags::values);
        phi_pres_old_p.reinit(face);
        phi_pres_old_p.gather_evaluate(src[2], EvaluationFlags::values);
        phi_pres_old_m.reinit(face);
        phi_pres_old_m.gather_evaluate(src[2], EvaluationFlags::values);

        phi_rho_tmp_2_p.reinit(face);
        phi_rho_tmp_2_p.gather_evaluate(src[3], EvaluationFlags::values);
        phi_rho_tmp_2_m.reinit(face);
        phi_rho_tmp_2_m.gather_evaluate(src[3], EvaluationFlags::values);
        phi_u_tmp_2_p.reinit(face);
        phi_u_tmp_2_p.gather_evaluate(src[4], EvaluationFlags::values);
        phi_u_tmp_2_m.reinit(face);
        phi_u_tmp_2_m.gather_evaluate(src[4], EvaluationFlags::values);
        phi_pres_tmp_2_p.reinit(face);
        phi_pres_tmp_2_p.gather_evaluate(src[5], EvaluationFlags::values);
        phi_pres_tmp_2_m.reinit(face);
        phi_pres_tmp_2_m.gather_evaluate(src[5], EvaluationFlags::values);

        phi_rho_tmp_3_p.reinit(face);
        phi_rho_tmp_3_p.gather_evaluate(src[6], EvaluationFlags::values);
        phi_rho_tmp_3_m.reinit(face);
        phi_rho_tmp_3_m.gather_evaluate(src[6], EvaluationFlags::values);
        phi_u_tmp_3_p.reinit(face);
        phi_u_tmp_3_p.gather_evaluate(src[7], EvaluationFlags::values);
        phi_u_tmp_3_m.reinit(face);
        phi_u_tmp_3_m.gather_evaluate(src[7], EvaluationFlags::values);
        phi_pres_tmp_3_p.reinit(face);
        phi_pres_tmp_3_p.gather_evaluate(src[8], EvaluationFlags::values);
        phi_pres_tmp_3_m.reinit(face);
        phi_pres_tmp_3_m.gather_evaluate(src[8], EvaluationFlags::values);

        phi_p.reinit(face);
        phi_m.reinit(face);

        /*--- Loop over all quadrature points ---*/
        for(unsigned int q = 0; q < phi_p.n_q_points; ++q) {
          const auto& n_plus             = phi_p.get_normal_vector(q);

          const auto& rho_old_p          = phi_rho_old_p.get_value(q);
          const auto& rho_old_m          = phi_rho_old_m.get_value(q);
          const auto& u_old_p            = phi_u_old_p.get_value(q);
          const auto& u_old_m            = phi_u_old_m.get_value(q);
          const auto& avg_kinetic_old    = 0.5*(0.5*scalar_product(u_old_p, u_old_p)*rho_old_p*u_old_p +
                                                0.5*scalar_product(u_old_m, u_old_m)*rho_old_m*u_old_m);

          const auto& pres_old_p         = phi_pres_old_p.get_value(q);
          const auto& pres_old_m         = phi_pres_old_m.get_value(q);
          const auto& E_old_p            = 1.0/(EquationData::Cp_Cv - 1.0)*pres_old_p/rho_old_p
                                         + 0.5*Ma*Ma*scalar_product(u_old_p, u_old_p);
          const auto& E_old_m            = 1.0/(EquationData::Cp_Cv - 1.0)*pres_old_m/rho_old_m
                                         + 0.5*Ma*Ma*scalar_product(u_old_m, u_old_m);
          const auto& avg_enthalpy_old   = 0.5*
                                           (((E_old_p - 0.5*Ma*Ma*scalar_product(u_old_p, u_old_p))*rho_old_p + pres_old_p)*u_old_p +
                                            ((E_old_m - 0.5*Ma*Ma*scalar_product(u_old_m, u_old_m))*rho_old_m + pres_old_m)*u_old_m);

          const auto& lambda_old         = std::max(std::abs(scalar_product(u_old_p, n_plus)),
                                                    std::abs(scalar_product(u_old_m, n_plus)));
          const auto& jump_rhoE_old      = rho_old_p*E_old_p - rho_old_m*E_old_m;

          const auto& rho_tmp_2_p        = phi_rho_tmp_2_p.get_value(q);
          const auto& rho_tmp_2_m        = phi_rho_tmp_2_m.get_value(q);
          const auto& u_tmp_2_p          = phi_u_tmp_2_p.get_value(q);
          const auto& u_tmp_2_m          = phi_u_tmp_2_m.get_value(q);
          const auto& avg_kinetic_tmp_2  = 0.5*(0.5*scalar_product(u_tmp_2_p, u_tmp_2_p)*rho_tmp_2_p*u_tmp_2_p +
                                                0.5*scalar_product(u_tmp_2_m, u_tmp_2_m)*rho_tmp_2_m*u_tmp_2_m);

          const auto& pres_tmp_2_p       = phi_pres_tmp_2_p.get_value(q);
          const auto& pres_tmp_2_m       = phi_pres_tmp_2_m.get_value(q);
          const auto& E_tmp_2_p          = 1.0/(EquationData::Cp_Cv - 1.0)*pres_tmp_2_p/rho_tmp_2_p
                                         + 0.5*Ma*Ma*scalar_product(u_tmp_2_p, u_tmp_2_p);
          const auto& E_tmp_2_m          = 1.0/(EquationData::Cp_Cv - 1.0)*pres_tmp_2_m/rho_tmp_2_m
                                         + 0.5*Ma*Ma*scalar_product(u_tmp_2_m, u_tmp_2_m);
          const auto& avg_enthalpy_tmp_2 = 0.5*
                                           (((E_tmp_2_p - 0.5*Ma*Ma*scalar_product(u_tmp_2_p, u_tmp_2_p))*rho_tmp_2_p +
                                              pres_tmp_2_p)*u_tmp_2_p +
                                            ((E_tmp_2_m - 0.5*Ma*Ma*scalar_product(u_tmp_2_m, u_tmp_2_m))*rho_tmp_2_m +
                                              pres_tmp_2_m)*u_tmp_2_m);

          const auto& lambda_tmp_2       = std::max(std::abs(scalar_product(u_tmp_2_p, n_plus)),
                                                    std::abs(scalar_product(u_tmp_2_m, n_plus)));
          const auto& jump_rhoE_tmp_2    = rho_tmp_2_p*E_tmp_2_p - rho_tmp_2_m*E_tmp_2_m;

          const auto& rho_tmp_3_p        = phi_rho_tmp_3_p.get_value(q);
          const auto& rho_tmp_3_m        = phi_rho_tmp_3_m.get_value(q);
          const auto& u_tmp_3_p          = phi_u_tmp_3_p.get_value(q);
          const auto& u_tmp_3_m          = phi_u_tmp_3_m.get_value(q);
          const auto& avg_kinetic_tmp_3  = 0.5*(0.5*scalar_product(u_tmp_3_p, u_tmp_3_p)*rho_tmp_3_p*u_tmp_3_p +
                                                0.5*scalar_product(u_tmp_3_m, u_tmp_3_m)*rho_tmp_3_m*u_tmp_3_m);

          const auto& pres_tmp_3_p       = phi_pres_tmp_3_p.get_value(q);
          const auto& pres_tmp_3_m       = phi_pres_tmp_3_m.get_value(q);
          const auto& E_tmp_3_p          = 1.0/(EquationData::Cp_Cv - 1.0)*pres_tmp_3_p/rho_tmp_3_p
                                         + 0.5*Ma*Ma*scalar_product(u_tmp_3_p, u_tmp_3_p);
          const auto& E_tmp_3_m          = 1.0/(EquationData::Cp_Cv - 1.0)*pres_tmp_3_m/rho_tmp_3_m
                                         + 0.5*Ma*Ma*scalar_product(u_tmp_3_m, u_tmp_3_m);
          const auto& avg_enthalpy_tmp_3 = 0.5*
                                           (((E_tmp_3_p - 0.5*Ma*Ma*scalar_product(u_tmp_3_p, u_tmp_3_p))*rho_tmp_3_p +
                                              pres_tmp_3_p)*u_tmp_3_p +
                                            ((E_tmp_3_m - 0.5*Ma*Ma*scalar_product(u_tmp_3_m, u_tmp_3_m))*rho_tmp_3_m +
                                              pres_tmp_3_m)*u_tmp_3_m);

          const auto& lambda_tmp_3       = std::max(std::abs(scalar_product(u_tmp_3_p, n_plus)),
                                                    std::abs(scalar_product(u_tmp_3_m, n_plus)));
          const auto& jump_rhoE_tmp_3    = rho_tmp_3_p*E_tmp_3_p - rho_tmp_3_m*E_tmp_3_m;

          phi_p.submit_value(-b1*dt*Ma*Ma*scalar_product(avg_kinetic_old, n_plus)
                             -b1*dt*scalar_product(avg_enthalpy_old, n_plus)
                             -b1*dt*0.5*lambda_old*jump_rhoE_old
                             -b2*dt*Ma*Ma*scalar_product(avg_kinetic_tmp_2, n_plus)
                             -b2*dt*scalar_product(avg_enthalpy_tmp_2, n_plus)
                             -b2*dt*0.5*lambda_tmp_2*jump_rhoE_tmp_2
                             -b3*dt*Ma*Ma*scalar_product(avg_kinetic_tmp_3, n_plus)
                             -b3*dt*scalar_product(avg_enthalpy_tmp_3, n_plus)
                             -b3*dt*0.5*lambda_tmp_3*jump_rhoE_tmp_3, q);
          phi_m.submit_value(b1*dt*Ma*Ma*scalar_product(avg_kinetic_old, n_plus) +
                             b1*dt*scalar_product(avg_enthalpy_old, n_plus) +
                             b1*dt*0.5*lambda_old*jump_rhoE_old +
                             b2*dt*Ma*Ma*scalar_product(avg_kinetic_tmp_2, n_plus) +
                             b2*dt*scalar_product(avg_enthalpy_tmp_2, n_plus) +
                             b2*dt*0.5*lambda_tmp_2*jump_rhoE_tmp_2 +
                             b3*dt*Ma*Ma*scalar_product(avg_kinetic_tmp_3, n_plus) +
                             b3*dt*scalar_product(avg_enthalpy_tmp_3, n_plus) +
                             b3*dt*0.5*lambda_tmp_3*jump_rhoE_tmp_3, q);
        }
        phi_p.integrate_scatter(EvaluationFlags::values, dst);
        phi_m.integrate_scatter(EvaluationFlags::values, dst);
      }
    }
  }


  // Assemble rhs boundary term for the pressure
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void EULEROperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  assemble_rhs_boundary_term_pressure(const MatrixFree<dim, Number>&               data,
                                      Vec&                                         dst,
                                      const std::vector<Vec>&                      src,
                                      const std::pair<unsigned int, unsigned int>& face_range) const {
    if(HYPERBOLIC_stage == 1) {
      /*--- We first start by declaring the suitable instances to read the available quantities. ---*/
      FEFaceEvaluation<dim, fe_degree_T, n_q_points_1d_u, 1, Number>   phi(data, true, 1),
                                                                       phi_pres_old(data, true, 1);
      FEFaceEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number> phi_u_old(data, true, 0);
      FEFaceEvaluation<dim, fe_degree_rho, n_q_points_1d_u, 1, Number> phi_rho_old(data, true, 2);

      /*--- Loop over all internal faces ---*/
      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        phi_rho_old.reinit(face);
        phi_rho_old.gather_evaluate(src[0], EvaluationFlags::values);
        phi_u_old.reinit(face);
        phi_u_old.gather_evaluate(src[1], EvaluationFlags::values);
        phi_pres_old.reinit(face);
        phi_pres_old.gather_evaluate(src[2], EvaluationFlags::values);

        phi.reinit(face);

        /*--- Loop over quadrature points ---*/
        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          const auto& n_plus           = phi.get_normal_vector(q);

          const auto& rho_old          = phi_rho_old.get_value(q);
          const auto& u_old            = phi_u_old.get_value(q);
          auto rho_old_D               = VectorizedArray<Number>();
          auto u_old_D                 = Tensor<1, dim, VectorizedArray<Number>>();
          const auto& point_vectorized = phi.quadrature_point(q);
          for(unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v) {
            Point<dim> point; /*--- The point returned by the 'quadrature_point' function is not an instance of Point
                                    and so it is not ready to be directly used. We need to pay attention to the
                                    vectorization ---*/
            for(unsigned int d = 0; d < dim; ++d) {
              point[d] = point_vectorized[d][v];
            }
            rho_old_D = rho_exact.value(point);
            for(unsigned int d = 0; d < dim; ++d) {
              u_old_D[d][v] = u_exact.value(point, d);
            }
          }
          const auto& avg_kinetic_old  = 0.5*(0.5*scalar_product(u_old, u_old)*rho_old*u_old +
                                              0.5*scalar_product(u_old_D, u_old_D)*rho_old_D*u_old_D);

          const auto& pres_old         = phi_pres_old.get_value(q);
          auto pres_old_D              = VectorizedArray<Number>();
          for(unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v) {
            Point<dim> point;
            for(unsigned int d = 0; d < dim; ++d) {
              point[d] = point_vectorized[d][v];
            }
            pres_old_D = pres_exact.value(point);
          }
          const auto& E_old            = 1.0/(EquationData::Cp_Cv - 1.0)*pres_old/rho_old
                                       + 0.5*Ma*Ma*scalar_product(u_old, u_old);
          const auto& E_old_D          = 1.0/(EquationData::Cp_Cv - 1.0)*pres_old_D/rho_old_D
                                       + 0.5*Ma*Ma*scalar_product(u_old_D, u_old_D);
          const auto& avg_enthalpy_old = 0.5*(((E_old - 0.5*Ma*Ma*scalar_product(u_old, u_old))*rho_old + pres_old)*u_old +
                                              ((E_old_D - 0.5*Ma*Ma*scalar_product(u_old_D, u_old_D))*rho_old_D + pres_old_D)*u_old_D);

          const auto& lambda_old       = std::max(std::abs(scalar_product(u_old, n_plus)),
                                                  std::abs(scalar_product(u_old_D, n_plus)));
          const auto& jump_rho_kin_old = rho_old*0.5*scalar_product(u_old, u_old) -
                                         rho_old_D*0.5*scalar_product(u_old_D, u_old_D);
          const auto& jump_rho_e_old   = rho_old*(E_old - 0.5*Ma*Ma*scalar_product(u_old, u_old)) -
                                         rho_old_D*(E_old_D - 0.5*Ma*Ma*scalar_product(u_old_D, u_old_D));

          phi.submit_value(-a21*dt*Ma*Ma*(scalar_product(avg_kinetic_old, n_plus) + 0.5*lambda_old*jump_rho_kin_old)
                           -a21_tilde*dt*(scalar_product(avg_enthalpy_old, n_plus) + 0.5*lambda_old*jump_rho_e_old), q);
        }
        phi.integrate_scatter(EvaluationFlags::values, dst);
      }
    }
    else if(HYPERBOLIC_stage == 2) {
      /*--- We first start by declaring the suitable instances to read the available quantities. ---*/
      FEFaceEvaluation<dim, fe_degree_T, n_q_points_1d_u, 1, Number>   phi(data, true, 1),
                                                                       phi_pres_old(data, true, 1),
                                                                       phi_pres_tmp_2(data, true, 1);
      FEFaceEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number> phi_u_old(data, true, 0),
                                                                       phi_u_tmp_2(data, true, 0);
      FEFaceEvaluation<dim, fe_degree_rho, n_q_points_1d_u, 1, Number> phi_rho_old(data, true, 2),
                                                                       phi_rho_tmp_2(data, true, 2);

      /*--- Loop over all internal faces ---*/
      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        phi_rho_old.reinit(face);
        phi_rho_old.gather_evaluate(src[0], EvaluationFlags::values);
        phi_u_old.reinit(face);
        phi_u_old.gather_evaluate(src[1], EvaluationFlags::values);
        phi_pres_old.reinit(face);
        phi_pres_old.gather_evaluate(src[2], EvaluationFlags::values);

        phi_rho_tmp_2.reinit(face);
        phi_rho_tmp_2.gather_evaluate(src[3], EvaluationFlags::values);
        phi_u_tmp_2.reinit(face);
        phi_u_tmp_2.gather_evaluate(src[4], EvaluationFlags::values);
        phi_pres_tmp_2.reinit(face);
        phi_pres_tmp_2.gather_evaluate(src[5], EvaluationFlags::values);

        phi.reinit(face);

        /*--- Loop over all quadrature points ---*/
        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          const auto& n_plus             = phi.get_normal_vector(q);

          const auto& rho_old            = phi_rho_old.get_value(q);
          const auto& u_old              = phi_u_old.get_value(q);
          auto rho_old_D                 = VectorizedArray<Number>();
          auto u_old_D                   = Tensor<1, dim, VectorizedArray<Number>>();
          const auto& point_vectorized   = phi.quadrature_point(q);
          for(unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v) {
            Point<dim> point;
            for(unsigned int d = 0; d < dim; ++d) {
              point[d] = point_vectorized[d][v];
            }
            rho_old_D = rho_exact.value(point);
            for(unsigned int d = 0; d < dim; ++d) {
              u_old_D[d][v] = u_exact.value(point, d);
            }
          }
          const auto& avg_kinetic_old    = 0.5*(0.5*scalar_product(u_old, u_old)*rho_old*u_old +
                                                0.5*scalar_product(u_old_D, u_old_D)*rho_old_D*u_old_D);

          const auto& pres_old           = phi_pres_old.get_value(q);
          auto pres_old_D                = VectorizedArray<Number>();
          for(unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v) {
            Point<dim> point;
            for(unsigned int d = 0; d < dim; ++d) {
              point[d] = point_vectorized[d][v];
            }
            pres_old_D = pres_exact.value(point);
          }
          const auto& E_old              = 1.0/(EquationData::Cp_Cv - 1.0)*pres_old/rho_old
                                         + 0.5*Ma*Ma*scalar_product(u_old, u_old);
          const auto& E_old_D            = 1.0/(EquationData::Cp_Cv - 1.0)*pres_old_D/rho_old_D
                                         + 0.5*Ma*Ma*scalar_product(u_old_D, u_old_D);
          const auto& avg_enthalpy_old   = 0.5*(((E_old - 0.5*Ma*Ma*scalar_product(u_old, u_old))*rho_old + pres_old)*u_old +
                                                ((E_old_D - 0.5*Ma*Ma*scalar_product(u_old_D, u_old_D))*rho_old_D + pres_old_D)*u_old_D);

          const auto& lambda_old         = std::max(std::abs(scalar_product(u_old, n_plus)),
                                                    std::abs(scalar_product(u_old_D, n_plus)));
          const auto& jump_rho_kin_old   = rho_old*0.5*scalar_product(u_old, u_old) -
                                           rho_old_D*0.5*scalar_product(u_old_D, u_old_D);
          const auto& jump_rho_e_old     = rho_old*(E_old - 0.5*Ma*Ma*scalar_product(u_old, u_old)) -
                                           rho_old_D*(E_old_D - 0.5*Ma*Ma*scalar_product(u_old_D, u_old_D));

          const auto& rho_tmp_2          = phi_rho_tmp_2.get_value(q);
          const auto& u_tmp_2            = phi_u_tmp_2.get_value(q);
          auto rho_tmp_2_D               = VectorizedArray<Number>();
          auto u_tmp_2_D                 = Tensor<1, dim, VectorizedArray<Number>>();
          for(unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v) {
            Point<dim> point;
            for(unsigned int d = 0; d < dim; ++d) {
              point[d] = point_vectorized[d][v];
            }
            rho_tmp_2_D = rho_exact.value(point);
            for(unsigned int d = 0; d < dim; ++d) {
              u_tmp_2_D[d][v] = u_exact.value(point, d);
            }
          }
          const auto& avg_kinetic_tmp_2  = 0.5*(0.5*scalar_product(u_tmp_2, u_tmp_2)*rho_tmp_2*u_tmp_2 +
                                                0.5*scalar_product(u_tmp_2_D, u_tmp_2_D)*rho_tmp_2_D*u_tmp_2_D);

          const auto& pres_tmp_2         = phi_pres_tmp_2.get_value(q);
          auto pres_tmp_2_D              = VectorizedArray<Number>();
          for(unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v) {
            Point<dim> point;
            for(unsigned int d = 0; d < dim; ++d) {
              point[d] = point_vectorized[d][v];
            }
            pres_tmp_2_D = pres_exact.value(point);
          }
          const auto& E_tmp_2            = 1.0/(EquationData::Cp_Cv - 1.0)*pres_tmp_2/rho_tmp_2
                                         + 0.5*Ma*Ma*scalar_product(u_tmp_2, u_tmp_2);
          const auto& E_tmp_2_D          = 1.0/(EquationData::Cp_Cv - 1.0)*pres_tmp_2_D/rho_tmp_2_D
                                         + 0.5*Ma*Ma*scalar_product(u_tmp_2_D, u_tmp_2_D);
          const auto& avg_enthalpy_tmp_2 = 0.5*(((E_tmp_2 - 0.5*Ma*Ma*scalar_product(u_tmp_2, u_tmp_2))*rho_tmp_2 + pres_tmp_2)*
                                                u_tmp_2 +
                                                ((E_tmp_2_D - 0.5*Ma*Ma*scalar_product(u_tmp_2_D, u_tmp_2_D))*rho_tmp_2_D + pres_tmp_2_D)*
                                                u_tmp_2_D);

          const auto& lambda_tmp_2       = std::max(std::abs(scalar_product(u_tmp_2, n_plus)),
                                                    std::abs(scalar_product(u_tmp_2_D, n_plus)));
          const auto& jump_rho_kin_tmp_2 = rho_tmp_2*0.5*scalar_product(u_tmp_2, u_tmp_2) -
                                           rho_tmp_2_D*0.5*scalar_product(u_tmp_2_D, u_tmp_2_D);
          const auto& jump_rho_e_tmp_2   = rho_tmp_2*(E_tmp_2 - 0.5*Ma*Ma*scalar_product(u_tmp_2, u_tmp_2)) -
                                           rho_tmp_2_D*(E_tmp_2_D - 0.5*Ma*Ma*scalar_product(u_tmp_2_D, u_tmp_2_D));

          phi.submit_value(-a31*dt*Ma*Ma*(scalar_product(avg_kinetic_old, n_plus) + 0.5*lambda_old*jump_rho_kin_old)
                           -a31_tilde*dt*(scalar_product(avg_enthalpy_old, n_plus) + 0.5*lambda_old*jump_rho_e_old)
                           -a32*dt*Ma*Ma*(scalar_product(avg_kinetic_tmp_2, n_plus) + 0.5*lambda_tmp_2*jump_rho_kin_tmp_2)
                           -a32_tilde*dt*(scalar_product(avg_enthalpy_tmp_2, n_plus) + 0.5*lambda_tmp_2*jump_rho_e_tmp_2), q);
        }
        phi.integrate_scatter(EvaluationFlags::values, dst);
      }
    }
    else {
      /*--- We first start by declaring the suitable instances to read the available quantities. ---*/
      FEFaceEvaluation<dim, fe_degree_T, n_q_points_1d_u, 1, Number>   phi(data, true, 1),
                                                                       phi_pres_old(data, true, 1),
                                                                       phi_pres_tmp_2(data, true, 1),
                                                                       phi_pres_tmp_3(data, true, 1);
      FEFaceEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number> phi_u_old(data, true, 0),
                                                                       phi_u_tmp_2(data, true, 0),
                                                                       phi_u_tmp_3(data, true, 0);
      FEFaceEvaluation<dim, fe_degree_rho, n_q_points_1d_u, 1, Number> phi_rho_old(data, true, 2),
                                                                       phi_rho_tmp_2(data, true, 2),
                                                                       phi_rho_tmp_3(data, true, 2);

      /*--- loop over all internal faces ---*/
      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        phi_rho_old.reinit(face);
        phi_rho_old.gather_evaluate(src[0], EvaluationFlags::values);
        phi_u_old.reinit(face);
        phi_u_old.gather_evaluate(src[1], EvaluationFlags::values);
        phi_pres_old.reinit(face);
        phi_pres_old.gather_evaluate(src[2], EvaluationFlags::values);

        phi_rho_tmp_2.reinit(face);
        phi_rho_tmp_2.gather_evaluate(src[3], EvaluationFlags::values);
        phi_u_tmp_2.reinit(face);
        phi_u_tmp_2.gather_evaluate(src[4], EvaluationFlags::values);
        phi_pres_tmp_2.reinit(face);
        phi_pres_tmp_2.gather_evaluate(src[5], EvaluationFlags::values);

        phi_rho_tmp_3.reinit(face);
        phi_rho_tmp_3.gather_evaluate(src[6], EvaluationFlags::values);
        phi_u_tmp_3.reinit(face);
        phi_u_tmp_3.gather_evaluate(src[7], EvaluationFlags::values);
        phi_pres_tmp_3.reinit(face);
        phi_pres_tmp_3.gather_evaluate(src[8], EvaluationFlags::values);

        phi.reinit(face);

        /*--- Loop over all quadrature points ---*/
        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          const auto& n_plus             = phi.get_normal_vector(q);

          const auto& rho_old            = phi_rho_old.get_value(q);
          const auto& u_old              = phi_u_old.get_value(q);
          auto rho_old_D                 = VectorizedArray<Number>();
          auto u_old_D                   = Tensor<1, dim, VectorizedArray<Number>>();
          const auto& point_vectorized   = phi.quadrature_point(q);
          for(unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v) {
            Point<dim> point;
            for(unsigned int d = 0; d < dim; ++d) {
              point[d] = point_vectorized[d][v];
            }
            rho_old_D = rho_exact.value(point);
            for(unsigned int d = 0; d < dim; ++d) {
              u_old_D[d][v] = u_exact.value(point, d);
            }
          }
          const auto& avg_kinetic_old    = 0.5*(0.5*scalar_product(u_old, u_old)*rho_old*u_old +
                                                0.5*scalar_product(u_old_D, u_old_D)*rho_old_D*u_old_D);

          const auto& pres_old           = phi_pres_old.get_value(q);
          auto pres_old_D                = VectorizedArray<Number>();
          for(unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v) {
            Point<dim> point;
            for(unsigned int d = 0; d < dim; ++d) {
              point[d] = point_vectorized[d][v];
            }
            pres_old_D = pres_exact.value(point);
          }
          const auto& E_old              = 1.0/(EquationData::Cp_Cv - 1.0)*pres_old/rho_old
                                         + 0.5*Ma*Ma*scalar_product(u_old, u_old);
          const auto& E_old_D            = 1.0/(EquationData::Cp_Cv - 1.0)*pres_old_D/rho_old_D
                                         + 0.5*Ma*Ma*scalar_product(u_old_D, u_old_D);
          const auto& avg_enthalpy_old   = 0.5*(((E_old - 0.5*Ma*Ma*scalar_product(u_old, u_old))*rho_old + pres_old)*u_old +
                                                ((E_old_D - 0.5*Ma*Ma*scalar_product(u_old_D, u_old_D))*rho_old_D + pres_old_D)*u_old_D);

          const auto& lambda_old         = std::max(std::abs(scalar_product(u_old, n_plus)),
                                                    std::abs(scalar_product(u_old_D, n_plus)));
          const auto& jump_rhoE_old      = rho_old*E_old - rho_old_D*E_old_D;

          const auto& rho_tmp_2          = phi_rho_tmp_2.get_value(q);
          const auto& u_tmp_2            = phi_u_tmp_2.get_value(q);
          auto rho_tmp_2_D               = VectorizedArray<Number>();
          auto u_tmp_2_D                 = Tensor<1, dim, VectorizedArray<Number>>();
          for(unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v) {
            Point<dim> point;
            for(unsigned int d = 0; d < dim; ++d) {
              point[d] = point_vectorized[d][v];
            }
            rho_tmp_2_D = rho_exact.value(point);
            for(unsigned int d = 0; d < dim; ++d) {
              u_tmp_2_D[d][v] = u_exact.value(point, d);
            }
          }
          const auto& avg_kinetic_tmp_2  = 0.5*(0.5*scalar_product(u_tmp_2, u_tmp_2)*rho_tmp_2*u_tmp_2 +
                                                0.5*scalar_product(u_tmp_2_D, u_tmp_2_D)*rho_tmp_2_D*u_tmp_2_D);

          const auto& pres_tmp_2         = phi_pres_tmp_2.get_value(q);
          auto pres_tmp_2_D              = VectorizedArray<Number>();
          for(unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v) {
            Point<dim> point;
            for(unsigned int d = 0; d < dim; ++d) {
              point[d] = point_vectorized[d][v];
            }
            pres_tmp_2_D = pres_exact.value(point);
          }
          const auto& E_tmp_2            = 1.0/(EquationData::Cp_Cv - 1.0)*pres_tmp_2/rho_tmp_2
                                         + 0.5*Ma*Ma*scalar_product(u_tmp_2, u_tmp_2);
          const auto& E_tmp_2_D          = 1.0/(EquationData::Cp_Cv - 1.0)*pres_tmp_2_D/rho_tmp_2_D
                                         + 0.5*Ma*Ma*scalar_product(u_tmp_2_D, u_tmp_2_D);
          const auto& avg_enthalpy_tmp_2 = 0.5*(((E_tmp_2 - 0.5*Ma*Ma*scalar_product(u_tmp_2, u_tmp_2))*rho_tmp_2 + pres_tmp_2)*
                                                u_tmp_2 +
                                                ((E_tmp_2_D - 0.5*Ma*Ma*scalar_product(u_tmp_2_D, u_tmp_2_D))*rho_tmp_2_D + pres_tmp_2_D)*
                                                u_tmp_2_D);

          const auto& lambda_tmp_2       = std::max(std::abs(scalar_product(u_tmp_2, n_plus)),
                                                    std::abs(scalar_product(u_tmp_2_D, n_plus)));
          const auto& jump_rhoE_tmp_2    = rho_tmp_2*E_tmp_2 - rho_tmp_2_D*E_tmp_2_D;

          const auto& rho_tmp_3          = phi_rho_tmp_3.get_value(q);
          const auto& u_tmp_3            = phi_u_tmp_3.get_value(q);
          auto rho_tmp_3_D               = VectorizedArray<Number>();
          auto u_tmp_3_D                 = Tensor<1, dim, VectorizedArray<Number>>();
          for(unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v) {
            Point<dim> point;
            for(unsigned int d = 0; d < dim; ++d) {
              point[d] = point_vectorized[d][v];
            }
            rho_tmp_3_D = rho_exact.value(point);
            for(unsigned int d = 0; d < dim; ++d) {
              u_tmp_3_D[d][v] = u_exact.value(point, d);
            }
          }
          const auto& avg_kinetic_tmp_3  = 0.5*(0.5*scalar_product(u_tmp_3, u_tmp_3)*rho_tmp_3*u_tmp_3 +
                                                0.5*scalar_product(u_tmp_3_D, u_tmp_3_D)*rho_tmp_3_D*u_tmp_3_D);

          const auto& pres_tmp_3         = phi_pres_tmp_3.get_value(q);
          auto pres_tmp_3_D              = VectorizedArray<Number>();
          for(unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v) {
            Point<dim> point;
            for(unsigned int d = 0; d < dim; ++d) {
              point[d] = point_vectorized[d][v];
            }
            pres_tmp_3_D = pres_exact.value(point);
          }
          const auto& E_tmp_3            = 1.0/(EquationData::Cp_Cv - 1.0)*pres_tmp_3/rho_tmp_3
                                         + 0.5*Ma*Ma*scalar_product(u_tmp_3, u_tmp_3);
          const auto& E_tmp_3_D          = 1.0/(EquationData::Cp_Cv - 1.0)*pres_tmp_3_D/rho_tmp_3_D
                                         + 0.5*Ma*Ma*scalar_product(u_tmp_3_D, u_tmp_3_D);
          const auto& avg_enthalpy_tmp_3 = 0.5*(((E_tmp_3 - 0.5*Ma*Ma*scalar_product(u_tmp_3, u_tmp_3))*rho_tmp_3 + pres_tmp_3)*
                                                u_tmp_3 +
                                                ((E_tmp_3_D - 0.5*Ma*Ma*scalar_product(u_tmp_3_D, u_tmp_3_D))*rho_tmp_3_D + pres_tmp_3_D)*
                                                u_tmp_3_D);

          const auto& lambda_tmp_3       = std::max(std::abs(scalar_product(u_tmp_3, n_plus)),
                                                    std::abs(scalar_product(u_tmp_3_D, n_plus)));
          const auto& jump_rhoE_tmp_3    = rho_tmp_3*E_tmp_3 - rho_tmp_3_D*E_tmp_3_D;

          phi.submit_value(-b1*dt*Ma*Ma*scalar_product(avg_kinetic_old, n_plus)
                           -b1*dt*scalar_product(avg_enthalpy_old, n_plus)
                           -b1*dt*0.5*lambda_old*jump_rhoE_old
                           -b2*dt*Ma*Ma*scalar_product(avg_kinetic_tmp_2, n_plus)
                           -b2*dt*scalar_product(avg_enthalpy_tmp_2, n_plus)
                           -b2*dt*0.5*lambda_tmp_2*jump_rhoE_tmp_2
                           -b3*dt*Ma*Ma*scalar_product(avg_kinetic_tmp_3, n_plus)
                           -b3*dt*scalar_product(avg_enthalpy_tmp_3, n_plus)
                           -b3*dt*0.5*lambda_tmp_3*jump_rhoE_tmp_3, q);
        }
        phi.integrate_scatter(EvaluationFlags::values, dst);
      }
    }
  }


  // Put together all the previous steps for pressure
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void EULEROperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  vmult_rhs_pressure(Vec& dst, const std::vector<Vec>& src) const {
    for(unsigned int d = 0; d < src.size(); ++d) {
      src[d].update_ghost_values();
    }

    this->data->loop(&EULEROperator::assemble_rhs_cell_term_pressure,
                     &EULEROperator::assemble_rhs_face_term_pressure,
                     &EULEROperator::assemble_rhs_boundary_term_pressure,
                     this, dst, src, true,
                     MatrixFree<dim, Number>::DataAccessOnFaces::unspecified,
                     MatrixFree<dim, Number>::DataAccessOnFaces::unspecified);
  }


  // Assemble cell term for the contribution due to internal energy
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void EULEROperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  assemble_cell_term_internal_energy(const MatrixFree<dim, Number>&               data,
                                     Vec&                                         dst,
                                     const Vec&                                   src,
                                     const std::pair<unsigned int, unsigned int>& cell_range) const {
    FEEvaluation<dim, fe_degree_T, n_q_points_1d_u, 1, Number> phi(data, 1);

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
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void EULEROperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  assemble_cell_term_enthalpy(const MatrixFree<dim, Number>&               data,
                              Vec&                                         dst,
                              const Vec&                                   src,
                              const std::pair<unsigned int, unsigned int>& cell_range) const {
    /*--- We first start by declaring the suitable instances to read also available quantities.
          Since here we have just one 'src' vector, but we also need to deal with the current pressure
           in the fixed point loop, we employ the auxiliary vector 'pres_fixed' where we setted this information ---*/
    FEEvaluation<dim, fe_degree_T, n_q_points_1d_u, 1, Number>   phi(data, 1),
                                                                 phi_pres_fixed(data, 1);
    FEEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number> phi_src(data, 0);

    const double coeff = (HYPERBOLIC_stage == 1) ? a22_tilde : a33_tilde;

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

        phi.submit_gradient(-coeff*dt*EquationData::Cp_Cv/(EquationData::Cp_Cv - 1.0)*pres_fixed*phi_src.get_value(q), q);
      }
      phi.integrate_scatter(EvaluationFlags::gradients, dst);
    }
  }


  // Assemble face term for the contribution due to enthalpy
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void EULEROperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  assemble_face_term_enthalpy(const MatrixFree<dim, Number>&               data,
                              Vec&                                         dst,
                              const Vec&                                   src,
                              const std::pair<unsigned int, unsigned int>& face_range) const {
    FEFaceEvaluation<dim, fe_degree_T, n_q_points_1d_u, 1, Number>   phi_p(data, true, 1),
                                                                     phi_m(data, false, 1),
                                                                     phi_pres_fixed_p(data, true, 1),
                                                                     phi_pres_fixed_m(data, false, 1);
    FEFaceEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number> phi_src_p(data, true, 0),
                                                                     phi_src_m(data, false, 0);

    /*--- This term changes between second and third stage of the IMEX scheme, but its structure not, so we do not need
          to explicitly distinguish the two cases as done for the rhs. ---*/
    const double coeff = (HYPERBOLIC_stage == 1) ? a22_tilde : a33_tilde;

    /*--- Loop over all faces ---*/
    for(unsigned int face = face_range.first; face < face_range.second; ++face) {
      phi_pres_fixed_p.reinit(face);
      phi_pres_fixed_p.gather_evaluate(pres_fixed, EvaluationFlags::values);
      phi_pres_fixed_m.reinit(face);
      phi_pres_fixed_m.gather_evaluate(pres_fixed, EvaluationFlags::values);

      phi_src_p.reinit(face);
      phi_src_p.gather_evaluate(src, EvaluationFlags::values);
      phi_src_m.reinit(face);
      phi_src_m.gather_evaluate(src, EvaluationFlags::values);

      phi_p.reinit(face);
      phi_m.reinit(face);

      /*--- Loop over all quadrature points ---*/
      for(unsigned int q = 0; q < phi_p.n_q_points; ++q) {
        const auto& n_plus            = phi_p.get_normal_vector(q);

        const auto& pres_fixed_p      = phi_pres_fixed_p.get_value(q);
        const auto& pres_fixed_m      = phi_pres_fixed_m.get_value(q);

        const auto& avg_flux_enthalpy = 0.5*EquationData::Cp_Cv/(EquationData::Cp_Cv - 1.0)*
                                        (pres_fixed_p*phi_src_p.get_value(q) + pres_fixed_m*phi_src_m.get_value(q));

        const auto& lambda_fixed      = std::max(std::abs(scalar_product(phi_src_p.get_value(q), n_plus)),
                                                 std::abs(scalar_product(phi_src_m.get_value(q), n_plus)));
        const auto& jump_rho_e_fixed  = 1.0/(EquationData::Cp_Cv - 1.0)*(pres_fixed_p - pres_fixed_m);

        phi_p.submit_value(coeff*dt*(scalar_product(avg_flux_enthalpy, n_plus) + 0.5*lambda_fixed*jump_rho_e_fixed), q);
        phi_m.submit_value(-coeff*dt*(scalar_product(avg_flux_enthalpy, n_plus) + 0.5*lambda_fixed*jump_rho_e_fixed), q);
      }
      phi_p.integrate_scatter(EvaluationFlags::values, dst);
      phi_m.integrate_scatter(EvaluationFlags::values, dst);
    }
  }


  // Assemble boundary term for the contribution due to enthalpy
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void EULEROperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  assemble_boundary_term_enthalpy(const MatrixFree<dim, Number>&               data,
                                  Vec&                                         dst,
                                  const Vec&                                   src,
                                  const std::pair<unsigned int, unsigned int>& face_range) const {
    FEFaceEvaluation<dim, fe_degree_T, n_q_points_1d_u, 1, Number>   phi(data, true, 1),
                                                                     phi_pres_fixed(data, true, 1);
    FEFaceEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number> phi_src(data, true, 0);

    /*--- This term changes between second and third stage of the IMEX scheme, but its structure not, so we do not need
          to explicitly distinguish the two cases as done for the rhs. ---*/
    const double coeff = (HYPERBOLIC_stage == 1) ? a22_tilde : a33_tilde;

    /*--- Loop over all faces ---*/
    for(unsigned int face = face_range.first; face < face_range.second; ++face) {
      phi_pres_fixed.reinit(face);
      phi_pres_fixed.gather_evaluate(pres_fixed, EvaluationFlags::values);

      phi_src.reinit(face);
      phi_src.gather_evaluate(src, EvaluationFlags::values);

      phi.reinit(face);

      /*--- Loop over all quadrature points ---*/
      for(unsigned int q = 0; q < phi.n_q_points; ++q) {
        const auto& n_plus            = phi.get_normal_vector(q);

        const auto& pres_fixed        = phi_pres_fixed.get_value(q);
        auto pres_fixed_D             = VectorizedArray<Number>();
        auto u_fixed_D                = Tensor<1, dim, VectorizedArray<Number>>();
        const auto& point_vectorized  = phi.quadrature_point(q);
        for(unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v) {
          Point<dim> point; /*--- The point returned by the 'quadrature_point' function is not an instance of Point
                                  and so it is not ready to be directly used. We need to pay attention to the
                                  vectorization ---*/
          for(unsigned int d = 0; d < dim; ++d) {
            point[d] = point_vectorized[d][v];
          }
          pres_fixed_D = pres_exact.value(point);
          for(unsigned int d = 0; d < dim; ++d) {
            u_fixed_D[d][v] = u_exact.value(point, d);
          }
        }

        const auto& avg_flux_enthalpy = 0.5*EquationData::Cp_Cv/(EquationData::Cp_Cv - 1.0)*
                                        (pres_fixed*phi_src.get_value(q) + pres_fixed_D*u_fixed_D);

        const auto& lambda_fixed      = std::max(std::abs(scalar_product(phi_src.get_value(q), n_plus)),
                                                 std::abs(scalar_product(u_fixed_D, n_plus)));
        const auto& jump_rho_e_fixed  = 1.0/(EquationData::Cp_Cv - 1.0)*(pres_fixed - pres_fixed_D);

        phi.submit_value(coeff*dt*(scalar_product(avg_flux_enthalpy, n_plus) + 0.5*lambda_fixed*jump_rho_e_fixed), q);
      }
      phi.integrate_scatter(EvaluationFlags::values, dst);
    }
  }


  // Assemble cell term for the pressure
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void EULEROperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  assemble_cell_term_pressure(const MatrixFree<dim, Number>&               data,
                              Vec&                                         dst,
                              const Vec&                                   src,
                              const std::pair<unsigned int, unsigned int>& cell_range) const {
    /*--- We first start by declaring the suitable instances to read quantities. This operator we are going to implement
          represents a rectangular matrix (we start from the pressure FE space and we end up with the velocity FE space).
          This is the reason of the distinction between 'phi' and 'phi_src'. ---*/
    FEEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number> phi(data, 0);
    FEEvaluation<dim, fe_degree_T, n_q_points_1d_u, 1, Number>   phi_src(data, 1);

    const double coeff = (HYPERBOLIC_stage == 1) ? a22_tilde : a33_tilde;

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
      phi_src.reinit(cell);
      phi_src.gather_evaluate(src, EvaluationFlags::values);

      phi.reinit(cell);

      for(unsigned int q = 0; q < phi.n_q_points; ++q) {
        /*--- Here we are testing against the divergence of the test function and, therefore, we employ 'submit_divergence'. ---*/
        phi.submit_divergence(-coeff*dt/(Ma*Ma)*phi_src.get_value(q), q);
      }
      phi.integrate_scatter(EvaluationFlags::gradients, dst);
    }
  }


  // Assemble face term for the pressure
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void EULEROperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  assemble_face_term_pressure(const MatrixFree<dim, Number>&               data,
                              Vec&                                         dst,
                              const Vec&                                   src,
                              const std::pair<unsigned int, unsigned int>& face_range) const {
    FEFaceEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number> phi_p(data, true, 0),
                                                                     phi_m(data, false, 0);
    FEFaceEvaluation<dim, fe_degree_T, n_q_points_1d_u, 1, Number>   phi_src_p(data, true, 1),
                                                                     phi_src_m(data, false, 1);

    const double coeff = (HYPERBOLIC_stage == 1) ? a22_tilde : a33_tilde;

    /*--- Loop over all internal faces ---*/
    for(unsigned int face = face_range.first; face < face_range.second; ++face) {
      phi_src_p.reinit(face);
      phi_src_p.gather_evaluate(src, EvaluationFlags::values);
      phi_src_m.reinit(face);
      phi_src_m.gather_evaluate(src, EvaluationFlags::values);

      phi_p.reinit(face);
      phi_m.reinit(face);

      /*--- Loop over all quadrature points ---*/
      for(unsigned int q = 0; q < phi_p.n_q_points; ++q) {
        const auto& n_plus   = phi_p.get_normal_vector(q);

        const auto& avg_term = 0.5*(phi_src_p.get_value(q) + phi_src_m.get_value(q));

        phi_p.submit_value(coeff*dt/(Ma*Ma)*avg_term*n_plus, q);
        phi_m.submit_value(-coeff*dt/(Ma*Ma)*avg_term*n_plus, q);
      }
      phi_p.integrate_scatter(EvaluationFlags::values, dst);
      phi_m.integrate_scatter(EvaluationFlags::values, dst);
    }
  }


  // Assemble boundary term for the pressure
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void EULEROperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  assemble_boundary_term_pressure(const MatrixFree<dim, Number>&               data,
                                  Vec&                                         dst,
                                  const Vec&                                   src,
                                  const std::pair<unsigned int, unsigned int>& face_range) const {
    FEFaceEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number> phi(data, true, 0);
    FEFaceEvaluation<dim, fe_degree_T, n_q_points_1d_u, 1, Number>   phi_src(data, true, 1);

    const double coeff = (HYPERBOLIC_stage == 1) ? a22_tilde : a33_tilde;

    /*--- Loop over all boundary faces ---*/
    for(unsigned int face = face_range.first; face < face_range.second; ++face) {
      phi_src.reinit(face);
      phi_src.gather_evaluate(src, EvaluationFlags::values | EvaluationFlags::gradients);

      phi.reinit(face);

      /*--- Loop over all quadrature points ---*/
      for(unsigned int q = 0; q < phi.n_q_points; ++q) {
        const auto& n_plus            = phi.get_normal_vector(q);

        auto pres_fixed_D             = VectorizedArray<Number>();
        const auto& point_vectorized  = phi.quadrature_point(q);
        for(unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v) {
          Point<dim> point; /*--- The point returned by the 'quadrature_point' function is not an instance of Point
                                  and so it is not ready to be directly used. We need to pay attention to the
                                  vectorization ---*/
          for(unsigned int d = 0; d < dim; ++d) {
            point[d] = point_vectorized[d][v];
          }
          pres_fixed_D = pres_exact.value(point);
        }

        phi.submit_value(coeff*dt/(Ma*Ma)*0.5*(phi_src.get_value(q) + pres_fixed_D)*n_plus, q);
      }
      phi.integrate_scatter(EvaluationFlags::values, dst);
    }
  }


  // Assemble rhs cell term for the velocity update
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void EULEROperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  assemble_rhs_cell_term_velocity_fixed(const MatrixFree<dim, Number>&               data,
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

    if(HYPERBOLIC_stage == 1) {
      /*--- We first start by declaring the suitable instances to read the available quantities. ---*/
      FEEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number> phi(data, 0),
                                                                   phi_u_old(data, 0);
      FEEvaluation<dim, fe_degree_T, n_q_points_1d_u, 1, Number>   phi_pres_old(data, 1);
      FEEvaluation<dim, fe_degree_rho, n_q_points_1d_u, 1, Number> phi_rho_old(data, 2),
                                                                   phi_rho_tmp_2(data, 2);

      /*--- Loop over all cells ---*/
      for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
        phi_rho_old.reinit(cell);
        phi_rho_old.gather_evaluate(src[0], EvaluationFlags::values | EvaluationFlags::gradients);
        phi_u_old.reinit(cell);
        phi_u_old.gather_evaluate(src[1], EvaluationFlags::values | EvaluationFlags::gradients);
        phi_pres_old.reinit(cell);
        phi_pres_old.gather_evaluate(src[2], EvaluationFlags::values);

        phi_rho_tmp_2.reinit(cell);
        phi_rho_tmp_2.gather_evaluate(src[3], EvaluationFlags::values | EvaluationFlags::gradients);

        phi.reinit(cell);

        /*--- Loop over all quadrature points ---*/
        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          const auto& rho_old            = phi_rho_old.get_value(q);
          const auto& u_old              = phi_u_old.get_value(q);
          const auto& pres_old           = phi_pres_old.get_value(q);

          const auto& tensor_product_u_n = outer_product(u_old, u_old);
          /*--- For the sake of compatibility, since after integration by parts, the pressure gradient
                would be tested against the divergence of the test function. This is equaivalent to test a diagonal matrix
                with diagonal entries equal to the pressure itself agains the gradient of the test function. ---*/
          auto p_n_times_identity        = tensor_product_u_n;
          p_n_times_identity = 0;
          for(unsigned int d = 0; d < dim; ++d) {
            p_n_times_identity[d][d] = pres_old;
          }

          const auto& rho_tmp_2          = phi_rho_tmp_2.get_value(q);

          phi.submit_value(rho_old*u_old -
                           a21_tilde*dt/(Fr*Fr)*rho_old*e_k -
                           a22_tilde*dt/(Fr*Fr)*rho_tmp_2*e_k, q);
          phi.submit_gradient(a21*dt*rho_old*tensor_product_u_n + a21_tilde*dt/(Ma*Ma)*p_n_times_identity, q);
        }
        phi.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
      }
    }
    else if(HYPERBOLIC_stage == 2) {
      /*--- We first start by declaring the suitable instances to read the available quantities. ---*/
      FEEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number> phi(data, 0),
                                                                   phi_u_old(data, 0),
                                                                   phi_u_tmp_2(data, 0);
      FEEvaluation<dim, fe_degree_T, n_q_points_1d_u, 1, Number>   phi_pres_old(data, 1),
                                                                   phi_pres_tmp_2(data, 1);
      FEEvaluation<dim, fe_degree_rho, n_q_points_1d_u, 1, Number> phi_rho_old(data, 2),
                                                                   phi_rho_tmp_2(data, 2),
                                                                   phi_rho_curr(data, 2);

      /*--- Loop over all cells ---*/
      for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
        phi_rho_old.reinit(cell);
        phi_rho_old.gather_evaluate(src[0], EvaluationFlags::values);
        phi_u_old.reinit(cell);
        phi_u_old.gather_evaluate(src[1], EvaluationFlags::values);
        phi_pres_old.reinit(cell);
        phi_pres_old.gather_evaluate(src[2], EvaluationFlags::values);

        phi_rho_tmp_2.reinit(cell);
        phi_rho_tmp_2.gather_evaluate(src[3], EvaluationFlags::values);
        phi_u_tmp_2.reinit(cell);
        phi_u_tmp_2.gather_evaluate(src[4], EvaluationFlags::values);
        phi_pres_tmp_2.reinit(cell);
        phi_pres_tmp_2.gather_evaluate(src[5], EvaluationFlags::values);

        phi_rho_curr.reinit(cell);
        phi_rho_curr.gather_evaluate(src[6], EvaluationFlags::values);

        phi.reinit(cell);

        /*--- Loop over all quadrature points ---*/
        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          const auto& rho_old               = phi_rho_old.get_value(q);
          const auto& u_old                 = phi_u_old.get_value(q);
          const auto& pres_old              = phi_pres_old.get_value(q);

          const auto& tensor_product_u_n    = outer_product(u_old, u_old);
          auto p_n_times_identity           = tensor_product_u_n;
          p_n_times_identity = 0;
          for(unsigned int d = 0; d < dim; ++d) {
            p_n_times_identity[d][d] = pres_old;
          }

          const auto& rho_tmp_2              = phi_rho_tmp_2.get_value(q);
          const auto& u_tmp_2                = phi_u_tmp_2.get_value(q);
          const auto& pres_tmp_2             = phi_pres_tmp_2.get_value(q);

          const auto& tensor_product_u_tmp_2 = outer_product(u_tmp_2, u_tmp_2);
          auto p_tmp_2_times_identity        = tensor_product_u_tmp_2;
          p_tmp_2_times_identity = 0;
          for(unsigned int d = 0; d < dim; ++d) {
            p_tmp_2_times_identity[d][d] = pres_tmp_2;
          }

          const auto& rho_curr               = phi_rho_curr.get_value(q);

          phi.submit_value(rho_old*u_old -
                           a31_tilde*dt/(Fr*Fr)*rho_old*e_k -
                           a32_tilde*dt/(Fr*Fr)*rho_tmp_2*e_k -
                           a33_tilde*dt/(Fr*Fr)*rho_curr*e_k, q);
          phi.submit_gradient(a31*dt*rho_old*tensor_product_u_n + a31_tilde*dt/(Ma*Ma)*p_n_times_identity +
                              a32*dt*rho_tmp_2*tensor_product_u_tmp_2 + a32_tilde*dt/(Ma*Ma)*p_tmp_2_times_identity, q);
        }
        phi.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
      }
    }
    else {
      /*--- We first start by declaring the suitable instances to read the available quantities. ---*/
      FEEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number> phi(data, 0),
                                                                   phi_u_old(data, 0),
                                                                   phi_u_tmp_2(data, 0),
                                                                   phi_u_tmp_3(data, 0);
      FEEvaluation<dim, fe_degree_T, n_q_points_1d_u, 1, Number>   phi_pres_old(data, 1),
                                                                   phi_pres_tmp_2(data, 1),
                                                                   phi_pres_tmp_3(data, 1);
      FEEvaluation<dim, fe_degree_rho, n_q_points_1d_u, 1, Number> phi_rho_old(data, 2),
                                                                   phi_rho_tmp_2(data, 2),
                                                                   phi_rho_tmp_3(data, 2);

      /*--- Loop over all cells ---*/
      for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
        phi_rho_old.reinit(cell);
        phi_rho_old.gather_evaluate(src[0], EvaluationFlags::values | EvaluationFlags::gradients);
        phi_u_old.reinit(cell);
        phi_u_old.gather_evaluate(src[1], EvaluationFlags::values | EvaluationFlags::gradients);
        phi_pres_old.reinit(cell);
        phi_pres_old.gather_evaluate(src[2], EvaluationFlags::values);

        phi_rho_tmp_2.reinit(cell);
        phi_rho_tmp_2.gather_evaluate(src[3], EvaluationFlags::values | EvaluationFlags::gradients);
        phi_u_tmp_2.reinit(cell);
        phi_u_tmp_2.gather_evaluate(src[4], EvaluationFlags::values | EvaluationFlags::gradients);
        phi_pres_tmp_2.reinit(cell);
        phi_pres_tmp_2.gather_evaluate(src[5], EvaluationFlags::values);

        phi_rho_tmp_3.reinit(cell);
        phi_rho_tmp_3.gather_evaluate(src[6], EvaluationFlags::values | EvaluationFlags::gradients);
        phi_u_tmp_3.reinit(cell);
        phi_u_tmp_3.gather_evaluate(src[7], EvaluationFlags::values | EvaluationFlags::gradients);
        phi_pres_tmp_3.reinit(cell);
        phi_pres_tmp_3.gather_evaluate(src[8], EvaluationFlags::values);

        phi.reinit(cell);

        /*--- Loop over all quadrature points ---*/
        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          const auto& rho_old            = phi_rho_old.get_value(q);
          const auto& u_old              = phi_u_old.get_value(q);
          const auto& pres_old           = phi_pres_old.get_value(q);

          const auto& tensor_product_u_n = outer_product(u_old, u_old);
          auto p_n_times_identity        = tensor_product_u_n;
          p_n_times_identity = 0;
          for(unsigned int d = 0; d < dim; ++d) {
            p_n_times_identity[d][d] = pres_old;
          }

          const auto& rho_tmp_2              = phi_rho_tmp_2.get_value(q);
          const auto& u_tmp_2                = phi_u_tmp_2.get_value(q);
          const auto& pres_tmp_2             = phi_pres_tmp_2.get_value(q);

          const auto& tensor_product_u_tmp_2 = outer_product(u_tmp_2, u_tmp_2);
          auto p_tmp_2_times_identity        = tensor_product_u_tmp_2;
          p_tmp_2_times_identity = 0;
          for(unsigned int d = 0; d < dim; ++d) {
            p_tmp_2_times_identity[d][d] = pres_tmp_2;
          }

          const auto& rho_tmp_3              = phi_rho_tmp_3.get_value(q);
          const auto& u_tmp_3                = phi_u_tmp_3.get_value(q);
          const auto& pres_tmp_3             = phi_pres_tmp_3.get_value(q);

          const auto& tensor_product_u_tmp_3 = outer_product(u_tmp_3, u_tmp_3);
          auto p_tmp_3_times_identity        = tensor_product_u_tmp_3;
          p_tmp_3_times_identity = 0;
          for(unsigned int d = 0; d < dim; ++d) {
            p_tmp_3_times_identity[d][d] = pres_tmp_3;
          }

          phi.submit_value(rho_old*u_old -
                           b1*dt/(Fr*Fr)*rho_old*e_k -
                           b2*dt/(Fr*Fr)*rho_tmp_2*e_k -
                           b3*dt/(Fr*Fr)*rho_tmp_3*e_k, q);
          phi.submit_gradient(b1*dt*rho_old*tensor_product_u_n + b1*dt/(Ma*Ma)*p_n_times_identity +
                              b2*dt*rho_tmp_2*tensor_product_u_tmp_2 + b2*dt/(Ma*Ma)*p_tmp_2_times_identity +
                              b3*dt*rho_tmp_3*tensor_product_u_tmp_3 + b3*dt/(Ma*Ma)*p_tmp_3_times_identity, q);
        }
        phi.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
      }
    }
  }


  // Assemble rhs face term for the velocity update
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void EULEROperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  assemble_rhs_face_term_velocity_fixed(const MatrixFree<dim, Number>&               data,
                                        Vec&                                         dst,
                                        const std::vector<Vec>&                      src,
                                        const std::pair<unsigned int, unsigned int>& face_range) const {
    if(HYPERBOLIC_stage == 1) {
      /*--- We first start by declaring the suitable instances to read the available quantities. ---*/
      FEFaceEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number> phi_p(data, true, 0),
                                                                       phi_m(data, false, 0),
                                                                       phi_u_old_p(data, true, 0),
                                                                       phi_u_old_m(data, false, 0);
      FEFaceEvaluation<dim, fe_degree_T, n_q_points_1d_u, 1, Number>   phi_pres_old_p(data, true, 1),
                                                                       phi_pres_old_m(data, false, 1);
      FEFaceEvaluation<dim, fe_degree_rho, n_q_points_1d_u, 1, Number> phi_rho_old_p(data, true, 2),
                                                                       phi_rho_old_m(data, false, 2);

      /*--- Loop over all internal faces ---*/
      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        phi_rho_old_p.reinit(face);
        phi_rho_old_p.gather_evaluate(src[0], EvaluationFlags::values);
        phi_rho_old_m.reinit(face);
        phi_rho_old_m.gather_evaluate(src[0], EvaluationFlags::values);
        phi_u_old_p.reinit(face);
        phi_u_old_p.gather_evaluate(src[1], EvaluationFlags::values);
        phi_u_old_m.reinit(face);
        phi_u_old_m.gather_evaluate(src[1], EvaluationFlags::values);
        phi_pres_old_p.reinit(face);
        phi_pres_old_p.gather_evaluate(src[2], EvaluationFlags::values);
        phi_pres_old_m.reinit(face);
        phi_pres_old_m.gather_evaluate(src[2], EvaluationFlags::values);

        phi_p.reinit(face);
        phi_m.reinit(face);

        /*--- Loop over all quadrature points ---*/
        for(unsigned int q = 0; q < phi_p.n_q_points; ++q) {
          const auto& n_plus                 = phi_p.get_normal_vector(q);

          const auto& rho_old_p              = phi_rho_old_p.get_value(q);
          const auto& rho_old_m              = phi_rho_old_m.get_value(q);
          const auto& u_old_p                = phi_u_old_p.get_value(q);
          const auto& u_old_m                = phi_u_old_m.get_value(q);
          const auto& pres_old_p             = phi_pres_old_p.get_value(q);
          const auto& pres_old_m             = phi_pres_old_m.get_value(q);
          const auto& avg_tensor_product_u_n = 0.5*(outer_product(rho_old_p*u_old_p, u_old_p) +
                                                    outer_product(rho_old_m*u_old_m, u_old_m));
          const auto& avg_pres_old           = 0.5*(pres_old_p + pres_old_m);
          const auto& jump_rhou_old          = rho_old_p*u_old_p - rho_old_m*u_old_m;
          const auto& lambda_old             = std::max(std::abs(scalar_product(u_old_p, n_plus)),
                                                        std::abs(scalar_product(u_old_m, n_plus)));

          phi_p.submit_value(-a21*dt*avg_tensor_product_u_n*n_plus
                             -a21_tilde*dt/(Ma*Ma)*avg_pres_old*n_plus
                             -a21*dt*0.5*lambda_old*jump_rhou_old, q);
          phi_m.submit_value(a21*dt*avg_tensor_product_u_n*n_plus +
                             a21_tilde*dt/(Ma*Ma)*avg_pres_old*n_plus +
                             a21*dt*0.5*lambda_old*jump_rhou_old, q);
        }
        phi_p.integrate_scatter(EvaluationFlags::values, dst);
        phi_m.integrate_scatter(EvaluationFlags::values, dst);
      }
    }
    else if(HYPERBOLIC_stage == 2) {
      /*--- We first start by declaring the suitable instances to read the available quantities. ---*/
      FEFaceEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number> phi_p(data, true, 0),
                                                                       phi_m(data, false, 0),
                                                                       phi_u_old_p(data, true, 0),
                                                                       phi_u_old_m(data, false, 0),
                                                                       phi_u_tmp_2_p(data, true, 0),
                                                                       phi_u_tmp_2_m(data, false, 0);
      FEFaceEvaluation<dim, fe_degree_T, n_q_points_1d_u, 1, Number>   phi_pres_old_p(data, true, 1),
                                                                       phi_pres_old_m(data, false, 1),
                                                                       phi_pres_tmp_2_p(data, true, 1),
                                                                       phi_pres_tmp_2_m(data, false, 1);
      FEFaceEvaluation<dim, fe_degree_rho, n_q_points_1d_u, 1, Number> phi_rho_old_p(data, true, 2),
                                                                       phi_rho_old_m(data, false, 2),
                                                                       phi_rho_tmp_2_p(data, true, 2),
                                                                       phi_rho_tmp_2_m(data, false, 2);

      /*---Loop over all internal faces ---*/
      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        phi_rho_old_p.reinit(face);
        phi_rho_old_p.gather_evaluate(src[0], EvaluationFlags::values);
        phi_rho_old_m.reinit(face);
        phi_rho_old_m.gather_evaluate(src[0], EvaluationFlags::values);
        phi_u_old_p.reinit(face);
        phi_u_old_p.gather_evaluate(src[1], EvaluationFlags::values);
        phi_u_old_m.reinit(face);
        phi_u_old_m.gather_evaluate(src[1], EvaluationFlags::values);
        phi_pres_old_p.reinit(face);
        phi_pres_old_p.gather_evaluate(src[2], EvaluationFlags::values);
        phi_pres_old_m.reinit(face);
        phi_pres_old_m.gather_evaluate(src[2], EvaluationFlags::values);

        phi_rho_tmp_2_p.reinit(face);
        phi_rho_tmp_2_p.gather_evaluate(src[3], EvaluationFlags::values);
        phi_rho_tmp_2_m.reinit(face);
        phi_rho_tmp_2_m.gather_evaluate(src[3], EvaluationFlags::values);
        phi_u_tmp_2_p.reinit(face);
        phi_u_tmp_2_p.gather_evaluate(src[4], EvaluationFlags::values);
        phi_u_tmp_2_m.reinit(face);
        phi_u_tmp_2_m.gather_evaluate(src[4], EvaluationFlags::values);
        phi_pres_tmp_2_p.reinit(face);
        phi_pres_tmp_2_p.gather_evaluate(src[5], EvaluationFlags::values);
        phi_pres_tmp_2_m.reinit(face);
        phi_pres_tmp_2_m.gather_evaluate(src[5], EvaluationFlags::values);

        phi_p.reinit(face);
        phi_m.reinit(face);

        /*--- Loop over all quadrature points ---*/
        for(unsigned int q = 0; q < phi_p.n_q_points; ++q) {
          const auto& n_plus                     = phi_p.get_normal_vector(q);

          const auto& rho_old_p                  = phi_rho_old_p.get_value(q);
          const auto& rho_old_m                  = phi_rho_old_m.get_value(q);
          const auto& u_old_p                    = phi_u_old_p.get_value(q);
          const auto& u_old_m                    = phi_u_old_m.get_value(q);
          const auto& pres_old_p                 = phi_pres_old_p.get_value(q);
          const auto& pres_old_m                 = phi_pres_old_m.get_value(q);
          const auto& avg_tensor_product_u_n     = 0.5*(outer_product(rho_old_p*u_old_p, u_old_p) +
                                                        outer_product(rho_old_m*u_old_m, u_old_m));
          const auto& avg_pres_old               = 0.5*(pres_old_p + pres_old_m);
          const auto& jump_rhou_old              = rho_old_p*u_old_p - rho_old_m*u_old_m;
          const auto& lambda_old                 = std::max(std::abs(scalar_product(u_old_p, n_plus)),
                                                            std::abs(scalar_product(u_old_m, n_plus)));

          const auto& rho_tmp_2_p                = phi_rho_tmp_2_p.get_value(q);
          const auto& rho_tmp_2_m                = phi_rho_tmp_2_m.get_value(q);
          const auto& u_tmp_2_p                  = phi_u_tmp_2_p.get_value(q);
          const auto& u_tmp_2_m                  = phi_u_tmp_2_m.get_value(q);
          const auto& pres_tmp_2_p               = phi_pres_tmp_2_p.get_value(q);
          const auto& pres_tmp_2_m               = phi_pres_tmp_2_m.get_value(q);
          const auto& avg_tensor_product_u_tmp_2 = 0.5*(outer_product(rho_tmp_2_p*u_tmp_2_p, u_tmp_2_p) +
                                                        outer_product(rho_tmp_2_m*u_tmp_2_m, u_tmp_2_m));
          const auto& avg_pres_tmp_2             = 0.5*(pres_tmp_2_p + pres_tmp_2_m);
          const auto& jump_rhou_tmp_2            = rho_tmp_2_p*u_tmp_2_p - rho_tmp_2_m*u_tmp_2_m;
          const auto& lambda_tmp_2               = std::max(std::abs(scalar_product(u_tmp_2_p, n_plus)),
                                                            std::abs(scalar_product(u_tmp_2_m, n_plus)));

          phi_p.submit_value(-a31*dt*avg_tensor_product_u_n*n_plus
                             -a31_tilde*dt/(Ma*Ma)*avg_pres_old*n_plus
                             -a31*dt*0.5*lambda_old*jump_rhou_old
                             -a32*dt*avg_tensor_product_u_tmp_2*n_plus
                             -a32_tilde*dt/(Ma*Ma)*avg_pres_tmp_2*n_plus
                             -a32*dt*0.5*lambda_tmp_2*jump_rhou_tmp_2, q);
          phi_m.submit_value(a31*dt*avg_tensor_product_u_n*n_plus +
                             a31_tilde*dt/(Ma*Ma)*avg_pres_old*n_plus +
                             a31*dt*0.5*lambda_old*jump_rhou_old +
                             a32*dt*avg_tensor_product_u_tmp_2*n_plus +
                             a32_tilde*dt/(Ma*Ma)*avg_pres_tmp_2*n_plus +
                             a32*dt*0.5*lambda_tmp_2*jump_rhou_tmp_2, q);
        }
        phi_p.integrate_scatter(EvaluationFlags::values, dst);
        phi_m.integrate_scatter(EvaluationFlags::values, dst);
      }
    }
    else {
      /*--- We first start by declaring the suitable instances to read the available quantities. ---*/
      FEFaceEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number> phi_p(data, true, 0),
                                                                       phi_m(data, false, 0),
                                                                       phi_u_old_p(data, true, 0),
                                                                       phi_u_old_m(data, false, 0),
                                                                       phi_u_tmp_2_p(data, true, 0),
                                                                       phi_u_tmp_2_m(data, false, 0),
                                                                       phi_u_tmp_3_p(data, true, 0),
                                                                       phi_u_tmp_3_m(data, false, 0);
      FEFaceEvaluation<dim, fe_degree_T, n_q_points_1d_u, 1, Number>   phi_pres_old_p(data, true, 1),
                                                                       phi_pres_old_m(data, false, 1),
                                                                       phi_pres_tmp_2_p(data, true, 1),
                                                                       phi_pres_tmp_2_m(data, false, 1),
                                                                       phi_pres_tmp_3_p(data, true, 1),
                                                                       phi_pres_tmp_3_m(data, false, 1);
      FEFaceEvaluation<dim, fe_degree_rho, n_q_points_1d_u, 1, Number> phi_rho_old_p(data, true, 2),
                                                                       phi_rho_old_m(data, false, 2),
                                                                       phi_rho_tmp_2_p(data, true, 2),
                                                                       phi_rho_tmp_2_m(data, false, 2),
                                                                       phi_rho_tmp_3_p(data, true, 2),
                                                                       phi_rho_tmp_3_m(data, false, 2);

      /*--- Loop over all internal faces ---*/
      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        phi_rho_old_p.reinit(face);
        phi_rho_old_p.gather_evaluate(src[0], EvaluationFlags::values);
        phi_rho_old_m.reinit(face);
        phi_rho_old_m.gather_evaluate(src[0], EvaluationFlags::values);
        phi_u_old_p.reinit(face);
        phi_u_old_p.gather_evaluate(src[1], EvaluationFlags::values);
        phi_u_old_m.reinit(face);
        phi_u_old_m.gather_evaluate(src[1], EvaluationFlags::values);
        phi_pres_old_p.reinit(face);
        phi_pres_old_p.gather_evaluate(src[2], EvaluationFlags::values);
        phi_pres_old_m.reinit(face);
        phi_pres_old_m.gather_evaluate(src[2], EvaluationFlags::values);

        phi_rho_tmp_2_p.reinit(face);
        phi_rho_tmp_2_p.gather_evaluate(src[3], EvaluationFlags::values);
        phi_rho_tmp_2_m.reinit(face);
        phi_rho_tmp_2_m.gather_evaluate(src[3], EvaluationFlags::values);
        phi_u_tmp_2_p.reinit(face);
        phi_u_tmp_2_p.gather_evaluate(src[4], EvaluationFlags::values);
        phi_u_tmp_2_m.reinit(face);
        phi_u_tmp_2_m.gather_evaluate(src[4], EvaluationFlags::values);
        phi_pres_tmp_2_p.reinit(face);
        phi_pres_tmp_2_p.gather_evaluate(src[5], EvaluationFlags::values);
        phi_pres_tmp_2_m.reinit(face);
        phi_pres_tmp_2_m.gather_evaluate(src[5], EvaluationFlags::values);

        phi_rho_tmp_3_p.reinit(face);
        phi_rho_tmp_3_p.gather_evaluate(src[6], EvaluationFlags::values);
        phi_rho_tmp_3_m.reinit(face);
        phi_rho_tmp_3_m.gather_evaluate(src[6], EvaluationFlags::values);
        phi_u_tmp_3_p.reinit(face);
        phi_u_tmp_3_p.gather_evaluate(src[7], EvaluationFlags::values);
        phi_u_tmp_3_m.reinit(face);
        phi_u_tmp_3_m.gather_evaluate(src[7], EvaluationFlags::values);
        phi_pres_tmp_3_p.reinit(face);
        phi_pres_tmp_3_p.gather_evaluate(src[8], EvaluationFlags::values);
        phi_pres_tmp_3_m.reinit(face);
        phi_pres_tmp_3_m.gather_evaluate(src[8], EvaluationFlags::values);

        phi_p.reinit(face);
        phi_m.reinit(face);

        /*--- Loop over all quadrature points ---*/
        for(unsigned int q = 0; q < phi_p.n_q_points; ++q) {
          const auto& n_plus                     = phi_p.get_normal_vector(q);

          const auto& rho_old_p                  = phi_rho_old_p.get_value(q);
          const auto& rho_old_m                  = phi_rho_old_m.get_value(q);
          const auto& u_old_p                    = phi_u_old_p.get_value(q);
          const auto& u_old_m                    = phi_u_old_m.get_value(q);
          const auto& pres_old_p                 = phi_pres_old_p.get_value(q);
          const auto& pres_old_m                 = phi_pres_old_m.get_value(q);
          const auto& avg_tensor_product_u_n     = 0.5*(outer_product(rho_old_p*u_old_p, u_old_p) +
                                                        outer_product(rho_old_m*u_old_m, u_old_m));
          const auto& avg_pres_old               = 0.5*(pres_old_p + pres_old_m);
          const auto& jump_rhou_old              = rho_old_p*u_old_p - rho_old_m*u_old_m;
          const auto& lambda_old                 = std::max(std::abs(scalar_product(u_old_p, n_plus)),
                                                            std::abs(scalar_product(u_old_m, n_plus)));

          const auto& rho_tmp_2_p                = phi_rho_tmp_2_p.get_value(q);
          const auto& rho_tmp_2_m                = phi_rho_tmp_2_m.get_value(q);
          const auto& u_tmp_2_p                  = phi_u_tmp_2_p.get_value(q);
          const auto& u_tmp_2_m                  = phi_u_tmp_2_m.get_value(q);
          const auto& pres_tmp_2_p               = phi_pres_tmp_2_p.get_value(q);
          const auto& pres_tmp_2_m               = phi_pres_tmp_2_m.get_value(q);
          const auto& avg_tensor_product_u_tmp_2 = 0.5*(outer_product(rho_tmp_2_p*u_tmp_2_p, u_tmp_2_p) +
                                                        outer_product(rho_tmp_2_m*u_tmp_2_m, u_tmp_2_m));
          const auto& avg_pres_tmp_2             = 0.5*(pres_tmp_2_p + pres_tmp_2_m);
          const auto& jump_rhou_tmp_2            = rho_tmp_2_p*u_tmp_2_p - rho_tmp_2_m*u_tmp_2_m;
          const auto& lambda_tmp_2               = std::max(std::abs(scalar_product(u_tmp_2_p, n_plus)),
                                                            std::abs(scalar_product(u_tmp_2_m, n_plus)));

          const auto& rho_tmp_3_p                = phi_rho_tmp_3_p.get_value(q);
          const auto& rho_tmp_3_m                = phi_rho_tmp_3_m.get_value(q);
          const auto& u_tmp_3_p                  = phi_u_tmp_3_p.get_value(q);
          const auto& u_tmp_3_m                  = phi_u_tmp_3_m.get_value(q);
          const auto& pres_tmp_3_p               = phi_pres_tmp_3_p.get_value(q);
          const auto& pres_tmp_3_m               = phi_pres_tmp_3_m.get_value(q);
          const auto& avg_tensor_product_u_tmp_3 = 0.5*(outer_product(rho_tmp_3_p*u_tmp_3_p, u_tmp_3_p) +
                                                        outer_product(rho_tmp_3_m*u_tmp_3_m, u_tmp_3_m));
          const auto& avg_pres_tmp_3             = 0.5*(pres_tmp_3_p + pres_tmp_3_m);
          const auto& jump_rhou_tmp_3            = rho_tmp_3_p*u_tmp_3_p - rho_tmp_3_m*u_tmp_3_m;
          const auto& lambda_tmp_3               = std::max(std::abs(scalar_product(u_tmp_3_p, n_plus)),
                                                            std::abs(scalar_product(u_tmp_3_m, n_plus)));

          phi_p.submit_value(-b1*dt*avg_tensor_product_u_n*n_plus
                             -b1*dt/(Ma*Ma)*avg_pres_old*n_plus
                             -b1*dt*0.5*lambda_old*jump_rhou_old
                             -b2*dt*avg_tensor_product_u_tmp_2*n_plus
                             -b2*dt/(Ma*Ma)*avg_pres_tmp_2*n_plus
                             -b2*dt*0.5*lambda_tmp_2*jump_rhou_tmp_2
                             -b3*dt*avg_tensor_product_u_tmp_3*n_plus
                             -b3*dt/(Ma*Ma)*avg_pres_tmp_3*n_plus
                             -b3*dt*0.5*lambda_tmp_3*jump_rhou_tmp_3, q);
          phi_m.submit_value(b1*dt*avg_tensor_product_u_n*n_plus +
                             b1*dt/(Ma*Ma)*avg_pres_old*n_plus +
                             b1*dt*0.5*lambda_old*jump_rhou_old +
                             b2*dt*avg_tensor_product_u_tmp_2*n_plus +
                             b2*dt/(Ma*Ma)*avg_pres_tmp_2*n_plus +
                             b2*dt*0.5*lambda_tmp_2*jump_rhou_tmp_2 +
                             b3*dt*avg_tensor_product_u_tmp_3*n_plus +
                             b3*dt/(Ma*Ma)*avg_pres_tmp_3*n_plus +
                             b3*dt*0.5*lambda_tmp_3*jump_rhou_tmp_3, q);
        }
        phi_p.integrate_scatter(EvaluationFlags::values, dst);
        phi_m.integrate_scatter(EvaluationFlags::values, dst);
      }
    }
  }


  // Assemble rhs boundary term for the velocity update
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void EULEROperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  assemble_rhs_boundary_term_velocity_fixed(const MatrixFree<dim, Number>&               data,
                                            Vec&                                         dst,
                                            const std::vector<Vec>&                      src,
                                            const std::pair<unsigned int, unsigned int>& face_range) const {
    if(HYPERBOLIC_stage == 1) {
      /*--- We first start by declaring the suitable instances to read the available quantities. ---*/
      FEFaceEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number> phi(data, true, 0),
                                                                       phi_u_old(data, true, 0);
      FEFaceEvaluation<dim, fe_degree_T, n_q_points_1d_u, 1, Number>   phi_pres_old(data, true, 1);
      FEFaceEvaluation<dim, fe_degree_rho, n_q_points_1d_u, 1, Number> phi_rho_old(data, true, 2);

      /*--- Loop over all internal faces ---*/
      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        phi_rho_old.reinit(face);
        phi_rho_old.gather_evaluate(src[0], EvaluationFlags::values);
        phi_u_old.reinit(face);
        phi_u_old.gather_evaluate(src[1], EvaluationFlags::values);
        phi_pres_old.reinit(face);
        phi_pres_old.gather_evaluate(src[2], EvaluationFlags::values);

        phi.reinit(face);

        /*--- Loop over all quadrature points ---*/
        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          const auto& n_plus                 = phi.get_normal_vector(q);

          const auto& rho_old                = phi_rho_old.get_value(q);
          const auto& u_old                  = phi_u_old.get_value(q);
          const auto& pres_old               = phi_pres_old.get_value(q);
          auto rho_old_D                     = VectorizedArray<Number>();
          auto u_old_D                       = Tensor<1, dim, VectorizedArray<Number>>();
          auto pres_old_D                    = VectorizedArray<Number>();
          const auto& point_vectorized       = phi.quadrature_point(q);
          for(unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v) {
            Point<dim> point; /*--- The point returned by the 'quadrature_point' function is not an instance of Point
                                    and so it is not ready to be directly used. We need to pay attention to the
                                    vectorization ---*/
            for(unsigned int d = 0; d < dim; ++d) {
              point[d] = point_vectorized[d][v];
            }
            rho_old_D = rho_exact.value(point);
            for(unsigned int d = 0; d < dim; ++d) {
              u_old_D[d][v] = u_exact.value(point, d);
            }
            pres_old_D = pres_exact.value(point);
          }
          const auto& avg_tensor_product_u_n = 0.5*(outer_product(rho_old*u_old, u_old) +
                                                    outer_product(rho_old_D*u_old_D, u_old_D));
          const auto& avg_pres_old           = 0.5*(pres_old + pres_old_D);
          const auto& jump_rhou_old          = rho_old*u_old - rho_old_D*u_old_D;
          const auto& lambda_old             = std::max(std::abs(scalar_product(u_old, n_plus)),
                                                        std::abs(scalar_product(u_old_D, n_plus)));

          phi.submit_value(-a21*dt*avg_tensor_product_u_n*n_plus
                           -a21_tilde*dt/(Ma*Ma)*avg_pres_old*n_plus
                           -a21*dt*0.5*lambda_old*jump_rhou_old, q);
        }
        phi.integrate_scatter(EvaluationFlags::values, dst);
      }
    }
    else if(HYPERBOLIC_stage == 2) {
      /*--- We first start by declaring the suitable instances to read the available quantities. ---*/
      FEFaceEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number> phi(data, true, 0),
                                                                       phi_u_old(data, true, 0),
                                                                       phi_u_tmp_2(data, true, 0);
      FEFaceEvaluation<dim, fe_degree_T, n_q_points_1d_u, 1, Number>   phi_pres_old(data, true, 1),
                                                                       phi_pres_tmp_2(data, true, 1);
      FEFaceEvaluation<dim, fe_degree_rho, n_q_points_1d_u, 1, Number> phi_rho_old(data, true, 2),
                                                                       phi_rho_tmp_2(data, true, 2);

      /*---Loop over all internal faces ---*/
      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        phi_rho_old.reinit(face);
        phi_rho_old.gather_evaluate(src[0], EvaluationFlags::values);
        phi_u_old.reinit(face);
        phi_u_old.gather_evaluate(src[1], EvaluationFlags::values);
        phi_pres_old.reinit(face);
        phi_pres_old.gather_evaluate(src[2], EvaluationFlags::values);

        phi_rho_tmp_2.reinit(face);
        phi_rho_tmp_2.gather_evaluate(src[3], EvaluationFlags::values);
        phi_u_tmp_2.reinit(face);
        phi_u_tmp_2.gather_evaluate(src[4], EvaluationFlags::values);
        phi_pres_tmp_2.reinit(face);
        phi_pres_tmp_2.gather_evaluate(src[5], EvaluationFlags::values);

        phi.reinit(face);

        /*--- Loop over all quadrature points ---*/
        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          const auto& n_plus                     = phi.get_normal_vector(q);

          const auto& rho_old                    = phi_rho_old.get_value(q);
          const auto& u_old                      = phi_u_old.get_value(q);
          const auto& pres_old                   = phi_pres_old.get_value(q);
          auto rho_old_D                         = VectorizedArray<Number>();
          auto u_old_D                           = Tensor<1, dim, VectorizedArray<Number>>();
          auto pres_old_D                        = VectorizedArray<Number>();
          const auto& point_vectorized           = phi.quadrature_point(q);
          for(unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v) {
            Point<dim> point;
            for(unsigned int d = 0; d < dim; ++d) {
              point[d] = point_vectorized[d][v];
            }
            rho_old_D = rho_exact.value(point);
            for(unsigned int d = 0; d < dim; ++d) {
              u_old_D[d][v] = u_exact.value(point, d);
            }
            pres_old_D = pres_exact.value(point);
          }
          const auto& avg_tensor_product_u_n     = 0.5*(outer_product(rho_old*u_old, u_old) +
                                                        outer_product(rho_old_D*u_old_D, u_old_D));
          const auto& avg_pres_old               = 0.5*(pres_old + pres_old_D);
          const auto& jump_rhou_old              = rho_old*u_old - rho_old_D*u_old_D;
          const auto& lambda_old                 = std::max(std::abs(scalar_product(u_old, n_plus)),
                                                            std::abs(scalar_product(u_old_D, n_plus)));

          const auto& rho_tmp_2                  = phi_rho_tmp_2.get_value(q);
          const auto& u_tmp_2                    = phi_u_tmp_2.get_value(q);
          const auto& pres_tmp_2                 = phi_pres_tmp_2.get_value(q);
          auto rho_tmp_2_D                       = VectorizedArray<Number>();
          auto u_tmp_2_D                         = Tensor<1, dim, VectorizedArray<Number>>();
          auto pres_tmp_2_D                      = VectorizedArray<Number>();
          for(unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v) {
            Point<dim> point;
            for(unsigned int d = 0; d < dim; ++d) {
              point[d] = point_vectorized[d][v];
            }
            rho_tmp_2_D = rho_exact.value(point);
            for(unsigned int d = 0; d < dim; ++d) {
              u_tmp_2_D[d][v] = u_exact.value(point, d);
            }
            pres_tmp_2_D = pres_exact.value(point);
          }
          const auto& avg_tensor_product_u_tmp_2 = 0.5*(outer_product(rho_tmp_2*u_tmp_2, u_tmp_2) +
                                                        outer_product(rho_tmp_2_D*u_tmp_2_D, u_tmp_2_D));
          const auto& avg_pres_tmp_2             = 0.5*(pres_tmp_2 + pres_tmp_2_D);
          const auto& jump_rhou_tmp_2            = rho_tmp_2*u_tmp_2 - rho_tmp_2_D*u_tmp_2_D;
          const auto& lambda_tmp_2               = std::max(std::abs(scalar_product(u_tmp_2, n_plus)),
                                                            std::abs(scalar_product(u_tmp_2_D, n_plus)));

          phi.submit_value(-a31*dt*avg_tensor_product_u_n*n_plus
                           -a31_tilde*dt/(Ma*Ma)*avg_pres_old*n_plus
                           -a31*dt*0.5*lambda_old*jump_rhou_old
                           -a32*dt*avg_tensor_product_u_tmp_2*n_plus
                           -a32_tilde*dt/(Ma*Ma)*avg_pres_tmp_2*n_plus
                           -a32*dt*0.5*lambda_tmp_2*jump_rhou_tmp_2, q);
        }
        phi.integrate_scatter(EvaluationFlags::values, dst);
      }
    }
    else {
      /*--- We first start by declaring the suitable instances to read the available quantities. ---*/
      FEFaceEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number> phi(data, true, 0),
                                                                       phi_u_old(data, true, 0),
                                                                       phi_u_tmp_2(data, true, 0),
                                                                       phi_u_tmp_3(data, true, 0);
      FEFaceEvaluation<dim, fe_degree_T, n_q_points_1d_u, 1, Number>   phi_pres_old(data, true, 1),
                                                                       phi_pres_tmp_2(data, true, 1),
                                                                       phi_pres_tmp_3(data, true, 1);
      FEFaceEvaluation<dim, fe_degree_rho, n_q_points_1d_u, 1, Number> phi_rho_old(data, true, 2),
                                                                       phi_rho_tmp_2(data, true, 2),
                                                                       phi_rho_tmp_3(data, true, 2);

      /*--- Loop over all internal faces ---*/
      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        phi_rho_old.reinit(face);
        phi_rho_old.gather_evaluate(src[0], EvaluationFlags::values);
        phi_u_old.reinit(face);
        phi_u_old.gather_evaluate(src[1], EvaluationFlags::values);
        phi_pres_old.reinit(face);
        phi_pres_old.gather_evaluate(src[2], EvaluationFlags::values);

        phi_rho_tmp_2.reinit(face);
        phi_rho_tmp_2.gather_evaluate(src[3], EvaluationFlags::values);
        phi_u_tmp_2.reinit(face);
        phi_u_tmp_2.gather_evaluate(src[4], EvaluationFlags::values);
        phi_pres_tmp_2.reinit(face);
        phi_pres_tmp_2.gather_evaluate(src[5], EvaluationFlags::values);

        phi_rho_tmp_3.reinit(face);
        phi_rho_tmp_3.gather_evaluate(src[6], EvaluationFlags::values);
        phi_u_tmp_3.reinit(face);
        phi_u_tmp_3.gather_evaluate(src[7], EvaluationFlags::values);
        phi_pres_tmp_3.reinit(face);
        phi_pres_tmp_3.gather_evaluate(src[8], EvaluationFlags::values);

        phi.reinit(face);

        /*--- Loop over all quadrature points ---*/
        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          const auto& n_plus                     = phi.get_normal_vector(q);

          const auto& rho_old                    = phi_rho_old.get_value(q);
          const auto& u_old                      = phi_u_old.get_value(q);
          const auto& pres_old                   = phi_pres_old.get_value(q);
          auto rho_old_D                         = VectorizedArray<Number>();
          auto u_old_D                           = Tensor<1, dim, VectorizedArray<Number>>();
          auto pres_old_D                        = VectorizedArray<Number>();
          const auto& point_vectorized           = phi.quadrature_point(q);
          for(unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v) {
            Point<dim> point;
            for(unsigned int d = 0; d < dim; ++d) {
              point[d] = point_vectorized[d][v];
            }
            rho_old_D = rho_exact.value(point);
            for(unsigned int d = 0; d < dim; ++d) {
              u_old_D[d][v] = u_exact.value(point, d);
            }
            pres_old_D = pres_exact.value(point);
          }
          const auto& avg_tensor_product_u_n     = 0.5*(outer_product(rho_old*u_old, u_old) +
                                                        outer_product(rho_old_D*u_old_D, u_old_D));
          const auto& avg_pres_old               = 0.5*(pres_old + pres_old_D);
          const auto& jump_rhou_old              = rho_old*u_old - rho_old_D*u_old_D;
          const auto& lambda_old                 = std::max(std::abs(scalar_product(u_old, n_plus)),
                                                            std::abs(scalar_product(u_old_D, n_plus)));

          const auto& rho_tmp_2                  = phi_rho_tmp_2.get_value(q);
          const auto& u_tmp_2                    = phi_u_tmp_2.get_value(q);
          const auto& pres_tmp_2                 = phi_pres_tmp_2.get_value(q);
          auto rho_tmp_2_D                       = VectorizedArray<Number>();
          auto u_tmp_2_D                         = Tensor<1, dim, VectorizedArray<Number>>();
          auto pres_tmp_2_D                      = VectorizedArray<Number>();
          for(unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v) {
            Point<dim> point;
            for(unsigned int d = 0; d < dim; ++d) {
              point[d] = point_vectorized[d][v];
            }
            rho_tmp_2_D = rho_exact.value(point);
            for(unsigned int d = 0; d < dim; ++d) {
              u_tmp_2_D[d][v] = u_exact.value(point, d);
            }
            pres_tmp_2_D = pres_exact.value(point);
          }
          const auto& avg_tensor_product_u_tmp_2 = 0.5*(outer_product(rho_tmp_2*u_tmp_2, u_tmp_2) +
                                                        outer_product(rho_tmp_2_D*u_tmp_2_D, u_tmp_2_D));
          const auto& avg_pres_tmp_2             = 0.5*(pres_tmp_2 + pres_tmp_2_D);
          const auto& jump_rhou_tmp_2            = rho_tmp_2*u_tmp_2 - rho_tmp_2_D*u_tmp_2_D;
          const auto& lambda_tmp_2               = std::max(std::abs(scalar_product(u_tmp_2, n_plus)),
                                                            std::abs(scalar_product(u_tmp_2_D, n_plus)));

          const auto& rho_tmp_3                  = phi_rho_tmp_3.get_value(q);
          const auto& u_tmp_3                    = phi_u_tmp_3.get_value(q);
          const auto& pres_tmp_3                 = phi_pres_tmp_3.get_value(q);
          auto rho_tmp_3_D                       = VectorizedArray<Number>();
          auto u_tmp_3_D                         = Tensor<1, dim, VectorizedArray<Number>>();
          auto pres_tmp_3_D                      = VectorizedArray<Number>();
          for(unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v) {
            Point<dim> point;
            for(unsigned int d = 0; d < dim; ++d) {
              point[d] = point_vectorized[d][v];
            }
            rho_tmp_3_D = rho_exact.value(point);
            for(unsigned int d = 0; d < dim; ++d) {
              u_tmp_3_D[d][v] = u_exact.value(point, d);
            }
            pres_tmp_3_D = pres_exact.value(point);
          }
          const auto& avg_tensor_product_u_tmp_3 = 0.5*(outer_product(rho_tmp_3*u_tmp_3, u_tmp_3) +
                                                        outer_product(rho_tmp_3_D*u_tmp_3_D, u_tmp_3_D));
          const auto& avg_pres_tmp_3             = 0.5*(pres_tmp_3 + pres_tmp_3_D);
          const auto& jump_rhou_tmp_3            = rho_tmp_3*u_tmp_3 - rho_tmp_3_D*u_tmp_3_D;
          const auto& lambda_tmp_3               = std::max(std::abs(scalar_product(u_tmp_3, n_plus)),
                                                            std::abs(scalar_product(u_tmp_3_D, n_plus)));

          phi.submit_value(-b1*dt*avg_tensor_product_u_n*n_plus
                           -b1*dt/(Ma*Ma)*avg_pres_old*n_plus
                           -b1*dt*0.5*lambda_old*jump_rhou_old
                           -b2*dt*avg_tensor_product_u_tmp_2*n_plus
                           -b2*dt/(Ma*Ma)*avg_pres_tmp_2*n_plus
                           -b2*dt*0.5*lambda_tmp_2*jump_rhou_tmp_2
                           -b3*dt*avg_tensor_product_u_tmp_3*n_plus
                           -b3*dt/(Ma*Ma)*avg_pres_tmp_3*n_plus
                           -b3*dt*0.5*lambda_tmp_3*jump_rhou_tmp_3, q);
        }
        phi.integrate_scatter(EvaluationFlags::values, dst);
      }
    }
  }


  // Put together all the previous steps for velocity update
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void EULEROperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  vmult_rhs_velocity_fixed(Vec& dst, const std::vector<Vec>& src) const {
    for(unsigned int d = 0; d < src.size(); ++d) {
      src[d].update_ghost_values();
    }

    this->data->loop(&EULEROperator::assemble_rhs_cell_term_velocity_fixed,
                     &EULEROperator::assemble_rhs_face_term_velocity_fixed,
                     &EULEROperator::assemble_rhs_boundary_term_velocity_fixed,
                     this, dst, src, true,
                     MatrixFree<dim, Number>::DataAccessOnFaces::unspecified,
                     MatrixFree<dim, Number>::DataAccessOnFaces::unspecified);
  }


  // Assemble cell term for the velocity update
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void EULEROperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  assemble_cell_term_velocity_fixed(const MatrixFree<dim, Number>&               data,
                                     Vec&                                         dst,
                                     const Vec&                                   src,
                                     const std::pair<unsigned int, unsigned int>& cell_range) const {
    /*--- We first start by declaring the suitable instances to read also available quantities.
          Since here we have just one 'src' vector, but we also need to deal with the current density,
          we employ the auxiliary vector 'rho_for_fixed' where we setted this information ---*/
    FEEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number> phi(data, 0);
    FEEvaluation<dim, fe_degree_rho, n_q_points_1d_u, 1, Number> phi_rho_for_fixed(data, 2);

    /*--- Loop over all cells ---*/
    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
      phi_rho_for_fixed.reinit(cell);
      phi_rho_for_fixed.gather_evaluate(rho_for_fixed, EvaluationFlags::values);
      phi.reinit(cell);
      phi.gather_evaluate(src, EvaluationFlags::values);

      /*--- Loop over all qaudrature points ---*/
      for(unsigned int q = 0; q < phi.n_q_points; ++q) {
        phi.submit_value(phi_rho_for_fixed.get_value(q)*phi.get_value(q), q);
      }
      phi.integrate_scatter(EvaluationFlags::values, dst);
    }
  }


  // Put together all previous steps
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void EULEROperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  apply_add(Vec& dst, const Vec& src) const {
    AssertIndexRange(NS_stage, 7);
    Assert(NS_stage > 0, ExcInternalError());

    if(NS_stage == 1 || NS_stage == 4) {
      this->data->cell_loop(&EULEROperator::assemble_cell_term_rho_update,
                            this, dst, src, false);
    }
    else if(NS_stage == 2) {
      this->data->cell_loop(&EULEROperator::assemble_cell_term_internal_energy,
                            this, dst, src, false);

      /*--- Implementation of the Schur complement operations ---*/
      Vec tmp_1;
      this->data->initialize_dof_vector(tmp_1, 0);
      this->vmult_pressure(tmp_1, src);

      NS_stage = 3;
      const std::vector<unsigned int> tmp_reinit = {0};
      auto* tmp_matrix = const_cast<EULEROperator*>(this);
      Vec tmp_2;
      this->data->initialize_dof_vector(tmp_2, 0);
      tmp_2 = 0;
      tmp_matrix->initialize(tmp_matrix->get_matrix_free(), tmp_reinit, tmp_reinit);

      SolverControl solver_control(10000, 1e-12*tmp_1.l2_norm());
      SolverCG<Vec> cg(solver_control);
      PreconditionJacobi<EULEROperator> preconditioner_Jacobi;
      tmp_matrix->compute_diagonal();
      preconditioner_Jacobi.initialize(*tmp_matrix);

      cg.solve(*tmp_matrix, tmp_2, tmp_1, preconditioner_Jacobi);

      Vec tmp_3;
      this->data->initialize_dof_vector(tmp_3, 1);
      this->vmult_enthalpy(tmp_3, tmp_2);

      dst.add(-1.0, tmp_3);
      NS_stage = 2;
      const std::vector<unsigned int> tmp = {1};
      tmp_matrix->initialize(tmp_matrix->get_matrix_free(), tmp, tmp);
      tmp_matrix->compute_diagonal();
    }
    else if(NS_stage == 3 || NS_stage == 5) {
      this->data->cell_loop(&EULEROperator::assemble_cell_term_velocity_fixed,
                            this, dst, src, false);
    }
    else {
      this->data->cell_loop(&EULEROperator::assemble_cell_term_internal_energy,
                            this, dst, src, false);
    }
  }


  // Application of pressure matrix
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void EULEROperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  vmult_pressure(Vec& dst, const Vec& src) const {
    src.update_ghost_values();

    this->data->loop(&EULEROperator::assemble_cell_term_pressure,
                     &EULEROperator::assemble_face_term_pressure,
                     &EULEROperator::assemble_boundary_term_pressure,
                     this, dst, src, true,
                     MatrixFree<dim, Number>::DataAccessOnFaces::unspecified,
                     MatrixFree<dim, Number>::DataAccessOnFaces::unspecified);
  }


  // Application of enthalpy matrix
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void EULEROperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  vmult_enthalpy(Vec& dst, const Vec& src) const {
    src.update_ghost_values();

    this->data->loop(&EULEROperator::assemble_cell_term_enthalpy,
                     &EULEROperator::assemble_face_term_enthalpy,
                     &EULEROperator::assemble_boundary_term_enthalpy,
                     this, dst, src, true,
                     MatrixFree<dim, Number>::DataAccessOnFaces::unspecified,
                     MatrixFree<dim, Number>::DataAccessOnFaces::unspecified);
  }


  // Assemble diagonal cell term for the rho projection
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void EULEROperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  assemble_diagonal_cell_term_rho_update(const MatrixFree<dim, Number>&               data,
                                         Vec&                                         dst,
                                         const unsigned int&                          ,
                                         const std::pair<unsigned int, unsigned int>& cell_range) const {
    FEEvaluation<dim, fe_degree_rho, n_q_points_1d_u, 1, Number> phi(data, 2);

    AlignedVector<VectorizedArray<Number>> diagonal(phi.dofs_per_component);

    /*--- Loop over all cells ---*/
    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
      phi.reinit(cell);

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
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void EULEROperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  assemble_diagonal_cell_term_velocity_fixed(const MatrixFree<dim, Number>&               data,
                                             Vec&                                         dst,
                                             const unsigned int&                          ,
                                             const std::pair<unsigned int, unsigned int>& cell_range) const {
    FEEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number> phi(data, 0);
    FEEvaluation<dim, fe_degree_rho, n_q_points_1d_u, 1, Number> phi_rho_for_fixed(data, 2);

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


  // Assemble diagonal cell term for the contribution due to internal energy
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void EULEROperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  assemble_diagonal_cell_term_internal_energy(const MatrixFree<dim, Number>&               data,
                                              Vec&                                         dst,
                                              const unsigned int&                          ,
                                              const std::pair<unsigned int, unsigned int>& cell_range) const {
    FEEvaluation<dim, fe_degree_T, n_q_points_1d_u, 1, Number> phi(data, 1);

    AlignedVector<VectorizedArray<Number>> diagonal(phi.dofs_per_component);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
      phi.reinit(cell);

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
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void EULEROperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  compute_diagonal() {
    this->inverse_diagonal_entries.reset(new DiagonalMatrix<Vec>());
    auto& inverse_diagonal = this->inverse_diagonal_entries->get_vector();

    const unsigned int dummy = 0;

    if(NS_stage == 1 || NS_stage == 4) {
      this->data->initialize_dof_vector(inverse_diagonal, 2);

      this->data->cell_loop(&EULEROperator::assemble_diagonal_cell_term_rho_update,
                            this, inverse_diagonal, dummy, false);
    }
    else if(NS_stage == 2 || NS_stage == 6) {
      this->data->initialize_dof_vector(inverse_diagonal, 1);

      this->data->cell_loop(&EULEROperator::assemble_diagonal_cell_term_internal_energy,
                            this, inverse_diagonal, dummy, false);
    }
    else {
      this->data->initialize_dof_vector(inverse_diagonal, 0);

      this->data->cell_loop(&EULEROperator::assemble_diagonal_cell_term_velocity_fixed,
                            this, inverse_diagonal, dummy, false);
    }

    /*--- For the preconditioner, we actually need the inverse of the diagonal ---*/
    for(unsigned int i = 0; i < inverse_diagonal.local_size(); ++i) {
      Assert(inverse_diagonal.local_element(i) != 0.0,
             ExcMessage("No diagonal entry in a definite operator should be zero"));
      inverse_diagonal.local_element(i) = 1.0/inverse_diagonal.local_element(i);
    }
  }
}
