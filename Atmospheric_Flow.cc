/* Author: Giuseppe Orlando, 2023. */

// @sect{Include files}

// We start by including all the necessary deal.II header files
//
#include <deal.II/base/parallel.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/conditional_ostream.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/affine_constraints.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/manifold_lib.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q.h>

#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>

#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>
#include <deal.II/matrix_free/fe_evaluation.h>

#include <deal.II/base/timer.h>

#include <deal.II/multigrid/multigrid.h>
#include <deal.II/multigrid/mg_transfer_matrix_free.h>
#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_matrix.h>

#include "runtime_parameters.h"
#include "equation_data.h"

// This is the class that implements the discretization
//
namespace Atmospheric_Flow {
  using namespace dealii;

  // @sect{ <code>HYPERBOLICOperator::HYPERBOLICOperator</code> }
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  class HYPERBOLICOperator: public MatrixFreeOperators::Base<dim, Vec> {
  public:
    HYPERBOLICOperator();

    HYPERBOLICOperator(RunTimeParameters::Data_Storage& data);

    void set_dt(const double time_step);

    void set_Mach(const double Ma_);

    void set_Froude(const double Fr_);

    void set_HYPERBOLIC_stage(const unsigned int stage);

    void set_NS_stage(const unsigned int stage);

    void set_rho_for_fixed(const Vec& src);

    void set_pres_fixed(const Vec& src);

    void set_u_fixed(const Vec& src);

    void vmult_rhs_rho_update(Vec& dst, const std::vector<Vec>& src) const;

    void vmult_rhs_pressure(Vec& dst, const std::vector<Vec>& src) const;

    void vmult_rhs_velocity_fixed(Vec& dst, const std::vector<Vec>& src) const;

    void vmult_pressure(Vec& dst, const Vec& src) const;

    void vmult_enthalpy(Vec& dst, const Vec& src) const;

    virtual void compute_diagonal() override;

  protected:
    double       Ma;
    double       Fr;
    double       dt;

    double       gamma;
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

    unsigned int HYPERBOLIC_stage;
    mutable unsigned int NS_stage;

    virtual void apply_add(Vec& dst, const Vec& src) const override;

  private:
    Vec rho_for_fixed,
        pres_fixed,
        u_fixed;

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
                                               const std::pair<unsigned int, unsigned int>& face_range) const {}

    void assemble_cell_term_rho_update(const MatrixFree<dim, Number>&               data,
                                       Vec&                                         dst,
                                       const Vec&                                   src,
                                       const std::pair<unsigned int, unsigned int>& cell_range) const;

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

    void assemble_cell_term_velocity_fixed(const MatrixFree<dim, Number>&               data,
                                           Vec&                                         dst,
                                           const Vec&                                   src,
                                           const std::pair<unsigned int, unsigned int>& cell_range) const;

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
                                             const std::pair<unsigned int, unsigned int>& face_range) const {}

    void assemble_cell_term_internal_energy(const MatrixFree<dim, Number>&               data,
                                            Vec&                                         dst,
                                            const Vec&                                   src,
                                            const std::pair<unsigned int, unsigned int>& cell_range) const;

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

    void assemble_diagonal_cell_term_rho_update(const MatrixFree<dim, Number>&               data,
                                                Vec&                                         dst,
                                                const Vec&                                   src,
                                                const std::pair<unsigned int, unsigned int>& cell_range) const;
    void assemble_diagonal_face_term_rho_update(const MatrixFree<dim, Number>&               data,
                                                Vec&                                         dst,
                                                const Vec&                                   src,
                                                const std::pair<unsigned int, unsigned int>& face_range) const {}
    void assemble_diagonal_boundary_term_rho_update(const MatrixFree<dim, Number>&               data,
                                                    Vec&                                         dst,
                                                    const Vec&                                   src,
                                                    const std::pair<unsigned int, unsigned int>& face_range) const {}

    void assemble_diagonal_cell_term_velocity_fixed(const MatrixFree<dim, Number>&               data,
                                                    Vec&                                         dst,
                                                    const Vec&                                   src,
                                                    const std::pair<unsigned int, unsigned int>& cell_range) const;
    void assemble_diagonal_face_term_velocity_fixed(const MatrixFree<dim, Number>&               data,
                                                    Vec&                                         dst,
                                                    const Vec&                                   src,
                                                    const std::pair<unsigned int, unsigned int>& face_range) const {}
    void assemble_diagonal_boundary_term_velocity_fixed(const MatrixFree<dim, Number>&               data,
                                                        Vec&                                         dst,
                                                        const Vec&                                   src,
                                                        const std::pair<unsigned int, unsigned int>& face_range) const {}

    void assemble_diagonal_cell_term_internal_energy(const MatrixFree<dim, Number>&               data,
                                                     Vec&                                         dst,
                                                     const Vec&                                   src,
                                                     const std::pair<unsigned int, unsigned int>& cell_range) const;
    void assemble_diagonal_face_term_internal_energy(const MatrixFree<dim, Number>&               data,
                                                     Vec&                                         dst,
                                                     const Vec&                                   src,
                                                     const std::pair<unsigned int, unsigned int>& face_range) const {}
    void assemble_diagonal_boundary_term_internal_energy(const MatrixFree<dim, Number>&               data,
                                                         Vec&                                         dst,
                                                         const Vec&                                   src,
                                                         const std::pair<unsigned int, unsigned int>& face_range) const {}
  };


  // Default constructor
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  HYPERBOLICOperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                     n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  HYPERBOLICOperator(): MatrixFreeOperators::Base<dim, Vec>(), Ma(), Fr(), dt(), gamma(2.0 - std::sqrt(2.0)),
                        a21(gamma), a22(0.0), a31(0.5), a32(0.5), a33(0.0),
                        a21_tilde(0.5*gamma), a22_tilde(0.5*gamma), a31_tilde(std::sqrt(2)/4.0), a32_tilde(std::sqrt(2)/4.0),
                        a33_tilde(1.0 - std::sqrt(2)/2.0), b1(a31_tilde), b2(a32_tilde), b3(a33_tilde),
                        HYPERBOLIC_stage(1), NS_stage(1) {}


  // Constructor with runtime parameters storage
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  HYPERBOLICOperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                     n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  HYPERBOLICOperator(RunTimeParameters::Data_Storage& data): MatrixFreeOperators::Base<dim, Vec>(),
                                                             Ma(data.Mach), Fr(data.Froude), dt(data.dt), gamma(2.0 - std::sqrt(2.0)),
                                                             a21(gamma), a22(0.0), a31(0.5),
                                                             a32(0.5), a33(0.0),
                                                             a21_tilde(0.5*gamma), a22_tilde(0.5*gamma),
                                                             a31_tilde(std::sqrt(2)/4.0), a32_tilde(std::sqrt(2)/4.0),
                                                             a33_tilde(1.0 - std::sqrt(2)/2.0), b1(a31_tilde),
                                                             b2(a32_tilde), b3(a33_tilde), HYPERBOLIC_stage(1), NS_stage(1) {}


  // Setter of time-step
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void HYPERBOLICOperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  set_dt(const double time_step) {
    dt = time_step;
  }


  // Setter of Mach number
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void HYPERBOLICOperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  set_Mach(const double Ma_) {
    Ma = Ma_;
  }


  // Setter of Froude number
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void HYPERBOLICOperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  set_Froude(const double Fr_) {
    Fr = Fr_;
  }


  // Setter of HYPERBOLIC stage (this can be known only during the effective execution
  // and so it has to be demanded to the class that really solves the problem)
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void HYPERBOLICOperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
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
  void HYPERBOLICOperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
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
  void HYPERBOLICOperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  set_rho_for_fixed(const Vec& src) {
    rho_for_fixed = src;
    rho_for_fixed.update_ghost_values();
  }


  // Setter of pressure for fixed point
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void HYPERBOLICOperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  set_pres_fixed(const Vec& src) {
    pres_fixed = src;
    pres_fixed.update_ghost_values();
  }


  // Setter of velocity for fixed point
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void HYPERBOLICOperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  set_u_fixed(const Vec& src) {
    u_fixed = src;
    u_fixed.update_ghost_values();
  }


  // Assemble rhs cell term for the density update
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void HYPERBOLICOperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  assemble_rhs_cell_term_rho_update(const MatrixFree<dim, Number>&               data,
                                    Vec&                                         dst,
                                    const std::vector<Vec>&                      src,
                                    const std::pair<unsigned int, unsigned int>& cell_range) const {
    if(HYPERBOLIC_stage == 1) {
      FEEvaluation<dim, fe_degree_rho, n_q_points_1d_u, 1, Number> phi(data, 2),
                                                                   phi_rho_old(data, 2);
      FEEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number> phi_u_old(data, 0);

      for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
        phi_rho_old.reinit(cell);
        phi_rho_old.gather_evaluate(src[0], true, false);
        phi_u_old.reinit(cell);
        phi_u_old.gather_evaluate(src[1], true, false);

        phi.reinit(cell);

        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          const auto& rho_old = phi_rho_old.get_value(q);
          const auto& u_old   = phi_u_old.get_value(q);

          phi.submit_value(rho_old, q);
          phi.submit_gradient(a21*dt*rho_old*u_old, q);
        }
        phi.integrate_scatter(true, true, dst);
      }
    }
    else if(HYPERBOLIC_stage == 2) {
      FEEvaluation<dim, fe_degree_rho, n_q_points_1d_u, 1, Number> phi(data, 2),
                                                                   phi_rho_old(data, 2),
                                                                   phi_rho_tmp_2(data, 2);
      FEEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number> phi_u_old(data, 0),
                                                                   phi_u_tmp_2(data, 0);

      for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
        phi_rho_old.reinit(cell);
        phi_rho_old.gather_evaluate(src[0], true, false);
        phi_u_old.reinit(cell);
        phi_u_old.gather_evaluate(src[1], true, false);

        phi_rho_tmp_2.reinit(cell);
        phi_rho_tmp_2.gather_evaluate(src[2], true, false);
        phi_u_tmp_2.reinit(cell);
        phi_u_tmp_2.gather_evaluate(src[3], true, false);

        phi.reinit(cell);

        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          const auto& rho_old   = phi_rho_old.get_value(q);
          const auto& u_old     = phi_u_old.get_value(q);

          const auto& rho_tmp_2 = phi_rho_tmp_2.get_value(q);
          const auto& u_tmp_2   = phi_u_tmp_2.get_value(q);

          phi.submit_value(rho_old, q);
          phi.submit_gradient(a31*dt*rho_old*u_old +
                              a32*dt*rho_tmp_2*u_tmp_2, q);
        }
        phi.integrate_scatter(true, true, dst);
      }
    }
    else {
      FEEvaluation<dim, fe_degree_rho, n_q_points_1d_u, 1, Number> phi(data, 2),
                                                                   phi_rho_old(data, 2),
                                                                   phi_rho_tmp_2(data, 2),
                                                                   phi_rho_tmp_3(data, 2);
      FEEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number> phi_u_old(data, 0),
                                                                   phi_u_tmp_2(data, 0),
                                                                   phi_u_tmp_3(data, 0);

      for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
        phi_rho_old.reinit(cell);
        phi_rho_old.gather_evaluate(src[0], true, false);
        phi_u_old.reinit(cell);
        phi_u_old.gather_evaluate(src[1], true, false);

        phi_rho_tmp_2.reinit(cell);
        phi_rho_tmp_2.gather_evaluate(src[2], true, false);
        phi_u_tmp_2.reinit(cell);
        phi_u_tmp_2.gather_evaluate(src[3], true, false);

        phi_rho_tmp_3.reinit(cell);
        phi_rho_tmp_3.gather_evaluate(src[4], true, false);
        phi_u_tmp_3.reinit(cell);
        phi_u_tmp_3.gather_evaluate(src[5], true, false);

        phi.reinit(cell);

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
        phi.integrate_scatter(true, true, dst);
      }
    }
  }


  // Assemble rhs face term for the density update
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void HYPERBOLICOperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  assemble_rhs_face_term_rho_update(const MatrixFree<dim, Number>&               data,
                                    Vec&                                         dst,
                                    const std::vector<Vec>&                      src,
                                    const std::pair<unsigned int, unsigned int>& face_range) const {
    if(HYPERBOLIC_stage == 1) {
      FEFaceEvaluation<dim, fe_degree_rho, n_q_points_1d_u, 1, Number> phi_p(data, true, 2),
                                                                       phi_m(data, false, 2),
                                                                       phi_rho_old_p(data, true, 2),
                                                                       phi_rho_old_m(data, false, 2);
      FEFaceEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number> phi_u_old_p(data, true, 0),
                                                                       phi_u_old_m(data, false, 0);

      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        phi_rho_old_p.reinit(face);
        phi_rho_old_p.gather_evaluate(src[0], true, false);
        phi_rho_old_m.reinit(face);
        phi_rho_old_m.gather_evaluate(src[0], true, false);
        phi_u_old_p.reinit(face);
        phi_u_old_p.gather_evaluate(src[1], true, false);
        phi_u_old_m.reinit(face);
        phi_u_old_m.gather_evaluate(src[1], true, false);

        phi_p.reinit(face);
        phi_m.reinit(face);

        for(unsigned int q = 0; q < phi_p.n_q_points; ++q) {
          const auto& n_plus       = phi_p.get_normal_vector(q);

          const auto& avg_flux_old = 0.5*(phi_rho_old_p.get_value(q)*phi_u_old_p.get_value(q) +
                                          phi_rho_old_m.get_value(q)*phi_u_old_m.get_value(q));
          const auto& lambda_old   = std::max(std::abs(scalar_product(phi_u_old_p.get_value(q), n_plus)),
                                              std::abs(scalar_product(phi_u_old_m.get_value(q), n_plus)));
          const auto& jump_rho_old = phi_rho_old_p.get_value(q) - phi_rho_old_m.get_value(q);

          phi_p.submit_value(-a21*dt*(scalar_product(avg_flux_old, n_plus) + 0.5*lambda_old*jump_rho_old), q);
          phi_m.submit_value(a21*dt*(scalar_product(avg_flux_old, n_plus) + 0.5*lambda_old*jump_rho_old), q);
        }
        phi_p.integrate_scatter(true, false, dst);
        phi_m.integrate_scatter(true, false, dst);
      }
    }
    else if(HYPERBOLIC_stage == 2) {
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

      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        phi_rho_old_p.reinit(face);
        phi_rho_old_p.gather_evaluate(src[0], true, false);
        phi_rho_old_m.reinit(face);
        phi_rho_old_m.gather_evaluate(src[0], true, false);
        phi_u_old_p.reinit(face);
        phi_u_old_p.gather_evaluate(src[1], true, false);
        phi_u_old_m.reinit(face);
        phi_u_old_m.gather_evaluate(src[1], true, false);

        phi_rho_tmp_2_p.reinit(face);
        phi_rho_tmp_2_p.gather_evaluate(src[2], true, false);
        phi_rho_tmp_2_m.reinit(face);
        phi_rho_tmp_2_m.gather_evaluate(src[2], true, false);
        phi_u_tmp_2_p.reinit(face);
        phi_u_tmp_2_p.gather_evaluate(src[3], true, false);
        phi_u_tmp_2_m.reinit(face);
        phi_u_tmp_2_m.gather_evaluate(src[3], true, false);

        phi_p.reinit(face);
        phi_m.reinit(face);

        for(unsigned int q = 0; q < phi_p.n_q_points; ++q) {
          const auto& n_plus         = phi_p.get_normal_vector(q);

          const auto& avg_flux_old   = 0.5*(phi_rho_old_p.get_value(q)*phi_u_old_p.get_value(q) +
                                            phi_rho_old_m.get_value(q)*phi_u_old_m.get_value(q));
          const auto& lambda_old     = std::max(std::abs(scalar_product(phi_u_old_p.get_value(q), n_plus)),
                                                std::abs(scalar_product(phi_u_old_m.get_value(q), n_plus)));
          const auto& jump_rho_old   = phi_rho_old_p.get_value(q) - phi_rho_old_m.get_value(q);

          const auto& avg_flux_tmp_2 = 0.5*(phi_rho_tmp_2_p.get_value(q)*phi_u_tmp_2_p.get_value(q) +
                                            phi_rho_tmp_2_m.get_value(q)*phi_u_tmp_2_m.get_value(q));
          const auto& lambda_tmp_2   = std::max(std::abs(scalar_product(phi_u_tmp_2_p.get_value(q), n_plus)),
                                                std::abs(scalar_product(phi_u_tmp_2_m.get_value(q), n_plus)));
          const auto& jump_rho_tmp_2 = phi_rho_tmp_2_p.get_value(q) - phi_rho_tmp_2_m.get_value(q);

          phi_p.submit_value(-a31*dt*(scalar_product(avg_flux_old, n_plus) + 0.5*lambda_old*jump_rho_old)
                             -a32*dt*(scalar_product(avg_flux_tmp_2, n_plus) + 0.5*lambda_tmp_2*jump_rho_tmp_2), q);
          phi_m.submit_value(a31*dt*(scalar_product(avg_flux_old, n_plus) + 0.5*lambda_old*jump_rho_old) +
                             a32*dt*(scalar_product(avg_flux_tmp_2, n_plus) + 0.5*lambda_tmp_2*jump_rho_tmp_2), q);
        }
        phi_p.integrate_scatter(true, false, dst);
        phi_m.integrate_scatter(true, false, dst);
      }
    }
    else {
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

      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        phi_rho_old_p.reinit(face);
        phi_rho_old_p.gather_evaluate(src[0], true, false);
        phi_rho_old_m.reinit(face);
        phi_rho_old_m.gather_evaluate(src[0], true, false);
        phi_u_old_p.reinit(face);
        phi_u_old_p.gather_evaluate(src[1], true, false);
        phi_u_old_m.reinit(face);
        phi_u_old_m.gather_evaluate(src[1], true, false);

        phi_rho_tmp_2_p.reinit(face);
        phi_rho_tmp_2_p.gather_evaluate(src[2], true, false);
        phi_rho_tmp_2_m.reinit(face);
        phi_rho_tmp_2_m.gather_evaluate(src[2], true, false);
        phi_u_tmp_2_p.reinit(face);
        phi_u_tmp_2_p.gather_evaluate(src[3], true, false);
        phi_u_tmp_2_m.reinit(face);
        phi_u_tmp_2_m.gather_evaluate(src[3], true, false);

        phi_rho_tmp_3_p.reinit(face);
        phi_rho_tmp_3_p.gather_evaluate(src[4], true, false);
        phi_rho_tmp_3_m.reinit(face);
        phi_rho_tmp_3_m.gather_evaluate(src[4], true, false);
        phi_u_tmp_3_p.reinit(face);
        phi_u_tmp_3_p.gather_evaluate(src[5], true, false);
        phi_u_tmp_3_m.reinit(face);
        phi_u_tmp_3_m.gather_evaluate(src[5], true, false);

        phi_p.reinit(face);
        phi_m.reinit(face);

        for(unsigned int q = 0; q < phi_p.n_q_points; ++q) {
          const auto& n_plus         = phi_p.get_normal_vector(q);

          const auto& avg_flux_old   = 0.5*(phi_rho_old_p.get_value(q)*phi_u_old_p.get_value(q) +
                                            phi_rho_old_m.get_value(q)*phi_u_old_m.get_value(q));
          const auto& lambda_old     = std::max(std::abs(scalar_product(phi_u_old_p.get_value(q), n_plus)),
                                                std::abs(scalar_product(phi_u_old_m.get_value(q), n_plus)));
          const auto& jump_rho_old   = phi_rho_old_p.get_value(q) - phi_rho_old_m.get_value(q);

          const auto& avg_flux_tmp_2 = 0.5*(phi_rho_tmp_2_p.get_value(q)*phi_u_tmp_2_p.get_value(q) +
                                            phi_rho_tmp_2_m.get_value(q)*phi_u_tmp_2_m.get_value(q));
          const auto& lambda_tmp_2   = std::max(std::abs(scalar_product(phi_u_tmp_2_p.get_value(q), n_plus)),
                                                std::abs(scalar_product(phi_u_tmp_2_m.get_value(q), n_plus)));
          const auto& jump_rho_tmp_2 = phi_rho_tmp_2_p.get_value(q) - phi_rho_tmp_2_m.get_value(q);

          const auto& avg_flux_tmp_3 = 0.5*(phi_rho_tmp_3_p.get_value(q)*phi_u_tmp_3_p.get_value(q) +
                                            phi_rho_tmp_3_m.get_value(q)*phi_u_tmp_3_m.get_value(q));
          const auto& lambda_tmp_3   = std::max(std::abs(scalar_product(phi_u_tmp_3_p.get_value(q), n_plus)),
                                                std::abs(scalar_product(phi_u_tmp_3_m.get_value(q), n_plus)));
          const auto& jump_rho_tmp_3 = phi_rho_tmp_3_p.get_value(q) - phi_rho_tmp_3_m.get_value(q);

          phi_p.submit_value(-b1*dt*(scalar_product(avg_flux_old, n_plus) + 0.5*lambda_old*jump_rho_old)
                             -b2*dt*(scalar_product(avg_flux_tmp_2, n_plus) + 0.5*lambda_tmp_2*jump_rho_tmp_2)
                             -b3*dt*(scalar_product(avg_flux_tmp_3, n_plus) + 0.5*lambda_tmp_3*jump_rho_tmp_3), q);
          phi_m.submit_value(b1*dt*(scalar_product(avg_flux_old, n_plus) + 0.5*lambda_old*jump_rho_old) +
                             b2*dt*(scalar_product(avg_flux_tmp_2, n_plus) + 0.5*lambda_tmp_2*jump_rho_tmp_2) +
                             b3*dt*(scalar_product(avg_flux_tmp_3, n_plus) + 0.5*lambda_tmp_3*jump_rho_tmp_3), q);
        }
        phi_p.integrate_scatter(true, false, dst);
        phi_m.integrate_scatter(true, false, dst);
      }
    }
  }


  // Put together all the previous steps for density update
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void HYPERBOLICOperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  vmult_rhs_rho_update(Vec& dst, const std::vector<Vec>& src) const {
    for(unsigned int d = 0; d < src.size(); ++d)
      src[d].update_ghost_values();

    this->data->loop(&HYPERBOLICOperator::assemble_rhs_cell_term_rho_update,
                     &HYPERBOLICOperator::assemble_rhs_face_term_rho_update,
                     &HYPERBOLICOperator::assemble_rhs_boundary_term_rho_update,
                     this, dst, src, true,
                     MatrixFree<dim, Number>::DataAccessOnFaces::unspecified,
                     MatrixFree<dim, Number>::DataAccessOnFaces::unspecified);
  }


  // Assemble cell term for the density update
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void HYPERBOLICOperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  assemble_cell_term_rho_update(const MatrixFree<dim, Number>&               data,
                                    Vec&                                         dst,
                                    const Vec&                                   src,
                                    const std::pair<unsigned int, unsigned int>& cell_range) const {
    FEEvaluation<dim, fe_degree_rho, n_q_points_1d_u, 1, Number> phi(data, 2);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
      phi.reinit(cell);
      phi.gather_evaluate(src, true, false);

      for(unsigned int q = 0; q < phi.n_q_points; ++q)
        phi.submit_value(phi.get_value(q), q);

      phi.integrate_scatter(true, false, dst);
    }
  }


  // Assemble rhs cell term for the pressure
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void HYPERBOLICOperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  assemble_rhs_cell_term_pressure(const MatrixFree<dim, Number>&               data,
                                  Vec&                                         dst,
                                  const std::vector<Vec>&                      src,
                                  const std::pair<unsigned int, unsigned int>& cell_range) const {
    if(HYPERBOLIC_stage == 1) {
      FEEvaluation<dim, fe_degree_T, n_q_points_1d_u, 1, Number>   phi(data, 1),
                                                                   phi_pres_old(data, 1);
      FEEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number> phi_u_old(data, 0),
                                                                   phi_u_fixed(data, 0);
      FEEvaluation<dim, fe_degree_rho, n_q_points_1d_u, 1, Number> phi_rho_old(data, 2),
                                                                   phi_rho_tmp_2(data, 2);

      for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
        phi_rho_old.reinit(cell);
        phi_rho_old.gather_evaluate(src[0], true, false);
        phi_u_old.reinit(cell);
        phi_u_old.gather_evaluate(src[1], true, false);
        phi_pres_old.reinit(cell);
        phi_pres_old.gather_evaluate(src[2], true, false);

        phi_rho_tmp_2.reinit(cell);
        phi_rho_tmp_2.gather_evaluate(src[3], true, false);
        phi_u_fixed.reinit(cell);
        phi_u_fixed.gather_evaluate(src[4], true, false);

        phi.reinit(cell);

        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          const auto& rho_old   = phi_rho_old.get_value(q);
          const auto& u_old     = phi_u_old.get_value(q);
          const auto& pres_old  = phi_pres_old.get_value(q);
          const auto& E_old     = 1.0/(EquationData::Cp_Cv - 1.0)*(pres_old/rho_old)
                                + 0.5*Ma*Ma*scalar_product(u_old, u_old);

          const auto& rho_tmp_2 = phi_rho_tmp_2.get_value(q);
          const auto& u_fixed   = phi_u_fixed.get_value(q);

          phi.submit_value(rho_old*E_old -
                           0.5*rho_tmp_2*Ma*Ma*scalar_product(u_fixed, u_fixed) -
                           a21_tilde*dt*Ma*Ma/(Fr*Fr)*rho_old*u_old[dim - 1] -
                           a22_tilde*dt*Ma*Ma/(Fr*Fr)*rho_tmp_2*u_fixed[dim - 1], q);
          phi.submit_gradient(0.5*a21*dt*Ma*Ma*scalar_product(u_old, u_old)*rho_old*u_old +
                              a21_tilde*dt*((rho_old*(E_old - 0.5*Ma*Ma*scalar_product(u_old, u_old)) + pres_old)*u_old), q);
        }
        phi.integrate_scatter(true, true, dst);
      }
    }
    else if(HYPERBOLIC_stage == 2) {
      FEEvaluation<dim, fe_degree_T, n_q_points_1d_u, 1, Number>   phi(data, 1),
                                                                   phi_pres_old(data, 1),
                                                                   phi_pres_tmp_2(data, 1);
      FEEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number> phi_u_old(data, 0),
                                                                   phi_u_tmp_2(data, 0),
                                                                   phi_u_fixed(data, 0);
      FEEvaluation<dim, fe_degree_rho, n_q_points_1d_u, 1, Number> phi_rho_old(data, 2),
                                                                   phi_rho_tmp_2(data, 2),
                                                                   phi_rho_tmp_3(data, 2);

      for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
        phi_rho_old.reinit(cell);
        phi_rho_old.gather_evaluate(src[0], true, false);
        phi_u_old.reinit(cell);
        phi_u_old.gather_evaluate(src[1], true, false);
        phi_pres_old.reinit(cell);
        phi_pres_old.gather_evaluate(src[2], true, false);

        phi_rho_tmp_2.reinit(cell);
        phi_rho_tmp_2.gather_evaluate(src[3], true, false);
        phi_u_tmp_2.reinit(cell);
        phi_u_tmp_2.gather_evaluate(src[4], true, false);
        phi_pres_tmp_2.reinit(cell);
        phi_pres_tmp_2.gather_evaluate(src[5], true, false);

        phi_rho_tmp_3.reinit(cell);
        phi_rho_tmp_3.gather_evaluate(src[6], true, false);
        phi_u_fixed.reinit(cell);
        phi_u_fixed.gather_evaluate(src[7], true, false);

        phi.reinit(cell);

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
                              ((rho_old*(E_old - 0.5*Ma*Ma*scalar_product(u_old, u_old)) + pres_old)*u_old) +
                              0.5*a32*dt*Ma*Ma*scalar_product(u_tmp_2, u_tmp_2)*rho_tmp_2*u_tmp_2 +
                              a32_tilde*dt*
                              ((rho_tmp_2*(E_tmp_2 - 0.5*Ma*Ma*scalar_product(u_tmp_2, u_tmp_2)) + pres_tmp_2)*u_tmp_2), q);
        }
        phi.integrate_scatter(true, true, dst);
      }
    }
    else {
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

      for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
        phi_rho_old.reinit(cell);
        phi_rho_old.gather_evaluate(src[0], true, false);
        phi_u_old.reinit(cell);
        phi_u_old.gather_evaluate(src[1], true, false);
        phi_pres_old.reinit(cell);
        phi_pres_old.gather_evaluate(src[2], true, false);

        phi_rho_tmp_2.reinit(cell);
        phi_rho_tmp_2.gather_evaluate(src[3], true, false);
        phi_u_tmp_2.reinit(cell);
        phi_u_tmp_2.gather_evaluate(src[4], true, false);
        phi_pres_tmp_2.reinit(cell);
        phi_pres_tmp_2.gather_evaluate(src[5], true, false);

        phi_rho_tmp_3.reinit(cell);
        phi_rho_tmp_3.gather_evaluate(src[6], true, false);
        phi_u_tmp_3.reinit(cell);
        phi_u_tmp_3.gather_evaluate(src[7], true, false);
        phi_pres_tmp_3.reinit(cell);
        phi_pres_tmp_3.gather_evaluate(src[8], true, false);

        phi_rho_curr.reinit(cell);
        phi_rho_curr.gather_evaluate(src[9], true, false);
        phi_u_fixed.reinit(cell);
        phi_u_fixed.gather_evaluate(src[10], true, false);

        phi.reinit(cell);

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
                              b1*dt*((rho_old*(E_old - 0.5*Ma*Ma*scalar_product(u_old, u_old)) + pres_old)*u_old) +
                              0.5*b2*dt*Ma*Ma*scalar_product(u_tmp_2, u_tmp_2)*rho_tmp_2*u_tmp_2 +
                              b2*dt*((rho_tmp_2*(E_tmp_2 - 0.5*Ma*Ma*scalar_product(u_tmp_2, u_tmp_2)) + pres_tmp_2)*u_tmp_2) +
                              0.5*b3*dt*Ma*Ma*scalar_product(u_tmp_3, u_tmp_3)*rho_tmp_3*u_tmp_3 +
                              b3*dt*((rho_tmp_3*(E_tmp_3 - 0.5*Ma*Ma*scalar_product(u_tmp_3, u_tmp_3)) + pres_tmp_3)*u_tmp_3), q);
        }
        phi.integrate_scatter(true, true, dst);
      }
    }
  }


  // Assemble rhs face term for the pressure
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void HYPERBOLICOperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  assemble_rhs_face_term_pressure(const MatrixFree<dim, Number>&               data,
                                  Vec&                                         dst,
                                  const std::vector<Vec>&                      src,
                                  const std::pair<unsigned int, unsigned int>& face_range) const {
    if(HYPERBOLIC_stage == 1) {
      FEFaceEvaluation<dim, fe_degree_T, n_q_points_1d_u, 1, Number>   phi_p(data, true, 1),
                                                                       phi_m(data, false, 1),
                                                                       phi_pres_old_p(data, true, 1),
                                                                       phi_pres_old_m(data, false, 1);
      FEFaceEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number> phi_u_old_p(data, true, 0),
                                                                       phi_u_old_m(data, false, 0);
      FEFaceEvaluation<dim, fe_degree_rho, n_q_points_1d_u, 1, Number> phi_rho_old_p(data, true, 2),
                                                                       phi_rho_old_m(data, false, 2);

      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        phi_rho_old_p.reinit(face);
        phi_rho_old_p.gather_evaluate(src[0], true, false);
        phi_rho_old_m.reinit(face);
        phi_rho_old_m.gather_evaluate(src[0], true, false);
        phi_u_old_p.reinit(face);
        phi_u_old_p.gather_evaluate(src[1], true, false);
        phi_u_old_m.reinit(face);
        phi_u_old_m.gather_evaluate(src[1], true, false);
        phi_pres_old_p.reinit(face);
        phi_pres_old_p.gather_evaluate(src[2], true, false);
        phi_pres_old_m.reinit(face);
        phi_pres_old_m.gather_evaluate(src[2], true, false);

        phi_p.reinit(face);
        phi_m.reinit(face);

        for(unsigned int q = 0; q < phi_p.n_q_points; ++q) {
          const auto& n_plus           = phi_p.get_normal_vector(q);

          const auto& rho_old_p        = phi_rho_old_p.get_value(q);
          const auto& rho_old_m        = phi_rho_old_m.get_value(q);
          const auto& u_old_p          = phi_u_old_p.get_value(q);
          const auto& u_old_m          = phi_u_old_m.get_value(q);
          const auto& avg_kinetic_old  = 0.5*(0.5*scalar_product(u_old_p,u_old_p)*rho_old_p*u_old_p +
                                              0.5*scalar_product(u_old_m,u_old_m)*rho_old_m*u_old_m);
          const auto& pres_old_p       = phi_pres_old_p.get_value(q);
          const auto& pres_old_m       = phi_pres_old_m.get_value(q);
          const auto& E_old_p          = 1.0/(EquationData::Cp_Cv - 1.0)*pres_old_p/rho_old_p
                                       + 0.5*Ma*Ma*scalar_product(u_old_p, u_old_p);
          const auto& E_old_m          = 1.0/(EquationData::Cp_Cv - 1.0)*pres_old_m/rho_old_m
                                       + 0.5*Ma*Ma*scalar_product(u_old_m, u_old_m);
          const auto& avg_enthalpy_old = 0.5*(((E_old_p - 0.5*Ma*Ma*scalar_product(u_old_p,u_old_p))*rho_old_p + pres_old_p)*u_old_p +
                                              ((E_old_m - 0.5*Ma*Ma*scalar_product(u_old_m,u_old_m))*rho_old_m + pres_old_m)*u_old_m);
          const auto& lambda_old       = std::max(std::abs(scalar_product(u_old_p, n_plus)),
                                                  std::abs(scalar_product(u_old_m, n_plus)));
          const auto& jump_rhoE_old    = rho_old_p*E_old_p - rho_old_m*E_old_m;

          phi_p.submit_value(-a21*dt*Ma*Ma*scalar_product(avg_kinetic_old, n_plus)
                             -a21_tilde*dt*scalar_product(avg_enthalpy_old, n_plus)
                             -a21*dt*0.5*lambda_old*jump_rhoE_old, q);
          phi_m.submit_value(a21*dt*Ma*Ma*scalar_product(avg_kinetic_old, n_plus) +
                             a21_tilde*dt*scalar_product(avg_enthalpy_old, n_plus) +
                             a21*dt*0.5*lambda_old*jump_rhoE_old, q);
        }
        phi_p.integrate_scatter(true, false, dst);
        phi_m.integrate_scatter(true, false, dst);
      }
    }
    else if(HYPERBOLIC_stage == 2) {
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

      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        phi_rho_old_p.reinit(face);
        phi_rho_old_p.gather_evaluate(src[0], true, false);
        phi_rho_old_m.reinit(face);
        phi_rho_old_m.gather_evaluate(src[0], true, false);
        phi_u_old_p.reinit(face);
        phi_u_old_p.gather_evaluate(src[1], true, false);
        phi_u_old_m.reinit(face);
        phi_u_old_m.gather_evaluate(src[1], true, false);
        phi_pres_old_p.reinit(face);
        phi_pres_old_p.gather_evaluate(src[2], true, false);
        phi_pres_old_m.reinit(face);
        phi_pres_old_m.gather_evaluate(src[2], true, false);

        phi_rho_tmp_2_p.reinit(face);
        phi_rho_tmp_2_p.gather_evaluate(src[3], true, false);
        phi_rho_tmp_2_m.reinit(face);
        phi_rho_tmp_2_m.gather_evaluate(src[3], true, false);
        phi_u_tmp_2_p.reinit(face);
        phi_u_tmp_2_p.gather_evaluate(src[4], true, false);
        phi_u_tmp_2_m.reinit(face);
        phi_u_tmp_2_m.gather_evaluate(src[4], true, false);
        phi_pres_tmp_2_p.reinit(face);
        phi_pres_tmp_2_p.gather_evaluate(src[5], true, false);
        phi_pres_tmp_2_m.reinit(face);
        phi_pres_tmp_2_m.gather_evaluate(src[5], true, false);

        phi_p.reinit(face);
        phi_m.reinit(face);

        for(unsigned int q = 0; q < phi_p.n_q_points; ++q) {
          const auto& n_plus             = phi_p.get_normal_vector(q);

          const auto& rho_old_p          = phi_rho_old_p.get_value(q);
          const auto& rho_old_m          = phi_rho_old_m.get_value(q);
          const auto& u_old_p            = phi_u_old_p.get_value(q);
          const auto& u_old_m            = phi_u_old_m.get_value(q);
          const auto& avg_kinetic_old    = 0.5*(0.5*scalar_product(u_old_p,u_old_p)*rho_old_p*u_old_p +
                                                0.5*scalar_product(u_old_m,u_old_m)*rho_old_m*u_old_m);
          const auto& pres_old_p         = phi_pres_old_p.get_value(q);
          const auto& pres_old_m         = phi_pres_old_m.get_value(q);
          const auto& E_old_p            = 1.0/(EquationData::Cp_Cv - 1.0)*pres_old_p/rho_old_p
                                         + 0.5*Ma*Ma*scalar_product(u_old_p, u_old_p);
          const auto& E_old_m            = 1.0/(EquationData::Cp_Cv - 1.0)*pres_old_m/rho_old_m
                                         + 0.5*Ma*Ma*scalar_product(u_old_m, u_old_m);
          const auto& avg_enthalpy_old   = 0.5*
                                           (((E_old_p - 0.5*Ma*Ma*scalar_product(u_old_p,u_old_p))*rho_old_p + pres_old_p)*u_old_p +
                                            ((E_old_m - 0.5*Ma*Ma*scalar_product(u_old_m,u_old_m))*rho_old_m + pres_old_m)*u_old_m);
          const auto& lambda_old         = std::max(std::abs(scalar_product(u_old_p, n_plus)),
                                                    std::abs(scalar_product(u_old_m, n_plus)));
          const auto& jump_rhoE_old      = rho_old_p*E_old_p - rho_old_m*E_old_m;

          const auto& rho_tmp_2_p        = phi_rho_tmp_2_p.get_value(q);
          const auto& rho_tmp_2_m        = phi_rho_tmp_2_m.get_value(q);
          const auto& u_tmp_2_p          = phi_u_tmp_2_p.get_value(q);
          const auto& u_tmp_2_m          = phi_u_tmp_2_m.get_value(q);
          const auto& avg_kinetic_tmp_2  = 0.5*(0.5*scalar_product(u_tmp_2_p,u_tmp_2_p)*rho_tmp_2_p*u_tmp_2_p +
                                                0.5*scalar_product(u_tmp_2_m,u_tmp_2_m)*rho_tmp_2_m*u_tmp_2_m);
          const auto& pres_tmp_2_p       = phi_pres_tmp_2_p.get_value(q);
          const auto& pres_tmp_2_m       = phi_pres_tmp_2_m.get_value(q);
          const auto& E_tmp_2_p          = 1.0/(EquationData::Cp_Cv - 1.0)*pres_tmp_2_p/rho_tmp_2_p
                                         + 0.5*Ma*Ma*scalar_product(u_tmp_2_p, u_tmp_2_p);
          const auto& E_tmp_2_m          = 1.0/(EquationData::Cp_Cv - 1.0)*pres_tmp_2_m/rho_tmp_2_m
                                         + 0.5*Ma*Ma*scalar_product(u_tmp_2_m, u_tmp_2_m);
          const auto& avg_enthalpy_tmp_2 = 0.5*
                                           (((E_tmp_2_p - 0.5*Ma*Ma*scalar_product(u_tmp_2_p,u_tmp_2_p))*rho_tmp_2_p +
                                              pres_tmp_2_p)*u_tmp_2_p +
                                            ((E_tmp_2_m - 0.5*Ma*Ma*scalar_product(u_tmp_2_m,u_tmp_2_m))*rho_tmp_2_m +
                                              pres_tmp_2_m)*u_tmp_2_m);
          const auto& lambda_tmp_2       = std::max(std::abs(scalar_product(u_tmp_2_p, n_plus)),
                                                    std::abs(scalar_product(u_tmp_2_m, n_plus)));
          const auto& jump_rhoE_tmp_2    = rho_tmp_2_p*E_tmp_2_p - rho_tmp_2_m*E_tmp_2_m;

          phi_p.submit_value(-a31*dt*Ma*Ma*scalar_product(avg_kinetic_old, n_plus)
                             -a31_tilde*dt*scalar_product(avg_enthalpy_old, n_plus)
                             -a31*dt*0.5*lambda_old*jump_rhoE_old
                             -a32*dt*Ma*Ma*scalar_product(avg_kinetic_tmp_2, n_plus)
                             -a32_tilde*dt*scalar_product(avg_enthalpy_tmp_2, n_plus)
                             -a32*dt*0.5*lambda_tmp_2*jump_rhoE_tmp_2, q);
          phi_m.submit_value(a31*dt*Ma*Ma*scalar_product(avg_kinetic_old, n_plus) +
                             a31_tilde*dt*scalar_product(avg_enthalpy_old, n_plus) +
                             a31*dt*0.5*lambda_old*jump_rhoE_old +
                             a32*dt*Ma*Ma*scalar_product(avg_kinetic_tmp_2, n_plus) +
                             a32_tilde*dt*scalar_product(avg_enthalpy_tmp_2, n_plus) +
                             a32*dt*0.5*lambda_tmp_2*jump_rhoE_tmp_2, q);
        }
        phi_p.integrate_scatter(true, false, dst);
        phi_m.integrate_scatter(true, false, dst);
      }
    }
    else {
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

      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        phi_rho_old_p.reinit(face);
        phi_rho_old_p.gather_evaluate(src[0], true, false);
        phi_rho_old_m.reinit(face);
        phi_rho_old_m.gather_evaluate(src[0], true, false);
        phi_u_old_p.reinit(face);
        phi_u_old_p.gather_evaluate(src[1], true, false);
        phi_u_old_m.reinit(face);
        phi_u_old_m.gather_evaluate(src[1], true, false);
        phi_pres_old_p.reinit(face);
        phi_pres_old_p.gather_evaluate(src[2], true, false);
        phi_pres_old_m.reinit(face);
        phi_pres_old_m.gather_evaluate(src[2], true, false);

        phi_rho_tmp_2_p.reinit(face);
        phi_rho_tmp_2_p.gather_evaluate(src[3], true, false);
        phi_rho_tmp_2_m.reinit(face);
        phi_rho_tmp_2_m.gather_evaluate(src[3], true, false);
        phi_u_tmp_2_p.reinit(face);
        phi_u_tmp_2_p.gather_evaluate(src[4], true, false);
        phi_u_tmp_2_m.reinit(face);
        phi_u_tmp_2_m.gather_evaluate(src[4], true, false);
        phi_pres_tmp_2_p.reinit(face);
        phi_pres_tmp_2_p.gather_evaluate(src[5], true, false);
        phi_pres_tmp_2_m.reinit(face);
        phi_pres_tmp_2_m.gather_evaluate(src[5], true, false);

        phi_rho_tmp_3_p.reinit(face);
        phi_rho_tmp_3_p.gather_evaluate(src[6], true, false);
        phi_rho_tmp_3_m.reinit(face);
        phi_rho_tmp_3_m.gather_evaluate(src[6], true, false);
        phi_u_tmp_3_p.reinit(face);
        phi_u_tmp_3_p.gather_evaluate(src[7], true, false);
        phi_u_tmp_3_m.reinit(face);
        phi_u_tmp_3_m.gather_evaluate(src[7], true, false);
        phi_pres_tmp_3_p.reinit(face);
        phi_pres_tmp_3_p.gather_evaluate(src[8], true, false);
        phi_pres_tmp_3_m.reinit(face);
        phi_pres_tmp_3_m.gather_evaluate(src[8], true, false);

        phi_p.reinit(face);
        phi_m.reinit(face);

        for(unsigned int q = 0; q < phi_p.n_q_points; ++q) {
          const auto& n_plus             = phi_p.get_normal_vector(q);

          const auto& rho_old_p          = phi_rho_old_p.get_value(q);
          const auto& rho_old_m          = phi_rho_old_m.get_value(q);
          const auto& u_old_p            = phi_u_old_p.get_value(q);
          const auto& u_old_m            = phi_u_old_m.get_value(q);
          const auto& avg_kinetic_old    = 0.5*(0.5*scalar_product(u_old_p,u_old_p)*rho_old_p*u_old_p +
                                                0.5*scalar_product(u_old_m,u_old_m)*rho_old_m*u_old_m);
          const auto& pres_old_p         = phi_pres_old_p.get_value(q);
          const auto& pres_old_m         = phi_pres_old_m.get_value(q);
          const auto& E_old_p            = 1.0/(EquationData::Cp_Cv - 1.0)*pres_old_p/rho_old_p
                                         + 0.5*Ma*Ma*scalar_product(u_old_p, u_old_p);
          const auto& E_old_m            = 1.0/(EquationData::Cp_Cv - 1.0)*pres_old_m/rho_old_m
                                         + 0.5*Ma*Ma*scalar_product(u_old_m, u_old_m);
          const auto& avg_enthalpy_old   = 0.5*
                                           (((E_old_p - 0.5*Ma*Ma*scalar_product(u_old_p,u_old_p))*rho_old_p + pres_old_p)*u_old_p +
                                            ((E_old_m - 0.5*Ma*Ma*scalar_product(u_old_m,u_old_m))*rho_old_m + pres_old_m)*u_old_m);
          const auto& lambda_old         = std::max(std::abs(scalar_product(u_old_p, n_plus)),
                                                    std::abs(scalar_product(u_old_m, n_plus)));
          const auto& jump_rhoE_old      = rho_old_p*E_old_p - rho_old_m*E_old_m;

          const auto& rho_tmp_2_p        = phi_rho_tmp_2_p.get_value(q);
          const auto& rho_tmp_2_m        = phi_rho_tmp_2_m.get_value(q);
          const auto& u_tmp_2_p          = phi_u_tmp_2_p.get_value(q);
          const auto& u_tmp_2_m          = phi_u_tmp_2_m.get_value(q);
          const auto& avg_kinetic_tmp_2  = 0.5*(0.5*scalar_product(u_tmp_2_p,u_tmp_2_p)*rho_tmp_2_p*u_tmp_2_p +
                                                0.5*scalar_product(u_tmp_2_m,u_tmp_2_m)*rho_tmp_2_m*u_tmp_2_m);
          const auto& pres_tmp_2_p       = phi_pres_tmp_2_p.get_value(q);
          const auto& pres_tmp_2_m       = phi_pres_tmp_2_m.get_value(q);
          const auto& E_tmp_2_p          = 1.0/(EquationData::Cp_Cv - 1.0)*pres_tmp_2_p/rho_tmp_2_p
                                         + 0.5*Ma*Ma*scalar_product(u_tmp_2_p, u_tmp_2_p);
          const auto& E_tmp_2_m          = 1.0/(EquationData::Cp_Cv - 1.0)*pres_tmp_2_m/rho_tmp_2_m
                                         + 0.5*Ma*Ma*scalar_product(u_tmp_2_m, u_tmp_2_m);
          const auto& avg_enthalpy_tmp_2 = 0.5*
                                           (((E_tmp_2_p - 0.5*Ma*Ma*scalar_product(u_tmp_2_p,u_tmp_2_p))*rho_tmp_2_p +
                                              pres_tmp_2_p)*u_tmp_2_p +
                                            ((E_tmp_2_m - 0.5*Ma*Ma*scalar_product(u_tmp_2_m,u_tmp_2_m))*rho_tmp_2_m +
                                              pres_tmp_2_m)*u_tmp_2_m);
          const auto& lambda_tmp_2       = std::max(std::abs(scalar_product(u_tmp_2_p, n_plus)),
                                                    std::abs(scalar_product(u_tmp_2_m, n_plus)));
          const auto& jump_rhoE_tmp_2    = rho_tmp_2_p*E_tmp_2_p - rho_tmp_2_m*E_tmp_2_m;

          const auto& rho_tmp_3_p        = phi_rho_tmp_3_p.get_value(q);
          const auto& rho_tmp_3_m        = phi_rho_tmp_3_m.get_value(q);
          const auto& u_tmp_3_p          = phi_u_tmp_3_p.get_value(q);
          const auto& u_tmp_3_m          = phi_u_tmp_3_m.get_value(q);
          const auto& avg_kinetic_tmp_3  = 0.5*(0.5*scalar_product(u_tmp_3_p,u_tmp_3_p)*rho_tmp_3_p*u_tmp_3_p +
                                                0.5*scalar_product(u_tmp_3_m,u_tmp_3_m)*rho_tmp_3_m*u_tmp_3_m);
          const auto& pres_tmp_3_p       = phi_pres_tmp_3_p.get_value(q);
          const auto& pres_tmp_3_m       = phi_pres_tmp_3_m.get_value(q);
          const auto& E_tmp_3_p          = 1.0/(EquationData::Cp_Cv - 1.0)*pres_tmp_3_p/rho_tmp_3_p
                                         + 0.5*Ma*Ma*scalar_product(u_tmp_3_p, u_tmp_3_p);
          const auto& E_tmp_3_m          = 1.0/(EquationData::Cp_Cv - 1.0)*pres_tmp_3_m/rho_tmp_3_m
                                         + 0.5*Ma*Ma*scalar_product(u_tmp_3_m, u_tmp_3_m);
          const auto& avg_enthalpy_tmp_3 = 0.5*
                                           (((E_tmp_3_p - 0.5*Ma*Ma*scalar_product(u_tmp_3_p,u_tmp_3_p))*rho_tmp_3_p +
                                              pres_tmp_3_p)*u_tmp_3_p +
                                            ((E_tmp_3_m - 0.5*Ma*Ma*scalar_product(u_tmp_3_m,u_tmp_3_m))*rho_tmp_3_m +
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
        phi_p.integrate_scatter(true, false, dst);
        phi_m.integrate_scatter(true, false, dst);
      }
    }
  }


  // Put together all the previous steps for pressure
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void HYPERBOLICOperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  vmult_rhs_pressure(Vec& dst, const std::vector<Vec>& src) const {
    for(unsigned int d = 0; d < src.size(); ++d)
      src[d].update_ghost_values();

    this->data->loop(&HYPERBOLICOperator::assemble_rhs_cell_term_pressure,
                     &HYPERBOLICOperator::assemble_rhs_face_term_pressure,
                     &HYPERBOLICOperator::assemble_rhs_boundary_term_pressure,
                     this, dst, src, true,
                     MatrixFree<dim, Number>::DataAccessOnFaces::unspecified,
                     MatrixFree<dim, Number>::DataAccessOnFaces::unspecified);
  }


  // Assemble cell term for the contribution due to internal energy
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void HYPERBOLICOperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  assemble_cell_term_internal_energy(const MatrixFree<dim, Number>&               data,
                                     Vec&                                         dst,
                                     const Vec&                                   src,
                                     const std::pair<unsigned int, unsigned int>& cell_range) const {
    FEEvaluation<dim, fe_degree_T, n_q_points_1d_u, 1, Number> phi(data, 1);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
      phi.reinit(cell);
      phi.gather_evaluate(src, true, false);

      for(unsigned int q = 0; q < phi.n_q_points; ++q)
        phi.submit_value(1.0/(EquationData::Cp_Cv - 1.0)*phi.get_value(q), q);

      phi.integrate_scatter(true, false, dst);
    }
  }


  // Assemble cell term for the contribution due to enthalpy
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void HYPERBOLICOperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  assemble_cell_term_enthalpy(const MatrixFree<dim, Number>&               data,
                              Vec&                                         dst,
                              const Vec&                                   src,
                              const std::pair<unsigned int, unsigned int>& cell_range) const {
    FEEvaluation<dim, fe_degree_T, n_q_points_1d_u, 1, Number>   phi(data, 1),
                                                                 phi_pres_fixed(data, 1);
    FEEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number> phi_src(data, 0);

    const double coeff = (HYPERBOLIC_stage == 1) ? a22_tilde : a33_tilde;

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
      phi_pres_fixed.reinit(cell);
      phi_pres_fixed.gather_evaluate(pres_fixed, true, false);
      phi_src.reinit(cell);
      phi_src.gather_evaluate(src, true, false);
      phi.reinit(cell);

      for(unsigned int q = 0; q < phi.n_q_points; ++q) {
        const auto& pres_fixed = phi_pres_fixed.get_value(q);

        phi.submit_gradient(-coeff*dt*EquationData::Cp_Cv/(EquationData::Cp_Cv - 1.0)*pres_fixed*phi_src.get_value(q), q);
      }
      phi.integrate_scatter(false, true, dst);
    }
  }


  // Assemble face term for the contribution due to enthalpy
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void HYPERBOLICOperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
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

    const double coeff = (HYPERBOLIC_stage == 1) ? a22_tilde : a33_tilde;

    for(unsigned int face = face_range.first; face < face_range.second; ++face) {
      phi_pres_fixed_p.reinit(face);
      phi_pres_fixed_p.gather_evaluate(pres_fixed, true, false);
      phi_pres_fixed_m.reinit(face);
      phi_pres_fixed_m.gather_evaluate(pres_fixed, true, false);
      phi_src_p.reinit(face);
      phi_src_p.gather_evaluate(src, true, false);
      phi_src_m.reinit(face);
      phi_src_m.gather_evaluate(src, true, false);
      phi_p.reinit(face);
      phi_m.reinit(face);

      for(unsigned int q = 0; q < phi_p.n_q_points; ++q) {
        const auto& n_plus            = phi_p.get_normal_vector(q);

        const auto& pres_fixed_p      = phi_pres_fixed_p.get_value(q);
        const auto& pres_fixed_m      = phi_pres_fixed_m.get_value(q);
        const auto& avg_flux_enthalpy = 0.5*EquationData::Cp_Cv/(EquationData::Cp_Cv - 1.0)*
                                        (pres_fixed_p*phi_src_p.get_value(q) + pres_fixed_m*phi_src_m.get_value(q));

        phi_p.submit_value(coeff*dt*scalar_product(avg_flux_enthalpy, n_plus), q);
        phi_m.submit_value(-coeff*dt*scalar_product(avg_flux_enthalpy, n_plus), q);
      }
      phi_p.integrate_scatter(true, false, dst);
      phi_m.integrate_scatter(true, false, dst);
    }
  }


  // Assemble cell term for the pressure
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void HYPERBOLICOperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  assemble_cell_term_pressure(const MatrixFree<dim, Number>&               data,
                              Vec&                                         dst,
                              const Vec&                                   src,
                              const std::pair<unsigned int, unsigned int>& cell_range) const {
    FEEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number> phi(data, 0);
    FEEvaluation<dim, fe_degree_T, n_q_points_1d_u, 1, Number>   phi_src(data, 1);

    const double coeff = (HYPERBOLIC_stage == 1) ? a22_tilde : a33_tilde;

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
      phi_src.reinit(cell);
      phi_src.gather_evaluate(src, true, false);
      phi.reinit(cell);

      for(unsigned int q = 0; q < phi.n_q_points; ++q)
        phi.submit_divergence(-coeff*dt/(Ma*Ma)*phi_src.get_value(q), q);

      phi.integrate_scatter(false, true, dst);
    }
  }


  // Assemble face term for the pressure
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void HYPERBOLICOperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
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

    for(unsigned int face = face_range.first; face < face_range.second; ++face) {
      phi_src_p.reinit(face);
      phi_src_p.gather_evaluate(src, true, false);
      phi_src_m.reinit(face);
      phi_src_m.gather_evaluate(src, true, false);
      phi_p.reinit(face);
      phi_m.reinit(face);

      for(unsigned int q = 0; q < phi_p.n_q_points; ++q) {
        const auto& n_plus   = phi_p.get_normal_vector(q);

        const auto& avg_term = 0.5*(phi_src_p.get_value(q) + phi_src_m.get_value(q));

        phi_p.submit_value(coeff*dt/(Ma*Ma)*avg_term*n_plus, q);
        phi_m.submit_value(-coeff*dt/(Ma*Ma)*avg_term*n_plus, q);
      }
      phi_p.integrate_scatter(true, false, dst);
      phi_m.integrate_scatter(true, false, dst);
    }
  }


  // Assemble boundary term for the pressure
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void HYPERBOLICOperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  assemble_boundary_term_pressure(const MatrixFree<dim, Number>&               data,
                                  Vec&                                         dst,
                                  const Vec&                                   src,
                                  const std::pair<unsigned int, unsigned int>& face_range) const {
    FEFaceEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number>   phi(data, true, 0);
    FEFaceEvaluation<dim, fe_degree_T, n_q_points_1d_u, 1, Number>     phi_src(data, true, 1);

    const double coeff = (HYPERBOLIC_stage == 1) ? a22_tilde : a33_tilde;

    for(unsigned int face = face_range.first; face < face_range.second; ++face) {
      phi_src.reinit(face);
      phi_src.gather_evaluate(src, true, true);
      phi.reinit(face);

      for(unsigned int q = 0; q < phi.n_q_points; ++q) {
        const auto& n_plus       = phi.get_normal_vector(q);

        const auto& pres_fixed_D = phi_src.get_value(q);

        phi.submit_value(coeff*dt/(Ma*Ma)*0.5*(phi_src.get_value(q) + pres_fixed_D)*n_plus, q);
      }
      phi.integrate_scatter(true, false, dst);
    }
  }


  // Assemble rhs cell term for the velocity update
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void HYPERBOLICOperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  assemble_rhs_cell_term_velocity_fixed(const MatrixFree<dim, Number>&               data,
                                         Vec&                                         dst,
                                         const std::vector<Vec>&                      src,
                                         const std::pair<unsigned int, unsigned int>& cell_range) const {
    Tensor<1, dim, VectorizedArray<Number>> e_k;
    for(unsigned int d = 0; d < dim - 1; ++d)
      e_k[d] = make_vectorized_array<Number>(0.0);
    e_k[dim - 1] = make_vectorized_array<Number>(1.0);

    if(HYPERBOLIC_stage == 1) {
      FEEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number> phi(data, 0),
                                                                   phi_u_old(data, 0);
      FEEvaluation<dim, fe_degree_T, n_q_points_1d_u, 1, Number>   phi_pres_old(data, 1);
      FEEvaluation<dim, fe_degree_rho, n_q_points_1d_u, 1, Number> phi_rho_old(data, 2),
                                                                   phi_rho_tmp_2(data, 2);

      for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
        phi_rho_old.reinit(cell);
        phi_rho_old.gather_evaluate(src[0], true, true);
        phi_u_old.reinit(cell);
        phi_u_old.gather_evaluate(src[1], true, true);
        phi_pres_old.reinit(cell);
        phi_pres_old.gather_evaluate(src[2], true, false);

        phi_rho_tmp_2.reinit(cell);
        phi_rho_tmp_2.gather_evaluate(src[3], true, true);

        phi.reinit(cell);

        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          const auto& rho_old            = phi_rho_old.get_value(q);
          const auto& u_old              = phi_u_old.get_value(q);
          const auto& pres_old           = phi_pres_old.get_value(q);

          const auto& tensor_product_u_n = outer_product(u_old, u_old);
          auto p_n_times_identity        = tensor_product_u_n;
          p_n_times_identity = 0;
          for(unsigned int d = 0; d < dim; ++d)
            p_n_times_identity[d][d] = pres_old;

          const auto& rho_tmp_2 = phi_rho_tmp_2.get_value(q);

          phi.submit_value(rho_old*u_old -
                           a21_tilde*dt/(Fr*Fr)*rho_old*e_k -
                           a22_tilde*dt/(Fr*Fr)*rho_tmp_2*e_k, q);
          phi.submit_gradient(a21*dt*rho_old*tensor_product_u_n + a21_tilde*dt/(Ma*Ma)*p_n_times_identity, q);
        }
        phi.integrate_scatter(true, true, dst);
      }
    }
    else if(HYPERBOLIC_stage == 2) {
      FEEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number> phi(data, 0),
                                                                   phi_u_old(data, 0),
                                                                   phi_u_tmp_2(data, 0);
      FEEvaluation<dim, fe_degree_T, n_q_points_1d_u, 1, Number>   phi_pres_old(data, 1),
                                                                   phi_pres_tmp_2(data, 1);
      FEEvaluation<dim, fe_degree_rho, n_q_points_1d_u, 1, Number> phi_rho_old(data, 2),
                                                                   phi_rho_tmp_2(data, 2),
                                                                   phi_rho_curr(data, 2);

      for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
        phi_rho_old.reinit(cell);
        phi_rho_old.gather_evaluate(src[0], true, false);
        phi_u_old.reinit(cell);
        phi_u_old.gather_evaluate(src[1], true, false);
        phi_pres_old.reinit(cell);
        phi_pres_old.gather_evaluate(src[2], true, false);

        phi_rho_tmp_2.reinit(cell);
        phi_rho_tmp_2.gather_evaluate(src[3], true, false);
        phi_u_tmp_2.reinit(cell);
        phi_u_tmp_2.gather_evaluate(src[4], true, false);
        phi_pres_tmp_2.reinit(cell);
        phi_pres_tmp_2.gather_evaluate(src[5], true, false);

        phi_rho_curr.reinit(cell);
        phi_rho_curr.gather_evaluate(src[6], true, false);

        phi.reinit(cell);

        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          const auto& rho_old               = phi_rho_old.get_value(q);
          const auto& u_old                 = phi_u_old.get_value(q);
          const auto& pres_old              = phi_pres_old.get_value(q);

          const auto& tensor_product_u_n    = outer_product(u_old, u_old);
          auto p_n_times_identity           = tensor_product_u_n;
          p_n_times_identity = 0;
          for(unsigned int d = 0; d < dim; ++d)
            p_n_times_identity[d][d] = pres_old;

          const auto& rho_tmp_2              = phi_rho_tmp_2.get_value(q);
          const auto& u_tmp_2                = phi_u_tmp_2.get_value(q);
          const auto& pres_tmp_2             = phi_pres_tmp_2.get_value(q);

          const auto& tensor_product_u_tmp_2 = outer_product(u_tmp_2, u_tmp_2);
          auto p_tmp_2_times_identity        = tensor_product_u_tmp_2;
          p_tmp_2_times_identity = 0;
          for(unsigned int d = 0; d < dim; ++d)
            p_tmp_2_times_identity[d][d] = pres_tmp_2;

          const auto& rho_curr = phi_rho_curr.get_value(q);

          phi.submit_value(rho_old*u_old -
                           a31_tilde*dt/(Fr*Fr)*rho_old*e_k -
                           a32_tilde*dt/(Fr*Fr)*rho_tmp_2*e_k -
                           a33_tilde*dt/(Fr*Fr)*rho_curr*e_k, q);
          phi.submit_gradient(a31*dt*rho_old*tensor_product_u_n + a31_tilde*dt/(Ma*Ma)*p_n_times_identity +
                              a32*dt*rho_tmp_2*tensor_product_u_tmp_2 + a32_tilde*dt/(Ma*Ma)*p_tmp_2_times_identity, q);
        }
        phi.integrate_scatter(true, true, dst);
      }
    }
    else {
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

      for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
        phi_rho_old.reinit(cell);
        phi_rho_old.gather_evaluate(src[0], true, true);
        phi_u_old.reinit(cell);
        phi_u_old.gather_evaluate(src[1], true, true);
        phi_pres_old.reinit(cell);
        phi_pres_old.gather_evaluate(src[2], true, false);

        phi_rho_tmp_2.reinit(cell);
        phi_rho_tmp_2.gather_evaluate(src[3], true, true);
        phi_u_tmp_2.reinit(cell);
        phi_u_tmp_2.gather_evaluate(src[4], true, true);
        phi_pres_tmp_2.reinit(cell);
        phi_pres_tmp_2.gather_evaluate(src[5], true, false);

        phi_rho_tmp_3.reinit(cell);
        phi_rho_tmp_3.gather_evaluate(src[6], true, true);
        phi_u_tmp_3.reinit(cell);
        phi_u_tmp_3.gather_evaluate(src[7], true, true);
        phi_pres_tmp_3.reinit(cell);
        phi_pres_tmp_3.gather_evaluate(src[8], true, false);

        phi.reinit(cell);

        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          const auto& rho_old            = phi_rho_old.get_value(q);
          const auto& u_old              = phi_u_old.get_value(q);
          const auto& pres_old           = phi_pres_old.get_value(q);

          const auto& tensor_product_u_n = outer_product(u_old, u_old);
          auto p_n_times_identity        = tensor_product_u_n;
          p_n_times_identity = 0;
          for(unsigned int d = 0; d < dim; ++d)
            p_n_times_identity[d][d] = pres_old;

          const auto& rho_tmp_2              = phi_rho_tmp_2.get_value(q);
          const auto& u_tmp_2                = phi_u_tmp_2.get_value(q);
          const auto& pres_tmp_2             = phi_pres_tmp_2.get_value(q);

          const auto& tensor_product_u_tmp_2 = outer_product(u_tmp_2, u_tmp_2);
          auto p_tmp_2_times_identity        = tensor_product_u_tmp_2;
          p_tmp_2_times_identity = 0;
          for(unsigned int d = 0; d < dim; ++d)
            p_tmp_2_times_identity[d][d] = pres_tmp_2;

          const auto& rho_tmp_3              = phi_rho_tmp_3.get_value(q);
          const auto& u_tmp_3                = phi_u_tmp_3.get_value(q);
          const auto& pres_tmp_3             = phi_pres_tmp_3.get_value(q);

          const auto& tensor_product_u_tmp_3 = outer_product(u_tmp_3, u_tmp_3);
          auto p_tmp_3_times_identity        = tensor_product_u_tmp_3;
          p_tmp_3_times_identity = 0;
          for(unsigned int d = 0; d < dim; ++d)
            p_tmp_3_times_identity[d][d] = pres_tmp_3;

          phi.submit_value(rho_old*u_old -
                           b1*dt/(Fr*Fr)*rho_old*e_k -
                           b2*dt/(Fr*Fr)*rho_tmp_2*e_k -
                           b3*dt/(Fr*Fr)*rho_tmp_3*e_k, q);
          phi.submit_gradient(b1*dt*rho_old*tensor_product_u_n + b1*dt/(Ma*Ma)*p_n_times_identity +
                              b2*dt*rho_tmp_2*tensor_product_u_tmp_2 + b2*dt/(Ma*Ma)*p_tmp_2_times_identity +
                              b3*dt*rho_tmp_3*tensor_product_u_tmp_3 + b3*dt/(Ma*Ma)*p_tmp_3_times_identity, q);
        }
        phi.integrate_scatter(true, true, dst);
      }
    }
  }


  // Assemble rhs face term for the velocity update
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void HYPERBOLICOperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  assemble_rhs_face_term_velocity_fixed(const MatrixFree<dim, Number>&               data,
                                        Vec&                                         dst,
                                        const std::vector<Vec>&                      src,
                                        const std::pair<unsigned int, unsigned int>& face_range) const {
    if(HYPERBOLIC_stage == 1) {
      FEFaceEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number> phi_p(data, true, 0),
                                                                       phi_m(data, false, 0),
                                                                       phi_u_old_p(data, true, 0),
                                                                       phi_u_old_m(data, false, 0);
      FEFaceEvaluation<dim, fe_degree_T, n_q_points_1d_u, 1, Number>   phi_pres_old_p(data, true, 1),
                                                                       phi_pres_old_m(data, false, 1);
      FEFaceEvaluation<dim, fe_degree_rho, n_q_points_1d_u, 1, Number> phi_rho_old_p(data, true, 2),
                                                                       phi_rho_old_m(data, false, 2);

      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        phi_rho_old_p.reinit(face);
        phi_rho_old_p.gather_evaluate(src[0], true, false);
        phi_rho_old_m.reinit(face);
        phi_rho_old_m.gather_evaluate(src[0], true, false);
        phi_u_old_p.reinit(face);
        phi_u_old_p.gather_evaluate(src[1], true, false);
        phi_u_old_m.reinit(face);
        phi_u_old_m.gather_evaluate(src[1], true, false);
        phi_pres_old_p.reinit(face);
        phi_pres_old_p.gather_evaluate(src[2], true, false);
        phi_pres_old_m.reinit(face);
        phi_pres_old_m.gather_evaluate(src[2], true, false);

        phi_p.reinit(face);
        phi_m.reinit(face);

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
        phi_p.integrate_scatter(true, false, dst);
        phi_m.integrate_scatter(true, false, dst);
      }
    }
    else if(HYPERBOLIC_stage == 2) {
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

      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        phi_rho_old_p.reinit(face);
        phi_rho_old_p.gather_evaluate(src[0], true, false);
        phi_rho_old_m.reinit(face);
        phi_rho_old_m.gather_evaluate(src[0], true, false);
        phi_u_old_p.reinit(face);
        phi_u_old_p.gather_evaluate(src[1], true, false);
        phi_u_old_m.reinit(face);
        phi_u_old_m.gather_evaluate(src[1], true, false);
        phi_pres_old_p.reinit(face);
        phi_pres_old_p.gather_evaluate(src[2], true, false);
        phi_pres_old_m.reinit(face);
        phi_pres_old_m.gather_evaluate(src[2], true, false);

        phi_rho_tmp_2_p.reinit(face);
        phi_rho_tmp_2_p.gather_evaluate(src[3], true, false);
        phi_rho_tmp_2_m.reinit(face);
        phi_rho_tmp_2_m.gather_evaluate(src[3], true, false);
        phi_u_tmp_2_p.reinit(face);
        phi_u_tmp_2_p.gather_evaluate(src[4], true, false);
        phi_u_tmp_2_m.reinit(face);
        phi_u_tmp_2_m.gather_evaluate(src[4], true, false);
        phi_pres_tmp_2_p.reinit(face);
        phi_pres_tmp_2_p.gather_evaluate(src[5], true, false);
        phi_pres_tmp_2_m.reinit(face);
        phi_pres_tmp_2_m.gather_evaluate(src[5], true, false);

        phi_p.reinit(face);
        phi_m.reinit(face);

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
        phi_p.integrate_scatter(true, false, dst);
        phi_m.integrate_scatter(true, false, dst);
      }
    }
    else {
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

      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        phi_rho_old_p.reinit(face);
        phi_rho_old_p.gather_evaluate(src[0], true, false);
        phi_rho_old_m.reinit(face);
        phi_rho_old_m.gather_evaluate(src[0], true, false);
        phi_u_old_p.reinit(face);
        phi_u_old_p.gather_evaluate(src[1], true, false);
        phi_u_old_m.reinit(face);
        phi_u_old_m.gather_evaluate(src[1], true, false);
        phi_pres_old_p.reinit(face);
        phi_pres_old_p.gather_evaluate(src[2], true, false);
        phi_pres_old_m.reinit(face);
        phi_pres_old_m.gather_evaluate(src[2], true, false);

        phi_rho_tmp_2_p.reinit(face);
        phi_rho_tmp_2_p.gather_evaluate(src[3], true, false);
        phi_rho_tmp_2_m.reinit(face);
        phi_rho_tmp_2_m.gather_evaluate(src[3], true, false);
        phi_u_tmp_2_p.reinit(face);
        phi_u_tmp_2_p.gather_evaluate(src[4], true, false);
        phi_u_tmp_2_m.reinit(face);
        phi_u_tmp_2_m.gather_evaluate(src[4], true, false);
        phi_pres_tmp_2_p.reinit(face);
        phi_pres_tmp_2_p.gather_evaluate(src[5], true, false);
        phi_pres_tmp_2_m.reinit(face);
        phi_pres_tmp_2_m.gather_evaluate(src[5], true, false);

        phi_rho_tmp_3_p.reinit(face);
        phi_rho_tmp_3_p.gather_evaluate(src[6], true, false);
        phi_rho_tmp_3_m.reinit(face);
        phi_rho_tmp_3_m.gather_evaluate(src[6], true, false);
        phi_u_tmp_3_p.reinit(face);
        phi_u_tmp_3_p.gather_evaluate(src[7], true, false);
        phi_u_tmp_3_m.reinit(face);
        phi_u_tmp_3_m.gather_evaluate(src[7], true, false);
        phi_pres_tmp_3_p.reinit(face);
        phi_pres_tmp_3_p.gather_evaluate(src[8], true, false);
        phi_pres_tmp_3_m.reinit(face);
        phi_pres_tmp_3_m.gather_evaluate(src[8], true, false);

        phi_p.reinit(face);
        phi_m.reinit(face);

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
        phi_p.integrate_scatter(true, false, dst);
        phi_m.integrate_scatter(true, false, dst);
      }
    }
  }


  // Assemble rhs boundary term for the velocity update
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void HYPERBOLICOperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  assemble_rhs_boundary_term_velocity_fixed(const MatrixFree<dim, Number>&               data,
                                             Vec&                                         dst,
                                             const std::vector<Vec>&                      src,
                                             const std::pair<unsigned int, unsigned int>& face_range) const {
    if(HYPERBOLIC_stage == 1) {
      FEFaceEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number> phi(data, true, 0);
      FEFaceEvaluation<dim, fe_degree_T, n_q_points_1d_u, 1, Number>   phi_pres_old(data, true, 1);

      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        phi_pres_old.reinit(face);
        phi_pres_old.gather_evaluate(src[2], true, false);

        phi.reinit(face);

        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          const auto& n_plus       = phi.get_normal_vector(q);

          const auto& pres_old     = phi_pres_old.get_value(q);
          const auto& pres_old_D   = pres_old;

          const auto& avg_pres_old = 0.5*(pres_old + pres_old_D);

          phi.submit_value(-a21_tilde*dt/(Ma*Ma)*avg_pres_old*n_plus, q);
        }
        phi.integrate_scatter(true, false, dst);
      }
    }
    else if(HYPERBOLIC_stage == 2) {
      FEFaceEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number> phi(data, true, 0);
      FEFaceEvaluation<dim, fe_degree_T, n_q_points_1d_u, 1, Number>   phi_pres_old(data, true, 1),
                                                                       phi_pres_tmp_2(data, true, 1);

      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        phi_pres_old.reinit(face);
        phi_pres_old.gather_evaluate(src[2], true, false);

        phi_pres_tmp_2.reinit(face);
        phi_pres_tmp_2.gather_evaluate(src[5], true, false);

        phi.reinit(face);

        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          const auto& n_plus         = phi.get_normal_vector(q);

          const auto& pres_old       = phi_pres_old.get_value(q);
          const auto& pres_old_D     = pres_old;

          const auto& avg_pres_old   = 0.5*(pres_old + pres_old_D);

          const auto& pres_tmp_2     = phi_pres_tmp_2.get_value(q);
          const auto& pres_tmp_2_D   = pres_tmp_2;

          const auto& avg_pres_tmp_2 = 0.5*(pres_tmp_2 + pres_tmp_2_D);

          phi.submit_value(-a31_tilde*dt/(Ma*Ma)*avg_pres_old*n_plus
                           -a32_tilde*dt/(Ma*Ma)*avg_pres_tmp_2*n_plus, q);
        }
        phi.integrate_scatter(true, false, dst);
      }
    }
    else {
      FEFaceEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number> phi(data, true, 0);
      FEFaceEvaluation<dim, fe_degree_T, n_q_points_1d_u, 1, Number>   phi_pres_old(data, true, 1),
                                                                       phi_pres_tmp_2(data, true, 1),
                                                                       phi_pres_tmp_3(data, true, 1);

      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        phi_pres_old.reinit(face);
        phi_pres_old.gather_evaluate(src[2], true, false);

        phi_pres_tmp_2.reinit(face);
        phi_pres_tmp_2.gather_evaluate(src[5], true, false);

        phi_pres_tmp_3.reinit(face);
        phi_pres_tmp_3.gather_evaluate(src[8], true, false);

        phi.reinit(face);

        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          const auto& n_plus         = phi.get_normal_vector(q);

          const auto& pres_old       = phi_pres_old.get_value(q);
          const auto& pres_old_D     = pres_old;

          const auto& avg_pres_old   = 0.5*(pres_old + pres_old_D);

          const auto& pres_tmp_2     = phi_pres_tmp_2.get_value(q);
          const auto& pres_tmp_2_D   = pres_tmp_2;

          const auto& avg_pres_tmp_2 = 0.5*(pres_tmp_2 + pres_tmp_2_D);

          const auto& pres_tmp_3     = phi_pres_tmp_3.get_value(q);
          const auto& pres_tmp_3_D   = pres_tmp_3;

          const auto& avg_pres_tmp_3 = 0.5*(pres_tmp_3 + pres_tmp_3_D);

          phi.submit_value(-b1*dt/(Ma*Ma)*avg_pres_old*n_plus
                           -b2*dt/(Ma*Ma)*avg_pres_tmp_2*n_plus
                           -b3*dt/(Ma*Ma)*avg_pres_tmp_3*n_plus, q);
        }
        phi.integrate_scatter(true, false, dst);
      }
    }
  }


  // Put together all the previous steps for velocity update
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void HYPERBOLICOperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  vmult_rhs_velocity_fixed(Vec& dst, const std::vector<Vec>& src) const {
    for(unsigned int d = 0; d < src.size(); ++d)
      src[d].update_ghost_values();

    this->data->loop(&HYPERBOLICOperator::assemble_rhs_cell_term_velocity_fixed,
                     &HYPERBOLICOperator::assemble_rhs_face_term_velocity_fixed,
                     &HYPERBOLICOperator::assemble_rhs_boundary_term_velocity_fixed,
                     this, dst, src, true,
                     MatrixFree<dim, Number>::DataAccessOnFaces::unspecified,
                     MatrixFree<dim, Number>::DataAccessOnFaces::unspecified);
  }


  // Assemble cell term for the velocity update
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void HYPERBOLICOperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  assemble_cell_term_velocity_fixed(const MatrixFree<dim, Number>&               data,
                                     Vec&                                         dst,
                                     const Vec&                                   src,
                                     const std::pair<unsigned int, unsigned int>& cell_range) const {
    FEEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number> phi(data, 0);
    FEEvaluation<dim, fe_degree_rho, n_q_points_1d_u, 1, Number> phi_rho_for_fixed(data, 2);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
      phi_rho_for_fixed.reinit(cell);
      phi_rho_for_fixed.gather_evaluate(rho_for_fixed, true, false);
      phi.reinit(cell);
      phi.gather_evaluate(src, true, false);

      for(unsigned int q = 0; q < phi.n_q_points; ++q)
        phi.submit_value(phi_rho_for_fixed.get_value(q)*phi.get_value(q), q);

      phi.integrate_scatter(true, false, dst);
    }
  }


  // Put together all previous steps
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void HYPERBOLICOperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  apply_add(Vec& dst, const Vec& src) const {
    AssertIndexRange(NS_stage, 7);
    Assert(NS_stage > 0, ExcInternalError());

    if(NS_stage == 1 || NS_stage == 4) {
      this->data->cell_loop(&HYPERBOLICOperator::assemble_cell_term_rho_update,
                            this, dst, src, false);
    }
    else if(NS_stage == 2) {
      this->data->cell_loop(&HYPERBOLICOperator::assemble_cell_term_internal_energy,
                            this, dst, src, false);

      Vec tmp_1;
      this->data->initialize_dof_vector(tmp_1, 0);
      this->vmult_pressure(tmp_1, src);

      NS_stage = 3;
      const std::vector<unsigned int> tmp_reinit = {0};
      auto* tmp_matrix = const_cast<HYPERBOLICOperator*>(this);
      Vec tmp_2;
      this->data->initialize_dof_vector(tmp_2, 0);
      tmp_2 = 0;
      tmp_matrix->initialize(tmp_matrix->get_matrix_free(), tmp_reinit, tmp_reinit);

      SolverControl solver_control(10000, 1e-12*tmp_1.l2_norm());
      SolverCG<Vec> cg(solver_control);
      PreconditionJacobi<HYPERBOLICOperator> preconditioner_Jacobi;
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
      this->data->cell_loop(&HYPERBOLICOperator::assemble_cell_term_velocity_fixed,
                            this, dst, src, false);
    }
    else {
      this->data->cell_loop(&HYPERBOLICOperator::assemble_cell_term_internal_energy,
                            this, dst, src, false);
    }
  }


  // Application of pressure matrix
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void HYPERBOLICOperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  vmult_pressure(Vec& dst, const Vec& src) const {
    src.update_ghost_values();

    this->data->loop(&HYPERBOLICOperator::assemble_cell_term_pressure,
                     &HYPERBOLICOperator::assemble_face_term_pressure,
                     &HYPERBOLICOperator::assemble_boundary_term_pressure,
                     this, dst, src, true,
                     MatrixFree<dim, Number>::DataAccessOnFaces::unspecified,
                     MatrixFree<dim, Number>::DataAccessOnFaces::unspecified);
  }


  // Application of enthalpy matrix
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void HYPERBOLICOperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  vmult_enthalpy(Vec& dst, const Vec& src) const {
    src.update_ghost_values();

    this->data->loop(&HYPERBOLICOperator::assemble_cell_term_enthalpy,
                     &HYPERBOLICOperator::assemble_face_term_enthalpy,
                     &HYPERBOLICOperator::assemble_boundary_term_enthalpy,
                     this, dst, src, true,
                     MatrixFree<dim, Number>::DataAccessOnFaces::unspecified,
                     MatrixFree<dim, Number>::DataAccessOnFaces::unspecified);
  }


  // Assemble diagonal cell term for the rho projection
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void HYPERBOLICOperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  assemble_diagonal_cell_term_rho_update(const MatrixFree<dim, Number>&               data,
                                         Vec&                                         dst,
                                         const Vec&                                   src,
                                         const std::pair<unsigned int, unsigned int>& cell_range) const {
    FEEvaluation<dim, fe_degree_rho, n_q_points_1d_u, 1, Number> phi(data, 2);

    AlignedVector<VectorizedArray<Number>> diagonal(phi.dofs_per_component);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
      phi.reinit(cell);

      for(unsigned int i = 0; i < phi.dofs_per_component; ++i) {
        for(unsigned int j = 0; j < phi.dofs_per_component; ++j)
          phi.submit_dof_value(VectorizedArray<Number>(), j);
        phi.submit_dof_value(make_vectorized_array<Number>(1.0), i);
        phi.evaluate(true, false);

        for(unsigned int q = 0; q < phi.n_q_points; ++q)
          phi.submit_value(phi.get_value(q), q);

        phi.integrate(true, false);
        diagonal[i] = phi.get_dof_value(i);
      }
      for(unsigned int i = 0; i < phi.dofs_per_component; ++i)
        phi.submit_dof_value(diagonal[i], i);
      phi.distribute_local_to_global(dst);
    }
  }


  // Assemble diagonal cell term for the velocity update
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void HYPERBOLICOperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  assemble_diagonal_cell_term_velocity_fixed(const MatrixFree<dim, Number>&               data,
                                              Vec&                                         dst,
                                              const Vec&                                   src,
                                              const std::pair<unsigned int, unsigned int>& cell_range) const {
    FEEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number> phi(data, 0);
    FEEvaluation<dim, fe_degree_rho, n_q_points_1d_u, 1, Number> phi_rho_for_fixed(data, 2);

    AlignedVector<Tensor<1, dim, VectorizedArray<Number>>> diagonal(phi.dofs_per_component);
    Tensor<1, dim, VectorizedArray<Number>> tmp;
    for(unsigned int d = 0; d < dim; ++d)
      tmp[d] = make_vectorized_array<Number>(1.0);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
      phi_rho_for_fixed.reinit(cell);
      phi_rho_for_fixed.gather_evaluate(rho_for_fixed, true, false);
      phi.reinit(cell);

      for(unsigned int i = 0; i < phi.dofs_per_component; ++i) {
        for(unsigned int j = 0; j < phi.dofs_per_component; ++j)
          phi.submit_dof_value(Tensor<1, dim, VectorizedArray<Number>>(), j);
        phi.submit_dof_value(tmp, i);
        phi.evaluate(true, false);

        for(unsigned int q = 0; q < phi.n_q_points; ++q)
          phi.submit_value(phi_rho_for_fixed.get_value(q)*phi.get_value(q), q);

        phi.integrate(true, false);
        diagonal[i] = phi.get_dof_value(i);
      }
      for(unsigned int i = 0; i < phi.dofs_per_component; ++i)
        phi.submit_dof_value(diagonal[i], i);
      phi.distribute_local_to_global(dst);
    }
  }


  // Assemble diagonal cell term for the contribution due to internal energy
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void HYPERBOLICOperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  assemble_diagonal_cell_term_internal_energy(const MatrixFree<dim, Number>&               data,
                                              Vec&                                         dst,
                                              const Vec&                                   src,
                                              const std::pair<unsigned int, unsigned int>& cell_range) const {
    FEEvaluation<dim, fe_degree_T, n_q_points_1d_u, 1, Number> phi(data, 1);

    AlignedVector<VectorizedArray<Number>> diagonal(phi.dofs_per_component);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
      phi.reinit(cell);

      for(unsigned int i = 0; i < phi.dofs_per_component; ++i) {
        for(unsigned int j = 0; j < phi.dofs_per_component; ++j)
          phi.submit_dof_value(VectorizedArray<Number>(), j);
        phi.submit_dof_value(make_vectorized_array<Number>(1.0), i);
        phi.evaluate(true, false);

        for(unsigned int q = 0; q < phi.n_q_points; ++q)
          phi.submit_value(1.0/(EquationData::Cp_Cv - 1.0)*phi.get_value(q), q);

        phi.integrate(true, false);
        diagonal[i] = phi.get_dof_value(i);
      }
      for(unsigned int i = 0; i < phi.dofs_per_component; ++i)
        phi.submit_dof_value(diagonal[i], i);
      phi.distribute_local_to_global(dst);
    }
  }


  //Compute diagonal of various steps
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void HYPERBOLICOperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  compute_diagonal() {
    if(NS_stage == 1 || NS_stage == 4) {
      this->inverse_diagonal_entries.reset(new DiagonalMatrix<Vec>());
      auto& inverse_diagonal = this->inverse_diagonal_entries->get_vector();
      this->data->initialize_dof_vector(inverse_diagonal, 2);
      Vec dummy;
      dummy.reinit(inverse_diagonal.local_size());

      this->data->loop(&HYPERBOLICOperator::assemble_diagonal_cell_term_rho_update,
                       &HYPERBOLICOperator::assemble_diagonal_face_term_rho_update,
                       &HYPERBOLICOperator::assemble_diagonal_boundary_term_rho_update,
                       this, inverse_diagonal, dummy, false,
                       MatrixFree<dim, Number>::DataAccessOnFaces::unspecified,
                       MatrixFree<dim, Number>::DataAccessOnFaces::unspecified);

      for(unsigned int i = 0; i < inverse_diagonal.local_size(); ++i) {
        Assert(inverse_diagonal.local_element(i) != 0.0,
               ExcMessage("No diagonal entry in a definite operator should be zero"));
        inverse_diagonal.local_element(i) = 1.0/inverse_diagonal.local_element(i);
      }
    }
    else if(NS_stage == 2 || NS_stage == 6) {
      this->inverse_diagonal_entries.reset(new DiagonalMatrix<Vec>());
      auto& inverse_diagonal = this->inverse_diagonal_entries->get_vector();
      this->data->initialize_dof_vector(inverse_diagonal, 1);
      Vec dummy;
      dummy.reinit(inverse_diagonal.local_size());

      this->data->loop(&HYPERBOLICOperator::assemble_diagonal_cell_term_internal_energy,
                       &HYPERBOLICOperator::assemble_diagonal_face_term_internal_energy,
                       &HYPERBOLICOperator::assemble_diagonal_boundary_term_internal_energy,
                       this, inverse_diagonal, dummy, false,
                       MatrixFree<dim, Number>::DataAccessOnFaces::unspecified,
                       MatrixFree<dim, Number>::DataAccessOnFaces::unspecified);

      for(unsigned int i = 0; i < inverse_diagonal.local_size(); ++i) {
        Assert(inverse_diagonal.local_element(i) != 0.0,
               ExcMessage("No diagonal entry in a definite operator should be zero"));
        inverse_diagonal.local_element(i) = 1.0/inverse_diagonal.local_element(i);
      }
    }
    else {
      this->inverse_diagonal_entries.reset(new DiagonalMatrix<Vec>());
      auto& inverse_diagonal = this->inverse_diagonal_entries->get_vector();
      this->data->initialize_dof_vector(inverse_diagonal, 0);
      Vec dummy;
      dummy.reinit(inverse_diagonal.local_size());

      this->data->loop(&HYPERBOLICOperator::assemble_diagonal_cell_term_velocity_fixed,
                       &HYPERBOLICOperator::assemble_diagonal_face_term_velocity_fixed,
                       &HYPERBOLICOperator::assemble_diagonal_boundary_term_velocity_fixed,
                       this, inverse_diagonal, dummy, false,
                       MatrixFree<dim, Number>::DataAccessOnFaces::unspecified,
                       MatrixFree<dim, Number>::DataAccessOnFaces::unspecified);

      for(unsigned int i = 0; i < inverse_diagonal.local_size(); ++i) {
        Assert(inverse_diagonal.local_element(i) != 0.0,
               ExcMessage("No diagonal entry in a definite operator should be zero"));
        inverse_diagonal.local_element(i) = 1.0/inverse_diagonal.local_element(i);
      }
    }
  }


  // @sect{The <code>EulerSolver</code> class}

  // Now for the main class of the program. It implements the solver for the
  // Euler equations using the discretization previously implemented.
  //
  template<int dim>
  class EulerSolver {
  public:
    EulerSolver(RunTimeParameters::Data_Storage& data); /*--- Class constructor ---*/

    void run(const bool verbose = false, const unsigned int output_interval = 10); /*--- The run function which actually runs the simulation ---*/

  protected:
    const double t_0;              /*--- Initial time auxiliary variable ----*/
    const double T;                /*--- Final time auxiliary variable ----*/
    unsigned int HYPERBOLIC_stage; /*--- Flag to check at which current stage of the IMEX we are ---*/
    const double Ma;               /*--- Mach number auxiliary variable ----*/
    const double Fr;               /*--- Froude number auxiliary variable ---*/
    double       dt;               /*--- Time step auxiliary variable ---*/

    parallel::distributed::Triangulation<dim> triangulation;

    FESystem<dim> fe_density;
    FESystem<dim> fe_velocity;
    FESystem<dim> fe_temperature;

    DoFHandler<dim> dof_handler_density;
    DoFHandler<dim> dof_handler_velocity;
    DoFHandler<dim> dof_handler_temperature;

    QGaussLobatto<dim> quadrature_density;
    QGaussLobatto<dim> quadrature_velocity;
    QGaussLobatto<dim> quadrature_temperature;

    LinearAlgebra::distributed::Vector<double> rho_old;
    LinearAlgebra::distributed::Vector<double> rho_tmp_2;
    LinearAlgebra::distributed::Vector<double> rho_tmp_3;
    LinearAlgebra::distributed::Vector<double> rho_curr;
    LinearAlgebra::distributed::Vector<double> rhs_rho;

    LinearAlgebra::distributed::Vector<double> u_old;
    LinearAlgebra::distributed::Vector<double> u_tmp_2;
    LinearAlgebra::distributed::Vector<double> u_tmp_3;
    LinearAlgebra::distributed::Vector<double> u_curr;
    LinearAlgebra::distributed::Vector<double> u_fixed;
    LinearAlgebra::distributed::Vector<double> rhs_u;

    LinearAlgebra::distributed::Vector<double> pres_old;
    LinearAlgebra::distributed::Vector<double> pres_tmp_2;
    LinearAlgebra::distributed::Vector<double> pres_tmp_3;
    LinearAlgebra::distributed::Vector<double> pres_fixed;
    LinearAlgebra::distributed::Vector<double> pres_fixed_old;
    LinearAlgebra::distributed::Vector<double> pres_tmp;
    LinearAlgebra::distributed::Vector<double> rhs_pres;

    LinearAlgebra::distributed::Vector<double> tmp_1; /*--- Auxiliary vector for the Schur complement ---*/

    LinearAlgebra::distributed::Vector<double> rho_bar;
    LinearAlgebra::distributed::Vector<double> u_bar;
    LinearAlgebra::distributed::Vector<double> pres_bar;

    DeclException2(ExcInvalidTimeStep,
                   double,
                   double,
                   << " The time step " << arg1 << " is out of range."
                   << std::endl
                   << " The permitted range is (0," << arg2 << "]");

    void create_triangulation(const unsigned int n_refines);

    void setup_dofs();

    void initialize();

    void update_density();

    void pressure_fixed_point();

    void update_velocity();

    void update_pressure();

    void output_results(const unsigned int step);

  private:
    LinearAlgebra::distributed::Vector<double> dt_tau_rho;
    LinearAlgebra::distributed::Vector<double> dt_tau_u;
    LinearAlgebra::distributed::Vector<double> dt_tau_pres;
    LinearAlgebra::distributed::Vector<double> dt_tau_rho_aux;
    LinearAlgebra::distributed::Vector<double> dt_tau_u_aux;
    LinearAlgebra::distributed::Vector<double> dt_tau_pres_aux;

    LinearAlgebra::distributed::Vector<double> dt_tau_rho_right;
    LinearAlgebra::distributed::Vector<double> dt_tau_u_right;
    LinearAlgebra::distributed::Vector<double> dt_tau_pres_right;
    LinearAlgebra::distributed::Vector<double> dt_tau_rho_aux_right;
    LinearAlgebra::distributed::Vector<double> dt_tau_u_aux_right;
    LinearAlgebra::distributed::Vector<double> dt_tau_pres_aux_right;

    LinearAlgebra::distributed::Vector<double> dt_tau_rho_left;
    LinearAlgebra::distributed::Vector<double> dt_tau_u_left;
    LinearAlgebra::distributed::Vector<double> dt_tau_pres_left;
    LinearAlgebra::distributed::Vector<double> dt_tau_rho_aux_left;
    LinearAlgebra::distributed::Vector<double> dt_tau_u_aux_left;
    LinearAlgebra::distributed::Vector<double> dt_tau_pres_aux_left;

    EquationData::PushForward<dim> push_forward;
    EquationData::PullBack<dim>    pull_back;
    FunctionManifold<2, 2, 2>      manifold;

    EquationData::Density<dim>     rho_init;
    EquationData::Velocity<dim>    u_init;
    EquationData::Pressure<dim>    pres_init;

    EquationData::Raylegh<dim, 1>       dt_tau;
    EquationData::Raylegh_Aux<dim, 1>   dt_tau_aux;
    EquationData::Raylegh<dim, dim>     dt_tau_vel;
    EquationData::Raylegh_Aux<dim, dim> dt_tau_vel_aux;

    EquationData::Raylegh_Right<dim, 1>       dt_tau_right;
    EquationData::Raylegh_Aux_Right<dim, 1>   dt_tau_aux_right;
    EquationData::Raylegh_Right<dim, dim>     dt_tau_vel_right;
    EquationData::Raylegh_Aux_Right<dim, dim> dt_tau_vel_aux_right;

    EquationData::Raylegh_Left<dim, 1>       dt_tau_left;
    EquationData::Raylegh_Aux_Left<dim, 1>   dt_tau_aux_left;
    EquationData::Raylegh_Left<dim, dim>     dt_tau_vel_left;
    EquationData::Raylegh_Aux_Left<dim, dim> dt_tau_vel_aux_left;

    std::shared_ptr<MatrixFree<dim, double>> matrix_free_storage;

    HYPERBOLICOperator<dim, EquationData::degree_rho, EquationData::degree_T, EquationData::degree_u,
                            2*EquationData::degree_rho + 1, 2*EquationData::degree_T + 1, 2*EquationData::degree_u + 1,
                            LinearAlgebra::distributed::Vector<double>, double> euler_matrix;

    MGLevelObject<HYPERBOLICOperator<dim, EquationData::degree_rho, EquationData::degree_T, EquationData::degree_u,
                                          2*EquationData::degree_rho + 1, 2*EquationData::degree_T + 1, 2*EquationData::degree_u + 1,
                                          LinearAlgebra::distributed::Vector<double>, double>> mg_matrices_euler;

    std::vector<const DoFHandler<dim>*> dof_handlers;

    std::vector<const AffineConstraints<double>*> constraints;
    AffineConstraints<double> constraints_velocity,
                              constraints_temperature,
                              constraints_density;

    std::vector<QGauss<1>> quadratures;

    unsigned int max_its;
    double       eps;

    std::string saving_dir;

    ConditionalOStream pcout;

    std::ofstream      time_out;
    ConditionalOStream ptime_out;
    TimerOutput        time_table;

    std::ofstream output_n_dofs_velocity;
    std::ofstream output_n_dofs_temperature;
    std::ofstream output_n_dofs_density;

    Vector<double> Linfty_error_per_cell_vel,
                   Linfty_error_per_cell_pres,
                   Linfty_error_per_cell_rho;

    MGLevelObject<LinearAlgebra::distributed::Vector<double>> level_projection;

    double get_maximal_velocity();

    double get_minimal_density();

    double get_maximal_density();

    double compute_max_celerity();

    std::pair<double, double> compute_max_Cu_x_w();

    std::pair<double, double> compute_max_C_x_w();
  };


  // @sect{ <code>EulerSolver::EulerSolver</code> }

  // In the constructor, we just read all the data from the
  // <code>Data_Storage</code> object that is passed as an argument, verify that
  // the data we read are reasonable and, finally, create the triangulation and
  // load the initial data.
  //
  template<int dim>
  EulerSolver<dim>::EulerSolver(RunTimeParameters::Data_Storage& data):
    t_0(data.initial_time),
    T(data.final_time),
    HYPERBOLIC_stage(1),            //--- Initialize the flag for the TR_BDF2 stage
    Ma(data.Mach),
    Fr(data.Froude),
    dt(data.dt),
    triangulation(MPI_COMM_WORLD, parallel::distributed::Triangulation<dim>::limit_level_difference_at_vertices,
                  parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy),
    fe_density(FE_DGQ<dim>(EquationData::degree_rho), 1),
    fe_velocity(FE_DGQ<dim>(EquationData::degree_u), dim),
    fe_temperature(FE_DGQ<dim>(EquationData::degree_T), 1),
    dof_handler_density(triangulation),
    dof_handler_velocity(triangulation),
    dof_handler_temperature(triangulation),
    quadrature_density(EquationData::degree_rho + 1),
    quadrature_velocity(EquationData::degree_u + 1),
    quadrature_temperature(EquationData::degree_T + 1),
    push_forward(),
    pull_back(),
    manifold(push_forward, pull_back),
    rho_init(data.initial_time),
    u_init(data.initial_time),
    pres_init(data.initial_time),
    dt_tau(data.initial_time),
    dt_tau_aux(data.initial_time),
    dt_tau_vel(data.initial_time),
    dt_tau_vel_aux(data.initial_time),
    dt_tau_right(data.initial_time),
    dt_tau_aux_right(data.initial_time),
    dt_tau_vel_right(data.initial_time),
    dt_tau_vel_aux_right(data.initial_time),
    dt_tau_left(data.initial_time),
    dt_tau_aux_left(data.initial_time),
    dt_tau_vel_left(data.initial_time),
    dt_tau_vel_aux_left(data.initial_time),
    euler_matrix(data),
    max_its(data.max_iterations),
    eps(data.eps),
    saving_dir(data.dir),
    pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0),
    time_out("./" + data.dir + "/time_analysis_" +
             Utilities::int_to_string(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)) + "proc.dat"),
    ptime_out(time_out, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0),
    time_table(ptime_out, TimerOutput::summary, TimerOutput::cpu_and_wall_times),
    output_n_dofs_velocity("./" + data.dir + "/n_dofs_velocity.dat", std::ofstream::out),
    output_n_dofs_temperature("./" + data.dir + "/n_dofs_temperature.dat", std::ofstream::out),
    output_n_dofs_density("./" + data.dir + "/n_dofs_density.dat", std::ofstream::out) {
      AssertThrow(!((dt <= 0.0) || (dt > 0.5*T)), ExcInvalidTimeStep(dt, 0.5*T));

      matrix_free_storage = std::make_shared<MatrixFree<dim, double>>();

      dof_handlers.clear();

      constraints.clear();
      constraints_velocity.clear();
      constraints_temperature.clear();
      constraints_density.clear();

      quadratures.clear();

      create_triangulation(data.n_global_refines);
      setup_dofs();
      initialize();
  }


  // @sect{<code>EulerSolver::create_triangulation_and_dofs</code>}

  // The method that creates the triangulation.
  //
  template<int dim>
  void EulerSolver<dim>::create_triangulation(const unsigned int n_refines) {
    TimerOutput::Scope t(time_table, "Create triangulation");

    Point<dim> lower_left;
    lower_left[0] = 0.0;
    lower_left[1] = 0.0;
    Point<dim> upper_right;
    upper_right[0] = 40.0;
    upper_right[1] = 20.0;

    GridGenerator::subdivided_hyper_rectangle(triangulation, {25, 25}, lower_left, upper_right, true);

    std::vector<GridTools::PeriodicFacePair<typename parallel::distributed::Triangulation<dim>::cell_iterator>> periodic_faces;
    GridTools::collect_periodic_faces(triangulation, 0, 1, 0, periodic_faces);
    triangulation.add_periodicity(periodic_faces);

    triangulation.refine_global(n_refines);

    GridTools::transform([this](const Point<2>& chart_point) {
                                  return manifold.push_forward(chart_point);
                                },
                                triangulation);
    for(auto cell = triangulation.begin(); cell != triangulation.end(); ++cell)
      cell->set_all_manifold_ids(111);
    triangulation.set_manifold(111, manifold);
  }


  // After creating the triangulation, it creates the mesh dependent
  // data, i.e. it distributes degrees of freedom and renumbers them, and
  // initializes the matrices and vectors that we will use.
  //
  template<int dim>
  void EulerSolver<dim>::setup_dofs() {
    pcout << "Number of active cells: " << triangulation.n_global_active_cells() << std::endl;
    pcout << "Number of levels: "       << triangulation.n_global_levels()       << std::endl;

    dof_handler_velocity.distribute_dofs(fe_velocity);
    dof_handler_temperature.distribute_dofs(fe_temperature);
    dof_handler_density.distribute_dofs(fe_density);

    mg_matrices_euler.clear_elements();
    dof_handler_velocity.distribute_mg_dofs();
    dof_handler_temperature.distribute_mg_dofs();
    dof_handler_density.distribute_mg_dofs();
    level_projection = MGLevelObject<LinearAlgebra::distributed::Vector<double>>(0, triangulation.n_global_levels() - 1);

    pcout << "dim (V_h) = " << dof_handler_velocity.n_dofs()
          << std::endl
          << "dim (Q_h) = " << dof_handler_temperature.n_dofs()
          << std::endl
          << "dim (X_h) = " << dof_handler_density.n_dofs()
          << std::endl
          << "Ma        = " << Ma
          << std::endl
          << "Fr        = " << Fr << std::endl
          << std::endl;

    if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0) {
      output_n_dofs_velocity    << dof_handler_velocity.n_dofs()    << std::endl;
      output_n_dofs_temperature << dof_handler_temperature.n_dofs() << std::endl;
      output_n_dofs_density     << dof_handler_density.n_dofs()     << std::endl;
    }

    typename MatrixFree<dim, double>::AdditionalData additional_data;
    additional_data.mapping_update_flags                = (update_gradients | update_JxW_values |
                                                           update_quadrature_points | update_values);
    additional_data.mapping_update_flags_inner_faces    = (update_gradients | update_JxW_values | update_quadrature_points |
                                                           update_normal_vectors | update_values);
    additional_data.mapping_update_flags_boundary_faces = (update_gradients | update_JxW_values | update_quadrature_points |
                                                           update_normal_vectors | update_values);
    additional_data.tasks_parallel_scheme               = MatrixFree<dim, double>::AdditionalData::none;

    dof_handlers.push_back(&dof_handler_velocity);
    dof_handlers.push_back(&dof_handler_temperature);
    dof_handlers.push_back(&dof_handler_density);

    constraints.push_back(&constraints_velocity);
    constraints.push_back(&constraints_temperature);
    constraints.push_back(&constraints_density);

    quadratures.push_back(QGauss<1>(2*EquationData::degree_u + 1));

    matrix_free_storage->reinit(dof_handlers, constraints, quadratures, additional_data);

    matrix_free_storage->initialize_dof_vector(u_old, 0);
    matrix_free_storage->initialize_dof_vector(u_tmp_2, 0);
    matrix_free_storage->initialize_dof_vector(u_tmp_3, 0);
    matrix_free_storage->initialize_dof_vector(u_curr, 0);
    matrix_free_storage->initialize_dof_vector(u_fixed, 0);
    matrix_free_storage->initialize_dof_vector(rhs_u, 0);

    matrix_free_storage->initialize_dof_vector(pres_old, 1);
    matrix_free_storage->initialize_dof_vector(pres_tmp_2, 1);
    matrix_free_storage->initialize_dof_vector(pres_tmp_3, 1);
    matrix_free_storage->initialize_dof_vector(pres_fixed, 1);
    matrix_free_storage->initialize_dof_vector(pres_fixed_old, 1);
    matrix_free_storage->initialize_dof_vector(pres_tmp, 1);
    matrix_free_storage->initialize_dof_vector(rhs_pres, 1);

    matrix_free_storage->initialize_dof_vector(rho_old, 2);
    matrix_free_storage->initialize_dof_vector(rho_tmp_2, 2);
    matrix_free_storage->initialize_dof_vector(rho_tmp_3, 2);
    matrix_free_storage->initialize_dof_vector(rho_curr, 2);
    matrix_free_storage->initialize_dof_vector(rhs_rho, 2);

    matrix_free_storage->initialize_dof_vector(tmp_1, 0);
    tmp_1 = 0;

    matrix_free_storage->initialize_dof_vector(dt_tau_u, 0);
    matrix_free_storage->initialize_dof_vector(dt_tau_pres, 1);
    matrix_free_storage->initialize_dof_vector(dt_tau_rho, 2);
    matrix_free_storage->initialize_dof_vector(dt_tau_u_aux, 0);
    matrix_free_storage->initialize_dof_vector(dt_tau_pres_aux, 1);
    matrix_free_storage->initialize_dof_vector(dt_tau_rho_aux, 2);
    VectorTools::interpolate(dof_handler_velocity, dt_tau_vel, dt_tau_u);
    VectorTools::interpolate(dof_handler_temperature, dt_tau, dt_tau_pres);
    VectorTools::interpolate(dof_handler_density, dt_tau, dt_tau_rho);
    VectorTools::interpolate(dof_handler_velocity, dt_tau_vel_aux, dt_tau_u_aux);
    VectorTools::interpolate(dof_handler_temperature, dt_tau_aux, dt_tau_pres_aux);
    VectorTools::interpolate(dof_handler_density, dt_tau_aux, dt_tau_rho_aux);

    matrix_free_storage->initialize_dof_vector(dt_tau_u_right, 0);
    matrix_free_storage->initialize_dof_vector(dt_tau_pres_right, 1);
    matrix_free_storage->initialize_dof_vector(dt_tau_rho_right, 2);
    matrix_free_storage->initialize_dof_vector(dt_tau_u_aux_right, 0);
    matrix_free_storage->initialize_dof_vector(dt_tau_pres_aux_right, 1);
    matrix_free_storage->initialize_dof_vector(dt_tau_rho_aux_right, 2);
    VectorTools::interpolate(dof_handler_velocity, dt_tau_vel_right, dt_tau_u_right);
    VectorTools::interpolate(dof_handler_temperature, dt_tau_right, dt_tau_pres_right);
    VectorTools::interpolate(dof_handler_density, dt_tau_right, dt_tau_rho_right);
    VectorTools::interpolate(dof_handler_velocity, dt_tau_vel_aux_right, dt_tau_u_aux_right);
    VectorTools::interpolate(dof_handler_temperature, dt_tau_aux_right, dt_tau_pres_aux_right);
    VectorTools::interpolate(dof_handler_density, dt_tau_aux_right, dt_tau_rho_aux_right);

    matrix_free_storage->initialize_dof_vector(dt_tau_u_left, 0);
    matrix_free_storage->initialize_dof_vector(dt_tau_pres_left, 1);
    matrix_free_storage->initialize_dof_vector(dt_tau_rho_left, 2);
    matrix_free_storage->initialize_dof_vector(dt_tau_u_aux_left, 0);
    matrix_free_storage->initialize_dof_vector(dt_tau_pres_aux_left, 1);
    matrix_free_storage->initialize_dof_vector(dt_tau_rho_aux_left, 2);
    VectorTools::interpolate(dof_handler_velocity, dt_tau_vel_left, dt_tau_u_left);
    VectorTools::interpolate(dof_handler_temperature, dt_tau_left, dt_tau_pres_left);
    VectorTools::interpolate(dof_handler_density, dt_tau_left, dt_tau_rho_left);
    VectorTools::interpolate(dof_handler_velocity, dt_tau_vel_aux_left, dt_tau_u_aux_left);
    VectorTools::interpolate(dof_handler_temperature, dt_tau_aux_left, dt_tau_pres_aux_left);
    VectorTools::interpolate(dof_handler_density, dt_tau_aux_left, dt_tau_rho_aux_left);

    matrix_free_storage->initialize_dof_vector(u_bar, 0);
    matrix_free_storage->initialize_dof_vector(pres_bar, 1);
    matrix_free_storage->initialize_dof_vector(rho_bar, 2);
    VectorTools::interpolate(dof_handler_velocity, u_init, u_bar);
    VectorTools::interpolate(dof_handler_temperature, pres_init, pres_bar);
    VectorTools::interpolate(dof_handler_density, rho_init, rho_bar);
    dt_tau_u.scale(u_bar);
    dt_tau_pres.scale(pres_bar);
    dt_tau_rho.scale(rho_bar);
    dt_tau_u_right.scale(u_bar);
    dt_tau_pres_right.scale(pres_bar);
    dt_tau_rho_right.scale(rho_bar);
    dt_tau_u_left.scale(u_bar);
    dt_tau_pres_left.scale(pres_bar);
    dt_tau_rho_left.scale(rho_bar);

    Vector<double> error_per_cell_tmp(triangulation.n_active_cells());
    Linfty_error_per_cell_vel.reinit(error_per_cell_tmp);
    Linfty_error_per_cell_pres.reinit(error_per_cell_tmp);
    Linfty_error_per_cell_rho.reinit(error_per_cell_tmp);

    mg_matrices_euler.resize(0, triangulation.n_global_levels() - 1);
    for(unsigned int level = 0; level < triangulation.n_global_levels(); ++level) {
      mg_matrices_euler[level].set_dt(dt);
      mg_matrices_euler[level].set_Mach(Ma);
      mg_matrices_euler[level].set_Froude(Fr);
    }
  }


  // @sect{ <code>EulerSolver::initialize</code> }

  // This method loads the initial data
  //
  template<int dim>
  void EulerSolver<dim>::initialize() {
    TimerOutput::Scope t(time_table, "Initialize state");

    VectorTools::interpolate(dof_handler_density, rho_init, rho_old);
    VectorTools::interpolate(dof_handler_velocity, u_init, u_old);
    VectorTools::interpolate(dof_handler_temperature, pres_init, pres_old);
  }


  // @sect{<code>EulerSolver::update_density</code>}

  // This implements the update of the density for the hyperbolic part
  //
  template<int dim>
  void EulerSolver<dim>::update_density() {
    TimerOutput::Scope t(time_table, "Update density");

    const std::vector<unsigned int> tmp = {2};
    euler_matrix.initialize(matrix_free_storage, tmp, tmp);
    euler_matrix.set_NS_stage(1);

    if(HYPERBOLIC_stage == 1) {
      euler_matrix.vmult_rhs_rho_update(rhs_rho, {rho_old, u_old});
    }
    else if(HYPERBOLIC_stage == 2) {
      euler_matrix.vmult_rhs_rho_update(rhs_rho, {rho_old, u_old,
                                                  rho_tmp_2, u_tmp_2});
    }
    else {
      euler_matrix.set_NS_stage(4);
      euler_matrix.vmult_rhs_rho_update(rhs_rho, {rho_old, u_old,
                                                  rho_tmp_2, u_tmp_2,
                                                  rho_tmp_3, u_tmp_3});
    }

    SolverControl solver_control(max_its, eps*rhs_rho.l2_norm());
    SolverCG<LinearAlgebra::distributed::Vector<double>> cg(solver_control);

    //--- Compute multigrid preconditioner for density ---
    for(unsigned int level = 0; level < triangulation.n_global_levels(); ++level) {
      typename MatrixFree<dim, double>::AdditionalData additional_data_mg;
      additional_data_mg.tasks_parallel_scheme               = MatrixFree<dim, double>::AdditionalData::none;
      additional_data_mg.mapping_update_flags                = (update_values | update_JxW_values);
      additional_data_mg.mapping_update_flags_inner_faces    = update_default;
      additional_data_mg.mapping_update_flags_boundary_faces = update_default;
      additional_data_mg.mg_level = level;

      std::shared_ptr<MatrixFree<dim, double>> mg_mf_storage_level(new MatrixFree<dim, double>());
      mg_mf_storage_level->reinit(dof_handlers, constraints, quadratures, additional_data_mg);
      mg_matrices_euler[level].initialize(mg_mf_storage_level, tmp, tmp);
      if(HYPERBOLIC_stage == 3)
        mg_matrices_euler[level].set_NS_stage(4);
      else
        mg_matrices_euler[level].set_NS_stage(1);
    }

    MGTransferMatrixFree<dim, double> mg_transfer;
    mg_transfer.build(dof_handler_density);
    using SmootherType = PreconditionChebyshev<HYPERBOLICOperator<dim,
                                                                  EquationData::degree_rho,
                                                                  EquationData::degree_T,
                                                                  EquationData::degree_u,
                                                                  2*EquationData::degree_rho + 1,
                                                                  2*EquationData::degree_T + 1,
                                                                  2*EquationData::degree_u + 1,
                                                                  LinearAlgebra::distributed::Vector<double>, double>,
                                                LinearAlgebra::distributed::Vector<double>>;
    mg::SmootherRelaxation<SmootherType, LinearAlgebra::distributed::Vector<double>> mg_smoother;
    MGLevelObject<typename SmootherType::AdditionalData> smoother_data;
    smoother_data.resize(0, triangulation.n_global_levels() - 1);
    for(unsigned int level = 0; level < triangulation.n_global_levels(); ++level) {
      if(level > 0) {
        smoother_data[level].smoothing_range     = 15.0;
        smoother_data[level].degree              = 3;
        smoother_data[level].eig_cg_n_iterations = 10;
      }
      else {
        smoother_data[0].smoothing_range     = 2e-2;
        smoother_data[0].degree              = numbers::invalid_unsigned_int;
        smoother_data[0].eig_cg_n_iterations = mg_matrices_euler[0].m();
      }
      mg_matrices_euler[level].compute_diagonal();
      smoother_data[level].preconditioner = mg_matrices_euler[level].get_matrix_diagonal_inverse();
    }
    mg_smoother.initialize(mg_matrices_euler, smoother_data);

    PreconditionIdentity                                 identity;
    SolverCG<LinearAlgebra::distributed::Vector<double>> cg_mg(solver_control);
    MGCoarseGridIterativeSolver<LinearAlgebra::distributed::Vector<double>,
                                SolverCG<LinearAlgebra::distributed::Vector<double>>,
                                HYPERBOLICOperator<dim,
                                                   EquationData::degree_rho,
                                                   EquationData::degree_T,
                                                   EquationData::degree_u,
                                                   2*EquationData::degree_rho + 1,
                                                   2*EquationData::degree_T + 1,
                                                   2*EquationData::degree_u + 1,
                                                   LinearAlgebra::distributed::Vector<double>, double>,
                                PreconditionIdentity> mg_coarse(cg_mg, mg_matrices_euler[0], identity);
    mg::Matrix<LinearAlgebra::distributed::Vector<double>> mg_matrix(mg_matrices_euler);
    Multigrid<LinearAlgebra::distributed::Vector<double>> mg(mg_matrix, mg_coarse, mg_transfer, mg_smoother, mg_smoother);
    PreconditionMG<dim,
                   LinearAlgebra::distributed::Vector<double>,
                   MGTransferMatrixFree<dim, double>> preconditioner(dof_handler_density, mg, mg_transfer);

    //--- Solve the system for the density
    if(HYPERBOLIC_stage == 1) {
      rho_tmp_2.equ(1.0, rho_old);
      cg.solve(euler_matrix, rho_tmp_2, rhs_rho, preconditioner);
    }
    else if(HYPERBOLIC_stage == 2) {
      rho_tmp_3.equ(1.0, rho_tmp_2);
      cg.solve(euler_matrix, rho_tmp_3, rhs_rho, preconditioner);
    }
    else {
      rho_curr.equ(1.0, rho_tmp_3);
      cg.solve(euler_matrix, rho_curr, rhs_rho, preconditioner);
    }
  }


  // @sect{<code>EulerSolver::pressure_fixed_point</code>}

  // This implements a step of the fixed point procedure for the computation of the pressure in the hyperbolic part
  //
  template<int dim>
  void EulerSolver<dim>::pressure_fixed_point() {
    TimerOutput::Scope t(time_table, "Fixed point pressure");

    const std::vector<unsigned int> tmp = {1};
    euler_matrix.initialize(matrix_free_storage, tmp, tmp);
    euler_matrix.set_NS_stage(2);

    euler_matrix.set_pres_fixed(pres_fixed_old);
    euler_matrix.set_u_fixed(u_fixed);
    if(HYPERBOLIC_stage == 1) {
      euler_matrix.vmult_rhs_pressure(rhs_pres, {rho_old, u_old, pres_old,
                                                 rho_tmp_2, u_fixed});

      euler_matrix.vmult_rhs_velocity_fixed(rhs_u, {rho_old, u_old, pres_old,
                                                    rho_tmp_2});
    }
    else if(HYPERBOLIC_stage == 2) {
      euler_matrix.vmult_rhs_pressure(rhs_pres, {rho_old, u_old, pres_old,
                                                 rho_tmp_2, u_tmp_2, pres_tmp_2,
                                                 rho_tmp_3, u_fixed});

      euler_matrix.vmult_rhs_velocity_fixed(rhs_u, {rho_old, u_old, pres_old,
                                                    rho_tmp_2, u_tmp_2, pres_tmp_2,
                                                    rho_tmp_3});
    }

    SolverControl solver_control_schur(max_its, 1e-12*rhs_u.l2_norm());
    SolverCG<LinearAlgebra::distributed::Vector<double>> cg_schur(solver_control_schur);
    const std::vector<unsigned int> tmp_reinit = {0};
    euler_matrix.initialize(matrix_free_storage, tmp_reinit, tmp_reinit);
    euler_matrix.set_NS_stage(3);

    //--- Set MultiGrid for velocity matrix ---
    for(unsigned int level = 0; level < triangulation.n_global_levels(); ++level) {
      typename MatrixFree<dim, double>::AdditionalData additional_data_mg;
      additional_data_mg.tasks_parallel_scheme               = MatrixFree<dim, double>::AdditionalData::none;
      additional_data_mg.mapping_update_flags                = (update_values | update_JxW_values);
      additional_data_mg.mapping_update_flags_inner_faces    = update_default;
      additional_data_mg.mapping_update_flags_boundary_faces = update_default;
      additional_data_mg.mg_level = level;

      std::shared_ptr<MatrixFree<dim, double>> mg_mf_storage_level(new MatrixFree<dim, double>());
      mg_mf_storage_level->reinit(dof_handlers, constraints, quadratures, additional_data_mg);
      mg_matrices_euler[level].initialize(mg_mf_storage_level, tmp_reinit, tmp_reinit);
      mg_matrices_euler[level].set_NS_stage(3);
    }

    MGTransferMatrixFree<dim, double> mg_transfer;
    mg_transfer.build(dof_handler_velocity);
    using SmootherType = PreconditionChebyshev<HYPERBOLICOperator<dim,
                                                                  EquationData::degree_rho,
                                                                  EquationData::degree_T,
                                                                  EquationData::degree_u,
                                                                  2*EquationData::degree_rho + 1,
                                                                  2*EquationData::degree_T + 1,
                                                                  2*EquationData::degree_u + 1,
                                                                  LinearAlgebra::distributed::Vector<double>, double>,
                                                LinearAlgebra::distributed::Vector<double>>;
    mg::SmootherRelaxation<SmootherType, LinearAlgebra::distributed::Vector<double>> mg_smoother;
    MGLevelObject<typename SmootherType::AdditionalData> smoother_data;
    smoother_data.resize(0, triangulation.n_global_levels() - 1);
    for(unsigned int level = 0; level < triangulation.n_global_levels(); ++level) {
      if(level > 0) {
        smoother_data[level].smoothing_range     = 15.0;
        smoother_data[level].degree              = 3;
        smoother_data[level].eig_cg_n_iterations = 10;
      }
      else {
        smoother_data[0].smoothing_range     = 2e-2;
        smoother_data[0].degree              = numbers::invalid_unsigned_int;
        smoother_data[0].eig_cg_n_iterations = mg_matrices_euler[0].m();
      }
      mg_matrices_euler[level].compute_diagonal();
      smoother_data[level].preconditioner = mg_matrices_euler[level].get_matrix_diagonal_inverse();
    }
    mg_smoother.initialize(mg_matrices_euler, smoother_data);

    PreconditionIdentity                                 identity;
    SolverCG<LinearAlgebra::distributed::Vector<double>> cg_mg(solver_control_schur);
    MGCoarseGridIterativeSolver<LinearAlgebra::distributed::Vector<double>,
                                SolverCG<LinearAlgebra::distributed::Vector<double>>,
                                HYPERBOLICOperator<dim,
                                                   EquationData::degree_rho,
                                                   EquationData::degree_T,
                                                   EquationData::degree_u,
                                                   2*EquationData::degree_rho + 1,
                                                   2*EquationData::degree_T + 1,
                                                   2*EquationData::degree_u + 1,
                                                   LinearAlgebra::distributed::Vector<double>, double>,
                                PreconditionIdentity> mg_coarse(cg_mg, mg_matrices_euler[0], identity);
    mg::Matrix<LinearAlgebra::distributed::Vector<double>> mg_matrix(mg_matrices_euler);
    Multigrid<LinearAlgebra::distributed::Vector<double>> mg(mg_matrix, mg_coarse, mg_transfer, mg_smoother, mg_smoother);
    PreconditionMG<dim,
                   LinearAlgebra::distributed::Vector<double>,
                   MGTransferMatrixFree<dim, double>> preconditioner(dof_handler_velocity, mg, mg_transfer);

    //--- Solve to compute first contribution to rhs ---
    cg_schur.solve(euler_matrix, tmp_1, rhs_u, preconditioner);

    //--- Perform matrix-vector multiplication with enthalpy matrix ---
    LinearAlgebra::distributed::Vector<double> tmp_2;
    matrix_free_storage->initialize_dof_vector(tmp_2, 1);
    euler_matrix.vmult_enthalpy(tmp_2, tmp_1);

    //---  Conclude computation of rhs for pressure fixed point ---
    rhs_pres.add(-1.0, tmp_2);

    euler_matrix.set_NS_stage(2);
    euler_matrix.initialize(matrix_free_storage, tmp, tmp);

    // --- Solve the system for the pressure ---
    SolverControl solver_control(max_its, eps*rhs_pres.l2_norm());
    SolverGMRES<LinearAlgebra::distributed::Vector<double>> gmres(solver_control);
    pres_fixed.equ(1.0, pres_fixed_old);
    PreconditionJacobi<HYPERBOLICOperator<dim,
                                          EquationData::degree_rho,
                                          EquationData::degree_T,
                                          EquationData::degree_u,
                                          2*EquationData::degree_rho + 1,
                                          2*EquationData::degree_T + 1,
                                          2*EquationData::degree_u + 1,
                                          LinearAlgebra::distributed::Vector<double>, double>> preconditioner_Jacobi;
    euler_matrix.compute_diagonal();
    preconditioner_Jacobi.initialize(euler_matrix);
    gmres.solve(euler_matrix, pres_fixed, rhs_pres, preconditioner_Jacobi);
  }


  // @sect{<code>EulerSolver::update_velocity</code>}

  // This implements the velocity update in the fixed point procedure for the computation of the pressure in the hyperbolic part
  //
  template<int dim>
  void EulerSolver<dim>::update_velocity() {
    TimerOutput::Scope t(time_table, "Update velocity");

    const std::vector<unsigned int> tmp = {0};
    euler_matrix.initialize(matrix_free_storage, tmp, tmp);
    if(HYPERBOLIC_stage == 1 || HYPERBOLIC_stage == 2) {
      euler_matrix.set_NS_stage(3);

      LinearAlgebra::distributed::Vector<double> tmp_3;
      matrix_free_storage->initialize_dof_vector(tmp_3, 0);
      euler_matrix.vmult_pressure(tmp_3, pres_fixed);
      rhs_u.add(-1.0, tmp_3);
    }
    else {
      euler_matrix.set_NS_stage(5);
      euler_matrix.vmult_rhs_velocity_fixed(rhs_u, {rho_old, u_old, pres_old,
                                                    rho_tmp_2, u_tmp_2, pres_tmp_2,
                                                    rho_tmp_3, u_tmp_3, pres_tmp_3});
    }

    SolverControl solver_control(max_its, eps*rhs_u.l2_norm());
    SolverCG<LinearAlgebra::distributed::Vector<double>> cg(solver_control);

    //--- Compute MultiGrid for velocity matrix ---
    for(unsigned int level = 0; level < triangulation.n_global_levels(); ++level) {
      typename MatrixFree<dim, double>::AdditionalData additional_data_mg;
      additional_data_mg.tasks_parallel_scheme               = MatrixFree<dim, double>::AdditionalData::none;
      additional_data_mg.mapping_update_flags                = (update_values | update_JxW_values);
      additional_data_mg.mapping_update_flags_inner_faces    = update_default;
      additional_data_mg.mapping_update_flags_boundary_faces = update_default;
      additional_data_mg.mg_level                            = level;

      std::shared_ptr<MatrixFree<dim, double>> mg_mf_storage_level(new MatrixFree<dim, double>());
      mg_mf_storage_level->reinit(dof_handlers, constraints, quadratures, additional_data_mg);
      mg_matrices_euler[level].initialize(mg_mf_storage_level, tmp, tmp);
      if(HYPERBOLIC_stage == 3)
        mg_matrices_euler[level].set_NS_stage(5);
      else
        mg_matrices_euler[level].set_NS_stage(3);
    }

    MGTransferMatrixFree<dim, double> mg_transfer;
    mg_transfer.build(dof_handler_velocity);
    using SmootherType = PreconditionChebyshev<HYPERBOLICOperator<dim,
                                                                  EquationData::degree_rho,
                                                                  EquationData::degree_T,
                                                                  EquationData::degree_u,
                                                                  2*EquationData::degree_rho + 1,
                                                                  2*EquationData::degree_T + 1,
                                                                  2*EquationData::degree_u + 1,
                                                                  LinearAlgebra::distributed::Vector<double>, double>,
                                                LinearAlgebra::distributed::Vector<double>>;
    mg::SmootherRelaxation<SmootherType, LinearAlgebra::distributed::Vector<double>> mg_smoother;
    MGLevelObject<typename SmootherType::AdditionalData> smoother_data;
    smoother_data.resize(0, triangulation.n_global_levels() - 1);
    for(unsigned int level = 0; level < triangulation.n_global_levels(); ++level) {
      if(level > 0) {
        smoother_data[level].smoothing_range     = 15.0;
        smoother_data[level].degree              = 3;
        smoother_data[level].eig_cg_n_iterations = 10;
      }
      else {
        smoother_data[0].smoothing_range     = 2e-2;
        smoother_data[0].degree              = numbers::invalid_unsigned_int;
        smoother_data[0].eig_cg_n_iterations = mg_matrices_euler[0].m();
      }
      mg_matrices_euler[level].compute_diagonal();
      smoother_data[level].preconditioner = mg_matrices_euler[level].get_matrix_diagonal_inverse();
    }
    mg_smoother.initialize(mg_matrices_euler, smoother_data);

    PreconditionIdentity                                 identity;
    SolverCG<LinearAlgebra::distributed::Vector<double>> cg_mg(solver_control);
    MGCoarseGridIterativeSolver<LinearAlgebra::distributed::Vector<double>,
                                SolverCG<LinearAlgebra::distributed::Vector<double>>,
                                HYPERBOLICOperator<dim,
                                                   EquationData::degree_rho,
                                                   EquationData::degree_T,
                                                   EquationData::degree_u,
                                                   2*EquationData::degree_rho + 1,
                                                   2*EquationData::degree_T + 1,
                                                   2*EquationData::degree_u + 1,
                                                   LinearAlgebra::distributed::Vector<double>, double>,
                                PreconditionIdentity> mg_coarse(cg_mg, mg_matrices_euler[0], identity);
    mg::Matrix<LinearAlgebra::distributed::Vector<double>> mg_matrix(mg_matrices_euler);
    Multigrid<LinearAlgebra::distributed::Vector<double>> mg(mg_matrix, mg_coarse, mg_transfer, mg_smoother, mg_smoother);
    PreconditionMG<dim,
                   LinearAlgebra::distributed::Vector<double>,
                   MGTransferMatrixFree<dim, double>> preconditioner(dof_handler_velocity, mg, mg_transfer);

    //--- Solve the system for the velocity
    if(HYPERBOLIC_stage == 1 || HYPERBOLIC_stage == 2)
      cg.solve(euler_matrix, u_fixed, rhs_u, preconditioner);
    else {
      u_curr.equ(1.0, u_tmp_3);
      cg.solve(euler_matrix, u_curr, rhs_u, preconditioner);
    }
  }


  // @sect{<code>EulerSolver::update_pressure</code>}

  // This implements the update of the pressure for the hyperbolic part
  //
  template<int dim>
  void EulerSolver<dim>::update_pressure() {
    TimerOutput::Scope t(time_table, "Update pressure");

    const std::vector<unsigned int> tmp = {1};
    euler_matrix.initialize(matrix_free_storage, tmp, tmp);

    euler_matrix.set_NS_stage(6);
    euler_matrix.vmult_rhs_pressure(rhs_pres, {rho_old, u_old, pres_old,
                                               rho_tmp_2, u_tmp_2, pres_tmp_2,
                                               rho_tmp_3, u_tmp_3, pres_tmp_3,
                                               rho_curr, u_curr});

    SolverControl solver_control(max_its, eps*rhs_pres.l2_norm());
    SolverCG<LinearAlgebra::distributed::Vector<double>> cg(solver_control);

    //--- Compute multigrid preconditioner for density ---
    for(unsigned int level = 0; level < triangulation.n_global_levels(); ++level) {
      typename MatrixFree<dim, double>::AdditionalData additional_data_mg;
      additional_data_mg.tasks_parallel_scheme               = MatrixFree<dim, double>::AdditionalData::none;
      additional_data_mg.mapping_update_flags                = (update_values | update_JxW_values);
      additional_data_mg.mapping_update_flags_inner_faces    = update_default;
      additional_data_mg.mapping_update_flags_boundary_faces = update_default;
      additional_data_mg.mg_level                            = level;

      std::shared_ptr<MatrixFree<dim, double>> mg_mf_storage_level(new MatrixFree<dim, double>());
      mg_mf_storage_level->reinit(dof_handlers, constraints, quadratures, additional_data_mg);
      mg_matrices_euler[level].initialize(mg_mf_storage_level, tmp, tmp);
      mg_matrices_euler[level].set_NS_stage(6);
    }

    MGTransferMatrixFree<dim, double> mg_transfer;
    mg_transfer.build(dof_handler_density);
    using SmootherType = PreconditionChebyshev<HYPERBOLICOperator<dim,
                                                                  EquationData::degree_rho,
                                                                  EquationData::degree_T,
                                                                  EquationData::degree_u,
                                                                  2*EquationData::degree_rho + 1,
                                                                  2*EquationData::degree_T + 1,
                                                                  2*EquationData::degree_u + 1,
                                                                  LinearAlgebra::distributed::Vector<double>, double>,
                                                LinearAlgebra::distributed::Vector<double>>;
    mg::SmootherRelaxation<SmootherType, LinearAlgebra::distributed::Vector<double>> mg_smoother;
    MGLevelObject<typename SmootherType::AdditionalData> smoother_data;
    smoother_data.resize(0, triangulation.n_global_levels() - 1);
    for(unsigned int level = 0; level < triangulation.n_global_levels(); ++level) {
      if(level > 0) {
        smoother_data[level].smoothing_range     = 15.0;
        smoother_data[level].degree              = 3;
        smoother_data[level].eig_cg_n_iterations = 10;
      }
      else {
        smoother_data[0].smoothing_range     = 2e-2;
        smoother_data[0].degree              = numbers::invalid_unsigned_int;
        smoother_data[0].eig_cg_n_iterations = mg_matrices_euler[0].m();
      }
      mg_matrices_euler[level].compute_diagonal();
      smoother_data[level].preconditioner = mg_matrices_euler[level].get_matrix_diagonal_inverse();
    }
    mg_smoother.initialize(mg_matrices_euler, smoother_data);

    PreconditionIdentity                                 identity;
    SolverCG<LinearAlgebra::distributed::Vector<double>> cg_mg(solver_control);
    MGCoarseGridIterativeSolver<LinearAlgebra::distributed::Vector<double>,
                                SolverCG<LinearAlgebra::distributed::Vector<double>>,
                                HYPERBOLICOperator<dim,
                                                   EquationData::degree_rho,
                                                   EquationData::degree_T,
                                                   EquationData::degree_u,
                                                   2*EquationData::degree_rho + 1,
                                                   2*EquationData::degree_T + 1,
                                                   2*EquationData::degree_u + 1,
                                                   LinearAlgebra::distributed::Vector<double>, double>,
                                PreconditionIdentity> mg_coarse(cg_mg, mg_matrices_euler[0], identity);
    mg::Matrix<LinearAlgebra::distributed::Vector<double>> mg_matrix(mg_matrices_euler);
    Multigrid<LinearAlgebra::distributed::Vector<double>> mg(mg_matrix, mg_coarse, mg_transfer, mg_smoother, mg_smoother);
    PreconditionMG<dim,
                   LinearAlgebra::distributed::Vector<double>,
                   MGTransferMatrixFree<dim, double>> preconditioner(dof_handler_density, mg, mg_transfer);

    //--- Solve the system for the pressure
    pres_old.equ(1.0, pres_tmp_3);
    cg.solve(euler_matrix, pres_old, rhs_pres, preconditioner);
  }


  // @sect{ <code>EulerSolver::output_results</code> }

  // This method plots the current solution. The main difficulty is that we want
  // to create a single output file that contains the data for all velocity
  // components and the pressure. On the other hand, velocities and the pressure
  // live on separate DoFHandler objects, and
  // so can't be written to the same file using a single DataOut object. As a
  // consequence, we have to work a bit harder to get the various pieces of data
  // into a single DoFHandler object, and then use that to drive graphical
  // output.
  //
  template<int dim>
  void EulerSolver<dim>::output_results(const unsigned int step) {
    TimerOutput::Scope t(time_table, "Output results");

    DataOut<dim> data_out;

    rho_old.update_ghost_values();
    data_out.add_data_vector(dof_handler_density, rho_old, "rho", {DataComponentInterpretation::component_is_scalar});
    std::vector<std::string> velocity_names(dim, "u");
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
    component_interpretation_velocity(dim, DataComponentInterpretation::component_is_part_of_vector);
    u_old.update_ghost_values();
    data_out.add_data_vector(dof_handler_velocity, u_old, velocity_names, component_interpretation_velocity);
    pres_old.update_ghost_values();
    data_out.add_data_vector(dof_handler_temperature, pres_old, "p", {DataComponentInterpretation::component_is_scalar});

    rho_bar.update_ghost_values();
    data_out.add_data_vector(dof_handler_density, rho_bar, "rho_bar", {DataComponentInterpretation::component_is_scalar});
    u_bar.update_ghost_values();
    std::fill(velocity_names.begin(), velocity_names.end(), "u_bar");
    data_out.add_data_vector(dof_handler_velocity, u_bar, velocity_names, component_interpretation_velocity);
    pres_bar.update_ghost_values();
    data_out.add_data_vector(dof_handler_temperature, pres_bar, "p_bar", {DataComponentInterpretation::component_is_scalar});

    data_out.build_patches(MappingQ1<dim>(), EquationData::degree_u, DataOut<dim>::curved_inner_cells);

    DataOutBase::DataOutFilterFlags flags(false, true);
    DataOutBase::DataOutFilter      data_filter(flags);
    data_out.write_filtered_data(data_filter);
    std::string output = "./" + saving_dir + "/solution-" + Utilities::int_to_string(step, 5) + ".h5";
    data_out.write_hdf5_parallel(data_filter, output, MPI_COMM_WORLD);
    std::vector<XDMFEntry> xdmf_entries;
    auto new_xdmf_entry = data_out.create_xdmf_entry(data_filter, output, step, MPI_COMM_WORLD);
    xdmf_entries.push_back(new_xdmf_entry);
    output = "./" + saving_dir + "/solution-" + Utilities::int_to_string(step, 5) + ".xdmf";
    data_out.write_xdmf_file(xdmf_entries, output, MPI_COMM_WORLD);
    output = "./" + saving_dir + "/solution-" + Utilities::int_to_string(step, 5) + ".vtu";
    data_out.write_vtu_in_parallel(output, MPI_COMM_WORLD);
  }


  // The following function is used in determining the maximal velocity
  // in order to compute the CFL
  //
  template<int dim>
  double EulerSolver<dim>::get_maximal_velocity() {
    return u_old.linfty_norm();
  }


  // The following function is used in determining the minimal density
  //
  template<int dim>
  double EulerSolver<dim>::get_minimal_density() {
    FEValues<dim> fe_values(fe_density, quadrature_density, update_values);
    std::vector<double> solution_values(quadrature_density.size());

    double min_local_density = std::numeric_limits<double>::max();

    for(const auto& cell: dof_handler_density.active_cell_iterators()) {
      if(cell->is_locally_owned()) {
        fe_values.reinit(cell);
        if(HYPERBOLIC_stage == 1)
          fe_values.get_function_values(rho_tmp_2, solution_values);
        else if(HYPERBOLIC_stage == 2)
          fe_values.get_function_values(rho_tmp_3, solution_values);
        else
          fe_values.get_function_values(rho_curr, solution_values);
        for(unsigned int q = 0; q < quadrature_density.size(); ++q)
          min_local_density = std::min(min_local_density, solution_values[q]);
      }
    }

    return Utilities::MPI::min(min_local_density, MPI_COMM_WORLD);
  }


  // The following function is used in determining the maximal density
  //
  template<int dim>
  double EulerSolver<dim>::get_maximal_density() {
    if(HYPERBOLIC_stage == 1)
      return rho_tmp_2.linfty_norm();
    if(HYPERBOLIC_stage == 2)
      return rho_tmp_3.linfty_norm();

    return rho_curr.linfty_norm();
  }

  // The following function is used in determining the maximal celerity
  //
  template<int dim>
  double EulerSolver<dim>::compute_max_celerity() {
    FEValues<dim> fe_values(fe_temperature, quadrature_temperature, update_values);
    std::vector<double> solution_values_pressure(quadrature_temperature.size()),
                        solution_values_density(quadrature_temperature.size());

    double max_local_celerity = std::numeric_limits<double>::min();

    for(const auto& cell: dof_handler_temperature.active_cell_iterators()) {
      if(cell->is_locally_owned()) {
        fe_values.reinit(cell);
        fe_values.get_function_values(pres_old, solution_values_pressure);
        fe_values.get_function_values(rho_old, solution_values_density);
        for(unsigned int q = 0; q < quadrature_temperature.size(); ++q)
          max_local_celerity = std::max(max_local_celerity,
                                        std::sqrt(EquationData::Cp_Cv*solution_values_pressure[q]/solution_values_density[q]));
      }
    }

    return Utilities::MPI::max(max_local_celerity, MPI_COMM_WORLD);
  }


  // The following function is used in determining the maximal advective Courant numbers along the two directions
  //
  template<int dim>
  std::pair<double, double> EulerSolver<dim>::compute_max_Cu_x_w() {
    FEValues<dim>               fe_values(fe_velocity, quadrature_velocity, update_values);
    std::vector<Vector<double>> solution_values_velocity(quadrature_velocity.size(), Vector<double>(dim));

    double max_Cu_x = std::numeric_limits<double>::min();
    double max_Cu_w = std::numeric_limits<double>::min();

    for(const auto& cell: dof_handler_velocity.active_cell_iterators()) {
      if(cell->is_locally_owned()) {
        fe_values.reinit(cell);
        fe_values.get_function_values(u_old, solution_values_velocity);
        for(unsigned int q = 0; q < quadrature_temperature.size(); ++q) {
          max_Cu_x = std::max(max_Cu_x,
                              EquationData::degree_u*std::abs(solution_values_velocity[q](0))*dt/cell->extent_in_direction(0));
          max_Cu_w = std::max(max_Cu_w,
                              EquationData::degree_u*std::abs(solution_values_velocity[q](1))*dt/cell->extent_in_direction(1));
        }
      }
    }

    return std::make_pair(Utilities::MPI::max(max_Cu_x, MPI_COMM_WORLD), Utilities::MPI::max(max_Cu_w, MPI_COMM_WORLD));
  }


  // The following function is used in determining the maximal Courant number along the two directions
  //
  template<int dim>
  std::pair<double, double> EulerSolver<dim>::compute_max_C_x_w() {
    FEValues<dim> fe_values(fe_temperature, quadrature_temperature, update_values);
    std::vector<double> solution_values_pressure(quadrature_temperature.size()),
                        solution_values_density(quadrature_temperature.size());

    double max_C_x = std::numeric_limits<double>::min();
    double max_C_w = std::numeric_limits<double>::min();

    for(const auto& cell: dof_handler_temperature.active_cell_iterators()) {
      if(cell->is_locally_owned()) {
        fe_values.reinit(cell);
        fe_values.get_function_values(pres_old, solution_values_pressure);
        fe_values.get_function_values(rho_old, solution_values_density);
        for(unsigned int q = 0; q < quadrature_temperature.size(); ++q) {
          double local_celerity = std::sqrt(EquationData::Cp_Cv*solution_values_pressure[q]/solution_values_density[q]);
          max_C_x = std::max(max_C_x, 1.0/Ma*EquationData::degree_u*local_celerity*dt/cell->extent_in_direction(0));
          max_C_w = std::max(max_C_w, 1.0/Ma*EquationData::degree_u*local_celerity*dt/cell->extent_in_direction(1));
        }
      }
    }

    return std::make_pair(Utilities::MPI::max(max_C_x, MPI_COMM_WORLD), Utilities::MPI::max(max_C_w, MPI_COMM_WORLD));
  }


  // @sect{ <code>EulerSolver::run</code> }

  // This is the time marching function, which starting at <code>t_0</code>
  // advances in time using the projection method with time step <code>dt</code>
  // until <code>T</code>.
  //
  // Its second parameter, <code>verbose</code> indicates whether the function
  // should output information what it is doing at any given moment:
  // we use the ConditionalOStream class to do that for us.
  //
  template<int dim>
  void EulerSolver<dim>::run(const bool verbose, const unsigned int output_interval) {
    ConditionalOStream verbose_cout(std::cout, verbose && Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);

    output_results(0);
    double time = t_0;
    unsigned int n = 0;
    while(std::abs(T - time) > 1e-10) {
      time += dt;
      n++;
      pcout << "Step = " << n << " Time = " << time << std::endl;

      //--- First stage of HYPERBOLIC operator
      HYPERBOLIC_stage = 1;
      euler_matrix.set_HYPERBOLIC_stage(HYPERBOLIC_stage);

      verbose_cout << "  Update density stage 1" << std::endl;
      update_density();
      pcout<<"Minimal density "<<get_minimal_density()<<std::endl;
      pcout<<"Maximal density "<<get_maximal_density()<<std::endl;

      verbose_cout << "  Fixed point pressure stage 1" << std::endl;
      euler_matrix.set_rho_for_fixed(rho_tmp_2);
      MGTransferMatrixFree<dim, double> mg_transfer;
      mg_transfer.build(dof_handler_density);
      mg_transfer.interpolate_to_mg(dof_handler_density, level_projection, rho_tmp_2);
      for(unsigned int level = 0; level < triangulation.n_global_levels(); ++level)
        mg_matrices_euler[level].set_rho_for_fixed(level_projection[level]);
      pres_fixed_old.equ(1.0, pres_old);
      u_fixed.equ(1.0, u_old);
      for(unsigned int iter = 0; iter < 100; ++iter) {
        pressure_fixed_point();
        update_velocity();

        //Compute the relative error
        VectorTools::integrate_difference(dof_handler_temperature, pres_fixed, ZeroFunction<dim>(),
                                          Linfty_error_per_cell_pres, quadrature_temperature, VectorTools::Linfty_norm);
        const double den = VectorTools::compute_global_error(triangulation, Linfty_error_per_cell_pres, VectorTools::Linfty_norm);
        double error = 0.0;
        pres_tmp.equ(1.0, pres_fixed);
        pres_tmp.add(-1.0, pres_fixed_old);
        VectorTools::integrate_difference(dof_handler_temperature, pres_tmp, ZeroFunction<dim>(),
                                          Linfty_error_per_cell_pres, quadrature_temperature, VectorTools::Linfty_norm);
        error = VectorTools::compute_global_error(triangulation, Linfty_error_per_cell_pres, VectorTools::Linfty_norm)/den;
        if(error < 1e-10)
          break;

        pres_fixed_old.equ(1.0, pres_fixed);
      }
      pres_tmp_2.equ(1.0, pres_fixed);
      u_tmp_2.equ(1.0, u_fixed);

      //--- Second stage of HYPERBOLIC operator
      HYPERBOLIC_stage = 2;
      euler_matrix.set_HYPERBOLIC_stage(HYPERBOLIC_stage);

      verbose_cout << "  Update density stage 2" << std::endl;
      update_density();
      pcout<<"Minimal density "<<get_minimal_density()<<std::endl;
      pcout<<"Maximal density "<<get_maximal_density()<<std::endl;

      verbose_cout << "  Fixed point pressure stage 2" << std::endl;
      euler_matrix.set_rho_for_fixed(rho_tmp_3);
      mg_transfer.interpolate_to_mg(dof_handler_density, level_projection, rho_tmp_3);
      for(unsigned int level = 0; level < triangulation.n_global_levels(); ++level)
        mg_matrices_euler[level].set_rho_for_fixed(level_projection[level]);
      pres_fixed_old.equ(1.0, pres_tmp_2);
      u_fixed.equ(1.0, u_tmp_2);
      for(unsigned int iter = 0; iter < 100; ++iter) {
        pressure_fixed_point();
        update_velocity();

        //Compute the relative error
        VectorTools::integrate_difference(dof_handler_temperature, pres_fixed, ZeroFunction<dim>(),
                                          Linfty_error_per_cell_pres, quadrature_temperature, VectorTools::Linfty_norm);
        const double den = VectorTools::compute_global_error(triangulation, Linfty_error_per_cell_pres, VectorTools::Linfty_norm);
        double error = 0.0;
        pres_tmp.equ(1.0, pres_fixed);
        pres_tmp.add(-1.0, pres_fixed_old);
        VectorTools::integrate_difference(dof_handler_temperature, pres_tmp, ZeroFunction<dim>(),
                                          Linfty_error_per_cell_pres, quadrature_temperature, VectorTools::Linfty_norm);
        error = VectorTools::compute_global_error(triangulation, Linfty_error_per_cell_pres, VectorTools::Linfty_norm)/den;
        if(error < 1e-10)
          break;

        pres_fixed_old.equ(1.0, pres_fixed);
      }
      pres_tmp_3.equ(1.0, pres_fixed);
      u_tmp_3.equ(1.0, u_fixed);

      // --- Final stage of RK scheme to update
      HYPERBOLIC_stage = 3;
      euler_matrix.set_HYPERBOLIC_stage(HYPERBOLIC_stage);

      verbose_cout << "  Update density" << std::endl;
      update_density();
      pcout<<"Minimal density "<<get_minimal_density()<<std::endl;
      pcout<<"Maximal density "<<get_maximal_density()<<std::endl;

      verbose_cout << "  Update velocity" << std::endl;
      euler_matrix.set_rho_for_fixed(rho_curr);
      mg_transfer.interpolate_to_mg(dof_handler_density, level_projection, rho_curr);
      for(unsigned int level = 0; level < triangulation.n_global_levels(); ++level)
        mg_matrices_euler[level].set_rho_for_fixed(level_projection[level]);
      update_velocity();

      verbose_cout << "  Update pressure" << std::endl;
      update_pressure();

      rho_old.equ(1.0, rho_curr);
      u_old.equ(1.0, u_curr);

      rho_old.add(1.0, dt_tau_rho);
      rho_old.scale(dt_tau_rho_aux);
      u_old.add(1.0, dt_tau_u);
      u_old.scale(dt_tau_u_aux);
      pres_old.add(1.0, dt_tau_pres);
      pres_old.scale(dt_tau_pres_aux);

      rho_old.add(1.0, dt_tau_rho_right);
      rho_old.scale(dt_tau_rho_aux_right);
      u_old.add(1.0, dt_tau_u_right);
      u_old.scale(dt_tau_u_aux_right);
      pres_old.add(1.0, dt_tau_pres_right);
      pres_old.scale(dt_tau_pres_aux_right);

      rho_old.add(1.0, dt_tau_rho_left);
      rho_old.scale(dt_tau_rho_aux_left);
      u_old.add(1.0, dt_tau_u_left);
      u_old.scale(dt_tau_u_aux_left);
      pres_old.add(1.0, dt_tau_pres_left);
      pres_old.scale(dt_tau_pres_aux_left);

      const double max_celerity = compute_max_celerity();
      pcout<< "Maximal celerity = " << 1.0/Ma*max_celerity << std::endl;
      pcout << "CFL_c = " << 1.0/Ma*dt*max_celerity*EquationData::degree_u*
                             std::sqrt(dim)/GridTools::minimal_cell_diameter(triangulation) << std::endl;
      const auto max_C_x_w = compute_max_C_x_w();
      pcout << "CFL_c_x = " << max_C_x_w.first << std::endl;
      pcout << "CFL_c_w = " << max_C_x_w.second << std::endl;
      const double max_velocity = get_maximal_velocity();
      pcout<< "Maximal velocity = " << max_velocity << std::endl;
      pcout << "CFL_u = " << dt*max_velocity*EquationData::degree_u*
                             std::sqrt(dim)/GridTools::minimal_cell_diameter(triangulation) << std::endl;
      const auto max_Cu_x_w = compute_max_Cu_x_w();
      pcout << "CFL_u_x = " << max_Cu_x_w.first << std::endl;
      pcout << "CFL_u_w = " << max_Cu_x_w.second << std::endl;
      if(n % output_interval == 0) {
        verbose_cout << "Plotting Solution final" << std::endl;
        output_results(n);
      }
      if(T - time < dt && T - time > 1e-10) {
        dt = T - time;
        euler_matrix.set_dt(dt);
        for(unsigned int level = 0; level < triangulation.n_global_levels(); ++level)
          mg_matrices_euler[level].set_dt(dt);
      }
    }
    if(n % output_interval != 0) {
      verbose_cout << "Plotting Solution final" << std::endl;
      output_results(n);
    }
  }

} // namespace Atmospheric_Flow


// @sect{ The main function }

// The main function is quite standard. We just need to declare the EulerSolver
// instance and let the simulation run.
//
int main(int argc, char *argv[]) {
  try {
    using namespace Atmospheric_Flow;

    RunTimeParameters::Data_Storage data;
    data.read_data("parameter-file.prm");

    Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, -1);

    const auto& curr_rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
    deallog.depth_console(data.verbose && curr_rank == 0 ? 2 : 0);

    EulerSolver<2> test(data);
    test.run(data.verbose, data.output_interval);

    if(curr_rank == 0)
      std::cout << "----------------------------------------------------"
                << std::endl
                << "Apparently everything went fine!" << std::endl
                << "Don't forget to brush your teeth :-)" << std::endl
                << std::endl;

    return 0;
  }
  catch(std::exception& exc) {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Exception on processing: " << std::endl
              << exc.what() << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;
    return 1;
  }
  catch(...) {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Unknown exception!" << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;
    return 1;
  }

}
