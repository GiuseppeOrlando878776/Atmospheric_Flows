/* Author: Giuseppe Orlando, 2024. */

// @sect{Include files}

// We start by including the necessary header file
//
#include "euler_operator.h"

// This is the class that implements the discretization of the viscous operator
//
namespace Turbulent_Diffusivity {
  using namespace dealii;

  // @sect{ <code>TurbulentOperator::TurbulentOperator</code> }
  //
  template<int dim, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d, int n_q_points_1d_boundary, typename Vec>
  class TurbulentOperator: public MatrixFreeOperators::Base<dim, Vec> {
  public:
    using Number = typename Vec::value_type;

    TurbulentOperator(); /*--- Default constructor ---*/

    TurbulentOperator(RunTimeParameters::Data_Storage& data); /*--- Constructor with some input related data ---*/

    void set_dt(const double time_step); /*--- Setter of the time-step. This is useful both for multigrid purposes and also
                                               in case of modifications of the time step. ---*/

    void set_VISCOUS_stage(const unsigned int stage); /*--- Setter of the TR-BDF2 stage. ---*/

    void set_NS_stage(const unsigned int stage); /*--- Setter of the equation currently under solution. ---*/

    void set_u_curr(const Vec& src); /*--- Setter of the current velocity. This is for the assembling of the bilinear forms
                                           where only one source vector can be passed in input to linearize diffusion coefficient. ---*/

    void set_theta_curr(const Vec& src); /*--- Setter of the current potential temperature. This is for the assembling of the bilinear forms
                                               where only one source vector can be passed in input to linearize diffusion coefficient. ---*/

    void vmult_rhs_velocity(Vec& dst, const std::vector<Vec>& src) const;  /*--- Auxiliary function to assemble the rhs for the velocity. ---*/

    void vmult_rhs_temperature(Vec& dst, const std::vector<Vec>& src) const; /*--- Auxiliary function to assemble the rhs for the temperature. ---*/

    virtual void compute_diagonal() override; /*--- Compute the diagonal for several preconditioners ---*/

  protected:
    double       dt; /*--- Time step. ---*/

    const double gamma; /*--- TR-BDF2 (i.e. implicit part) parameter. ---*/
    /*--- The following variables follow the classical Butcher tableaux notation. Notice that we are using just the implicit part ---*/
    const double a21_tilde;
    const double a22_tilde;
    const double a31_tilde;
    const double a32_tilde;
    const double a33_tilde;

    unsigned int VISCOUS_stage; /*--- Flag for the TR-BDF2 stage ---*/
    unsigned int NS_stage;      /*--- Flag for the equation actually solved ---*/

    virtual void apply_add(Vec& dst, const Vec& src) const override; /*--- Overriden function which actually assembles the
                                                                           bilinear forms ---*/

  private:
    /*--- Parameters related to IP ---*/
    const double C_T = 1.0*(fe_degree_T + 1)*(fe_degree_T + 1);
    const double C_u = 1.0*(fe_degree_u + 1)*(fe_degree_u + 1);

    Vec u_curr,
        theta_curr;

    /*--- Assembler functions for the rhs related to the velocity equation. Here, and also in the following,
          we distinguish between the contribution for cells, faces and boundary. ---*/
    void assemble_rhs_cell_term_velocity(const MatrixFree<dim, Number>&               data,
                                         Vec&                                         dst,
                                         const std::vector<Vec>&                      src,
                                         const std::pair<unsigned int, unsigned int>& cell_range) const;
    void assemble_rhs_face_term_velocity(const MatrixFree<dim, Number>&               data,
                                         Vec&                                         dst,
                                         const std::vector<Vec>&                      src,
                                         const std::pair<unsigned int, unsigned int>& face_range) const;
    void assemble_rhs_boundary_term_velocity(const MatrixFree<dim, Number>&               data,
                                             Vec&                                         dst,
                                             const std::vector<Vec>&                      src,
                                             const std::pair<unsigned int, unsigned int>& face_range) const {}

    /*--- Assembler functions related to the bilinear form of the velocity equation. ---*/
    void assemble_cell_term_velocity(const MatrixFree<dim, Number>&               data,
                                     Vec&                                         dst,
                                     const Vec&                                   src,
                                     const std::pair<unsigned int, unsigned int>& cell_range) const;
    void assemble_face_term_velocity(const MatrixFree<dim, Number>&               data,
                                     Vec&                                         dst,
                                     const Vec&                                   src,
                                     const std::pair<unsigned int, unsigned int>& face_range) const;
    void assemble_boundary_term_velocity(const MatrixFree<dim, Number>&               data,
                                         Vec&                                         dst,
                                         const Vec&                                   src,
                                         const std::pair<unsigned int, unsigned int>& face_range) const {}

    /*--- Assembler functions for the rhs related to the potential temperature equation. ---*/
    void assemble_rhs_cell_term_temperature(const MatrixFree<dim, Number>&               data,
                                            Vec&                                         dst,
                                            const std::vector<Vec>&                      src,
                                            const std::pair<unsigned int, unsigned int>& cell_range) const;
    void assemble_rhs_face_term_temperature(const MatrixFree<dim, Number>&               data,
                                            Vec&                                         dst,
                                            const std::vector<Vec>&                      src,
                                            const std::pair<unsigned int, unsigned int>& face_range) const;
    void assemble_rhs_boundary_term_temperature(const MatrixFree<dim, Number>&               data,
                                                Vec&                                         dst,
                                                const std::vector<Vec>&                      src,
                                                const std::pair<unsigned int, unsigned int>& face_range) const {}

    /*--- Assembler functions for the potential temperature equation. ---*/
    void assemble_cell_term_temperature(const MatrixFree<dim, Number>&               data,
                                        Vec&                                         dst,
                                        const Vec&                                   src,
                                        const std::pair<unsigned int, unsigned int>& cell_range) const;
    void assemble_face_term_temperature(const MatrixFree<dim, Number>&               data,
                                        Vec&                                         dst,
                                        const Vec&                                   src,
                                        const std::pair<unsigned int, unsigned int>& face_range) const;
    void assemble_boundary_term_temperature(const MatrixFree<dim, Number>&               data,
                                            Vec&                                         dst,
                                            const Vec&                                   src,
                                            const std::pair<unsigned int, unsigned int>& face_range) const {}

    /*--- Assembler functions for the diagonal part of the matrix for the velocity equation. ---*/
    void assemble_diagonal_cell_term_velocity(const MatrixFree<dim, Number>&               data,
                                              Vec&                                         dst,
                                              const unsigned int&                          src,
                                              const std::pair<unsigned int, unsigned int>& cell_range) const;
    void assemble_diagonal_face_term_velocity(const MatrixFree<dim, Number>&               data,
                                              Vec&                                         dst,
                                              const unsigned int&                          src,
                                              const std::pair<unsigned int, unsigned int>& face_range) const;
    void assemble_diagonal_boundary_term_velocity(const MatrixFree<dim, Number>&               data,
                                                  Vec&                                         dst,
                                                  const unsigned int&                          src,
                                                  const std::pair<unsigned int, unsigned int>& face_range) const {}

    /*--- Assembler functions for the diagonal part of the matrix for the potential temperature equation. ---*/
    void assemble_diagonal_cell_term_temperature(const MatrixFree<dim, Number>&               data,
                                                 Vec&                                         dst,
                                                 const unsigned int&                          src,
                                                 const std::pair<unsigned int, unsigned int>& cell_range) const;
    void assemble_diagonal_face_term_temperature(const MatrixFree<dim, Number>&               data,
                                                 Vec&                                         dst,
                                                 const unsigned int&                          src,
                                                 const std::pair<unsigned int, unsigned int>& face_range) const;
    void assemble_diagonal_boundary_term_temperature(const MatrixFree<dim, Number>&               data,
                                                     Vec&                                         dst,
                                                     const unsigned int&                          src,
                                                     const std::pair<unsigned int, unsigned int>& face_range) const {}
  };


  // Default constructor
  //
  template<int dim, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d, int n_q_points_1d_boundary, typename Vec>
  TurbulentOperator<dim, fe_degree_T, fe_degree_u,
                         n_q_points_1d, n_q_points_1d_boundary, Vec>::
  TurbulentOperator(): MatrixFreeOperators::Base<dim, Vec>(), dt(), gamma(2.0 - std::sqrt(2.0)),
                       a21_tilde(0.5*gamma), a22_tilde(0.5*gamma),
                       a31_tilde(0.5 - 0.25*gamma), a32_tilde(0.5 - 0.25*gamma), a33_tilde(0.5*gamma),
                       VISCOUS_stage(1), NS_stage(1) {}

  // Constructor with runtime parameters storage
  //
  template<int dim, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d, int n_q_points_1d_boundary, typename Vec>
  TurbulentOperator<dim, fe_degree_T, fe_degree_u,
                         n_q_points_1d, n_q_points_1d_boundary, Vec>::
  TurbulentOperator(RunTimeParameters::Data_Storage& data): MatrixFreeOperators::Base<dim, Vec>(), dt(data.dt), gamma(2.0 - std::sqrt(2.0)),
                                                            a21_tilde(0.5*gamma), a22_tilde(0.5*gamma),
                                                            a31_tilde(0.5 - 0.25*gamma), a32_tilde(0.5 - 0.25*gamma), a33_tilde(0.5*gamma),
                                                            VISCOUS_stage(1), NS_stage(1) {}


  // Setter of time-step
  //
  template<int dim, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d, int n_q_points_1d_boundary, typename Vec>
  void TurbulentOperator<dim, fe_degree_T, fe_degree_u,
                              n_q_points_1d, n_q_points_1d_boundary, Vec>::
  set_dt(const double time_step) {
    dt = time_step;
  }

  // Setter of VISCOUS stage (this can be known only during the effective execution
  // and so it has to be demanded to the class that really solves the problem)
  //
  template<int dim, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d, int n_q_points_1d_boundary, typename Vec>
  void TurbulentOperator<dim, fe_degree_T, fe_degree_u,
                              n_q_points_1d, n_q_points_1d_boundary, Vec>::
  set_VISCOUS_stage(const unsigned int stage) {
    AssertIndexRange(stage, 4);
    Assert(stage > 0, ExcInternalError());

    VISCOUS_stage = stage;
  }

  // Setter of NS stage (this can be known only during the effective execution
  // and so it has to be demanded to the class that really solves the problem)
  //
  template<int dim, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d, int n_q_points_1d_boundary, typename Vec>
  void TurbulentOperator<dim, fe_degree_T, fe_degree_u,
                              n_q_points_1d, n_q_points_1d_boundary, Vec>::
  set_NS_stage(const unsigned int stage) {
    AssertIndexRange(stage, 3);
    Assert(stage > 0, ExcInternalError());

    NS_stage = stage;
  }

  // Setter of current velocity
  //
  template<int dim, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d, int n_q_points_1d_boundary, typename Vec>
  void TurbulentOperator<dim, fe_degree_T, fe_degree_u,
                              n_q_points_1d, n_q_points_1d_boundary, Vec>::
  set_u_curr(const Vec& src) {
    u_curr = src;
    u_curr.update_ghost_values();
  }

  // Setter of current potential temperature
  //
  template<int dim, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d, int n_q_points_1d_boundary, typename Vec>
  void TurbulentOperator<dim, fe_degree_T, fe_degree_u,
                              n_q_points_1d, n_q_points_1d_boundary, Vec>::
  set_theta_curr(const Vec& src) {
    theta_curr = src;
    theta_curr.update_ghost_values();
  }


  // Assemble rhs cell term for the velocity equation
  //
  template<int dim, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d, int n_q_points_1d_boundary, typename Vec>
  void TurbulentOperator<dim, fe_degree_T, fe_degree_u,
                              n_q_points_1d, n_q_points_1d_boundary, Vec>::
  assemble_rhs_cell_term_velocity(const MatrixFree<dim, Number>&               data,
                                  Vec&                                         dst,
                                  const std::vector<Vec>&                      src,
                                  const std::pair<unsigned int, unsigned int>& cell_range) const {
    if(VISCOUS_stage == 2) {
      /*--- We start by declaring suitable instances to read the available quantities ---*/
      FEEvaluation<dim, fe_degree_u, n_q_points_1d, dim, Number> phi(data, 0),
                                                                 phi_u_curr(data, 0);
      FEEvaluation<dim, fe_degree_T, n_q_points_1d, 1, Number>   phi_theta_curr(data, 1);

      /*--- Loop over all cells ---*/
      for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
        phi_u_curr.reinit(cell);
        phi_u_curr.gather_evaluate(src[0], EvaluationFlags::values | EvaluationFlags::gradients);
        phi_theta_curr.reinit(cell);
        phi_theta_curr.gather_evaluate(src[1], EvaluationFlags::gradients);

        phi.reinit(cell);

        /*--- Loop over all quadrature points. ---*/
        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          const auto& u_curr = phi_u_curr.get_value(q);

          /*--- Compute contribution at initialization ---*/
          const auto& grad_u_curr              = phi_u_curr.get_gradient(q);
          const auto& grad_theta_curr          = phi_theta_curr.get_gradient(q);
          const auto& mod_squared_grad_uz_curr = grad_u_curr[0][dim - 1]*grad_u_curr[0][dim - 1];
          const auto& Ri_curr                  = 1.0/(EquationData::Fr2)*grad_theta_curr[dim - 1]/mod_squared_grad_uz_curr;
          VectorizedArray<Number> b;
          VectorizedArray<Number> beta;
          for(unsigned int idx = 0; idx < VectorizedArray<Number>::size(); ++idx) {
            if(Ri_curr[idx] > 0.0) {
              b[idx]    = 5.0;
              beta[idx] = -2.0;
            }
            else {
              b[idx]    = 20.0;
              beta[idx] = 0.5;
            }
          }
          const auto& kappa_curr = EquationData::l_mixing*EquationData::l_mixing*std::sqrt(mod_squared_grad_uz_curr)*
                                   std::pow(1.0 + b*std::abs(Ri_curr), beta);
          Tensor<2, dim, VectorizedArray<Number>> diff_flux_curr;
          diff_flux_curr = 0;
          diff_flux_curr[0][dim - 1] = kappa_curr*grad_u_curr[0][dim - 1];

          phi.submit_value(u_curr, q);
          phi.submit_gradient(-a21_tilde*dt*diff_flux_curr, q);
        }

        phi.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
      }
    }
    else {
      /*--- We start by declaring suitable instances to read the available quantities ---*/
      FEEvaluation<dim, fe_degree_u, n_q_points_1d, dim, Number> phi(data, 0),
                                                                 phi_u_curr(data, 0),
                                                                 phi_u_tmp_2(data, 0);
      FEEvaluation<dim, fe_degree_T, n_q_points_1d, 1, Number>   phi_theta_curr(data, 1),
                                                                 phi_theta_tmp_2(data, 1);

      /*--- Loop over all cells ---*/
      for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
        phi_u_curr.reinit(cell);
        phi_u_curr.gather_evaluate(src[0], EvaluationFlags::values | EvaluationFlags::gradients);
        phi_theta_curr.reinit(cell);
        phi_theta_curr.gather_evaluate(src[1], EvaluationFlags::gradients);

        phi_u_tmp_2.reinit(cell);
        phi_u_tmp_2.gather_evaluate(src[2], EvaluationFlags::gradients);
        phi_theta_tmp_2.reinit(cell);
        phi_theta_tmp_2.gather_evaluate(src[3], EvaluationFlags::gradients);

        phi.reinit(cell);

        /*--- Loop over all quadrature points. ---*/
        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          const auto& u_curr = phi_u_curr.get_value(q);

          /*--- Compute contribution at initialization ---*/
          const auto& grad_u_curr              = phi_u_curr.get_gradient(q);
          const auto& grad_theta_curr          = phi_theta_curr.get_gradient(q);
          const auto& mod_squared_grad_uz_curr = grad_u_curr[0][dim - 1]*grad_u_curr[0][dim - 1];
          const auto& Ri_curr                  = 1.0/(EquationData::Fr2)*grad_theta_curr[dim - 1]/mod_squared_grad_uz_curr;
          VectorizedArray<Number> b;
          VectorizedArray<Number> beta;
          for(unsigned int idx = 0; idx < VectorizedArray<Number>::size(); ++idx) {
            if(Ri_curr[idx] > 0.0) {
              b[idx]    = 5.0;
              beta[idx] = -2.0;
            }
            else {
              b[idx]    = 20.0;
              beta[idx] = 0.5;
            }
          }
          const auto& kappa_curr = EquationData::l_mixing*EquationData::l_mixing*std::sqrt(mod_squared_grad_uz_curr)*
                                   std::pow(1.0 + b*std::abs(Ri_curr), beta);
          Tensor<2, dim, VectorizedArray<Number>> diff_flux_curr;
          diff_flux_curr = 0;
          diff_flux_curr[0][dim - 1] = kappa_curr*grad_u_curr[0][dim - 1];

          /*--- Compute contribution at previous stage ---*/
          const auto& grad_u_tmp_2              = phi_u_tmp_2.get_gradient(q);
          const auto& grad_theta_tmp_2          = phi_theta_tmp_2.get_gradient(q);
          const auto& mod_squared_grad_uz_tmp_2 = grad_u_tmp_2[0][dim - 1]*grad_u_tmp_2[0][dim - 1];
          const auto& Ri_tmp_2                  = 1.0/(EquationData::Fr2)*grad_theta_tmp_2[dim - 1]/mod_squared_grad_uz_tmp_2;
          for(unsigned int idx = 0; idx < VectorizedArray<Number>::size(); ++idx) {
            if(Ri_tmp_2[idx] > 0.0) {
              b[idx]    = 5.0;
              beta[idx] = -2.0;
            }
            else {
              b[idx]    = 20.0;
              beta[idx] = 0.5;
            }
          }
          const auto& kappa_tmp_2 = EquationData::l_mixing*EquationData::l_mixing*std::sqrt(mod_squared_grad_uz_tmp_2)*
                                    std::pow(1.0 + b*std::abs(Ri_tmp_2), beta);
          Tensor<2, dim, VectorizedArray<Number>> diff_flux_tmp_2;
          diff_flux_tmp_2 = 0;
          diff_flux_tmp_2[0][dim - 1] = kappa_tmp_2*grad_u_tmp_2[0][dim - 1];

          phi.submit_value(u_curr, q);
          phi.submit_gradient(-a31_tilde*dt*diff_flux_curr
                              -a32_tilde*dt*diff_flux_tmp_2, q);
        }

        phi.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
      }
    }
  }

  // Assemble rhs face term for the velocity equation
  //
  template<int dim, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d, int n_q_points_1d_boundary, typename Vec>
  void TurbulentOperator<dim, fe_degree_T, fe_degree_u,
                              n_q_points_1d, n_q_points_1d_boundary, Vec>::
  assemble_rhs_face_term_velocity(const MatrixFree<dim, Number>&               data,
                                  Vec&                                         dst,
                                  const std::vector<Vec>&                      src,
                                  const std::pair<unsigned int, unsigned int>& face_range) const {
    if(VISCOUS_stage == 2) {
      /*--- We start by declaring suitable quantities to read the available quantities ---*/
      FEFaceEvaluation<dim, fe_degree_u, n_q_points_1d, dim, Number> phi_p(data, true, 0),
                                                                     phi_m(data, false, 0),
                                                                     phi_u_curr_p(data, true, 0),
                                                                     phi_u_curr_m(data, false, 0);
      FEFaceEvaluation<dim, fe_degree_T, n_q_points_1d, 1, Number>   phi_theta_curr_p(data, true, 1),
                                                                     phi_theta_curr_m(data, false, 1);

      /*--- Loop over all internal faces ---*/
      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        phi_u_curr_p.reinit(face);
        phi_u_curr_p.gather_evaluate(src[0], EvaluationFlags::values | EvaluationFlags::gradients);
        phi_u_curr_m.reinit(face);
        phi_u_curr_m.gather_evaluate(src[0], EvaluationFlags::values | EvaluationFlags::gradients);
        phi_theta_curr_p.reinit(face);
        phi_theta_curr_p.gather_evaluate(src[1], EvaluationFlags::gradients);
        phi_theta_curr_m.reinit(face);
        phi_theta_curr_m.gather_evaluate(src[1], EvaluationFlags::gradients);

        phi_p.reinit(face);
        phi_m.reinit(face);

        const auto coef_jump = C_u*0.5*(std::abs((phi_p.get_normal_vector(0) * phi_p.inverse_jacobian(0))[dim - 1]) +
                                        std::abs((phi_m.get_normal_vector(0) * phi_m.inverse_jacobian(0))[dim - 1])); /*--- Jump constant for IP ---*/

        /*--- Loop over all quadrature points ---*/
        for(unsigned int q = 0; q < phi_p.n_q_points; ++q) {
          const auto& n_plus = phi_p.get_normal_vector(q);

          /*--- Compute contribution at initialization ---*/
          const auto& grad_u_curr_p              = phi_u_curr_p.get_gradient(q);
          const auto& grad_u_curr_m              = phi_u_curr_m.get_gradient(q);
          const auto& grad_theta_curr_p          = phi_theta_curr_p.get_gradient(q);
          const auto& grad_theta_curr_m          = phi_theta_curr_m.get_gradient(q);
          const auto& mod_squared_grad_uz_curr_p = grad_u_curr_p[0][dim - 1]*grad_u_curr_p[0][dim - 1];
          const auto& mod_squared_grad_uz_curr_m = grad_u_curr_m[0][dim - 1]*grad_u_curr_m[0][dim - 1];
          const auto& Ri_curr_p                  = 1.0/(EquationData::Fr2)*grad_theta_curr_p[dim - 1]/mod_squared_grad_uz_curr_p;
          const auto& Ri_curr_m                  = 1.0/(EquationData::Fr2)*grad_theta_curr_m[dim - 1]/mod_squared_grad_uz_curr_m;
          VectorizedArray<Number> b_p, b_m;
          VectorizedArray<Number> beta_p, beta_m;
          for(unsigned int idx = 0; idx < VectorizedArray<Number>::size(); ++idx) {
            if(Ri_curr_p[idx] > 0.0) {
              b_p[idx]    = 5.0;
              beta_p[idx] = -2.0;
            }
            else {
              b_p[idx]    = 20.0;
              beta_p[idx] = 0.5;
            }

            if(Ri_curr_m[idx] > 0.0) {
              b_m[idx]    = 5.0;
              beta_m[idx] = -2.0;
            }
            else {
              b_m[idx]    = 20.0;
              beta_m[idx] = 0.5;
            }
          }
          const auto& kappa_curr_p = EquationData::l_mixing*EquationData::l_mixing*std::sqrt(mod_squared_grad_uz_curr_p)*
                                     std::pow(1.0 + b_p*std::abs(Ri_curr_p), beta_p);
          const auto& kappa_curr_m = EquationData::l_mixing*EquationData::l_mixing*std::sqrt(mod_squared_grad_uz_curr_m)*
                                     std::pow(1.0 + b_m*std::abs(Ri_curr_m), beta_m);
          Tensor<2, dim, VectorizedArray<Number>> avg_diff_flux_curr;
          avg_diff_flux_curr = 0;
          avg_diff_flux_curr[0][dim - 1] = 0.5*(kappa_curr_p*grad_u_curr_p[0][dim - 1] +
                                                kappa_curr_m*grad_u_curr_m[0][dim - 1]);

          /*--- Consider also jump penalization contribution ---*/
          const auto& u_curr_p       = phi_u_curr_p.get_value(q);
          const auto& u_curr_m       = phi_u_curr_m.get_value(q);
          const auto& avg_kappa_curr = 2.0/(1.0/kappa_curr_p + 1.0/kappa_curr_m);
          auto jump_u_curr           = u_curr_p - u_curr_m;
          jump_u_curr[dim - 1] = make_vectorized_array<Number>(0.0);

          phi_p.submit_value(a21_tilde*dt*(avg_diff_flux_curr*n_plus -
                                           coef_jump*avg_kappa_curr*jump_u_curr), q);
          phi_m.submit_value(-a21_tilde*dt*(avg_diff_flux_curr*n_plus -
                                            coef_jump*avg_kappa_curr*jump_u_curr), q);
        }

        phi_p.integrate_scatter(EvaluationFlags::values, dst);
        phi_m.integrate_scatter(EvaluationFlags::values, dst);
      }
    }
    else {
      /*--- We start by declaring suitable instances to read the available quantities ---*/
      FEFaceEvaluation<dim, fe_degree_u, n_q_points_1d, dim, Number> phi_p(data, true, 0),
                                                                     phi_m(data, false, 0),
                                                                     phi_u_curr_p(data, true, 0),
                                                                     phi_u_curr_m(data, false, 0),
                                                                     phi_u_tmp_2_p(data, true, 0),
                                                                     phi_u_tmp_2_m(data, false, 0);
      FEFaceEvaluation<dim, fe_degree_T, n_q_points_1d, 1, Number>   phi_theta_curr_p(data, true, 1),
                                                                     phi_theta_curr_m(data, false, 1),
                                                                     phi_theta_tmp_2_p(data, true, 1),
                                                                     phi_theta_tmp_2_m(data, false, 1);

      /*--- Loop over all internal faces ---*/
      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        phi_u_curr_p.reinit(face);
        phi_u_curr_p.gather_evaluate(src[0], EvaluationFlags::values | EvaluationFlags::gradients);
        phi_u_curr_m.reinit(face);
        phi_u_curr_m.gather_evaluate(src[0], EvaluationFlags::values | EvaluationFlags::gradients);
        phi_theta_curr_p.reinit(face);
        phi_theta_curr_p.gather_evaluate(src[1], EvaluationFlags::gradients);
        phi_theta_curr_m.reinit(face);
        phi_theta_curr_m.gather_evaluate(src[1], EvaluationFlags::gradients);

        phi_u_tmp_2_p.reinit(face);
        phi_u_tmp_2_p.gather_evaluate(src[2], EvaluationFlags::values | EvaluationFlags::gradients);
        phi_u_tmp_2_m.reinit(face);
        phi_u_tmp_2_m.gather_evaluate(src[2], EvaluationFlags::values | EvaluationFlags::gradients);
        phi_theta_tmp_2_p.reinit(face);
        phi_theta_tmp_2_p.gather_evaluate(src[3], EvaluationFlags::gradients);
        phi_theta_tmp_2_m.reinit(face);
        phi_theta_tmp_2_m.gather_evaluate(src[3], EvaluationFlags::gradients);

        phi_p.reinit(face);
        phi_m.reinit(face);

        const auto coef_jump = C_u*0.5*(std::abs((phi_p.get_normal_vector(0) * phi_p.inverse_jacobian(0))[dim - 1]) +
                                        std::abs((phi_m.get_normal_vector(0) * phi_m.inverse_jacobian(0))[dim - 1]));

        /*--- Loop over all quadrature points ---*/
        for(unsigned int q = 0; q < phi_p.n_q_points; ++q) {
          const auto& n_plus = phi_p.get_normal_vector(q);

          /*--- Compute contribution at initialization ---*/
          const auto& grad_u_curr_p               = phi_u_curr_p.get_gradient(q);
          const auto& grad_u_curr_m               = phi_u_curr_m.get_gradient(q);
          const auto& grad_theta_curr_p           = phi_theta_curr_p.get_gradient(q);
          const auto& grad_theta_curr_m           = phi_theta_curr_m.get_gradient(q);
          const auto& mod_squared_grad_uz_curr_p  = grad_u_curr_p[0][dim - 1]*grad_u_curr_p[0][dim - 1];
          const auto& mod_squared_grad_uz_curr_m  = grad_u_curr_m[0][dim - 1]*grad_u_curr_m[0][dim - 1];
          const auto& Ri_curr_p                   = 1.0/(EquationData::Fr2)*grad_theta_curr_p[dim - 1]/mod_squared_grad_uz_curr_p;
          const auto& Ri_curr_m                   = 1.0/(EquationData::Fr2)*grad_theta_curr_m[dim - 1]/mod_squared_grad_uz_curr_m;
          VectorizedArray<Number> b_p, b_m;
          VectorizedArray<Number> beta_p, beta_m;
          for(unsigned int idx = 0; idx < VectorizedArray<Number>::size(); ++idx) {
            if(Ri_curr_p[idx] > 0.0) {
              b_p[idx]    = 5.0;
              beta_p[idx] = -2.0;
            }
            else {
              b_p[idx]    = 20.0;
              beta_p[idx] = 0.5;
            }

            if(Ri_curr_m[idx] > 0.0) {
              b_m[idx]    = 5.0;
              beta_m[idx] = -2.0;
            }
            else {
              b_m[idx]    = 20.0;
              beta_m[idx] = 0.5;
            }
          }
          const auto& kappa_curr_p = EquationData::l_mixing*EquationData::l_mixing*std::sqrt(mod_squared_grad_uz_curr_p)*
                                     std::pow(1.0 + b_p*std::abs(Ri_curr_p), beta_p);
          const auto& kappa_curr_m = EquationData::l_mixing*EquationData::l_mixing*std::sqrt(mod_squared_grad_uz_curr_m)*
                                     std::pow(1.0 + b_m*std::abs(Ri_curr_m), beta_m);
          Tensor<2, dim, VectorizedArray<Number>> avg_diff_flux_curr;
          avg_diff_flux_curr = 0;
          avg_diff_flux_curr[0][dim - 1] = 0.5*(kappa_curr_p*grad_u_curr_p[0][dim - 1] +
                                                kappa_curr_m*grad_u_curr_m[0][dim - 1]);

          /*--- Compute contribution at previous stage ---*/
          const auto& grad_u_tmp_2_p              = phi_u_tmp_2_p.get_gradient(q);
          const auto& grad_u_tmp_2_m              = phi_u_tmp_2_m.get_gradient(q);
          const auto& grad_theta_tmp_2_p          = phi_theta_tmp_2_p.get_gradient(q);
          const auto& grad_theta_tmp_2_m          = phi_theta_tmp_2_m.get_gradient(q);
          const auto& mod_squared_grad_uz_tmp_2_p = grad_u_tmp_2_p[0][dim - 1]*grad_u_tmp_2_p[0][dim - 1];
          const auto& mod_squared_grad_uz_tmp_2_m = grad_u_tmp_2_m[0][dim - 1]*grad_u_tmp_2_m[0][dim - 1];
          const auto& Ri_tmp_2_p                  = 1.0/(EquationData::Fr2)*grad_theta_tmp_2_p[dim - 1]/mod_squared_grad_uz_tmp_2_p;
          const auto& Ri_tmp_2_m                  = 1.0/(EquationData::Fr2)*grad_theta_tmp_2_m[dim - 1]/mod_squared_grad_uz_tmp_2_m;
          for(unsigned int idx = 0; idx < VectorizedArray<Number>::size(); ++idx) {
            if(Ri_tmp_2_p[idx] > 0.0) {
              b_p[idx]    = 5.0;
              beta_p[idx] = -2.0;
            }
            else {
              b_p[idx]    = 20.0;
              beta_p[idx] = 0.5;
            }

            if(Ri_tmp_2_m[idx] > 0.0) {
              b_m[idx]    = 5.0;
              beta_m[idx] = -2.0;
            }
            else {
              b_m[idx]    = 20.0;
              beta_m[idx] = 0.5;
            }
          }
          const auto& kappa_tmp_2_p = EquationData::l_mixing*EquationData::l_mixing*std::sqrt(mod_squared_grad_uz_tmp_2_p)*
                                      std::pow(1.0 + b_p*std::abs(Ri_tmp_2_p), beta_p);
          const auto& kappa_tmp_2_m = EquationData::l_mixing*EquationData::l_mixing*std::sqrt(mod_squared_grad_uz_tmp_2_m)*
                                      std::pow(1.0 + b_m*std::abs(Ri_tmp_2_m), beta_m);
          Tensor<2, dim, VectorizedArray<Number>> avg_diff_flux_tmp_2;
          avg_diff_flux_tmp_2 = 0;
          avg_diff_flux_tmp_2[0][dim - 1] = 0.5*(kappa_tmp_2_p*grad_u_tmp_2_p[0][dim - 1] +
                                                 kappa_tmp_2_m*grad_u_tmp_2_m[0][dim - 1]);

          /*--- Consider also jump penalization contribution ---*/
          const auto& u_curr_p       = phi_u_curr_p.get_value(q);
          const auto& u_curr_m       = phi_u_curr_m.get_value(q);
          const auto& avg_kappa_curr = 2.0/(1.0/kappa_curr_p + 1.0/kappa_curr_m);
          auto jump_u_curr           = u_curr_p - u_curr_m;
          jump_u_curr[dim - 1]       = make_vectorized_array<Number>(0.0);

          const auto& u_tmp_2_p       = phi_u_tmp_2_p.get_value(q);
          const auto& u_tmp_2_m       = phi_u_tmp_2_m.get_value(q);
          const auto& avg_kappa_tmp_2 = 2.0/(1.0/kappa_tmp_2_p + 1.0/kappa_tmp_2_m);
          auto jump_u_tmp_2           = u_tmp_2_p - u_tmp_2_m;
          jump_u_tmp_2[dim - 1]       = make_vectorized_array<Number>(0.0);

          phi_p.submit_value(a31_tilde*dt*(avg_diff_flux_curr*n_plus -
                                           coef_jump*avg_kappa_curr*jump_u_curr) +
                             a32_tilde*dt*(avg_diff_flux_tmp_2*n_plus -
                                           coef_jump*avg_kappa_tmp_2*jump_u_tmp_2), q);
          phi_m.submit_value(-a31_tilde*dt*(avg_diff_flux_curr*n_plus -
                                            coef_jump*avg_kappa_curr*jump_u_curr)
                             -a32_tilde*dt*(avg_diff_flux_tmp_2*n_plus -
                                            coef_jump*avg_kappa_tmp_2*jump_u_tmp_2), q);
        }

        phi_p.integrate_scatter(EvaluationFlags::values, dst);
        phi_m.integrate_scatter(EvaluationFlags::values, dst);
      }
    }
  }

  // Put together all the previous steps for the momentum equation
  //
  template<int dim, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d, int n_q_points_1d_boundary, typename Vec>
  void TurbulentOperator<dim, fe_degree_T, fe_degree_u,
                                 n_q_points_1d, n_q_points_1d_boundary, Vec>::
  vmult_rhs_velocity(Vec& dst, const std::vector<Vec>& src) const {
    for(unsigned int d = 0; d < src.size(); ++d) {
      src[d].update_ghost_values();
    }

    this->data->loop(&TurbulentOperator::assemble_rhs_cell_term_velocity,
                     &TurbulentOperator::assemble_rhs_face_term_velocity,
                     &TurbulentOperator::assemble_rhs_boundary_term_velocity,
                     this, dst, src, true,
                     MatrixFree<dim, Number>::DataAccessOnFaces::unspecified,
                     MatrixFree<dim, Number>::DataAccessOnFaces::unspecified);
  }

  // Assemble cell term for the velocity equation
  //
  template<int dim, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d, int n_q_points_1d_boundary, typename Vec>
  void TurbulentOperator<dim, fe_degree_T, fe_degree_u,
                              n_q_points_1d, n_q_points_1d_boundary, Vec>::
  assemble_cell_term_velocity(const MatrixFree<dim, Number>&               data,
                              Vec&                                         dst,
                              const Vec&                                   src,
                              const std::pair<unsigned int, unsigned int>& cell_range) const {
    /*--- We start declaring suitable instances to read the available quantities ---*/
    FEEvaluation<dim, fe_degree_u, n_q_points_1d, dim, Number> phi(data, 0),
                                                               phi_u_curr(data, 0);
    FEEvaluation<dim, fe_degree_T, n_q_points_1d, 1, Number>   phi_theta_curr(data, 1);

    const double coeff = (VISCOUS_stage == 2) ? a22_tilde*dt : a33_tilde*dt;

    /*--- Loop over all cells ---*/
    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
      phi_u_curr.reinit(cell);
      phi_u_curr.gather_evaluate(u_curr, EvaluationFlags::gradients);
      phi_theta_curr.reinit(cell);
      phi_theta_curr.gather_evaluate(theta_curr, EvaluationFlags::gradients);

      phi.reinit(cell);
      phi.gather_evaluate(src, EvaluationFlags::values | EvaluationFlags::gradients);

      /*--- Loop over all quadrature points ---*/
      for(unsigned int q = 0; q < phi.n_q_points; ++q) {
        const auto& u = phi.get_value(q);

        /*--- Compute contribution at current stage ---*/
        const auto& grad_u_curr              = phi_u_curr.get_gradient(q);
        const auto& grad_theta_curr          = phi_theta_curr.get_gradient(q);
        const auto& mod_squared_grad_uz_curr = grad_u_curr[0][dim - 1]*grad_u_curr[0][dim - 1];
        const auto& Ri_curr                  = 1.0/(EquationData::Fr2)*grad_theta_curr[dim - 1]/mod_squared_grad_uz_curr;
        VectorizedArray<Number> b;
        VectorizedArray<Number> beta;
        for(unsigned int idx = 0; idx < VectorizedArray<Number>::size(); ++idx) {
          if(Ri_curr[idx] > 0.0) {
            b[idx]    = 5.0;
            beta[idx] = -2.0;
          }
          else {
            b[idx]    = 20.0;
            beta[idx] = 0.5;
          }
        }
        const auto& kappa_curr = EquationData::l_mixing*EquationData::l_mixing*std::sqrt(mod_squared_grad_uz_curr)*
                                 std::pow(1.0 + b*std::abs(Ri_curr), beta);

        Tensor<2, dim, VectorizedArray<Number>> diff_flux;
        diff_flux = 0;
        const auto& grad_u    = phi.get_gradient(q);
        diff_flux[0][dim - 1] = kappa_curr*grad_u[0][dim - 1];

        phi.submit_value(u, q);
        phi.submit_gradient(coeff*diff_flux, q);
      }

      phi.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
    }
  }

  // Assemble face term for the velocity equation
  //
  template<int dim, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d, int n_q_points_1d_boundary, typename Vec>
  void TurbulentOperator<dim, fe_degree_T, fe_degree_u,
                              n_q_points_1d, n_q_points_1d_boundary, Vec>::
  assemble_face_term_velocity(const MatrixFree<dim, Number>&               data,
                              Vec&                                         dst,
                              const Vec&                                   src,
                              const std::pair<unsigned int, unsigned int>& face_range) const {
    /*--- We start by declaring suitable instances to read the available quantities ---*/
    FEFaceEvaluation<dim, fe_degree_u, n_q_points_1d, dim, Number> phi_p(data, true, 0),
                                                                   phi_m(data, false, 0),
                                                                   phi_u_curr_p(data, true, 0),
                                                                   phi_u_curr_m(data, false, 0);
    FEFaceEvaluation<dim, fe_degree_T, n_q_points_1d, 1, Number>   phi_theta_curr_p(data, true, 1),
                                                                   phi_theta_curr_m(data, false, 1);

    const double coeff = (VISCOUS_stage == 2) ? a22_tilde*dt : a33_tilde*dt;

    /*--- Loop over all internal faces ---*/
    for(unsigned int face = face_range.first; face < face_range.second; ++face) {
      phi_u_curr_p.reinit(face);
      phi_u_curr_p.gather_evaluate(u_curr, EvaluationFlags::gradients);
      phi_u_curr_m.reinit(face);
      phi_u_curr_m.gather_evaluate(u_curr, EvaluationFlags::gradients);
      phi_theta_curr_p.reinit(face);
      phi_theta_curr_p.gather_evaluate(theta_curr, EvaluationFlags::gradients);
      phi_theta_curr_m.reinit(face);
      phi_theta_curr_m.gather_evaluate(theta_curr, EvaluationFlags::gradients);

      phi_p.reinit(face);
      phi_p.gather_evaluate(src, EvaluationFlags::values | EvaluationFlags::gradients);
      phi_m.reinit(face);
      phi_m.gather_evaluate(src, EvaluationFlags::values | EvaluationFlags::gradients);

      const auto coef_jump = C_u*0.5*(std::abs((phi_p.get_normal_vector(0) * phi_p.inverse_jacobian(0))[dim - 1]) +
                                      std::abs((phi_m.get_normal_vector(0) * phi_m.inverse_jacobian(0))[dim - 1])); /*--- Jump constant for IP ---*/

      /*--- Loop over all quadrature points ---*/
      for(unsigned int q = 0; q < phi_p.n_q_points; ++q) {
        const auto& n_plus = phi_p.get_normal_vector(q);

        /*--- Compute contribution at current stage ---*/
        const auto& grad_u_curr_p              = phi_u_curr_p.get_gradient(q);
        const auto& grad_u_curr_m              = phi_u_curr_m.get_gradient(q);
        const auto& grad_theta_curr_p          = phi_theta_curr_p.get_gradient(q);
        const auto& grad_theta_curr_m          = phi_theta_curr_m.get_gradient(q);
        const auto& mod_squared_grad_uz_curr_p = grad_u_curr_p[0][dim - 1]*grad_u_curr_p[0][dim - 1];
        const auto& mod_squared_grad_uz_curr_m = grad_u_curr_m[0][dim - 1]*grad_u_curr_m[0][dim - 1];
        const auto& Ri_curr_p                  = 1.0/(EquationData::Fr2)*grad_theta_curr_p[dim - 1]/mod_squared_grad_uz_curr_p;
        const auto& Ri_curr_m                  = 1.0/(EquationData::Fr2)*grad_theta_curr_m[dim - 1]/mod_squared_grad_uz_curr_m;
        VectorizedArray<Number> b_p, b_m;
        VectorizedArray<Number> beta_p, beta_m;
        for(unsigned int idx = 0; idx < VectorizedArray<Number>::size(); ++idx) {
          if(Ri_curr_p[idx] > 0.0) {
            b_p[idx]    = 5.0;
            beta_p[idx] = -2.0;
          }
          else {
            b_p[idx]    = 20.0;
            beta_p[idx] = 0.5;
          }

          if(Ri_curr_m[idx] > 0.0) {
            b_m[idx]    = 5.0;
            beta_m[idx] = -2.0;
          }
          else {
            b_m[idx]    = 20.0;
            beta_m[idx] = 0.5;
          }
        }
        const auto& kappa_curr_p = EquationData::l_mixing*EquationData::l_mixing*std::sqrt(mod_squared_grad_uz_curr_p)*
                                   std::pow(1.0 + b_p*std::abs(Ri_curr_p), beta_p);
        const auto& kappa_curr_m = EquationData::l_mixing*EquationData::l_mixing*std::sqrt(mod_squared_grad_uz_curr_m)*
                                   std::pow(1.0 + b_m*std::abs(Ri_curr_m), beta_m);
        Tensor<2, dim, VectorizedArray<Number>> avg_diff_flux;
        avg_diff_flux = 0;
        const auto& grad_u_p      = phi_p.get_gradient(q);
        const auto& grad_u_m      = phi_m.get_gradient(q);
        avg_diff_flux[0][dim - 1] = 0.5*(kappa_curr_p*grad_u_p[0][dim - 1] +
                                         kappa_curr_m*grad_u_m[0][dim - 1]);

        /*--- Consider jump penalization ---*/
        const auto& u_p            = phi_p.get_value(q);
        const auto& u_m            = phi_m.get_value(q);
        const auto& avg_kappa_curr = 2.0/(1.0/kappa_curr_p + 1.0/kappa_curr_m);
        auto jump_u                = u_p - u_m;
        jump_u[dim - 1]            = make_vectorized_array<Number>(0.0);

        phi_p.submit_value(coeff*(-avg_diff_flux*n_plus + coef_jump*avg_kappa_curr*jump_u), q);
        phi_m.submit_value(-coeff*(-avg_diff_flux*n_plus + coef_jump*avg_kappa_curr*jump_u), q);
      }

      phi_p.integrate_scatter(EvaluationFlags::values, dst);
      phi_m.integrate_scatter(EvaluationFlags::values, dst);
    }
  }


  // Assemble rhs cell term for the the potential temperature equation
  //
  template<int dim, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d, int n_q_points_1d_boundary, typename Vec>
  void TurbulentOperator<dim, fe_degree_T, fe_degree_u,
                              n_q_points_1d, n_q_points_1d_boundary, Vec>::
  assemble_rhs_cell_term_temperature(const MatrixFree<dim, Number>&               data,
                                     Vec&                                         dst,
                                     const std::vector<Vec>&                      src,
                                     const std::pair<unsigned int, unsigned int>& cell_range) const {
    if(VISCOUS_stage == 2) {
      /*--- We start by declaring suitable instances to read the available quantities ---*/
      FEEvaluation<dim, fe_degree_T, n_q_points_1d, 1, Number>   phi(data, 1),
                                                                 phi_theta_curr(data, 1);
      FEEvaluation<dim, fe_degree_u, n_q_points_1d, dim, Number> phi_u_curr(data, 0);

      /*--- Loop over all cells ---*/
      for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
        phi_u_curr.reinit(cell);
        phi_u_curr.gather_evaluate(src[0], EvaluationFlags::gradients);
        phi_theta_curr.reinit(cell);
        phi_theta_curr.gather_evaluate(src[1], EvaluationFlags::values | EvaluationFlags::gradients);

        phi.reinit(cell);

        /*--- Loop over all quadrature points ---*/
        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          const auto& theta_curr = phi_theta_curr.get_value(q);

          /*--- Compute contribution at initialization ---*/
          const auto& grad_u_curr              = phi_u_curr.get_gradient(q);
          const auto& grad_theta_curr          = phi_theta_curr.get_gradient(q);
          const auto& mod_squared_grad_uz_curr = grad_u_curr[0][dim - 1]*grad_u_curr[0][dim - 1];
          const auto& Ri_curr                  = 1.0/(EquationData::Fr2)*grad_theta_curr[dim - 1]/mod_squared_grad_uz_curr;
          VectorizedArray<Number> b;
          VectorizedArray<Number> beta;
          for(unsigned int idx = 0; idx < VectorizedArray<Number>::size(); ++idx) {
            if(Ri_curr[idx] > 0.0) {
              b[idx]    = 5.0;
              beta[idx] = -2.0;
            }
            else {
              b[idx]    = 20.0;
              beta[idx] = 0.5;
            }
          }
          const auto& kappa_curr = EquationData::l_mixing*EquationData::l_mixing*std::sqrt(mod_squared_grad_uz_curr)*
                                   std::pow(1.0 + b*std::abs(Ri_curr), beta);
          Tensor<2, dim, VectorizedArray<Number>> diff_tensor_curr;
          diff_tensor_curr = 0;
          diff_tensor_curr[dim - 1][dim - 1] = kappa_curr;

          phi.submit_value(theta_curr, q);
          phi.submit_gradient(-a21_tilde*dt*diff_tensor_curr*grad_theta_curr, q);
        }

        phi.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
      }
    }
    else {
      /*--- We start by delaring suitable instances to read the available quantities ---*/
      FEEvaluation<dim, fe_degree_T, n_q_points_1d, 1, Number>   phi(data, 1),
                                                                 phi_theta_curr(data, 1),
                                                                 phi_theta_tmp_2(data, 1);
      FEEvaluation<dim, fe_degree_u, n_q_points_1d, dim, Number> phi_u_curr(data, 0),
                                                                 phi_u_tmp_2(data, 0);

      /*--- Loop over all cells ---*/
      for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
        phi_u_curr.reinit(cell);
        phi_u_curr.gather_evaluate(src[0], EvaluationFlags::gradients);
        phi_theta_curr.reinit(cell);
        phi_theta_curr.gather_evaluate(src[1], EvaluationFlags::values | EvaluationFlags::gradients);

        phi_u_tmp_2.reinit(cell);
        phi_u_tmp_2.gather_evaluate(src[2], EvaluationFlags::gradients);
        phi_theta_tmp_2.reinit(cell);
        phi_theta_tmp_2.gather_evaluate(src[3], EvaluationFlags::gradients);

        phi.reinit(cell);

        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          const auto& theta_curr = phi_theta_curr.get_value(q);

          /*--- Compute contribution at initialization ---*/
          const auto& grad_u_curr              = phi_u_curr.get_gradient(q);
          const auto& grad_theta_curr          = phi_theta_curr.get_gradient(q);
          const auto& mod_squared_grad_uz_curr = grad_u_curr[0][dim - 1]*grad_u_curr[0][dim - 1];
          const auto& Ri_curr                  = 1.0/(EquationData::Fr2)*grad_theta_curr[dim - 1]/mod_squared_grad_uz_curr;
          VectorizedArray<Number> b;
          VectorizedArray<Number> beta;
          for(unsigned int idx = 0; idx < VectorizedArray<Number>::size(); ++idx) {
            if(Ri_curr[idx] > 0.0) {
              b[idx]    = 5.0;
              beta[idx] = -2.0;
            }
            else {
              b[idx]    = 20.0;
              beta[idx] = 0.5;
            }
          }
          const auto& kappa_curr = EquationData::l_mixing*EquationData::l_mixing*std::sqrt(mod_squared_grad_uz_curr)*
                                   std::pow(1.0 + b*std::abs(Ri_curr), beta);
          Tensor<2, dim, VectorizedArray<Number>> diff_tensor_curr;
          diff_tensor_curr = 0;
          diff_tensor_curr[dim - 1][dim - 1] = kappa_curr;

          /*--- Compute contribution at previous stage ---*/
          const auto& grad_u_tmp_2              = phi_u_tmp_2.get_gradient(q);
          const auto& grad_theta_tmp_2          = phi_theta_tmp_2.get_gradient(q);
          const auto& mod_squared_grad_uz_tmp_2 = grad_u_tmp_2[0][dim - 1]*grad_u_tmp_2[0][dim - 1];
          const auto& Ri_tmp_2                  = 1.0/(EquationData::Fr2)*grad_theta_tmp_2[dim - 1]/mod_squared_grad_uz_tmp_2;
          for(unsigned int idx = 0; idx < VectorizedArray<Number>::size(); ++idx) {
            if(Ri_tmp_2[idx] > 0.0) {
              b[idx]    = 5.0;
              beta[idx] = -2.0;
            }
            else {
              b[idx]    = 20.0;
              beta[idx] = 0.5;
            }
          }
          const auto& kappa_tmp_2 = EquationData::l_mixing*EquationData::l_mixing*std::sqrt(mod_squared_grad_uz_tmp_2)*
                                    std::pow(1.0 + b*std::abs(Ri_tmp_2), beta);
          Tensor<2, dim, VectorizedArray<Number>> diff_tensor_tmp_2;
          diff_tensor_tmp_2 = 0;
          diff_tensor_tmp_2[dim - 1][dim - 1] = kappa_tmp_2;

          phi.submit_value(theta_curr, q);
          phi.submit_gradient(-a31_tilde*dt*diff_tensor_curr*grad_theta_curr
                              -a32_tilde*dt*diff_tensor_tmp_2*grad_theta_tmp_2, q);
        }

        phi.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
      }
    }
  }

  // Assemble rhs face term for the temperature equation
  //
  template<int dim, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d, int n_q_points_1d_boundary, typename Vec>
  void TurbulentOperator<dim, fe_degree_T, fe_degree_u,
                              n_q_points_1d, n_q_points_1d_boundary, Vec>::
  assemble_rhs_face_term_temperature(const MatrixFree<dim, Number>&               data,
                                     Vec&                                         dst,
                                     const std::vector<Vec>&                      src,
                                     const std::pair<unsigned int, unsigned int>& face_range) const {
    if(VISCOUS_stage == 2) {
      /*--- We start by declaring suitable instances to read the available quantities ---*/
      FEFaceEvaluation<dim, fe_degree_T, n_q_points_1d, 1, Number>   phi_p(data, true, 1),
                                                                     phi_m(data, false, 1),
                                                                     phi_theta_curr_p(data, true, 1),
                                                                     phi_theta_curr_m(data, false, 1);
      FEFaceEvaluation<dim, fe_degree_u, n_q_points_1d, dim, Number> phi_u_curr_p(data, true, 0),
                                                                     phi_u_curr_m(data, false, 0);

      /*--- Loop over all internal faces ---*/
      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        phi_u_curr_p.reinit(face);
        phi_u_curr_p.gather_evaluate(src[0], EvaluationFlags::gradients);
        phi_u_curr_m.reinit(face);
        phi_u_curr_m.gather_evaluate(src[0], EvaluationFlags::gradients);
        phi_theta_curr_p.reinit(face);
        phi_theta_curr_p.gather_evaluate(src[1], EvaluationFlags::values | EvaluationFlags::gradients);
        phi_theta_curr_m.reinit(face);
        phi_theta_curr_m.gather_evaluate(src[1], EvaluationFlags::values | EvaluationFlags::gradients);

        phi_p.reinit(face);
        phi_m.reinit(face);

        const auto coef_jump = C_T*0.5*(std::abs((phi_p.get_normal_vector(0) * phi_p.inverse_jacobian(0))[dim - 1]) +
                                        std::abs((phi_m.get_normal_vector(0) * phi_m.inverse_jacobian(0))[dim - 1])); /*--- Jump constant for IP ---*/

        /*--- Loop over all quadrature points. ---*/
        for(unsigned int q = 0; q < phi_p.n_q_points; ++q) {
          const auto& n_plus = phi_p.get_normal_vector(q);

          /*--- Compute contribution at initialization ---*/
          const auto& grad_u_curr_p              = phi_u_curr_p.get_gradient(q);
          const auto& grad_u_curr_m              = phi_u_curr_m.get_gradient(q);
          const auto& grad_theta_curr_p          = phi_theta_curr_p.get_gradient(q);
          const auto& grad_theta_curr_m          = phi_theta_curr_m.get_gradient(q);
          const auto& mod_squared_grad_uz_curr_p = grad_u_curr_p[0][dim - 1]*grad_u_curr_p[0][dim - 1];
          const auto& mod_squared_grad_uz_curr_m = grad_u_curr_m[0][dim - 1]*grad_u_curr_m[0][dim - 1];
          const auto& Ri_curr_p                  = 1.0/(EquationData::Fr2)*grad_theta_curr_p[dim - 1]/mod_squared_grad_uz_curr_p;
          const auto& Ri_curr_m                  = 1.0/(EquationData::Fr2)*grad_theta_curr_m[dim - 1]/mod_squared_grad_uz_curr_m;
          VectorizedArray<Number> b_p, b_m;
          VectorizedArray<Number> beta_p, beta_m;
          for(unsigned int idx = 0; idx < VectorizedArray<Number>::size(); ++idx) {
            if(Ri_curr_p[idx] > 0.0) {
              b_p[idx]    = 5.0;
              beta_p[idx] = -2.0;
            }
            else {
              b_p[idx]    = 20.0;
              beta_p[idx] = 0.5;
            }

            if(Ri_curr_m[idx] > 0.0) {
              b_m[idx]    = 5.0;
              beta_m[idx] = -2.0;
            }
            else {
              b_m[idx]    = 20.0;
              beta_m[idx] = 0.5;
            }
          }
          const auto& kappa_curr_p = EquationData::l_mixing*EquationData::l_mixing*std::sqrt(mod_squared_grad_uz_curr_p)*
                                     std::pow(1.0 + b_p*std::abs(Ri_curr_p), beta_p);
          const auto& kappa_curr_m = EquationData::l_mixing*EquationData::l_mixing*std::sqrt(mod_squared_grad_uz_curr_m)*
                                     std::pow(1.0 + b_m*std::abs(Ri_curr_m), beta_m);
          Tensor<2, dim, VectorizedArray<Number>> diff_tensor_curr_p,
                                                  diff_tensor_curr_m;
          diff_tensor_curr_p = 0;
          diff_tensor_curr_p[dim - 1][dim - 1] = kappa_curr_p;
          diff_tensor_curr_m = 0;
          diff_tensor_curr_m[dim - 1][dim - 1] = kappa_curr_m;
          const auto& avg_diff_flux_curr = 0.5*(diff_tensor_curr_p*grad_theta_curr_p +
                                                diff_tensor_curr_m*grad_theta_curr_m);

          /*--- Consider also jump penalization contribution ---*/
          const auto& theta_curr_p    = phi_theta_curr_p.get_value(q);
          const auto& theta_curr_m    = phi_theta_curr_m.get_value(q);
          const auto& avg_kappa_curr  = 2.0/(1.0/kappa_curr_p + 1.0/kappa_curr_m);
          const auto& jump_theta_curr = theta_curr_p - theta_curr_m;

          phi_p.submit_value(a21_tilde*dt*(scalar_product(avg_diff_flux_curr, n_plus) -
                                           coef_jump*avg_kappa_curr*jump_theta_curr), q);
          phi_m.submit_value(-a21_tilde*dt*(scalar_product(avg_diff_flux_curr, n_plus) -
                                            coef_jump*avg_kappa_curr*jump_theta_curr), q);
        }

        phi_p.integrate_scatter(EvaluationFlags::values, dst);
        phi_m.integrate_scatter(EvaluationFlags::values, dst);
      }
    }
    else {
      /*--- We start by declaring suitable instances to read the available quantities ---*/
      FEFaceEvaluation<dim, fe_degree_T, n_q_points_1d, 1, Number>   phi_p(data, true, 1),
                                                                     phi_m(data, false, 1),
                                                                     phi_theta_curr_p(data, true, 1),
                                                                     phi_theta_curr_m(data, false, 1),
                                                                     phi_theta_tmp_2_p(data, true, 1),
                                                                     phi_theta_tmp_2_m(data, false, 1);
      FEFaceEvaluation<dim, fe_degree_u, n_q_points_1d, dim, Number> phi_u_curr_p(data, true, 0),
                                                                     phi_u_curr_m(data, false, 0),
                                                                     phi_u_tmp_2_p(data, true, 0),
                                                                     phi_u_tmp_2_m(data, false, 0);

      /*--- Loop over all internal faces ---*/
      for(unsigned int face = face_range.first; face < face_range.second; ++face) {
        phi_u_curr_p.reinit(face);
        phi_u_curr_p.gather_evaluate(src[0], EvaluationFlags::gradients);
        phi_u_curr_m.reinit(face);
        phi_u_curr_m.gather_evaluate(src[0], EvaluationFlags::gradients);
        phi_theta_curr_p.reinit(face);
        phi_theta_curr_p.gather_evaluate(src[1], EvaluationFlags::values | EvaluationFlags::gradients);
        phi_theta_curr_m.reinit(face);
        phi_theta_curr_m.gather_evaluate(src[1], EvaluationFlags::values | EvaluationFlags::gradients);

        phi_u_tmp_2_p.reinit(face);
        phi_u_tmp_2_p.gather_evaluate(src[2], EvaluationFlags::gradients);
        phi_u_tmp_2_m.reinit(face);
        phi_u_tmp_2_m.gather_evaluate(src[2], EvaluationFlags::gradients);
        phi_theta_tmp_2_p.reinit(face);
        phi_theta_tmp_2_p.gather_evaluate(src[3], EvaluationFlags::values | EvaluationFlags::gradients);
        phi_theta_tmp_2_m.reinit(face);
        phi_theta_tmp_2_m.gather_evaluate(src[3], EvaluationFlags::values | EvaluationFlags::gradients);

        phi_p.reinit(face);
        phi_m.reinit(face);

        const auto coef_jump = C_T*0.5*(std::abs((phi_p.get_normal_vector(0) * phi_p.inverse_jacobian(0))[dim - 1]) +
                                        std::abs((phi_m.get_normal_vector(0) * phi_m.inverse_jacobian(0))[dim - 1]));

        /*--- Loop over all quadrature points ---*/
        for(unsigned int q = 0; q < phi_p.n_q_points; ++q) {
          const auto& n_plus = phi_p.get_normal_vector(q);

          /*--- Compute contribution at initialization ---*/
          const auto& grad_u_curr_p              = phi_u_curr_p.get_gradient(q);
          const auto& grad_u_curr_m              = phi_u_curr_m.get_gradient(q);
          const auto& grad_theta_curr_p          = phi_theta_curr_p.get_gradient(q);
          const auto& grad_theta_curr_m          = phi_theta_curr_m.get_gradient(q);
          const auto& mod_squared_grad_uz_curr_p = grad_u_curr_p[0][dim - 1]*grad_u_curr_p[0][dim - 1];
          const auto& mod_squared_grad_uz_curr_m = grad_u_curr_m[0][dim - 1]*grad_u_curr_m[0][dim - 1];
          const auto& Ri_curr_p                  = 1.0/(EquationData::Fr2)*grad_theta_curr_p[dim - 1]/mod_squared_grad_uz_curr_p;
          const auto& Ri_curr_m                  = 1.0/(EquationData::Fr2)*grad_theta_curr_m[dim - 1]/mod_squared_grad_uz_curr_m;
          VectorizedArray<Number> b_p, b_m;
          VectorizedArray<Number> beta_p, beta_m;
          for(unsigned int idx = 0; idx < VectorizedArray<Number>::size(); ++idx) {
            if(Ri_curr_p[idx] > 0.0) {
              b_p[idx]    = 5.0;
              beta_p[idx] = -2.0;
            }
            else {
              b_p[idx]    = 20.0;
              beta_p[idx] = 0.5;
            }

            if(Ri_curr_m[idx] > 0.0) {
              b_m[idx]    = 5.0;
              beta_m[idx] = -2.0;
            }
            else {
              b_m[idx]    = 20.0;
              beta_m[idx] = 0.5;
            }
          }
          const auto& kappa_curr_p = EquationData::l_mixing*EquationData::l_mixing*std::sqrt(mod_squared_grad_uz_curr_p)*
                                     std::pow(1.0 + b_p*std::abs(Ri_curr_p), beta_p);
          const auto& kappa_curr_m = EquationData::l_mixing*EquationData::l_mixing*std::sqrt(mod_squared_grad_uz_curr_m)*
                                     std::pow(1.0 + b_m*std::abs(Ri_curr_m), beta_m);
          Tensor<2, dim, VectorizedArray<Number>> diff_tensor_curr_p,
                                                  diff_tensor_curr_m;
          diff_tensor_curr_p = 0;
          diff_tensor_curr_p[dim - 1][dim - 1] = kappa_curr_p;
          diff_tensor_curr_m = 0;
          diff_tensor_curr_m[dim - 1][dim - 1] = kappa_curr_m;
          const auto& avg_diff_flux_curr = 0.5*(diff_tensor_curr_p*grad_theta_curr_p +
                                                diff_tensor_curr_m*grad_theta_curr_m);

          /*--- Compute contribution at previous stage ---*/
          const auto& grad_u_tmp_2_p              = phi_u_tmp_2_p.get_gradient(q);
          const auto& grad_u_tmp_2_m              = phi_u_tmp_2_m.get_gradient(q);
          const auto& grad_theta_tmp_2_p          = phi_theta_tmp_2_p.get_gradient(q);
          const auto& grad_theta_tmp_2_m          = phi_theta_tmp_2_m.get_gradient(q);
          const auto& mod_squared_grad_uz_tmp_2_p = grad_u_tmp_2_p[0][dim - 1]*grad_u_tmp_2_p[0][dim - 1];
          const auto& mod_squared_grad_uz_tmp_2_m = grad_u_tmp_2_m[0][dim - 1]*grad_u_tmp_2_m[0][dim - 1];
          const auto& Ri_tmp_2_p                  = 1.0/(EquationData::Fr2)*grad_theta_tmp_2_p[dim - 1]/mod_squared_grad_uz_tmp_2_p;
          const auto& Ri_tmp_2_m                  = 1.0/(EquationData::Fr2)*grad_theta_tmp_2_m[dim - 1]/mod_squared_grad_uz_tmp_2_m;
          for(unsigned int idx = 0; idx < VectorizedArray<Number>::size(); ++idx) {
            if(Ri_tmp_2_p[idx] > 0.0) {
              b_p[idx]    = 5.0;
              beta_p[idx] = -2.0;
            }
            else {
              b_p[idx]    = 20.0;
              beta_p[idx] = 0.5;
            }

            if(Ri_tmp_2_m[idx] > 0.0) {
              b_m[idx]    = 5.0;
              beta_m[idx] = -2.0;
            }
            else {
              b_m[idx]    = 20.0;
              beta_m[idx] = 0.5;
            }
          }
          const auto& kappa_tmp_2_p = EquationData::l_mixing*EquationData::l_mixing*std::sqrt(mod_squared_grad_uz_tmp_2_p)*
                                      std::pow(1.0 + b_p*std::abs(Ri_tmp_2_p), beta_p);
          const auto& kappa_tmp_2_m = EquationData::l_mixing*EquationData::l_mixing*std::sqrt(mod_squared_grad_uz_tmp_2_m)*
                                      std::pow(1.0 + b_m*std::abs(Ri_tmp_2_m), beta_m);
          Tensor<2, dim, VectorizedArray<Number>> diff_tensor_tmp_2_p,
                                                  diff_tensor_tmp_2_m;
          diff_tensor_tmp_2_p = 0;
          diff_tensor_tmp_2_p[dim - 1][dim - 1] = kappa_tmp_2_p;
          diff_tensor_tmp_2_m = 0;
          diff_tensor_tmp_2_m[dim - 1][dim - 1] = kappa_tmp_2_m;
          const auto& avg_diff_flux_tmp_2 = 0.5*(diff_tensor_tmp_2_p*grad_theta_tmp_2_p +
                                                 diff_tensor_tmp_2_m*grad_theta_tmp_2_m);

          /*--- Consider also jump penalization contribution ---*/
          const auto& theta_curr_p     = phi_theta_curr_p.get_value(q);
          const auto& theta_curr_m     = phi_theta_curr_m.get_value(q);
          const auto& avg_kappa_curr   = 2.0/(1.0/kappa_curr_p + 1.0/kappa_curr_m);
          const auto& jump_theta_curr  = theta_curr_p - theta_curr_m;

          const auto& theta_tmp_2_p    = phi_theta_tmp_2_p.get_value(q);
          const auto& theta_tmp_2_m    = phi_theta_tmp_2_m.get_value(q);
          const auto& avg_kappa_tmp_2  = 2.0/(1.0/kappa_tmp_2_p + 1.0/kappa_tmp_2_m);
          const auto& jump_theta_tmp_2 = theta_tmp_2_p - theta_tmp_2_m;

          phi_p.submit_value(a31_tilde*dt*(scalar_product(avg_diff_flux_curr, n_plus) -
                                           coef_jump*avg_kappa_curr*jump_theta_curr) +
                             a32_tilde*dt*(scalar_product(avg_diff_flux_tmp_2, n_plus) -
                                           coef_jump*avg_kappa_tmp_2*jump_theta_tmp_2), q);
          phi_m.submit_value(-a31_tilde*dt*(scalar_product(avg_diff_flux_curr, n_plus) -
                                            coef_jump*avg_kappa_curr*jump_theta_curr)
                             -a32_tilde*dt*(scalar_product(avg_diff_flux_tmp_2, n_plus) -
                                            coef_jump*avg_kappa_tmp_2*jump_theta_tmp_2), q);
        }

        phi_p.integrate_scatter(EvaluationFlags::values, dst);
        phi_m.integrate_scatter(EvaluationFlags::values, dst);
      }
    }
  }

  // Put together all the previous steps for the temperature equation
  //
  template<int dim, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d, int n_q_points_1d_boundary, typename Vec>
  void TurbulentOperator<dim, fe_degree_T, fe_degree_u,
                              n_q_points_1d, n_q_points_1d_boundary, Vec>::
  vmult_rhs_temperature(Vec& dst, const std::vector<Vec>& src) const {
    for(unsigned int d = 0; d < src.size(); ++d) {
      src[d].update_ghost_values();
    }

    this->data->loop(&TurbulentOperator::assemble_rhs_cell_term_temperature,
                     &TurbulentOperator::assemble_rhs_face_term_temperature,
                     &TurbulentOperator::assemble_rhs_boundary_term_temperature,
                     this, dst, src, true,
                     MatrixFree<dim, Number>::DataAccessOnFaces::unspecified,
                     MatrixFree<dim, Number>::DataAccessOnFaces::unspecified);
  }

  // Assemble cell term for the temperature equation
  //
  template<int dim, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d, int n_q_points_1d_boundary, typename Vec>
  void TurbulentOperator<dim, fe_degree_T, fe_degree_u,
                              n_q_points_1d, n_q_points_1d_boundary, Vec>::
  assemble_cell_term_temperature(const MatrixFree<dim, Number>&               data,
                                 Vec&                                         dst,
                                 const Vec&                                   src,
                                 const std::pair<unsigned int, unsigned int>& cell_range) const {
    /*--- We start by declaring suitable instances to read the available quantities ---*/
    FEEvaluation<dim, fe_degree_T, n_q_points_1d, 1, Number>   phi(data, 1),
                                                               phi_theta_curr(data, 1);
    FEEvaluation<dim, fe_degree_u, n_q_points_1d, dim, Number> phi_u_curr(data, 0);

    const double coeff = (VISCOUS_stage == 2) ? a22_tilde*dt : a33_tilde*dt;

    /*--- Loop over all cells. ---*/
    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
      phi_u_curr.reinit(cell);
      phi_u_curr.gather_evaluate(u_curr, EvaluationFlags::gradients);
      phi_theta_curr.reinit(cell);
      phi_theta_curr.gather_evaluate(theta_curr, EvaluationFlags::gradients);

      phi.reinit(cell);
      phi.gather_evaluate(src, EvaluationFlags::values | EvaluationFlags::gradients);

      /*--- Loop over all quadrature points. ---*/
      for(unsigned int q = 0; q < phi.n_q_points; ++q) {
        /*--- Compute contribution at current stage ---*/
        const auto& grad_u_curr              = phi_u_curr.get_gradient(q);
        const auto& grad_theta_curr          = phi_theta_curr.get_gradient(q);
        const auto& mod_squared_grad_uz_curr = grad_u_curr[0][dim - 1]*grad_u_curr[0][dim - 1];
        const auto& Ri_curr                  = 1.0/(EquationData::Fr2)*grad_theta_curr[dim - 1]/mod_squared_grad_uz_curr;
        VectorizedArray<Number> b;
        VectorizedArray<Number> beta;
        for(unsigned int idx = 0; idx < VectorizedArray<Number>::size(); ++idx) {
          if(Ri_curr[idx] > 0.0) {
            b[idx]    = 5.0;
            beta[idx] = -2.0;
          }
          else {
            b[idx]    = 20.0;
            beta[idx] = 0.5;
          }
        }
        const auto& kappa_curr = EquationData::l_mixing*EquationData::l_mixing*std::sqrt(mod_squared_grad_uz_curr)*
                                 std::pow(1.0 + b*std::abs(Ri_curr), beta);
        Tensor<2, dim, VectorizedArray<Number>> diff_tensor_curr;
        diff_tensor_curr = 0;
        diff_tensor_curr[dim - 1][dim - 1] = kappa_curr;

        phi.submit_value(phi.get_value(q), q);
        phi.submit_gradient(coeff*diff_tensor_curr*phi.get_gradient(q), q);
      }

      phi.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
    }
  }

  // Assemble face term for the temperature equation
  //
  template<int dim, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d, int n_q_points_1d_boundary, typename Vec>
  void TurbulentOperator<dim, fe_degree_T, fe_degree_u,
                              n_q_points_1d, n_q_points_1d_boundary, Vec>::
  assemble_face_term_temperature(const MatrixFree<dim, Number>&               data,
                                 Vec&                                         dst,
                                 const Vec&                                   src,
                                 const std::pair<unsigned int, unsigned int>& face_range) const {
    /*--- We start by declaring suitable instances to read the available quantities ---*/
    FEFaceEvaluation<dim, fe_degree_T, n_q_points_1d, 1, Number>   phi_p(data, true, 1),
                                                                   phi_m(data, false, 1),
                                                                   phi_theta_curr_p(data, true, 1),
                                                                   phi_theta_curr_m(data, false, 1);
    FEFaceEvaluation<dim, fe_degree_u, n_q_points_1d, dim, Number> phi_u_curr_p(data, true, 0),
                                                                   phi_u_curr_m(data, false, 0);

    const double coeff = (VISCOUS_stage == 2) ? a22_tilde*dt : a33_tilde*dt;

    /*--- Loop over all internal faces ---*/
    for(unsigned int face = face_range.first; face < face_range.second; ++face) {
      phi_u_curr_p.reinit(face);
      phi_u_curr_p.gather_evaluate(u_curr, EvaluationFlags::gradients);
      phi_u_curr_m.reinit(face);
      phi_u_curr_m.gather_evaluate(u_curr, EvaluationFlags::gradients);
      phi_theta_curr_p.reinit(face);
      phi_theta_curr_p.gather_evaluate(theta_curr, EvaluationFlags::gradients);
      phi_theta_curr_m.reinit(face);
      phi_theta_curr_m.gather_evaluate(theta_curr, EvaluationFlags::gradients);

      phi_p.reinit(face);
      phi_p.gather_evaluate(src, EvaluationFlags::values | EvaluationFlags::gradients);
      phi_m.reinit(face);
      phi_m.gather_evaluate(src, EvaluationFlags::values | EvaluationFlags::gradients);

      const auto coef_jump = C_T*0.5*(std::abs((phi_p.get_normal_vector(0) * phi_p.inverse_jacobian(0))[dim - 1]) +
                                      std::abs((phi_m.get_normal_vector(0) * phi_m.inverse_jacobian(0))[dim - 1])); /*--- Jump cosntant for IP ---*/

      /*--- Loop over all quadrature points ---*/
      for(unsigned int q = 0; q < phi_p.n_q_points; ++q) {
        const auto& n_plus = phi_p.get_normal_vector(q);

        /*--- Compute contribution at current stage ---*/
        const auto& grad_u_curr_p              = phi_u_curr_p.get_gradient(q);
        const auto& grad_u_curr_m              = phi_u_curr_m.get_gradient(q);
        const auto& grad_theta_curr_p          = phi_theta_curr_p.get_gradient(q);
        const auto& grad_theta_curr_m          = phi_theta_curr_m.get_gradient(q);
        const auto& mod_squared_grad_uz_curr_p = grad_u_curr_p[0][dim - 1]*grad_u_curr_p[0][dim - 1];
        const auto& mod_squared_grad_uz_curr_m = grad_u_curr_m[0][dim - 1]*grad_u_curr_m[0][dim - 1];
        const auto& Ri_curr_p                  = 1.0/(EquationData::Fr2)*grad_theta_curr_p[dim - 1]/mod_squared_grad_uz_curr_p;
        const auto& Ri_curr_m                  = 1.0/(EquationData::Fr2)*grad_theta_curr_m[dim - 1]/mod_squared_grad_uz_curr_m;
        VectorizedArray<Number> b_p, b_m;
        VectorizedArray<Number> beta_p, beta_m;
        for(unsigned int idx = 0; idx < VectorizedArray<Number>::size(); ++idx) {
          if(Ri_curr_p[idx] > 0.0) {
            b_p[idx]    = 5.0;
            beta_p[idx] = -2.0;
          }
          else {
            b_p[idx]    = 20.0;
            beta_p[idx] = 0.5;
          }

          if(Ri_curr_m[idx] > 0.0) {
            b_m[idx]    = 5.0;
            beta_m[idx] = -2.0;
          }
          else {
            b_m[idx]    = 20.0;
            beta_m[idx] = 0.5;
          }
        }
        const auto& kappa_curr_p = EquationData::l_mixing*EquationData::l_mixing*std::sqrt(mod_squared_grad_uz_curr_p)*
                                   std::pow(1.0 + b_p*std::abs(Ri_curr_p), beta_p);
        const auto& kappa_curr_m = EquationData::l_mixing*EquationData::l_mixing*std::sqrt(mod_squared_grad_uz_curr_m)*
                                   std::pow(1.0 + b_m*std::abs(Ri_curr_m), beta_m);
        Tensor<2, dim, VectorizedArray<Number>> diff_tensor_curr_p,
                                                diff_tensor_curr_m;
        diff_tensor_curr_p = 0;
        diff_tensor_curr_p[dim - 1][dim - 1] = kappa_curr_p;
        diff_tensor_curr_m = 0;
        diff_tensor_curr_m[dim - 1][dim - 1] = kappa_curr_m;
        const auto& avg_diff_flux = 0.5*(diff_tensor_curr_p*phi_p.get_gradient(q) +
                                         diff_tensor_curr_m*phi_m.get_gradient(q));

        /*--- Consider also IP term ---*/
        const auto& theta_p        = phi_p.get_value(q);
        const auto& theta_m        = phi_m.get_value(q);
        const auto& avg_kappa_curr = 2.0/(1.0/kappa_curr_p + 1.0/kappa_curr_m);
        const auto& jump_theta     = theta_p - theta_m;

        phi_p.submit_value(coeff*(-scalar_product(avg_diff_flux, n_plus) + coef_jump*avg_kappa_curr*jump_theta), q);
        phi_m.submit_value(-coeff*(-scalar_product(avg_diff_flux, n_plus) + coef_jump*avg_kappa_curr*jump_theta), q);
      }

      phi_p.integrate_scatter(EvaluationFlags::values, dst);
      phi_m.integrate_scatter(EvaluationFlags::values, dst);
    }
  }


  // Put together all previous steps
  //
  template<int dim, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d, int n_q_points_1d_boundary, typename Vec>
  void TurbulentOperator<dim, fe_degree_T, fe_degree_u,
                                 n_q_points_1d, n_q_points_1d_boundary, Vec>::
  apply_add(Vec& dst, const Vec& src) const {
    AssertIndexRange(NS_stage, 3);
    Assert(NS_stage > 0, ExcInternalError());

    if(NS_stage == 1) {
      this->data->loop(&TurbulentOperator::assemble_cell_term_velocity,
                       &TurbulentOperator::assemble_face_term_velocity,
                       &TurbulentOperator::assemble_boundary_term_velocity,
                       this, dst, src, false,
                       MatrixFree<dim, Number>::DataAccessOnFaces::unspecified,
                       MatrixFree<dim, Number>::DataAccessOnFaces::unspecified);
    }
    else if(NS_stage == 2) {
      this->data->loop(&TurbulentOperator::assemble_cell_term_temperature,
                       &TurbulentOperator::assemble_face_term_temperature,
                       &TurbulentOperator::assemble_boundary_term_temperature,
                       this, dst, src, false);
    }
    else {
      Assert(false, ExcInternalError());
    }
  }


  // Assemble diagonal cell term for the velocity equation
  //
  template<int dim, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d, int n_q_points_1d_boundary, typename Vec>
  void TurbulentOperator<dim, fe_degree_T, fe_degree_u,
                              n_q_points_1d, n_q_points_1d_boundary, Vec>::
  assemble_diagonal_cell_term_velocity(const MatrixFree<dim, Number>&               data,
                                       Vec&                                         dst,
                                       const unsigned int&                          ,
                                       const std::pair<unsigned int, unsigned int>& cell_range) const {
    /*--- We start by declaring suitable instances to read the available quantities ---*/
    FEEvaluation<dim, fe_degree_u, n_q_points_1d, dim, Number> phi(data, 0),
                                                               phi_u_curr(data, 0);
    FEEvaluation<dim, fe_degree_T, n_q_points_1d, 1, Number>   phi_theta_curr(data, 1);

    /*--- We are in a matrix-free framework. Hence, in order to compute the diagonal, we need to test the operator against
          a vector which is 1 for the node of interest and 0 elsewhere. This is what 'tmp' does. ---*/
    AlignedVector<Tensor<1, dim, VectorizedArray<Number>>> diagonal(phi.dofs_per_component);
    Tensor<1, dim, VectorizedArray<Number>> tmp;
    for(unsigned int d = 0; d < dim; ++d) {
      tmp[d] = make_vectorized_array<Number>(1.0);
    }

    const double coeff = (VISCOUS_stage == 2) ? a22_tilde*dt : a33_tilde*dt;

    /*--- Loop over all cells ---*/
    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
      phi_u_curr.reinit(cell);
      phi_u_curr.gather_evaluate(u_curr, EvaluationFlags::gradients);
      phi_theta_curr.reinit(cell);
      phi_theta_curr.gather_evaluate(theta_curr, EvaluationFlags::gradients);

      phi.reinit(cell);

      /*--- Loop over all dofs ---*/
      for(unsigned int i = 0; i < phi.dofs_per_component; ++i) {
        for(unsigned int j = 0; j < phi.dofs_per_component; ++j) {
          phi.submit_dof_value(Tensor<1, dim, VectorizedArray<Number>>(), j);
        }
        phi.submit_dof_value(tmp, i);
        phi.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);

        /*--- Loop over all quadrature points. ---*/
        for(unsigned int q = 0; q < phi.n_q_points; ++q) {
          const auto& u = phi.get_value(q);

          /*--- Compute contribution at current stage ---*/
          const auto& grad_u_curr              = phi_u_curr.get_gradient(q);
          const auto& grad_theta_curr          = phi_theta_curr.get_gradient(q);
          const auto& mod_squared_grad_uz_curr = grad_u_curr[0][dim - 1]*grad_u_curr[0][dim - 1];
          const auto& Ri_curr                  = 1.0/(EquationData::Fr2)*grad_theta_curr[dim - 1]/mod_squared_grad_uz_curr;
          VectorizedArray<Number> b;
          VectorizedArray<Number> beta;
          for(unsigned int idx = 0; idx < VectorizedArray<Number>::size(); ++idx) {
            if(Ri_curr[idx] > 0.0) {
              b[idx]    = 5.0;
              beta[idx] = -2.0;
            }
            else {
              b[idx]    = 20.0;
              beta[idx] = 0.5;
            }
          }
          const auto& kappa_curr = EquationData::l_mixing*EquationData::l_mixing*std::sqrt(mod_squared_grad_uz_curr)*
                                   std::pow(1.0 + b*std::abs(Ri_curr), beta);

          Tensor<2, dim, VectorizedArray<Number>> diff_flux;
          diff_flux = 0;
          const auto& grad_u    = phi.get_gradient(q);
          diff_flux[0][dim - 1] = kappa_curr*grad_u[0][dim - 1];

          phi.submit_value(u, q);
          phi.submit_gradient(coeff*diff_flux, q);
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

  // Assemble diagonal face term for the velocity equation
  //
  template<int dim, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d, int n_q_points_1d_boundary, typename Vec>
  void TurbulentOperator<dim, fe_degree_T, fe_degree_u,
                              n_q_points_1d, n_q_points_1d_boundary, Vec>::
  assemble_diagonal_face_term_velocity(const MatrixFree<dim, Number>&               data,
                                       Vec&                                         dst,
                                       const unsigned int&                          ,
                                       const std::pair<unsigned int, unsigned int>& face_range) const {
    /*--- We start by declaring suitable instances to read the available quantities ---*/
    FEFaceEvaluation<dim, fe_degree_u, n_q_points_1d, dim, Number> phi_p(data, true, 0),
                                                                   phi_m(data, false, 0),
                                                                   phi_u_curr_p(data, true, 0),
                                                                   phi_u_curr_m(data, false, 0);
    FEFaceEvaluation<dim, fe_degree_T, n_q_points_1d, 1, Number>   phi_theta_curr_p(data, true, 1),
                                                                   phi_theta_curr_m(data, false, 1);

    AlignedVector<Tensor<1, dim, VectorizedArray<Number>>> diagonal_p(phi_p.dofs_per_component),
                                                           diagonal_m(phi_m.dofs_per_component);
    Tensor<1, dim, VectorizedArray<Number>> tmp;
    for(unsigned int d = 0; d < dim; ++d)
      tmp[d] = make_vectorized_array<Number>(1.0);

    const double coeff = (VISCOUS_stage == 2) ? a22_tilde*dt : a33_tilde*dt;

    /*--- Loop over all internal faces ---*/
    for(unsigned int face = face_range.first; face < face_range.second; ++face) {
      phi_u_curr_p.reinit(face);
      phi_u_curr_p.gather_evaluate(u_curr, EvaluationFlags::gradients);
      phi_u_curr_m.reinit(face);
      phi_u_curr_m.gather_evaluate(u_curr, EvaluationFlags::gradients);
      phi_theta_curr_p.reinit(face);
      phi_theta_curr_p.gather_evaluate(theta_curr, EvaluationFlags::gradients);
      phi_theta_curr_m.reinit(face);
      phi_theta_curr_m.gather_evaluate(theta_curr, EvaluationFlags::gradients);

      phi_p.reinit(face);
      phi_m.reinit(face);

      const auto coef_jump = C_u*0.5*(std::abs((phi_p.get_normal_vector(0) * phi_p.inverse_jacobian(0))[dim - 1]) +
                                      std::abs((phi_m.get_normal_vector(0) * phi_m.inverse_jacobian(0))[dim - 1])); /*--- Jump constant for IP ---*/

      /*--- Loop over all dofs ---*/
      for(unsigned int i = 0; i < phi_p.dofs_per_component; ++i) {
        for(unsigned int j = 0; j < phi_p.dofs_per_component; ++j) {
          phi_p.submit_dof_value(Tensor<1, dim, VectorizedArray<Number>>(), j);
          phi_m.submit_dof_value(Tensor<1, dim, VectorizedArray<Number>>(), j);
        }
        phi_p.submit_dof_value(tmp, i);
        phi_p.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);
        phi_m.submit_dof_value(tmp, i);
        phi_m.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);

        /*--- Loop over all quadrature points ---*/
        for(unsigned int q = 0; q < phi_p.n_q_points; ++q) {
          const auto& n_plus = phi_p.get_normal_vector(q);

          /*--- Compute contribution at current stage ---*/
          const auto& grad_u_curr_p              = phi_u_curr_p.get_gradient(q);
          const auto& grad_u_curr_m              = phi_u_curr_m.get_gradient(q);
          const auto& grad_theta_curr_p          = phi_theta_curr_p.get_gradient(q);
          const auto& grad_theta_curr_m          = phi_theta_curr_m.get_gradient(q);
          const auto& mod_squared_grad_uz_curr_p = grad_u_curr_p[0][dim - 1]*grad_u_curr_p[0][dim - 1];
          const auto& mod_squared_grad_uz_curr_m = grad_u_curr_m[0][dim - 1]*grad_u_curr_m[0][dim - 1];
          const auto& Ri_curr_p                  = 1.0/(EquationData::Fr2)*grad_theta_curr_p[dim - 1]/mod_squared_grad_uz_curr_p;
          const auto& Ri_curr_m                  = 1.0/(EquationData::Fr2)*grad_theta_curr_m[dim - 1]/mod_squared_grad_uz_curr_m;
          VectorizedArray<Number> b_p, b_m;
          VectorizedArray<Number> beta_p, beta_m;
          for(unsigned int idx = 0; idx < VectorizedArray<Number>::size(); ++idx) {
            if(Ri_curr_p[idx] > 0.0) {
              b_p[idx]    = 5.0;
              beta_p[idx] = -2.0;
            }
            else {
              b_p[idx]    = 20.0;
              beta_p[idx] = 0.5;
            }

            if(Ri_curr_m[idx] > 0.0) {
              b_m[idx]    = 5.0;
              beta_m[idx] = -2.0;
            }
            else {
              b_m[idx]    = 20.0;
              beta_m[idx] = 0.5;
            }
          }
          const auto& kappa_curr_p = EquationData::l_mixing*EquationData::l_mixing*std::sqrt(mod_squared_grad_uz_curr_p)*
                                     std::pow(1.0 + b_p*std::abs(Ri_curr_p), beta_p);
          const auto& kappa_curr_m = EquationData::l_mixing*EquationData::l_mixing*std::sqrt(mod_squared_grad_uz_curr_m)*
                                     std::pow(1.0 + b_m*std::abs(Ri_curr_m), beta_m);
          Tensor<2, dim, VectorizedArray<Number>> avg_diff_flux;
          avg_diff_flux = 0;
          const auto& grad_u_p      = phi_p.get_gradient(q);
          const auto& grad_u_m      = phi_m.get_gradient(q);
          avg_diff_flux[0][dim - 1] = 0.5*(kappa_curr_p*grad_u_p[0][dim - 1] +
                                           kappa_curr_m*grad_u_m[0][dim - 1]);

          /*--- Consider also IP term ---*/
          const auto& u_p            = phi_p.get_value(q);
          const auto& u_m            = phi_m.get_value(q);
          const auto& avg_kappa_curr = 2.0/(1.0/kappa_curr_p + 1.0/kappa_curr_m);
          auto jump_u                = u_p - u_m;
          jump_u[dim - 1]            = make_vectorized_array<Number>(0.0);

          phi_p.submit_value(coeff*(-avg_diff_flux*n_plus + coef_jump*avg_kappa_curr*jump_u), q);
          phi_m.submit_value(-coeff*(-avg_diff_flux*n_plus + coef_jump*avg_kappa_curr*jump_u), q);
        }

        phi_p.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
        diagonal_p[i] = phi_p.get_dof_value(i);
        phi_m.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
        diagonal_m[i] = phi_m.get_dof_value(i);
      }

      for(unsigned int i = 0; i < phi_p.dofs_per_component; ++i) {
        phi_p.submit_dof_value(diagonal_p[i], i);
        phi_m.submit_dof_value(diagonal_m[i], i);
      }
      phi_p.distribute_local_to_global(dst);
      phi_m.distribute_local_to_global(dst);
    }
  }


  // Assemble diagonal cell term for the temperature equation
  //
  template<int dim, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d, int n_q_points_1d_boundary, typename Vec>
  void TurbulentOperator<dim, fe_degree_T, fe_degree_u,
                              n_q_points_1d, n_q_points_1d_boundary, Vec>::
  assemble_diagonal_cell_term_temperature(const MatrixFree<dim, Number>&               data,
                                          Vec&                                         dst,
                                          const unsigned int&                          ,
                                          const std::pair<unsigned int, unsigned int>& cell_range) const {
    /*--- We start by decalring suitable instances to read the available quantities ---*/
    FEEvaluation<dim, fe_degree_T, n_q_points_1d, 1, Number>   phi(data, 1),
                                                               phi_theta_curr(data, 1);
    FEEvaluation<dim, fe_degree_u, n_q_points_1d, dim, Number> phi_u_curr(data, 0);

    AlignedVector<VectorizedArray<Number>> diagonal(phi.dofs_per_component);

    const double coeff = (VISCOUS_stage == 2) ? a22_tilde*dt : a33_tilde*dt;

    /*--- Loop over all cells ---*/
    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
      phi_u_curr.reinit(cell);
      phi_u_curr.gather_evaluate(u_curr, EvaluationFlags::gradients);
      phi_theta_curr.reinit(cell);
      phi_theta_curr.gather_evaluate(theta_curr, EvaluationFlags::gradients);

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
          /*--- Compute contribution at current stage ---*/
          const auto& grad_u_curr              = phi_u_curr.get_gradient(q);
          const auto& grad_theta_curr          = phi_theta_curr.get_gradient(q);
          const auto& mod_squared_grad_uz_curr = grad_u_curr[0][dim - 1]*grad_u_curr[0][dim - 1];
          const auto& Ri_curr                  = 1.0/(EquationData::Fr2)*grad_theta_curr[dim - 1]/mod_squared_grad_uz_curr;
          VectorizedArray<Number> b;
          VectorizedArray<Number> beta;
          for(unsigned int idx = 0; idx < VectorizedArray<Number>::size(); ++idx) {
            if(Ri_curr[idx] > 0.0) {
              b[idx]    = 5.0;
              beta[idx] = -2.0;
            }
            else {
              b[idx]    = 20.0;
              beta[idx] = 0.5;
            }
          }
          const auto& kappa_curr = EquationData::l_mixing*EquationData::l_mixing*std::sqrt(mod_squared_grad_uz_curr)*
                                   std::pow(1.0 + b*std::abs(Ri_curr), beta);
          Tensor<2, dim, VectorizedArray<Number>> diff_tensor_curr;
          diff_tensor_curr = 0;
          diff_tensor_curr[dim - 1][dim - 1] = kappa_curr;

          phi.submit_value(phi.get_value(q), q);
          phi.submit_gradient(coeff*diff_tensor_curr*phi.get_gradient(q), q);
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


  // Assemble diagonal face term for the temperature equation
  //
  template<int dim, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d, int n_q_points_1d_boundary, typename Vec>
  void TurbulentOperator<dim, fe_degree_T, fe_degree_u,
                              n_q_points_1d, n_q_points_1d_boundary, Vec>::
  assemble_diagonal_face_term_temperature(const MatrixFree<dim, Number>&               data,
                                          Vec&                                         dst,
                                          const unsigned int&                          ,
                                          const std::pair<unsigned int, unsigned int>& face_range) const {
    /*--- We start by decalring suitable instances to read the available quantities ---*/
    FEFaceEvaluation<dim, fe_degree_T, n_q_points_1d, 1, Number>   phi_p(data, true, 1),
                                                                   phi_m(data, false, 1),
                                                                   phi_theta_curr_p(data, true, 1),
                                                                   phi_theta_curr_m(data, false, 1);
    FEFaceEvaluation<dim, fe_degree_u, n_q_points_1d, dim, Number> phi_u_curr_p(data, true, 0),
                                                                   phi_u_curr_m(data, false, 0);

    AlignedVector<VectorizedArray<Number>> diagonal_p(phi_p.dofs_per_component),
                                           diagonal_m(phi_m.dofs_per_component);

    const double coeff = (VISCOUS_stage == 2) ? a22_tilde*dt : a33_tilde*dt;

    /*--- Loop over all face ---*/
    for(unsigned int face = face_range.first; face < face_range.second; ++face) {
      phi_u_curr_p.reinit(face);
      phi_u_curr_p.gather_evaluate(u_curr, EvaluationFlags::gradients);
      phi_u_curr_m.reinit(face);
      phi_u_curr_m.gather_evaluate(u_curr, EvaluationFlags::gradients);
      phi_theta_curr_p.reinit(face);
      phi_theta_curr_p.gather_evaluate(theta_curr, EvaluationFlags::gradients);
      phi_theta_curr_m.reinit(face);
      phi_theta_curr_m.gather_evaluate(theta_curr, EvaluationFlags::gradients);

      phi_p.reinit(face);
      phi_m.reinit(face);

      const auto coef_jump = C_T*0.5*(std::abs((phi_p.get_normal_vector(0) * phi_p.inverse_jacobian(0))[dim - 1]) +
                                      std::abs((phi_m.get_normal_vector(0) * phi_m.inverse_jacobian(0))[dim - 1])); /*--- Jump constant for IP ---*/

      /*--- Loop over all dofs ---*/
      for(unsigned int i = 0; i < phi_p.dofs_per_component; ++i) {
        for(unsigned int j = 0; j < phi_p.dofs_per_component; ++j) {
          phi_p.submit_dof_value(VectorizedArray<Number>(), j);
          phi_m.submit_dof_value(VectorizedArray<Number>(), j);
        }
        phi_p.submit_dof_value(make_vectorized_array<Number>(1.0), i);
        phi_m.submit_dof_value(make_vectorized_array<Number>(1.0), i);
        phi_p.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);
        phi_m.evaluate(EvaluationFlags::values | EvaluationFlags::gradients);

        /*--- Loop over all quadrature points ---*/
        for(unsigned int q = 0; q < phi_p.n_q_points; ++q) {
          const auto& n_plus = phi_p.get_normal_vector(q);

          /*--- Compute contribution at current stage ---*/
          const auto& grad_u_curr_p              = phi_u_curr_p.get_gradient(q);
          const auto& grad_u_curr_m              = phi_u_curr_m.get_gradient(q);
          const auto& grad_theta_curr_p          = phi_theta_curr_p.get_gradient(q);
          const auto& grad_theta_curr_m          = phi_theta_curr_m.get_gradient(q);
          const auto& mod_squared_grad_uz_curr_p = grad_u_curr_p[0][dim - 1]*grad_u_curr_p[0][dim - 1];
          const auto& mod_squared_grad_uz_curr_m = grad_u_curr_m[0][dim - 1]*grad_u_curr_m[0][dim - 1];
          const auto& Ri_curr_p                  = 1.0/(EquationData::Fr2)*grad_theta_curr_p[dim - 1]/mod_squared_grad_uz_curr_p;
          const auto& Ri_curr_m                  = 1.0/(EquationData::Fr2)*grad_theta_curr_m[dim - 1]/mod_squared_grad_uz_curr_m;
          VectorizedArray<Number> b_p, b_m;
          VectorizedArray<Number> beta_p, beta_m;
          for(unsigned int idx = 0; idx < VectorizedArray<Number>::size(); ++idx) {
            if(Ri_curr_p[idx] > 0.0) {
              b_p[idx]    = 5.0;
              beta_p[idx] = -2.0;
            }
            else {
              b_p[idx]    = 20.0;
              beta_p[idx] = 0.5;
            }

            if(Ri_curr_m[idx] > 0.0) {
              b_m[idx]    = 5.0;
              beta_m[idx] = -2.0;
            }
            else {
              b_m[idx]    = 20.0;
              beta_m[idx] = 0.5;
            }
          }
          const auto& kappa_curr_p = EquationData::l_mixing*EquationData::l_mixing*std::sqrt(mod_squared_grad_uz_curr_p)*
                                     std::pow(1.0 + b_p*std::abs(Ri_curr_p), beta_p);
          const auto& kappa_curr_m = EquationData::l_mixing*EquationData::l_mixing*std::sqrt(mod_squared_grad_uz_curr_m)*
                                     std::pow(1.0 + b_m*std::abs(Ri_curr_m), beta_m);
          Tensor<2, dim, VectorizedArray<Number>> diff_tensor_curr_p,
                                                    diff_tensor_curr_m;
          diff_tensor_curr_p = 0;
          diff_tensor_curr_p[dim - 1][dim - 1] = kappa_curr_p;
          diff_tensor_curr_m = 0;
          diff_tensor_curr_m[dim - 1][dim - 1] = kappa_curr_m;
          const auto& avg_diff_flux = 0.5*(diff_tensor_curr_p*phi_p.get_gradient(q) +
                                           diff_tensor_curr_m*phi_m.get_gradient(q));

          /*--- Consider also IP term ---*/
          const auto& theta_p        = phi_p.get_value(q);
          const auto& theta_m        = phi_m.get_value(q);
          const auto& avg_kappa_curr = 2.0/(1.0/kappa_curr_p + 1.0/kappa_curr_m);
          const auto& jump_theta     = theta_p - theta_m;

          phi_p.submit_value(coeff*(-scalar_product(avg_diff_flux, n_plus) + coef_jump*avg_kappa_curr*jump_theta), q);
          phi_m.submit_value(-coeff*(-scalar_product(avg_diff_flux, n_plus) + coef_jump*avg_kappa_curr*jump_theta), q);
        }

        phi_p.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
        diagonal_p[i] = phi_p.get_dof_value(i);
        phi_m.integrate(EvaluationFlags::values | EvaluationFlags::gradients);
        diagonal_m[i] = phi_m.get_dof_value(i);
      }

      for(unsigned int i = 0; i < phi_p.dofs_per_component; ++i) {
        phi_p.submit_dof_value(diagonal_p[i], i);
        phi_m.submit_dof_value(diagonal_m[i], i);
      }
      phi_p.distribute_local_to_global(dst);
      phi_m.distribute_local_to_global(dst);
    }
  }


  // Compute diagonal of various steps
  //
  template<int dim, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d, int n_q_points_1d_boundary, typename Vec>
  void TurbulentOperator<dim, fe_degree_T, fe_degree_u,
                          n_q_points_1d, n_q_points_1d_boundary, Vec>::
  compute_diagonal() {
    AssertIndexRange(NS_stage, 3);
    Assert(NS_stage > 0, ExcInternalError());

    this->inverse_diagonal_entries.reset(new DiagonalMatrix<Vec>());
    auto& inverse_diagonal = this->inverse_diagonal_entries->get_vector();

    const unsigned int dummy = 0;

    if(NS_stage == 1) {
      this->data->initialize_dof_vector(inverse_diagonal, 0);

      this->data->loop(&TurbulentOperator::assemble_diagonal_cell_term_velocity,
                       &TurbulentOperator::assemble_diagonal_face_term_velocity,
                       &TurbulentOperator::assemble_diagonal_boundary_term_velocity,
                       this, inverse_diagonal, dummy, false,
                       MatrixFree<dim, Number>::DataAccessOnFaces::unspecified,
                       MatrixFree<dim, Number>::DataAccessOnFaces::unspecified);
    }
    else if(NS_stage == 2) {
      this->data->initialize_dof_vector(inverse_diagonal, 1);

      this->data->loop(&TurbulentOperator::assemble_diagonal_cell_term_temperature,
                       &TurbulentOperator::assemble_diagonal_face_term_temperature,
                       &TurbulentOperator::assemble_diagonal_boundary_term_temperature,
                       this, inverse_diagonal, dummy, false,
                       MatrixFree<dim, Number>::DataAccessOnFaces::unspecified,
                       MatrixFree<dim, Number>::DataAccessOnFaces::unspecified);
    }
    else {
      Assert(false, ExcInternalError());
    }

    /*--- For the preconditioner, we actually need the inverse of the diagonal ---*/
    for(unsigned int i = 0; i < inverse_diagonal.local_size(); ++i) {
      Assert(inverse_diagonal.local_element(i) != 0.0,
             ExcMessage("No diagonal entry in a definite operator should be zero"));
      inverse_diagonal.local_element(i) = 1.0/inverse_diagonal.local_element(i);
    }
  }

} // End of namespace
