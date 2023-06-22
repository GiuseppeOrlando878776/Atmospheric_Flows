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

    void set_NS_stage(const unsigned int stage); /*--- Setter of the equation currently under solution. ---*/

    void vmult_rhs_rho_update(Vec& dst, const std::vector<Vec>& src) const; /*--- Auxiliary function to assemble the rhs
                                                                                  for the density. ---*/

    void vmult_rhs_momentum_update(Vec& dst, const std::vector<Vec>& src) const;  /*--- Auxiliary function to assemble the rhs
                                                                                        for the momentum. ---*/

    void vmult_rhs_energy_update(Vec& dst, const std::vector<Vec>& src) const;  /*--- Auxiliary function to assemble the rhs
                                                                                      for the energy. ---*/

    virtual void compute_diagonal() override; /*--- Overriden function to compute the diagonal. ---*/

  protected:
    double       Ma;  /*--- Mach number. ---*/
    double       dt;  /*--- Time step. ---*/

    unsigned int NS_stage; /*--- Flag for the IMEX stage ---*/

    virtual void apply_add(Vec& dst, const Vec& src) const override; /*--- Overriden function which actually assembles the
                                                                           bilinear forms ---*/

  private:
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
                                               const std::pair<unsigned int, unsigned int>& face_range) const {}
                                               /*-- No flux, so no contribution from this function ---*/

    /*--- Assembler function related to the bilinear form of the continuity equation. Only cell contribution is present,
          since, basically, we end up with a mass matrix. ---*/
    void assemble_cell_term_rho_update(const MatrixFree<dim, Number>&               data,
                                       Vec&                                         dst,
                                       const Vec&                                   src,
                                       const std::pair<unsigned int, unsigned int>& cell_range) const;

    /*--- Assembler functions for the rhs related to the momentum equation. ---*/
    void assemble_rhs_cell_term_momentum_update(const MatrixFree<dim, Number>&               data,
                                                Vec&                                         dst,
                                                const std::vector<Vec>&                      src,
                                                const std::pair<unsigned int, unsigned int>& cell_range) const;
    void assemble_rhs_face_term_momentum_update(const MatrixFree<dim, Number>&               data,
                                                Vec&                                         dst,
                                                const std::vector<Vec>&                      src,
                                                const std::pair<unsigned int, unsigned int>& face_range) const;
    void assemble_rhs_boundary_term_momentum_update(const MatrixFree<dim, Number>&               data,
                                                   Vec&                                         dst,
                                                   const std::vector<Vec>&                      src,
                                                   const std::pair<unsigned int, unsigned int>& face_range) const;

    /*--- Assembler function related to the bilinear form of the momentum equation. Only cell contribution is present,
          since, basically, we end up with a mass matrix. ---*/
    void assemble_cell_term_momentum_update(const MatrixFree<dim, Number>&               data,
                                            Vec&                                         dst,
                                            const Vec&                                   src,
                                            const std::pair<unsigned int, unsigned int>& cell_range) const;

    /*--- Assembler functions for the rhs of the the energy equation. ---*/
    void assemble_rhs_cell_term_energy_update(const MatrixFree<dim, Number>&               data,
                                              Vec&                                         dst,
                                              const std::vector<Vec>&                      src,
                                              const std::pair<unsigned int, unsigned int>& cell_range) const;
    void assemble_rhs_face_term_energy_update(const MatrixFree<dim, Number>&               data,
                                              Vec&                                         dst,
                                              const std::vector<Vec>&                      src,
                                              const std::pair<unsigned int, unsigned int>& face_range) const;
    void assemble_rhs_boundary_term_energy_update(const MatrixFree<dim, Number>&               data,
                                                  Vec&                                         dst,
                                                  const std::vector<Vec>&                      src,
                                                  const std::pair<unsigned int, unsigned int>& face_range) const {}
                                             /*-- No flux, so no contribution from this function ---*/

    /*--- Assembler function related to the bilinear form of the energy equation. Only cell contribution is present,
          since, basically, we end up with a mass matrix. ---*/
    void assemble_cell_term_energy_update(const MatrixFree<dim, Number>&               data,
                                          Vec&                                         dst,
                                          const Vec&                                   src,
                                          const std::pair<unsigned int, unsigned int>& cell_range) const;

    /*--- Assembler function for the diagonal part of the matrix for the continuity equation. ---*/
    void assemble_diagonal_cell_term_rho_update(const MatrixFree<dim, Number>&               data,
                                                Vec&                                         dst,
                                                const unsigned int&                          src,
                                                const std::pair<unsigned int, unsigned int>& cell_range) const;

    /*--- Assembler function for the diagonal part of the matrix for the momentum equation. ---*/
    void assemble_diagonal_cell_term_momentum_update(const MatrixFree<dim, Number>&               data,
                                                     Vec&                                         dst,
                                                     const unsigned int&                          src,
                                                     const std::pair<unsigned int, unsigned int>& cell_range) const;

    /*--- Assembler function for the diagonal part of the matrix for the energy equation. ---*/
    void assemble_diagonal_cell_term_energy_update(const MatrixFree<dim, Number>&               data,
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
  EULEROperator(): MatrixFreeOperators::Base<dim, Vec>(), Ma(), dt(), NS_stage(1) {}


  // Constructor with runtime parameters storage
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  EULEROperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                     n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  EULEROperator(RunTimeParameters::Data_Storage& data): MatrixFreeOperators::Base<dim, Vec>(),
                                                        Ma(data.Mach), dt(data.dt), NS_stage(1) {}


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


  // Setter of NS stage (this can be known only during the effective execution
  // and so it has to be demanded to the class that really solves the problem)
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void EULEROperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                          n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  set_NS_stage(const unsigned int stage) {
    AssertIndexRange(stage, 4);
    Assert(stage > 0, ExcInternalError());

    NS_stage = stage;
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
    FEEvaluation<dim, fe_degree_rho, n_q_points_1d_u, 1, Number> phi(data, 2),
                                                                 phi_rho(data, 2);
    FEEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number> phi_rhou(data, 0);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
      phi_rho.reinit(cell);
      phi_rho.gather_evaluate(src[0], EvaluationFlags::values);

      phi_rhou.reinit(cell);
      phi_rhou.gather_evaluate(src[1], EvaluationFlags::values);

      phi.reinit(cell);

      for(unsigned int q = 0; q < phi.n_q_points; ++q) {
        const auto& rho  = phi_rho.get_value(q);

        const auto& rhou = phi_rhou.get_value(q);

        phi.submit_value(rho, q);
        phi.submit_gradient(dt*rhou, q);
      }
      phi.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
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
    FEFaceEvaluation<dim, fe_degree_rho, n_q_points_1d_u, 1, Number> phi_p(data, true, 2),
                                                                     phi_m(data, false, 2),
                                                                     phi_rho_p(data, true, 2),
                                                                     phi_rho_m(data, false, 2);
    FEFaceEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number> phi_rhou_p(data, true, 0),
                                                                     phi_rhou_m(data, false, 0);
    FEFaceEvaluation<dim, fe_degree_T, n_q_points_1d_u, 1, Number>   phi_rhoE_p(data, true, 1),
                                                                     phi_rhoE_m(data, false, 1);

    for(unsigned int face = face_range.first; face < face_range.second; ++face) {
      phi_rho_p.reinit(face);
      phi_rho_p.gather_evaluate(src[0], EvaluationFlags::values);
      phi_rho_m.reinit(face);
      phi_rho_m.gather_evaluate(src[0], EvaluationFlags::values);

      phi_rhou_p.reinit(face);
      phi_rhou_p.gather_evaluate(src[1], EvaluationFlags::values);
      phi_rhou_m.reinit(face);
      phi_rhou_m.gather_evaluate(src[1], EvaluationFlags::values);

      phi_rhoE_p.reinit(face);
      phi_rhoE_p.gather_evaluate(src[2], EvaluationFlags::values);
      phi_rhoE_m.reinit(face);
      phi_rhoE_m.gather_evaluate(src[2], EvaluationFlags::values);

      phi_p.reinit(face);
      phi_m.reinit(face);

      for(unsigned int q = 0; q < phi_p.n_q_points; ++q) {
        const auto& n_plus   = phi_p.get_normal_vector(q);

        const auto& rho_p    = phi_rho_p.get_value(q);
        const auto& rho_m    = phi_rho_m.get_value(q);

        const auto& rhou_p   = phi_rhou_p.get_value(q);
        const auto& rhou_m   = phi_rhou_m.get_value(q);

        const auto& rhoE_p   = phi_rhoE_p.get_value(q);
        const auto& rhoE_m   = phi_rhoE_m.get_value(q);

        const auto& pres_p   = (EquationData::Cp_Cv - 1.0)*
                               (rhoE_p - 0.5*Ma*Ma*scalar_product(rhou_p, rhou_p)/rho_p);
        const auto& pres_m   = (EquationData::Cp_Cv - 1.0)*
                               (rhoE_m - 0.5*Ma*Ma*scalar_product(rhou_m, rhou_m/rho_m));

        const auto& avg_flux = 0.5*(rhou_p + rhou_m);

        const auto& c2_p     = EquationData::Cp_Cv*pres_p/rho_p;
        const auto& c2_m     = EquationData::Cp_Cv*pres_m/rho_m;
        const auto& lambda   = std::max(std::sqrt(scalar_product(rhou_p/rho_p, rhou_p/rho_p)) + 1.0/Ma*std::sqrt(std::abs(c2_p)),
                                        std::sqrt(scalar_product(rhou_m/rho_m, rhou_m/rho_m)) + 1.0/Ma*std::sqrt(std::abs(c2_m)));
        const auto& jump_rho = rho_p - rho_m;

        phi_p.submit_value(-dt*(scalar_product(avg_flux, n_plus) + 0.5*lambda*jump_rho), q);
        phi_m.submit_value(dt*(scalar_product(avg_flux, n_plus) + 0.5*lambda*jump_rho), q);
      }
      phi_p.integrate_scatter(EvaluationFlags::values, dst);
      phi_m.integrate_scatter(EvaluationFlags::values, dst);
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


  // Assemble rhs cell term for the momentum update
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void EULEROperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                     n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  assemble_rhs_cell_term_momentum_update(const MatrixFree<dim, Number>&               data,
                                         Vec&                                         dst,
                                         const std::vector<Vec>&                      src,
                                         const std::pair<unsigned int, unsigned int>& cell_range) const {
    FEEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number> phi(data, 0),
                                                                 phi_rhou(data, 0);
    FEEvaluation<dim, fe_degree_T, n_q_points_1d_u, 1, Number>   phi_rhoE(data, 1);
    FEEvaluation<dim, fe_degree_rho, n_q_points_1d_u, 1, Number> phi_rho(data, 2);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
      phi_rho.reinit(cell);
      phi_rho.gather_evaluate(src[0], EvaluationFlags::values);

      phi_rhou.reinit(cell);
      phi_rhou.gather_evaluate(src[1], EvaluationFlags::values);

      phi_rhoE.reinit(cell);
      phi_rhoE.gather_evaluate(src[2], EvaluationFlags::values);

      phi.reinit(cell);

      for(unsigned int q = 0; q < phi.n_q_points; ++q) {
        const auto& rho              = phi_rho.get_value(q);

        const auto& rhou             = phi_rhou.get_value(q);

        const auto& rhoE             = phi_rhoE.get_value(q);

        const auto& pres             = (EquationData::Cp_Cv - 1.0)*
                                       (rhoE - 0.5*Ma*Ma*scalar_product(rhou, rhou)/rho);

        const auto& tensor_product_u = outer_product(rhou, rhou/rho);

        auto p_times_identity        = tensor_product_u;
        p_times_identity = 0;
        for(unsigned int d = 0; d < dim; ++d) {
          p_times_identity[d][d] = pres;
        }

        phi.submit_value(rhou, q);
        phi.submit_gradient(dt*tensor_product_u + dt/(Ma*Ma)*p_times_identity, q);
      }
      phi.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
    }
  }


  // Assemble rhs face term for the momentum update
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void EULEROperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                     n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  assemble_rhs_face_term_momentum_update(const MatrixFree<dim, Number>&               data,
                                         Vec&                                         dst,
                                         const std::vector<Vec>&                      src,
                                         const std::pair<unsigned int, unsigned int>& face_range) const {
    FEFaceEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number> phi_p(data, true, 0),
                                                                     phi_m(data, false, 0),
                                                                     phi_rhou_p(data, true, 0),
                                                                     phi_rhou_m(data, false, 0);
    FEFaceEvaluation<dim, fe_degree_rho, n_q_points_1d_u, 1, Number> phi_rho_p(data, true, 2),
                                                                     phi_rho_m(data, false, 2);
    FEFaceEvaluation<dim, fe_degree_T, n_q_points_1d_u, 1, Number>   phi_rhoE_p(data, true, 1),
                                                                     phi_rhoE_m(data, false, 1);

    for(unsigned int face = face_range.first; face < face_range.second; ++face) {
      phi_rho_p.reinit(face);
      phi_rho_p.gather_evaluate(src[0], EvaluationFlags::values);
      phi_rho_m.reinit(face);
      phi_rho_m.gather_evaluate(src[0], EvaluationFlags::values);

      phi_rhou_p.reinit(face);
      phi_rhou_p.gather_evaluate(src[1], EvaluationFlags::values);
      phi_rhou_m.reinit(face);
      phi_rhou_m.gather_evaluate(src[1], EvaluationFlags::values);

      phi_rhoE_p.reinit(face);
      phi_rhoE_p.gather_evaluate(src[2], EvaluationFlags::values);
      phi_rhoE_m.reinit(face);
      phi_rhoE_m.gather_evaluate(src[2], EvaluationFlags::values);

      phi_p.reinit(face);
      phi_m.reinit(face);

      for(unsigned int q = 0; q < phi_p.n_q_points; ++q) {
        const auto& n_plus               = phi_p.get_normal_vector(q);

        const auto& rho_p                = phi_rho_p.get_value(q);
        const auto& rho_m                = phi_rho_m.get_value(q);

        const auto& rhou_p               = phi_rhou_p.get_value(q);
        const auto& rhou_m               = phi_rhou_m.get_value(q);

        const auto& rhoE_p               = phi_rhoE_p.get_value(q);
        const auto& rhoE_m               = phi_rhoE_m.get_value(q);

        const auto& avg_tensor_product_u = 0.5*(outer_product(rhou_p, rhou_p/rho_p) + outer_product(rhou_m, rhou_m/rho_m));

        const auto& pres_p               = (EquationData::Cp_Cv - 1.0)*
                                           (rhoE_p - 0.5*Ma*Ma*scalar_product(rhou_p, rhou_p)/rho_p);
        const auto& pres_m               = (EquationData::Cp_Cv - 1.0)*
                                           (rhoE_m - 0.5*Ma*Ma*scalar_product(rhou_m, rhou_m)/rho_m);
        const auto& avg_pres             = 0.5*(pres_p + pres_m);

        const auto& c2_p                 = EquationData::Cp_Cv*pres_p/rho_p;
        const auto& c2_m                 = EquationData::Cp_Cv*pres_m/rho_m;
        const auto& lambda               = std::max(std::sqrt(scalar_product(rhou_p/rho_p, rhou_p/rho_p)) + 1.0/Ma*std::sqrt(std::abs(c2_p)),
                                                    std::sqrt(scalar_product(rhou_m/rho_m, rhou_m/rho_m)) + 1.0/Ma*std::sqrt(std::abs(c2_m)));
        const auto& jump_rhou            = rhou_p - rhou_m;

        phi_p.submit_value(-dt*(avg_tensor_product_u*n_plus +
                                1.0/(Ma*Ma)*avg_pres*n_plus +
                                0.5*lambda*jump_rhou), q);
        phi_m.submit_value(dt*(avg_tensor_product_u*n_plus +
                               1.0/(Ma*Ma)*avg_pres*n_plus +
                               0.5*lambda*jump_rhou), q);
      }
      phi_p.integrate_scatter(EvaluationFlags::values, dst);
      phi_m.integrate_scatter(EvaluationFlags::values, dst);
    }
  }


  // Assemble rhs boundary term for the momentum update
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void EULEROperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                     n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  assemble_rhs_boundary_term_momentum_update(const MatrixFree<dim, Number>&               data,
                                             Vec&                                         dst,
                                             const std::vector<Vec>&                      src,
                                             const std::pair<unsigned int, unsigned int>& face_range) const {
    FEFaceEvaluation<dim, fe_degree_rho, n_q_points_1d_u, 1, Number> phi_rho(data, true, 2);
    FEFaceEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number> phi(data, true, 0),
                                                                     phi_rhou(data, true, 0);
    FEFaceEvaluation<dim, fe_degree_T, n_q_points_1d_u, 1, Number>   phi_rhoE(data, true, 1);

    for(unsigned int face = face_range.first; face < face_range.second; ++face) {
      phi_rho.reinit(face);
      phi_rho.gather_evaluate(src[0], EvaluationFlags::values);

      phi_rhou.reinit(face);
      phi_rhou.gather_evaluate(src[1], EvaluationFlags::values);

      phi_rhoE.reinit(face);
      phi_rhoE.gather_evaluate(src[2], EvaluationFlags::values);

      phi.reinit(face);

      for(unsigned int q = 0; q < phi.n_q_points; ++q) {
        const auto& n_plus   = phi.get_normal_vector(q);

        const auto& rho      = phi_rho.get_value(q);

        const auto& rhou     = phi_rhou.get_value(q);

        const auto& rhoE     = phi_rhoE.get_value(q);

        const auto& pres     = (EquationData::Cp_Cv - 1.0)*
                               (rhoE - 0.5*Ma*Ma*scalar_product(rhou, rhou)/rho);

        const auto& pres_D   = pres;

        const auto& avg_pres = 0.5*(pres + pres_D);

        phi.submit_value(-dt*(1.0/(Ma*Ma)*avg_pres*n_plus), q);
      }
      phi.integrate_scatter(EvaluationFlags::values, dst);
    }
  }


  // Put together all the previous steps for momentum update
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void EULEROperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                     n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  vmult_rhs_momentum_update(Vec& dst, const std::vector<Vec>& src) const {
    for(unsigned int d = 0; d < src.size(); ++d) {
      src[d].update_ghost_values();
    }

    this->data->loop(&EULEROperator::assemble_rhs_cell_term_momentum_update,
                     &EULEROperator::assemble_rhs_face_term_momentum_update,
                     &EULEROperator::assemble_rhs_boundary_term_momentum_update,
                     this, dst, src, true,
                     MatrixFree<dim, Number>::DataAccessOnFaces::unspecified,
                     MatrixFree<dim, Number>::DataAccessOnFaces::unspecified);
  }


  // Assemble cell term for the momentum update
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void EULEROperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                     n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  assemble_cell_term_momentum_update(const MatrixFree<dim, Number>&               data,
                                     Vec&                                         dst,
                                     const Vec&                                   src,
                                     const std::pair<unsigned int, unsigned int>& cell_range) const {
    FEEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number> phi(data, 0);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
      phi.reinit(cell);
      phi.gather_evaluate(src, EvaluationFlags::values);

      for(unsigned int q = 0; q < phi.n_q_points; ++q) {
        phi.submit_value(phi.get_value(q), q);
      }
      phi.integrate_scatter(EvaluationFlags::values, dst);
    }
  }


  // Assemble rhs cell term for the energy update
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void EULEROperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                     n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  assemble_rhs_cell_term_energy_update(const MatrixFree<dim, Number>&               data,
                                       Vec&                                         dst,
                                       const std::vector<Vec>&                      src,
                                       const std::pair<unsigned int, unsigned int>& cell_range) const {
    FEEvaluation<dim, fe_degree_T, n_q_points_1d_u, 1, Number>   phi(data, 1),
                                                                 phi_rhoE(data, 1);
    FEEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number> phi_rhou(data, 0);
    FEEvaluation<dim, fe_degree_rho, n_q_points_1d_u, 1, Number> phi_rho(data, 2);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
      phi_rho.reinit(cell);
      phi_rho.gather_evaluate(src[0], EvaluationFlags::values);

      phi_rhou.reinit(cell);
      phi_rhou.gather_evaluate(src[1], EvaluationFlags::values);

      phi_rhoE.reinit(cell);
      phi_rhoE.gather_evaluate(src[2], EvaluationFlags::values);

      phi.reinit(cell);

      for(unsigned int q = 0; q < phi.n_q_points; ++q) {
        const auto& rho  = phi_rho.get_value(q);

        const auto& rhou = phi_rhou.get_value(q);

        const auto& rhoE = phi_rhoE.get_value(q);

        const auto& pres = (EquationData::Cp_Cv - 1.0)*
                           (rhoE - 0.5*Ma*Ma*scalar_product(rhou, rhou)/rho);

        phi.submit_value(rhoE, q);
        phi.submit_gradient(dt*(Ma*Ma*0.5*scalar_product(rhou/rho, rhou/rho)*rhou +
                                (rhoE - 0.5*Ma*Ma*scalar_product(rhou, rhou)/rho + pres)*rhou/rho), q);
      }
      phi.integrate_scatter(EvaluationFlags::values | EvaluationFlags::gradients, dst);
    }
  }


  // Assemble rhs face term for the energy update
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void EULEROperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                     n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  assemble_rhs_face_term_energy_update(const MatrixFree<dim, Number>&               data,
                                       Vec&                                         dst,
                                       const std::vector<Vec>&                      src,
                                       const std::pair<unsigned int, unsigned int>& face_range) const {
    FEFaceEvaluation<dim, fe_degree_T, n_q_points_1d_u, 1, Number>   phi_p(data, true, 1),
                                                                     phi_m(data, false, 1),
                                                                     phi_rhoE_p(data, true, 1),
                                                                     phi_rhoE_m(data, false, 1);
    FEFaceEvaluation<dim, fe_degree_rho, n_q_points_1d_u, 1, Number> phi_rho_p(data, true, 2),
                                                                     phi_rho_m(data, false, 2);
    FEFaceEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number> phi_rhou_p(data, true, 0),
                                                                     phi_rhou_m(data, false, 0);

    for(unsigned int face = face_range.first; face < face_range.second; ++face) {
      phi_rho_p.reinit(face);
      phi_rho_p.gather_evaluate(src[0], EvaluationFlags::values);
      phi_rho_m.reinit(face);
      phi_rho_m.gather_evaluate(src[0], EvaluationFlags::values);

      phi_rhou_p.reinit(face);
      phi_rhou_p.gather_evaluate(src[1], EvaluationFlags::values);
      phi_rhou_m.reinit(face);
      phi_rhou_m.gather_evaluate(src[1], EvaluationFlags::values);

      phi_rhoE_p.reinit(face);
      phi_rhoE_p.gather_evaluate(src[2], EvaluationFlags::values);
      phi_rhoE_m.reinit(face);
      phi_rhoE_m.gather_evaluate(src[2], EvaluationFlags::values);

      phi_p.reinit(face);
      phi_m.reinit(face);

      for(unsigned int q = 0; q < phi_p.n_q_points; ++q) {
        const auto& n_plus       = phi_p.get_normal_vector(q);

        const auto& rho_p        = phi_rho_p.get_value(q);
        const auto& rho_m        = phi_rho_m.get_value(q);
        const auto& rhou_p       = phi_rhou_p.get_value(q);
        const auto& rhou_m       = phi_rhou_m.get_value(q);
        const auto& rhoE_p       = phi_rhoE_p.get_value(q);
        const auto& rhoE_m       = phi_rhoE_m.get_value(q);

        const auto& pres_p       = (EquationData::Cp_Cv - 1.0)*
                                   (rhoE_p - 0.5*Ma*Ma*scalar_product(rhou_p, rhou_p)/rho_p);
        const auto& pres_m       = (EquationData::Cp_Cv - 1.0)*
                                   (rhoE_m - 0.5*Ma*Ma*scalar_product(rhou_m, rhou_m)/rho_m);

        const auto& avg_enthalpy = 0.5*((rhoE_p - 0.5*Ma*Ma/rho_p*scalar_product(rhou_p, rhou_p) + pres_p)*rhou_p/rho_p +
                                        (rhoE_m - 0.5*Ma*Ma/rho_m*scalar_product(rhou_m, rhou_m) + pres_m)*rhou_m/rho_m);
        const auto& avg_kinetic  = 0.5*(0.5*rhou_p*scalar_product(rhou_p/rho_p, rhou_p/rho_p) +
                                        0.5*rhou_m*scalar_product(rhou_m/rho_m, rhou_m/rho_m));

        const auto& c2_p         = EquationData::Cp_Cv*pres_p/rho_p;
        const auto& c2_m         = EquationData::Cp_Cv*pres_m/rho_m;
        const auto& lambda       = std::max(std::sqrt(scalar_product(rhou_p/rho_p, rhou_p/rho_p)) + 1.0/Ma*std::sqrt(std::abs(c2_p)),
                                            std::sqrt(scalar_product(rhou_m/rho_m, rhou_m/rho_m)) + 1.0/Ma*std::sqrt(std::abs(c2_m)));
        const auto& jump_rhoE    = rhoE_p - rhoE_m;

        phi_p.submit_value(-dt*(scalar_product(avg_enthalpy, n_plus) +
                                Ma*Ma*scalar_product(avg_kinetic, n_plus) +
                                0.5*lambda*jump_rhoE), q);
        phi_m.submit_value(dt*(scalar_product(avg_enthalpy, n_plus) +
                               Ma*Ma*scalar_product(avg_kinetic, n_plus) +
                               0.5*lambda*jump_rhoE), q);
      }
      phi_p.integrate_scatter(EvaluationFlags::values, dst);
      phi_m.integrate_scatter(EvaluationFlags::values, dst);
    }
  }


  // Put together all the previous steps for energy
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void EULEROperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                     n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  vmult_rhs_energy_update(Vec& dst, const std::vector<Vec>& src) const {
    for(unsigned int d = 0; d < src.size(); ++d) {
      src[d].update_ghost_values();
    }

    this->data->loop(&EULEROperator::assemble_rhs_cell_term_energy_update,
                     &EULEROperator::assemble_rhs_face_term_energy_update,
                     &EULEROperator::assemble_rhs_boundary_term_energy_update,
                     this, dst, src, true,
                     MatrixFree<dim, Number>::DataAccessOnFaces::unspecified,
                     MatrixFree<dim, Number>::DataAccessOnFaces::unspecified);
  }


  // Assemble cell term for the energy
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void EULEROperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                     n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  assemble_cell_term_energy_update(const MatrixFree<dim, Number>&               data,
                                   Vec&                                         dst,
                                   const Vec&                                   src,
                                   const std::pair<unsigned int, unsigned int>& cell_range) const {
    FEEvaluation<dim, fe_degree_T, n_q_points_1d_u, 1, Number> phi(data, 1);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell) {
      phi.reinit(cell);
      phi.gather_evaluate(src, EvaluationFlags::values);

      for(unsigned int q = 0; q < phi.n_q_points; ++q) {
        phi.submit_value(phi.get_value(q), q);
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
    AssertIndexRange(NS_stage, 4);
    Assert(NS_stage > 0, ExcInternalError());

    if(NS_stage == 1) {
      this->data->cell_loop(&EULEROperator::assemble_cell_term_rho_update,
                            this, dst, src, false);
    }
    else if(NS_stage == 2) {
      this->data->cell_loop(&EULEROperator::assemble_cell_term_momentum_update,
                            this, dst, src, false);
    }
    else if(NS_stage == 3) {
      this->data->cell_loop(&EULEROperator::assemble_cell_term_energy_update,
                            this, dst, src, false);
    }
    else {
      Assert(false, ExcNotImplemented());
    }
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
  assemble_diagonal_cell_term_momentum_update(const MatrixFree<dim, Number>&               data,
                                              Vec&                                         dst,
                                              const unsigned int&                          ,
                                              const std::pair<unsigned int, unsigned int>& cell_range) const {
    FEEvaluation<dim, fe_degree_u, n_q_points_1d_u, dim, Number> phi(data, 0);

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
      phi.reinit(cell);

      for(unsigned int i = 0; i < phi.dofs_per_component; ++i) {
        for(unsigned int j = 0; j < phi.dofs_per_component; ++j) {
          phi.submit_dof_value(Tensor<1, dim, VectorizedArray<Number>>(), j);
        }
        phi.submit_dof_value(tmp, i);
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


  // Assemble diagonal cell term for the contribution due to internal energy
  //
  template<int dim, int fe_degree_rho, int fe_degree_T, int fe_degree_u,
           int n_q_points_1d_rho, int n_q_points_1d_T, int n_q_points_1d_u, typename Vec, typename Number>
  void EULEROperator<dim, fe_degree_rho, fe_degree_T, fe_degree_u,
                     n_q_points_1d_rho, n_q_points_1d_T, n_q_points_1d_u, Vec, Number>::
  assemble_diagonal_cell_term_energy_update(const MatrixFree<dim, Number>&               data,
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

    if(NS_stage == 1) {
      this->data->initialize_dof_vector(inverse_diagonal, 2);

      this->data->cell_loop(&EULEROperator::assemble_diagonal_cell_term_rho_update,
                            this, inverse_diagonal, dummy, false);
    }
    else if(NS_stage == 2) {
      this->data->initialize_dof_vector(inverse_diagonal, 0);

      this->data->cell_loop(&EULEROperator::assemble_diagonal_cell_term_momentum_update,
                            this, inverse_diagonal, dummy, false);
    }
    else if(NS_stage == 3) {
      this->data->initialize_dof_vector(inverse_diagonal, 1);

      this->data->cell_loop(&EULEROperator::assemble_diagonal_cell_term_energy_update,
                            this, inverse_diagonal, dummy, false);
    }
    else {
      Assert(false, ExcNotImplemented());
    }

    /*--- For the preconditioner, we actually need the inverse of the diagonal ---*/
    for(unsigned int i = 0; i < inverse_diagonal.local_size(); ++i) {
      Assert(inverse_diagonal.local_element(i) != 0.0,
             ExcMessage("No diagonal entry in a definite operator should be zero"));
      inverse_diagonal.local_element(i) = 1.0/inverse_diagonal.local_element(i);
    }
  }
}
