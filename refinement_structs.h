/*--- Author: Giuseppe Orlando, 2022 ---*/

#include <deal.II/fe/fe_values.h>

using namespace dealii;

// Auxiliary structs for mesh adaptation procedure
//
template <int dim>
struct ScratchData {
  ScratchData(const FiniteElement<dim>& fe,
              const unsigned int        quadrature_degree,
              const UpdateFlags         update_flags): fe_values(fe, QGauss<dim>(quadrature_degree), update_flags) {}

  ScratchData(const ScratchData<dim>& scratch_data): fe_values(scratch_data.fe_values.get_fe(),
                                                               scratch_data.fe_values.get_quadrature(),
                                                               scratch_data.fe_values.get_update_flags()) {}
  FEValues<dim> fe_values;
};


struct CopyData {
  CopyData() : cell_index(numbers::invalid_unsigned_int), value(0.0)  {}

  CopyData(const CopyData &) = default;

  unsigned int cell_index;
  double       value;
};
