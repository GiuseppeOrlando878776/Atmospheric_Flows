#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>

#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_in.h>

#include <fstream>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/fe/fe_system.h>

#include <deal.II/fe/mapping_q.h>

#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>

#include <deal.II/numerics/vector_tools.h>

#include "equation_data.h"

int main() {
  using namespace dealii;

  Triangulation<2, 3> triangulation;
  GridGenerator::hyper_sphere(triangulation);

  triangulation.refine_global(3);

  GridOut grid_out;

  std::ofstream outfile("mesh.ucd");
  grid_out.write_ucd(triangulation, outfile);

  std::ofstream outfile_vtk("mesh.vtk");
  grid_out.write_vtk(triangulation, outfile_vtk);

  FESystem<2, 3> fe_density(FE_DGQ<2, 3>(2), 1);
  DoFHandler<2, 3> dof_handler_density(triangulation);
  dof_handler_density.distribute_dofs(fe_density);

  EquationData::Density<3> rho_init;

  Vector<double> rho;
  rho.reinit(dof_handler_density.n_dofs());

  VectorTools::interpolate(MappingQ<2, 3>(2), dof_handler_density, rho_init, rho);

  DataOut<2, DoFHandler<2, 3>> data_out;

  data_out.attach_dof_handler(dof_handler_density);
  data_out.add_data_vector(rho, "rho", DataOut<2, DoFHandler<2, 3>>::type_dof_data, {DataComponentInterpretation::component_is_scalar});
  data_out.build_patches(MappingQ<2, 3>(2), 2, DataOut<2, DoFHandler<2, 3>>::curved_inner_cells);

  std::ofstream outfile_sol_vtk("solution.vtk");
  data_out.write_vtk(outfile_sol_vtk);

  /*GridIn<2> grid_in;
  grid_in.attach_triangulation(triangulation);
  std::ifstream infile("mesh.ucd");
  grid_in.read_ucd(infile);*/
}
