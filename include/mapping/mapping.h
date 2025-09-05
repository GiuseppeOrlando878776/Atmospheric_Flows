/*--- Author: Giuseppe Orlando, 2025. ---*/

// @sect{Include files}

// We start by including the necessary deal.II header file and a related
// header file with some constant values
//
#include <deal.II/base/function.h>

#include "../equation_data.h"

// @sect{Mapping reference-physical domain}

// In this namespace, we declare the mapping between reference and physical
// elements using the Gal-Chen mapping
//
namespace GalChenMapping {
  using namespace dealii;

  static const unsigned degree_mapping          = 2;                                                             /*--- Mapping degree ---*/
  static const unsigned extra_quadrature_degree = (degree_mapping == 1) ? 0 : my_ceil(0.5*(degree_mapping - 2)); /*--- Extra accuracy
                                                                                                                       for quadratures ---*/

  static const double h  = 400.0;   /*--- Hill height ---*/
  static const double xc = 30000.0; /*--- x-Center of the hill ---*/
  static const double yc = 20000.0; /*--- x-Center of the hill ---*/
  static const double ac = 1000.0;  /*--- Width of the hill ---*/

  /**
   * Now we can focus on mappings from reference element to the physical one
     using the Gal-Chen. Notice that lenghts are in kilometers because of
     the non-dimensional version (the characteristic length is assumed 1 km).
   */
  template<unsigned dim, typename T = double>
  class PushForward: public Function<dim, T> {
  public:
    PushForward(): Function<dim, T>(dim, static_cast<T>(0.0)),
                   z_max(static_cast<T>(EquationData::z_max)/static_cast<T>(EquationData::L_ref)) {}

    virtual ~PushForward() {};

    virtual T value(const Point<dim, T>& p,
                    const unsigned       component = 0) const override;

  private:
    const T z_max;
  };

  // Mapping from reference to physical
  //
  template<unsigned dim, typename T>
  T PushForward<dim, T>::value(const Point<dim, T>& p,
                               const unsigned       component) const {
    // x component
    if(component == 0) {
      return p[0];
    }
    // y component
    else if(component == 1) {
      return p[1];
    }
    // z component
    else if(component == 2) {
      auto hX = static_cast<T>(GalChenMapping::h)/
                std::pow(static_cast<T>(1.0) +
                         (p[0]*static_cast<T>(EquationData::L_ref) - static_cast<T>(GalChenMapping::xc))/
                         static_cast<T>(GalChenMapping::ac)*
                         (p[0]*static_cast<T>(EquationData::L_ref) - static_cast<T>(GalChenMapping::xc))/
                         static_cast<T>(GalChenMapping::ac) +
                         (p[1]*static_cast<T>(EquationData::L_ref) - static_cast<T>(GalChenMapping::yc))/
                         static_cast<T>(GalChenMapping::ac)*
                         (p[1]*static_cast<T>(EquationData::L_ref) - static_cast<T>(GalChenMapping::yc))/
                         static_cast<T>(GalChenMapping::ac), static_cast<T>(1.5));
      hX /= static_cast<T>(EquationData::L_ref);

      return p[2] + ((z_max - p[2])/z_max)*hX;
    }
  }


  /**
   * We compute now the inverse mapping (from physical to reference).
     Notice that lenghts are in kilometers becasue of the non-dimensional version.
   */
  template<unsigned dim, typename T = double>
  class PullBack: public Function<dim, T> {
  public:
    PullBack(): Function<dim, T>(dim, static_cast<T>(0.0)),
                z_max(static_cast<T>(EquationData::z_max)/static_cast<T>(EquationData::L_ref)) {}

    virtual ~PullBack() {};

    virtual T value(const Point<dim, T>& p,
                    const unsigned       component = 0) const override;

  private:
    const T z_max;
  };

  // Mapping from physical to reference
  //
  template<unsigned dim, typename T>
  T PullBack<dim, T>::value(const Point<dim, T>& p,
                            const unsigned       component) const {
    // x component
    if(component == 0) {
      return p[0];
    }
    // y component
    else if(component == 1) {
      return p[1];
    }
    // z component
    else if(component == 2) {
      auto hx = static_cast<T>(GalChenMapping::h)/
                std::pow(static_cast<T>(1.0) +
                         (p[0]*static_cast<T>(EquationData::L_ref) - static_cast<T>(GalChenMapping::xc))/
                         static_cast<T>(GalChenMapping::ac)*
                         (p[0]*static_cast<T>(EquationData::L_ref) - static_cast<T>(GalChenMapping::xc))/
                         static_cast<T>(GalChenMapping::ac) +
                         (p[1]*static_cast<T>(EquationData::L_ref) - static_cast<T>(GalChenMapping::yc))/
                         static_cast<T>(GalChenMapping::ac)*
                         (p[1]*static_cast<T>(EquationData::L_ref) - static_cast<T>(GalChenMapping::yc))/
                         static_cast<T>(GalChenMapping::ac), static_cast<T>(1.5));
      hx /= static_cast<T>(EquationData::L_ref);

      return z_max*(p[2] - hx)/(z_max - hx);
    }
  }

} // namespace Mapping
