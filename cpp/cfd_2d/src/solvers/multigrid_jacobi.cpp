#include "cfd.h"

using namespace CFD;

void FluidSimulation::solveWithMultigridJacobi() {
    // reset norm check
    this->res_norm = 0.0;

    this->setBoundaryConditionsP();
    this->setBoundaryConditionsPGeometry();

    while (this->res_norm > this->eps || this->res_norm == 0) {

        Multigrid::vcycle(this->multigrid_hierarchy, this->multigrid_hierarchy->numLevels() - 1, this->omg, 1);

        this->computeDiscreteL2Norm();
        this->res_norm_over_it_with_pressure_solver(this->it) = this->res_norm;
        this->it++;
    }

    this->setBoundaryConditionsP();
    this->setBoundaryConditionsPGeometry();
}