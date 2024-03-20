#include "cfd.h"

using namespace CFD;

void FluidSimulation::solveWithMultigridJacobi() {
    // reset norm check
    this->res_norm = 0.0;
    this->n_cg = 0;

    this->resetPressure();

    while ((this->res_norm > this->eps || this->res_norm == 0) && this->n_cg < this->maxiterations_cg) {
        this->setBoundaryConditionsP();
        this->setBoundaryConditionsPGeometry();

        Multigrid::vcycle(this->multigrid_hierarchy, this->multigrid_hierarchy->numLevels() - 1, this->omg, this->num_sweeps);

        this->computeResidual();
        this->res_norm_over_it_with_pressure_solver(this->it) = this->res_norm;
        this->it++;
        this->n_cg++;
    }

    this->setBoundaryConditionsP();
    this->setBoundaryConditionsPGeometry();
}