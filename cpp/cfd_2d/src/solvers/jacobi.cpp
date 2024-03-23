#include "cfd.h"
#include "omp.h"

using namespace CFD;

void FluidSimulation::solveWithJacobi() {
    this->resetPressure();

    this->res_norm = 0.0;
    this->n_cg = 0;
    float inv_d = 1.0/(-2.0*this->grid.dx2 - 2.0*this->grid.dy2);
    this->maxiterations_cg = 300;
    FluidSimulation* sim = this;
    while ((this->res_norm > this->eps || this->res_norm == 0) && this->n_cg < this->maxiterations_cg) {
        this->setBoundaryConditionsP();
        this->setBoundaryConditionsPGeometry();
        // Jacobi smoother with relaxation factor (omega)
        this->grid.po = this->grid.p;
        #pragma omp parallel for num_threads(2) shared(sim)
        for (int j = 1; j < sim->grid.jmax + 1; j++) {
            for (int i = 1; i < sim->grid.imax + 1; i++) {
                sim->grid.p(i, j) = (
                    (inv_d) // 1/Aii
                    *
                    (
                        sim->grid.RHS(i,j)*sim->grid.dx2dy2 - sim->grid.dx2*(sim->grid.po(i+1,j) + sim->grid.po(i-1,j)) - sim->grid.dy2*(sim->grid.po(i,j+1) + sim->grid.po(i,j-1))
                    )
                );
            }
        }

        this->computeResidual();
        this->res_norm_over_it_with_pressure_solver(this->it) = this->res_norm;
        this->it++;
        this->n_cg++;
    }

    this->setBoundaryConditionsP();
    this->setBoundaryConditionsPGeometry();
}