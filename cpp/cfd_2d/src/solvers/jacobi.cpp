#include "cfd.h"

using namespace CFD;

void FluidSimulation::solveWithJacobi() {
    this->resetPressure();

    this->res_norm = 0.0;
    this->n_cg = 0;
    float inv_d = 1.0/(-2.0*this->grid.dx2 - 2.0*this->grid.dy2);
    while ((this->res_norm > this->eps || this->res_norm == 0) && this->n_cg < this->maxiterations_cg) {
        this->setBoundaryConditionsP();
        this->setBoundaryConditionsPGeometry();
        // Jacobi smoother with relaxation factor (omega)
        this->grid.po = this->grid.p;
        for (int i = 1; i < this->grid.imax + 1; i++) {
            for (int j = 1; j < this->grid.jmax + 1; j++) {
                this->grid.p(i, j) = (
                    (inv_d) // 1/Aii
                    *
                    (
                        this->grid.RHS(i,j)*this->grid.dx2dy2 - this->grid.dx2*(this->grid.po(i+1,j) + this->grid.po(i-1,j)) - this->grid.dy2*(this->grid.po(i,j+1) + this->grid.po(i,j-1))
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