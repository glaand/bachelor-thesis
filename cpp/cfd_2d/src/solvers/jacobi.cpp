#include "cfd.h"

using namespace CFD;

void FluidSimulation::solveWithJacobi() {
    this->resetPressure();
    this->setBoundaryConditionsP();
    this->setBoundaryConditionsPGeometry();

    this->res_norm = 0.0;
    while (this->res_norm > this->eps || this->res_norm == 0) {
        // Jacobi smoother with relaxation factor (omega)
        for (int i = 1; i <= this->grid.imax; i++) {
            for (int j = 1; j <= this->grid.jmax; j++) {
                this->grid.po(i,j) = this->grid.p(i,j); // smart residual preparation
                this->grid.p(i, j) = (
                    (1/(-2*this->grid.dx2 - 2*this->grid.dy2)) // 1/Aii
                    *
                    (
                        this->grid.RHS(i,j)*this->grid.dx2dy2 - this->grid.dy2*(this->grid.p(i+1,j) + this->grid.p(i-1,j)) - this->grid.dx2*(this->grid.p(i,j+1) + this->grid.p(i,j-1))
                    )
                );
            }
        }

        this->computeDiscreteL2Norm();
        this->res_norm_over_it_with_pressure_solver(this->it) = this->res_norm;
        this->it++;
    }

    this->setBoundaryConditionsP();
    this->setBoundaryConditionsPGeometry();
}