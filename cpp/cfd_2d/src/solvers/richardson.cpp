#include "cfd.h"
#include "omp.h"

using namespace CFD;

void FluidSimulation::solveWithRichardson() {
    this->resetPressure();

    this->res_norm = 0.0;
    this->n_cg = 0;
    while ((this->res_norm > this->eps || this->res_norm == 0) && this->n_cg < this->maxiterations_cg) {
        this->setBoundaryConditionsP();
        this->setBoundaryConditionsPGeometry();

        this->grid.po = this->grid.p;
        for (int j = 1; j < this->grid.jmax + 1; j++) {
            for (int i = 1; i < this->grid.imax + 1; i++) {
                this->grid.res(i,j) = this->grid.RHS(i,j) - (
                    (1/this->grid.dx2)*(this->grid.po(i+1,j) - 2*this->grid.po(i,j) + this->grid.po(i-1,j)) +
                    (1/this->grid.dy2)*(this->grid.po(i,j+1) - 2*this->grid.po(i,j) + this->grid.po(i,j-1))
                );
                this->grid.p(i,j) = this->grid.po(i,j) + this->omg * this->grid.res(i,j);
            }
        }

        // Compute residual
        for (int j = 1; j < this->grid.jmax + 1; j++) {
            for (int i = 1; i < this->grid.imax + 1; i++) {
                this->grid.res(i,j) = this->grid.RHS(i,j) - (
                    (1/this->grid.dx2)*(this->grid.p(i+1,j) - 2*this->grid.p(i,j) + this->grid.p(i-1,j)) +
                    (1/this->grid.dy2)*(this->grid.p(i,j+1) - 2*this->grid.p(i,j) + this->grid.p(i,j-1))
                );
            }
        }

        this->computeResidualNorm();
        this->res_norm_over_it_with_pressure_solver(this->it) = this->res_norm;
        this->it++;
        this->n_cg++;
    }

    this->setBoundaryConditionsP();
    this->setBoundaryConditionsPGeometry();
}