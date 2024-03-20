#include "cfd.h"

using namespace CFD;

void FluidSimulation::solveWithSteepestDescent() {
    this->resetPressure();
    this->setBoundaryConditionsP();
    this->setBoundaryConditionsPGeometry();

    // reset norm check
    this->res_norm = 0.0;
    double lambda = 0.0;
    double alpha_top = 0.0;
    double alpha_bottom = 0.0;
    double alpha_top_new = 0.0;
    this->n_cg = 0;
    int maxiterations = std::max(this->grid.imax, this->grid.jmax);

    // Initial residual vector of Ax=b
    for (int i = 1; i < this->grid.imax + 1; i++) {
        for (int j = 1; j < this->grid.jmax + 1; j++) {
            this->grid.res(i,j) = this->grid.RHS(i,j);
            this->grid.search_vector(i,j) = this->grid.res(i,j);
        }
    }

    while ((this->res_norm > this->eps || this->res_norm == 0) && this->n_cg < this->maxiterations_cg) {
        this->setBoundaryConditionsP();
        this->setBoundaryConditionsPGeometry();
        alpha_top = 0.0;
        alpha_bottom = 0.0;
        // Laplacian operator of grid.res, because of dot product of <res, Asearch_vector>, A-Matrix is the laplacian operator
        for (int i = 1; i < this->grid.imax + 1; i++) {
            for (int j = 1; j < this->grid.jmax + 1; j++) {
                this->grid.Asearch_vector(i,j) = (
                    // Sparse matrix A
                    (1/this->grid.dx2)*(this->grid.search_vector(i+1,j) - 2*this->grid.search_vector(i,j) + this->grid.search_vector(i-1,j)) +
                    (1/this->grid.dy2)*(this->grid.search_vector(i,j+1) - 2*this->grid.search_vector(i,j) + this->grid.search_vector(i,j-1))
                );
                alpha_top += this->grid.res(i,j)*this->grid.res(i,j);
                alpha_bottom += this->grid.search_vector(i,j)*this->grid.Asearch_vector(i,j);
            }
        }
        lambda = alpha_top/alpha_bottom;
        for (int i = 1; i < this->grid.imax + 1; i++) {
            for (int j = 1; j < this->grid.jmax + 1; j++) {
                this->grid.p(i,j) += lambda*this->grid.search_vector(i,j);
                this->grid.res(i,j) -= lambda*this->grid.Asearch_vector(i,j);
                this->grid.search_vector(i,j) = this->grid.res(i,j);
            }
        }

        this->computeResidualNorm();
        this->res_norm_over_it_with_pressure_solver(this->it) = this->res_norm;
        this->it++;
        this->n_cg++;
    }
    this->n_cg_over_it(this->it_wo_pressure_solver) = this->n_cg;
    this->setBoundaryConditionsP();
    this->setBoundaryConditionsPGeometry();
}