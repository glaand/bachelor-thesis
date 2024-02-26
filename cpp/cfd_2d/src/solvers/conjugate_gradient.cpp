#include "cfd.h"

using namespace CFD;

void FluidSimulation::solveWithConjugatedGradient() {
    this->resetPressure();
    this->setBoundaryConditionsP();
    this->setBoundaryConditionsPGeometry();

    // reset norm check
    this->res_norm = 0.0;
    int n = 0;
    double lambda = 0.0;
    double alpha_top = 0.0;
    double alpha_bottom = 0.0;
    double alpha_top_new = 0.0;
    int maxiterations = std::max(this->grid.imax, this->grid.jmax);

    // Initial residual vector of Ax=b
    for (int i = 1; i < this->grid.imax + 1; i++) {
        for (int j = 1; j < this->grid.jmax + 1; j++) {
            this->grid.res(i,j) = this->grid.RHS(i,j) - (
                // Sparse matrix A
                (1/this->grid.dx2)*(this->grid.p(i+1,j) - 2*this->grid.p(i,j) + this->grid.p(i-1,j)) +
                (1/this->grid.dy2)*(this->grid.p(i,j+1) - 2*this->grid.p(i,j) + this->grid.p(i,j-1))
            );
            // copy residual to search_vector (with initial guess from multigrid)
            this->grid.search_vector(i,j) = this->grid.res(i,j);
            alpha_top += this->grid.res(i,j)*this->grid.res(i,j);
        }
    }

    while ((this->res_norm > this->eps || this->res_norm == 0) && n < maxiterations) {
        alpha_bottom = 0.0;
        // Laplacian operator of grid.res, because of dot product of <res, Asearch_vector>, A-Matrix is the laplacian operator
        for (int i = 1; i < this->grid.imax + 1; i++) {
            for (int j = 1; j < this->grid.jmax + 1; j++) {
                this->grid.Asearch_vector(i,j) = (
                    // Sparse matrix A
                    (1/this->grid.dx2)*(this->grid.search_vector(i+1,j) - 2*this->grid.search_vector(i,j) + this->grid.search_vector(i-1,j)) +
                    (1/this->grid.dy2)*(this->grid.search_vector(i,j+1) - 2*this->grid.search_vector(i,j) + this->grid.search_vector(i,j-1))
                );
                alpha_bottom += this->grid.search_vector(i,j)*this->grid.Asearch_vector(i,j);
            }
        }
        // Update pressure and new residual
        lambda = alpha_top/alpha_bottom;
        alpha_top_new = 0.0;
        for (int i = 1; i < this->grid.imax + 1; i++) {
            for (int j = 1; j < this->grid.jmax + 1; j++) {
                this->grid.po(i,j) = this->grid.p(i,j); // smart residual preparation
                this->grid.p(i,j) += lambda*this->grid.search_vector(i,j);
                this->grid.res(i,j) -= lambda*this->grid.Asearch_vector(i,j);
                alpha_top_new += this->grid.res(i,j)*this->grid.res(i,j);
            }
        }

        // Calculate new search vector
        lambda = alpha_top_new/alpha_top;
        for (int i = 1; i < this->grid.imax + 1; i++) {
            for (int j = 1; j < this->grid.jmax + 1; j++) {
                this->grid.search_vector(i,j) = this->grid.res(i,j) + lambda*this->grid.search_vector(i,j);
            }
        }
        alpha_top = alpha_top_new;

        this->computeDiscreteL2Norm();
        this->res_norm_over_it_with_pressure_solver(this->it) = this->res_norm;
        this->it++;
        n++;
    }
    this->setBoundaryConditionsP();
    this->setBoundaryConditionsPGeometry();
}