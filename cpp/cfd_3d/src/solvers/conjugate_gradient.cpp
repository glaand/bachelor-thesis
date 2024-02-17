#include "cfd.h"

using namespace CFD;

void FluidSimulation::solveWithConjugatedGradient() {
    this->setBoundaryConditionsP();
    this->setBoundaryConditionsPGeometry();

    // reset norm check
    this->res_norm = 0.0;
    int n = 0;
    double lambda = 0.0;
    double alpha_top = 0.0;
    double alpha_bottom = 0.0;
    double alpha_top_new = 0.0;
    int maxiterations = std::max(this->grid.imax, std::max(this->grid.jmax, this->grid.kmax));

    //Multigrid::vcycle(this->multigrid_hierarchy, this->multigrid_hierarchy->numLevels() - 1, 1, 1);

    // Initial residual vector of Ax=b
    for (int i = 1; i < this->grid.imax + 1; i++) {
        for (int j = 1; j < this->grid.jmax + 1; j++) {
            for (int k = 1; k < this->grid.kmax + 1; k++) {
                /*without initiaul guess*/
                this->grid.res(i,j,k) = this->grid.RHS(i,j,k) - (
                    // Sparse matrix A
                    (1/this->grid.dx2)*(this->grid.p(i+1,j,k) - 2*this->grid.p(i,j,k) + this->grid.p(i-1,j,k)) +
                    (1/this->grid.dy2)*(this->grid.p(i,j+1,k) - 2*this->grid.p(i,j,k) + this->grid.p(i,j-1,k)) +
                    (1/this->grid.dz2)*(this->grid.p(i,j,k+1) - 2*this->grid.p(i,j,k) + this->grid.p(i,j,k-1))
                );
                // copy residual to search_vector (with initial guess from multigrid)
                /*this->grid.search_vector(i, j, k) = this->grid.res(i, j, k);*/
                alpha_top += this->grid.res(i, j, k) * this->grid.res(i, j, k);
            }
        }
    }

    while ((this->res_norm > this->eps || this->res_norm == 0) && n < maxiterations) {
        alpha_bottom = 0.0;
        // Laplacian operator of grid.res, because of dot product of <res, Asearch_vector>, A-Matrix is the laplacian operator
        for (int i = 1; i < this->grid.imax + 1; i++) {
            for (int j = 1; j < this->grid.jmax + 1; j++) {
                for (int k = 1; k < this->grid.kmax + 1; k++) {
                    this->grid.Asearch_vector(i, j, k) = (
                        // Sparse matrix A
                        (1 / this->grid.dx2) * (this->grid.search_vector(i + 1, j, k) - 2 * this->grid.search_vector(i, j, k) + this->grid.search_vector(i - 1, j, k)) +
                        (1 / this->grid.dy2) * (this->grid.search_vector(i, j + 1, k) - 2 * this->grid.search_vector(i, j, k) + this->grid.search_vector(i, j - 1, k)) +
                        (1 / this->grid.dz2) * (this->grid.search_vector(i, j, k + 1) - 2 * this->grid.search_vector(i, j, k) + this->grid.search_vector(i, j, k - 1))
                    );
                    alpha_bottom += this->grid.search_vector(i, j, k) * this->grid.Asearch_vector(i, j, k);
                }
            }
        }
        // Update pressure and new residual
        lambda = alpha_top / alpha_bottom;
        alpha_top_new = 0.0;
        for (int i = 1; i < this->grid.imax + 1; i++) {
            for (int j = 1; j < this->grid.jmax + 1; j++) {
                for (int k = 1; k < this->grid.kmax + 1; k++) {
                    this->grid.po(i, j, k) = this->grid.p(i, j, k); // smart residual preparation
                    this->grid.p(i, j, k) += lambda * this->grid.search_vector(i, j, k);
                    this->grid.res(i, j, k) -= lambda * this->grid.Asearch_vector(i, j, k);
                    alpha_top_new += this->grid.res(i, j, k) * this->grid.res(i, j, k);
                }
            }
        }

        // Calculate new search vector
        lambda = alpha_top_new / alpha_top;
        for (int i = 1; i < this->grid.imax + 1; i++) {
            for (int j = 1; j < this->grid.jmax + 1; j++) {
                for (int k = 1; k < this->grid.kmax + 1; k++) {
                    this->grid.search_vector(i, j, k) = this->grid.res(i, j, k) + lambda * this->grid.search_vector(i, j, k);
                }
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
