#include "cfd.h"

using namespace CFD;

void FluidSimulation::solveWithMultigridPCG() {
    this->setBoundaryConditionsP();
    this->setBoundaryConditionsPGeometry();

    // reset norm check
    this->res_norm = 0.0;
    int n = 0;
    double alpha = 0.0;
    double alpha_top = 0.0;
    double alpha_bottom = 0.0;
    double beta = 0.0;
    double beta_top = 0.0;
    double beta_bottom = 0.0;

    int maxiterations = std::max(this->grid.imax, this->grid.jmax);

    Multigrid::vcycle(this->multigrid_hierarchy, this->multigrid_hierarchy->numLevels() - 1, 1, 1); // initial guess with multigrid

    // Initial residual vector of Ax=b
    for (int i = 1; i < this->grid.imax + 1; i++) {
        for (int j = 1; j < this->grid.jmax + 1; j++) {
            /* copy residual to preconditioner and reset grids (without initiaul guess)
            this->grid.res(i,j) = this->grid.RHS(i,j) - (
                // Sparse matrix A
                (1/this->grid.dx2)*(this->grid.p(i+1,j) - 2*this->grid.p(i,j) + this->grid.p(i-1,j)) +
                (1/this->grid.dy2)*(this->grid.p(i,j+1) - 2*this->grid.p(i,j) + this->grid.p(i,j-1))
            );*/
            this->preconditioner.RHS(i,j) = this->grid.res(i,j);
            this->preconditioner.p(i,j) = 0.0;
        }
    }

    // Initial guess for error vector
    Multigrid::vcycle(this->multigrid_hierarchy_preconditioner, this->multigrid_hierarchy_preconditioner->numLevels() - 1, 1, 1);

    // Initial search vector
    this->grid.search_vector = this->preconditioner.p;

    while ((this->res_norm > this->eps || this->res_norm == 0) && n < maxiterations) {
        alpha_top = 0.0;
        alpha_bottom = 0.0;

        // Calculate alpha
        // Laplacian operator of error_vector from multigrid, because of dot product of <A, Pi>, A-Matrix is the laplacian operator
        for (int i = 1; i < this->grid.imax + 1; i++) {
            for (int j = 1; j < this->grid.jmax + 1; j++) {
                this->grid.Asearch_vector(i,j) = (
                    // Sparse matrix A
                    (1/this->grid.dx2)*(this->grid.search_vector(i+1,j) - 2*this->grid.search_vector(i,j) + this->grid.search_vector(i-1,j)) +
                    (1/this->grid.dy2)*(this->grid.search_vector(i,j+1) - 2*this->grid.search_vector(i,j) + this->grid.search_vector(i,j-1))
                );
                alpha_top += this->preconditioner.p(i,j)*this->grid.res(i,j);
                alpha_bottom += this->grid.search_vector(i,j)*this->grid.Asearch_vector(i,j);
            }
        }
        alpha = alpha_top/alpha_bottom;

        // Update pressure and new residual
        for (int i = 1; i < this->grid.imax + 1; i++) {
            for (int j = 1; j < this->grid.jmax + 1; j++) {
                this->grid.po(i,j) = this->grid.p(i,j); // smart residual preparation
                this->grid.p(i,j) += alpha*this->grid.search_vector(i,j);
                this->grid.res(i,j) -= alpha*this->grid.Asearch_vector(i,j);
                this->preconditioner.RHS(i,j) = this->grid.res(i,j);
            }
        }
        
        // New guess for error vector
        Multigrid::vcycle(this->multigrid_hierarchy_preconditioner, this->multigrid_hierarchy_preconditioner->numLevels() - 1, 1, 1);

        // Calculate beta
        for (int i = 1; i < this->grid.imax + 1; i++) {
            for (int j = 1; j < this->grid.jmax + 1; j++) {
                beta_top += this->preconditioner.p(i,j)*this->grid.res(i,j);
            }
        }
        beta = beta_top/alpha_top;

        // Calculate new search vector
        for (int i = 1; i < this->grid.imax + 1; i++) {
            for (int j = 1; j < this->grid.jmax + 1; j++) {
                this->grid.search_vector(i,j) = this->preconditioner.p(i,j) + beta*this->grid.search_vector(i,j);
            }
        }

        this->computeDiscreteL2Norm();
        this->res_norm_over_it_with_pressure_solver(this->it) = this->res_norm;
        this->it++;
        n++;
    }
    this->setBoundaryConditionsP();
    this->setBoundaryConditionsPGeometry();
}