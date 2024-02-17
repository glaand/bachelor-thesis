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

    int maxiterations = std::max(this->grid.imax, std::max(this->grid.jmax, this->grid.kmax));

    Multigrid::vcycle(this->multigrid_hierarchy, this->multigrid_hierarchy->numLevels() - 1, 1, 1); // initial guess with multigrid

    // Initial residual vector of Ax=b
    for (int i = 1; i <= this->grid.imax; i++) {
        for (int j = 1; j <= this->grid.jmax; j++) {
            for (int k = 1; k <= this->grid.kmax; k++) {
                this->preconditioner.RHS(i, j, k) = this->grid.res(i, j, k);
                this->preconditioner.p(i, j, k) = 0.0;
            }
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
        for (int i = 1; i <= this->grid.imax; i++) {
            for (int j = 1; j <= this->grid.jmax; j++) {
                for (int k = 1; k <= this->grid.kmax; k++) {
                    this->grid.Asearch_vector(i, j, k) = (
                        // Sparse matrix A
                        (1 / this->grid.dx2) * (this->grid.search_vector(i+1, j, k) - 2 * this->grid.search_vector(i, j, k) + this->grid.search_vector(i-1, j, k)) +
                        (1 / this->grid.dy2) * (this->grid.search_vector(i, j+1, k) - 2 * this->grid.search_vector(i, j, k) + this->grid.search_vector(i, j-1, k)) +
                        (1 / this->grid.dz2) * (this->grid.search_vector(i, j, k+1) - 2 * this->grid.search_vector(i, j, k) + this->grid.search_vector(i, j, k-1))
                    );
                    alpha_top += this->preconditioner.p(i, j, k) * this->grid.res(i, j, k);
                    alpha_bottom += this->grid.search_vector(i, j, k) * this->grid.Asearch_vector(i, j, k);
                }
            }
        }
        alpha = alpha_top / alpha_bottom;

        // Update pressure and new residual
        for (int i = 1; i <= this->grid.imax; i++) {
            for (int j = 1; j <= this->grid.jmax; j++) {
                for (int k = 1; k <= this->grid.kmax; k++) {
                    this->grid.po(i, j, k) = this->grid.p(i, j, k); // smart residual preparation
                    this->grid.p(i, j, k) += alpha * this->grid.search_vector(i, j, k);
                    this->grid.res(i, j, k) -= alpha * this->grid.Asearch_vector(i, j, k);
                    this->preconditioner.RHS(i, j, k) = this->grid.res(i, j, k);
                }
            }
        }

        // New guess for error vector
        Multigrid::vcycle(this->multigrid_hierarchy_preconditioner, this->multigrid_hierarchy_preconditioner->numLevels() - 1, 1, 1);

        // Calculate beta
        for (int i = 1; i <= this->grid.imax; i++) {
            for (int j = 1; j <= this->grid.jmax; j++) {
                for (int k = 1; k <= this->grid.kmax; k++) {
                    beta_top += this->preconditioner.p(i, j, k) * this->grid.res(i, j, k);
                }
            }
        }
        beta = beta_top / alpha_top;

        // Calculate new search vector
        for (int i = 1; i <= this->grid.imax; i++) {
            for (int j = 1; j <= this->grid.jmax; j++) {
                for (int k = 1; k <= this->grid.kmax; k++) {
                    this->grid.search_vector(i, j, k) = this->preconditioner.p(i, j, k) + beta * this->grid.search_vector(i, j, k);
                }
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
