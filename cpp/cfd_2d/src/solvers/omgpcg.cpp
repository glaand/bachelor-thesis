#include "cfd.h"
#include <limits>
#include "mpi.h"

using namespace CFD;

void FluidSimulation::solveWithOMGPCG() {
    this->resetPressure();

    // reset norm check
    this->res_norm = 0.0;
    this->n_cg = 0;
    this->alpha_cg = 0.0;
    this->alpha_top_cg = 0.0;
    this->alpha_bottom_cg = 0.0;
    this->beta_cg = 0.0;
    this->beta_top_cg = 0.0;

    for (int i = 1; i < this->grid.imax + 1; i++) {
        for (int j = 1; j < this->grid.jmax + 1; j++) {
            this->grid.res(i,j) = this->grid.RHS(i,j);
            this->preconditioner.RHS(i,j) = this->grid.res(i,j);
        }
    }

    float htop = 1.0;
    float hbottom = 1.0;

    this->computeResidualNorm();
    float previous_res_norm = this->res_norm;
    float res_norm_ratio = 0.0;

    // List of Eigen::MatrixXf
    std::vector<Eigen::MatrixXf> p_list;

    int n_ml = 0;
    int n_mgpcg = 0;
    int local_done = 0;
    int global_done = 0;
    bool all_equal = true;

    while ((this->res_norm > this->eps || this->res_norm == 0) && this->n_cg < this->maxiterations_cg) {
        MPI_Allreduce(&local_done, &global_done, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        this->setBoundaryConditionsP();
        this->setBoundaryConditionsPGeometry();
        this->alpha_top_cg = 0.0;
        this->alpha_bottom_cg = 0.0;
        this->beta_top_cg= 0.0;
        this->res_norm = 0.0;

        this->preconditioner.p.setZero();
        Multigrid::vcycle(this->multigrid_hierarchy_preconditioner, this->multigrid_hierarchy_preconditioner->numLevels() - 1, this->omg, this->num_sweeps);
        for (int i = 1; i < this->grid.imax + 1; i++) {
            for (int j = 1; j < this->grid.jmax + 1; j++) {
                this->grid.search_vector(i,j) = this->preconditioner.p(i,j);
            }
        }
        if (this->save_ml) {
            this->saveMLData();
        }
        p_list.push_back(this->grid.search_vector);

        for (int v = this->n_cg - 3; v < this->n_cg; v++) {
            if (v < 0) {
                continue;
            }
            for (int i = 1; i < this->grid.imax + 1; i++) {
                for (int j = 1; j < this->grid.jmax + 1; j++) {
                    this->grid.Asearch_vector(i,j) = (
                        // Sparse matrix A
                        (1/this->grid.dx2)*(p_list[v](i+1,j) - 2*p_list[v](i,j) + p_list[v](i-1,j)) +
                        (1/this->grid.dy2)*(p_list[v](i,j+1) - 2*p_list[v](i,j) + p_list[v](i,j-1))
                    );
                    htop += p_list[n_cg](i,j)*this->grid.Asearch_vector(i,j);
                    hbottom += p_list[v](i,j)*this->grid.Asearch_vector(i,j);
                }
            }
            p_list[n_cg] -= htop/hbottom*p_list[v];
        }

        // Calculate alpha
        // Laplacian operator of error_vector from multigrid, because of dot product of <A, Pi>, A-Matrix is the laplacian operator
        for (int i = 1; i < this->grid.imax + 1; i++) {
            for (int j = 1; j < this->grid.jmax + 1; j++) {
                this->grid.Asearch_vector(i,j) = (
                    // Sparse matrix A
                    (1/this->grid.dx2)*(p_list[n_cg](i+1,j) - 2*p_list[n_cg](i,j) + p_list[n_cg](i-1,j)) +
                    (1/this->grid.dy2)*(p_list[n_cg](i,j+1) - 2*p_list[n_cg](i,j) + p_list[n_cg](i,j-1))
                );
                this->alpha_top_cg += p_list[n_cg](i,j)*this->grid.res(i,j);
                this->alpha_bottom_cg += p_list[n_cg](i,j)*this->grid.Asearch_vector(i,j);
            }
        }
        this->alpha_cg = this->alpha_top_cg/this->alpha_bottom_cg;
        if (std::isnan(this->alpha_cg)) {
            this->alpha_cg = 0.0;
        }

        // Update pressure and new residual
        for (int i = 1; i < this->grid.imax + 1; i++) {
            for (int j = 1; j < this->grid.jmax + 1; j++) {
                this->grid.p(i,j) += this->alpha_cg*p_list[n_cg](i,j);
                this->grid.res(i,j) -= this->alpha_cg*this->grid.Asearch_vector(i,j);
            }
        }

        // Copy res to preconditioner RHS
        this->preconditioner.RHS = this->grid.res;

        // Calculate norm of residual
        this->computeResidualNorm();

        // Convergence check
        this->res_norm_over_it_with_pressure_solver(this->it) = this->res_norm;
        res_norm_ratio = this->res_norm/previous_res_norm;
        previous_res_norm = this->res_norm;
        if (this->res_norm < this->eps) {
            this->it++;
            break;
        }

        this->it++;
        this->n_cg++;

        // Check if last 100 residuals are the same
        if (n_cg > 0) {
            all_equal = true;
            for (size_t i = this->it - 100; i < this->it; i++) {
                if (std::abs(res_norm_over_it_with_pressure_solver(i) - res_norm_over_it_with_pressure_solver(this->it - 100)) > 1e-6) {
                    all_equal = false;
                    break;
                }
            }
            if (all_equal) {
                break;
            }
        }
    }
    this->n_cg_over_it(this->it_wo_pressure_solver) = this->n_cg;
    local_done = 1;
    while (global_done != this->world_size) {
        MPI_Allreduce(&local_done, &global_done, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        this->setBoundaryConditionsP();
    }
    this->setBoundaryConditionsP();
    this->setBoundaryConditionsPGeometry();
}