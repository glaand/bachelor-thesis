#include "cfd.h"

using namespace CFD;

void FluidSimulation::inferenceExp1() {
    // Initial guess for error vector with deep learning
    auto options = torch::TensorOptions().dtype(torch::kFloat);
    torch::Tensor T = torch::from_blob(this->grid.res.data(), {this->grid.res.rows(), this->grid.res.cols()}, options).clone();
    T = T.to(torch::kCUDA);
    T = T.unsqueeze(0).unsqueeze(0);
    auto output = this->model.forward({ T }).toTensor();
    output = output.to(torch::kCPU).squeeze(0).squeeze(0);

    // set preconditioner p to 0
    for (int i = 0; i < this->grid.imax + 2; i++) {
        for (int j = 0; j < this->grid.jmax + 2; j++) {
            this->preconditioner.p(i,j) = 0;
        }
    }

    auto output_acc = output.accessor<float,2>();
    for(int i = 1; i < this->grid.imax + 1; i++) {
        for(int j = 1; j < this->grid.jmax + 1; j++) {
            this->preconditioner.p(i,j) = output_acc[j][i];
        }
    }
}

void FluidSimulation::solveWithML() {
    this->resetPressure();
    this->setBoundaryConditionsP();
    this->setBoundaryConditionsPGeometry();

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
            this->grid.res(i,j) = this->grid.RHS(i,j) - (
                // Sparse matrix A
                (1/this->grid.dx2)*(this->grid.p(i+1,j) - 2*this->grid.p(i,j) + this->grid.p(i-1,j)) +
                (1/this->grid.dy2)*(this->grid.p(i,j+1) - 2*this->grid.p(i,j) + this->grid.p(i,j-1))
            );
            this->preconditioner.RHS(i,j) = this->grid.res(i,j);
        }
    }

    int initial_res_norm = this->grid.res.norm();

    // set preconditioner p to 0
    for (int i = 0; i < this->grid.imax + 2; i++) {
        for (int j = 0; j < this->grid.jmax + 2; j++) {
            this->preconditioner.p(i,j) = 0;
        }
    }

    // Initial guess for error vector
    Multigrid::vcycle(this->multigrid_hierarchy_preconditioner, this->multigrid_hierarchy_preconditioner->numLevels() - 1, this->omg, this->num_sweeps);

    // Initial search vector
    this->grid.search_vector = this->preconditioner.p;

    while ((this->res_norm > this->eps || this->res_norm == 0) && this->n_cg < this->maxiterations_cg) {
        this->alpha_top_cg = 0.0;
        this->alpha_bottom_cg = 0.0;
        this->beta_top_cg = 0.0;
        this->res_norm = 0.0;

        // Calculate alpha
        // Laplacian operator of error_vector from multigrid, because of dot product of <A, Pi>, A-Matrix is the laplacian operator
        for (int i = 1; i < this->grid.imax + 1; i++) {
            for (int j = 1; j < this->grid.jmax + 1; j++) {
                this->grid.Asearch_vector(i,j) = (
                    // Sparse matrix A
                    (1/this->grid.dx2)*(this->grid.search_vector(i+1,j) - 2*this->grid.search_vector(i,j) + this->grid.search_vector(i-1,j)) +
                    (1/this->grid.dy2)*(this->grid.search_vector(i,j+1) - 2*this->grid.search_vector(i,j) + this->grid.search_vector(i,j-1))
                );
                this->alpha_top_cg += this->preconditioner.p(i,j)*this->grid.res(i,j);
                this->alpha_bottom_cg += this->grid.search_vector(i,j)*this->grid.Asearch_vector(i,j);
            }
        }
        this->alpha_cg = this->alpha_top_cg/this->alpha_bottom_cg;

        // Update pressure and new residual
        for (int i = 1; i < this->grid.imax + 1; i++) {
            for (int j = 1; j < this->grid.jmax + 1; j++) {
                this->grid.po(i,j) = this->grid.p(i,j); // smart residual preparation
                this->grid.p(i,j) += this->alpha_cg*this->grid.search_vector(i,j);
                this->grid.res(i,j) -= this->alpha_cg*this->grid.Asearch_vector(i,j);
                this->preconditioner.RHS(i,j) = this->grid.res(i,j);
            }
        }

        // Calculate norm of residual
        this->res_norm = this->grid.res.norm() / initial_res_norm;

        // Convergence check
        this->res_norm_over_it_with_pressure_solver(this->it) = this->res_norm;
        if (this->res_norm < this->eps) {
            this->it++;
            break;
        }
        
        this->inferenceExp1();

        // Calculate beta
        for (int i = 1; i < this->grid.imax + 1; i++) {
            for (int j = 1; j < this->grid.jmax + 1; j++) {
                this->beta_top_cg += this->preconditioner.p(i,j)*this->grid.res(i,j);
            }
        }
        this->beta_cg = this->beta_top_cg/this->alpha_top_cg;
        
        this->betas(this->it) = this->beta_cg;

        // Calculate new search vector
        for (int i = 1; i < this->grid.imax + 1; i++) {
            for (int j = 1; j < this->grid.jmax + 1; j++) {
                this->grid.search_vector(i,j) = this->preconditioner.p(i,j) + this->beta_cg*this->grid.search_vector(i,j);
            }
        }

        this->res_norm_over_it_with_pressure_solver(this->it) = this->res_norm;
        this->it++;
        this->n_cg++;
    }

    this->setBoundaryConditionsP();
    this->setBoundaryConditionsPGeometry();

}