#include "cfd.h"

using namespace CFD;


void FluidSimulation::inferenceExp1() {
    // Calculate the error correction with deep learning
    auto options = torch::TensorOptions().dtype(torch::kFloat);
    torch::Tensor T = torch::zeros({this->grid.jmax+2, this->grid.imax+2}, options);
    for(int i = 1; i < this->grid.imax + 1; i++) {
        for(int j = 1; j < this->grid.jmax + 1; j++) {
            T[j][i] = this->preconditioner.RHS(j,i);
        }
    }
    T = T.to(torch::kCUDA);
    T = T.unsqueeze(0).unsqueeze(0);
    auto output = this->model.forward({ T }).toTensor();
    output = output.to(torch::kCPU).squeeze(0).squeeze(0);

    auto output_acc = output.accessor<float,2>();

    for(int i = 1; i < this->grid.imax + 1; i++) {
        for(int j = 1; j < this->grid.jmax + 1; j++) {
            this->preconditioner.p(i,j) = this->safety_factor*output_acc[j][i];
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
            this->grid.res(i,j) = this->grid.RHS(i,j);
            this->preconditioner.RHS(i,j) = this->grid.res(i,j);
        }
    }

    // Initial guess for error vector
    this->inferenceExp1();

    for(int i = 1; i < this->grid.imax + 1; i++) {
        for(int j = 1; j < this->grid.jmax + 1; j++) {
            this->grid.search_vector(i,j) = this->preconditioner.p(i,j);
        }
    }

    while ((this->res_norm > this->eps || this->res_norm == 0) && this->n_cg < this->maxiterations_cg) {
        this->setBoundaryConditionsP();
        this->setBoundaryConditionsPGeometry();
        this->alpha_top_cg = 0.0;
        this->alpha_bottom_cg = 0.0;
        this->beta_top_cg= 0.0;
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
                this->grid.p(i,j) += this->alpha_cg*this->grid.search_vector(i,j);
                this->grid.res(i,j) -= this->alpha_cg*this->grid.Asearch_vector(i,j);
                this->preconditioner.p(i,j) = this->grid.res(i,j);
                this->preconditioner.RHS(i,j) = this->grid.res(i,j);
            }
        }

        // Calculate norm of residual
        this->computeResidualNorm();

        // Convergence check
        this->res_norm_over_it_with_pressure_solver(this->it) = this->res_norm;
        if (this->res_norm < this->eps) {
            this->it++;
            break;
        }

        // New guess for error vector
        this->inferenceExp1();

        // Calculate beta
        for (int i = 1; i < this->grid.imax + 1; i++) {
            for (int j = 1; j < this->grid.jmax + 1; j++) {
                this->beta_top_cg += this->preconditioner.p(i,j)*this->grid.res(i,j); // Orthogonality of b - Mx
            }
        }
        this->beta_cg = this->beta_top_cg/this->alpha_top_cg;

        // Calculate new search vector
        for (int i = 1; i < this->grid.imax + 1; i++) {
            for (int j = 1; j < this->grid.jmax + 1; j++) {
                this->grid.search_vector(i,j) = this->preconditioner.p(i,j) + this->beta_cg*this->grid.search_vector(i,j);
            }
        }

        this->it++;
        this->n_cg++;
    }
    this->n_cg_over_it(this->it_wo_pressure_solver) = this->n_cg;
    this->setBoundaryConditionsP();
    this->setBoundaryConditionsPGeometry();

}