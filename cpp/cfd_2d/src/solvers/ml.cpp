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

    this->preconditioner.p.setZero();
    for (int k = 0; k < 1; k++) {
        Multigrid::vcycle(this->multigrid_hierarchy_preconditioner, this->multigrid_hierarchy_preconditioner->numLevels() - 1, this->omg, 1);
    }
    std::string filename = "mgpcg_" + std::to_string(this->n_cg) + ".dat";
    Kernel::saveMatrix(filename.c_str(), &this->preconditioner.p);

    for(int i = 1; i < this->grid.imax + 1; i++) {
        for(int j = 1; j < this->grid.jmax + 1; j++) {
            this->preconditioner.p(i,j) = output_acc[i][j];
            this->grid.search_vector(i,j) = this->preconditioner.p(i,j);
        }
    }
    filename = "ml_" + std::to_string(this->n_cg) + ".dat";
    Kernel::saveMatrix(filename.c_str(), &this->preconditioner.p);
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

    float htop = 1.0;
    float hbottom = 1.0;

    // List of Eigen::MatrixXf
    std::vector<Eigen::MatrixXf> p_list;

    while ((this->res_norm > this->eps || this->res_norm == 0) && this->n_cg < this->maxiterations_cg) {
        this->setBoundaryConditionsP();
        this->setBoundaryConditionsPGeometry();
        this->alpha_top_cg = 0.0;
        this->alpha_bottom_cg = 0.0;
        this->beta_top_cg= 0.0;
        this->res_norm = 0.0;

        if (this->n_cg < this->num_sweeps) {
            this->inferenceExp1();
        }
        else {
            this->preconditioner.p.setZero();
            for (int k = 0; k < 1; k++) {
                Multigrid::vcycle(this->multigrid_hierarchy_preconditioner, this->multigrid_hierarchy_preconditioner->numLevels() - 1, this->omg, 1);
            }
            for (int i = 1; i < this->grid.imax + 1; i++) {
                for (int j = 1; j < this->grid.jmax + 1; j++) {
                    this->grid.search_vector(i,j) = this->preconditioner.p(i,j);
                }
            }
        }
        if (this->save_ml && this->n_cg == 0) {
            this->saveMLData();
        }
        p_list.push_back(this->grid.search_vector);

        for (int v = this->n_cg - 2; v < this->n_cg; v++) {
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

        // Update pressure and new residual
        for (int i = 1; i < this->grid.imax + 1; i++) {
            for (int j = 1; j < this->grid.jmax + 1; j++) {
                this->grid.p(i,j) += this->alpha_cg*p_list[n_cg](i,j);
                this->grid.res(i,j) -= this->alpha_cg*this->grid.Asearch_vector(i,j);
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

        this->it++;
        this->n_cg++;
    }
    this->n_cg_over_it(this->it_wo_pressure_solver) = this->n_cg;
    this->setBoundaryConditionsP();
    this->setBoundaryConditionsPGeometry();

}