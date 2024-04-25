#include "cfd.h"
#include <limits>

using namespace CFD;


void FluidSimulation::inferenceExp1() {
    // Calculate the error correction with deep learning
    for(int i = 1; i < this->grid.imax + 1; i++) {
        for(int j = 1; j < this->grid.jmax + 1; j++) {
            this->grid.input_ml[j][i] = this->preconditioner.RHS(j,i);
        }
    }
    this->grid.output_ml = this->model.forward({ this->grid.input_ml.to(torch::kCUDA).unsqueeze(0).unsqueeze(0) }).toTensor().to(torch::kCPU).squeeze(0).squeeze(0);

    auto output_acc = this->grid.output_ml.accessor<float,2>();

    for (int i = 1; i < this->grid.imax + 1; i++) {
        for (int j = 1; j < this->grid.jmax + 1; j++) {
            // ML inference
            this->grid.search_vector(i,j) = output_acc[i][j];
        }
    }

    /*std::string filename = "ml_" + std::to_string(this->n_cg) + ".dat";
    Kernel::saveMatrix(filename.c_str(), &this->preconditioner.p);

    this->preconditioner.p.setZero();
    for (int k = 0; k < this->num_sweeps; k++) {
        Multigrid::vcycle(this->multigrid_hierarchy_preconditioner, this->multigrid_hierarchy_preconditioner->numLevels() - 1, this->omg, this->num_sweeps);
    }

    filename = "mgpcg_" + std::to_string(this->n_cg) + ".dat";
    Kernel::saveMatrix(filename.c_str(), &this->preconditioner.p);

    filename = "RHS_" + std::to_string(this->n_cg) + ".dat";
    Kernel::saveMatrix(filename.c_str(), &this->preconditioner.RHS);*/
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

    this->computeResidualNorm();
    float previous_res_norm = this->res_norm;
    float res_norm_ratio = 0.0;

    // List of Eigen::MatrixXf
    std::vector<Eigen::MatrixXf> p_list;

    int n_ml = 0;
    int n_mgpcg = 0;

    while ((this->res_norm > this->eps || this->res_norm == 0) && this->n_cg < this->maxiterations_cg) {
        this->setBoundaryConditionsP();
        this->setBoundaryConditionsPGeometry();
        this->alpha_top_cg = 0.0;
        this->alpha_bottom_cg = 0.0;
        this->beta_top_cg= 0.0;
        this->res_norm = 0.0;

        if (this->n_cg >= 0) {
            this->inferenceExp1();
            n_ml++;
        }
        else {
            this->preconditioner.p.setZero();
            for (int k = 0; k < this->num_sweeps; k++) {
                Multigrid::vcycle(this->multigrid_hierarchy_preconditioner, this->multigrid_hierarchy_preconditioner->numLevels() - 1, this->omg, this->num_sweeps);
            }
            for (int i = 1; i < this->grid.imax + 1; i++) {
                for (int j = 1; j < this->grid.jmax + 1; j++) {
                    this->grid.search_vector(i,j) = this->preconditioner.p(i,j);
                }
            }
            n_mgpcg++;
        }
        if (this->save_ml) {
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

        // Update pressure
        for (int i = 1; i < this->grid.imax + 1; i++) {
            for (int j = 1; j < this->grid.jmax + 1; j++) {
                this->grid.p(i,j) += this->alpha_cg*p_list[n_cg](i,j);
            }
        }

        // New residual
        for (int i = 1; i < this->grid.imax + 1; i++) {
            for (int j = 1; j < this->grid.jmax + 1; j++) {
                this->grid.res(i,j) = this->grid.RHS(i,j) - (
                    (1/this->grid.dx2)*(this->grid.p(i+1,j) - 2*this->grid.p(i,j) + this->grid.p(i-1,j)) +
                    (1/this->grid.dy2)*(this->grid.p(i,j+1) - 2*this->grid.p(i,j) + this->grid.p(i,j-1))
                );
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
    }
    std::cout << "ML: " << n_ml << " MGPCG: " << n_mgpcg << std::endl;
    this->n_cg_over_it(this->it_wo_pressure_solver) = this->n_cg;
    this->setBoundaryConditionsP();
    this->setBoundaryConditionsPGeometry();

}