#include "cfd.h"

using namespace CFD;

void FluidSimulation::solveWithML() {

    this->setBoundaryConditionsP();
    this->setBoundaryConditionsPGeometry();

    // reset norm check
    this->res_norm = 0.0;
    
    auto options = torch::TensorOptions().dtype(torch::kFloat);
    torch::Tensor T = torch::from_blob(this->grid.RHS.data(), {this->grid.RHS.rows(), this->grid.RHS.cols()}, options).clone();
    T = T.to(torch::kCUDA);
    T = T.unsqueeze(0);
    auto output = this->model.forward({ T }).toTensor();
    output = output.to(torch::kCPU).squeeze(0);
    float* output_ptr = output.data_ptr<float>();
    for (int i = 0; i < output.size(0); i++) {
        for (int j = 0; j < output.size(1); j++) {
            this->grid.p(i, j) = output_ptr[i * output.size(1) + j];
        }
    }
    this->it++;

    this->setBoundaryConditionsP();
    this->setBoundaryConditionsPGeometry();

}