#include "cfd.h"
#include "mpi.h"

using namespace CFD;

void FluidSimulation::solveWithMultigridJacobi() {
    // reset norm check
    this->res_norm = -1;
    this->n_cg = 0;
    int local_done = 0;
    int global_done = 0;
    bool all_equal = true;

    this->resetPressure();

    while ((this->res_norm > this->eps || this->res_norm == -1) && this->n_cg < this->maxiterations_cg) {
        MPI_Allreduce(&local_done, &global_done, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        this->setBoundaryConditionsP();
        this->setBoundaryConditionsPGeometry();

        Multigrid::vcycle(this->multigrid_hierarchy, this->multigrid_hierarchy->numLevels() - 1, this->omg, this->num_sweeps);

        this->computeResidual();
        this->res_norm_over_it_with_pressure_solver(this->it) = this->res_norm;
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
    local_done = 1;
    while (global_done != this->world_size) {
        MPI_Allreduce(&local_done, &global_done, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        this->setBoundaryConditionsP();
    }
    this->setBoundaryConditionsP();
    this->setBoundaryConditionsPGeometry();
}