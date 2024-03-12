#include <filesystem>
#include <iostream>
#include <chrono>
#include "cfd.h"

namespace CFD {
    namespace ME_X {
        // Slide 15
        float uu_x(StaggeredGrid *grid, FluidSimulation *sim, int i, int j) {
            return (
                (
                    (1/grid->dx)*(pow((grid->u(i,j) + grid->u(i+1,j))/2,2) - pow((grid->u(i-1,j)+grid->u(i,j))/2,2))
                )
                +
                (
                    (sim->alpha/grid->dx)*(std::abs(grid->u(i,j) + grid->u(i+1,j))*(grid->u(i,j) - grid->u(i+1,j))/4 - std::abs(grid->u(i-1,j) + grid->u(i,j))*(grid->u(i-1,j) - grid->u(i,j))/4)
                )
            );
        }

        float uv_y(StaggeredGrid *grid, FluidSimulation *sim, int i, int j) {
            return (
                (
                    (1/grid->dy)*((grid->v(i,j) + grid->v(i+1,j))*(grid->u(i,j) + grid->u(i,j+1))/4 - (grid->v(i,j-1) + grid->v(i+1,j-1))*(grid->u(i,j-1) + grid->u(i,j))/4)
                )
                +
                (
                    (sim->alpha/grid->dy)*(std::abs(grid->v(i,j) + grid->v(i+1,j))*(grid->u(i,j) - grid->u(i,j+1))/4 - std::abs(grid->v(i,j-1) + grid->v(i+1,j-1))*(grid->u(i,j-1) - grid->u(i,j))/4)
                )
            );
        }

        float uu_xx(StaggeredGrid *grid, FluidSimulation *sim, int i, int j) {
            return (
                (grid->u(i+1,j) - 2*grid->u(i,j) + grid->u(i-1,j))/pow(grid->dx, 2)
            );
        }

        float uu_yy(StaggeredGrid *grid, FluidSimulation *sim, int i, int j) {
            return (
                (grid->u(i,j+1) - 2*grid->u(i,j) + grid->u(i,j-1))/pow(grid->dy, 2)
            );
        }

        float p_x(StaggeredGrid *grid, FluidSimulation *sim, int i, int j) {
            return (
                (grid->p(i+1,j) - grid->p(i,j))/grid->dx
            );
        }
    }

    namespace ME_Y {
        float uv_x(StaggeredGrid *grid, FluidSimulation *sim, int i, int j) {
            return (
                (
                    (1/grid->dx)*((grid->u(i,j) + grid->u(i,j+1))*(grid->v(i,j) + grid->v(i+1,j))/4 - (grid->u(i-1,j) + grid->u(i-1,j+1))*(grid->v(i-1,j) + grid->v(i,j))/4)
                )
                +
                (
                    (sim->alpha/grid->dx)*(std::abs(grid->u(i,j) + grid->u(i,j+1))*(grid->v(i,j) - grid->v(i+1,j))/4 - std::abs(grid->u(i-1,j) + grid->u(i-1,j+1))*(grid->v(i-1,j) - grid->v(i,j))/4)
                )
            );
        }

        float vv_y(StaggeredGrid *grid, FluidSimulation *sim, int i, int j) {
            return (
                (
                    (1/grid->dy)*(pow((grid->v(i,j) + grid->v(i,j+1))/2,2) - pow((grid->v(i,j-1)+grid->v(i,j))/2,2))
                )
                +
                (
                    (sim->alpha/grid->dy)*(std::abs(grid->v(i,j) + grid->v(i,j+1))*(grid->v(i,j) - grid->v(i,j+1))/4 - std::abs(grid->v(i,j-1) + grid->v(i,j))*(grid->v(i,j-1) - grid->v(i,j))/4)
                )
            );
        }

        float vv_xx(StaggeredGrid *grid, FluidSimulation *sim, int i, int j) {
            return (
                (grid->v(i+1,j) - 2*grid->v(i,j) + grid->v(i-1,j))/pow(grid->dx, 2)
            );
        }

        float vv_yy(StaggeredGrid *grid, FluidSimulation *sim, int i, int j) {
            return (
                (grid->v(i,j+1) - 2*grid->v(i,j) + grid->v(i,j-1))/pow(grid->dy, 2)
            );
        }

        float p_y(StaggeredGrid *grid, FluidSimulation *sim, int i, int j) {
            return (
                (grid->p(i,j+1) - grid->p(i,j))/grid->dy
            );
        }
    }

    namespace CE {
        float u_x(StaggeredGrid *grid, FluidSimulation *sim, int i, int j) {
            return (
                (grid->u(i,j) - grid->u(i-1,j))/(grid->dx)
            );
        }

        float v_y(StaggeredGrid *grid, FluidSimulation *sim, int i, int j) {
            return (
                (grid->v(i,j) - grid->v(i,j-1))/(grid->dy)
            );
        }
    }

    float StaggeredGrid::findMaxAbsoluteU() const {
        float max = 0.0;
        for (int j = 1; j < jmax + 2; j++) {
            for (int i = 1; i < imax + 1; i++) {
                if (std::abs(u(i, j)) > max) {
                    max = std::abs(u(i, j));
                }
            }
        }
        return max;
    }
    float StaggeredGrid::findMaxAbsoluteV() const {
        float max = 0.0;
        for (int j = 1; j < jmax + 1; j++) {
            for (int i = 1; i < imax + 2; i++) {
                if (std::abs(v(i, j)) > max) {
                    max = std::abs(v(i, j));
                }
            }
        }
        return max;
    }
    void StaggeredGrid::interpolateVelocity() {
        for (int j = 1; j < jmax + 1; j++) { // Start from 1 to exclude boundary cells
            for (int i = 1; i < imax + 1; i++) { // Start from 1 to exclude boundary cells
                // Interpolate u velocity component at half-integer grid points
                u_interpolated(i, j) = (u(i, j) + u(i, j+1)) / 2;
                // Interpolate v velocity component at half-integer grid points
                v_interpolated(i, j) = (v(i, j) + v(i+1, j)) / 2;

                // Fine interpolation along x direction
                double delta_x = 0.5 * (u(i, j) + u(i, j+1) - u(i+1, j) - u(i+1, j+1));
                u_interpolated(i + 1,j) = u(i, j+1) + delta_x;
                
                // Fine interpolation along y direction
                double delta_y = 0.5 * (v(i, j) + v(i+1, j) - v(i, j+1) - v(i+1, j+1));
                v_interpolated(i, j + 1) = v(i+1, j) + delta_y;
            }
        }
    }
    void FluidSimulation::resetPressure() {
        for (int j = 0; j < this->grid.jmax + 2; j++) {
            for (int i = 0; i < this->grid.imax + 2; i++) {
                this->grid.p(i, j) = 0;
            }
        }
    }
    void FluidSimulation::selectDtAccordingToStabilityCondition() {
        float left = (this->Re/2) * pow(((1/pow(this->grid.dx, 2))+(1/pow(this->grid.dy, 2))), -1);
        float middle = this->grid.dx / this->grid.findMaxAbsoluteU();
        float right = this->grid.dy / this->grid.findMaxAbsoluteV();
        this->dt = this->tau * std::min(std::min(left, middle), right);
    }

    void FluidSimulation::computeF() {
        for (int j = 1; j < this->grid.jmax + 2; j++) {
            for (int i = 1; i < this->grid.imax + 1; i++) {
                this->grid.F(i,j) = this->grid.u(i,j) + this->dt * (
                    (1/this->Re) * (ME_X::uu_xx(&this->grid, this, i, j) + ME_X::uu_yy(&this->grid, this, i, j)) 
                    - 
                    ME_X::uu_x(&this->grid, this, i, j)
                    -
                    ME_X::uv_y(&this->grid, this, i, j)
                );
            }
        }
    }

    void FluidSimulation::computeG() {
        for (int j = 1; j < this->grid.jmax + 1; j++) {
            for (int i = 1; i < this->grid.imax + 2; i++) {
                this->grid.G(i,j) = this->grid.v(i,j) + this->dt * (
                    (1/this->Re) * (ME_Y::vv_xx(&this->grid, this, i, j) + ME_Y::vv_yy(&this->grid, this, i, j)) 
                    - 
                    ME_Y::uv_x(&this->grid, this, i, j)
                    -
                    ME_Y::vv_y(&this->grid, this, i, j)
                );
            }
        }
    }

    void FluidSimulation::computeRHS() {
        for (int j = 1; j < this->grid.jmax + 1; j++) {
            for (int i = 1; i < this->grid.imax + 1; i++) {
                this->grid.RHS(i,j) = (1/this->dt)*((this->grid.F(i,j) - this->grid.F(i-1,j))/this->grid.dx+(this->grid.G(i,j) - this->grid.G(i,j-1))/this->grid.dy);
            }
        }
    }

    void FluidSimulation::computeDiscreteL2Norm() {
        // Extract grid dimensions
        int imax = this->grid.imax;
        int jmax = this->grid.jmax;

        // Compute the difference between po and p
        Eigen::MatrixXf diff = this->grid.po.block(1, 1, imax, jmax) - this->grid.p.block(1, 1, imax, jmax);

        // Element-wise square of differences
        Eigen::MatrixXf squared_diff = diff.array().square();

        // Sum of squared differences
        float sum_squared_diff = squared_diff.sum();

        // Multiply by dx and dy to get the integral
        this->res_norm = std::sqrt(sum_squared_diff * this->grid.dx * this->grid.dy);
    }

    void FluidSimulation::computeU() {
        for (int j = 1; j < this->grid.jmax + 2; j++) {
            for (int i = 1; i < this->grid.imax + 1; i++) {
                this->grid.u(i,j) = this->grid.F(i,j) - (this->dt/this->grid.dx) * (this->grid.p(i+1,j) - this->grid.p(i,j));
            }
        }
    }

    void FluidSimulation::computeV() {
        for (int j = 1; j < this->grid.jmax + 1; j++) {
            for (int i = 1; i < this->grid.imax + 2; i++) {
                this->grid.v(i,j) = this->grid.G(i,j) - (this->dt/this->grid.dy) * (this->grid.p(i,j+1) - this->grid.p(i,j));
            }
        }
    }

    void FluidSimulation::setBoundaryConditionsPGeometry() {
        // Geometry boundaries
        float tmp_p = 0.0;
        int counter = 0;
        for (int j = 1; j < this->grid.jmax + 1; j++) {
            for (int i = 1; i < this->grid.imax + 1; i++) {
                tmp_p = 0.0;
                counter = 0;
                // check if is obstacle
                if (isObstacleCell(i, j)) {
                    // obstacle cell
                    if (this->grid.flag_field(i, j) & FlagFieldMask::FLUID_NORTH) {
                        tmp_p += this->grid.p(i, j+1);
                        counter++;
                    }
                    if (this->grid.flag_field(i, j) & FlagFieldMask::FLUID_SOUTH) {
                        tmp_p += this->grid.p(i, j-1);
                        counter++;
                    }
                    if (this->grid.flag_field(i, j) & FlagFieldMask::FLUID_WEST) {
                        tmp_p += this->grid.p(i-1, j);
                        counter++;
                    }
                    if (this->grid.flag_field(i, j) & FlagFieldMask::FLUID_EAST) {
                        tmp_p += this->grid.p(i+1, j);
                        counter++;
                    }
                    if (counter > 0) {
                        this->grid.p(i, j) = tmp_p / counter;
                    }
                }
            }
        }
    }

    void FluidSimulation::setBoundaryConditionsVelocityGeometry() {
        for (int j = 1; j < this->grid.jmax + 1; j++) {
            for (int i = 1; i < this->grid.imax + 1; i++) {
                if (isObstacleCell(i, j)) {
                    if (isCornerCell(i, j)) {
                        handleCornerCell(i, j);
                    } else {
                        handleObstacleCell(i, j);
                    }
                }
            }
        }
    }

    bool FluidSimulation::isObstacleCell(int i, int j) {
        return (this->grid.flag_field(i, j) & FlagFieldMask::MASK_CELL_TYPE) == FlagFieldMask::CELL_OBSTACLE;
    }

    bool FluidSimulation::isCornerCell(int i, int j) {
        return (this->grid.flag_field(i, j) & (FlagFieldMask::FLUID_NORTH | FlagFieldMask::FLUID_SOUTH)) &&
            (this->grid.flag_field(i, j) & (FlagFieldMask::FLUID_EAST | FlagFieldMask::FLUID_WEST));
    }

    void FluidSimulation::handleCornerCell(int i, int j) {
        this->grid.u(i, j) = 0.0;
        this->grid.v(i, j) = 0.0;
        this->grid.F(i, j) = 0.0;
        this->grid.G(i, j) = 0.0;
        this->grid.u(i-1, j) = -this->grid.u(i-1, j + (i == this->grid.imax ? -1 : 1));
        this->grid.v(i, j-1) = -this->grid.v(i + (j == this->grid.jmax ? -1 : 1), j-1);
        this->grid.F(i-1, j) = -this->grid.F(i-1, j + (i == this->grid.imax ? -1 : 1));
        this->grid.G(i, j-1) = -this->grid.G(i + (j == this->grid.jmax ? -1 : 1), j-1);
    }

    void FluidSimulation::handleObstacleCell(int i, int j) {
        if (this->grid.flag_field(i, j) & FlagFieldMask::FLUID_NORTH) {
            this->grid.u(i, j) = -this->grid.u(i, j + 1);
            this->grid.u(i-1, j) = -this->grid.u(i-1, j + 1);
            this->grid.v(i, j) = 0.0;

            this->grid.F(i, j) = -this->grid.F(i, j + 1);
            this->grid.F(i-1, j) = -this->grid.F(i-1, j + 1);
            this->grid.G(i, j) = 0.0;
        } else if (this->grid.flag_field(i, j) & FlagFieldMask::FLUID_SOUTH) {
            this->grid.u(i, j) = -this->grid.u(i, j - 1);
            this->grid.u(i-1, j) = -this->grid.u(i-1, j - 1);
            this->grid.v(i, j) = 0.0;

            this->grid.F(i, j) = -this->grid.F(i, j - 1);
            this->grid.F(i-1, j) = -this->grid.F(i-1, j - 1);
            this->grid.G(i, j) = 0.0;
        } else if (this->grid.flag_field(i, j) & FlagFieldMask::FLUID_WEST) {
            this->grid.v(i, j-1) = -this->grid.v(i-1, j-1);
            this->grid.v(i, j) = -this->grid.v(i-1, j);
            this->grid.u(i-1, j) = 0.0;

            this->grid.G(i, j-1) = -this->grid.G(i-1, j-1);
            this->grid.G(i, j) = -this->grid.G(i-1, j);
            this->grid.F(i-1, j) = 0.0;
        } else if (this->grid.flag_field(i, j) & FlagFieldMask::FLUID_EAST) {
            this->grid.v(i, j-1) = -this->grid.v(i + 1, j - 1);
            this->grid.v(i, j) = -this->grid.v(i + 1, j);
            this->grid.u(i + 1, j) = 0.0;

            this->grid.G(i, j-1) = -this->grid.G(i + 1, j - 1);
            this->grid.G(i, j) = -this->grid.G(i + 1, j);
            this->grid.F(i + 1, j) = 0.0;
        } else {
            // interior obstacle cell, so no-slip
            this->grid.u(i,j) = 0.0;
            this->grid.v(i,j) = 0.0;

            this->grid.F(i,j) = 0.0;
            this->grid.G(i,j) = 0.0;
        }
    }

    void FluidSimulation::loadTorchScriptModel(const std::string& modelPath) {
        try {
            // Deserialize the ScriptModule from a file using torch::jit::load().
            this->model = torch::jit::load(modelPath);
        } catch (const c10::Error& e) {
            std::cerr << "Error loading the model\n";
            // print the error and stack trace
            std::cerr << e.what();
        }
    }

    void FluidSimulation::saveMLData() {
        // check if directory exists
        if (!std::filesystem::exists("ML_data")) {
            std::filesystem::create_directory("ML_data");
        }
        std::string filename = "ML_data/res_" + std::to_string(this->current_file_number) + ".dat";
        Kernel::saveMatrix(filename.c_str(), &this->preconditioner.RHS);
        filename = "ML_data/e_" + std::to_string(this->current_file_number) + ".dat";
        Kernel::saveMatrix(filename.c_str(), &this->preconditioner.p);
        this->current_file_number++;
    }

    void FluidSimulation::run() {
        float last_saved = 0.0;
        std::string solver_name = "";

        if (!this->no_vtk) {
            saveVTKGeometry(this);
        }

        // Function pointer to solver
        void (CFD::FluidSimulation::*pressure_solver)();

        int imax_levels = std::log2(this->grid.imax);
        int jmax_levels = std::log2(this->grid.jmax);
        int levels = std::min(imax_levels, jmax_levels);
        this->preconditioner = StaggeredGrid(this->grid.imax, this->grid.jmax, this->grid.xlength, this->grid.ylength);
        this->multigrid_hierarchy = new MultigridHierarchy(levels, &this->grid);
        this->multigrid_hierarchy_preconditioner = new MultigridHierarchy(levels, &this->preconditioner);

        if (this->solver_type == SolverType::JACOBI) {
            pressure_solver = &FluidSimulation::solveWithJacobi;
            solver_name = "Jacobi";
            std::cout << "Solver: Jacobi (" << this->grid.imax << "x" << this->grid.jmax << ")" << std::endl;
        }
        else if (this->solver_type == SolverType::MULTIGRID_JACOBI) {
            pressure_solver = &FluidSimulation::solveWithMultigridJacobi;
            solver_name = "Multigrid Jacobi";
            std::cout << "Solver: Multigrid Jacobi (" << this->grid.imax << "x" << this->grid.jmax << ")" << std::endl;
        }
        else if (this->solver_type == SolverType::CONJUGATE_GRADIENT) {
            pressure_solver = &FluidSimulation::solveWithConjugateGradient;
            solver_name = "Conjugate Gradient";
            std::cout << "Solver: Conjugate Gradient (" << this->grid.imax << "x" << this->grid.jmax << ")" << std::endl;
        }
        else if (this->solver_type == SolverType::MGPCG) {
            pressure_solver = &FluidSimulation::solveWithMultigridPCG;
            solver_name = "Multigrid PCG";
            std::cout << "Solver: Multigrid PCG (" << this->grid.imax << "x" << this->grid.jmax << ")" << std::endl;
        }
        else if (this->solver_type == SolverType::MGPCG_FASTER) {
            pressure_solver = &FluidSimulation::solveWithMultigridPCGFaster;
            solver_name = "Multigrid PCG (Faster)";
            std::cout << "Solver: Multigrid PCG Faster (" << this->grid.imax << "x" << this->grid.jmax << ")" << std::endl;
        }
        else if (this->solver_type == SolverType::ML) {
            pressure_solver = &FluidSimulation::solveWithML;
            loadTorchScriptModel(this->ml_model_path);
            solver_name = "ML";
            std::cout << "Solver: ML (" << this->grid.imax << "x" << this->grid.jmax << ")" << std::endl;
        }
        else {
            throw std::invalid_argument("Invalid solver type");
        }


        while(this->t < this->t_end) {
            this->lastTimestamp = static_cast<int>(std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count());
            this->selectDtAccordingToStabilityCondition();
            this->setBoundaryConditionsU();
            this->setBoundaryConditionsV();
            this->setBoundaryConditionsVelocityGeometry();
            this->computeF();
            this->computeG();
            this->setBoundaryConditionsU();
            this->setBoundaryConditionsV();
            this->setBoundaryConditionsVelocityGeometry();
            this->computeRHS();
            
            (this->*pressure_solver)();

            this->res_norm_over_it_without_pressure_solver(this->it_wo_pressure_solver) = this->res_norm;

            this->computeU();
            this->computeV();
            std::cout << "Solver: " << solver_name << "\t" << " t: " << this->t << "\t" << " dt: " << this->dt << "\t" << " res: " << this->res_norm << "\t" << " p-norm: " << this->p_norm << "\t" << " it: " << this->it << "\t" << " it_wo_pressure_solver: " << this->it_wo_pressure_solver << "\t" << " n_cg: " << this->n_cg << "\t" << " duration: " << this->duration << std::endl;
            if (this->t - last_saved >= this->save_interval) {
                this->grid.interpolateVelocity();
                if (!this->no_vtk) {
                    saveVTK(this);
                }
                last_saved += this->save_interval;
            }
            this->duration += static_cast<int>(std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count()) - this->lastTimestamp;
            this->res_norm_over_time(this->duration) = this->res_norm;
            this->t = this->t + this->dt;
            this->it_wo_pressure_solver++;
        }

        this->setBoundaryConditionsU();
        this->setBoundaryConditionsV();
        this->setBoundaryConditionsVelocityGeometry();

        this->grid.interpolateVelocity();

        this->setBoundaryConditionsP();
        this->setBoundaryConditionsPGeometry();

        this->res_norm_over_it_with_pressure_solver.conservativeResize(this->it);
        this->res_norm_over_it_without_pressure_solver.conservativeResize(this->it_wo_pressure_solver);
        this->res_norm_over_time.conservativeResize(this->duration);
        this->betas.conservativeResize(this->it);

        return;
    }

    void FluidSimulation::saveData() {
        Kernel::saveMatrix("u.dat", &this->grid.u_interpolated);
        Kernel::saveMatrix("v.dat", &this->grid.v_interpolated);
        Kernel::saveMatrix("p.dat", &this->grid.p);
        Kernel::saveVector("residuals_with_pressure_solver.dat", &this->res_norm_over_it_with_pressure_solver);
        Kernel::saveVector("residuals_without_pressure_solver.dat", &this->res_norm_over_it_without_pressure_solver);
        Kernel::saveVector("residuals_over_time.dat", &this->res_norm_over_time);
        Kernel::saveVector("betas.dat", &this->betas);
    }
}