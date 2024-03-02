#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "kernel.h"
#include "argparse.h"
#include "staggered_grid.h"
#include "multigrid.h"

namespace CFD {
    using namespace Eigen;

    enum FlagFieldMask {
        CELL_FLUID = 0b11111,
        FLUID_WEST = 0b00100,
        FLUID_EAST = 0b01000,
        FLUID_SOUTH = 0b00010,
        FLUID_NORTH = 0b00001,
        CELL_OBSTACLE = 0b00000,
        OBSTACLE_WEST = 0b11011,
        OBSTACLE_EAST = 0b11101,
        OBSTACLE_SOUTH = 0b11110,
        OBSTACLE_NORTH = 0b11100,
        MASK_CELL_TYPE = 0b10000,
        MASK_WEST = 0b00100,
        MASK_EAST = 0b01000,
        MASK_SOUTH = 0b00010,
        MASK_NORTH = 0b00001
    };

    enum SolverType {
        JACOBI,
        MULTIGRID_JACOBI,
        CONJUGATED_GRADIENT,
        MULTIGRID_PCG,
        ML
    };
    SolverType convertSolverType(const std::string& solver);

    class FluidParams {
        public:
            FluidParams(const std::string name, int argc, char* argv[]);
            int imax = 100;
            int jmax = 100;
            float xlength = 1.0;
            float ylength = 1.0;
            float t_end = 5.0;
            float tau = 0.5;
            float eps = 1e-3;
            float omg = 1.0;
            float alpha = 0.9;
            float Re = 100.0;
            float t = 0;
            float dt = 0.05;
            float save_interval = 0.5;
            bool save_ml = false;
            bool no_vtk = false;
            SolverType solver_type = SolverType::JACOBI;

            argparse::ArgumentParser argument_parser;
    };

    class FluidSimulation {
        public:
            FluidSimulation(const FluidParams& params) {
                imax = params.imax;
                jmax = params.jmax;
                xlength = params.xlength;
                ylength = params.ylength;
                t_end = params.t_end;
                tau = params.tau;
                eps = params.eps;
                omg = params.omg;
                alpha = params.alpha;
                Re = params.Re;
                t = params.t;
                dt = params.dt;
                solver_type = params.solver_type;
                save_interval = params.save_interval;
                res_norm = 0.0;
                e_norm = 0.0;
                multigrid_hierarchy = nullptr;
                res_norm_over_it_with_pressure_solver = VectorXd::Zero(1e8);
                res_norm_over_it_without_pressure_solver = VectorXd::Zero(1e8);
                res_norm_over_time = VectorXd::Zero(1e8);
                betas = VectorXd::Zero(1e8);
                save_ml = params.save_ml;
                no_vtk = params.no_vtk;
                maxiterations_cg = std::max(imax, jmax);
            }
            int imax;
            int jmax;
            float xlength;
            float ylength;
            float t_end;
            float tau;
            float eps;
            float omg;
            float alpha;
            float Re;
            float t;
            float dt;
            float p_norm;
            float e_norm;
            float res_norm;
            int it = 0;
            int it_wo_pressure_solver = 0;
            int lastTimestamp = 0;
            int duration = 0;
            StaggeredGrid grid;
            Kernel::Timer timer;
            SolverType solver_type;
            float save_interval;
            bool save_ml;
            bool no_vtk;
            VectorXd res_norm_over_it_with_pressure_solver;
            VectorXd res_norm_over_it_without_pressure_solver;
            VectorXd res_norm_over_time;
            VectorXd betas;

            // Conjugate Gradient components
            int n_cg = 0;
            int maxiterations_cg;
            float alpha_cg = 0.0;
            float alpha_top_cg = 0.0;
            float alpha_bottom_cg = 0.0;
            float beta_cg = 0.0;
            float beta_top_cg = 0.0;

            // Multigrid components
            MultigridHierarchy *multigrid_hierarchy;

            // Preconditioner Conjugated Gradient components
            MultigridHierarchy *multigrid_hierarchy_preconditioner;
            StaggeredGrid preconditioner;

            // Deep Learning
            torch::jit::script::Module model;

            void resetPressure();
            void selectDtAccordingToStabilityCondition();
            void computeF();
            void computeG();
            void computeRHS();
            void solveWithJacobi();
            void solveWithMultigridJacobi();
            void solveWithConjugatedGradient();
            void solveWithMultigridPCG();
            void solveWithML();
            void computeDiscreteL2Norm();
            void computeU();
            void computeV();
            void run();
            void saveData();
            void saveMLData();
            int current_file_number = 0;

            // Local functions
            bool isObstacleCell(int i, int j);
            bool isCornerCell(int i, int j);
            void handleCornerCell(int i, int j);
            void handleObstacleCell(int i, int j);

            // Deep Learning
            void loadTorchScriptModel(const std::string& modelPath);
            void inferenceExp1();

            // Virtual functions
            virtual void setBoundaryConditionsU() = 0;
            virtual void setBoundaryConditionsV() = 0;
            virtual void setBoundaryConditionsP() = 0;
            virtual void setBoundaryConditionsPGeometry();
            virtual void setBoundaryConditionsVelocityGeometry();
            virtual void setBoundaryConditionsInterpolatedVelocityGeometry();

            virtual ~FluidSimulation() = default;
    };

    // Functions
    void saveVTK(FluidSimulation* sim);
    void saveVTKGeometry(FluidSimulation* sim);
}
