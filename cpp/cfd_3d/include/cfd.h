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
        CELL_FLUID = 0b1111111,
        FLUID_WEST = 0b0010000,
        FLUID_EAST = 0b0100000,
        FLUID_SOUTH = 0b0001000,
        FLUID_NORTH = 0b0000100,
        FLUID_BOTTOM = 0b0000010,
        FLUID_TOP = 0b0000001,
        CELL_OBSTACLE = 0b0000000,
        OBSTACLE_WEST = 0b1101110,
        OBSTACLE_EAST = 0b1110110,
        OBSTACLE_SOUTH = 0b1111100,
        OBSTACLE_NORTH = 0b1111000,
        OBSTACLE_BOTTOM = 0b1111010,
        OBSTACLE_TOP = 0b1111110,
        MASK_CELL_TYPE = 0b1000000,
        MASK_WEST = 0b0010000,
        MASK_EAST = 0b0100000,
        MASK_SOUTH = 0b0001000,
        MASK_NORTH = 0b0000100,
        MASK_BOTTOM = 0b0000010,
        MASK_TOP = 0b0000001
    };

    enum SolverType {
        JACOBI,
        MULTIGRID_JACOBI,
        CONJUGATED_GRADIENT,
        MULTIGRID_PCG
    };
    SolverType convertSolverType(const std::string& solver);

    class FluidParams {
    public:
        FluidParams(const std::string name, int argc, char* argv[]);
        int imax = 100;
        int jmax = 100;
        int kmax = 100;
        double xlength = 1.0;
        double ylength = 1.0;
        double zlength = 1.0;
        double t_end = 5.0;
        double tau = 0.5;
        double eps = 1e-3;
        double omg = 1.0;
        double alpha = 0.9;
        double Re = 100.0;
        double t = 0;
        double dt = 0.05;
        double save_interval = 0.5;
        bool save_hdf5 = false;
        SolverType solver_type = SolverType::JACOBI;

        argparse::ArgumentParser argument_parser;
    };

    class FluidSimulation {
    public:
        FluidSimulation(const FluidParams& params) {
            imax = params.imax;
            jmax = params.jmax;
            kmax = params.kmax;
            xlength = params.xlength;
            ylength = params.ylength;
            zlength = params.zlength;
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
            multigrid_hierarchy = nullptr;
            res_norm_over_it_with_pressure_solver = VectorXd::Zero(1e7);
            res_norm_over_it_without_pressure_solver = VectorXd::Zero(1e7);
            res_norm_over_time = VectorXd::Zero(1e7);
            save_hdf5 = params.save_hdf5;
        }
        int imax;
        int jmax;
        int kmax;
        double xlength;
        double ylength;
        double zlength;
        double t_end;
        double tau;
        double eps;
        double omg;
        double alpha;
        double Re;
        double t;
        double dt;
        double res_norm;
        int it = 0;
        int it_wo_pressure_solver = 0;
        int lastTimestamp = 0;
        int duration = 0;
        StaggeredGrid grid;
        Kernel::Timer timer;
        SolverType solver_type;
        double save_interval;
        bool save_hdf5;
        VectorXd res_norm_over_it_with_pressure_solver;
        VectorXd res_norm_over_it_without_pressure_solver;
        VectorXd res_norm_over_time;

        // Multigrid components
        MultigridHierarchy *multigrid_hierarchy;

        // Preconditioner Conjugated Gradient components
        MultigridHierarchy *multigrid_hierarchy_preconditioner;
        StaggeredGrid preconditioner;

        void selectDtAccordingToStabilityCondition();
        void computeF();
        void computeG();
        void computeH(); // 3D
        void computeRHS();
        void solveWithJacobi();
        void solveWithMultigridJacobi();
        void solveWithConjugatedGradient();
        void solveWithMultigridPCG();
        void computeDiscreteL2Norm();
        void computeU();
        void computeV();
        void computeW(); // 3D
        void run();
        void saveData();
        void saveHDF5();

        // Local functions
        bool isObstacleCell(int i, int j, int k); // 3D
        bool isCornerCell(int i, int j, int k); // 3D
        void handleCornerCell(int i, int j, int k); // 3D
        void handleObstacleCell(int i, int j, int k); // 3D

        // Virtual functions
        virtual void setBoundaryConditionsU() = 0;
        virtual void setBoundaryConditionsV() = 0;
        virtual void setBoundaryConditionsW() = 0; // 3D
        virtual void setBoundaryConditionsP() = 0;
        virtual void setBoundaryConditionsPGeometry();
        virtual void setBoundaryConditionsVelocityGeometry();

        virtual ~FluidSimulation() = default;
    };

    // Functions
    void saveVTK(FluidSimulation* sim);
    void saveVTKGeometry(FluidSimulation* sim);
}
