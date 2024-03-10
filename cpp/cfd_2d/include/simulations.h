#pragma once
#include "cfd.h"

namespace CFD {
    using namespace Eigen;

    class FluidWithoutObstacles2D : public FluidSimulation {
        public:
            FluidWithoutObstacles2D(const FluidParams& params) : FluidSimulation(params) {
                grid = StaggeredGrid(imax, jmax, xlength, ylength);
            }
            void setBoundaryConditionsU() override;
            void setBoundaryConditionsV() override;
            void setBoundaryConditionsP() override;
            void setBoundaryConditionsVelocityGeometry() override {};
            void setBoundaryConditionsPGeometry() override {};
    };

    class FluidWithObstacles2D : public FluidSimulation {
        public:
            FluidWithObstacles2D(const FluidParams& params) : FluidSimulation(params) {
                grid = StaggeredGrid(imax, jmax, xlength, ylength);
            }
            void setBoundaryConditionsU() override;
            void setBoundaryConditionsV() override;
            void setBoundaryConditionsP() override;
            void run();
    };
}
