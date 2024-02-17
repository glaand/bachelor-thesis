#pragma once
#include "cfd.h"

namespace CFD {
    using namespace Eigen;

    class FluidWithoutObstacles3D : public FluidSimulation {
        public:
            FluidWithoutObstacles3D(const FluidParams& params) : FluidSimulation(params) {
                grid = StaggeredGrid(imax, jmax, kmax, xlength, ylength, zlength);
            }
            void setBoundaryConditionsU() override;
            void setBoundaryConditionsV() override;
            void setBoundaryConditionsW() override;
            void setBoundaryConditionsP() override;
            void setBoundaryConditionsVelocityGeometry() override {};
            void setBoundaryConditionsPGeometry() override {};
    };

    class FluidWithObstacles3D : public FluidSimulation {
        public:
            FluidWithObstacles3D(const FluidParams& params) : FluidSimulation(params) {
                grid = StaggeredGrid(imax, jmax, kmax, xlength, ylength, zlength);
            }
            void setBoundaryConditionsU() override;
            void setBoundaryConditionsV() override;
            void setBoundaryConditionsW() override;
            void setBoundaryConditionsP() override;
            void run();
    };
}
