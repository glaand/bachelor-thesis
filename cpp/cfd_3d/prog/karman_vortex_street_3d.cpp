#include <iostream>
#include "simulations.h"

using namespace CFD;

void FluidWithObstacles3D::setBoundaryConditionsU() {
    // Inflow and outflow at left and right boundary
    for (int k = 0; k < this->grid.kmax + 2; k++) {
        for (int j = 0; j < this->grid.jmax + 3; j++) {
            // Inflow at left boundary (Left wall)
            this->grid.u(0, j, k) = 1.0;
            // Outflow at right boundary (Right wall)
            this->grid.u(this->grid.imax + 1, j, k) = this->grid.u(this->grid.imax, j, k);
        }
    }

    // no-slip at top and bottom
    for (int i = 0; i < this->grid.imax + 2; i++) {
        for (int k = 0; k < this->grid.kmax + 2; k++) {
            // Bottom wall
            this->grid.u(i, 0, k) = -this->grid.u(i, 1, k);
            // Top wall
            this->grid.u(i, this->grid.jmax + 2, k) = -this->grid.u(i, this->grid.jmax + 1, k);
        }
    }
}

void FluidWithObstacles3D::setBoundaryConditionsV() {
    // Inflow and outflow at left and right boundary
    for (int k = 0; k < this->grid.kmax + 2; k++) {
        for (int j = 0; j < this->grid.jmax + 2; j++) {
            // Inflow at left boundary (Left wall)
            this->grid.v(0, j, k) = -this->grid.v(1, j, k);
            // Outflow at right boundary (Right wall)
            this->grid.v(this->grid.imax + 1, j, k) = this->grid.v(this->grid.imax, j, k);
        }
    }

    // no-slip at top and bottom
    for (int i = 0; i < this->grid.imax + 3; i++) {
        for (int k = 0; k < this->grid.kmax + 2; k++) {
            // Bottom wall
            this->grid.v(i, 0, k) = 0.0;
            // Top wall
            this->grid.v(i, this->grid.jmax + 1, k) = 0.0;
        }
    }
}

void FluidWithObstacles3D::setBoundaryConditionsW() {
    // Inflow and outflow at left and right boundary
    for (int k = 0; k < this->grid.kmax + 3; k++) {
        for (int j = 0; j < this->grid.jmax + 2; j++) {
            // Inflow at left boundary (Left wall)
            this->grid.w(0, j, k) = -this->grid.w(1, j, k);
            // Outflow at right boundary (Right wall)
            this->grid.w(this->grid.imax + 1, j, k) = this->grid.w(this->grid.imax, j, k);
        }
    }

    // no-slip at top and bottom
    for (int i = 0; i < this->grid.imax + 2; i++) {
        for (int j = 0; j < this->grid.jmax + 2; j++) {
            // Bottom wall
            this->grid.w(i, j, 0) = -this->grid.w(i, j, 1);
            // Top wall
            this->grid.w(i, j, this->grid.kmax + 1) = -this->grid.w(i, j, this->grid.kmax);
        }
    }
}

void FluidWithObstacles3D::setBoundaryConditionsP() {
    for (int i = 0; i < this->grid.imax + 2; i++) {
        for (int j = 0; j < this->grid.jmax + 2; j++) {
            this->grid.p(i, j, 0) = this->grid.p(i, j, 1);
            this->grid.p(i, j, this->grid.kmax + 1) = this->grid.p(i, j, this->grid.kmax);
        }
    }
    for (int k = 0; k < this->grid.kmax + 2; k++) {
        for (int j = 0; j < this->grid.jmax + 2; j++) {
            this->grid.p(0, j, k) = this->grid.p(1, j, k);
            this->grid.p(this->grid.imax + 1, j, k) = this->grid.p(this->grid.imax, j, k);
        }
    }

    for (int i = 0; i < this->grid.imax + 2; i++) {
        for (int k = 0; k < this->grid.kmax + 2; k++) {
            this->grid.p(i, 0, k) = this->grid.p(i, 1, k);
            this->grid.p(i, this->grid.jmax + 1, k) = this->grid.p(i, this->grid.jmax, k);
        }
    }
}

void FluidWithObstacles3D::run() {
    // Manage Flag Field with Bitmasks
    // Cube in the middle with a fifth of the size of the domain
    int width = floor(this->grid.jmax / 2.0);
    int distanceTop = 0;
    int distanceBottom = this->grid.jmax;
    int distanceLeft = floor((this->grid.jmax - width) / 2.0);
    int distanceRight = floor((this->grid.jmax - width) / 2.0) + width;
    int distanceFront = floor((this->grid.jmax - width) / 2.0);
    int distanceBack = floor((this->grid.jmax - width) / 2.0) + width;

    for (int i = 1; i < this->grid.imax + 1; i++) {
        for (int j = 1; j < this->grid.jmax + 1; j++) {
            for (int k = 1; k < this->grid.kmax + 1; k++) {

                // Check if cell is inside the cube
                if (i >= distanceLeft && i <= distanceRight &&
                    j >= distanceTop && j <= distanceBottom &&
                    k >= distanceFront && k <= distanceBack) {

                    // Inside the cube
                    this->grid.flag_field(i, j, k) = FlagFieldMask::CELL_OBSTACLE;

                    // Top Layer of the cube
                    if (j == distanceTop) {
                        // South neighbor
                        this->grid.flag_field(i, j, k) |= FlagFieldMask::FLUID_SOUTH;
                    }
                    if (j == distanceBottom) {
                        // North neighbor
                        this->grid.flag_field(i, j, k) |= FlagFieldMask::FLUID_NORTH;
                    }
                    if (i == distanceLeft) {
                        // West neighbor
                        this->grid.flag_field(i, j, k) |= FlagFieldMask::FLUID_WEST;
                    }
                    if (i == distanceRight) {
                        // East neighbor
                        this->grid.flag_field(i, j, k) |= FlagFieldMask::FLUID_EAST;
                    }
                    if (k == distanceFront) {
                        // Bottom neighbor
                        this->grid.flag_field(i, j, k) |= FlagFieldMask::FLUID_BOTTOM;
                    }
                    if (k == distanceBack) {
                        // Top neighbor
                        this->grid.flag_field(i, j, k) |= FlagFieldMask::FLUID_TOP;
                    }
                } else {
                    // Fluid cells have fluid neighbors
                    this->grid.flag_field(i, j, k) = FlagFieldMask::CELL_FLUID;
                }
            }
        }
    }

    /*// loop through flag field and print with std::bitset
    for (int i = 0; i < this->grid.imax + 2; i++) {
        for (int j = 0; j < this->grid.jmax + 2; j++) {
            for (int k = 0; k < this->grid.kmax + 2; k++) {
                std::cout << std::bitset<6>(this->grid.flag_field(i, j, k)) << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
    std::exit(0);*/

    for (int i = 0; i < this->grid.imax + 2; i++) {
        for (int j = 0; j < this->grid.jmax + 3; j++) {
            for (int k = 0; k < this->grid.kmax + 2; k++) {
                this->grid.u(i, j, k) = 1.0;
            }
        }
    }

    FluidSimulation::run();

    return;
}

int main(int argc, char* argv[]) {
    FluidParams flow_params = FluidParams("karman_vortex_street_3d", argc, argv);
    FluidWithObstacles3D sim = FluidWithObstacles3D(flow_params);
    sim.run();
    sim.saveData();
}