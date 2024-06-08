#include <iostream>
#include "simulations.h"

using namespace CFD;

void FluidWithObstacles2D::setBoundaryConditionsU() {
    // Inflow and outflow at left and right boundary
    for (int j = 0; j < this->grid.jmax + 3; j++) {
        // Inflow at left boundary (Left wall)
        this->grid.u(0, j) = 1.5;
        // Outflow at right boundary (Right wall)
        this->grid.u(this->grid.imax + 1, j) = this->grid.u(this->grid.imax, j);
    }

    // no-slip at top and bottom
    for (int i = 0; i < this->grid.imax + 2; i++) {
        // Bottom wall
        this->grid.u(i, 0) = -this->grid.u(i, 1);
        // Top wall
        this->grid.u(i, this->grid.jmax + 2) = -this->grid.u(i, this->grid.jmax + 1);
    }
}

void FluidWithObstacles2D::setBoundaryConditionsV() {
    // Inflow and outflow at left and right boundary
    for (int j = 0; j < this->grid.jmax + 2; j++) {
        // Inflow at left boundary (Left wall)
        this->grid.v(0, j) = -this->grid.v(1, j);
        // Outflow at right boundary (Right wall)
        this->grid.v(this->grid.imax + 1, j) = this->grid.v(this->grid.imax, j);
    }

    // no-slip at top and bottom
    for (int i = 0; i < this->grid.imax + 3; i++) {
        // Bottom wall
        this->grid.v(i, 0) = 0.0;
        // Top wall
        this->grid.v(i, this->grid.jmax + 1) = 0.0;
    }
}

void FluidWithObstacles2D::setBoundaryConditionsP() {
    for (int i = 0; i < this->grid.imax + 2; i++) {
        this->grid.p(i, 0) = this->grid.p(i, 1);
        this->grid.p(i, this->grid.jmax + 1) = this->grid.p(i, this->grid.jmax);
    }
    for (int j = 0; j < this->grid.jmax + 2; j++) {
        this->grid.p(0, j) = this->grid.p(1, j);
        this->grid.p(this->grid.imax + 1, j) = this->grid.p(this->grid.imax, j);
    }
}

void FluidWithObstacles2D::run() {
    // Manage Flag Field with Bitmasks
    // Circle with a radius of one-fifth of the domain's width, positioned one-fifth of the width from the left
    int radius = floor(this->grid.jmax / this->radius);
    int centerY = floor(this->grid.jmax / 2.0); // Centered vertically
    int centerX = centerY; // One-fifth of the width from the left

    for (int i = 1; i < this->grid.imax + 1; i++) {
        for (int j = 1; j < this->grid.jmax + 1; j++) {

            // Check if cell is inside the circle
            if ((i - centerX) * (i - centerX) + (j - centerY) * (j - centerY) <= radius * radius) {
                // Inside the circle
                this->grid.flag_field(i, j) = FlagFieldMask::CELL_OBSTACLE;

                // Top Layer of the circle
                if ((i - centerX) * (i - centerX) + (j - (centerY - radius)) * (j - (centerY - radius)) <= 1) {
                    // South neighbor
                    this->grid.flag_field(i, j) |= FlagFieldMask::FLUID_SOUTH;
                }
                if ((i - centerX) * (i - centerX) + (j - (centerY + radius)) * (j - (centerY + radius)) <= 1) {
                    // North neighbor
                    this->grid.flag_field(i, j) |= FlagFieldMask::FLUID_NORTH;
                }
                if ((i - (centerX - radius)) * (i - (centerX - radius)) + (j - centerY) * (j - centerY) <= 1) {
                    // West neighbor
                    this->grid.flag_field(i, j) |= FlagFieldMask::FLUID_WEST;
                }
                if ((i - (centerX + radius)) * (i - (centerX + radius)) + (j - centerY) * (j - centerY) <= 1) {
                    // East neighbor
                    this->grid.flag_field(i, j) |= FlagFieldMask::FLUID_EAST;
                }
            } else {
                // Fluid cells have fluid neighbors
                this->grid.flag_field(i, j) = FlagFieldMask::CELL_FLUID;
            }
        }
    }



    /*// loop through flag field and print with std::bitset
    for (int i = 0; i < this->grid.imax + 2; i++) {
        for (int j = 0; j < this->grid.jmax + 2; j++) {
            std::cout << std::bitset<5>(this->grid.flag_field(i, j)) << " ";
        }
        std::cout << std::endl;
    }
    std::exit(0);*/

    for (int i = 0; i < this->grid.imax + 2; i++) {
        for (int j = 0; j < this->grid.jmax + 3; j++) {
            this->grid.u(i, j) = 1.0;
            this->grid.F(i, j) = this->grid.u(i, j);
        }
    }

    FluidSimulation::run();

    return;
}


int main(int argc, char* argv[]) {
    FluidParams flow_params = FluidParams("karman_vortex_street_2d", argc, argv);
    FluidWithObstacles2D sim = FluidWithObstacles2D(flow_params);
    sim.run();
    sim.saveData();
}
