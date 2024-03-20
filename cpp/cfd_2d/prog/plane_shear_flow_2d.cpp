#include <iostream>
#include "simulations.h"

using namespace CFD;


void FluidWithoutObstacles2D::setBoundaryConditionsU() {
    // Inflow and outflow at left and right boundary
    for (int j = 0; j < this->grid.jmax + 3; j++) {
        // Inflow at left boundary (Left wall)
        this->grid.u(0, j) = 1.0;
        this->grid.F(0, j) = 1.0;
        // Outflow at right boundary (Right wall)
        this->grid.u(this->grid.imax + 1, j) = this->grid.u(this->grid.imax, j);
        this->grid.F(this->grid.imax + 1, j) = this->grid.F(this->grid.imax, j);
    }

    // no-slip at top and bottom
    for (int i = 0; i < this->grid.imax + 2; i++) {
        // Bottom wall
        this->grid.u(i, 0) = -this->grid.u(i, 1);
        this->grid.F(i, 0) = -this->grid.F(i, 1);
        // Top wall
        this->grid.u(i, this->grid.jmax + 2) = -this->grid.u(i, this->grid.jmax + 1);
        this->grid.F(i, this->grid.jmax + 2) = -this->grid.F(i, this->grid.jmax + 1);
    }
}

void FluidWithoutObstacles2D::setBoundaryConditionsV() {
    // Inflow and outflow at left and right boundary
    for (int j = 0; j < this->grid.jmax + 2; j++) {
        // Inflow at left boundary (Left wall)
        this->grid.v(0, j) = -this->grid.v(1, j);
        this->grid.G(0, j) = -this->grid.G(1, j);
        // Outflow at right boundary (Right wall)
        this->grid.v(this->grid.imax + 1, j) = this->grid.v(this->grid.imax, j);
        this->grid.G(this->grid.imax + 1, j) = this->grid.G(this->grid.imax, j);
    }

    // no-slip at top and bottom
    for (int i = 0; i < this->grid.imax + 3; i++) {
        // Bottom wall
        this->grid.v(i, 0) = 0.0;
        this->grid.G(i, 0) = 0.0;
        // Top wall
        this->grid.v(i, this->grid.jmax + 1) = 0.0;
        this->grid.G(i, this->grid.jmax + 1) = 0.0;
    }
}

void FluidWithoutObstacles2D::setBoundaryConditionsP() {
    for (int i = 0; i < this->grid.imax + 2; i++) {
        this->grid.p(i, 0) = this->grid.p(i, 1);
        this->grid.p(i, this->grid.jmax + 1) = this->grid.p(i, this->grid.jmax);
    }
    for (int j = 0; j < this->grid.jmax + 2; j++) {
        this->grid.p(0, j) = this->grid.p(1, j);
        this->grid.p(this->grid.imax + 1, j) = this->grid.p(this->grid.imax, j);
    }
}


int main(int argc, char* argv[]) {
    FluidParams flow_params = FluidParams("plane_shear_flow_2d", argc, argv);
    FluidWithoutObstacles2D sim = FluidWithoutObstacles2D(flow_params);
    for (int i = 0; i < sim.grid.imax + 2; i++) {
        for (int j = 0; j < sim.grid.jmax + 3; j++) {
            sim.grid.u(i, j) = 1.0;
        }
    }
    sim.run();
    sim.saveData();
}
