#include <iostream>
#include "simulations.h"

using namespace CFD;

void FluidWithoutObstacles3D::setBoundaryConditionsU() {
    // Everything no-slip in u-rows
    for (int k = 0; k < this->grid.kmax + 3; k++) {
        for (int j = 0; j < this->grid.jmax + 3; j++) {
            // Left wall
            this->grid.u(0, j, k) = 0.0;
            // Right wall
            this->grid.u(this->grid.imax, j, k) = 0.0;
        }
    }

    // Interpolate with inner wall
    for (int i = 0; i < this->grid.imax + 2; i++) {
        for (int k = 0; k < this->grid.kmax + 2; k++) {
            // Bottom wall
            this->grid.u(i, 0, k) = -this->grid.u(i, 1, k);
            // Top wall
            this->grid.u(i, this->grid.jmax + 1, k) = 2.0 - this->grid.u(i, this->grid.jmax, k);
        }
    }
}

void FluidWithoutObstacles3D::setBoundaryConditionsV() {
    // Everything no-slip in v-rows
    for (int k = 0; k < this->grid.kmax + 2; k++) {
        for (int j = 0; j < this->grid.jmax + 2; j++) {
            // Left wall
            this->grid.v(0, j, k) = -this->grid.v(1, j, k);
            // Right wall
            this->grid.v(this->grid.imax + 1, j, k) = -this->grid.v(this->grid.imax, j, k);
        }
    }

    // Interpolate with inner wall
    for (int i = 0; i < this->grid.imax + 3; i++) {
        for (int k = 0; k < this->grid.kmax + 2; k++) {
            // Bottom wall
            this->grid.v(i, 0, k) = 0.0;
            // Top wall
            this->grid.v(i, this->grid.jmax, k) = 0.0;
        }
    }
}

void FluidWithoutObstacles3D::setBoundaryConditionsW() {
    // Everything no-slip in w-rows
    for (int k = 0; k < this->grid.kmax + 2; k++) {
        for (int j = 0; j < this->grid.jmax + 3; j++) {
            // Left wall
            this->grid.w(0, j, k) = -this->grid.w(1, j, k);
            // Right wall
            this->grid.w(this->grid.imax + 1, j, k) = -this->grid.w(this->grid.imax, j, k);
        }
    }

    // Interpolate with inner wall
    for (int i = 0; i < this->grid.imax + 2; i++) {
        for (int j = 0; j < this->grid.jmax + 2; j++) {
            // Bottom wall
            this->grid.w(i, j, 0) = 0.0;
            // Top wall
            this->grid.w(i, j, this->grid.kmax + 1) = 0.0;
        }
    }
}

void FluidWithoutObstacles3D::setBoundaryConditionsP() {
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
}

int main(int argc, char* argv[]) {
    FluidParams flow_params = FluidParams("lid_driven_cavity_3d", argc, argv);
    FluidWithoutObstacles3D sim = FluidWithoutObstacles3D(flow_params);
    sim.run();
    sim.saveData();
}
