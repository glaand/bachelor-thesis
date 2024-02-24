#include <iostream>
#include <filesystem>
#include <chrono>
#include "cfd.h"

namespace CFD {
    namespace ME_X {
        // Slide 15
        double uu_x(StaggeredGrid *grid, FluidSimulation *sim, int i, int j, int k) {
            return (
                (1 / grid->dx) * (pow((grid->u(i, j, k) + grid->u(i + 1, j, k)) / 2, 2) - pow((grid->u(i - 1, j, k) + grid->u(i, j, k)) / 2, 2)) +
                (sim->alpha / grid->dx) * (std::abs(grid->u(i, j, k) + grid->u(i + 1, j, k)) * (grid->u(i, j, k) - grid->u(i + 1, j, k)) / 4 - std::abs(grid->u(i - 1, j, k) + grid->u(i, j, k)) * (grid->u(i - 1, j, k) - grid->u(i, j, k)) / 4)
            );
        }

        double uv_y(StaggeredGrid *grid, FluidSimulation *sim, int i, int j, int k) {
            return (
                (1 / grid->dy) * ((grid->v(i, j, k) + grid->v(i + 1, j, k)) * (grid->u(i, j, k) + grid->u(i, j + 1, k)) / 4 - (grid->v(i, j - 1, k) + grid->v(i + 1, j - 1, k)) * (grid->u(i, j - 1, k) + grid->u(i, j, k)) / 4) +
                (sim->alpha / grid->dy) * (std::abs(grid->v(i, j, k) + grid->v(i + 1, j, k)) * (grid->u(i, j, k) - grid->u(i, j + 1, k)) / 4 - std::abs(grid->v(i, j - 1, k) + grid->v(i + 1, j - 1, k)) * (grid->u(i, j - 1, k) - grid->u(i, j, k)) / 4)
            );
        }

        double uu_xx(StaggeredGrid *grid, FluidSimulation *sim, int i, int j, int k) {
            return (
                (grid->u(i + 1, j, k) - 2 * grid->u(i, j, k) + grid->u(i - 1, j, k)) / pow(grid->dx, 2)
            );
        }

        double uu_yy(StaggeredGrid *grid, FluidSimulation *sim, int i, int j, int k) {
            return (
                (grid->u(i, j + 1, k) - 2 * grid->u(i, j, k) + grid->u(i, j - 1, k)) / pow(grid->dy, 2)
            );
        }

        double uu_zz(StaggeredGrid *grid, FluidSimulation *sim, int i, int j, int k) {
            return (
                (grid->u(i, j, k + 1) - 2 * grid->u(i, j, k) + grid->u(i, j, k - 1)) / pow(grid->dz, 2)
            );
        }

        double uw_z(StaggeredGrid *grid, FluidSimulation *sim, int i, int j, int k) {
            return (
                (1 / grid->dz) * ((grid->w(i, j, k) + grid->w(i + 1, j, k)) * (grid->u(i, j, k) + grid->u(i, j, k + 1)) / 4 - (grid->w(i, j, k - 1) + grid->w(i + 1, j, k - 1)) * (grid->u(i, j, k - 1) + grid->u(i, j, k)) / 4) +
                (sim->alpha / grid->dz) * (std::abs(grid->w(i, j, k) + grid->w(i + 1, j, k)) * (grid->u(i, j, k) - grid->u(i, j, k + 1)) / 4 - std::abs(grid->w(i, j, k - 1) + grid->w(i + 1, j, k - 1)) * (grid->u(i, j, k - 1) - grid->u(i, j, k)) / 4)
            );
        }

        double p_x(StaggeredGrid *grid, FluidSimulation *sim, int i, int j, int k) {
            return (
                (grid->p(i + 1, j, k) - grid->p(i, j, k)) / grid->dx
            );
        }
    }

    namespace ME_Y {
        double uv_x(StaggeredGrid *grid, FluidSimulation *sim, int i, int j, int k) {
            return (
                (1 / grid->dx) * ((grid->u(i, j, k) + grid->u(i, j + 1, k)) * (grid->v(i, j, k) + grid->v(i + 1, j, k)) / 4 - (grid->u(i - 1, j, k) + grid->u(i - 1, j + 1, k)) * (grid->v(i - 1, j, k) + grid->v(i, j, k)) / 4) +
                (sim->alpha / grid->dx) * (std::abs(grid->u(i, j, k) + grid->u(i, j + 1, k)) * (grid->v(i, j, k) - grid->v(i + 1, j, k)) / 4 - std::abs(grid->u(i - 1, j, k) + grid->u(i - 1, j + 1, k)) * (grid->v(i - 1, j, k) - grid->v(i, j, k)) / 4)
            );
        }

        double vv_y(StaggeredGrid *grid, FluidSimulation *sim, int i, int j, int k) {
            return (
                (1 / grid->dy) * (pow((grid->v(i, j, k) + grid->v(i, j + 1, k)) / 2, 2) - pow((grid->v(i, j - 1, k) + grid->v(i, j, k)) / 2, 2)) +
                (sim->alpha / grid->dy) * (std::abs(grid->v(i, j, k) + grid->v(i, j + 1, k)) * (grid->v(i, j, k) - grid->v(i, j + 1, k)) / 4 - std::abs(grid->v(i, j - 1, k) + grid->v(i, j, k)) * (grid->v(i, j - 1, k) - grid->v(i, j, k)) / 4)
            );
        }

        double vv_xx(StaggeredGrid *grid, FluidSimulation *sim, int i, int j, int k) {
            return (
                (grid->v(i + 1, j, k) - 2 * grid->v(i, j, k) + grid->v(i - 1, j, k)) / pow(grid->dx, 2)
            );
        }

        double vv_yy(StaggeredGrid *grid, FluidSimulation *sim, int i, int j, int k) {
            return (
                (grid->v(i, j + 1, k) - 2 * grid->v(i, j, k) + grid->v(i, j - 1, k)) / pow(grid->dy, 2)
            );
        }

        double vv_zz(StaggeredGrid *grid, FluidSimulation *sim, int i, int j, int k) {
            return (
                (grid->v(i, j, k + 1) - 2 * grid->v(i, j, k) + grid->v(i, j, k - 1)) / pow(grid->dz, 2)
            );
        }

        double vw_z(StaggeredGrid *grid, FluidSimulation *sim, int i, int j, int k) {
            return (
                (1 / grid->dz) * ((grid->w(i, j, k) + grid->w(i, j + 1, k)) * (grid->v(i, j, k) + grid->v(i, j, k + 1)) / 4 - (grid->w(i, j, k - 1) + grid->w(i, j + 1, k - 1)) * (grid->v(i, j, k - 1) + grid->v(i, j, k)) / 4) +
                (sim->alpha / grid->dz) * (std::abs(grid->w(i, j, k) + grid->w(i, j + 1, k)) * (grid->v(i, j, k) - grid->v(i, j, k + 1)) / 4 - std::abs(grid->w(i, j, k - 1) + grid->w(i, j + 1, k - 1)) * (grid->v(i, j, k - 1) - grid->v(i, j, k)) / 4)
            );
        }

        double p_y(StaggeredGrid *grid, FluidSimulation *sim, int i, int j, int k) {
            return (
                (grid->p(i, j + 1, k) - grid->p(i, j, k)) / grid->dy
            );
        }
    }

    namespace ME_Z {
        double uw_x(StaggeredGrid *grid, FluidSimulation *sim, int i, int j, int k) {
            return (
                (1 / grid->dx) * ((grid->u(i, j, k) + grid->u(i, j, k + 1)) * (grid->w(i, j, k) + grid->w(i + 1, j, k)) / 4 - (grid->u(i - 1, j, k) + grid->u(i - 1, j, k + 1)) * (grid->w(i - 1, j, k) + grid->w(i, j, k)) / 4) +
                (sim->alpha / grid->dx) * (std::abs(grid->u(i, j, k) + grid->u(i, j, k + 1)) * (grid->w(i, j, k) - grid->w(i + 1, j, k)) / 4 - std::abs(grid->u(i - 1, j, k) + grid->u(i - 1, j, k + 1)) * (grid->w(i - 1, j, k) - grid->w(i, j, k)) / 4)
            );
        }

        double vw_y(StaggeredGrid *grid, FluidSimulation *sim, int i, int j, int k) {
            return (
                (1 / grid->dy) * ((grid->v(i, j, k) + grid->v(i, j, k + 1)) * (grid->w(i, j, k) + grid->w(i, j + 1, k)) / 4 - (grid->v(i, j - 1, k) + grid->v(i, j - 1, k + 1)) * (grid->w(i, j - 1, k) + grid->w(i, j, k)) / 4) +
                (sim->alpha / grid->dy) * (std::abs(grid->v(i, j, k) + grid->v(i, j, k + 1)) * (grid->w(i, j, k) - grid->w(i, j + 1, k)) / 4 - std::abs(grid->v(i, j - 1, k) + grid->v(i, j - 1, k + 1)) * (grid->w(i, j - 1, k) - grid->w(i, j, k)) / 4)
            );
        }

        double ww_z(StaggeredGrid *grid, FluidSimulation *sim, int i, int j, int k) {
            return (
                (1 / grid->dz) * (pow((grid->w(i, j, k) + grid->w(i, j, k + 1)) / 2, 2) - pow((grid->w(i, j, k - 1) + grid->w(i, j, k)) / 2, 2)) +
                (sim->alpha / grid->dz) * (std::abs(grid->w(i, j, k) + grid->w(i, j, k + 1)) * (grid->w(i, j, k) - grid->w(i, j, k + 1)) / 4 - std::abs(grid->w(i, j, k - 1) + grid->w(i, j, k)) * (grid->w(i, j, k - 1) - grid->w(i, j, k)) / 4)
            );
        }

        double uu_zz(StaggeredGrid *grid, FluidSimulation *sim, int i, int j, int k) {
            return (
                (grid->u(i, j, k + 1) - 2 * grid->u(i, j, k) + grid->u(i, j, k - 1)) / pow(grid->dz, 2)
            );
        }

        double vv_zz(StaggeredGrid *grid, FluidSimulation *sim, int i, int j, int k) {
            return (
                (grid->v(i, j, k + 1) - 2 * grid->v(i, j, k) + grid->v(i, j, k - 1)) / pow(grid->dz, 2)
            );
        }

        double ww_xx(StaggeredGrid *grid, FluidSimulation *sim, int i, int j, int k) {
            return (
                (grid->w(i + 1, j, k) - 2 * grid->w(i, j, k) + grid->w(i - 1, j, k)) / pow(grid->dx, 2)
            );
        }

        double ww_yy(StaggeredGrid *grid, FluidSimulation *sim, int i, int j, int k) {
            return (
                (grid->w(i, j + 1, k) - 2 * grid->w(i, j, k) + grid->w(i, j - 1, k)) / pow(grid->dy, 2)
            );
        }

        double ww_zz(StaggeredGrid *grid, FluidSimulation *sim, int i, int j, int k) {
            return (
                (grid->w(i, j, k + 1) - 2 * grid->w(i, j, k) + grid->w(i, j, k - 1)) / pow(grid->dz, 2)
            );
        }

        double p_z(StaggeredGrid *grid, FluidSimulation *sim, int i, int j, int k) {
            return (
                (grid->p(i, j, k + 1) - grid->p(i, j, k)) / grid->dz
            );
        }
    }

    double StaggeredGrid::findMaxAbsoluteU() const {
        double max = 0.0;
        for (int i = 1; i < imax + 1; i++) {
            for (int j = 1; j < jmax + 2; j++) {
                for (int k = 1; k < kmax + 1; k++) {
                    if (std::abs(u(i, j, k)) > max) {
                        max = std::abs(u(i, j, k));
                    }
                }
            }
        }
        return max;
    }

    double StaggeredGrid::findMaxAbsoluteV() const {
        double max = 0.0;
        for (int i = 1; i < imax + 2; i++) {
            for (int j = 1; j < jmax + 1; j++) {
                for (int k = 1; k < kmax + 1; k++) {
                    if (std::abs(v(i, j, k)) > max) {
                        max = std::abs(v(i, j, k));
                    }
                }
            }
        }
        return max;
    }

    double StaggeredGrid::findMaxAbsoluteW() const {
        double max = 0.0;
        for (int i = 1; i < imax + 2; i++) {
            for (int j = 1; j < jmax + 2; j++) {
                for (int k = 1; k < kmax + 1; k++) {
                    if (std::abs(w(i, j, k)) > max) {
                        max = std::abs(w(i, j, k));
                    }
                }
            }
        }
        return max;
    }

    void StaggeredGrid::interpolateVelocity() {
        for (int i = 0; i < imax + 2; i++) {
            for (int j = 0; j < jmax + 2; j++) {
                for (int k = 0; k < kmax + 2; k++) {
                    u_interpolated(i, j, k) = (u(i, j, k) + u(i, j + 1, k)) / 2;
                    v_interpolated(i, j, k) = (v(i, j, k) + v(i + 1, j, k)) / 2;
                    w_interpolated(i, j, k) = (w(i, j, k) + w(i, j, k + 1)) / 2;
                }
            }
        }
    }

    void FluidSimulation::selectDtAccordingToStabilityCondition() {
        double left = (this->Re/3) * pow(((1/pow(this->grid.dx, 2)) + (1/pow(this->grid.dy, 2)) + (1/pow(this->grid.dz, 2))), -1);
        double middle = this->grid.dx / this->grid.findMaxAbsoluteU();
        double right = this->grid.dy / this->grid.findMaxAbsoluteV();
        double top = this->grid.dz / this->grid.findMaxAbsoluteW();
        
        this->dt = this->tau * std::min(std::min(std::min(left, middle), right), top);
    }


    void FluidSimulation::computeF() {
        for (int i = 1; i < this->grid.imax + 1; i++) {
            for (int j = 1; j < this->grid.jmax + 2; j++) {
                for (int k = 1; k < this->grid.kmax + 2; k++) {
                    this->grid.F(i, j, k) = this->grid.u(i, j, k) + this->dt * (
                        (1/this->Re) * (ME_X::uu_xx(&this->grid, this, i, j, k) + ME_X::uu_yy(&this->grid, this, i, j, k) + ME_X::uu_zz(&this->grid, this, i, j, k))
                        -
                        ME_X::uu_x(&this->grid, this, i, j, k)
                        -
                        ME_X::uv_y(&this->grid, this, i, j, k)
                        -
                        ME_X::uw_z(&this->grid, this, i, j, k)
                    );
                }
            }
        }
        // Boundary conditions
        for (int j = 0; j < this->grid.jmax + 3; j++) {
            for (int k = 0; k < this->grid.kmax + 3; k++) {
                this->grid.F(0, j, k) = this->grid.u(0, j, k);
                this->grid.F(this->grid.imax, j, k) = this->grid.u(this->grid.imax, j, k);
            }
        }
    }

    void FluidSimulation::computeG() {
        for (int i = 1; i < this->grid.imax + 2; i++) {
            for (int j = 1; j < this->grid.jmax + 1; j++) {
                for (int k = 1; k < this->grid.kmax + 2; k++) {
                    this->grid.G(i, j, k) = this->grid.v(i, j, k) + this->dt * (
                        (1/this->Re) * (ME_Y::vv_xx(&this->grid, this, i, j, k) + ME_Y::vv_yy(&this->grid, this, i, j, k) + ME_Y::vv_zz(&this->grid, this, i, j, k))
                        -
                        ME_Y::uv_x(&this->grid, this, i, j, k)
                        -
                        ME_Y::vv_y(&this->grid, this, i, j, k)
                        -
                        ME_Y::vw_z(&this->grid, this, i, j, k)
                    );
                }
            }
        }
        // Boundary conditions
        for (int i = 0; i < this->grid.imax + 3; i++) {
            for (int k = 0; k < this->grid.kmax + 3; k++) {
                this->grid.G(i, 0, k) = this->grid.v(i, 0, k);
                this->grid.G(i, this->grid.jmax, k) = this->grid.v(i, this->grid.jmax, k);
            }
        }
    }

    void FluidSimulation::computeH() {
        for (int i = 1; i < this->grid.imax + 1; i++) {
            for (int j = 1; j < this->grid.jmax + 1; j++) {
                for (int k = 1; k < this->grid.kmax + 2; k++) {
                    this->grid.H(i, j, k) = this->grid.w(i, j, k) + this->dt * (
                        (1/this->Re) * (ME_Z::ww_xx(&this->grid, this, i, j, k) + ME_Z::ww_yy(&this->grid, this, i, j, k) + ME_Z::ww_zz(&this->grid, this, i, j, k))
                        -
                        ME_Z::uw_x(&this->grid, this, i, j, k)
                        -
                        ME_Z::vw_y(&this->grid, this, i, j, k)
                        -
                        ME_Z::ww_z(&this->grid, this, i, j, k)
                    );
                }
            }
        }
        // Boundary conditions
        for (int i = 0; i < this->grid.imax + 3; i++) {
            for (int j = 0; j < this->grid.jmax + 3; j++) {
                this->grid.H(i, j, 0) = this->grid.w(i, j, 0);
                this->grid.H(i, j, this->grid.kmax) = this->grid.w(i, j, this->grid.kmax);
            }
        }
    }


    void FluidSimulation::computeRHS() {
        for (int i = 1; i < this->grid.imax + 1; i++) {
            for (int j = 1; j < this->grid.jmax + 1; j++) {
                for (int k = 1; k < this->grid.kmax + 1; k++) {
                    this->grid.RHS(i, j, k) = (1/this->dt) * (
                        (this->grid.F(i, j, k) - this->grid.F(i-1, j, k))/this->grid.dx +
                        (this->grid.G(i, j, k) - this->grid.G(i, j-1, k))/this->grid.dy +
                        (this->grid.H(i, j, k) - this->grid.H(i, j, k-1))/this->grid.dz
                    );
                }
            }
        }
    }

    void FluidSimulation::computeDiscreteL2Norm() {
        this->res_norm = 0.0;
        for (int i = 1; i < this->grid.imax + 1; i++) {
            for (int j = 1; j < this->grid.jmax + 1; j++) {
                for (int k = 1; k < this->grid.kmax + 1; k++) {
                    this->res_norm += pow((this->grid.po(i, j, k) - this->grid.p(i, j, k)), 2);
                }
            }
        }
        // Multiply by dx, dy, and dz to get the integral
        this->res_norm *= this->grid.dx * this->grid.dy * this->grid.dz;
        this->res_norm = sqrt(this->res_norm);
    }

    void FluidSimulation::computeU() {
        for (int i = 1; i < this->grid.imax + 3; i++) {
            for (int j = 1; j < this->grid.jmax + 1; j++) {
                for (int k = 1; k < this->grid.kmax + 1; k++) {
                    this->grid.u(i, j, k) = this->grid.F(i, j, k) - (this->dt/this->grid.dx) * (this->grid.p(i+1, j, k) - this->grid.p(i, j, k));
                }
            }
        }
    }

    void FluidSimulation::computeV() {
        for (int i = 1; i < this->grid.imax + 1; i++) {
            for (int j = 1; j < this->grid.jmax + 3; j++) {
                for (int k = 1; k < this->grid.kmax + 1; k++) {
                    this->grid.v(i, j, k) = this->grid.G(i, j, k) - (this->dt / this->grid.dy) * (this->grid.p(i, j + 1, k) - this->grid.p(i, j, k));
                }
            }
        }
    }

    void FluidSimulation::computeW() {
        for (int i = 1; i < this->grid.imax + 1; i++) {
            for (int j = 1; j < this->grid.jmax + 1; j++) {
                for (int k = 1; k < this->grid.kmax + 3; k++) {
                    this->grid.w(i, j, k) = this->grid.H(i, j, k) - (this->dt/this->grid.dz) * (this->grid.p(i, j, k+1) - this->grid.p(i, j, k));
                }
            }
        }
    }

    void FluidSimulation::setBoundaryConditionsPGeometry() {
        // Geometry boundaries
        double tmp_p = 0.0;
        int counter = 0;
        for (int i = 1; i < this->grid.imax + 1; i++) {
            for (int j = 1; j < this->grid.jmax + 1; j++) {
                for (int k = 1; k < this->grid.kmax + 1; k++) {
                    tmp_p = 0.0;
                    counter = 0;
                    // check if is obstacle
                    if (isObstacleCell(i, j, k)) {
                        // obstacle cell
                        if (this->grid.flag_field(i, j, k) & FlagFieldMask::FLUID_NORTH) {
                            tmp_p += this->grid.p(i, j+1, k);
                            counter++;
                        }
                        if (this->grid.flag_field(i, j, k) & FlagFieldMask::FLUID_SOUTH) {
                            tmp_p += this->grid.p(i, j-1, k);
                            counter++;
                        }
                        if (this->grid.flag_field(i, j, k) & FlagFieldMask::FLUID_WEST) {
                            tmp_p += this->grid.p(i-1, j, k);
                            counter++;
                        }
                        if (this->grid.flag_field(i, j, k) & FlagFieldMask::FLUID_EAST) {
                            tmp_p += this->grid.p(i+1, j, k);
                            counter++;
                        }
                        if (counter > 0) {
                            this->grid.p(i, j, k) = tmp_p / counter;
                        }
                    }
                }
            }
        }
    }


    void FluidSimulation::setBoundaryConditionsVelocityGeometry() {
        for (int i = 1; i < this->grid.imax + 1; i++) {
            for (int j = 1; j < this->grid.jmax + 1; j++) {
                for (int k = 1; k < this->grid.kmax + 1; k++) {
                    if (isObstacleCell(i, j, k)) {
                        if (isCornerCell(i, j, k)) {
                            handleCornerCell(i, j, k);
                        } else {
                            handleObstacleCell(i, j, k);
                        }
                    }
                }
            }
        }
    }

    bool FluidSimulation::isObstacleCell(int i, int j, int k) {
        return (this->grid.flag_field(i, j, k) & FlagFieldMask::MASK_CELL_TYPE) == FlagFieldMask::CELL_OBSTACLE;
    }

    bool FluidSimulation::isCornerCell(int i, int j, int k) {
        return (this->grid.flag_field(i, j, k) & (FlagFieldMask::FLUID_NORTH | FlagFieldMask::FLUID_SOUTH)) &&
            (this->grid.flag_field(i, j, k) & (FlagFieldMask::FLUID_EAST | FlagFieldMask::FLUID_WEST));
    }

    void FluidSimulation::handleCornerCell(int i, int j, int k) {
        this->grid.u(i, j, k) = 0.0;
        this->grid.v(i, j, k) = 0.0;
        this->grid.w(i, j, k) = 0.0;
        this->grid.u(i-1, j, k) = -this->grid.u(i-1, j + (i == this->grid.imax ? -1 : 1), k);
        this->grid.v(i, j-1, k) = -this->grid.v(i + (j == this->grid.jmax ? -1 : 1), j-1, k);
        this->grid.w(i, j, k-1) = -this->grid.w(i, j, k + (k == this->grid.kmax ? -1 : 1));
        this->grid.G(i, j, k) = this->grid.v(i, j, k);
        this->grid.F(i, j, k) = this->grid.u(i, j, k);
        this->grid.H(i, j, k) = this->grid.w(i, j, k);
    }

    void FluidSimulation::handleObstacleCell(int i, int j, int k) {
        if (this->grid.flag_field(i, j, k) & FlagFieldMask::FLUID_NORTH) {
            this->grid.u(i, j, k) = -this->grid.u(i, j + 1, k);
            this->grid.u(i-1, j, k) = -this->grid.u(i-1, j + 1, k);
            this->grid.v(i, j, k) = 0.0;
            this->grid.G(i, j, k) = this->grid.v(i, j, k);
        } else if (this->grid.flag_field(i, j, k) & FlagFieldMask::FLUID_SOUTH) {
            this->grid.u(i, j, k) = -this->grid.u(i, j - 1, k);
            this->grid.u(i-1, j, k) = -this->grid.u(i-1, j - 1, k);
            this->grid.v(i, j, k) = 0.0;
            this->grid.G(i, j, k) = this->grid.v(i, j, k);
        } else if (this->grid.flag_field(i, j, k) & FlagFieldMask::FLUID_WEST) {
            this->grid.v(i, j-1, k) = -this->grid.v(i-1, j-1, k);
            this->grid.v(i, j, k) = -this->grid.v(i-1, j, k);
            this->grid.u(i-1, j, k) = 0.0;
            this->grid.F(i-1, j, k) = this->grid.u(i-1, j, k);
        } else if (this->grid.flag_field(i, j, k) & FlagFieldMask::FLUID_EAST) {
            this->grid.v(i, j-1, k) = -this->grid.v(i + 1, j - 1, k);
            this->grid.v(i, j, k) = -this->grid.v(i + 1, j, k);
            this->grid.u(i + 1, j, k) = 0.0;
            this->grid.F(i + 1, j, k) = this->grid.u(i + 1, j, k);
        } else if (this->grid.flag_field(i, j, k) & FlagFieldMask::FLUID_TOP) {
            this->grid.w(i, j, k-1) = -this->grid.w(i, j, k);
            this->grid.w(i, j-1, k) = -this->grid.w(i, j-1, k);
            this->grid.u(i, j, k) = 0.0;
            this->grid.v(i, j, k) = 0.0;
        } else if (this->grid.flag_field(i, j, k) & FlagFieldMask::FLUID_BOTTOM) {
            this->grid.w(i, j, k-1) = -this->grid.w(i, j, k-1);
            this->grid.w(i, j-1, k) = -this->grid.w(i, j-1, k-1);
            this->grid.u(i, j, k) = 0.0;
            this->grid.v(i, j, k) = 0.0;
        } else {
            // interior obstacle cell, so no-slip
            this->grid.u(i, j, k) = 0.0;
            this->grid.v(i, j, k) = 0.0;
            this->grid.w(i, j, k) = 0.0;
        }
    }

    void FluidSimulation::run() {
        double last_saved = 0.0;
        std::string solver_name = "";

        // Function pointer to solver
        void (CFD::FluidSimulation::*pressure_solver)();

        if (this->solver_type == SolverType::JACOBI) {
            pressure_solver = &FluidSimulation::solveWithJacobi;
            solver_name = "Jacobi";
            std::cout << "Solver: Jacobi (" << this->grid.imax << "x" << this->grid.jmax << "x" << this->grid.kmax << ")" << std::endl;
        }
        else if (this->solver_type == SolverType::MULTIGRID_JACOBI) {
            pressure_solver = &FluidSimulation::solveWithMultigridJacobi;
            solver_name = "Multigrid Jacobi";

            // check if imax, jmax, and kmax are powers of 2, if not throw exception
            if ((this->grid.imax & (this->grid.imax - 1)) != 0 || 
                (this->grid.jmax & (this->grid.jmax - 1)) != 0 ||
                (this->grid.kmax & (this->grid.kmax - 1)) != 0) {
                throw std::invalid_argument("imax, jmax, and kmax must be powers of 2");
            }

            int imax_levels = std::log2(this->grid.imax);
            int jmax_levels = std::log2(this->grid.jmax);
            int kmax_levels = std::log2(this->grid.kmax);
            int levels = std::min(imax_levels, std::min(jmax_levels, kmax_levels));

            this->multigrid_hierarchy = new MultigridHierarchy(levels, &this->grid);

            std::cout << "Solver: Multigrid Jacobi (" << this->grid.imax << "x" << this->grid.jmax << "x" << this->grid.kmax << ")" << std::endl;
        }
        else if (this->solver_type == SolverType::CONJUGATED_GRADIENT) {
            pressure_solver = &FluidSimulation::solveWithConjugatedGradient;
            solver_name = "Conjugated Gradient";

            // check if imax, jmax, and kmax are powers of 2, if not throw exception
            if ((this->grid.imax & (this->grid.imax - 1)) != 0 || 
                (this->grid.jmax & (this->grid.jmax - 1)) != 0 ||
                (this->grid.kmax & (this->grid.kmax - 1)) != 0) {
                throw std::invalid_argument("imax, jmax, and kmax must be powers of 2");
            }

            int imax_levels = std::log2(this->grid.imax);
            int jmax_levels = std::log2(this->grid.jmax);
            int kmax_levels = std::log2(this->grid.kmax);
            int levels = std::min(imax_levels, std::min(jmax_levels, kmax_levels));

            this->multigrid_hierarchy = new MultigridHierarchy(levels, &this->grid);
            std::cout << "Solver: Conjugated Gradient (" << this->grid.imax << "x" << this->grid.jmax << "x" << this->grid.kmax << ")" << std::endl;
        }
        else if (this->solver_type == SolverType::MULTIGRID_PCG) {
            pressure_solver = &FluidSimulation::solveWithMultigridPCG;
            solver_name = "Multigrid PCG";

            // check if imax, jmax, and kmax are powers of 2, if not throw exception
            if ((this->grid.imax & (this->grid.imax - 1)) != 0 || 
                (this->grid.jmax & (this->grid.jmax - 1)) != 0 ||
                (this->grid.kmax & (this->grid.kmax - 1)) != 0) {
                throw std::invalid_argument("imax, jmax, and kmax must be powers of 2");
            }

            int imax_levels = std::log2(this->grid.imax);
            int jmax_levels = std::log2(this->grid.jmax);
            int kmax_levels = std::log2(this->grid.kmax);
            int levels = std::min(imax_levels, std::min(jmax_levels, kmax_levels));

            this->preconditioner = StaggeredGrid(this->grid.imax, this->grid.jmax, this->grid.kmax, this->grid.xlength, this->grid.ylength, this->grid.zlength);

            this->multigrid_hierarchy = new MultigridHierarchy(levels, &this->grid);
            this->multigrid_hierarchy_preconditioner = new MultigridHierarchy(levels, &this->preconditioner);

            std::cout << "Solver: Multigrid PCG (" << this->grid.imax << "x" << this->grid.jmax << "x" << this->grid.kmax << ")" << std::endl;
        }
        else {
            throw std::invalid_argument("Invalid solver type");
        }


        while (this->t < this->t_end) {
            this->lastTimestamp = static_cast<int>(std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count());
            this->selectDtAccordingToStabilityCondition();
            this->setBoundaryConditionsU();
            this->setBoundaryConditionsV();
            this->setBoundaryConditionsW();
            this->setBoundaryConditionsVelocityGeometry();
            this->computeF();
            this->computeG();
            this->computeH();
            this->setBoundaryConditionsVelocityGeometry();
            this->computeRHS();
            this->computeU();
            this->computeV();
            this->computeW();

            (this->*pressure_solver)();

            this->res_norm_over_it_without_pressure_solver(this->it_wo_pressure_solver) = this->res_norm;

            this->computeU();
            this->computeV();
            this->computeW();
            this->setBoundaryConditionsU();
            this->setBoundaryConditionsV();
            this->setBoundaryConditionsW();
            this->setBoundaryConditionsVelocityGeometry();
            this->setBoundaryConditionsPGeometry();
            if (this->t - last_saved >= this->save_interval) {
                std::cout << "Solver: " << solver_name << " t: " << this->t << " dt: " << this->dt << " res: " << this->res_norm << std::endl;
                this->grid.interpolateVelocity();
                saveVTK(this);
                last_saved += this->save_interval;
            }
            this->duration += static_cast<int>(std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count()) - this->lastTimestamp;
            this->res_norm_over_time(this->duration) = this->res_norm;
            this->t = this->t + this->dt;
            this->it_wo_pressure_solver++;
        }

        this->setBoundaryConditionsU();
        this->setBoundaryConditionsV();
        this->setBoundaryConditionsW();
        this->setBoundaryConditionsVelocityGeometry();
        this->setBoundaryConditionsP();
        this->setBoundaryConditionsPGeometry();

        this->grid.interpolateVelocity();

        this->res_norm_over_it_with_pressure_solver.conservativeResize(this->it);
        this->res_norm_over_it_without_pressure_solver.conservativeResize(this->it_wo_pressure_solver);
        this->res_norm_over_time.conservativeResize(this->duration);

        saveVTKGeometry(this);
    }

    void FluidSimulation::saveData() {
        // Additional save commands for 3D can be added similarly...
        Kernel::saveVector("residuals_with_pressure_solver.dat", &this->res_norm_over_it_with_pressure_solver);
        Kernel::saveVector("residuals_without_pressure_solver.dat", &this->res_norm_over_it_without_pressure_solver);
        Kernel::saveVector("residuals_over_time.dat", &this->res_norm_over_time);
    }
}