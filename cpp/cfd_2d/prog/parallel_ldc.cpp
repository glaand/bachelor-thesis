#include <iostream>
#include <cmath>
#include "simulations.h"
#include "mpi.h"

using namespace CFD;

void FluidWithoutObstacles2D::setBoundaryConditionsU() {
    int world_rank_x = world_rank % proc_grid_x;
    int world_rank_y = world_rank / proc_grid_x;

    // Identify the neighboring processes
    int left_neighbor = (world_rank_x == 0) ? MPI_PROC_NULL : world_rank - 1;
    int right_neighbor = (world_rank_x == proc_grid_x - 1) ? MPI_PROC_NULL : world_rank + 1;
    int bottom_neighbor = (world_rank_y == 0) ? MPI_PROC_NULL : world_rank - proc_grid_x;
    int top_neighbor = (world_rank_y == proc_grid_y - 1) ? MPI_PROC_NULL : world_rank + proc_grid_x;

    // Communicate boundary values with neighboring processes
    std::vector<double> send_left(this->grid.jmax + 3), send_right(this->grid.jmax + 3);
    std::vector<double> recv_left(this->grid.jmax + 3), recv_right(this->grid.jmax + 3);

    std::vector<double> send_bottom(this->grid.imax + 2), send_top(this->grid.imax + 2);
    std::vector<double> recv_bottom(this->grid.imax + 2), recv_top(this->grid.imax + 2);

    // Fill send buffers with u-velocity values from the boundary cells
    for (int j = 0; j < this->grid.jmax + 3; j++) {
        send_left[j] = this->grid.u(1, j);
        send_right[j] = this->grid.u(this->grid.imax, j);
    }

    for (int i = 0; i < this->grid.imax + 2; i++) {
        send_bottom[i] = this->grid.u(i, 1);
        send_top[i] = this->grid.u(i, this->grid.jmax + 1);
    }

    // Communicate boundary values with neighboring processes
    MPI_Sendrecv(send_left.data(), this->grid.jmax + 3, MPI_DOUBLE, left_neighbor, 0,
                 recv_right.data(), this->grid.jmax + 3, MPI_DOUBLE, right_neighbor, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Sendrecv(send_right.data(), this->grid.jmax + 3, MPI_DOUBLE, right_neighbor, 1,
                 recv_left.data(), this->grid.jmax + 3, MPI_DOUBLE, left_neighbor, 1,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    MPI_Sendrecv(send_bottom.data(), this->grid.imax + 2, MPI_DOUBLE, bottom_neighbor, 2,
                 recv_top.data(), this->grid.imax + 2, MPI_DOUBLE, top_neighbor, 2,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Sendrecv(send_top.data(), this->grid.imax + 2, MPI_DOUBLE, top_neighbor, 3,
                 recv_bottom.data(), this->grid.imax + 2, MPI_DOUBLE, bottom_neighbor, 3,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // Use received values to set boundary conditions for the current subdomain
    for (int j = 0; j < this->grid.jmax + 3; j++) {
        if (left_neighbor != MPI_PROC_NULL) {
            this->grid.u(0, j) = recv_left[j];
        }
        if (right_neighbor != MPI_PROC_NULL) {
            this->grid.u(this->grid.imax + 1, j) = recv_right[j];
        }
    }

    for (int i = 0; i < this->grid.imax + 2; i++) {
        if (bottom_neighbor != MPI_PROC_NULL) {
            this->grid.u(i, 0) = recv_bottom[i];
        }
        if (top_neighbor != MPI_PROC_NULL) {
            this->grid.u(i, this->grid.jmax + 2) = recv_top[i];
        }
    }

    // Fill send buffers with F values from the boundary cells
    for (int j = 0; j < this->grid.jmax + 3; j++) {
        send_left[j] = this->grid.F(1, j);
        send_right[j] = this->grid.F(this->grid.imax, j);
    }

    for (int i = 0; i < this->grid.imax + 2; i++) {
        send_bottom[i] = this->grid.F(i, 1);
        send_top[i] = this->grid.F(i, this->grid.jmax + 1);
    }

    // Communicate boundary values with neighboring processes
    MPI_Sendrecv(send_left.data(), this->grid.jmax + 3, MPI_DOUBLE, left_neighbor, 4,
                 recv_right.data(), this->grid.jmax + 3, MPI_DOUBLE, right_neighbor, 4,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Sendrecv(send_right.data(), this->grid.jmax + 3, MPI_DOUBLE, right_neighbor, 5,
                 recv_left.data(), this->grid.jmax + 3, MPI_DOUBLE, left_neighbor, 5,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Sendrecv(send_bottom.data(), this->grid.imax + 2, MPI_DOUBLE, bottom_neighbor, 6,
                 recv_top.data(), this->grid.imax + 2, MPI_DOUBLE, top_neighbor, 6,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Sendrecv(send_top.data(), this->grid.imax + 2, MPI_DOUBLE, top_neighbor, 7,
                 recv_bottom.data(), this->grid.imax + 2, MPI_DOUBLE, bottom_neighbor, 7,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    
    // Use received values to set boundary conditions for the current subdomain
    for (int j = 0; j < this->grid.jmax + 3; j++) {
        if (left_neighbor != MPI_PROC_NULL) {
            this->grid.F(0, j) = recv_left[j];
        }
        if (right_neighbor != MPI_PROC_NULL) {
            this->grid.F(this->grid.imax + 1, j) = recv_right[j];
        }
    }

    for (int i = 0; i < this->grid.imax + 2; i++) {
        if (bottom_neighbor != MPI_PROC_NULL) {
            this->grid.F(i, 0) = recv_bottom[i];
        }
        if (top_neighbor != MPI_PROC_NULL) {
            this->grid.F(i, this->grid.jmax + 2) = recv_top[i];
        }
    }

    // No-slip boundary conditions for u-velocity
    for (int j = 0; j < this->grid.jmax + 3; j++) {
        // Left wall (only for process at the left edge)
        if (world_rank_x == 0) {
            this->grid.u(0, j) = 0.0;
            this->grid.F(0, j) = 0.0;
        }
        // Right wall (only for process at the right edge)
        if (world_rank_x == proc_grid_x - 1) {
            this->grid.u(this->grid.imax + 1, j) = 0.0;
            this->grid.F(this->grid.imax + 1, j) = 0.0;
        }
    }

    for (int i = 0; i < this->grid.imax + 2; i++) {
        // Bottom wall (only for process at the bottom edge)
        if (world_rank_y == 0) {
            this->grid.u(i, 0) = -this->grid.u(i, 1);
            this->grid.F(i, 0) = -this->grid.F(i, 1);
        }
        // Top wall (only for process at the top edge)
        if (world_rank_y == proc_grid_y - 1) {
            this->grid.u(i, this->grid.jmax + 2) = 2.0 - this->grid.u(i, this->grid.jmax+1);
            this->grid.F(i, this->grid.jmax + 2) = 2.0 - this->grid.F(i, this->grid.jmax+1);
        }
    }

    //std::cout << "Rank " << world_rank << " has u: \n" << this->grid.u.transpose().colwise().reverse() << std::endl;
}

void FluidWithoutObstacles2D::setBoundaryConditionsV() {
    int world_rank_x = world_rank % proc_grid_x;
    int world_rank_y = world_rank / proc_grid_x;

    // Identify the neighboring processes
    int left_neighbor = (world_rank_x == 0) ? MPI_PROC_NULL : world_rank - 1;
    int right_neighbor = (world_rank_x == proc_grid_x - 1) ? MPI_PROC_NULL : world_rank + 1;
    int bottom_neighbor = (world_rank_y == 0) ? MPI_PROC_NULL : world_rank - proc_grid_x;
    int top_neighbor = (world_rank_y == proc_grid_y - 1) ? MPI_PROC_NULL : world_rank + proc_grid_x;

    // Communicate boundary values with neighboring processes
    std::vector<double> send_left(this->grid.jmax + 2), send_right(this->grid.jmax + 2);
    std::vector<double> recv_left(this->grid.jmax + 2), recv_right(this->grid.jmax + 2);

    std::vector<double> send_bottom(this->grid.imax + 3), send_top(this->grid.imax + 3);
    std::vector<double> recv_bottom(this->grid.imax + 3), recv_top(this->grid.imax + 3);

    // Fill send buffers with v-velocity values from the boundary cells
    for (int j = 0; j < this->grid.jmax + 2; j++) {
        send_left[j] = this->grid.v(1, j);
        send_right[j] = this->grid.v(this->grid.imax + 1, j);
    }

    for (int i = 0; i < this->grid.imax + 3; i++) {
        send_bottom[i] = this->grid.v(i, 1);
        send_top[i] = this->grid.v(i, this->grid.jmax);
    }

    // Communicate boundary values with neighboring processes
    MPI_Sendrecv(send_left.data(), this->grid.jmax + 2, MPI_DOUBLE, left_neighbor, 4,
                 recv_right.data(), this->grid.jmax + 2, MPI_DOUBLE, right_neighbor, 4,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Sendrecv(send_right.data(), this->grid.jmax + 2, MPI_DOUBLE, right_neighbor, 5,
                 recv_left.data(), this->grid.jmax + 2, MPI_DOUBLE, left_neighbor, 5,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    MPI_Sendrecv(send_bottom.data(), this->grid.imax + 3, MPI_DOUBLE, bottom_neighbor, 6,
                 recv_top.data(), this->grid.imax + 3, MPI_DOUBLE, top_neighbor, 6,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Sendrecv(send_top.data(), this->grid.imax + 3, MPI_DOUBLE, top_neighbor, 7,
                 recv_bottom.data(), this->grid.imax + 3, MPI_DOUBLE, bottom_neighbor, 7,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // Use received values to set boundary conditions for the current subdomain
    for (int j = 0; j < this->grid.jmax + 2; j++) {
        if (left_neighbor != MPI_PROC_NULL) {
            this->grid.v(0, j) = recv_left[j];
        }
        if (right_neighbor != MPI_PROC_NULL) {
            this->grid.v(this->grid.imax + 2, j) = recv_right[j];
        }
    }

    for (int i = 0; i < this->grid.imax + 3; i++) {
        if (bottom_neighbor != MPI_PROC_NULL) {
            this->grid.v(i, 0) = recv_bottom[i];
        }
        if (top_neighbor != MPI_PROC_NULL) {
            this->grid.v(i, this->grid.jmax + 1) = recv_top[i];
        }
    }

    // Fill send buffers with G values from the boundary cells
    for (int j = 0; j < this->grid.jmax + 2; j++) {
        send_left[j] = this->grid.G(1, j);
        send_right[j] = this->grid.G(this->grid.imax + 1, j);
    }

    for (int i = 0; i < this->grid.imax + 3; i++) {
        send_bottom[i] = this->grid.G(i, 1);
        send_top[i] = this->grid.G(i, this->grid.jmax);
    }

    // Communicate boundary values with neighboring processes
    MPI_Sendrecv(send_left.data(), this->grid.jmax + 2, MPI_DOUBLE, left_neighbor, 8,
                 recv_right.data(), this->grid.jmax + 2, MPI_DOUBLE, right_neighbor, 8,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    MPI_Sendrecv(send_right.data(), this->grid.jmax + 2, MPI_DOUBLE, right_neighbor, 9,
                 recv_left.data(), this->grid.jmax + 2, MPI_DOUBLE, left_neighbor, 9,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    MPI_Sendrecv(send_bottom.data(), this->grid.imax + 3, MPI_DOUBLE, bottom_neighbor, 10,
                 recv_top.data(), this->grid.imax + 3, MPI_DOUBLE, top_neighbor, 10,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE); 

    MPI_Sendrecv(send_top.data(), this->grid.imax + 3, MPI_DOUBLE, top_neighbor, 11,
                    recv_bottom.data(), this->grid.imax + 3, MPI_DOUBLE, bottom_neighbor, 11,
                    MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // Use received values to set boundary conditions for the current subdomain
    for (int j = 0; j < this->grid.jmax + 2; j++) {
        if (left_neighbor != MPI_PROC_NULL) {
            this->grid.G(0, j) = recv_left[j];
        }
        if (right_neighbor != MPI_PROC_NULL) {
            this->grid.G(this->grid.imax + 2, j) = recv_right[j];
        }
    }

    for (int i = 0; i < this->grid.imax + 3; i++) {
        if (bottom_neighbor != MPI_PROC_NULL) {
            this->grid.G(i, 0) = recv_bottom[i];
        }
        if (top_neighbor != MPI_PROC_NULL) {
            this->grid.G(i, this->grid.jmax + 1) = recv_top[i];
        }
    }

    // No-slip boundary conditions for v-velocity
    for (int j = 0; j < this->grid.jmax + 2; j++) {
        // Left wall (only for process at the left edge)
        if (world_rank_x == 0) {
            this->grid.v(0, j) = -this->grid.v(1, j);
            this->grid.G(0, j) = -this->grid.G(1, j);
        }
        // Right wall (only for process at the right edge)
        if (world_rank_x == proc_grid_x - 1) {
            this->grid.v(this->grid.imax + 2, j) = -this->grid.v(this->grid.imax + 1, j);
            this->grid.G(this->grid.imax + 2, j) = -this->grid.G(this->grid.imax + 1, j);
        }
    }

    for (int i = 0; i < this->grid.imax + 3; i++) {
        // Bottom wall (only for process at the bottom edge)
        if (world_rank_y == 0) {
            this->grid.v(i, 0) = 0.0;
            this->grid.G(i, 0) = 0.0;
        }
        // Top wall (only for process at the top edge)
        if (world_rank_y == proc_grid_y - 1) {
            this->grid.v(i, this->grid.jmax + 1) = 0.0;
            this->grid.G(i, this->grid.jmax + 1) = 0.0;
        }
    }
}


void FluidWithoutObstacles2D::setBoundaryConditionsP() {
    int world_rank_x = world_rank % proc_grid_x;
    int world_rank_y = world_rank / proc_grid_x;

    int left_neighbor = (world_rank_x == 0) ? MPI_PROC_NULL : world_rank - 1;
    int right_neighbor = (world_rank_x == proc_grid_x - 1) ? MPI_PROC_NULL : world_rank + 1;
    int bottom_neighbor = (world_rank_y == 0) ? MPI_PROC_NULL : world_rank - proc_grid_x;
    int top_neighbor = (world_rank_y == proc_grid_y - 1) ? MPI_PROC_NULL : world_rank + proc_grid_x;

    // Buffer arrays to send and receive boundary data
    std::vector<double> send_left(this->grid.jmax + 2), send_right(this->grid.jmax + 2);
    std::vector<double> recv_left(this->grid.jmax + 2), recv_right(this->grid.jmax + 2);

    std::vector<double> send_bottom(this->grid.imax + 2), send_top(this->grid.imax + 2);
    std::vector<double> recv_bottom(this->grid.imax + 2), recv_top(this->grid.imax + 2);

    // Fill send buffers with pressure values from the boundary cells
    for (int j = 0; j < this->grid.jmax + 2; j++) {
        send_left[j] = this->grid.p(1, j);
        send_right[j] = this->grid.p(this->grid.imax, j);
    }

    for (int i = 0; i < this->grid.imax + 2; i++) {
        send_bottom[i] = this->grid.p(i, 1);
        send_top[i] = this->grid.p(i, this->grid.jmax);
    }

    // Communicate boundary values with neighboring processes for grid.p
    MPI_Sendrecv(send_left.data(), this->grid.jmax + 2, MPI_DOUBLE, left_neighbor, 0,
                 recv_right.data(), this->grid.jmax + 2, MPI_DOUBLE, right_neighbor, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Sendrecv(send_right.data(), this->grid.jmax + 2, MPI_DOUBLE, right_neighbor, 1,
                 recv_left.data(), this->grid.jmax + 2, MPI_DOUBLE, left_neighbor, 1,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    MPI_Sendrecv(send_bottom.data(), this->grid.imax + 2, MPI_DOUBLE, bottom_neighbor, 2,
                 recv_top.data(), this->grid.imax + 2, MPI_DOUBLE, top_neighbor, 2,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Sendrecv(send_top.data(), this->grid.imax + 2, MPI_DOUBLE, top_neighbor, 3,
                 recv_bottom.data(), this->grid.imax + 2, MPI_DOUBLE, bottom_neighbor, 3,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // Use received values to set boundary conditions for the current subdomain
    for (int j = 0; j < this->grid.jmax + 2; j++) {
        if (left_neighbor != MPI_PROC_NULL) {
            this->grid.p(0, j) = recv_left[j];
        }
        if (right_neighbor != MPI_PROC_NULL) {
            this->grid.p(this->grid.imax + 1, j) = recv_right[j];
        }
    }

    for (int i = 0; i < this->grid.imax + 2; i++) {
        if (bottom_neighbor != MPI_PROC_NULL) {
            this->grid.p(i, 0) = recv_bottom[i];
        }
        if (top_neighbor != MPI_PROC_NULL) {
            this->grid.p(i, this->grid.jmax + 1) = recv_top[i];
        }
    }

    
    for (int i = 0; i < this->grid.imax + 2; i++) {
        if (world_rank_y == 0) {
            this->grid.p(i, 0) = this->grid.p(i, 1);
        }
        if (world_rank_y == proc_grid_y - 1) {
            this->grid.p(i, this->grid.jmax + 1) = this->grid.p(i, this->grid.jmax);
        }
    }
    for (int j = 0; j < this->grid.jmax + 2; j++) {
        if (world_rank_x == 0) {
            this->grid.p(0, j) = this->grid.p(1, j);
        }
        if (world_rank_x == proc_grid_x - 1) {
            this->grid.p(this->grid.imax + 1, j) = this->grid.p(this->grid.imax, j);
        }
    }
}

void FluidWithoutObstacles2D::run() {
    FluidSimulation::run();
}

int main(int argc, char* argv[]) {
    FluidParams flow_params = FluidParams("parallel_ldc", argc, argv);
    FluidWithoutObstacles2D sim = FluidWithoutObstacles2D(flow_params);
    sim.run();
    if (sim.world_rank == 0) {
        sim.saveData();
    }
    return 0;
}
