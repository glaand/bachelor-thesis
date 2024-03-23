#include "cfd.h"
#include <iostream>

using namespace CFD;

int main() {
    // Initialize the problem
    StaggeredGrid grid( 32, 32, 1.0, 1.0 );
    MultigridHierarchy * multigrid_hierarchy = new MultigridHierarchy(5, &grid);

    // Set the RHS
    for (int i = 1; i < grid.imax + 1; i++) {
        for (int j = 1; j < grid.jmax + 1; j++) {
            grid.RHS(i,j) = 1.0;
        }
    }

    // Solve the problem
    float e_norm = 9999999999.0;
    while (e_norm > 1e-3) {
        Multigrid::vcycle(multigrid_hierarchy, multigrid_hierarchy->numLevels() - 1, 1.0, 1);
        e_norm = grid.res.norm();
        std::cout << "Error norm: " << e_norm << std::endl;
    }

    // Print the solution
    std::cout << "Solution (Multigrid):" << std::endl;

    // True solution
    Eigen::VectorXf RHS = Eigen::VectorXf::Ones(grid.imax*grid.jmax);
    Eigen::VectorXf x = Eigen::VectorXf::Zero(grid.imax*grid.jmax);
    Eigen::MatrixXf A = Eigen::MatrixXf::Zero(grid.imax*grid.jmax, grid.imax*grid.jmax);

    // Fill the matrix A with the Laplacian operator
    for (int i = 0; i < grid.imax; i++) {
        for (int j = 0; j < grid.jmax; j++) {
            int idx = i*grid.jmax + j;
            A(idx, idx) = -2.0/grid.dx2 - 2.0/grid.dy2;
            if (i > 0) {
                A(idx, idx - grid.jmax) = 1.0/grid.dx2;
            }
            if (i < grid.imax - 1) {
                A(idx, idx + grid.jmax) = 1.0/grid.dx2;
            }
            if (j > 0) {
                A(idx, idx - 1) = 1.0/grid.dy2;
            }
            if (j < grid.jmax - 1) {
                A(idx, idx + 1) = 1.0/grid.dy2;
            }
        }
    }

    // Solve the system
    x = A.inverse() * RHS;

    // Print the solution
    Eigen::Map<Eigen::MatrixXf> x_map(x.data(), grid.imax, grid.jmax);
    std::cout << "Solution (Conjugate Gradient [EIGEN]):" << std::endl;

    // Calculate error
    Eigen::MatrixXf error = Eigen::MatrixXf::Zero(grid.imax, grid.jmax);
    for (int i = 0; i < grid.imax; i++) {
        for (int j = 0; j < grid.jmax; j++) {
            error(i,j) = grid.p(i+1,j+1) - x_map(i,j);
        }
    }

    // Print the error norm
    std::cout << "Error norm: " << error.norm() << std::endl;
    
    return 0;
}