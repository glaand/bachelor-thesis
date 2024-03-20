#include "multigrid.h"

using namespace CFD;

void Multigrid::vcycle(MultigridHierarchy *hierarchy, int currentLevel, float omg, int numSweeps) {
    // Briggs, Multigrid Tutorial, p. 41
    // reusing p and RHS for error and residual for simplicity and memory efficiency
    // (Ae = res) and (Ap = RHS)

    if (currentLevel == 0) {
        // Relax on the coarset grid
        relax(hierarchy->grids[0].get(), numSweeps, omg);
    } else {
        // Relax on the current grid
        relax(hierarchy->grids[currentLevel].get(), numSweeps, omg);

        // Restrict the residual to the coarser grid
        restrict_operator(hierarchy->grids[currentLevel].get(), hierarchy->grids[currentLevel-1].get());

        // Recursively call vcycle
        vcycle(hierarchy, currentLevel-1, omg, numSweeps*2);

        // Prolongate the error to the finer grid
        prolongate_operator(hierarchy->grids[currentLevel-1].get(), hierarchy->grids[currentLevel].get());

        // Post-smooth on the current grid
        relax(hierarchy->grids[currentLevel].get(), numSweeps, omg);
    }
}

void Multigrid::restrict_operator(const StaggeredGrid *fine, StaggeredGrid *coarse) {
    // Restrict with full weighting
    // Briggs, Multigrid Tutorial, p. 36

    // Reset coarse p
    for (int i = 0; i < coarse->imax + 2; i++) {
        for (int j = 0; j < coarse->jmax + 2; j++) {
            coarse->p(i,j) = 0;
        }
    }

    // Restrict res^h to RHS^{2h} but saving on RHS^{2h}
    for (int i = 1; i < coarse->imax + 1; i++) {
        for (int j = 1; j < coarse->jmax + 1; j++) {
            coarse->RHS(i,j) = 0.0625*(
                fine->res(2*i-1,2*j-1) + fine->res(2*i-1,2*j+1) + fine->res(2*i+1,2*j-1) + fine->res(2*i+1,2*j+1)
                + 2 * (fine->res(2*i,2*j-1) + fine->res(2*i,2*j+1) + fine->res(2*i-1,2*j) + fine->res(2*i+1,2*j))
                + 4 * fine->res(2*i,2*j)
            );
        }
    }
}

void Multigrid::prolongate_operator(const StaggeredGrid *coarse, StaggeredGrid *fine) {
    // Prolongate with linear interpolation
    // Briggs, Multigrid Tutorial, p. 35

    // Prolongate p^{2h} to p^{2h} adding p^{2h}
    for (int i = 1; i < coarse->imax + 1; i++) {
        for (int j = 1; j < coarse->jmax + 1; j++) {
            fine->p(2*i,2*j) += coarse->p(i,j);
            fine->p(2*i+1,2*j) += 0.5 * (coarse->p(i,j) + coarse->p(i+1,j));
            fine->p(2*i,2*j+1) += 0.5 * (coarse->p(i,j) + coarse->p(i,j+1));
            fine->p(2*i+1,2*j+1) += 0.25 * (coarse->p(i,j) + coarse->p(i+1,j) + coarse->p(i,j+1) + coarse->p(i+1,j+1));
        }
    }

    // Set boundaries to zero
    for (int i = 0; i < fine->imax + 2; i++) {
        fine->p(i,0) = 0;
        fine->p(i,fine->jmax+1) = 0;
    }
    for (int j = 0; j < fine->jmax + 2; j++) {
        fine->p(0,j) = 0;
        fine->p(fine->imax+1,j) = 0;
    }
}

void Multigrid::relax(StaggeredGrid *grid, int numSweeps, float omg) {
    // Relaxation with Jacobi
    // Its in abstract form, so it can be used for (Ae = res) or (Ap = RHS)
    
    for (int sweep = 0; sweep < numSweeps; sweep++) {
        // Jacobi smoother with relaxation factor (omega)
        for (int i = 1; i < grid->imax + 1; i++) {
            for (int j = 1; j < grid->jmax + 1; j++) {
                grid->po(i,j) = grid->p(i,j);
                grid->p(i, j) = (
                    (1.0/(-2.0*grid->dx2 - 2.0*grid->dy2)) // 1/Aii
                    *
                    (
                        grid->RHS(i,j)*grid->dx2dy2 - grid->dx2*(grid->po(i+1,j) + grid->po(i-1,j)) - grid->dy2*(grid->po(i,j+1) + grid->po(i,j-1))
                    )
                );
            }
        }
    }

    // calculate residual
    for (int i = 1; i < grid->imax + 1; i++) {
        for (int j = 1; j < grid->jmax + 1; j++) {
            grid->res(i,j) = grid->RHS(i,j) - (
                // Sparse matrix A
                (1.0/grid->dx2)*(grid->p(i+1,j) - 2.0*grid->p(i,j) + grid->p(i-1,j)) +
                (1.0/grid->dy2)*(grid->p(i,j+1) - 2.0*grid->p(i,j) + grid->p(i,j-1))
            );
        }
    }
}