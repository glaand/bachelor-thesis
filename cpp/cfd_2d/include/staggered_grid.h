#pragma once

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include <torch/script.h>

namespace CFD {
    using namespace Eigen;

    class StaggeredGrid {
    public:
        StaggeredGrid(
            const int p_imax = 50,
            const int p_jmax = 50,
            const double p_xlength = 1.0,
            const double p_ylength = 1.0
        ) {
            imax = p_imax;
            jmax = p_jmax;
            xlength = p_xlength;
            ylength = p_ylength;
            p = MatrixXf::Zero(imax + 2, jmax + 2);
            po = MatrixXf::Zero(imax + 2, jmax + 2);
            RHS = MatrixXf::Zero(imax + 2, jmax + 2);
            res = MatrixXf::Zero(imax + 2, jmax + 2);
            u = MatrixXf::Zero(imax + 2, jmax + 3);
            F = MatrixXf::Zero(imax + 2, jmax + 3);
            v = MatrixXf::Zero(imax + 3, jmax + 2);
            G = MatrixXf::Zero(imax + 3, jmax + 2);
            u_interpolated = MatrixXf::Zero(imax + 2, jmax + 2);
            v_interpolated = MatrixXf::Zero(imax + 2, jmax + 2);
            flag_field = MatrixXi::Ones(imax + 2, jmax + 2);

            // Conjugated Gradient components
            search_vector = MatrixXf::Zero(imax + 2, jmax + 2);
            Asearch_vector = MatrixXf::Zero(imax + 2, jmax + 2);

            dx = xlength / imax;
            dy = ylength / jmax;
            dx2 = dx * dx;
            dy2 = dy * dy;
            dx2dy2 = dx2 * dy2;
        }
        double findMaxAbsoluteU() const;
        double findMaxAbsoluteV() const;
        void interpolateVelocity();
        double dx;
        double dy;
        double dx2;
        double dy2;
        double dx2dy2;
        int imax;
        int jmax;
        double xlength;
        double ylength;
        MatrixXf p;
        MatrixXf po;
        MatrixXf RHS;
        MatrixXf res;
        MatrixXf u;
        MatrixXf v;
        MatrixXf F;
        MatrixXf G;
        MatrixXf u_interpolated;
        MatrixXf v_interpolated;
        MatrixXi flag_field;

        // Conjugated Gradient components
        MatrixXf search_vector;
        MatrixXf Asearch_vector;
    };
}
