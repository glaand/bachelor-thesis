#pragma once

#include <unsupported/Eigen/CXX11/Tensor>
#include <Eigen/Dense>
#include <Eigen/Sparse>

namespace CFD {
    using namespace Eigen;

    class StaggeredGrid {
    public:
        StaggeredGrid(
            int p_imax = 50,
            int p_jmax = 50,
            int p_kmax = 50,
            double p_xlength = 1.0,
            double p_ylength = 1.0,
            double p_zlength = 1.0
        ) : imax(p_imax), jmax(p_jmax), kmax(p_kmax),
            xlength(p_xlength), ylength(p_ylength), zlength(p_zlength),
            p(imax + 2, jmax + 2, kmax + 2), po(imax + 2, jmax + 2, kmax + 2),
            RHS(imax + 2, jmax + 2, kmax + 2), res(imax + 2, jmax + 2, kmax + 2),
            u(imax + 2, jmax + 3, kmax + 2), F(imax + 2, jmax + 3, kmax + 2),
            v(imax + 3, jmax + 2, kmax + 2), G(imax + 3, jmax + 2, kmax + 2),
            w(imax + 2, jmax + 2, kmax + 3), H(imax + 2, jmax + 2, kmax + 3),
            u_interpolated(imax + 2, jmax + 2, kmax + 2),
            v_interpolated(imax + 2, jmax + 2, kmax + 2),
            w_interpolated(imax + 2, jmax + 2, kmax + 2),
            flag_field(imax + 2, jmax + 2, kmax + 2), // initialize with the appropriate values
            search_vector(imax + 2, jmax + 2, kmax + 2),
            Asearch_vector(imax + 2, jmax + 2, kmax + 2) {
                flag_field.setConstant(1.0);
                dx = xlength / imax;
                dy = ylength / jmax;
                dz = zlength / kmax;
                dx2 = dx * dx;
                dy2 = dy * dy;
                dz2 = dz * dz;
                dx2dy2 = dx2 * dy2;
                dx2dz2 = dx2 * dz2; 
                dy2dz2 = dy2 * dz2;
                dx2dy2dz2 = dx2 * dy2 * dz2;

        }
        double findMaxAbsoluteU() const;
        double findMaxAbsoluteV() const;
        double findMaxAbsoluteW() const;
        void interpolateVelocity();
        double dx;
        double dy;
        double dz;
        double dx2;
        double dy2;
        double dz2;
        double dx2dy2;
        double dx2dz2;
        double dy2dz2;
        double dx2dy2dz2;
        int imax;
        int jmax;
        int kmax;
        double xlength;
        double ylength;
        double zlength;
        Tensor<double, 3> p;
        Tensor<double, 3> po;
        Tensor<double, 3> RHS;
        Tensor<double, 3> res;
        Tensor<double, 3> u;
        Tensor<double, 3> v;
        Tensor<double, 3> w;
        Tensor<double, 3> F;
        Tensor<double, 3> G;
        Tensor<double, 3> H;
        Tensor<double, 3> u_interpolated;
        Tensor<double, 3> v_interpolated;
        Tensor<double, 3> w_interpolated;
        Tensor<int, 3> flag_field;

        // Conjugated Gradient components
        Tensor<double, 3> search_vector;
        Tensor<double, 3> Asearch_vector;
    };
}
