//
// Created by user on 4/15/23.
//

#ifndef NLA_GAUSS_SEIDL_H
#define NLA_GAUSS_SEIDL_H
#include <Eigen/Dense>


using Eigen::Matrix;
using Eigen::Vector;

template <typename T, int n>
Vector<T, n> gauss_seidl(Matrix<T, n, n> A, Vector<T,n> b, int iters, Vector<T,n> x0 = Vector<T,n>::Zero()) {
    Vector<T, n> y;
    for (int k = 0; k < iters; ++k) {
        y = b-A*x0;
        y = A.template triangularView<Eigen::Lower>().solve(y);
        x0 = x0 + y;
    }
    return x0;
}

#endif //NLA_GAUSS_SEIDL_H
