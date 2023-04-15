//
// Created by user on 4/15/23.
//

#ifndef NLA_SOR_H
#define NLA_SOR_H


#include <Eigen/Dense>

using Eigen::Matrix;
using Eigen::Vector;


template <typename T, int n>
Vector<T,n> sor(Matrix<T,n,n> A, Vector<T,n> b, T omega, int iters, Vector<T,n> x0 = Vector<T,n>::Zero()) {
    Matrix<T,n,n> R = A * omega;
    R += A.diagonal().asDiagonal() * (1-omega);

    Matrix<T,n,n> rhs = A.template triangularView<Eigen::Upper>();
    rhs *= -omega;
    rhs += A.diagonal().asDiagonal();

    Vector<T,n> y;
    for (int k = 0; k < iters; ++k) {
        y = rhs * x0 + omega*b;
        x0 = R.template triangularView<Eigen::Lower>().solve(y);
    }
    return x0;
}

#endif //NLA_SOR_H
