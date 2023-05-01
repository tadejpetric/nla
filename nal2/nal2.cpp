//
// Created by user on 4/27/23.
//

#include "nal2.h"
#include <iostream>
#include <limits>
#include <unsupported/Eigen/SparseExtra>
#include <Eigen/Sparse>
#include <Eigen/Dense>


typedef Eigen::SparseMatrix<double> smatrix;

using Eigen::Vector;

template <int n>
void d_lanczos(smatrix& A, Vector<double, n>& x, Vector<double, n>& b) {
    Vector<double, n> r = b - A*x;
    Vector<double, n> v_this = r / r.norm();
    Vector<double, n> v_last = Vector<double, n>::Zero();
    Vector<double, n> p;
    double beta_this;
    double beta_last = 0;
    double zeta = r.norm();

    double lambda = 0;
    double eta = 0;

    for (int j = 0; j < 50; ++j) {
        Vector<double, n> z = A*v_this;
        double alpha = v_this.dot(z);
        z = z - alpha*v_this - beta_last*v_last;

        beta_this = z.norm();
        if (j > 1) {
            lambda = beta_last/eta;
        }
        eta = alpha - beta_last*lambda;
        if (eta < 0.00003 && eta > -0.00003) {
            std::cout << "eta";
            break;
        }
        if (j > 1) {
            zeta = -lambda * zeta;
        }
        p = (1/eta)*(v_this - beta_last*p);
        x += zeta*p;
        std::cout << "---\n" << x << "\n---\n";

        if (beta_this < 0.00003 && beta_this > -0.00003) {
            std::cout << "beta_this";
            break;
        }
        v_last = v_this;
        v_this = z/beta_this;
        beta_last = beta_this;
    }
}

int main(int argc, char** argv) {
    /*bool s = false;
    smatrix mat;
    if (argc <= 1) {
        char opt = '1';
        load_matrix(mat, &opt);
    } else {
        load_matrix(mat, argv[1]);
    }
    mat.makeCompressed();
    std::cout << mat.rows() << " " << mat.cols();

    for (int i = 0; i < 100; ++i) {
        for (int j = 0; j < i; ++j) {
        }
    }*/
    srand(0);
    const int i1 = 5;

    Eigen::Matrix<double, i1, i1> x1 = Eigen::Matrix<double,i1,i1>::Random();
    x1 = x1 + x1.transpose().eval();
    smatrix A(i1, i1);
    for (int j = 0; j < i1; ++j) {
        for (int i = 0; i < i1; ++i) {
            if (i == j) A.insert(i,i) = i+1;
            //A.insert(i,j) = x1(i,j);
        }
    }
    Vector<double, i1> x = Vector<double, i1>::Zero();
    Vector<double, i1> b = Vector<double, i1>::Ones();
    d_lanczos(A, x, b);
    std::cout << A << "\n";
    //std::cout << x;
    std::cout << "\n-----\n";
    Eigen::SimplicialLDLT<smatrix> solver;
    solver.compute(A);
    Vector<double, i1> y = solver.solve(b);
    //std::cout << y;
    Eigen::Matrix<double, i1, 2> all;
    all.col(0) = x;
    all.col(1) = y;
    std::cout << all;
}