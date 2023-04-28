//
// Created by user on 4/27/23.
//

#include "nal1.h"
#include <iostream>

//#include <unsupported/Eigen/SparseExtra>
#include <Eigen/Sparse>
#include <Eigen/Dense>

typedef Eigen::SparseMatrix<double> smatrix;
typedef Eigen::SparseVector<double> svector;



void build_matrix(smatrix& mat, const area& omega) {
    // returns
    // outer upper diagonal
    const int w = omega.divisions * omega.divisions;
    const int d = omega.divisions;

    for (int i = d; i < w; ++i) {
        mat.insert(i-d, i) = 1;
        mat.insert(i, i-d) = 1;
    }

    // inner upper diagonal
    for (int i = 0; i < d; ++i) {
        for (int j = 1; j < d; ++j) {
            mat.insert(i*d + j-1,i*d + j) = 1;
            mat.insert(i*d + j,i*d + j-1) = 1;
        }
    }

    // diagonal
    for (int i = 0; i < d; ++i) {
        for (int j = 0; j < d; ++j) {
            mat.insert(i*d + j,i*d + j) = omega.penalty(i,j)*omega.step_width2 - 4;
        }
    }
}

/*
template <int n>
void build_b(Eigen::Vector<double, n>& b, const area& omega) {
    for (int i = 0; i < n*n; ++i) {
        b.insert(i) = omega.step_width2;
    }
}
*/
template <int n>
void solve_system(smatrix& mat, Eigen::Vector<double, n*n>& x, Eigen::Vector<double, n*n>& b){
    Eigen::SimplicialLDLT<smatrix> solver;
    solver.compute(mat);
    x = solver.solve(b);
}

int main() {
    const int n = 101;
    area omega(n, 100);
    smatrix mat(n*n,n*n);
    Eigen::Vector<double, n*n> b;
    b.setOnes();
    b *= omega.step_width2;

    build_matrix(mat, omega);
    //build_b(b, omega);
    mat.makeCompressed();
    Eigen::Vector<double, n*n> x;
    solve_system<n>(mat, x, b);
    std::cout << x;
}