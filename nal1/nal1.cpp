//
// Created by user on 4/27/23.
//

#include "nal1.h"
#include <iostream>
#include <fstream>
//#include <unsupported/Eigen/SparseExtra>
#include <Eigen/Sparse>
#include <Eigen/Dense>

typedef Eigen::SparseMatrix<double> smatrix;
typedef Eigen::SparseVector<double> svector;
using Eigen::VectorXd;



void build_matrix(smatrix& mat, const area& omega) {
    // returns

    using Eigen::Triplet;

    const int w = omega.divisions * omega.divisions;
    const int d = omega.divisions;
    std::vector<Triplet<double>> triplets;

    // outer upper diagonal
    for (int i = d; i < w; ++i) {
        triplets.emplace_back(i-d, i, 1);
        triplets.emplace_back(i,i-d,1);
    }

    for (int i = 0; i < d; ++i) {
        for (int j = 1; j < d; ++j) {
            // inner upper diagonal
            triplets.emplace_back(i*d+j-1, i*d+j,1);
            triplets.emplace_back(i*d+j, i*d+j-1,1);

        }
    }
    for (int i = 0; i < d; ++i) {
        for (int j = 0; j < d; ++j) {
            triplets.emplace_back(i*d+j, i*d+j, omega.penalty(i,j)*omega.step_width2 - 4);
        }
    }

    mat.setFromTriplets(triplets.begin(), triplets.end());
}


void gauss_seidl(smatrix& A, VectorXd& x0, VectorXd& b, int iters) {
    VectorXd y(A.cols());
    for (int k = 0; k < iters; ++k) {
        y = b-A*x0;
        y = A.template triangularView<Eigen::Lower>().solve(y);
        x0 = y;
    }
}

void gauss_jacobi(smatrix& A, VectorXd& x0, VectorXd& b, int iters) {
    VectorXd y(A.cols());
    VectorXd D = A.diagonal();
    for (int k = 0; k < iters; ++k) {
        y = b-A*x0;
        y = D.cwiseInverse().cwiseProduct(y).eval();
        x0 = x0 + y;
    }
}




template <typename vec> // for static vec = Eigen::Vector<double, n*n>
void solve_system(smatrix& mat, vec& x, vec& b){
    // this is Eigen's default method
    Eigen::SparseLU<smatrix> solver;
    solver.compute(mat);
    x = solver.solve(b);
}

int main() {
    const int n = 600;


    area omega(n, 600);
    smatrix mat(n*n,n*n);

    VectorXd b(n*n);
    b.setOnes();
    b *= omega.step_width2;

    auto p = start();
    build_matrix(mat, omega);

    mat.makeCompressed();
    VectorXd x(n*n);

    auto t = start();
    solve_system(mat, x, b);
    stop(t);

    auto t2 = start();
    VectorXd x2(n*n);
    x2.setZero();
    gauss_seidl(mat, x2, b, 7);
    stop(t2);



    //std::cout << "\n" << (x-x2).norm() << "\n";


    std::ofstream out("../nal1/outmat.txt");
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            out << x(i*n + j) << " ";
        }
        out << "\n";
    }
    out.close();

}