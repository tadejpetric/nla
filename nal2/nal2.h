//
// Created by user on 4/27/23.
//

#ifndef NLA_NAL2_H
#define NLA_NAL2_H

#include <iostream>
#include <unsupported/Eigen/SparseExtra>
#include <Eigen/Sparse>
#include <utility>
#include <cmath>

typedef Eigen::SparseMatrix<double> smatrix;
using Eigen::VectorXd;

template <int n>
void make_sample(smatrix& A) {
    srand(0);

    A.resize(n,n);
    Eigen::Matrix<double, n, n> x1 = Eigen::Matrix<double,n,n>::Random();
    x1 = x1 + x1.transpose().eval();
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            if (i == j) A.insert(i,i) = i+1;
            //A.insert(i,j) = x1(i,j);
        }
    }
}

void load_matrix(smatrix& matrix, VectorXd& b, const char* file) {

    std::string data_location = "../nal2/data/";
    std::string filename = "ncvxbqp1.mtx"; //"data/ncvxbqp1.mtx";
    if (file[0] == '2') filename = "spmsrtls.mtx";
    if (file[1] == '3') filename = "wing.mtx";

    filename = data_location + filename;

    if (file[0] == '4') make_sample<10>(matrix);
    else
        Eigen::loadMarket(matrix, filename);

    b = VectorXd::Ones(matrix.rows());
}

std::tuple<double, double> givens(double& alpha, double& beta) {
    double r = alpha*alpha + beta*beta;
    r = std::sqrt(r);

    return {alpha/r, -beta/r};
}



#endif //NLA_NAL2_H
