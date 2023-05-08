//
// Created by user on 4/27/23.
//

#ifndef NLA_NAL4_H
#define NLA_NAL4_H

#include <iostream>
#include <unsupported/Eigen/SparseExtra>
#include <Eigen/Sparse>
#include <utility>
#include <cmath>

typedef Eigen::SparseMatrix<double> smatrix;
using Eigen::VectorXd;


void load_matrix(smatrix& train, smatrix& test) {

    std::string data_location = "../nal4/";

    Eigen::loadMarket(train, data_location + "train.mtx");
    Eigen::loadMarket(test, data_location + "test.mtx");

}

#endif //NLA_NAL4_H
