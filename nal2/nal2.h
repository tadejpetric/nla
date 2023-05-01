//
// Created by user on 4/27/23.
//

#ifndef NLA_NAL2_H
#define NLA_NAL2_H

#include <iostream>
#include <unsupported/Eigen/SparseExtra>
#include <Eigen/Sparse>

using Eigen::SparseMatrix;

bool load_matrix(SparseMatrix<double>& matrix, const char* file) {
    std::string data_location = "../nal2/data/";
    std::string filename = "ncvxbqp1.mtx"; //"data/ncvxbqp1.mtx";
    if (file[0] == '2') filename = "spmsrtls.mtx";
    if (file[1] == '3') filename = "wing.mtx";
    filename = data_location + filename;



    auto x = Eigen::loadMarket(matrix, filename);
    return x;
}


#endif //NLA_NAL2_H
