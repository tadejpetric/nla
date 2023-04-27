//
// Created by user on 4/27/23.
//

#include "nal2.h"
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

int main(int argc, char** argv) {
    bool s = false;
    Eigen::SparseMatrix<double> mat;
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
    }
}