//
// Created by user on 4/27/23.
//

#ifndef NLA_NAL3_H
#define NLA_NAL3_H
#include <iostream>
#include <limits>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <unsupported/Eigen/SparseExtra>
#include <cmath>


typedef Eigen::SparseMatrix<double> smatrix;

using Eigen::VectorXd;

void readb(VectorXd& b, std::string file) {
    std::ifstream in(file.c_str());
    std::string line;
    int i = 0;
    bool fst = true;
    while (getline(in, line)) {
        if (line.length() < 1) continue;
        if (line[0] == '%') continue;
        if (fst) {
            fst = false;
            continue;
        }
        b(i) = atof(line.c_str());
        ++i;
    }
    in.close();
}


#endif //NLA_NAL3_H
