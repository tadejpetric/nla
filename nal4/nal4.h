//
// Created by user on 4/27/23.
//

#ifndef NLA_NAL4_H
#define NLA_NAL4_H

#include <fstream>
#include <iostream>
#include <unsupported/Eigen/SparseExtra>
#include <Eigen/Sparse>
#include <utility>
#include <cmath>
#include <chrono>

typedef Eigen::SparseMatrix<double> smatrix;
using Eigen::VectorXd;


void load_matrix(smatrix& train, smatrix& test) {

    std::string data_location = "../nal4/";

    Eigen::loadMarket(train, data_location + "train.mtx");
    Eigen::loadMarket(test, data_location + "test.mtx");

}

std::chrono::time_point<std::chrono::high_resolution_clock> start() {
    return std::chrono::high_resolution_clock::now();
}

void stop(std::chrono::time_point<std::chrono::high_resolution_clock> t) {
    using std::chrono::high_resolution_clock;
    using std::chrono::duration_cast;
    using std::chrono::duration;
    using std::chrono::seconds;
    duration<double> delta = high_resolution_clock::now() - t;
    std::cout << "duration: " << delta.count() << "s";
}

template <typename T, typename V>
void save_chart(std::vector<std::tuple<T, V>>& v, const char* file) {
    std::ofstream out(file);

    for(auto&& [x, y] : v) {
        out << "(" << x <<", " << y << ")\n";
    }
    out.close();
}




#endif //NLA_NAL4_H
