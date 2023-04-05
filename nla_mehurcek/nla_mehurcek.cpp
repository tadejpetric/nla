#include <iostream>
#include <vector>
#include <math.h>
#include <Eigen/Dense>
#include <matplot/matplot.h>
using Eigen::MatrixXd;
using Eigen::VectorXd;

MatrixXd gauss_seidel(MatrixXd U) {
    for (int iteration = 0; iteration < 2000; ++iteration) {
        for (int i = 1; i < U.rows()-1; ++i) {
            for (int j = 1; j < U.cols()-1; ++j)
                U(i,j) = (U(i-1,j)+U(i+1,j)+U(i,j-1)+U(i,j+1))/4.0;
        }
    }
    return U;
}

double reparametrise_to_double(double start, double end, int new_end, int current_step) {
    return start + current_step*(end-start)/(new_end);
}
int reparametrise_to_int(double start, double end, int new_end, double current_step) {
    return std::round((end-start)/(current_step-start) * new_end);
}

MatrixXd setup_matrix(int divisions) {
    auto U = MatrixXd(divisions,divisions).setZero();
    for (int i = 0; i < divisions; ++i) {
        double x = reparametrise_to_double(-1, 1, divisions-1, i);
        U(i,divisions-1) = 1-x*x;
        U(divisions-1, i) = 2-2*x*x;
    }
    return U;
}

std::vector<std::vector<double>> matrix_to_vector(MatrixXd matrika) {
    auto x = std::vector<std::vector<double>>();
    for (int i = 0; i < matrika.rows(); ++i) {
        auto row = std::vector<double>();
        for (int j = 0; j < matrika.cols(); ++j) {
            row.emplace_back(matrika(i,j));
        }
        x.emplace_back(row);
    }
    return x;
}

int main()
{
    int divisions = 20;
    auto matrika = setup_matrix(divisions);
    matrika = gauss_seidel(matrika);

    //matplot::fmesh([&matrika](int x, int y) {return matrika(x,y);},{1, 98, 1, 98});
    using matplot::meshgrid;
    using matplot::linspace;
    auto [X, Y] = meshgrid(linspace(-1, +1, divisions), linspace(-1, +1, divisions));
    for (auto x: X) {
        for (auto y: x) {
            std::cout << y << " ";
        }
        std::cout << "\n---\n";
    }
//matplot::surf(X,Y, matrix_to_vector(matrika));
    //  matplot::show();

    /*
    MatrixXd m(2,2);
    m(0,0) = 3;
    m(1,0) = 2.5;
    m(0,1) = -1;
    m(1,1) = m(1,0) + m(0,1);
    std::cout << m << std::endl;
     */
}