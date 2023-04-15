#include <iostream>
#include <Eigen/Dense>
#include "sor.h"
#include "../gauss_seidl/gauss_seidl.h"
using Eigen::Matrix;
using Eigen::Vector;



int main() {
    constexpr int s = 30;
    std::srand(time(nullptr));
    Matrix<double, s, s> A;

    A = Matrix<double, s, s>::Random();
    A = A.transpose() * A;
    Vector<double, s> b = Vector<double,s>::Random();
    //b << 10, 10, 10;//, -4, 27;
    std::cout << sor(A, b, 1., 50) -  gauss_seidl(A,b,50)<< "\n";//, {1,0,1});
}