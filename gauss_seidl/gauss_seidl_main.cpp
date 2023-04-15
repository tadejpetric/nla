#include <iostream>
#include <Eigen/Dense>
#include "gauss_seidl.h"

using Eigen::Matrix;
using Eigen::Vector;




int main() {
    Matrix<double, 3, 3> A;
    A << 12, -3, 1,
            -1,  9, 2,
            1, -1, 10;
    Vector<double, 3> b;
    b << 10, 10, 10;//, -4, 27;
    std::cout << gauss_seidl(A, b, 2,{1,0,1});
}
