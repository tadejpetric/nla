#include <iostream>
#include <Eigen/Dense>


using Eigen::Matrix;
using Eigen::Vector;


template <typename T, int n>
Vector<T, n> gauss_seidl(Matrix<T, n, n> A, Vector<T,n> b, Vector<T,n> x0 = Vector<T,n>::Zero()) {
    Vector<T, n> y;
    for (int k = 0; k < 2; ++k) {
        y = b-A*x0;
        y = A.template triangularView<Eigen::Lower>().solve(y);
        x0 = x0 + y;
    }
    return x0;
}

int main() {
    Matrix<double, 3, 3> A;
    A << 12, -3, 1,
            -1,  9, 2,
            1, -1, 10;
    Vector<double, 3> b;
    b << 10, 10, 10;//, -4, 27;
    std::cout << gauss_seidl(A, b, {1,0,1});
}
