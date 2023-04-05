#include <iostream>
#include <vector>
#include <math.h>
#include <Eigen/Dense>

//using Eigen::MatrixXd;
//using Eigen::VectorXd;
using Eigen::Matrix;
using Eigen::Vector;

template <typename T, int n, int m>
void display(Matrix<T,n,m> A) {

}
template <typename T, int n>
Vector<T,n> init(Matrix<T, n, n> A, Vector<T,n> b) {
    Vector<T, n> x0 = b;
    for (int i = 0; i < n; ++i) {
        x0(i) /= A(i, i);
    }
    return x0;
}

template <typename T, int n>
Vector<T, n> jacobi(Matrix<T, n, n> A, Vector<T,n> b, Vector<T,n> x0 = Vector<T,n>::Zero()) {
    /*
     *  A = L + D + U
     *  Ax = b => (L+U)x + Dx = b
     *  Dx = b-(L+U)x
     */
    Vector<T, n> y;
    for (int k = 0; k < 2; ++k) {
        y = b - A*x0;
        for (int i = 0; i < n; ++i) {
            y(i) /= A(i,i);
        }
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
    std::cout << jacobi(A, b, {1,0,1});
}
