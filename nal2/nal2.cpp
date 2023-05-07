//
// Created by user on 4/27/23.
//

#include "nal2.h"
#include <iostream>
#include <limits>
#include <unsupported/Eigen/SparseExtra>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <cmath>


typedef Eigen::SparseMatrix<double> smatrix;

using Eigen::VectorXd;
using Eigen::Vector3d;

bool small(double x) {
    return (x < 0.00003) && (x > -0.00003);

}


void first_iter(smatrix& A, VectorXd& p, double& alpha, double& beta, VectorXd& v_this, VectorXd& v_last) {
    VectorXd z = A*v_this;
    alpha = v_this.dot(z);
    z = z - alpha*v_this;

    beta = z.norm();
    p = v_this/alpha;

    v_last = v_this;
    v_this = z/beta;
}

void second_iter(smatrix& A,
                 VectorXd& p_this,
                 VectorXd& p_last,
                 double& alpha, double& beta, double& zeta,
                 VectorXd& v_this,
                 VectorXd& v_last,
                 Vector3d& col) {

    VectorXd z = A*v_this - beta*v_last;
    auto [c, s] = givens(alpha, beta);

    alpha = v_this.dot(z);
    z = z - alpha*v_this;

    p_this = v_this - p_last*(c*beta - s*alpha);
    p_this /= (s*beta + c*alpha);
    alpha = s*beta + c*alpha;

    beta = z.norm();

    zeta *= s/c;

    col(0) = -s*beta;
    col(1) =  c*beta;
}


void qr_d_lanczos(smatrix& A, VectorXd& x, VectorXd& b) {
    VectorXd r = b - A*x;
    VectorXd v_this = r / r.norm();
    VectorXd v_last = VectorXd::Zero(r.size());

    VectorXd p_k2;
    VectorXd p_k1;
    VectorXd p_k;

    double alpha_last, beta_last;
    double alpha_this, beta_this;
    double zeta = r.norm();

    Vector3d last_col = Vector3d::Zero();


    // compute first manually
    first_iter(A, p_k2, alpha_last, beta_last, v_this, v_last);
    x += zeta*p_k2;
    if (small(beta_last)) return;
    std::cout << x;
    second_iter(A, p_k1, p_k2, alpha_last, beta_last, zeta, v_this, v_last, last_col);

    x += zeta*p_k1;
    if (small(beta_last)) return;
    std::cout << "\n-\n" << x << "\n-\n";

    for (int j = 2; j < 10; ++j) {

        VectorXd z = A*v_this - beta_last*v_last;
        alpha_this = v_this.dot(z);
        z = z - alpha_this*v_this;
        beta_this = z.norm();

        auto [c, s] = givens(alpha_last, beta_last);

        last_col(2) = s*last_col(1) + c*alpha_this;
        last_col(1) = c*last_col(1) - s*alpha_this;

        alpha_last = c*beta_last + s*alpha_this;

        p_k = v_this - last_col(1)*p_k1 - last_col(0)*p_k2;
        p_k /= last_col(2);

        zeta *= s/c;
        std::cout << "\n-\n";
        std::cout << x;
        std::cout << "-\n";
        x += zeta*p_k;

        last_col(0) = -s*beta_this;
        last_col(1) =  c*beta_this;

        if (small(beta_this)) break;

        v_last = v_this;
        v_this = z/beta_this;
    }
}


void d_lanczos(smatrix& A, VectorXd& x, VectorXd& b) {
    VectorXd r = b - A*x;
    VectorXd v_this = r / r.norm();
    VectorXd v_last = VectorXd::Zero(r.size());
    VectorXd p = VectorXd::Zero(r.size());
    double beta_this;
    double beta_last = 0;
    double zeta = r.norm();

    double lambda = 0;
    double eta = 0;

    for (int j = 0; j < 5000; ++j) {

        VectorXd z = A*v_this - beta_last*v_last;
        double alpha = v_this.dot(z);
        z = z - alpha*v_this;

        beta_this = z.norm();
        if (j >= 1) {
            lambda = beta_last/eta;
        }
        eta = alpha - beta_last*lambda;
        if (eta < 0.00003 && eta > -0.00003) {
            break;
        }
        if (j >= 1) {
            zeta = -lambda * zeta;
        }

        // beta(k-1) p(k-1) + eta(k)p(k) = v(k)
        p = (1/eta)*(v_this - beta_last*p);
        x += zeta*p;

        if (beta_this < 0.00003 && beta_this > -0.00003) {
            break;
        }
        if (std::abs(beta_this * zeta/eta) < std::pow(10.,-10.))
            break;
        v_last = v_this;
        v_this = z/beta_this;
        beta_last = beta_this;

    }
}



int main(int argc, char** argv) {
    bool s = false;

    smatrix A;
    VectorXd b;
    
    if (argc <= 1) {
        char opt = '1';
        load_matrix(A, b, &opt);
    } else {
        load_matrix(A, b,  argv[1]);
    }
    A.makeCompressed();
    std::cout << A.rows() << " " << A.cols() << "\n";
    std::cout << b.rows() << " " << b.cols() << "\n";

    VectorXd x = VectorXd::Zero(A.rows());


    qr_d_lanczos(A, x, b);

    Eigen::SimplicialLDLT<smatrix> solver;
    solver.compute(A);
    VectorXd y = solver.solve(b);
    //std::cout << y;
    Eigen::MatrixXd all(x.size(),2);
    all.col(0) = x;
    all.col(1) = y;
    std::cout << all;
}