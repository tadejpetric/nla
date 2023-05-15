//
// Created by user on 4/27/23.
//

#include "nal2.h"
#include <iostream>
#include <limits>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <cmath>


typedef Eigen::SparseMatrix<double> smatrix;

using Eigen::VectorXd;
using Eigen::Vector3d;

bool small(double x) {
    return (x < 0.00003) && (x > -0.00003);

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
    VectorXd z = A*v_this;
    alpha_last = v_this.dot(z);
    z = z - alpha_last*v_this;

    beta_last = z.norm();
    p_k2 = v_this/alpha_last;

    v_last = v_this;
    v_this = z/beta_last;



    if (small(beta_last)) return;

    // compute second manually
    z = A*v_this - beta_last*v_last;
    auto [c, s] = givens(alpha_last, beta_last);

    x += c*zeta*p_k2;
    alpha_last = v_this.dot(z);
    z = z - alpha_last*v_this;

    p_k1 = v_this - p_k2*(c*beta_last - s*alpha_last);
    p_k1 /= (s*beta_last + c*alpha_last);
    alpha_last = s*beta_last + c*alpha_last;

    beta_last = z.norm();

    zeta *= s;

    last_col(0) = -s*beta_last;
    last_col(1) =  c*beta_last;



    if (small(beta_last)) return;
    std::cout << "\n-\n" << x << "\n-\n";

    for (int j = 2; j < 30; ++j) {

        z = A*v_this - beta_last*v_last;
        alpha_this = v_this.dot(z);
        z = z - alpha_this*v_this;
        beta_this = z.norm();

        auto [ci, si] = givens(alpha_last, beta_last);

        std::cout << x << "\n\n";

        last_col(2) = si * last_col(1) + ci * alpha_this; // f~
        last_col(1) = ci * last_col(1) - si * alpha_this;

        alpha_last = ci * beta_last + si * alpha_this;


        p_k = v_this - last_col(1)*p_k1 - last_col(0)*p_k2;
        p_k /= last_col(2);

        x += ci * zeta * p_k;
        zeta *= si;


        last_col(0) = -si * beta_this;
        last_col(1) = ci * beta_this;

        if (small(beta_this)) break;

        v_last = v_this;
        v_this = z/beta_this;

        p_k2 = p_k1;
        p_k1 = p_k;
        beta_last = beta_this;
    }
}


void d_lanczos(smatrix& A, VectorXd& x, VectorXd& b) {
    auto r = b - A*x; // don't have to save it. Auto doesn't compute it, solves the abstract expression
    VectorXd v_this = r / r.norm();
    VectorXd v_last = VectorXd::Zero(v_this.size());
    VectorXd p = VectorXd::Zero(v_this.size());
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