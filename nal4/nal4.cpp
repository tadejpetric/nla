//
// Created by user on 4/27/23.
//

#include "nal4.h"
#include <iostream>
#include <limits>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <cmath>


typedef Eigen::SparseMatrix<double> smatrix;

using Eigen::VectorXd;

class implicit_matrix;

namespace Eigen {
    namespace internal {
        // adapted from https://eigen.tuxfamily.org/dox/group__MatrixfreeSolverExample.html
        template<>
        struct traits<implicit_matrix> :  public Eigen::internal::traits<Eigen::SparseMatrix<double> >
        {};
    }
}

class implicit_matrix : public Eigen::EigenBase<implicit_matrix> {
public:
    typedef double Scalar;
    typedef double RealScalar;
    typedef int StorageIndex;
    enum {
        ColsAtCompileTime = Eigen::Dynamic,
        MaxColsAtCompileTime = Eigen::Dynamic,
        IsRowMajor = false
    };
    implicit_matrix(smatrix& mat) : X(&mat), alp(1) {}

    Index rows() const { return X->rows(); }
    Index cols() const { return X->rows(); }


    VectorXd operator*(const VectorXd& x) const {
        return (1- alp) *  ((*X) * (X->transpose() * x)) + alp * x;

    }

    double alp;
    const smatrix *X;
};

double residue(implicit_matrix& A, VectorXd& x, VectorXd& b) {
    return (*A.X * (A.X->transpose() * x) - b).norm();
}


auto convergence_speed(implicit_matrix& A, VectorXd b, int steps=100)
-> std::tuple<std::vector<std::tuple<float, int>>, std::vector<std::tuple<float, float>>> {
    A.alp = 0;
    Eigen::ConjugateGradient<implicit_matrix, Eigen::Lower|Eigen::Upper, Eigen::IdentityPreconditioner> cg;
    //cg.setTolerance(1E-5);
    std::vector<std::tuple<float, int>> iters_list;
    std::vector<std::tuple<float, float>> acc_list;

    for (int i = 1; i < steps; ++i) {

        A.alp = i/(steps-1.);
        cg.compute(A);
        VectorXd novel = cg.solve(b);
        //std::cout << "iter " << i << " " << cg.iterations() << "\n";
        std::tuple<float, int> el = {i/(steps-1.), cg.iterations()};
        iters_list.emplace_back(el);

        std::tuple<float, float> el2 = {i/(steps-1.),residue(A, novel, b)};
        acc_list.emplace_back(el2);
    }
    return std::make_tuple(iters_list, acc_list);
}


void save_full_matrix(implicit_matrix&A, smatrix& test, const char* file){
    std::ofstream out(file);

    Eigen::ConjugateGradient<implicit_matrix, Eigen::Lower|Eigen::Upper, Eigen::IdentityPreconditioner> cg;
    cg.compute(A);
    cg.setTolerance(1E-5);
    for (int i = 0; i < 500; ++i) {
        //std::cout << "col " << i;
        VectorXd result = cg.solve(test.col(i));
        result = test.transpose() * result;
        for (double el : result) {
            out << el << " ";
        }
        out << "\n";
    }
    out.close();
}

int main(int argc, char** argv) {
    std::string filename;
    if (argc <= 1) {
        filename = ".";
    }
    else {
        filename = argv[1];
    }
    // vpsina 2010
    const int V = 4*2*0 + 1*0; // my project is not solvable in matlab, because it is 1-indexed

    smatrix train;
    smatrix test;
    load_matrix(train, test);

    //smatrix test(5, 5);
    //test.setIdentity();
    Eigen::ConjugateGradient<implicit_matrix, Eigen::Lower|Eigen::Upper, Eigen::IdentityPreconditioner> cg;
    cg.setTolerance(1E-10);
    implicit_matrix A(train);

    std::cout << "Writing charts for best alpha\n";
    auto timer = start();
    auto [chart_iter, chart_acc] = convergence_speed(A, test.col(V));
    stop(timer);
    save_chart(chart_iter, (filename+"iters.txt").c_str());
    save_chart(chart_acc, (filename+"alpha.txt").c_str());
    std::cout << "\n";

    // I determined that alpha=0.6 is a good compromise
    std::cout << "saving matrix\n";
    A.alp = 0.6;
    timer = start();
    save_full_matrix(A, test, (filename+"matrix.txt").c_str());
    stop(timer);
}