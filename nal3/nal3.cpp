//
// Created by user on 4/27/23.
//
#include "nal3.h"
#include <iostream>
#include <limits>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <unsupported/Eigen/SparseExtra>
#include <unsupported/Eigen/IterativeSolvers>
#include <cmath>


typedef Eigen::SparseMatrix<double> smatrix;
typedef Eigen::SparseMatrix<std::complex<double>> cmatrix;

using Eigen::VectorXd;
typedef Eigen::Vector<std::complex<double>, Eigen::Dynamic> cvec;

template <typename V, typename T>
struct solvers {
    const V* A;
    const T* b;
    double error;
    long steps;
    solvers (V& A, T& b) : A(&A), b(&b) {}

    void minres() {
        Eigen::MINRES<V, Eigen::Lower> s;
        s.setTolerance(1E-6);
        s.setMaxIterations(500);
        s.compute((*A).template triangularView<Eigen::Lower>());
        s.solve(*b).eval();
        error = s.error();
        steps = s.iterations();
    }

    template <typename preconditioner>
    void gmres() {
        Eigen::GMRES<V, preconditioner> s;
        s.setMaxIterations(500);
        s.setTolerance(1e-6);
        s.compute(*A);
        s.solve(*b).eval();
        error = s.error();
        steps = s.iterations();
    }

    void bicgstabl() {
        Eigen::BiCGSTABL<V> s;
        s.setMaxIterations(500);
        s.setTolerance(1e-6);
        s.compute(*A);
        s.solve(*b).eval();
        error = s.error();
        steps = s.iterations();
    }

    void bicgstab() {
        Eigen::BiCGSTAB<V> s;
        s.setMaxIterations(500);
        s.setTolerance(1e-6);
        s.compute(*A);
        s.solve(*b).eval();
        error = s.error();
        steps = s.iterations();
    }

    template <typename preconditioner>
    void cg() {
        Eigen::ConjugateGradient<V, Eigen::Lower, preconditioner> cg;
        cg.setMaxIterations(500);
        cg.setTolerance(1e-6);
        cg.compute(*A);
        cg.solve(*b).eval();
        error = cg.error();
        steps = cg.iterations();
    }
};


void c63(std::string& location) {
    // symmetric, not positive definite, real, not cholesky
    // methods: MINRES, SYMMLQ, BICGSTABL, BICGSTAB, GCR, GMRES
    smatrix mat;
    VectorXd b(44234);
    Eigen::loadMarket(mat, location+"c-63.mtx");
    readb(b, location+"c-63_b.mtx");
    solvers<smatrix, VectorXd> S(mat, b);
    S.minres();
    std::cout << S.error << " " << S.steps;
}
void gridgena() {
    // symmetric, positive definite, real, cholesky decomposable
    // all methods
}

void rfdevcol(cmatrix& A, cvec& b) {
    solvers<cmatrix, cvec> S(A, b);
    S.gmres<Eigen::IdentityPreconditioner>();
    std::cout << "gmres: (err " << S.error << ", iter " << S.steps << ")\n";
    S.bicgstabl();
    std::cout << "bicgstabl: (err " << S.error << ", iter " << S.steps << ")\n";
    S.bicgstab();
    std::cout << "bicgstab: (err " << S.error << ", iter " << S.steps << ")\n";
}
void rfdevice(std::string& location) {
    // not positive definite, not symmetric, complex, not cholesky decomposable
    // methods: GMRES, GCR, BICGSTAB, BICGSTABL
    //smatrix mat;
    cmatrix mat;
    cmatrix b;
    Eigen::loadMarket(mat, location+"RFdevice.mtx");
    Eigen::loadMarket(b, location+"RFdevice_b.mtx");

    for (int i = 0; i < 9; ++i){
        cvec bi = b.col(i);
        std::cout << "stolpec " << i << "\n";
        rfdevcol(mat, bi);
    }
}

int main(int argc, char** argv) {
    smatrix A;
    smatrix b;

    std::string location = "../nal3/data/";
    //c63(location);
    rfdevice(location);

}
#include "nal3.h"
