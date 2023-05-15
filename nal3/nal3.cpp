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

    template <typename preconditioner=Eigen::IdentityPreconditioner>
    void minres() {
        Eigen::MINRES<V, Eigen::Lower, preconditioner> s;
        s.setTolerance(1E-6);
        s.setMaxIterations(500);
        s.compute((*A).template triangularView<Eigen::Lower>());
        s.solve(*b).eval();
        error = s.error();
        steps = s.iterations();
    }

    template <typename preconditioner>
    void gmres(int restart) {
        Eigen::GMRES<V, preconditioner> s;
        s.set_restart(restart);
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

    template <typename preconditioner=Eigen::IdentityPreconditioner>
    void bicgstab() {
        Eigen::BiCGSTAB<V, preconditioner> s;
        s.setMaxIterations(500);
        s.setTolerance(1e-6);
        s.compute(*A);
        s.solve(*b).eval();
        error = s.error();
        steps = s.iterations();
    }

    template <typename preconditioner=Eigen::IdentityPreconditioner>
    void cg() {
        Eigen::ConjugateGradient<V, Eigen::Lower, preconditioner> cg;
        cg.setMaxIterations(500);
        cg.setTolerance(1e-6);
        cg.compute(*A);
        cg.solve(*b).eval();
        error = cg.error();
        steps = cg.iterations();
    }

    template <typename preconditioner=Eigen::IdentityPreconditioner>
    void idrs() {
        Eigen::IDRS<V, preconditioner> s;
        s.setMaxIterations(500);
        s.setTolerance(1e-6);
        s.compute(*A);
        s.solve(*b).eval();
        error = s.error();
        steps = s.iterations();
    }

    template <typename preconditioner=Eigen::IdentityPreconditioner>
    void idrstabl() {
        Eigen::IDRSTABL<V, preconditioner> s;
        s.setMaxIterations(500);
        s.setTolerance(1e-6);
        s.compute(*A);
        s.solve(*b).eval();

        error = s.error();
        steps = s.iterations();
    }

    void noniterative() {
        Eigen::SparseLU<V> solver;
        solver.compute(*A);
        auto x = solver.solve(*b).eval();
        error =  (*b - *A * x).norm() / (*b).norm();
    }
};


void c63(std::string& location) {
    // symmetric, not positive definite, real, not cholesky
    // methods: MINRES, BICGSTABL, BICGSTAB, GCR, GMRES
    smatrix mat;
    VectorXd b(44234);
    Eigen::loadMarket(mat, location+"c-63.mtx");
    readb(b, location+"c-63_b.mtx");
    solvers<smatrix, VectorXd> S(mat, b);
    std::cout << "\n-------------c-63-------------\n\n";
    S.minres();
    std::cout << "minres (err " << S.error << ", iter " << S.steps << ")\n";
    S.minres<Eigen::DiagonalPreconditioner<double>>();
    std::cout << "minres diag precond  (err " << S.error << ", iter " << S.steps << ")\n";

    S.idrs();
    std::cout << "idrs: (err " << S.error << ", iter " << S.steps << ")\n";
    S.idrs<Eigen::DiagonalPreconditioner<double>>();
    std::cout << "idrs diag precond: (err " << S.error << ", iter " << S.steps << ")\n";
    S.idrstabl();
    std::cout << "idrstabl: (err " << S.error << ", iter " << S.steps << ")\n";
    S.idrstabl<Eigen::DiagonalPreconditioner<double>>();
    std::cout << "idrstabl diag precond: (err " << S.error << ", iter " << S.steps << ")\n";
    S.idrstabl<Eigen::LeastSquareDiagonalPreconditioner<double>>();
    std::cout << "idrstabl LS precond: (err " << S.error << ", iter " << S.steps << ")\n";

    S.bicgstab();
    std::cout << "bicgstab (err " << S.error << ", iter " << S.steps << ")\n";
    S.bicgstab<Eigen::DiagonalPreconditioner<double>>();
    std::cout << "bicgstab diag precond (err " << S.error << ", iter " << S.steps << ")\n";

    S.bicgstabl();
    std::cout << "bicgstabl (err " << S.error << ", iter " << S.steps << ")\n";

    for (int i = 1; i < 6; ++i) {
        S.gmres<Eigen::IdentityPreconditioner>(4<<i);
        std::cout << "gmres " << (4<<i) << " restart : (err " << S.error << ", iter " << S.steps << ")\n";
    }

    for (int i = 1; i < 6; ++i) {
        S.gmres<Eigen::DiagonalPreconditioner<double>>(4<<i);
        std::cout << "gmres diag precond " << (4<<i) << " restart : (err " << S.error << ", iter " << S.steps << ")\n";
    }

    for (int i = 1; i < 6; ++i) {
        S.gmres<Eigen::LeastSquareDiagonalPreconditioner<double>>(4<<i);
        std::cout << "gmres LS precond " << (4<<i) << " restart : (err " << S.error << ", iter " << S.steps << ")\n";
    }

    S.noniterative();
    std::cout << "LU (err " << S.error << ")\n";
}

void gridgena(std::string location) {
    // symmetric, positive definite, real, cholesky decomposable
    // all methods
    smatrix mat;

    Eigen::loadMarket(mat, location+"gridgena.mtx");
    VectorXd b = mat.col(0);
    solvers<smatrix, VectorXd> S(mat, b);

    std::cout << "\n-------------gridgena-------------\n\n";

    S.minres();
    std::cout << "minres (err " << S.error << ", iter " << S.steps << ")\n";
    S.minres<Eigen::DiagonalPreconditioner<double>>();
    std::cout << "minres diag precond (err " << S.error << ", iter " << S.steps << ")\n";

    S.cg();
    std::cout << "cg (err " << S.error << ", iter " << S.steps << ")\n";
    S.cg<Eigen::IncompleteCholesky<double>>();
    std::cout << "cg chol precond (err " << S.error << ", iter " << S.steps << ")\n";
    S.cg<Eigen::IncompleteLUT<double>>();
    std::cout << "cg LUT precond (err " << S.error << ", iter " << S.steps << ")\n";

    S.idrs();
    std::cout << "idrs: (err " << S.error << ", iter " << S.steps << ")\n";
    S.idrs<Eigen::DiagonalPreconditioner<double>>();
    std::cout << "idrs diag precond: (err " << S.error << ", iter " << S.steps << ")\n";
    S.idrstabl();
    std::cout << "idrstabl: (err " << S.error << ", iter " << S.steps << ")\n";
    S.idrstabl<Eigen::DiagonalPreconditioner<double>>();
    std::cout << "idrstabl diag precond: (err " << S.error << ", iter " << S.steps << ")\n";
    S.idrstabl<Eigen::IncompleteLUT<double>>();
    std::cout << "idrstabl LS precond: (err " << S.error << ", iter " << S.steps << ")\n";
    S.idrstabl<Eigen::IncompleteCholesky<double>>();
    std::cout << "idrstabl chol precond: (err " << S.error << ", iter " << S.steps << ")\n";

    S.bicgstab();
    std::cout << "bicgstab (err " << S.error << ", iter " << S.steps << ")\n";
    S.bicgstab<Eigen::DiagonalPreconditioner<double>>();
    std::cout << "bicgstab diag precond (err " << S.error << ", iter " << S.steps << ")\n";
    S.bicgstab<Eigen::IncompleteCholesky<double>>();
    std::cout << "bicgstab chol precond (err " << S.error << ", iter " << S.steps << ")\n";

    S.bicgstabl();
    std::cout << "bicgstabl (err " << S.error << ", iter " << S.steps << ")\n";



    for (int i = 1; i < 6; ++i) {
        S.gmres<Eigen::IdentityPreconditioner>(4<<i);
        std::cout << "gmres " << (4<<i) << " restart : (err " << S.error << ", iter " << S.steps << ")\n";
    }

    for (int i = 1; i < 6; ++i) {
        S.gmres<Eigen::DiagonalPreconditioner<double>>(4<<i);
        std::cout << "gmres diag precond " << (4<<i) << " restart : (err " << S.error << ", iter " << S.steps << ")\n";
    }

    for (int i = 1; i < 6; ++i) {
        S.gmres<Eigen::LeastSquareDiagonalPreconditioner<double>>(4<<i);
        std::cout << "gmres LS precond " << (4<<i) << " restart : (err " << S.error << ", iter " << S.steps << ")\n";
    }

    for (int i = 1; i < 6; ++i) {
        S.gmres<Eigen::IncompleteCholesky<double>>(4<<i);
        std::cout << "gmres chol precond " << (4<<i) << " restart : (err " << S.error << ", iter " << S.steps << ")\n";
    }

    S.noniterative();
    std::cout << "LU (err " << S.error << ")\n";
}

void rfdevcol(cmatrix& A, cvec& b) {
    solvers<cmatrix, cvec> S(A, b);
    for (int i = 1; i < 5; ++i) {
        S.gmres<Eigen::IdentityPreconditioner>(4<<i);
        std::cout << "gmres " << (4<<i) << " restart : (err " << S.error << ", iter " << S.steps << ")\n";
    }
    for (int i = 1; i < 5; ++i) {
        S.gmres<Eigen::DiagonalPreconditioner<std::complex<double>>>(4<<i);
        std::cout << "gmres diag precond " << (4<<i) << " restart : (err " << S.error << ", iter " << S.steps << ")\n";
    }
    for (int i = 1; i < 5; ++i) {
        S.gmres<Eigen::LeastSquareDiagonalPreconditioner<std::complex<double>>>(4<<i);
        std::cout << "gmres LS precond " << (4<<i) << " restart : (err " << S.error << ", iter " << S.steps << ")\n";
    }
    S.bicgstabl();
    std::cout << "bicgstabl: (err " << S.error << ", iter " << S.steps << ")\n";

    S.bicgstab();
    std::cout << "bicgstab: (err " << S.error << ", iter " << S.steps << ")\n";
    S.bicgstab<Eigen::DiagonalPreconditioner<std::complex<double>>>();
    std::cout << "bicgstab diag precond: (err " << S.error << ", iter " << S.steps << ")\n";
    S.bicgstab<Eigen::IncompleteLUT<std::complex<double>>>();
    std::cout << "bicgstab diag precond: (err " << S.error << ", iter " << S.steps << ")\n";

    S.idrs();
    std::cout << "idrs: (err " << S.error << ", iter " << S.steps << ")\n";
    S.idrs<Eigen::DiagonalPreconditioner<std::complex<double>>>();
    std::cout << "idrs diag precond: (err " << S.error << ", iter " << S.steps << ")\n";
    S.idrstabl();
    std::cout << "idrstabl: (err " << S.error << ", iter " << S.steps << ")\n";
    S.idrstabl<Eigen::DiagonalPreconditioner<std::complex<double>>>();
    std::cout << "idrstabl diag precond: (err " << S.error << ", iter " << S.steps << ")\n";
    S.idrstabl<Eigen::LeastSquareDiagonalPreconditioner<std::complex<double>>>();
    std::cout << "idrstabl LS precond: (err " << S.error << ", iter " << S.steps << ")\n";

    S.noniterative();
    std::cout << "LU (err " << S.error << ")\n";

}
void rfdevice(std::string& location) {
    // not positive definite, not symmetric, complex, not cholesky decomposable
    // methods: GMRES, BICGSTAB, BICGSTABL
    //smatrix mat;
    cmatrix mat;
    cmatrix b;
    Eigen::loadMarket(mat, location+"RFdevice.mtx");
    Eigen::loadMarket(b, location+"RFdevice_b.mtx");
    std::cout << "\n-------------RFdevice-------------\n\n";
    for (int i = 0; i < 9; ++i){
        cvec bi = b.col(i);
        std::cout << "\tstolpec " << i << "\n";
        rfdevcol(mat, bi);
    }
}

int main(int argc, char** argv) {

    std::string location = "../nal3/data/";
    auto t = start();
    c63(location);
    stop(t);
    t= start();
    gridgena(location);
    stop(t);
    t = start();
    rfdevice(location);
    stop(t);
}
#include "nal3.h"
