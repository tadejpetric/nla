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
    implicit_matrix(smatrix& mat) : X(&mat) {}

    Index rows() const { return X->rows(); }
    Index cols() const { return X->rows(); }


    VectorXd operator*(const VectorXd& x) const {
        return (1- alp) *  ((*X) * (X->transpose() * x)) + alp * x;

    }
    double alp;
    const smatrix *X;
};
namespace Eigen {
    namespace internal {
        template <>
        struct generic_product_impl<implicit_matrix, VectorXd, SparseShape, DenseShape, GemvProduct>
                : generic_product_impl_base<implicit_matrix,VectorXd,generic_product_impl<implicit_matrix,VectorXd> >
        {
            typedef typename Product<implicit_matrix,VectorXd>::Scalar Scalar;

            template<typename Dest>
            static void scaleAndAddTo(Dest& dst, const implicit_matrix& lhs, const VectorXd& rhs, const Scalar& alpha)
            {
                dst += alpha * lhs.alp*rhs;
                VectorXd y = lhs.X->transpose() * rhs;
                dst += alpha * (1-lhs.alp) *  (*(lhs.X) * y);
            }
        };
    }
}



int main() {
    // vpsina 2010
    const int V = 4*2*0 + 1*0; // my project is not solvable in matlab, because it is 1-indexed

    smatrix train;
    smatrix test;
    load_matrix(train, test);

    //smatrix test(5, 5);
    //test.setIdentity();
    Eigen::ConjugateGradient<implicit_matrix, Eigen::Lower|Eigen::Upper, Eigen::IdentityPreconditioner> cg;
    cg.setTolerance(0.000001);
    implicit_matrix A(test);
    A.alp = 1;

    VectorXd v = VectorXd::Random(20489);
    cg.compute(A);
    VectorXd x = cg.solve(v);
    std::cout << v;
    std::cout << "\n-\n" << x;

    Eigen::MatrixXd all(20489, 2);
    all.col(0) = x;
    all.col(1) = y;
    std::cout << all;



}