 # -*- coding: utf-8 -*-
"""
Maximum Entropy calculation, brute-force w/o assuming exponential models.

Entropy is a concave function in the pmf. Negative entropy is convex.  Convex
minimization then allows us to find the distribution with minimal negative
entropy. This means we can find the maximum entropy distribution, which, under
no other constraints, is the uniform distribution. The goal of this module is
to try to minimize the negentropy via two routes.

Here, we focus on n=1 individual but let the range of values be user-specified.
So rather than focus on n=1 binary, we can do n=1 drawing from k-symbols.
In order to match CVXOPT notation, in the following, we refer to the alphabet
size as n.

Route 1:
    Treat all p_i as independent and use a normalization constraint.

Route 2:
    Treat all but the final p_i as independent and do not use a normalization
    constraint.

We'll also try this with automatic differentiators.

The true gradient and Hessian in the case that all components are independent.

    \frac{\partial H}{\partial p_i}
        = - \log_2 p_j - 1 / \log(2)
    \frac{\partial^2 H}{\partial p_j \partial p_j}
        = - \frac{ \delta_{ij} }{ p_i \log(2)

The true gradient and Hessian in the case that only the first N-1 components
are independent.

    \frac{\partial H}{\partial p_i}
        = \log_2 (p_N / p_i)
    \frac{\partial^2 H}{\partial p_j \partial p_j}
        = \frac{1}{p_N \log(2)} - \frac{ \delta_{ij} }{ p_i \log(2)

Using MaxEntropyImplicit() doesn't really work. It turns out it is too
difficult to obtain a probability distribution that is properly normalized if
we do not enforce it explicitly.

Lesson: Always enforce constraints explicitly. Use MaxEntropyExplicit().

"""
from __future__ import division

import numpy as np
# np.log2(0) doesn't need a warning...
np.seterr(divide='ignore', invalid='ignore')

import numdifftools

from cvxopt import matrix
from cvxopt.solvers import cp, options

def negentropy_1(p):
    """
    Entropy which operates on vectors of length N.

    """
    return np.nansum(p * np.log2(p))

def negentropy_2(p):
    """
    Entropy which operates on vectors of length N-1.

    """
    pN = 1 - p.sum()
    d = np.array(p.tolist() + [pN])
    return np.nansum(d * np.log2(d))

grad_1 = numdifftools.Gradient(negentropy_1)
hess_1 = numdifftools.Hessian(negentropy_1)

grad_2 = numdifftools.Gradient(negentropy_2)
hess_2 = numdifftools.Hessian(negentropy_2)

class MaxEntropyExplicit(object):
    """
    Find maximum entropy distribution with explicit normalization constraints.

    """
    negentropy = negentropy_1
    gradient = grad_1
    hessian = hess_1

    def __init__(self, n, prng=None):
        self.n = n
        if prng is None:
            prng = np.random.RandomState()
        self.prng = prng

        self.negentropy = negentropy_1
        self.gradient = grad_1
        self.hessian = hess_1

        self.tol = {}

        self.init()

    def init(self):

        # Dimension of optimization variable
        n = self.n

        # Number of nonlinear constraints
        self.m = 0

        self.build_nonnegativity_constraints()
        self.build_linear_equality_constraints()
        self.build_F()

    def build_nonnegativity_constraints(self):

        n = self.n

        # Nonnegativity constraint
        #
        # We have M = N = 0 (no 2nd order cones or positive semidefinite cones)
        # So, K = l where l is the dimension of the nonnegative orthant. Thus,
        # we have l = n.
        G = matrix( -1 * np.eye(n) )   # G should have shape: (K,n) = (n,n)
        h = matrix( np.zeros((n,1)) )  # h should have shape: (K,1) = (n,1)

        self.G = G
        self.h = h

    def build_linear_equality_constraints(self):

        n = self.n

        #
        # Linear equality constraints (these are not independent constraints)
        #
        A = []
        b = []

        # 1) Normalization: \sum_i q_i = 1
        A.append( np.ones(n) )
        b.append( 1 )

        A = np.asarray(A, dtype=float)
        b = np.asarray(b, dtype=float)

        A = matrix(A)
        b = matrix(b)  # now a column vector

        self.A = A
        self.b = b

    def build_F(self):

        n = self.n
        m = self.m

        def F(x=None, z=None):
            # x has shape: (n,1)   and is the distribution
            # z has shape: (m+1,1) and is the Hessian of f_0

            if x is None and z is None:
                # Initial point is the original distribution.
                d = self.prng.dirichlet([1]*n)
                return (m, matrix(d))

            xarr = np.array(x)[:,0]

            # Verify that x is in domain.
            # Does G,h and A,b take care of this?
            #
            if np.any(xarr > 1) or np.any(xarr < 0):
                return None
            if not np.allclose(np.sum(xarr), 1, **self.tol):
                print "Discarding due to bad normalization."
                return None

            f = self.negentropy(xarr)
            Df = self.gradient(xarr)
            Df = matrix(Df.reshape((1, n)))

            if z is None:
                return (f, Df)
            H = self.hessian(xarr)
            H = matrix(H)

            return (f, Df, z[0] * H)

        self.F = F

    def optimize(self, show_progress=False):

        old = options.get('show_progress', None)
        out = None

        try:
            options['show_progress'] = show_progress
            result = cp(F=self.F,
                        G=self.G,
                        h=self.h,
                        dims={'l':self.n, 'q':[], 's':[]},
                        A=self.A,
                        b=self.b)

            self.result = result
            out = np.asarray(result['x'])
        except:
            if old is None:
                del options['show_progress']
            else:
                options['show_progress'] = old

        return out

class MaxEntropyImplicit(MaxEntropyExplicit):
    """
    Find maximum entropy distribution with implicit normalization constraints.

    """
    negentropy = negentropy_2
    gradient = grad_2
    hessian = hess_2

    def __init__(self, n, prng=None):
        super(MaxEntropyImplicit, self).__init__(n, prng)
        self.tol = {'rtol': 1e-2, 'atol': 1e-3}

    def build_linear_equality_constraints(self):
        self.A = None
        self.b = None
