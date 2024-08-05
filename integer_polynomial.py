from __future__ import annotations

import copy
import sys

import numpy as np

from integer import Integer


class IntegerPolynomial:
    def __init__(self, coefficients):
        assert len(coefficients) > 0
        assert isinstance(coefficients[0], Integer)
        if isinstance(coefficients, list):
            self.coefficients: np.ndarray = np.array(coefficients, dtype=Integer)
        elif isinstance(coefficients, np.ndarray):
            self.coefficients: np.ndarray = copy.deepcopy(coefficients)
        else:
            assert False, 'Invalid argument type for creating IntegerPolynomial'

    def __add__(self, other: IntegerPolynomial) -> IntegerPolynomial:
        assert isinstance(other, IntegerPolynomial)
        self._resize_same(self.coefficients, other.coefficients)
        return IntegerPolynomial(np.trim_zeros(self.coefficients + other.coefficients, trim='b'))

    def __sub__(self, other: IntegerPolynomial) -> IntegerPolynomial:
        assert isinstance(other, IntegerPolynomial)
        self._resize_same(self.coefficients, other.coefficients)
        return IntegerPolynomial(np.trim_zeros(self.coefficients - other.coefficients, trim='b'))

    def __mul__(self, other: IntegerPolynomial) -> IntegerPolynomial:
        assert isinstance(other, IntegerPolynomial)
        return IntegerPolynomial(np.trim_zeros(np.convolve(self.coefficients, other.coefficients), trim='b'))

    def __eq__(self, other: IntegerPolynomial) -> bool:
        for i in range(len(self.coefficients)):
            if self.coefficients[i] != other.coefficients[i]:
                return False
        return True

    def __len__(self) -> int:
        return len(self.coefficients) - 1

    def __getitem__(self, key) -> Integer:
        return self.coefficients[key]

    def evaluate(self, x: Integer) -> Integer:
        res = Integer(0)
        for i in range(len(self.coefficients)):
            res += self.coefficients[i] * (x ** Integer(i))
        return res

    def get_payload(self):
        return sys.getsizeof(self.coefficients)

    @staticmethod
    def _resize_same(poly0, poly1):
        len0 = len(poly0)
        len1 = len(poly1)
        if len0 < len1:
            poly0 = np.resize(poly0, (len1,))
        elif len0 > len1:
            poly1 = np.resize(poly1, (len0,))
        return poly0, poly1

    @staticmethod
    def _gauss_elimination(A: np.ndarray, b: np.ndarray):
        n = A.shape[0]
        for i in range(n):
            pivot_reciprocal = Integer(1) / A[i, i]
            for j in range(i + 1, n):
                factor = A[j][i] * pivot_reciprocal
                A[j, i:] -= A[i, i:] * factor
                b[j] -= b[i] * factor
        x = np.array([Integer(0)] * n)
        for i in range(n - 1, -1, -1):
            x[i] = b[i] / A[i][i]
            b[:i] -= A[:i, i] * x[i]
        return x

    @staticmethod
    def interpolate(points: np.ndarray) -> IntegerPolynomial:
        assert len(points) >= 1
        A = np.array([[Integer(idx ** i) for i in range(len(points))] for idx in range(len(points))], dtype=object)
        b = copy.deepcopy(points)
        coefficients = IntegerPolynomial._gauss_elimination(A, b)
        return IntegerPolynomial(coefficients)
