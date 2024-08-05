from __future__ import annotations

import math
import sys
import time
from typing import Union

from gmpy2 import mpz, random_state, mpz_random, next_prime, c_mod, invert, powmod, cmp


class Integer:
    __base: mpz = mpz(2)
    __random_state = random_state(int(time.time()))

    @staticmethod
    def set_base(base: Union[mpz, int]):
        assert type(base) is int or type(base) is mpz
        Integer.__base = -mpz(base)

    @staticmethod
    def get_base() -> mpz:
        return -Integer.__base

    @staticmethod
    def get_random(min: mpz = mpz(0)) -> Integer:
        return Integer(mpz_random(Integer.__random_state, Integer.__base - min) + min)

    @staticmethod
    def set_prime_base(min: mpz):
        Integer.__base = -next_prime(min)

    def __init__(self, n: Union[mpz, int]):
        super().__init__()
        assert type(n) is int or type(n) is mpz

        self.n: mpz = c_mod(n, Integer.__base)

    def __repr__(self):
        return str(self.n)

    def __add__(self, other: Integer) -> Integer:
        return Integer(c_mod(self.n + other.n, Integer.__base))

    def __sub__(self, other: Integer) -> Integer:
        return Integer(c_mod(self.n + (-other).n, Integer.__base))

    def __neg__(self) -> Integer:
        return Integer(-self.n)

    def __truediv__(self, other: Integer) -> Integer:
        return Integer(c_mod(self.n * invert(other.n, Integer.__base), Integer.__base))

    def __mul__(self, other: Integer) -> Integer:
        return Integer(c_mod(self.n * other.n, Integer.__base))

    def __pow__(self, other: Integer) -> Integer:
        return Integer(powmod(self.n, other.n, Integer.__base))

    def __eq__(self, other: Integer) -> bool:
        if isinstance(other, Integer):
            return cmp(self.n, other.n) == 0
        elif isinstance(other, float):
            return self == Integer(int(other))
        elif isinstance(other, int):
            return self == Integer(other)
        else:
            assert False

    def __gt__(self, other: Integer) -> bool:
        return cmp(self.n, other.n) > 0

    def __lt__(self, other: Integer) -> bool:
        return cmp(self.n, other.n) < 0

    def __ge__(self, other: Integer) -> bool:
        return cmp(self.n, other.n) >= 0

    def __le__(self, other: Integer) -> bool:
        return cmp(self.n, other.n) <= 0

    def __ne__(self, other) -> bool:
        if isinstance(other, Integer):
            return cmp(self.n, other.n) != 0
        elif isinstance(other, float):
            return self != Integer(int(other))
        elif isinstance(other, int):
            return self != Integer(other)
        else:
            assert False

    def get_size(self) -> int:
        return sys.getsizeof(self.n)
