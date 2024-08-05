import datetime
import math
import sys
import time

import matplotlib
import numpy as np
from matplotlib import pyplot as plt

from integer import Integer
from integer_polynomial import IntegerPolynomial

np.seterr(over='ignore')
matplotlib.use('TkAgg')


class InnerProduct3PCSimulator:
    def __init__(self, length: int, network_model: list):
        self.length = length
        self.network_model = network_model

    def delay_by_network(self, payload_size: int):
        return np.polynomial.polynomial.polyval(payload_size, self.network_model)

    @staticmethod
    def _calc_searching_rounds_recursive(length: int, lam: int, mem):
        if math.ceil(length / lam) <= 1:
            return 0

        res = 0
        for i in range(2, math.ceil(length / lam) + 1):
            if mem[length, i] == 0:
                res += 1
                mem[length, i] = 1
            res += InnerProduct3PCSimulator._calc_searching_rounds_recursive(math.ceil(length / lam), i, mem)
        return res

    def calc_searching_rounds(self):
        res = 0
        mem = np.zeros((self.length + 1, self.length + 1))
        for next_lambda in range(2, self.length):
            if mem[self.length, next_lambda] == 0:
                res += 1
                mem[self.length, next_lambda] = 1
            res += self._calc_searching_rounds_recursive(self.length, next_lambda, mem)
        return res

    def simulate_one_round(self, length: int, compression: int):
        s = math.ceil(length / compression)
        u = np.array([Integer.get_random() for _ in range(length)], dtype=object)
        v = np.array([Integer.get_random() for _ in range(length)], dtype=object)
        out_0 = np.sum(u * v)
        out_1 = Integer(0)

        start_time = time.time_ns()

        if s > 1:
            u.resize(s * compression)
            v.resize(s * compression)

            u[u == 0] = Integer(0)
            v[v == 0] = Integer(0)

            gx = IntegerPolynomial([Integer(0)])
            for i in range(s):
                p = IntegerPolynomial.interpolate(u[compression * i:compression * (i + 1)])
                q = IntegerPolynomial.interpolate(v[compression * i:compression * (i + 1)])
                gx += p * q

            c0_coeff = []
            c1_coeff = []
            for i in range(len(gx) + 1):
                random_int = Integer.get_random()
                c0_coeff.append(random_int)
                c1_coeff.append(gx[i] - random_int)

            c0 = IntegerPolynomial(c0_coeff)
            c1 = IntegerPolynomial(c1_coeff)

            b1 = Integer(0)
            for i in range(compression):
                b1 += c0.evaluate(Integer(i))
            b1 -= out_0
            r = Integer.get_random()
            out_0 = c0.evaluate(r)

            end_time = time.time_ns()
            end_time += int(self.delay_by_network(c0.get_payload() + sys.getsizeof(compression)) * 1e+6 +
                            self.delay_by_network(sys.getsizeof(r) + sys.getsizeof(b1)) * 1e+6)

            b2 = Integer(0)
            for i in range(compression):
                b2 += c1.evaluate(Integer(i))
            b2 -= out_1
            out_1 = c1.evaluate(r)

            assert b1 + b2 == Integer(0)
        else:
            p = IntegerPolynomial.interpolate(np.concatenate([np.array([Integer.get_random()]), u]))
            q = IntegerPolynomial.interpolate(np.concatenate([np.array([Integer.get_random()]), v]))
            gx = p * q

            c0_coeff = []
            c1_coeff = []
            for i in range(len(gx) + 1):
                random_int = Integer.get_random()
                c0_coeff.append(random_int)
                c1_coeff.append(gx[i] - random_int)

            c0 = IntegerPolynomial(c0_coeff)
            c1 = IntegerPolynomial(c1_coeff)

            r = Integer.get_random()
            out_0 = c0.evaluate(r)
            u_res = p.evaluate(r)

            end_time = time.time_ns()
            end_time += int(self.delay_by_network(c0.get_payload() + sys.getsizeof(compression)) * 1e+6 +
                            self.delay_by_network(sys.getsizeof(out_0) + sys.getsizeof(u_res)) * 1e+6)

            out_1 = c1.evaluate(r)
            v_res = q.evaluate(r)

            assert u_res * v_res == out_0 + out_1

        return (end_time - start_time) / 1e+6

    def _find_best_lambda_schedule(self, length: int, mem):
        if length <= 1:
            return 0., [], 0., []
        total_min = float('inf')
        min_schedule = []
        total_max = float('-inf')
        max_schedule = []
        for next_lambda in range(2, length + 1):
            if mem[length][next_lambda] == 0.:
                mem[length, next_lambda] = self.simulate_one_round(length, next_lambda)
            min_time, optim_schedule, max_time, worst_schedule = self._find_best_lambda_schedule(math.ceil(length / next_lambda), mem)
            curr_min = mem[length][next_lambda] + min_time
            curr_max = mem[length][next_lambda] + max_time
            if total_min > curr_min:
                total_min = curr_min
                min_schedule = [next_lambda] + optim_schedule
            if total_max < curr_max:
                total_max = curr_max
                max_schedule = [next_lambda] + worst_schedule
        return total_min, min_schedule, total_max, max_schedule

    def find_best_lambda_schedule(self):
        mem = np.zeros((self.length + 1, self.length + 1), dtype=np.float64)
        return self._find_best_lambda_schedule(self.length, mem)


if __name__ == '__main__':
    Integer.set_prime_base(2 ** 31)
    mins = []
    for i in range(2, 65):
        sim = InnerProduct3PCSimulator(i, [10., 0.025])
        min_time, min_path, max_time, max_path = sim.find_best_lambda_schedule()
        print(f'LEN {i} MIN_TIME {min_time} MIN_SCHEDULE {min_path} MAX_TIME {max_time} MAX_SCHEDULE {max_path}')
        mins.append(min_time)

    fig, ax = plt.subplots()

    ax.plot(range(2, 65), mins, color='blue', marker='^', label='min time')

    plt.xlabel('length of vector')
    plt.ylabel('total time (ms)')
    plt.legend()
    plt.tight_layout()
    fig.savefig(f'fig_{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}.png', dpi=300, format='png',
                bbox_inches='tight', pad_inches=0.1, )
    plt.show()
