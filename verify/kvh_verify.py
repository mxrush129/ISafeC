import re
from functools import reduce

import cvxpy as cp
import numpy as np
import sympy as sp

from rl_train.Examples import Example, Zones


class KVH:
    def __init__(self, example: Example, n, l):
        self.barrier = sp.sympify('0')
        self.ex = example
        self.n, self.l = n, l
        self.f = example.f
        x = sp.symbols([f'x{i + 1}' for i in range(self.n)])

        self.poly, self.poly_list = self.generate_monomials(self.n, self.l)
        self.sp_poly = np.array([sp.sympify(e) for e in self.poly])
        self.lam_sp = [sp.lambdify(x, e) for e in self.sp_poly]
        self.len_vector = len(self.poly)
        self.dic_forward = dict(zip([tuple(e) for e in self.poly_list], range(self.len_vector)))
        self.coefficient_matrix = None

        self.init_obj, self.unsafe_obj, self.lie_obj, self.lie_b = None, None, None, None
        self.lie, self.pos = None, 0
        self.initialize()
        print('Initialization completed!')
        position = -1
        for i, item in enumerate(self.poly_list):
            if sum(item) > 2:
                position = i
                break
        self.poly_fine = self.poly[:position]
        self.sp_poly_fine = [sp.sympify(e) for e in self.poly_fine]
        self.lam_sp_fine = [sp.lambdify(x, e) for e in self.sp_poly_fine]

    def verify_all(self):
        init_obj = list(-1 * np.array(self.init_obj))
        if not self.verify_positive(init_obj, s='initial'):
            print('Initial zone verification failed!')
            return False

        if not self.verify_positive(self.unsafe_obj, s='unsafe'):
            print('Unsafe zone verification failed!')
            return False

        if not self.verify_positive(None, True, s='domain'):
            print('Lie zone verification failed!')
            return False

        return True

    def verify_positive(self, expr_obj, lie=False, s='initial'):
        if lie:
            if self.pos > 0:
                obj = list(-1 * np.array(self.lie_obj))
                y = self.compute_lp(obj, self.lie_b)
            else:
                obj = list(-1 * np.array(self.lie))
                y = self.compute_linear_programming(obj)
        else:
            y = self.compute_linear_programming(expr_obj)
        print(f'The gamma value of {s} condition:', y)
        if y >= 0:
            return True
        else:
            return False

    def update_barrier(self, barrier, multiplier, controller):
        # update the barrier
        if isinstance(barrier, str):
            barrier = sp.sympify(barrier)
        self.barrier = barrier

        x = sp.symbols([f'x{i + 1}' for i in range(self.n)])

        # update the objective of zones
        self.init_obj = self.get_objective(self.barrier, self.get_interval(self.ex.I_zones))
        self.unsafe_obj = self.get_objective(self.barrier, self.get_interval(self.ex.U_zones))
        if self.pos > 0:
            db = sum([sp.diff(self.barrier, x[i]) * self.f[i](x, [controller]) for i in range(self.n)])
            self.lie_obj = self.get_objective(db, self.get_interval(self.ex.D_zones))
            self.lie_b = self.get_objective(self.barrier, self.get_interval(self.ex.D_zones))
        else:
            lie = sum([sp.diff(self.barrier, x[i]) * self.f[i](x, [controller]) for i in
                       range(self.n)]) - multiplier * self.barrier
            lie = sp.expand(lie)
            self.lie = self.get_objective(lie, self.get_interval(self.ex.D_zones))

    @staticmethod
    def get_interval(zone: Zones):
        if zone.shape == 'box':
            return zip(zone.low, zone.up)

    def get_objective(self, item, interval):
        origin_obj = [0] * self.len_vector
        dic = self.convert(item, interval)
        d = {e: i for i, e in enumerate(self.poly)}
        for key, value in dic.items():
            origin_obj[d[key]] += value
        return origin_obj

    def compute_lp(self, db_obj, b_obj):
        len_memory = self.coefficient_matrix.shape[1]
        x = cp.Variable((len_memory, 1))
        z = cp.Variable()
        y = cp.Variable()
        no_constant = self.coefficient_matrix[1:]
        constant = self.coefficient_matrix[0:1]

        b = np.array([db_obj[1:]]).T
        p = np.array([b_obj[1:]]).T
        A = np.diag(np.ones(len_memory))
        obj = cp.Maximize(y)
        zero = np.zeros((len_memory, 1))

        constraints = [A @ x >= zero, no_constant @ x == b + z * p, constant @ x == db_obj[0] + z * b_obj[0] - y]

        prob = cp.Problem(obj, constraints)
        prob.solve(solver=cp.GUROBI)
        # print(f'z:{z.value}')
        if prob.status == cp.OPTIMAL:
            s = self.coefficient_matrix @ x.value
            s = [[e[0]] if abs(e[0]) > 1e-6 else [0] for e in s]
            # print('sum:', sum(self.sp_poly @ s))
            return float(y.value)
        else:
            return None

    def compute_linear_programming(self, objective):
        len_memory = self.coefficient_matrix.shape[1]
        x = cp.Variable((len_memory, 1))
        y = cp.Variable()
        no_constant = self.coefficient_matrix[1:]
        constant = self.coefficient_matrix[0:1]

        b = np.array([objective[1:]]).T
        A = np.diag(np.ones(len_memory))

        obj = cp.Maximize(y)
        zero = np.zeros((len_memory, 1))

        constraints = [A @ x >= zero, no_constant @ x == b, constant @ x == objective[0] - y]

        prob = cp.Problem(obj, constraints)
        prob.solve(solver=cp.GUROBI)
        if prob.status == cp.OPTIMAL:
            s = self.coefficient_matrix @ x.value
            s = [[e[0]] if abs(e[0]) > 1e-6 else [0] for e in s]
            # print('sum:', sum(self.sp_poly @ s))
            return float(y.value)
        else:
            return None

    def initialize(self):
        _, tds = self.generate_monomials(self.n * 2, self.l)

        # ray.init(num_cpus=psutil.cpu_count() // 3)
        # base = ray.get([self.get_all_base.remote(self, item) for item in tds[1:]])

        base = []
        for item in tds[1:]:
            base.append(self.get_all_base(item))
        # print(base)

        self.coefficient_matrix = np.array(base).T

        # self.init_obj = [0] * self.len_vector
        # self.unsafe_obj = [0] * self.len_vector
        # self.lie_obj = [0] * self.len_vector
        # print(tds)
        # print(base)
        # for item in base:
        #     sum = 0
        #     for x, y in zip(self.poly, item):
        #         sum += sp.sympify(x) * y
        #     print(sum)

    def generate_monomials(self, variable_number, degree):
        def key_sort(item: list):
            ans = [-e for e in item]
            ans = [sum(item)] + ans
            return ans

        poly = [x.copy() for x in self.depth_first_search(variable_number, degree)]
        poly.sort(key=key_sort)
        x = sp.symbols([f'x{i + 1}' for i in range(variable_number)])
        polynomial = [str(reduce(lambda a, b: a * b, [x[i] ** exp for i, exp in enumerate(e)])) for e in poly]
        return polynomial, poly

    # @ray.remote
    def get_all_base(self, td):
        res = [1] + [0] * (self.len_vector - 1)
        for i, deg in enumerate(td):
            if deg > 0:
                for j in range(deg):
                    res = self.mul_vector(res, i)
        return res

    def mul_vector(self, a, pos):
        if pos % 2 != 0:  # mul 1-x
            pos //= 2
            ans = a.copy()
            for i, item in enumerate(a):
                if item != 0:
                    wl_i_love_you = self.poly_list[i].copy()
                    wl_i_love_you[pos] += 1
                    ans[self.dic_forward[tuple(wl_i_love_you)]] -= item
        else:  # mul x
            pos //= 2
            ans = [0] * len(a)
            for i, item in enumerate(a):
                if item != 0:
                    wl_i_love_you = self.poly_list[i].copy()
                    wl_i_love_you[pos] += 1
                    ans[self.dic_forward[tuple(wl_i_love_you)]] += item
        return ans

    # @staticmethod
    # @ray.remote
    # def generate_base(len_vector, pos):
    #     b1, b2 = [0] * len_vector, [0] * len_vector
    #     b1[pos + 1], b2[0], b2[pos + 1] = 1, 1, -1
    #     return [b1, b2]

    @staticmethod
    def depth_first_search(n: int, m: int):
        now = [0] * n
        while True:
            yield now
            now[0] += 1
            for i in range(n):
                if now[i] > m or sum(now) > m:
                    now[i] = 0
                    if i + 1 >= n:
                        return
                    now[i + 1] += 1
                else:
                    break

    @staticmethod
    def convert(expression, interval):
        if not isinstance(expression, str):
            expression = str(expression)

        # for i, (low, high) in enumerate(interval):
        #     expression = expression.replace(f'x{i + 1}', f'({high - low}*x{i + 1}+{low})')

        interval = list(interval)

        def replace_function(match):
            res = match.group()
            pos = int(res[1:]) - 1
            return f'({interval[pos][1] - interval[pos][0]}*x{pos + 1}+{interval[pos][0]})'

        expression = re.sub('x\d+', replace_function, expression)

        expression = sp.sympify(expression)
        expression = sp.expand(expression)

        terms = expression.as_ordered_terms()
        dic = {}
        for term in terms:
            item = term.as_coeff_Mul()
            dic[str(item[1])] = item[0]

        return dic
