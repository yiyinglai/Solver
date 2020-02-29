import numpy as np
from toyplex.components import Var, LinExpr, LinConstr
from toyplex.simplex import Simplex
import math
__version__ = '0'


class Model:
    """Primal simplex method that solves a linear program expressed in standard form.

    It starts with a feasible basic solution, and moves from one feasible basic solution to
    another to decrease objective function, until reaching the minimum.
    """
    def __init__(self):
        """Initiates a Simplex object."""
        self.constrs = []
        self.objective = None
        self.tab = None
        self.objval = 0
        # all variables
        self.vars = {}
        self.conts = {}
        self.bins = {}
        self.ints = {}
        self.slacks = {}
        self.surplus = {}
        self.n_constrs = 0
        self.n_vars = 0

    def add_var(self, type='cont', lb=0, ub=math.inf, name=None):
        """Adds a decision variable."""
        if type == 'cont':
            if name is None:
                name = 'x' + str(int(len(self.conts))+1)
            self.conts[name] = Var(name, type=type)
            self.vars.update(self.conts)
            if lb > 0:
                self.add_constr(self.conts[name] >= lb)
            if ub is not math.inf:
                self.add_constr(self.conts[name] <= ub)
        elif type == 'bin':
            if name is None:
                name = 'b' + str(int(len(self.bins))+1)
            self.bins[name] = Var(name, type=type)
            self.vars.update(self.bins)
            self.add_constr(self.bins[name] <= 1)
        elif type == 'int':
            if name is None:
                name = 'i' + str(int(len(self.ints))+1)
            self.ints[name] = Var(name, type=type)
            self.vars.update(self.ints)
        return self.vars[name]

    def add_constr(self, constr: LinConstr):
        """Adds a linear constraint, and add slack and surplus variables as needed."""
        # beautiful
        if constr.sense == '==':
            pass
        # slack variable
        elif constr.sense == '<=':
            name = 's' + str(int(len(self.slacks))+1)
            self.slacks[name] = Var(name, type='cont')
            constr.coeffs[name] = 1
        # surplus variable
        elif constr.sense == '>=':
            name = 'p' + str(int(len(self.surplus)) + 1)
            self.surplus[name] = Var(name, type='cont')
            constr.coeffs[name] = -1
        self.constrs.append(constr)

    def set_tab(self):
        """Sets tab."""
        var_col = {}
        for idx, key in enumerate([*self.vars.keys()]+[*self.slacks.keys()]+[*self.surplus.keys()]):
            var_col[key] = idx
        self.n_constrs = len(self.constrs)
        self.n_vars = len(self.vars) + len(self.slacks) + len(self.surplus)
        self.tab = np.zeros((self.n_constrs, self.n_vars + 1))
        for idx, constr in enumerate(self.constrs):
            for key in constr.coeffs.keys():
                if constr.coeffs[key] != 0:
                    self.tab[idx][var_col[key]] = constr.coeffs[key]
                self.tab[idx][-1] = constr.b
        self.tab = np.vstack((self.tab, self.objective))

    def set_objective(self, objective: LinExpr, sense='min'):
        """Sets objective."""
        self.objective = np.zeros(len(self.vars) + len(self.slacks) + len(self.surplus) + 1)

        if 'const' in objective.coeffs.keys():
            self.objective[-1] = -objective.coeffs['const']
            del objective.coeffs['const']
        else:
            self.objective[-1] = 0

        var_col = {}
        for idx, key in enumerate(self.vars.keys()):
            var_col[key] = idx
        for key in objective.coeffs.keys():
            if objective.coeffs[key] != 0:
                self.objective[var_col[key]] = objective.coeffs[key]

        if sense == 'max':
            self.objective = - self.objective

    def optimize(self, verbose=False):
        """Solves linear program."""
        print('Toyplex', __version__)
        self.set_tab()

        # simplex algorithm
        names = [name for name in self.vars] + [name for name in self.slacks] + [name for name in self.surplus]
        spx = Simplex(self.tab, names=names)
        spx.solve(verbose=verbose)

        # result
        self.objval = spx.tab[-1][-1]
        for idx, key in enumerate(self.vars.keys()):
            self.vars[key].val = 0
            if len(np.where(spx.tab[:-1, idx] > 0)[0]) == 1:
                arr = np.where(spx.tab[:, idx] == 1)[0]
                if len(arr) == 1:
                    self.vars[key].val = spx.tab[:, -1][arr[0]]


if __name__ == '__main__':
    m = Model()
    x = m.add_var(type='cont', name='x')
    y = m.add_var(type='cont', name='y')
    m.add_constr(3*x + 5*y <= 78)
    m.add_constr(4*x + y <= 36)
    m.set_objective(5*x + 4*y, sense='max')
    m.optimize(verbose=False)
    for var in m.vars.values():
        print(var.name + ':', var.val)
    print('Objective value:', m.objval)
