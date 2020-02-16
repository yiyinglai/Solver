import numpy as np
from toyplex.components import Var, LinExpr, LinConstr


class Simplex:
    """Primal simplex method that solves a linear program expressed in standard form.

    It starts with a feasible basic solution, and moves from one feasible basic solution to
    another to decrease objective function, until reaching the minimum.
    """
    def __init__(self):
        """Initiates a Simplex object."""
        self.constrs = []
        self.objective = None
        self.tab = None
        self.vars = {}
        self.slacks = {}
        self.n_constrs = 0
        self.n_vars = 0

        # code -1: solving, 0: solved, 1: unbounded, 2: infeasible
        self.code = -1

    def add_var(self, type='continuous', name=None):
        """Adds a decision variable."""
        if name is None:
            name = 'x' + str(int(len(self.vars))+1)
        self.vars[name] = Var(name, type=type)
        return self.vars[name]

    def add_constr(self, constr: LinConstr):
        """Adds a linear constraint."""
        self.constrs.append(constr)

    def set_tab(self):
        """Sets tab."""
        var_col = {}
        for idx, key in enumerate(self.vars.keys()):
            var_col[key] = idx
        self.n_constrs = len(self.constrs)
        self.n_vars = (len(self.vars) + len(self.slacks))
        self.tab = np.zeros((self.n_constrs, self.n_vars + 1))
        for idx, constr in enumerate(self.constrs):
            for key in constr.coeffs.keys():
                if constr.coeffs[key] != 0:
                    self.tab[idx][var_col[key]] = constr.coeffs[key]
                self.tab[idx][-1] = constr.b
        self.tab = np.vstack((self.tab, self.objective))

    def set_objective(self, objective: LinExpr):
        """Sets objective."""
        self.objective = np.zeros(len(self.vars) + len(self.slacks) + 1)

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

    def print_tab(self, width=8):
        """Prints current tab."""
        print(''.join(('x' + str(i + 1)).rjust(width) for i in range(self.tab.shape[1] - 1)))
        for i in range(self.tab.shape[0]):
            print(''.join('{0:0.3f}'.format(self.tab[i][j]).rjust(width) for j in range(self.tab.shape[1])))

    def indices(self):
        """Returns indices of basic vars in tab."""
        candidates = np.where(self.tab[:self.n_constrs, :self.n_vars] == 1)
        indices = []
        for n in range(len(candidates[0])):
            if len(np.where(self.tab[:, candidates[1][n]] == 0)[0]) == self.n_constrs:
                indices.append((candidates[0][n], candidates[1][n]))
        return indices

    def is_canonical(self, verbose=False):
        """Returns whether tab is in canonical form or not."""
        # condition 3: RHS non-negative
        if not all(self.tab[:, -1] >= 0):
            if verbose:
                print('not in canonical form for violating condition 3: RHS non-negative')
            return False

        # condition 4: isolated var
        if len(self.indices()) != self.n_constrs:
            if verbose:
                print('not in canonical form for violating condition 4: isolated var')
            return False

        # is canonical
        if verbose:
            print('in canonical form')
        return True

    def put_canonical(self, verbose=False):
        """Puts tab in canonical form."""
        indices = self.indices()
        i_indices = [idx[0] for idx in indices]
        # number of artificial vars
        n_art = self.n_constrs - len(indices)

        # artificial constr
        art_constr = np.copy(self.tab[:self.n_constrs, :-1])
        art_i_indices = []
        aux = np.zeros((self.n_constrs, n_art))
        art_j = 0
        for i in range(self.n_constrs):
            if i not in i_indices:
                aux[i][art_j] = 1
                art_i_indices.append(i)
                art_j += 1
        art_constr = np.hstack((art_constr, aux, np.expand_dims(self.tab[:self.n_constrs, -1], axis=-1)))

        # artificial objective
        art_objective = np.zeros((1, art_constr.shape[1]))
        for i in range(self.n_constrs):
            if i in art_i_indices:
                art_objective -= art_constr[i, :]
        art_objective[0, self.n_vars:-1] = 0

        # new tab
        objective = np.hstack((self.objective[:-1], np.zeros(n_art), self.objective[-1]))
        self.tab = np.vstack((art_constr, objective, art_objective))

        if verbose:
            print('\noriginal problem with artificial vars:')
            self.print_tab()

        # try to put in canonical form
        while self.should_continue():
            self.pivot(verbose=verbose)
        if self.tab[-1][-1] == 0:
            # necessary to remove art_objective because entering var is found using the last row in tab in self.pivot()
            self.tab = np.hstack((self.tab[:-1, :self.n_vars], np.expand_dims(self.tab[:-1, -1], axis=-1)))
            if verbose:
                print('\noriginal problem has feasible soln and is now in canonical form:')
                self.print_tab()
            return True
        else:
            self.code = 2
            return False

    def pivot(self, verbose=False):
        """Pivots the tab."""
        # find pivot point
        enters = np.argmin(self.tab[-1, :self.n_vars])
        ratios = []
        for i in range(self.n_constrs):
            if self.tab[i, enters] > 0:
                ratios.append(self.tab[i, -1] / self.tab[i, enters])
            else:
                ratios.append(float('inf'))
        leaves = np.argmin(ratios)

        # row of pivot point
        self.tab[leaves] = self.tab[leaves] / self.tab[leaves, enters]
        # the remaining rows
        for i in range(self.tab.shape[0]):
            if i != leaves:
                self.tab[i] = self.tab[i] - self.tab[i, enters] * self.tab[leaves]

        if verbose:
            print('pivoting at:', leaves + 1, enters + 1)
            self.print_tab()

    def should_continue(self):
        """Returns whether should continue pivoting or not."""
        # theorem O: reached optimality when all c_k > 0
        if all(c >= 0 for c in self.tab[-1, :-1]):
            self.code = 0
            return False

        # theorem U: unbounded if all a_ik <= 0, c_k < 0
        else:
            for k in range(self.n_vars):
                if self.tab[-1, k] < 0:
                    if all(a <= 0 for a in self.tab[:-1, k]):
                        self.code = 1
                        return False
        return True

    def solve(self, verbose=False):
        """Solves linear program."""
        # new
        self.set_tab()

        # start solving
        print('original problem:')
        self.print_tab()

        # stage 1
        if not self.is_canonical(verbose=verbose):
            self.put_canonical(verbose=verbose)

        # stage 2
        if self.code == -1:
            while self.should_continue():
                self.pivot(verbose=verbose)

        # report
        print('\nresult:')
        if self.code == 0:
            soln = np.zeros(self.n_vars)
            for i in range(self.n_constrs):
                soln[np.where(self.tab[i, :] == 1)] = self.tab[i, -1]
            print('solution:')
            print('(' + ', '.join('{0:0.3f}'.format(x) for x in soln) + ')')
            print('objective function:')
            print('{0:0.3f}'.format(-self.tab[-1, -1]))
        elif self.code == 1:
            print('problem is unbounded')
        elif self.code == 2:
            print('original problem has no feasible soln')


if __name__ == '__main__':
    # new implementation with interface
    m = Simplex()
    x1 = m.add_var()
    x2 = m.add_var()
    x3 = m.add_var()
    x4 = m.add_var()
    x5 = m.add_var()
    m.add_constr(x1 + 2 * x4 - x5 == 10)
    m.add_constr(x2 - x4 - 5 * x5 == 20)
    m.add_constr(x3 + 6 * x4 - 12 * x5 == 18)
    m.set_objective(-2 * x4 + 3 * x5 - 60)
    m.solve(verbose=True)

    # m = Simplex()
    # x1 = m.add_var()
    # x2 = m.add_var()
    # x3 = m.add_var()
    # x4 = m.add_var()
    # m.add_constr(x1 - 2 * x2 - 3 * x3 - 2 * x4 == 3)
    # m.add_constr(x1 - x2 + 2 * x3 + x4 == 11)
    # m.set_objective(2 * x1 - 3 * x2 + x3 + x4)
    # m.solve(verbose=True)
