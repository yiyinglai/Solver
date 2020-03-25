import time
import numpy as np


class Simplex:
    """Primal simplex method that solves a linear program expressed in standard form.

    It starts with a feasible basic solution, and moves from one feasible basic solution to
    another to decrease objective function, until reaching the minimum.
    """

    def __init__(self, arg_tab, names=None):
        """Initiates a Simplex object."""
        self.iterations = 0
        self.solvetime = None
        # code -1: solving, 0: solved, 1: unbounded, 2: infeasible
        self.code = -1

        self.constrs = arg_tab[:-1, :]
        self.objective = arg_tab[-1, :]
        self.tab = np.vstack((self.constrs, self.objective))
        self.n_constrs = self.constrs.shape[0]
        self.n_vars = self.constrs.shape[1] - 1
        self.n_arts = 0
        if names is None:
            self.names = ['x' + str(i + 1) for i in range(self.n_vars)]
        else:
            self.names = names

    def print_tab(self, width=8):
        """Prints current tab."""
        print(''.join(name.rjust(width) for name in self.names))
        for i in range(self.tab.shape[0]):
            print(''.join('{0:0.3f}'.format(self.tab[i][j]).rjust(width) for j in range(self.tab.shape[1])))

    def indices(self):
        """Returns indices of basic vars in tab."""
        candidates = np.where(self.tab[:self.n_constrs, :-1] == 1)
        indices = []
        for n in range(len(candidates[0])):
            if len(np.where(self.tab[:self.n_constrs, candidates[1][n]] == 0)[0]) == self.n_constrs-1:
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
        if len(self.indices()) < self.n_constrs:
            if verbose:
                print('not in canonical form for violating condition 4: isolated var')
            return False

        # is canonical
        if verbose:
            print('in canonical form')
        return True

    def put_canonical(self, verbose=False):
        """Puts tab in canonical form."""
        # indices of basic variables
        indices = self.indices()
        i_indices = [idx[0] for idx in indices]

        # number of artificial vars needed
        self.n_arts = self.n_constrs - len(indices)
        self.names = self.names + ['a' + str(i+1) for i in range(self.n_arts)]

        # artificial constr
        art_constr = np.copy(self.constrs[:, :-1])
        art_i_indices = []
        aux = np.zeros((self.n_constrs, self.n_arts))
        art_j = 0
        for i in range(self.n_constrs):
            if i not in i_indices:
                aux[i][art_j] = 1
                art_i_indices.append(i)
                art_j += 1
        art_constr = np.hstack((art_constr, aux, np.expand_dims(self.constrs[:, -1], axis=-1)))

        # artificial objective
        art_objective = np.zeros((1, art_constr.shape[1]))
        for i in art_i_indices:
            art_objective -= art_constr[i, :]
        art_objective[0, self.n_vars:-1] = 0

        # new tab
        objective = np.hstack((self.objective[:-1], np.zeros(self.n_arts), self.objective[-1]))
        self.tab = np.vstack((art_constr, objective, art_objective))

        if verbose:
            print('\noriginal problem with artificial vars:')
            self.print_tab()

        # try to put in canonical form
        while self.should_continue():
            self.pivot(verbose=verbose)

        # remove redundancy
        indices = self.indices()
        for idx in indices:
            # if artificial variable in basic variables set
            if idx[1] >= self.n_vars:
                if verbose:
                    print('artificial variable a{} in basic variables set'.format(idx[1]-self.n_vars+1))
                leaves = idx[0]
                # remove redundant constraint
                if all(self.tab[leaves, :self.n_vars] == 0):
                    if verbose:
                        print('constraint {} is redundant'.format(idx[0]+1))
                else:
                    enters = np.where(self.tab[leaves, :self.n_vars] != 0)[0][0]
                    self.pivot(enters=enters, leaves=leaves, verbose=verbose)

        # original problem is feasible
        if abs(self.tab[-1][-1]) <= 0.00001:
            # remove artificial objective and artificial variables
            # reset self.n_vars to the correct number
            self.names = self.names[:-self.n_arts]
            self.tab = np.hstack((self.tab[:-1, :self.n_vars], np.expand_dims(self.tab[:-1, -1], axis=-1)))
            self.code = -1
            if verbose:
                print('\noriginal problem has feasible soln and is now in canonical form:')
                self.print_tab()
            return True
        # original problem is infeasible
        else:
            self.code = 2
            return False

    def pivot(self, enters=None, leaves=None, verbose=False):
        """Pivots the tab."""
        self.iterations += 1
        if enters is None:
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
        if all(c >= 0 for c in self.tab[-1, :self.n_vars]):
            if all(self.tab[:self.n_constrs, -1] >= 0):
                self.code = 0
            else:
                self.code = 2
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
        # start solving
        if verbose:
            print('original problem:')
            self.print_tab()

        start = time.time()

        # stage 1
        if not self.is_canonical(verbose=verbose):
            self.put_canonical(verbose=verbose)

        # stage 2
        if self.code == -1:
            while self.should_continue():
                self.pivot(verbose=verbose)

        self.solvetime = time.time() - start

        # report
        if verbose:
            if self.code == 0:
                soln = np.zeros(self.n_vars)
                for i in range(self.n_vars):
                    if len(np.where(self.tab[:-1, i] > 0)[0]) == 1:
                        arr = np.where(self.tab[:, i] == 1)[0]
                        if len(arr) == 1:
                            soln[i] = self.tab[:, -1][arr[0]]
                print('solution:')
                print('(' + ', '.join('{0:0.3f}'.format(x) for x in soln) + ')')
                print('objective function:')
                print('{0:0.3f}'.format(-self.tab[-1, -1]))
            elif self.code == 1:
                print('problem is unbounded')
            elif self.code == 2:
                print('original problem has no feasible soln')


if __name__ == '__main__':
    # in canonical form
    tab1 = np.array([[1, 0, 0, 2, -1, 10],
                     [0, 1, 0, -1, -5, 20],
                     [0, 0, 1, 6, -12, 18],
                     [0, 0, 0, -2, 3, 60]], dtype=float)

    tab2 = np.array([[3, 2, 0, 1, 0, 0, 60],
                     [-1, 1, 4, 0, 1, 0, 10],
                     [2, -2, 5, 0, 0, 1, 50],
                     [-2, -3, -3, 0, 0, 0, 0]], dtype=float)

    # in standard form
    std1 = np.array([[1, -2, -3, -2, 3],
                     [1, -1, 2, 1, 11],
                     [2, -3, 1, 1, 0]], dtype=float)

    # infeasible
    std2 = np.array([[1, 2, 1, 1, 0, 1],
                     [-1, 0, 2, 0, -1, 4],
                     [1, -1, 2, 0, 0, 4],
                     [1, 1, 1, 0, 0, 0]], dtype=float)

    # https://www.youtube.com/watch?v=CiLG14fsPdc&list=PLbxFfU5GKZz1Tm_9RR5M_uvdOXpJJ8LC3&index=9 at 36:58
    std3 = np.array([[1, -1, 0, 1],
                     [2, 1, -1, 3],
                     [0, 0, 0, 0]], dtype=float)

    # https://www.youtube.com/watch?v=CiLG14fsPdc&list=PLbxFfU5GKZz1Tm_9RR5M_uvdOXpJJ8LC3&index=9 at 47:57
    std4 = np.array([[1, -2, -3, -2, 3],
                     [1, -1, 2, 1, 11],
                     [2, -3, 1, 1, 0]], dtype=float)

    # https://www.youtube.com/watch?v=_uhTN6KvCC8 at 52:42
    # redundant
    red1 = np.array([[1, -2, 3, 1, 6],
                     [-1, 1, 2, 2/3, 4],
                     [2, -1, 1, -1, 0]], dtype=float)

    # https://www.youtube.com/watch?v=_uhTN6KvCC8 at 1:04:46
    # redundant
    red2 = np.array([[1, 2, 0, 1, 20],
                     [2, 1, 1, 0, 10],
                     [-1, 4, -2, 3, 40],
                     [1, 4, 3, 2, 0]], dtype=float)

    # s = Simplex(tab1)
    # s = Simplex(tab2)
    # s = Simplex(std1)
    # s = Simplex(std2)
    # s = Simplex(std3)
    # s = Simplex(std4)
    # s = Simplex(red1)
    s = Simplex(red2)
    s.solve(verbose=True)
