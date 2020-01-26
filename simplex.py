import numpy as np


class Simplex:
    """algorithm that finds the optimal solution for a linear program in CANONICAL form.

    It starts with a feasible basic solution, and moves from one feasible basic solution to
    another to decrease objective function, until reaching the minimum.
    """
    def __init__(self, arg_tab):
        """Initiates a Simplex object."""
        self.constr = arg_tab[:-1, :]
        self.objective = arg_tab[-1, :]
        self.tab = np.vstack((self.constr, self.objective))
        self.n_constr = self.constr.shape[0]
        self.n_var = self.constr.shape[1] - 1
        # code -1: solving, 0: solved, 1: unbounded, 2: infeasible
        self.code = -1

    def print_tab(self, width=8):
        """Prints current tab."""
        print(''.join(('x' + str(i + 1)).rjust(width) for i in range(self.tab.shape[1] - 1)))
        for i in range(self.tab.shape[0]):
            print(''.join('{0:0.3f}'.format(self.tab[i][j]).rjust(width) for j in range(self.tab.shape[1])))

    def indices(self):
        """Returns indices of basic vars in tab."""
        candidates = np.where(self.tab[:self.n_constr, :self.n_var] == 1)
        indices = []
        for n in range(len(candidates[0])):
            if len(np.where(self.tab[:, candidates[1][n]] == 0)[0]) == self.n_constr:
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
        if len(self.indices()) != self.n_constr:
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
        n_art = self.n_constr - len(indices)

        # artificial constr
        art_constr = np.copy(self.constr[:, :-1])
        art_i_indices = []
        aux = np.zeros((self.n_constr, n_art))
        art_j = 0
        for i in range(self.n_constr):
            if i not in i_indices:
                aux[i][art_j] = 1
                art_i_indices.append(i)
                art_j += 1
        art_constr = np.hstack((art_constr, aux, np.expand_dims(self.constr[:, -1], axis=-1)))

        # artificial objective
        art_objective = np.zeros((1, art_constr.shape[1]))
        for i in range(self.n_constr):
            if i in art_i_indices:
                art_objective -= art_constr[i, :]
        art_objective[0, self.n_var:-1] = 0

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
            self.tab = np.hstack((self.tab[:-1, :self.n_var], np.expand_dims(self.tab[:-1, -1], axis=-1)))
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
        enters = np.argmin(self.tab[-1, :self.n_var])
        ratios = []
        for i in range(self.n_constr):
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
            for k in range(self.n_var):
                if self.tab[-1, k] < 0:
                    if all(a <= 0 for a in self.tab[:-1, k]):
                        self.code = 1
                        return False
        return True

    def solve(self, verbose=False):
        """Solves linear program."""
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
            soln = np.zeros(self.n_var)
            for i in range(self.n_constr):
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
    # todo: add interface
    # todo: unbounded example

    # in canonical form
    tab1 = np.array([[1, 0, 0, 2, -1, 10],
                    [0, 1, 0, -1, -5, 20],
                    [0, 0, 1, 6, -12, 18],
                    [0, 0, 0, -2, 3, 60]], dtype=float)

    tab2 = np.array([[3, 2, 0, 1, 0, 0, 60],
                    [-1, 1, 4, 0, 1, 0, 10],
                    [2, -2, 5, 0, 0, 1, 50],
                    [-2, -3, -3, 0, 0, 0, 0]], dtype=float)

    # unbounded
    # tab3 = np.array([[],
    #                  [],
    #                  [],
    #                  []], dtype=float)

    # in standard form
    std1 = np.array([[1, -2, -3, -2, 3],
                    [1, -1, 2, 1, 11],
                    [2, -3, 1, 1, 0]], dtype=float)
    # infeasible
    std2 = np.array([[1, 2, 1, 1, 0, 1],
                    [-1, 0, 2, 0, -1, 4],
                    [1, -1, 2, 0, 0, 4],
                    [1, 1, 1, 0, 0, 0]], dtype=float)

    # s = Simplex(tab1)
    # s = Simplex(tab2)
    # s = Simplex(std1)
    s = Simplex(std2)
    s.solve(verbose=True)
