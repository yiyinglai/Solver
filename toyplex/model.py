import numpy as np
from toyplex.components import Var, LinExpr, LinConstr
from toyplex.simplex import Simplex
import math
import copy


class Node:
    """Primal simplex method that solves a linear program expressed in standard form.

    It starts with a feasible basic solution, and moves from one feasible basic solution to
    another to decrease objective function, until reaching the minimum.
    """
    def __init__(self, key):
        """Initiates a Simplex object."""
        self.key = key
        # code -1: solving, 0: solved, 1: unbounded, 2: infeasible
        self.code = -1

        # default minimization
        self.sense = 'min'

        # decision vars, union of conts, bins and ints
        self.vars = {}
        self.conts = {}
        self.bins = {}
        self.ints = {}
        # non-decision vars
        self.slacks = {}
        self.surplus = {}
        # constraints
        self.constrs = []
        self.n_vars = 0
        self.n_constrs = 0
        # objective expression, objective array
        self.objval = None
        self.objexpr = None
        self.objective = None
        # tab
        self.tab = None
        self.spx = None

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

    def add_constr(self, _constr: LinConstr):
        """Adds a linear constraint, and add slack and surplus variables as needed."""
        constr = copy.deepcopy(_constr)
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

    def set_objective(self, objexpr: LinExpr, sense='min'):
        """Sets objective."""
        self.objexpr = objexpr
        self.objective = np.zeros(len(self.vars) + len(self.slacks) + len(self.surplus) + 1)

        if 'const' in objexpr.coeffs.keys():
            self.objective[-1] = -objexpr.coeffs['const']
            del objexpr.coeffs['const']
        else:
            self.objective[-1] = 0

        var_col = {}
        for idx, key in enumerate(self.vars.keys()):
            var_col[key] = idx
        for key in objexpr.coeffs.keys():
            if objexpr.coeffs[key] != 0:
                self.objective[var_col[key]] = objexpr.coeffs[key]

        if sense == 'max':
            self.sense = sense
            self.objective = -self.objective

        # set tab
        self.set_tab()

    def describe(self):
        """Describes the linear program."""
        print('{}\t{}'.format(self.sense, self.objexpr))
        for idx, constr in enumerate(self.constrs):
            if idx == 0:
                print('st\t{}'.format(constr))
            else:
                print('\t{}'.format(constr))

    def optimize(self, verbose=False):
        """Solves linear program."""
        # simplex algorithm
        names = [name for name in self.vars] + [name for name in self.slacks] + [name for name in self.surplus]
        self.spx = Simplex(self.tab, names=names)
        self.spx.solve(verbose=verbose)
        self.code = self.spx.code

        # result
        if self.code == 0:
            if self.sense == 'min':
                self.objval = -self.spx.tab[-1][-1]
            elif self.sense == 'max':
                self.objval = self.spx.tab[-1][-1]
            for idx, key in enumerate(self.vars.keys()):
                self.vars[key].val = 0
                if len(np.where(self.spx.tab[:-1, idx] > 0)[0]) == 1:
                    arr = np.where(self.spx.tab[:, idx] == 1)[0]
                    if len(arr) == 1:
                        self.vars[key].val = self.spx.tab[:, -1][arr[0]]

    def int_soln(self):
        """Returns if solution is integral and the number of fractional values."""
        integral = False
        num_fracvar = 0
        for var in [*self.ints.values()] + [*self.bins.values()]:
            if not float(var.val).is_integer():
                num_fracvar += 1
        if num_fracvar == 0:
            integral = True
        return integral, num_fracvar


class Model:
    """A mixed integer programming model.
    """
    def __init__(self):
        """Initiates a MIP model."""
        # code -1: solving, 0: solved, 1: unbounded, 2: infeasible
        self.code = -1

        # tree
        root = Node(0)
        self.nodes = {root.key: root}
        self.candidates = {root.key: root}
        self.icmbkey = None
        self.icmbval = math.inf
        self.strategy = 'best first'

        # default minimization
        self.sense = 'min'

        # decision vars, union of conts, bins and ints
        self.vars = {}
        self.conts = {}
        self.bins = {}
        self.ints = {}
        # non-decision vars
        self.slacks = {}
        self.surplus = {}
        # constraints
        self.constrs = []
        self.n_vars = 0
        self.n_constrs = 0
        # objective expression, objective array
        self.objkey = 0
        self.objval = math.inf
        self.objexpr = None
        self.objective = None

        # callback functions
        self.int_cb = None
        self.frac_cb = None
        self.parent_node = None
        self.child_node = None

        self.verbose = False

    def add_var(self, type='cont', lb=None, ub=None, name=None):
        """Adds a decision variable."""
        if lb is None and ub is None:
            self.nodes[0].add_var(type=type, name=name)
        elif lb is None and ub is not None:
            self.nodes[0].add_var(type=type, ub=ub, name=name)
        elif lb is not None and ub is None:
            self.nodes[0].add_var(type=type, lb=lb, name=name)

        # decision vars
        self.vars.update(self.nodes[0].vars)
        self.conts.update(self.nodes[0].conts)
        self.bins.update(self.nodes[0].bins)
        self.ints.update(self.nodes[0].ints)
        # non-decision vars
        self.slacks.update(self.nodes[0].slacks)
        self.surplus.update(self.nodes[0].surplus)
        # update model constrs
        self.constrs = self.nodes[0].constrs[:]
        return self.nodes[0].vars[name]

    def add_constr(self, constr: LinConstr):
        """Adds a linear constraint, and add slack and surplus variables as needed."""
        self.nodes[0].add_constr(constr)
        self.constrs = self.nodes[0].constrs[:]

    def add_lzcut(self, lzcut: LinConstr):
        """Adds a lazy cut, and add slack and surplus variables as needed."""
        self.child_node = self.add_node(self.parent_node, lzcut, add2candidates=True)
        self.relax(self.child_node)
        self.parent_node = None

    def add_node(self, parent_node: Node, constr: LinConstr, add2candidates=True):
        """Adds a new node to the tree, with additional constraint and returns it."""
        child_node = copy.deepcopy(parent_node)
        child_node.key = len(self.nodes)
        child_node.code = -1
        child_node.add_constr(constr)
        child_node.set_objective(self.objexpr, sense=self.sense)
        self.nodes[child_node.key] = child_node
        if add2candidates:
            self.candidates[child_node.key] = child_node
        return child_node

    def set_objective(self, objexpr: LinExpr, sense='min'):
        """Sets objective."""
        self.nodes[0].set_objective(objexpr, sense=sense)
        self.objexpr = objexpr
        if sense == 'max':
            self.sense = 'max'
            self.objval = -math.inf
            self.icmbval = -math.inf

    def describe(self):
        """Describes the linear program."""
        print('\n{}\t{}'.format(self.sense, self.objexpr))
        for idx, constr in enumerate(self.constrs):
            if idx == 0:
                print('st\t{}'.format(constr))
            else:
                print('\t{}'.format(constr))
        if self.conts:
            print('continuous: ' + ', '.join(str(var) for var in self.conts))
        if self.ints:
            print('integral: ' + ', '.join(str(var) for var in self.ints))
        if self.bins:
            print('binary: ' + ', '.join(str(var) for var in self.bins))
        print()

    def candidates_queue(self):
        """Returns a queue of keys of the candidate nodes."""
        if self.strategy == 'best first':
            if self.sense == 'min':
                return sorted(self.candidates, key=lambda key: self.candidates[key].objval if self.candidates[key].objval is not None else math.inf)
            else:
                return sorted(self.candidates, key=lambda key: self.candidates[key].objval if self.candidates[key].objval is not None else -math.inf, reverse=True)

    def update_icmb(self, node: Node):
        """Updates incumbent key and objective value as needed."""
        if (self.sense == 'max' and node.objval > self.icmbval) or (
                self.sense == 'min' and node.objval < self.icmbval):
            self.icmbkey = node.key
            self.icmbval = node.objval

    def relax(self, node):
        """Solves the relaxation of a node."""
        node.optimize(verbose=self.verbose)

        if self.verbose:
            print('\nNode {}'.format(node.key))
            node.describe()
            if node.code == 0:
                print(', '.join(str(node.vars[key].val) for key in self.vars.keys()))
                print('Objval: {}'.format(node.objval))
            else:
                print(node.code)
            print()
        # relaxation unbounded or infeasible
        if node.code != 0:
            del self.candidates[node.key]
            # unbounded
            if node.code == 1:
                self.code = 1
        # relaxation feasible
        else:
            # integral solution
            integral, num_fracvar = node.int_soln()
            if integral:
                del self.candidates[node.key]
                # no callback, simply update incumbent
                if self.int_cb is None:
                    self.update_icmb(node)
                # integral solution callback
                else:
                    self.parent_node = node
                    self.int_cb(self)
                    # no lazy constraints added
                    if self.parent_node is not None:
                        self.update_icmb(node)
                        self.parent_node = None

    def branch(self, key):
        """Branch at the node specified by the key."""
        node = self.nodes[key]
        del self.candidates[key]

        # todo: which_var()
        # var = which_var()
        var = [var for var in node.vars.values() if not float(var.val).is_integer()][0]

        if (self.sense == 'max' and node.objval > self.icmbval) or (
                self.sense == 'min' and node.objval < self.icmbval):
            # branch down
            left_node = self.add_node(node, var <= math.floor(var.val))
            self.relax(left_node)
            # branch up
            right_node = self.add_node(node, var >= math.ceil(var.val))
            self.relax(right_node)

    def optimize(self, int_cb=None, frac_cb=None, verbose=False):
        """Optimizes the mixed integer programming model."""
        if int_cb is not None:
            self.int_cb = int_cb
        if frac_cb is not None:
            self.frac_cb = frac_cb
        self.verbose = verbose

        print('Variables: {} continuous, {} binary, {} integer'.format(len(self.conts), len(self.bins), len(self.ints)))

        # root relaxation
        self.relax(self.nodes[0])

        while self.code == -1:
            # decide which node to branch next
            if self.candidates:
                key = self.candidates_queue()[0]
                self.branch(key)

            # no more candidates
            elif self.icmbkey:
                self.code = 0
            else:
                self.code = 2

        # finally
        if self.code == 0:
            self.objkey = self.icmbkey
            self.objval = self.icmbval
            for key in self.vars.keys():
                self.vars[key].val = self.nodes[self.objkey].vars[key].val
            print('\nOptimal objective value: {}'.format(self.objval))
        elif self.code == 1:
            print('Model unbounded')
        elif self.code == 2:
            print('Model infeasible')


if __name__ == '__main__':
    def my_integral_callback(model):
        x_val = model.parent_node.vars['x'].val
        y_val = model.parent_node.vars['y'].val
        if 4 * x_val + y_val > 36.5:
            print('Inside my integral callback: {}, {}'.format(x_val, y_val))
            print('Add lazy cut', str(4 * x + y <= 36.5))
            model.add_lzcut(4 * x + y <= 36.5)
        else:
            print('Inside my integral callback: {}, {}'.format(x_val, y_val))
            print('No lazy cuts added')

    m = Model()
    x = m.add_var(type='int', name='x')
    y = m.add_var(type='bin', name='y')
    m.add_constr(3*x + 5*y <= 78.8)
    # m.add_constr(4*x + y <= 36.5)
    m.set_objective(5*x + 4*y, sense='max')
    m.describe()
    m.strategy = 'best first'
    # m.optimize(verbose=False)
    m.optimize(int_cb=my_integral_callback, verbose=False)

    if m.code == 0:
        for var in m.vars.values():
            print("{}({}): {}".format(var.name, var.type, var.val))
