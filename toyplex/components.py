from copy import deepcopy
import unittest


class Var:
    """Variables in a solver.
    """
    def __init__(self, name, type='cont'):
        self.type = type
        self.name = name
        self.val = None

    def __neg__(self):
        return LinExpr({self.name: -1})

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return LinExpr({self.name: other})

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return LinExpr({self.name: 1/other})

    def __add__(self, other):
        # print('Var__add__')
        if isinstance(other, (int, float)):
            return LinExpr({self.name: 1, 'const': other})
        elif isinstance(other, Var):
            if other.name == self.name:
                return LinExpr({self.name: 2})
            else:
                return LinExpr({self.name: 1, other.name: 1})
        elif isinstance(other, LinExpr):
            return other.__radd__(self)

    def __eq__(self, other):
        if isinstance(other, (int, float)):
            return LinConstr(1 * self, '==', LinExpr({'const': other}))
        elif isinstance(other, Var):
            if other.name == self.name:
                print('This is valid, but not necessary')
                return LinConstr(1 * self, '==', 1 * other)
            else:
                return LinConstr(1 * self, '==', 1 * other)

    def __le__(self, other):
        if isinstance(other, (int, float)):
            return LinConstr(1 * self, '<=', LinExpr({'const': other}))
        elif isinstance(other, Var):
            return LinConstr(1 * self, '<=', 1 * other)
        elif isinstance(other, LinExpr):
            return LinConstr(1 * self, '<=', other)

    def __ge__(self, other):
        if isinstance(other, (int, float)):
            return LinConstr(1 * self, '>=', LinExpr({'const': other}))
        elif isinstance(other, Var):
            return LinConstr(1 * self, '>=', 1 * other)
        elif isinstance(other, LinExpr):
            return LinConstr(1 * self, '>=', other)

    def __sub__(self, other):
        return self.__add__(-other)

    def __rsub__(self, other):
        return -self.__sub__(other)

    __rmul__ = __mul__
    __radd__ = __add__


class LinExpr:
    """Linear expression.
    """
    def __init__(self, coeffs):
        self.coeffs = coeffs

    def __neg__(self):
        for key in self.coeffs.keys():
            self.coeffs[key] = -self.coeffs[key]
        return self

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            selfkeys = self.coeffs.keys()
            for key in selfkeys:
                self.coeffs[key] *= other
            return self

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            selfkeys = self.coeffs.keys()
            for key in selfkeys:
                self.coeffs[key] /= other
            return self

    def __add__(self, other):
        if isinstance(other, (int, float)):
            if 'const' in self.coeffs.keys():
                self.coeffs['const'] += other
            else:
                self.coeffs['const'] = other
            return self
        elif isinstance(other, Var):
            if other.name in self.coeffs.keys():
                self.coeffs[other.name] += 1
            else:
                self.coeffs[other.name] = 1
            return self
        elif isinstance(other, LinExpr):
            selfkeys = self.coeffs.keys()
            otherkeys = other.coeffs.keys()
            for key in otherkeys:
                if key in selfkeys:
                    self.coeffs[key] += other.coeffs[key]
                else:
                    self.coeffs[key] = other.coeffs[key]
            return self

    def __sub__(self, other):
        if isinstance(other, (int, float)):
            if 'const' in self.coeffs.keys():
                self.coeffs['const'] -= other
            else:
                self.coeffs['const'] = -other
            return self
        elif isinstance(other, Var):
            if other.name in self.coeffs.keys():
                self.coeffs[other.name] -= 1
            else:
                self.coeffs[other.name] = -1
            return self
        elif isinstance(other, LinExpr):
            selfkeys = self.coeffs.keys()
            otherkeys = other.coeffs.keys()
            for key in otherkeys:
                if key in selfkeys:
                    self.coeffs[key] -= other.coeffs[key]
                else:
                    self.coeffs[key] = -other.coeffs[key]
            return self

    def __rsub__(self, other):
        return -self.__sub__(other)

    __rmul__ = __mul__
    __radd__ = __add__

    def __str__(self):
        _str = ''
        for idx, key in enumerate(self.coeffs.keys()):
            if key == 'const':
                if self.coeffs[key] == 0:
                    if len(self.coeffs.keys()) == 1:
                        _str += '0'
                elif self.coeffs[key] < 0:
                    _str += str(self.coeffs[key])
                elif self.coeffs[key] > 0:
                    if idx == 0:
                        _str += str(self.coeffs[key])
                    else:
                        _str += '+' + str(self.coeffs[key])
            else:
                if self.coeffs[key] == 1:
                    if idx == 0:
                        _str += key
                    else:
                        _str += '+' + key
                elif self.coeffs[key] == -1:
                    _str += '-' + key
                elif self.coeffs[key] < 0:
                    _str += str(self.coeffs[key]) + '*' + key
                else:
                    if idx == 0:
                        _str += str(self.coeffs[key]) + '*' + key
                    else:
                        _str += '+' + str(self.coeffs[key]) + '*' + key
        _str = _str.replace('+', ' + ')
        _str = _str.replace('-', ' - ')
        _str = _str.replace('*', ' * ')
        _str = _str.replace('/', ' / ')
        _str = _str.replace('  ', ' ')
        return _str

    def __eq__(self, other):
        if isinstance(other, (int, float)):
            return LinConstr(self, '==', LinExpr({'const': other}))
        elif isinstance(other, Var):
            return LinConstr(self, '==', 1 * other)
        elif isinstance(other, LinExpr):
            return LinConstr(self, '==', rhs=other)

    def __le__(self, other):
        if isinstance(other, (int, float)):
            return LinConstr(self, '<=', LinExpr({'const': other}))
        elif isinstance(other, Var):
            return LinConstr(self, '<=', 1 * other)
        elif isinstance(other, LinExpr):
            return LinConstr(self, '<=', other)

    def __ge__(self, other):
        if isinstance(other, (int, float)):
            return LinConstr(self, '>=', LinExpr({'const': other}))
        elif isinstance(other, Var):
            return LinConstr(self, '>=', 1 * other)
        elif isinstance(other, LinExpr):
            return LinConstr(self, '>=', other)


class LinConstr:
    """Linear constraint.
    """
    def __init__(self, lhs: LinExpr, sense, rhs: LinExpr):
        self.lhs = lhs
        self.sense = sense
        self.rhs = rhs
        if 'const' not in lhs.coeffs.keys():
            lhs.coeffs['const'] = 0
        if 'const' not in rhs.coeffs.keys():
            rhs.coeffs['const'] = 0
        lhskeys = lhs.coeffs.keys()
        rhskeys = rhs.coeffs.keys()
        coeffs = deepcopy(lhs.coeffs)
        for key in rhskeys:
            if key in lhskeys:
                coeffs[key] -= rhs.coeffs[key]
            else:
                coeffs[key] = -rhs.coeffs[key]
        # final right hand side
        self.b = -coeffs['const']
        del coeffs['const']
        # final left hand side
        self.coeffs = coeffs

    def __str__(self):
        _str = str(self.lhs) + self.sense + str(self.rhs)
        _str = _str.replace('+', ' + ')
        _str = _str.replace('-', ' - ')
        _str = _str.replace('*', ' * ')
        _str = _str.replace('/', ' / ')
        _str = _str.replace('==', ' == ')
        _str = _str.replace('<=', ' <= ')
        _str = _str.replace('>=', ' >= ')
        _str = _str.replace('  ', ' ')
        return _str


class Test(unittest.TestCase):
    def test_Var__neg__(self):
        x = Var('x', type='cont')
        self.assertEqual((-x).coeffs, {'x': -1})

    def test_Var__mul__(self):
        x = Var('x', type='cont')
        self.assertEqual((x * 1).coeffs, {'x': 1})

    def test_Var__rmul__(self):
        x = Var('x', type='cont')
        self.assertEqual((1 * x).coeffs, {'x': 1})

    def test_Var__truediv__(self):
        x = Var('x', type='cont')
        self.assertEqual((x/2).coeffs, {'x': 1/2})

    def test_Var__add__(self):
        x = Var('x', type='cont')
        y = Var('y', type='cont')
        self.assertEqual((x + 1).coeffs, {'x': 1, 'const': 1})
        self.assertEqual((x + x).coeffs, {'x': 2})
        self.assertEqual((x + y).coeffs, {'x': 1, 'y': 1})
        self.assertEqual((x + (x + 1)).coeffs, {'x': 2, 'const': 1})

    def test_Var__radd__(self):
        x = Var('x', type='cont')
        y = Var('y', type='cont')
        self.assertEqual((1 + x).coeffs, {'x': 1, 'const': 1})
        self.assertEqual((x + x).coeffs, {'x': 2})
        self.assertEqual((x + y).coeffs, {'x': 1, 'y': 1})
        self.assertEqual(((x + 1) + x).coeffs, {'x': 2, 'const': 1})

    def test_Var__sub__(self):
        x = Var('x', type='cont')
        y = Var('y', type='cont')
        self.assertEqual((x - 1).coeffs, {'x': 1, 'const': -1})
        self.assertEqual((x - x).coeffs, {'x': 0})
        self.assertEqual((x - y).coeffs, {'x': 1, 'y': -1})
        self.assertEqual((x - (x + 1)).coeffs, {'x': 0, 'const': -1})

    def test_Var__rsub__(self):
        x = Var('x', type='cont')
        y = Var('y', type='cont')
        self.assertEqual((1 - x).coeffs, {'x': -1, 'const': 1})
        self.assertEqual((x - x).coeffs, {'x': 0})
        self.assertEqual((x - y).coeffs, {'x': 1, 'y': -1})
        self.assertEqual(((x + 1) - x).coeffs, {'x': 0, 'const': 1})

    def test_LinExpr__neg__(self):
        x = Var('x', type='cont')
        y = Var('y', type='cont')
        self.assertEqual((-(x + 1)).coeffs, {'x': -1, 'const': -1})
        self.assertEqual((-(x + y)).coeffs, {'x': -1, 'y': -1})

    def test_LinExpr__mul__(self):
        x = Var('x', type='cont')
        y = Var('y', type='cont')
        self.assertEqual(((x + 1) * 1).coeffs, {'x': 1, 'const': 1})
        self.assertEqual(((x + y) * 1).coeffs, {'x': 1, 'y': 1})

    def test_LinExpr__rmul__(self):
        x = Var('x', type='cont')
        y = Var('y', type='cont')
        self.assertEqual((1 * (x + 1)).coeffs, {'x': 1, 'const': 1})
        self.assertEqual((1 * (x + y)).coeffs, {'x': 1, 'y': 1})

    def test_LinExpr__truediv__(self):
        x = Var('x', type='cont')
        y = Var('y', type='cont')
        self.assertEqual(((x + 1)/2).coeffs, {'x': 1/2, 'const': 1/2})
        self.assertEqual(((x + y)/2).coeffs, {'x': 1/2, 'y': 1/2})

    def test_LinExpr__add__(self):
        x = Var('x', type='cont')
        y = Var('y', type='cont')
        self.assertEqual((x + 1 + 1).coeffs, {'x': 1, 'const': 2})
        self.assertEqual((x + x + y).coeffs, {'x': 2, 'y': 1})
        self.assertEqual((x + y + y).coeffs, {'x': 1, 'y': 2})
        self.assertEqual(((x + 1) + x).coeffs, {'x': 2, 'const': 1})

    def test_LinExpr__radd__(self):
        x = Var('x', type='cont')
        y = Var('y', type='cont')
        self.assertEqual((x + 1 + 1).coeffs, {'x': 1, 'const': 2})
        self.assertEqual((x + (x + y)).coeffs, {'x': 2, 'y': 1})
        self.assertEqual((x + (y + y)).coeffs, {'x': 1, 'y': 2})
        self.assertEqual((x + (1 + x)).coeffs, {'x': 2, 'const': 1})

    def test_LinExpr__sub__(self):
        x = Var('x', type='cont')
        y = Var('y', type='cont')
        self.assertEqual((x + 1 - 1).coeffs, {'x': 1, 'const': 0})
        self.assertEqual((x + x - y).coeffs, {'x': 2, 'y': -1})
        self.assertEqual((x + y - y).coeffs, {'x': 1, 'y': 0})
        self.assertEqual(((x + 1) - x).coeffs, {'x': 0, 'const': 1})

    def test_LinExpr__rsub__(self):
        x = Var('x', type='cont')
        y = Var('y', type='cont')
        self.assertEqual((1 - (x + 1)).coeffs, {'x': -1, 'const': 0})
        self.assertEqual((x - (x - y)).coeffs, {'x': 0, 'y': 1})
        self.assertEqual((x - (y - y)).coeffs, {'x': 1, 'y': 0})
        self.assertEqual((x - (1 - x)).coeffs, {'x': 2, 'const': -1})

    def test_LinConstr__eq__(self):
        x = Var('x', type='cont')
        y = Var('y', type='cont')
        self.assertEqual(str(x + y == 2), 'x + y == 2')
        self.assertEqual(str(1 * x + 2 * y == 2), 'x + 2 * y == 2')

    def test_Var__eq__(self):
        x = Var('x', type='cont')
        self.assertEqual(str(x == 2), 'x == 2')

    def test_Var__le__(self):
        x = Var('x', type='cont')
        self.assertEqual(str(x <= 2), 'x <= 2')

    def test_Var__ge__(self):
        x = Var('x', type='cont')
        self.assertEqual(str(x >= 2), 'x >= 2')


if __name__ == '__main__':
    unittest.main()
