from toyplex.model import Model

# Model parameters
# Purchasing cost of one package of 100 bags for each brand
c = [512, 451, 373]

# Number of bags for three coffee types for each brand
n = [[42, 45, 13],
     [32, 37, 31],
     [24, 22, 54]]

# Minimum number of bags for each coffee types that must be purchased
d = [330, 300, 100]

# Ranges for brand and coffee type
I = range(3)
J = range(3)

# Create model
m = Model()

# Add decision variables
y = []
for i in I:
    y.append(m.add_var(type='bin', name='y'+str(i+1)))
x = []
for i in I:
    x.append(m.add_var(type='int', name='x'+str(i+1)))

# Add constraints
for j in J:
    m.add_constr(sum(x[i]*n[i][j] for i in I) >= d[j])
for i in I:
    m.add_constr(x[i] <= 5*y[i])

# Set model objective
m.set_objective(sum(c[i] * x[i] for i in I), sense='min')
m.describe()

# Solve
m.optimize(verbose=False)

brands = {0: 'Starbucks',
          1: 'Second Cup',
          2: 'Tim Hortons'}


# Print solution
print('\nTOTAL COST: %g' % m.objval)
print('SOLUTION:')

for i, var in enumerate(x):
    print("Place {} orders from {}".format(int(var.val), brands[i]))
