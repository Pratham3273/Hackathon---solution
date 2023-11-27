import pyomo.environ as pyo
from pyomo.environ import *
from pyomo.opt import SolverFactory, SolverStatus, TerminationCondition
import pandas as pd
import numpy as np

####################    Data    ########################

model = pyo.ConcreteModel()

df1 = pd.read_excel('Warehouse_1_data.xlsx', sheet_name='Distance Matrix')
df2 = pd.read_excel('Warehouse_1_data.xlsx', sheet_name='Demand')

dist_mat = df1.drop('Node_ID', axis = 1).to_numpy()

############# Parameters ###############
V = 5   # Number of vehicles
N = 9   # Total Number of nodes
C = 7   # Total Number of stores

########### Variables ############

model.x = pyo.Var(range(N), range(N), range(V), within = pyo.Integers, bounds = (0,1))
x = model.x

model.s = pyo.Var(range(N), range(V), bounds = (0, None))
s = model.s 

########### Constraints #############

#C1: Visit only one customer
model.C1 = pyo.ConstraintList()
for i in range(1, N-1):
    model.C1.add(expr = sum(x[i,j,k] for j in range(N) for k in range(V)) == 1)

#C2: Vehicles go out from warehouse
model.C2 = pyo.ConstraintList()
for k in range(V):
    model.C2.add(expr = sum(x[0,j,k] for j in range(N)) == 1)

#C3: Flow balance at stores
model.C3 = pyo.ConstraintList()
for h in range(1,N-1):
    for k in range(V):
        model.C3.add(expr = sum(x[i,h,k] for i in range(N)) == sum(x[h,j,k] for j in range(N)))

#C4: Vehicles return to warehouse
model.C4 = pyo.ConstraintList()
for k in range(V):
    model.C4.add(expr = sum(x[i,N-1, k] for i in range(N)) == 1)

#C5 and C6: Total demand is less than equal to vehicle capacity
q_w = 61200 #Weight capacity for vehicle
q_v = 2389  #Volume capacity for vehicle

model.C5 = pyo.ConstraintList()
for k in range(V):
    model.C5.add(expr = sum(df2['Demand1'][i] * x[i,j,k] for i in range(1, N-1) for j in range(N)) <= q_w)

model.C6 = pyo.ConstraintList()
for k in range(V):
    model.C6.add(expr = sum(df2['Demand2'][i]*x[i,j,k] for i in range(1, N-1) for j in range(N)) <= q_v)

#C7: Time window constraint
speed = 40
time_mat = dist_mat/speed
model.C7 = pyo.ConstraintList()
for i in range(N):
    for j in range(N):
        for k in range(V):
            model.C7.add(expr = s[i,k] + time_mat[i,j] - 100000*(1 - x[i,j,k]) <= s[j,k])

#C8 and 9: Time window limits

model.C8 = pyo.ConstraintList()
for i in range(N):
    for k in range(V):
        model.C8.add(expr = df2['D_start'][i] <= s[i,k] )


model.C9 = pyo.ConstraintList()
for i in range(N):
    for k in range(V):
        model.C9.add(expr = s[i,k] <= df2['D_end'][i]) 

#C10:
model.C10 = pyo.ConstraintList()
for i in range(N):
    for j in range(N):
        if i == j:
            model.C10.add(sum(x[i,j,k] for k in range(V)) == 0)

########### Objective function ############

vehicle_dist = sum([dist_mat[i,j]*x[i,j,k] for i in range(N) for j in range(N) for k in range(V)])

model.obj = pyo.Objective(expr = vehicle_dist, sense = minimize)
opt = SolverFactory('glpk')
results = opt.solve(model)
#model.pprint()

if (results.solver.status == SolverStatus.ok) and (
        results.solver.termination_condition == TerminationCondition.optimal):
    print("Optimal Obj. value: " , pyo.value(model.obj))
    model.pprint()
    x_values = model.x.extract_values()

# Printing the values of x
    for key, value in x_values.items():
        print(f"x{key} = {value}")
elif results.solver.termination_condition == TerminationCondition.infeasible:
    print('****************************')
    print("Infeasible. Change Parameters")
else:
    print("Solver Status: ", results.solver.status)