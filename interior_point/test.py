from casadi import *
import numpy as np
from problem import interior_point_problem

nv =2
x = MX.sym('x',nv)

x_test = [2,3] 


#Objective Function
O = Function('F',[x],[(x[0]-4)**2+(x[1]-4)**2])

#Equality Constraint
G = Function('G',[x],[sin(x[0])-x[1]**2])

#Inequality Constraint
H = Function('H',[x],[x[0]**2+x[1]**2-4])

p = interior_point_problem(x,O,G,H,80, np.array([-2,-8]), 10*np.ones(1), 10*np.ones(1), 10*np.ones(1),2,1,1)

p.solve()



