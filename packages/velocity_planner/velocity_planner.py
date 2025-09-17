import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

def velocity_planner_simple(N,dt,v0,v_max, a_max, a_min, opti_silent=False):
    opti = ca.Opti()
    v = opti.variable(N+1)
    a = opti.variable(N)
    v_max_opti =  opti.parameter(N+1)
    a_max_opti =  opti.parameter(1)
    a_min_opti =  opti.parameter(1)

    opti.subject_to(v[0] == v0)
    for k in range(N):
        opti.subject_to(v[k+1] == v[k] + dt * a[k])

    opti.subject_to(opti.bounded(0.01, v, v_max_opti))
    opti.subject_to(opti.bounded(a_min_opti, a, a_max_opti))
    opti.subject_to(v[N] == 1.0)

    distance = ca.sum1(v) * dt
    opti.minimize(-distance + ca.sum1(a*a)*0.0)

    if opti_silent:
        p_opts = {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.sb': 'yes'}
        #p_opts = {'print_time':False, 'verbose':False}
        s_opts = {'print_level':0}
    else:
        p_opts = {}
        s_opts = {}

    opti.solver('ipopt',p_opts,s_opts)

    opti.set_value(v_max_opti, v_max)
    opti.set_value(a_max_opti, a_max)
    opti.set_value(a_min_opti, a_min)

    return opti, v, a