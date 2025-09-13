import numpy as np
from ..classes import WorkSpace, Calibration
from .model_funs import consumption
from ..learning import xi_fun, NeuralNetwork

def finite_diff(cal, ws, t, xi_fun):

    for j in range(cal.Nj):
        if ws.z_drift[j] > 0:
            ws.xi_z[j,:] = (xi_fun(ws.model, ws.z[j] + cal.delta_diff,ws.a[ j,:],ws.a,ws.A)
                              - xi_fun(ws.model,ws.z[j],ws.a[j,:],ws.a,ws.A)) / cal.delta_diff
        elif ws.z_drift[j] < 0:
            ws.xi_z[j,:] = -(xi_fun(ws.model, ws.z[j] - cal.delta_diff,ws.a[ j,:],ws.a,ws.A)
                              - xi_fun(ws.model,ws.z[j],ws.a[j,:],ws.a,ws.A)) / cal.delta_diff
        else:
            ws.xi_z[j,:] = ((xi_fun(ws.model, ws.z[j] + cal.delta_diff,ws.a[ j,:],ws.a,ws.A)
                              - xi_fun(ws.model,ws.z[j] - cal.delta_diff,ws.a[ j,:],ws.a,ws.A))
                              / (2 * cal.delta_diff))

    for k in range(cal.Nk):
        if ws.A_drift[k] > 0:
            ws.xi_A[:,k] = ((xi_fun(ws.model,ws.z,ws.a[ :,k],ws.a[:,k],ws.A[k]
                                      + cal.delta_diff) - xi_fun(ws.model,ws.z,ws.a[:,k],ws.a,ws.A[k]))
                              / cal.delta_diff)
        elif ws.A_drift[k] < 0:
            ws.xi_A[:,k] = -((xi_fun(ws.model,ws.z,ws.a[ :,k],ws.a[:,k],ws.A[k]
                                      - cal.delta_diff) - xi_fun(ws.model,ws.z,ws.a[ :,k],ws.a,ws.A[k]))
                              / cal.delta_diff)
        else:
            ws.xi_A[:,k] = (xi_fun(ws.model,ws.z,ws.a[ :,k],ws.a[:,k],ws.A[k]
                                      + cal.delta_diff) - xi_fun(ws.model,ws.z,ws.a[ :,k],ws.a,ws.A[k]
                                      - cal.delta_diff)) / (2 * cal.delta_diff)

def upwind(cal:Calibration, ws:WorkSpace, t, xi_fun):
    ws.xiF  = (xi_fun(ws.model,ws.z, ws.a + cal.delta_diff, ws.a, ws.A)
                       - xi_fun(ws.model,ws.z, ws.a, ws.a, ws.A)) / cal.delta_diff
    ws.xiB  = -(xi_fun(ws.model,ws.z, ws.a - cal.delta_diff, ws.a, ws.A)
                        - xi_fun(ws.model,ws.z, ws.a, ws.a, ws.A)) / cal.delta_diff
    ws.cF  = consumption(ws.xiF, cal.gamma)
    ws.cB  = consumption(ws.xiB, cal.gamma)
    ws.c0  = ws.r[np.newaxis,:] * ws.a + ws.w[np.newaxis,:] * ws.z[ :, np.newaxis]

    ws.sF = ws.c0 - ws.cF
    ws.sB= ws.c0 - ws.cB

    ws.indF  = ws.sF > 0 & ws.xiF > 0
    ws.indB  = ws.sB  < 0 & ws.xiB > 0



    ws.ind0  = (~ws.indF ) & (~ws.indB )

    ws.c = (ws.cF * ws.indF + ws.cB * ws.indB
                     + ws.c0* ws.ind0)

    ws.a = min(max(ws.a + (ws.sF  * ws.indF  + ws.sB  * ws.indB ) * cal.dt, cal.a0_low), cal.a0_high)