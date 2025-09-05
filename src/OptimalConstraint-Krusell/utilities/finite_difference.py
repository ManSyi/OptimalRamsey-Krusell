import numpy as np
from ..classes import WorkSpace, Calibration
from .model_funs import consumption

def finite_diff(cal, ws, t, xi_fun):

    for j in range(cal.Nj):
        if ws.z_drift[j] > 0:
            ws.xi_z[j,:] = (xi_fun(ws.z[t,j] + cal.delta_diff,ws.a[t, j,:],ws.a[t,:,:],ws.A[t,:])
                              - xi_fun(ws.z[t,j],ws.a[t,j,:],ws.a[t,:,:],ws.A[t,:])) / cal.delta_diff
        elif ws.z_drift[j] < 0:
            ws.xi_z[j,:] = -(xi_fun(ws.z[t,j] - cal.delta_diff,ws.a[t, j,:],ws.a[t,:,:],ws.A[t,:])
                              - xi_fun(ws.z[t,j],ws.a[t,j,:],ws.a[t,:,:],ws.A[t,:])) / cal.delta_diff
        else:
            ws.xi_z[j,:] = ((xi_fun(ws.z[t,j] + cal.delta_diff,ws.a[t, j,:],ws.a[t,:,:],ws.A[t,:])
                              - xi_fun(ws.z[t,j] - cal.delta_diff,ws.a[t, j,:],ws.a[t,:,:],ws.A[t,:]))
                              / (2 * cal.delta_diff))

    for k in range(cal.Nk):
        if ws.A_drift[k] > 0:
            ws.xi_A[:,k] = ((xi_fun(ws.z[t,:],ws.a[t, :,k],ws.a[t,:,:],ws.A[t,:]
                                      + cal.delta_diff) - xi_fun(ws.z[t,:],ws.a[t, :,k],ws.a[t,:,:],ws.A[t,:]))
                              / cal.delta_diff)
        elif ws.A_drift[k] < 0:
            ws.xi_A[:,k] = -((xi_fun(ws.z[t,:],ws.a[t, :,k],ws.a[t,:,:],ws.A[t,:]
                                      - cal.delta_diff) - xi_fun(ws.z[t,:],ws.a[t, :,k],ws.a[t,:,:],ws.A[t,:]))
                              / cal.delta_diff)
        else:
            ws.xi_A[:,k] = (xi_fun(ws.z[t,:],ws.a[t, :,k],ws.a[t,:,:],ws.A[t,:]
                                      + cal.delta_diff) - xi_fun(ws.z[t,:],ws.a[t, :,k],ws.a[t,:,:],ws.A[t,:]
                                      - cal.delta_diff)) / (2 * cal.delta_diff)

def upwind(cal,ws, t, xi_fun):
    ws.xiF  = (xi_fun(ws.z[t, :], ws.a[t, :, :] + cal.delta_diff, ws.a[t, :, :], ws.A[t, :])
                       - xi_fun(ws.z[t, :], ws.a[t, :, :], ws.a[t, :, :], ws.A[t, :])) / cal.delta_diff
    ws.xiB  = -(xi_fun(ws.z[t, :], ws.a[t, :, :] - cal.delta_diff, ws.a[t, :, :], ws.A[t, :])
                        - xi_fun(ws.z[t, :], ws.a[t, :, :], ws.a[t, :, :], ws.A[t, :])) / cal.delta_diff
    ws.cF  = consumption(ws.xiF, cal.gamma)
    ws.cB  = consumption(ws.xiB, cal.gamma)
    ws.c0  = ws.r[np.newaxis,:] * ws.a[t, :, :] + ws.w[np.newaxis,:] * ws.z[t, :, np.newaxis]

    ws.sF = ws.c0 - ws.cF
    ws.sB= ws.c0 - ws.cB

    ws.indF  = ws.ws.sF > 0
    ws.indB  = ws.ws.sB  < 0
    ws.ind0  = (~ws.indF ) & (~ws.indB )

    ws.c = (ws.cF * ws.indF + ws.cB * ws.indB
                     + ws.c0* ws.ind0)

    ws.a[t+1,:,:] = ws.a[t,:,:] + (ws.sF  * ws.indF  + ws.sB  * ws.indB ) * cal.dt