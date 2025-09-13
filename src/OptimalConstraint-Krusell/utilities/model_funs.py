import numpy as np

def capital_labor(asset, productivity):
    return asset.sum(len(asset.shape)-1), productivity.sum(len(productivity.shape)-1)

def prices(tfp, alpha, delta, K, L, marginal, asset, productivity):
    Y = np.exp(tfp) * K ** alpha * L ** (1-alpha)
    r = alpha * Y / K - delta
    w = (1 - alpha) * Y / L
    r_k = alpha * (alpha - 1) * np.exp(tfp) * K ** (alpha - 1) * L ** (1-alpha)
    w_k = alpha * (1 - alpha) * np.exp(tfp) * K ** alpha * L ** (-alpha)
    Lambda = (marginal * (r_k.unsqueeze(0) * asset + w_k.unsqueeze(0)  * productivity.unsqueeze(1) )).sum(1)
    return Y, r, w, Lambda
# end

# procedure iterates on aggregate and idiosyncratic shock both
def consumption(marginal, gamma):
    return np.maximum(marginal,0.0001) ** (-1/gamma)

def risk_loadings(xi_z, xi_A, sigma_z, sigma):
    varsigma_z = - xi_z * sigma_z
    varsigma = - xi_A * sigma
    return varsigma_z, varsigma
# end



def siml_xi(xi, rho, r, dt, Lambda, varsigma_z, varsigma, dWj, dWk):
    return 1 / ((rho - r) * dt) * (xi - Lambda * dt - varsigma_z * dWj.unsqueeze(1) - varsigma * dWk.unsqueeze(0))

def siml_z(z, theta, hat_z, dt, sigma_z, dWj, bound):
    z_drift = theta * (hat_z - z)
    z_next = z + z_drift * dt + sigma_z * dWj
    z_next = max(bound[0], z_next)
    z_next = min(bound[1], z_next)
    return z_next, z_drift

def siml_A(A, eta, hat_A, dt, sigma, dWk, bound):
    A_drift = eta * (hat_A - A)
    A_next = A + A_drift * dt + sigma * dWk
    A_next = max(bound[0], A_next)
    A_next = min(bound[1], A_next)
    return A_next, A_drift