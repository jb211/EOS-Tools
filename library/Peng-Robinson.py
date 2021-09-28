import math
from scipy.optimize import fsolve

R_ig = 8.3145 #J*mol^{-1}*K^{-1}

def alpha_fn(w,T, Tc, Pc):
    if w < 0.5:
        m = 0.3764 + 1.54226*w -0.26992*(w**2)
    else:
        m = 0.37960 +1.485*w - 0.1644*(w**2)
    T_r = T / Tc
    alf = (1 + m * (1-math.sqrt(T_r)))**2
    return alf

def a_fn(Tc,Pc):
    return 0.457235 * R_ig**2 * Tc**2 / Pc

def b_fn(Tc,Pc):
    return 0.077796  * R_ig * Tc / Pc

def A_fn(P,T,Tc,Pc,w):
    alfa = alpha_fn(w,T, Tc, Pc)
    a = a_fn(Tc,Pc)
    return a*alfa*P / (R_ig**2 * T**2)

def B_fn(P,T,Tc,Pc):
    b = b_fn(Tc,Pc)
    return b * P / (R_ig * T)

def solve_PR_for_V(P, T, a, b):
    def PR_root(x):
        return P - R_ig*T / (x[0] - b) + a/(x[0]**2 + 2*x[0]*b - b**2)
    root = fsolve(PR_root, [0.001])
    return (P,root[0])

def solve_idealfor_V(P,T):
    return R_ig * T / P

def calc_z_factor(T,P,V):
    return P * V / (R_ig * T)

def solve_Z_factor_root(A,B):
    def Z_poly(x):
        return x[0]**3 - (1-B)* x[0]**2 + (A - 2*B -3*B**2)*x[0] - (A*B - B**2 - B**3)
    z_root = fsolve(Z_poly, [0.5])
    return z_root


if __name__ == "__main__":
    w_t = 0.228
    t_sc = 304.12
    p_sc = 7380000
    tt = 373.69
    Pt = 101325
    
    a_test = a_fn(t_sc, p_sc)
    b_test = b_fn(t_sc, p_sc)
    A_test = A_fn(Pt, tt, t_sc, p_sc, w_t)
    B_test = B_fn(Pt, tt, t_sc, p_sc)

    p_test_psi = [
    14.7 
    , 31.6 
    , 75.2
    , 162.8 
    , 301.2 
    , 750.1 
    , 1510
    , 2901.1 
    , 7368.1 
    , 14527 
    , 24650 ]

    psi_to_Pa = lambda x: x * 6894.76
    lmap = lambda func, *iterable: list(map(func, *iterable))
    p_test_Pa = lmap(psi_to_Pa, p_test_psi)

    V_test_data = lmap(lambda  p: solve_PR_for_V(p, tt, a_test, b_test), p_test_Pa)
    z_factors_data = zip(p_test_Pa, lmap(lambda (v,p): calc_z_factor(tt, v,p), V_test_data))