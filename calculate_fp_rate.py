
import math
import random
import sys
import numpy as np



def fp_rate(weight, k_val, sum_jacc, N, R, m_val):
    return (1 - ((1 - (1 / (weight * R))) ** (k_val * N * (sum_jacc ** m_val)))) ** k_val

def vanila_fp_rate(N, R):
    k_val = math.log(2) * R / N
    return (1 - ((1 - (1 / R)) ** (k_val * N))) ** k_val


B = 100 # Say we have 100 categories (100 bloom filters)
# N_vals = [10000, 100000, 1000000]
# R_vals = [1000, 10000, 100000, 1000000]
N_vals = [10000]
R_vals = [10000, 100000]
m_vals = [1, 5, 9, 11]
avg_jacc_to_m_vals = [0.8, 0.5, 0.3, 0.1] # jacc_sim(x_a, x_u) ** m



# Say we know distribution beforehand
# And k is optimized to k = ln(2) * c_i * R / (jacc_sim(x_a, x_u) ** m  * N)
# for individual bloom filters...
for s in avg_jacc_to_m_vals:
    for R in R_vals:
        for N in N_vals:
            for m in m_vals:
                k = math.log(2) * s * R / ((s ** m) * N)
                print("""
                ### Knowing distribution beforehand (weight = avg jacc sim over N)
                    Total hash space size {}   Num hash func per bloom filter {}   avg jacc sim  {}    N {}
                    Num min hash func {}
                    fp_rate   :    {}
                    vanila_fp_rate : {}
                """.format(
                    R, k, s, N, m, fp_rate(s, k, s, N, R, m), vanila_fp_rate(N, R)
                ))






