
import math
import random
import sys
import numpy as np



def fp_rate(weight, k_val, etau_over_N, N, R, m_val):
    return (1 - ((1 - (1 / (weight * R))) ** (k_val * N * etau_over_N))) ** k_val

def vanila_fp_rate(N, R):
    k_val = math.log(2) * R / N
    return (1 - ((1 - (1 / R)) ** (k_val * N))) ** k_val


B = 100 # Say we have 100 categories (100 bloom filters)
# N_vals = [10000, 100000, 1000000]
# R_vals = [1000, 10000, 100000, 1000000]
N_vals = [100000]
R_vals = [1000000]
m_vals = [1, 3, 5]
e_tau_over_N_vals = [0.3, 0.1, 0.01, 0.005] # sum(jacc_sim(x_a, x_u) ** m) / N



# Say we know distribution beforehand
# And k is optimized to k = ln(2) * c_i * R / (jacc_sim(x_a, x_u) ** m  * N)
# for individual bloom filters...
for R in R_vals:
    for m in m_vals:
        for N in N_vals:
            for s in e_tau_over_N_vals:
                num_buckets = 4 ** m
                # k = math.log(2) * (R / num_buckets) / N
                k = math.ceil(math.log(2) * R / N)
                print("""
                    Total hash space size {}   Num hash func per bloom filter {}   jacc sim^m sum N  {}
                    Num min hash func {}  Hash space size per bloom filter {}
                    fp_rate   :    {}
                    vanila_fp_rate : {}
                """.format(
                    R, k, s * N, m, R / num_buckets, fp_rate(1 / num_buckets, k, s, N, R, m), vanila_fp_rate(N, R)
                ))






