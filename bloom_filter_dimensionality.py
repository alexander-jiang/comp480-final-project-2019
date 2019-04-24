# does bloom filter false positive rate depend on dimensionality of the data?

# i.e. if the input objects are more similar, does that affect the bloom filter
# false positive rate? (it shouldn't, right)

import math
import random
import sys
import numpy as np
import json
import time
from tqdm import tqdm
from bitarray import bitarray
import matplotlib
import matplotlib.pyplot as plt

# hash into a 32-bit unsigned int i.e. R = 2^32, assuming 64-bit machine word
def universal_hash():
    M = np.uint64(32) # how many bits for # of bins?
    w = np.uint64(64) # how many bits for a machine word? (for efficient execution)
    # use multiply-add-shift (choose odd a and non-negative b)
    a = random.randrange(2 ** w)
    if a % 2 == 0:
        a += 1
    b = random.randrange(2 ** (w - M))
    def hash(x):
        return np.uint32(np.uint64(a * x + b) >> (w - M))
    return hash

def universal_hash_vectors(vector_dim):
    hashfuncs = [universal_hash() for j in range(vector_dim)]
    def hash(vec):
        hash_val = 0
        for j in range(vector_dim):
            hash_val += hashfuncs[j](vec[j])
        return hash_val
    return hash

class BloomFilter():
    def __init__(self, num_keys, fp_rate, hash_factory, *hash_factory_args):
        self.initialized = False
        R_over_N = np.log(fp_rate) / np.log(0.618)
        self.k = math.ceil(R_over_N * np.log(2))
        self.R = math.ceil(R_over_N * num_keys)
        self.initialize_hashfuncs_and_filter(hash_factory, *hash_factory_args)

    def set_range_and_k(self, R, k, hash_factory, *hash_factory_args):
        self.initialized = False
        self.k = math.ceil(k)
        self.R = R
        self.initialize_hashfuncs_and_filter(hash_factory, *hash_factory_args)

    def initialize_hashfuncs_and_filter(self, hash_factory, *hash_factory_args):
        self.hashfuncs = [hash_factory(*hash_factory_args) for i in range(self.k)]
        self.filter = bitarray(self.R)
        self.filter.setall(False)
        self.initialized = True

    def insert(self, key):
        assert self.initialized, "Must call initialize_hashfuncs_and_filter first!"
        # print(f"before inserting {key}, {self.filter.count()} bits set to True")
        for hashfunc in self.hashfuncs:
            self.filter[hashfunc(key) % self.R] = True
        # print(f"after inserting {key}, {self.filter.count()} bits set to True")

    def test(self, key):
        assert self.initialized, "Must call initialize_hashfuncs_and_filter first!"
        for hashfunc in self.hashfuncs:
            if not self.filter[hashfunc(key) % self.R]:
                return False
        return True

def generate_random_vectors(vector_dim=100, num_vectors=100000):
    return np.random.randn(num_vectors, vector_dim) # dtype is float64

def generate_plots(vector_dims, desired_fp_rate):
    with open(f"bf_dimensionality_fp_rates.json", "r") as f:
        empirical_fp_rates = json.load(f)
    with open(f"bf_dimensionality_insert_times.json", "r") as f:
        insert_times = json.load(f)

    empirical_fp_rate_means = []
    empirical_fp_rate_stds = []
    for vector_dim in vector_dims:
        empirical_fp_rate_means.append(np.mean(empirical_fp_rates[str(vector_dim)]))
        empirical_fp_rate_stds.append(np.std(empirical_fp_rates[str(vector_dim)]))

    fig1, ax1 = plt.subplots()
    width = 0.15
    ind = np.arange(len(vector_dims))
    newp = ax1.bar(ind, empirical_fp_rate_means, width, yerr=empirical_fp_rate_stds)
    ax1.axhline(y=desired_fp_rate, linestyle='--', color='black')
    ax1.set_xticks(ind)
    ax1.set_xticklabels([f"d = {vector_dim}" for vector_dim in vector_dims])
    ax1.set_ylabel("Mean & std dev of empirical FP rate (10 trials)")
    ax1.set_title("Empirical FP rate vs. vector dimension")
    fig1.savefig("figures/bf_dimensionality_fp_rates.png")


def main():
    random.seed(99423)
    vector_dims = [10, 30, 50, 100, 300]
    num_vectors = 10000
    desired_fp_rate = 0.01
    num_trials = 10

    empirical_fp_rates = {}
    insert_times = {}
    for vector_dim in vector_dims:
        empirical_fp_rates[vector_dim] = []
        insert_times[vector_dim] = []
        for trial_num in range(num_trials):
            print(f"Vector dimension = {vector_dim}, trial {trial_num + 1}")
            vectors = generate_random_vectors(vector_dim=vector_dim, num_vectors=num_vectors)
            bf = BloomFilter(num_vectors, desired_fp_rate, universal_hash_vectors, vector_dim)

            print(f"Inserting {num_vectors} vectors")
            insert_start = time.process_time()
            count = 0
            for vec in tqdm(vectors):
                # if count % 1000 == 0:
                #     print(f"inserted {count} / {num_vectors}")
                bf.insert(vec)
                count += 1
            insert_time = time.process_time() - insert_start

            # count = 0
            # for vec in vectors:
            #     if count % 1000 == 0:
            #         print(f"tested {count} / {num_vectors}")
            #     if not bf.test(vec):
            #         print(f"ERROR false negative detected in vanilla Bloom filter!!")
            #         print(f"key: {vec}")
            #     count += 1

            num_test_vectors = 5000
            test_vectors = generate_random_vectors(vector_dim=vector_dim, num_vectors=num_test_vectors)
            fp_count = 0
            for test_vec in test_vectors:
                if test_vec in vectors:
                    num_test_vectors -= 1
                    continue
                if bf.test(test_vec):
                    fp_count += 1
            fp_rate = float(fp_count) / num_test_vectors
            print("Actual fp rate: ", fp_rate)
            print("Expected fp rate: ", desired_fp_rate)

            empirical_fp_rates[vector_dim].append(fp_rate)
            insert_times[vector_dim].append(insert_time)

        with open(f"bf_dimensionality_fp_rates.json", "w") as f:
            json.dump(empirical_fp_rates, f)
        with open(f"bf_dimensionality_insert_times.json", "w") as f:
            json.dump(insert_times, f)

if __name__ == "__main__": main()
