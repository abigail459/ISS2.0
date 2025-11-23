#randomparticlesetting
import csv
import random
import numpy as np
import os

n_falling = 216

def WRITE(file, data):
    os.chdir("/Users/liliy/Documents/GitHub/ISS2.0/data/")
    with open(file, "w", newline='') as fin:
        writer = csv.writer(fin)
        writer.writerows(data)
        print(f"data written to '{file}'")


def s_gen():
    return float(random.uniform(0.015, 0.185))
def v_gen():
    return float(random.uniform(-0.005, 0.003))
def r_gen():
    return float(random.uniform(0.003, 0.007))



s_falling = np.array([[s_gen(), s_gen(), 0.0] for _ in range(n_falling)]) # range is no. of particles
v_falling = np.array([[v_gen(), 0.0, 0.0] for _ in range(n_falling)])
R_falling = np.array([r_gen() for _ in range(n_falling)])

WRITE("s_falling_data.csv", s_falling)
WRITE("v_falling_data.csv", v_falling)
WRITE("R_falling_data.csv", R_falling[:, np.newaxis])

np.savez(
    "falling_data.npz", 
    s_falling = s_falling,
    v_falling = v_falling,
    R_falling = R_falling
    )

# print(len(s_falling))
# s_fallingx = np.array([
#     [0.05, 0.16, 0.0],
#     [0.08, 0.16, 0.0],
#     [0.10, 0.16, 0.0],
#     [0.12, 0.16, 0.0],
#     [0.15, 0.16, 0.0],
#     [0.07, 0.13, 0.0],
#     [0.13, 0.13, 0.0]
# ])

# v_falling = np.array([
#     [0.005, 0.0, 0.0],
#     [-0.005, 0.0, 0.0],
#     [0.003, 0.0, 0.0],
#     [-0.003, 0.0, 0.0],
#     [0.002, 0.0, 0.0],
#     [0.004, 0.0, 0.0],
#     [-0.004, 0.0, 0.0]
# ])

# R_falling = np.array([0.004, 0.0042, 0.0038, 0.0045, 0.004, 0.0043, 0.0041])
