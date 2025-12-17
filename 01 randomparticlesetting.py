#01randomparticlesetting.py
import csv
import random
import numpy as np
import os

n_falling = 216 #no. of particles! :3

def WRITE(file, data):
    os.chdir("/Users/liliy/Documents/GitHub/ISS2.0/data/") #change this as needed
    with open(file, "w", newline='') as fin:
        writer = csv.writer(fin)
        writer.writerows(data)
        print(f"data written to '{file}'")


# RANGES OF PARTICLE PROPERTIES: (generates randomly between these numbers)
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
