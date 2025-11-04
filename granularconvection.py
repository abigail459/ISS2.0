# rmb to change directory if needed
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from matplotlib.patches import Circle
import math
import csv

### DIRECTORY
os.chdir(r"/Users/Abigail/Desktop/Sciences/ISS/ISS2.0/Figures/")
current_directory = os.getcwd()

### LISTS
g = np.array([0.0, 0.0, 0.0])  # (gravitational field strength)
s = np.array([[ # (0 to 0.2) (3d array)
    [0.13, 0.08, 0], # (x, y, z)
    [0.13, 0.1, 0],
    [0.04, 0.19, 0],
    [0, 0.16, 0]
    ]]) 
v = np.array([[  # velocity of travel
    [0.0, 0.2, 0],
    [0.0, -0.1, 0],
    [-0.3, 0.01, 0],
    [-0.01, -0.3, 0]
    ]])

R = np.array([0.007, 0.008, 0.01, 0.008])
rho = np.array([0.9, 0.9, 0.9, 0.9])
fps = 48
simulation_duration = 2 # seconds

### VARIABLES
t_step = 1/24 # (seconds)
Mu = 1.82*10**-4 # (air viscosity [Pa*s])

### FUNCTIONS
def WRITE(file, data):
    with open(file, "w", newline='') as fin:
        writer = csv.writer(fin)
        writer.writerows(data)
        print(f"data written to '{file}'")
    
def get_Vol(R): # (scalar)
    vol = (4/3)*math.pi*(R**3)
    return vol

def get_m(Vol, rho): # (1d array)
    return Vol*rho

def get_W(m, g): # (2d array)
    return np.transpose([m])@[g]

def get_Fnet(W, Fdrag, fCollisions):
    return W+Fdrag+fCollisions

def get_Fdrag(Mu, Ri, Vi): # (air resistance using stokes formula)
    const = -6*math.pi*Mu
    Ri_new = np.transpose([Ri])@([[const, const, const]])
    Fdrag = Ri_new*Vi # vi is. at one t step. ok.
    return Fdrag

def get_allSij(latest_s): # (3d arrays) 
    size = len(latest_s)
    
    allSij = np.array([[np.array([0.0]*3)]*size]*size)
    for row in range(size):
        for col in range(size):
            allSij[row][col] = latest_s[row] - latest_s[col]
            allSij[col][row] = -(allSij[row][col])
    return allSij

def get_allHij(R, allSij): #(2d array) 
    size = len(R) # number of particles
    allHij = np.zeros((size, size))
    allSij_magnitudes = get_allSij_magnitudes(allSij)
    
    for row in range(size):
        for col in range(row, size):
            allHij[row][col] = R[row] + R[col] - allSij_magnitudes[row][col]
            allHij[col][row] = allHij[row][col]
    return allHij

def get_magnitude(s): # (3d array)
    return math.sqrt(sum(i**2 for i in s))

def get_unit_vector(s): # (particle)
    return s/get_magnitude(s)

def get_allSij_magnitudes(allSij): # (2d array)
    size = len(allSij)
    allSij_magnitudes = np.array([[0.0]*size]*size)
    for row in range(size):
        for col in range(size):
            allSij_magnitudes[row][col] = get_magnitude(allSij[row][col])

    return allSij_magnitudes

def get_allReff(R): # (2d array)
    size = len(R)
    allReff = np.array([[0.0]*size]*size)
    for row in range(size):
        for col in range(size):
            allReff[row][col] = (R[row]*R[col])/(R[row]+R[col])
            allReff[col][row] = allReff[row][col]
    return allReff

def get_Etilde(youngE, possionNu):
    return youngE/(1-possionNu**2)

def get_allSijhat(allSij): # (3d array)
    allSij_magnitudes = get_allSij_magnitudes(allSij)
    allSijhat = allSij/allSij_magnitudes[:, :, np.newaxis]
    # return np.nan_to_num(allSijhat)
    return allSijhat

def get_allfij(R, Etilde, allReff, allHij, allSijhat):
    size = len(R)
    allfij = np.array([[[0.0]*3]*size]*size)
    for row in range(size):
        for col in range(size):
            if row != col and allHij[row][col] > 0:
                allfij[row][col] = ((2/3)*Etilde*
                                    (math.sqrt(allReff[row][col]))*
                                    ((max(allHij[row][col], 0.0))**(3/2))*
                                    allSijhat[row][col]
                                    ) 
            allfij[col][row] = -allfij[row][col]
    return allfij

def get_allfCollisions(R, allfij): # (2d array)
    size = len(R)
    allfCollisions = np.array([[0.0]*3]*size)
    for row in range(size):
        for col in range(size):
            allfCollisions[row] += allfij[row][col]
    return allfCollisions


def create_a(Fnet, m): 
    m = np.transpose([m])@np.array([[1,1,1]])

    return [Fnet/m]

def append_a(a, Fnet, m):
    m = np.transpose([m])@np.array([[1,1,1]])
    
    new_a = Fnet/m
    a = np.vstack((a, [new_a]))  # ?? double bracket ??
    
    return a

def append_v(v, a, t_step, t_index):
    new_v = v[t_index]+a[t_index]*t_step
    v = np.vstack((v, [new_v]))
    
    return v

def append_s(s, v, a, t_step, t_index):
    new_s = s[t_index]+v[t_index]*t_step+0.5*a[t_index]*(t_step)**2
    s = np.vstack((s, [new_s]))
    
    return s


def graphs_out(s, R): 

    for index, t_index in enumerate(s):
        ax = plt.gca() # (get current axes)

        # x and y lists
        nump = len(t_index)
        x = []
        y = []
        for p_index in range(nump):
            x.append(t_index[p_index][0])
            y.append(t_index[p_index][1])
            
        
        for xi, yi, ri in zip(x, y, R): # (i for particle)
            circle = Circle((xi, yi), ri, edgecolor='blue', facecolor='skyblue', alpha=0.7)
            ax.add_patch(circle)
        ax.set_aspect('equal')

        # setting box limits
        ax.set_xlim(0, 0.2)
        ax.set_ylim(0, 0.2)
        
        plt.savefig("fig %04d.png" %(int(index)))
        plt.close()


#============================================================================================================#


#==== execution ====#

Vol = get_Vol(R)
m = get_m(Vol, rho)
W = get_W(m, g)
Etilde = get_Etilde(10**9, 0.4)


for t_index in range(fps*simulation_duration):
    allSij = get_allSij(s[-1])
    allSijhat = get_allSijhat(allSij)
    allHij = get_allHij(R, allSij)
    allReff = get_allReff(R)
    allfij = get_allfij(R, Etilde, allReff, allHij, allSijhat)
    fCollisions = get_allfCollisions(R, allfij)
    if t_index == 0:
        Fdrag = get_Fdrag(Mu, R, v[t_index])
    else:
        Fdrag = get_Fdrag(Mu, R, v[t_index-1])
    
    Fnet = get_Fnet(W, Fdrag, fCollisions)
    if t_index == 0:
        a = create_a(Fnet, m)
    else:
        a = append_a(a, Fnet, m)
##    print(v[t_index-1], Fdrag, Fnet, "-----------", sep="\n\n")
    v = append_v(v, a, t_step, t_index)
    s = append_s(s, v, a, t_step, t_index)
    
    if t_index == 0 or t_index == 1:
        print("allfCollisions", fCollisions)

graphs_out(s, R)

#==== export to video ====#
    
images = [img for img in os.listdir(current_directory) if img.endswith(".png")]
images.sort()

frame = cv2.imread(os.path.join(current_directory, images[0]))
height, width, layers = frame.shape
size = (width, height)
video_name = '0_video.mp4'
out = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

for image in images:
    img_path = os.path.join(current_directory, image)
    frame = cv2.imread(img_path)
    out.write(frame)

out.release()
print(f"video '{video_name}' created successfully")

#==== write to csv ====#

os.chdir(r"/Users/Abigail/Desktop/Sciences/ISS/ISS2.0/")
WRITE("data.csv", [[f"particle{i+1}" for i in range(len(R))], *s.tolist()]) # stores each particle's coordinates for each timestep
