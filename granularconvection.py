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
g = np.array([0.0, -9.8, 0.0])  # (gravitational field strength)
s = np.array([[ # 0 to 0.2
    [0.12, 0.1, 0], # x, y, z
    [0.1, 0.2, 0],
    [0.13, 0.19, 0],
    [0, 0.16, 0]
    ]])
v = np.array([[ 
    [0.2, 0.6, 0],
    [0.02, 0.3, 0],
    [-0.3, 0.01, 0],
    [-0.01, -0.3, 0]
    ]])

R = np.array([0.007, 0.008, 0.01, 0.008])
rho = np.array([0.9, 0.9, 0.9, 0.9])

### VARIABLES
t_step = 1/24 # (seconds)
Mu = 1.82*10**-4 # (air viscosity [Pa*s])

### FUNCTIONS
def WRITE(file, data):
    with open(file, "w", newline='') as fin:
        writer = csv.writer(fin)
        writer.writerows(data)
        print(f"data written to '{file}'")
    
def get_Vol(R):
    vol = (4/3)*math.pi*(R**3)
    return vol

def get_m(Vol, rho):
    return Vol*rho

def get_W(m, g):
    return np.transpose([m])@[g]

def get_Fnet(W, Fdrag):
    return W+Fdrag

def get_Fdrag(Mu, Ri, Vi): # (air resistance using stokes formula)
    const = -6*math.pi*Mu
    Ri_new = np.transpose([Ri])@([[const, const, const]])
    Fdrag = Ri_new*Vi # vi is. at one t step. ok.
    return Fdrag

def get_allSij(Si, Sj): # 2d arrays
    size = len(Si)
    
    Sij = np.array([[np.array([0.0]*3)]*size]*size)
    for row in range(size):
        for col in range(row+1, size):
            Sij[row][col] = Si[row][col] - Sj[row][col]
            Sij[col][row] = -(Si[row][col] - Sj[row][col])
    return Sij

def get_allHij(R, allSij):
    size = len(R)
    allHij = np.array([[0.0]*size]*size)
    allSij_magnitudes = get_allSij_magnitudes(allSij)
    
    for row in range(size):
        for col in range(row, size):
            allHij[row][col] = R[row] + R[col] - allSij_magnitudes[row][col]
            allHij[col][row] = allHij[row][col]
    return allHij

def get_magnitude(s):
    return math.sqrt(sum(i**2 for i in s))

def get_unit_vector(s): # particle
    return s/get_magnitude(s)

def get_allSij_magnitudes(allSij):
    size = len(allSij)
    allSij_magnitudes = np.array([[0.0]*size]*size)
    for row in range(size):
        for col in range(size):
            allSij_magnitudes[row][col] = get_magnitude(allSij[row][col])

    return allSij_magnitude

def get_allReff(R):
    size = len(R)
    allReff = np.array([[0.0]*size]*size)
    for row in range(size):
        for col in range(size):
            allReff[row][col] = (R[row]*R[col])/(R[row]+R[col])
            allReff[col][row] = allReff[row][col]
    return allReff

def get_Etilde(youngE, possionNu):
    return youngE/(1-possionNu**2)


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

for t_index in range(48):
    if t_index == 0:
        Fdrag = get_Fdrag(Mu, R, v[t_index])
    else:
        Fdrag =get_Fdrag(Mu, R, v[t_index-1])
    
    Fnet = get_Fnet(W, Fdrag)
    if t_index == 0:
        a = create_a(Fnet, m)
    else:
        a = append_a(a, Fnet, m)
##    print(v[t_index-1], Fdrag, Fnet, "-----------", sep="\n\n")
    v = append_v(v, a, t_step, t_index)
    s = append_s(s, v, a, t_step, t_index)
    

graphs_out(s, R)

#==== export to video ====#
    
images = [img for img in os.listdir(current_directory) if img.endswith(".png")]
images.sort()

frame = cv2.imread(os.path.join(current_directory, images[0]))
height, width, layers = frame.shape
size = (width, height)
video_name = '0_video.mp4'
fps = 24
out = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

for image in images:
    img_path = os.path.join(current_directory, image)
    frame = cv2.imread(img_path)
    out.write(frame)

out.release()
print(f"video '{video_name}' created successfully")

#==== write to csv ====#
os.chdir(r"/Users/Abigail/Desktop/Sciences/ISS/ISS2.0/")
WRITE("data.csv", [[f"particle{i+1}" for i in range(len(R))], *s.tolist()])
