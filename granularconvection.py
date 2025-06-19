import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from matplotlib.patches import Circle
import math

### DIRECTORY
os.chdir(r"/Users/Abigail/Desktop/Sciences/ISS/ISS2.0/Figures/")
current_directory = os.getcwd()

### LISTS
g = np.array([0, -10, 0])  # <- gravitational field strength
s = np.array([[
    [5, 5, 0],
    [7, 10, 0],
    [8, 3, 0],
    [7, 4, 0]
    ]])
v = np.array([[
    [0, 0.2, 0],
    [0, 0.5, 0],
    [0, 9, 0],
    [0, 6, 0]
    ]])
##a = np.array([[
##    [0, g, 0],
##    [0, g, 0],
##    [0, g, 0],
##    [0, g, 0]
##    ]])
R = np.array([2, 1, 1, 0.5])
rho = np.array([0.9, 0.9, 0.9, 0.9])

### VARIABLES
t_step = 1/24 # seconds
t_index = 0 # each frame
p_index = 0 # each particle

### FUNCTIONS
def get_Vol(R):
    vol = (4/3)*math.pi*(R**3)
    return vol

def get_m(Vol, rho):
    return Vol*rho

def get_W(m, g):
    return np.transpose([m])@[g]

def get_Fnet(W): # tbc
    return W

def create_a(Fnet, m):
    m = np.transpose([m])@np.array([[1,1,1]])

    return [Fnet/m]
def append_a(a, Fnet, m):
    m = np.transpose([m])@np.array([[1,1,1]])
    
    new_a = Fnet/m
    a = np.vstack((a, [new_a]))  # double bracket?
    
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
        ax = plt.gca()

        # x and y lists
        nump = len(t_index)
        x = []
        y = []
        for p_index in range(nump):
            x.append(t_index[p_index][0])
            y.append(t_index[p_index][1])
        
        for xi, yi, ri in zip(x, y, R):
            circle = Circle((xi, yi), ri, edgecolor='blue', facecolor='skyblue', alpha=0.7)
            ax.add_patch(circle)
        ax.set_aspect('equal')

        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        
        plt.savefig("fig %04d.png" %(int(index)))
        plt.close()



#============================================================#


#==== execution ====#

Vol = get_Vol(R)
m = get_m(Vol, rho)
W = get_W(m, g)
print(W)



for t_index in range(48):
    Fnet = get_Fnet(W)
    if t_index == 0:
        a = create_a(Fnet, m)
    else:
        a = append_a(a, Fnet, m)
    v = append_v(v, a, t_step, t_index)
    s = append_s(s, v, a, t_step, t_index)

graphs_out(s, R)
print(a)


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

