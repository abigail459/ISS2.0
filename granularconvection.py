import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from matplotlib.patches import Circle

### DIRECTORY
os.chdir(r"/Users/Abigail/Desktop/Sciences/ISS/ISS2.0/Figures/")
current_directory = os.getcwd()

### LISTS
x = np.array([5,7,8,7])
y = np.array([5,10, 3,4])
R = np.array([2, 1, 1, 0.5])

### VARIABLES
t_step = 1/24 # seconds
t_index = 0 # each frame
g_index = 0 # each particle


#============================================================#

#==== plotting ====#

for t_index in range(24):
    ax = plt.gca()
    y_mod = [val -  t_index*0.2 for val in y]
    for xi, yi, ri in zip(x, y_mod, R):
        circle = Circle((xi, yi), ri, edgecolor='blue', facecolor='skyblue', alpha=0.7)
        ax.add_patch(circle)
    ax.set_aspect('equal')

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    
    plt.savefig("fig %04d.png" %(t_index))
    plt.close()


#==== export to video ====#
    
images = [img for img in os.listdir(current_directory) if img.endswith(".png")]
images.sort()

frame = cv2.imread(os.path.join(current_directory, images[0]))
height, width, layers = frame.shape
size = (width, height)
video_name = 'output_video.mp4'
fps = 24
out = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

for image in images:
    img_path = os.path.join(current_directory, image)
    frame = cv2.imread(img_path)
    out.write(frame)

out.release()
print(f"video '{video_name}' created successfully")

