import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

os.chdir(r"/Users/Abigail/Desktop/Sciences/ISS/ISS2.0/Graphs/")
current_directory = os.getcwd()

x = [5,7,8,7,2,17,2,9,4,11,12,9,6]
y = [99,86,87,88,111,86,103,87,94,78,77,85,86]




for i in range(24):
    ax = plt.gca()
    ax.set_xlim(0, 20)
    ax.set_ylim(3, 120)
    y_mod = [val - i*2 for val in y]
    plt.scatter(x, y_mod)
    plt.savefig("fig %04d.png" %(i))
    plt.close()
    
    

### export to video ###
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
print(f"Video '{video_name}' created successfully.")

