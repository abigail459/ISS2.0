import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from matplotlib.patches import Circle
import csv
from collections import defaultdict

### DIRECTORY SETUP
rootdir = "/Users/Abigail/Desktop/Sciences" # js change this
os.chdir(f"{rootdir}/ISS2.0/data")
current_directory = os.getcwd()


### DATA SETUP
data = np.load("data.npz")
n_frames = data["n_frames"]
s_history = data["s_history"]
R = data["R"]
n_falling = int(data["n_falling"])
time_history = data["time_history"]

# SIMULATION PARAMETERS
t_step = 2e-5  # 20 microseconds
simulation_duration = 5.0  # 1 second 
display_fps = 30.0  # 20 fps
numrendered = display_fps*simulation_duration


### SIMULATION
def run_simulation_visually():
    for frame in range(int(numrendered)+1):
        create_frame(s_history[frame], R, frame, n_falling, time_history[frame])

## FUNCTIONS
oscillation_amplitude = 0.003  # 3mm
oscillation_frequency = 2.0  # 2 Hz
omega = 2 * np.pi * oscillation_frequency

def get_box_displacement(time):
    return oscillation_amplitude * np.sin(omega * time)

def get_box_velocity(time):
    return oscillation_amplitude * omega * np.cos(omega * time)


### VISUALISATION 
def create_frame(s_current, R, frame_index, n_falling, time):
    fig = plt.figure(figsize=(6, 6), dpi=80)  # Smaller/lower DPI = faster
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    ax.set_facecolor('#e8e8e8')
    
    # box walls (gray circles)
    for i in range(n_falling, min(len(R), s_current.shape[0])):
        x, y = s_current[i, 0], s_current[i, 1]
        circle = Circle((x, y), R[i], edgecolor='none', facecolor='#202020', alpha=1.0)
        ax.add_patch(circle)
    
    # falling particles (coloured)
    colors = ['#FF3333', '#3333FF', '#33FF33', '#FFAA00', '#FF33FF', '#33FFFF', '#FFFF33']
    
    for i in range(n_falling):
        x, y = s_current[i, 0], s_current[i, 1]
        color = colors[i % len(colors)]
        
        circle = Circle((x, y), R[i], edgecolor='black', facecolor=color, 
                       alpha=0.9, linewidth=2)
        ax.add_patch(circle)
        
        ax.text(x, y, str(i+1), ha='center', va='center', fontsize=9, 
                fontweight='bold', color='white')
    
    # Fixed limits --> full screen
    ax.set_xlim(0, 0.2)
    ax.set_ylim(0, 0.2)
    ax.set_xlabel('x (m)', fontsize=11, fontweight='bold')
    ax.set_ylabel('y (m)', fontsize=11, fontweight='bold')
    
    # Title
    box_disp = get_box_displacement(time)
    ax.set_title(f'Time: {time:.2f}s | Oscillation: {box_disp*1000:.1f}mm', 
                fontsize=12, fontweight='bold', pad=10)
    
    # Light grid
    ax.grid(True, alpha=0.15, linestyle='--', linewidth=0.5) 
    
    fig.set_size_inches(6, 6)
    plt.savefig(f"fig_{frame_index:04d}.png", dpi=80, format='png')
    plt.close(fig) # Close the figure window
    plt.clf() #clears the entire content of the currently active Matplotlib figure,


def create_video(output_name, fps, directory):
    print("Creating video...")
    
    #find all frame files
    images = sorted([img for img in os.listdir(directory) 
                     if img.endswith(".png") and img.startswith("fig_")])
    
    #error msg if legit nothing was found
    if not images:
        print("Error: No frames found.")
        return
    #count
    print(f"  Found {len(images)} frames")
    
    #read first frame to define dimensions of the video
    frame = cv2.imread(os.path.join(directory, images[0]))
    height, width = frame.shape[:2] # --> Returns (height, width, channels), e.g. (480, 480, 3) -> 480×480 RGB image
    # ^^ only take height & weight tho. 
    
    #write video.
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  #fourcc aka "Four Character Code" is a video codec identifier
    out = cv2.VideoWriter(output_name, fourcc, fps, (width, height)) #videowriter object
    
    #write all frames
    for img in images:
        frame = cv2.imread(os.path.join(directory, img))
        if frame is not None and frame.shape[0] == height and frame.shape[1] == width: #ensure dimension matches
            out.write(frame) # appends frame to video file
    
    out.release() # impt!! close video file (finalises encoding)
    print(f"Video: {output_name}")

    

### RUNNING
if __name__ == "__main__": #Only runs if script executed directly (not imported as module)
    print("\nCleaning old frames...")
    for file in os.listdir(current_directory):
        if file.startswith("fig_") and file.endswith(".png"):
            os.remove(os.path.join(current_directory, file)) # Delete file
    print("Cleaned")
    
    os.chdir(f"{rootdir}/ISS2.0/Figures")
    current_directory = os.getcwd()

    run_simulation_visually() # executes the main simulation loop
    create_video('granular_3.mp4', display_fps, current_directory)
    
    print("\n" + "-"*60)
    print("DONE!")
    print(f"Video: granular_3.mp4")
    print(f"Frames: {numrendered}")
    print("-"*60)
