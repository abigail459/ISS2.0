#plottinggraphs
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
rootdir = "/Users/liliy/Documents/GitHub/" # js change this
os.chdir(f"{rootdir}/ISS2.0/data")
current_directory = os.getcwd()


### DATA SETUP
data = np.load("generated_values.npz")

n_frames = data["n_frames"]
s_history = data["s_history"]
R = data["R"]
n_falling = int(data["n_falling"])
time_history = data["time_history"]

osc_enable_x = bool(data.get("oscillation_enable_x", False))
osc_enable_y = bool(data.get("oscillation_enable_y", True))

amp_x = float(data.get("oscillation_amplitude_x", 0.003))
amp_y = float(data.get("oscillation_amplitude_y", 0.003))
freq_x = float(data.get("oscillation_frequency_x", 2.0))
freq_y = float(data.get("oscillation_frequency_y", 2.0))


# SIMULATION PARAMETERS
t_step = float(data.get("t_step"))  # timestep
simulation_duration = float(data.get("simulation_duration"))  
display_fps = float(data.get("display_fps")) #fps
numrendered = display_fps*simulation_duration


### SIMULATION
def run_simulation_visually():
    print(f"Frames: {numrendered}")
    for frame in range(int(numrendered)+1):
        create_frame(s_history[frame], R, frame, n_falling, time_history[frame])

## FUNCTIONS
if osc_enable_y and not osc_enable_x:
    oscillation_amplitude = amp_y
    oscillation_frequency = freq_y
elif osc_enable_x and not osc_enable_y:
    oscillation_amplitude = amp_x
    oscillation_frequency = freq_x
else:
    # if both enabled then show magnitude of vector (?) right....
    oscillation_amplitude = max(amp_x, amp_y)
    oscillation_frequency = max(freq_x, freq_y)

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
    for i in range(n_falling):
        x, y = s_current[i, 0], s_current[i, 1]
        highcutoff = max(R)-((max(R)-min(R))/3) # relative to size sample
        lowcutoff = min(R)+((max(R)-min(R))/3)
        if R[i] > highcutoff:
            color = "#004CFF"
        elif R[i] < lowcutoff:
           color = "#FF0000"
        else:
            color = "#33FF33"
        
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
    create_video('output.mp4', display_fps, current_directory)
    
    print("\n" + "-"*60)
    print("DONE!")
    print(f"Video: 'output.mp4'")
    print("-"*60)
