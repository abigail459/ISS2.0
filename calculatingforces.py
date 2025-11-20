import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from matplotlib.patches import Circle
import csv
from collections import defaultdict

# mu_something --> micro, 10^-6



### DIRECTORY SETUP
rootdir = "/Users/Abigail/Desktop/Sciences" # js change this
os.chdir(f"{rootdir}/ISS2.0/data")
current_directory = os.getcwd()

### PHYSICAL PARAMETERS
rho_particle = 7630  # kg/m^3, eq 2.1a
E_tilde = 1e7  # Pa, the effective Young’s modulus in Hertzian contact formula.
               # Lower value -> “softer” particles -> deeper overlaps -> slower/stable simulation.
gamma_n_over_R = 3e5  #Pa*s/m,, more damping --> less bounce,, controls energy loss during collisions.s
w_adhesion = 0.0  # J/m^2,, Surface energy density for JKR cohesive contact.
                  # Set to zero → no sticking forces between particles (non-cohesive granular flow).
                  # If sticky or very fine powders -> this would be nonzero.
                  # Eq. 2.4: JKR adhesive force.
Mu_air = 1.82e-5  # Pa·s,, Dynamic viscosity of air. Used in Stokes drag Eq 2.9. Acts like a slow-down force for moving particles.

# FRICTION + TANGENTIAL HISTORY
k_t = 2e7           # produces tangential (shear) force, F = -k * ξ, tangential stiffness (N/m). 
                    # 𝜉 is stored tangential displacement.
                    # ^^ Spring constant for tangential deformation (Cundall–Strack model).
                    # If too high -> simulation may become “stiff” or jitter.
mu_t = 0.5          # sliding (Coulomb) friction coefficient,, Higher µₜ --> particles grip more, form stable piles.
mu_r = 0.01         # rolling friction coefficient (approx.)
tangential_history = {}  # keys=(i,j), value = tangential displacement vector (ξ_ij)

# Helper to unify contact key ordering
def contact_key(i, j):
    return (i, j) if i < j else (j, i) #- store in same format, smaller, larger

# sIMULATION PARAMETERS
t_step = 2e-5  # 20 microseconds 
simulation_duration = 1.0  # 1 second 
display_fps = 30  # 30 fps
save_every_n_steps = int(1.0 / (display_fps * t_step))

print(f"SETTINGS:")
print(f"  Timestep: {t_step*1e6:.0f} μs")
print(f"  Duration: {simulation_duration}s")
print(f"  Total steps: {int(simulation_duration/t_step):,}")
print(f"  Frames: {int(simulation_duration * display_fps)}")

# gravity
g = np.array([0.0, -15.0, 0.0])  # 15 m/s² (realistic settling)





### BOX OSCILLATION
oscillation_amplitude = 0.003  # 3mm
oscillation_frequency = 2.0  # 2 Hz
omega = 2 * np.pi * oscillation_frequency

def get_box_displacement(time):
    return oscillation_amplitude * np.sin(omega * time)

def get_box_velocity(time):
    return oscillation_amplitude * omega * np.cos(omega * time)

### FULL-SCREEN BOX (0-20cm graph)
box_left = 0.01  # 1cm margin
box_right = 0.19  # 19cm
box_bottom = 0.01  # 1cm margin
box_top = 0.19  # 19cm

box_width = box_right - box_left  # 18cm
box_height = box_top - box_bottom  # 18cm

print(f"\nBOX: {box_width*100:.0f}cm × {box_height*100:.0f}cm")

# OPTIMISED WALL DENSITY (balance: no gaps + speed)
wall_spacing = 0.008  # 8mm spacing
box_particle_radius = 0.005  # 5mm radius (slight overlap = sealed)

# BUILD WALLS
bottom_x = np.arange(box_left - box_particle_radius, box_right + box_particle_radius, wall_spacing)
bottom_wall = np.array([[x, box_bottom, 0.0] for x in bottom_x])

top_x = np.arange(box_left - box_particle_radius, box_right + box_particle_radius, wall_spacing)
top_wall = np.array([[x, box_top, 0.0] for x in top_x])

left_y = np.arange(box_bottom, box_top + wall_spacing, wall_spacing)
left_wall = np.array([[box_left, y, 0.0] for y in left_y])

right_y = np.arange(box_bottom, box_top + wall_spacing, wall_spacing)
right_wall = np.array([[box_right, y, 0.0] for y in right_y])

box_positions_initial = np.vstack([bottom_wall, top_wall, left_wall, right_wall])
n_box = len(box_positions_initial)
box_R = np.full(n_box, box_particle_radius)

print(f"Wall particles: {n_box}")
print(f"  Spacing: {wall_spacing*1000:.0f}mm")
print(f"  Radius: {box_particle_radius*1000:.0f}mm")






### FALLING PARTICLES
s_falling = np.array([
    [0.05, 0.16, 0.0],
    [0.08, 0.16, 0.0],
    [0.10, 0.16, 0.0],
    [0.12, 0.16, 0.0],
    [0.15, 0.16, 0.0],
    [0.07, 0.13, 0.0],
    [0.13, 0.13, 0.0]
])

v_falling = np.array([
    [0.005, 0.0, 0.0],
    [-0.005, 0.0, 0.0],
    [0.003, 0.0, 0.0],
    [-0.003, 0.0, 0.0],
    [0.002, 0.0, 0.0],
    [0.004, 0.0, 0.0],
    [-0.004, 0.0, 0.0]
])

R_falling = np.array([0.004, 0.0042, 0.0038, 0.0045, 0.004, 0.0043, 0.0041])

s = np.vstack([s_falling, box_positions_initial])
v = np.vstack([v_falling, np.zeros((n_box, 3))])
R = np.concatenate([R_falling, box_R])

n_particles = len(R)
n_falling = len(R_falling)

print(f"\nPARTICLES num: {n_falling} falling + {n_box} walls = {n_particles} total")


Vol = (4.0/3.0) * np.pi * R**3
m = Vol * rho_particle
gamma_n = gamma_n_over_R * R




# ----------------------------- # 
### SPATIAL HASHING --> what its about: http://mauveweb.co.uk/posts/2011/05/introduction-to-spatial-hashes.html 
# .self -> https://www.geeksforgeeks.org/python/self-in-python-class/
# O(N) instaed of O(N^2)

class SpatialHash: #class so it can store particles in grid “cells” so collisions can be detected ffast,,
    def __init__(self, cell_size):
        self.cell_size = cell_size  #the width/height of each grid cell.
        self.grid = {}  #dictionary mapping integer cell coordinates -> lists of particle IDs.
        #instead of checking every pair of particles, only check pairs in the same or neighbouring cells -> reducing work.
    
    def clear(self):  #Called every timestep 
        self.grid.clear() #wipes the dictionary --> rebuild it with updated positions.
    
    def _hash(self, x, y):
        return (int(x / self.cell_size), int(y / self.cell_size)) # Returns a tuple: (grid_x, grid_y).
    
    def insert(self, particle_id, x, y):
        cell = self._hash(x, y)
        if cell not in self.grid:
            self.grid[cell] = []
        self.grid[cell].append(particle_id)  # Each cell contains the IDs of all particles currently inside that region.
    
    def get_nearby(self, x, y):
        # Only check 9 neighboring cells instead of all particles --> Looks at the 3×3 block around one cell: 
        # 9 cells --> Because in DEM, a particle can only touch others within a radius slightly larger than its diameter. 
        ###     If cell size ≥ ~2R, then anything outside the 3×3 region is guaranteed too far away to collide -> avoids scanning all N particles.
    
        cell = self._hash(x, y)
        nearby = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                neighbor = (cell[0] + dx, cell[1] + dy)
                nearby.extend(self.grid.get(neighbor, []))
        return nearby

max_radius = np.max(R)  # largest particle radius in the system.
spatial_hash = SpatialHash(cell_size = 3.0 * max_radius)  
# ensures that:  - The grid cell size must be big enough that a whole particle fits inside one cell. 
#                   bcs if the cell is too small, the particle gets “split” across cells & lose collision accuracy.
##               - neighbour search is efficient 
##               - no risk of missing contactss

print(f"\n Spatial hash cell size: {3.0 * max_radius * 1000:.1f}mm")





### FORCE CALCULATION
def get_forces_optimised(s, v, R, m, gamma_n, E_tilde, n_falling, box_velocity, spatial_hash):
    """
    Returns F_total (n x 3) array of forces for all particles.
    This function now includes:
      - Hertz elastic normal
      - viscous normal damping
      - Stokes drag (air)
      - Cundall-Strack tangential spring + Coulomb cap
      - approximate rolling resistance (converted to tangential force)
    """

    n = len(s)
    F_total = np.zeros((n, 3)) #empty force array 
    
    # Gravity (vectorised for speed) - only applied to falling particles
    F_total[:n_falling, 1] = m[:n_falling] * g[1]
    
    # Air drag (vectorised)
    F_total[:n_falling] -= 6.0 * np.pi * Mu_air * R[:n_falling, np.newaxis] * v[:n_falling]
    
    # Build spatial hash
    spatial_hash.clear()
    for i in range(n):
        spatial_hash.insert(i, s[i, 0], s[i, 1])
    
    # Contact forces - ONLY CHECK NEARBY PARTICLES!
    for i in range(n):
        nearby = spatial_hash.get_nearby(s[i, 0], s[i, 1])
        
        for j in nearby:
            if i >= j:  # Skip duplicates
                continue
            
            r_ij = s[i] - s[j]
            dist_sq = np.dot(r_ij, r_ij)  # Faster than norm
            
            if dist_sq < 1e-24:
                continue
            
            dist = np.sqrt(dist_sq)
            h_ij = R[i] + R[j] - dist
            
            if h_ij > 0:  # Contact!
                # normal unit vector from j to i
                r_hat = r_ij / dist
                R_eff = (R[i] * R[j]) / (R[i] + R[j])
                
                # --- Elastic (Hertz)
                f_elastic = (2.0/3.0) * E_tilde * np.sqrt(R_eff) * (h_ij ** 1.5) * r_hat
                
                # --- Viscous normal damping
                # velocity of particle or wall: walls have velocity along x = box_velocity
                v_i = v[i] if i < n_falling else np.array([box_velocity, 0.0, 0.0])
                v_j = v[j] if j < n_falling else np.array([box_velocity, 0.0, 0.0])
                v_rel = v_i - v_j
                v_rel_normal = np.dot(v_rel, r_hat)
                f_viscous = -gamma_n[i] * np.sqrt(R_eff * h_ij) * v_rel_normal * r_hat
                
                # Combined normal contact vector
                f_normal = f_elastic + f_viscous
                
                # --- Tangential (Cundall-Strack)
                # tangential relative velocity
                v_t = v_rel - v_rel_normal * r_hat
                
                key = contact_key(i, j)
                if key not in tangential_history:
                    tangential_history[key] = np.zeros(3)
                
                # Integrate tangential spring: xi <- xi + v_t * dt
                xi_t = tangential_history[key] + v_t * t_step
                tangential_history[key] = xi_t.copy()
                
                # Raw tangential force from spring
                f_t_raw = - k_t * xi_t
                
                # Normal force magnitude for Coulomb limit
                # Use the normal component (positive compressive); clamp to small positive
                f_n_mag = np.dot(f_normal, r_hat)
                if f_n_mag < 0:
                    # if tensile (unlikely without cohesion), use small positive to avoid zero division
                    f_n_mag = 1e-12
                else:
                    f_n_mag = max(f_n_mag, 1e-12)
                
                f_t_mag = np.linalg.norm(f_t_raw)
                if f_t_mag > mu_t * f_n_mag:
                    # sliding: scale tangential force to Coulomb limit
                    f_t = f_t_raw * (mu_t * f_n_mag / f_t_mag)
                    # update stored xi consistently: xi = -f_t / k_t
                    tangential_history[key] = -f_t / k_t
                else:
                    # sticking: keep spring force
                    f_t = f_t_raw
                
                # numerical cleanup: remove any tiny normal component in f_t
                f_t = f_t - np.dot(f_t, r_hat) * r_hat
                
                # --- Rolling resistance (approximate)
                # We don't track angular velocity of spheres; approximate rolling resistance
                # as a tangential force opposing v_t with magnitude μ_r * F_n
                v_t_norm = np.linalg.norm(v_t)
                if v_t_norm > 1e-12:
                    f_roll_mag = mu_r * f_n_mag
                    f_roll_dir = -v_t / v_t_norm
                    f_roll = f_roll_mag * f_roll_dir
                else:
                    f_roll = np.zeros(3)
                
                # Total contact force = normal + tangential + rolling
                f_contact = f_normal + f_t + f_roll
                
                # Only falling particles respond
                if i < n_falling:
                    F_total[i] += f_contact
                if j < n_falling:
                    F_total[j] -= f_contact
            else:
                # contact ended: remove tangential history to avoid memory growth and reset spring
                key = contact_key(i, j)
                if key in tangential_history:
                    del tangential_history[key]
    
    return F_total




### SIMULATION LOOP
def run_simulation():
    print("\n" + "-"*60)
    print("GRANULAR CONVECTION SIMULATION")
    print("-"*60)
    
    s_current = s.copy()
    v_current = v.copy()
    s_history = [s_current.copy()]
    
    time = 0.0
    time_history = [0.0]
    box_velocity = get_box_velocity(time)
    F = get_forces_optimised(s_current, v_current, R, m, gamma_n, E_tilde, n_falling, box_velocity, spatial_hash)
    a_current = F / m[:, np.newaxis]
    
    frame_counter = 1
    
    n_steps = int(simulation_duration / t_step) #iterations
    
    import time as time_module  # avoid meddling w our actual values
    start_time = time_module.time() # actual seconds elapsed on computer
    last_print = start_time # records when simulation started
    
    print("\nRunning simulation...")
    
    for step in range(1, n_steps):
        time = step * t_step #time: Simulation time (physics time, not wall-clock time)
        
        box_disp = get_box_displacement(time) 
        box_velocity = get_box_velocity(time)
        
        #progress reporting!! 
        current_time = time_module.time()
        if current_time - last_print >= 10.0:
            elapsed = current_time - start_time
            progress = 100 * step / n_steps
            steps_per_sec = step / elapsed
            eta = (n_steps - step) / steps_per_sec
            print(f"  {progress:.1f}% | {elapsed:.0f}s elapsed | {eta:.0f}s remaining")
            last_print = current_time
        
        # Verlet integration ! 
        ## a numerical method used in physics simulations to calculate the position of objects over time 
        ## by using their current and previous positions, without explicitly needing velocity. 

        s_new = s_current.copy() # current position of particle
        s_new[:n_falling] = s_current[:n_falling] + v_current[:n_falling] * t_step + \
                            0.5 * a_current[:n_falling] * t_step**2  #Linear motion 
        # ^^ Update positions!!
        

        # Move box walls
        s_new[n_falling:] = box_positions_initial + np.array([box_disp, 0.0, 0.0])
        
        # Forces at new positions
        F_new = get_forces_optimised(s_new, v_current, R, m, gamma_n, E_tilde, n_falling, box_velocity, spatial_hash)
            # ^^ update v coz positions are at time t+Δt, but velocities still at time t
            # implied to update v after s in eq 2.2
        a_new = F_new / m[:, np.newaxis]
        
        # Update velocities
        v_new = v_current.copy()
        v_new[:n_falling] = v_current[:n_falling] + 0.5 * (a_current[:n_falling] + a_new[:n_falling]) * t_step
        
        s_current = s_new
        v_current = v_new
        a_current = a_new
        
        # Save frame
        if step % save_every_n_steps == 0:
            s_history.append(s_current.copy())
            time_history.append(time)
            frame_counter += 1

        # os.chdir(f"{rootdir}/ISS2.0/data")
        np.savez(
            "data.npz", 
            s_history = np.array(s_history),
            n_frames = frame_counter,
            R = R,
            n_falling = n_falling,
            time_history = time_history
        )
    
    total_time = time_module.time() - start_time
    print(f"\n{'-'*60}")
    print(f"Generated {frame_counter} frames")
    print(f"{'-'*60}\n")
    
    return frame_counter, s_history


### RUNNING
if __name__ == "__main__": #Only runs if script executed directly (not imported as module)
    n_frames, s_history = run_simulation() #executes the main simulation loop
 
