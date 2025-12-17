# ===================== main.py (full) =====================
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from matplotlib.patches import Circle
import csv
from collections import defaultdict

from numba import jit, prange

# mu_something --> micro, 10^-6

### DIRECTORY SETUP
rootdir = "/Users/liliy/Documents/GitHub/"  # js change this
os.chdir(f"{rootdir}/ISS2.0/data")
current_directory = os.getcwd()

### PHYSICAL PARAMETERS
rho_particle = 7630        # kg/m^3, eq. 2.1a
E_tilde = 1e7              # Pa, the effective Young's modulus in Hertzian contact formula.
                           # Lower value -> "softer" particles -> deeper overlaps -> slower/stable simulation.
gamma_n_over_R = 1e5       #Pa*s/m,, more damping --> less bounce,, controls energy loss during collisions.s
w_adhesion = 0.0           # J/m^2,, Surface energy density for JKR cohesive contact.
                           # Set to zero → no sticking forces between particles (non-cohesive granular flow).
Mu_air = 1.82e-5           # Pa*s

# FRICTION PARAMETERS
k_t = 2e7                  # tangential stiffness,, produces tangential (shear) force, F = -k * ξ, N/m. 
mu_t = 0.5                 # # sliding (Coulomb) friction coefficient,, Higher µₜ --> particles grip more, form stable piles.
mu_r = 0.01                # rolling friction coefficient

# gravity
g = np.array([0.0, -9.8, 0.0])

# SIMULATION PARAMETERS
t_step = 2e-5              # 20 microseconds
simulation_duration = 5.0  # s
display_fps = 60
save_every_n_steps = int(1.0 / (display_fps * t_step))

print(f"SETTINGS:")
print(f"  Timestep: {t_step*1e6:.0f} μs")
print(f"  Duration: {simulation_duration}s")
print(f"  Total steps: {int(simulation_duration/t_step):,}")
print(f"  Frames: {int(simulation_duration * display_fps)}")

### OSCILLATION CONFIGURATION
class oscillation_config:
    def __init__(self):
        # make False/True for which way we want it to vibrate. 
        self.enable_x = False
        self.enable_y = True

        self.amplitude_x = 0.0045
        self.amplitude_y = 0.01

        self.frequency_x = 5.0
        self.frequency_y = 8.0

        #ignore for now till needed. this should make it go diagonal to diagonal.
        self.phase_x = 0.0
        self.phase_y = 0.0
        self.omega_x = 2 * np.pi * self.frequency_x
        self.omega_y = 2 * np.pi * self.frequency_y

    def get_displacement(self, time):
        x = self.amplitude_x * np.sin(self.omega_x * time + self.phase_x) if self.enable_x else 0.0
        y = self.amplitude_y * np.sin(self.omega_y * time + self.phase_y) if self.enable_y else 0.0
        return np.array([x, y, 0.0])

    def get_velocity(self, time):
        vx = self.amplitude_x * self.omega_x * np.cos(self.omega_x * time + self.phase_x) if self.enable_x else 0.0
        vy = self.amplitude_y * self.omega_y * np.cos(self.omega_y * time + self.phase_y) if self.enable_y else 0.0
        return np.array([vx, vy, 0.0])

    def print_info(self, g_magnitude):
        print(f"\n OSCILLATION SETTINGS")
        if self.enable_x:
            Gamma_x = self.amplitude_x * self.omega_x**2 / g_magnitude
            print(f"X (Horizontal): f={self.frequency_x:.1f} Hz, A={self.amplitude_x*1000:.1f} mm, Γ={Gamma_x:.3f}")
        else:
            print("X: Disabled")

        if self.enable_y:
            Gamma_y = self.amplitude_y * self.omega_y**2 / g_magnitude
            print(f"Y (Vertical):   f={self.frequency_y:.1f} Hz, A={self.amplitude_y*1000:.1f} mm, Γ={Gamma_y:.3f}")
        else:
            print("Y: Disabled")

oscil_config = oscillation_config()
oscil_config.print_info(abs(g[1]))

### FULL-SCREEN BOX (0-20cm graph)
box_left = 0.01
box_right = 0.19
box_bottom = 0.01
box_top = 0.19

box_width = box_right - box_left
box_height = box_top - box_bottom
print(f"\nBOX: {box_width*100:.0f}cm × {box_height*100:.0f}cm")

wall_spacing = 0.008
box_particle_radius = 0.005

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

### CSV
def READ(file):
    completefile = []
    with open(file, 'r', newline='') as fin:
        reader = csv.reader(fin)
        for row in reader:
            completefile.append([float(x) for x in row])
    return completefile

### FALLING PARTICLES
data = np.load("falling_data.npz")
s_falling = data["s_falling"]
v_falling = data["v_falling"]
R_falling = data["R_falling"]

s = np.vstack([s_falling, box_positions_initial])
v = np.vstack([v_falling, np.zeros((n_box, 3))])
R = np.concatenate([R_falling, box_R])

n_particles = len(R)
n_falling = len(R_falling)

print(f"\nPARTICLES num: {n_falling} falling + {n_box} walls = {n_particles} total")

Vol = (4.0/3.0) * np.pi * R**3
m = Vol * rho_particle
gamma_n = gamma_n_over_R * R

# Tangential history (full matrix, upper triangle used)
tangential_history = np.zeros((n_particles, n_particles, 3), dtype=np.float64)



# spatial hash that's no longer needed... i think. 
'''
class SpatialHash:
    def __init__(self, cell_size):
        self.cell_size = cell_size
        self.grid = {}
    def clear(self):
        self.grid.clear()
    def _hash(self, x, y):
        return (int(x / self.cell_size), int(y / self.cell_size))
    def insert(self, particle_id, x, y):
        cell = self._hash(x, y)
        if cell not in self.grid:
            self.grid[cell] = []
        self.grid[cell].append(particle_id)
    def get_nearby(self, x, y):
        cell = self._hash(x, y)
        nearby = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                neighbor = (cell[0] + dx, cell[1] + dy)
                nearby.extend(self.grid.get(neighbor, []))
        return nearby

max_radius = np.max(R)
spatial_hash = SpatialHash(cell_size=3.0 * max_radius)
print(f"\nSpatial hash cell size: {3.0 * max_radius * 1000:.1f}mm")
'''




# === NUMBA CONTACT KERNEL ===

@jit (nopython=True, parallel=True, fastmath=True) #fastmath and parallel is basically optimisation for ur CPU 
    #nopython difference --> https://stackoverflow.com/questions/71510827/numba-when-to-use-nopython-true 

def get_forces_numba(s, v, R, gamma_n, E_tilde,
                                 n_falling, box_velocity,
                                 k_t, mu_t, mu_r,
                                 t_step, tang_hist):
    """
    Numba --> contact force & fricti on
    - Hertz normal force
    - viscous normal damping
    - Cundall-Strack tangential spring with Coulomb cap
    -  v. simple rolling resistance
    """

    n = len(s)
    F_contact = np.zeros((n, 3)) #2D array with n rows + 3 columns

    for i in prange(n): #parallel range,, outer loop 
        for j in range(i + 1, n): #inner loop 
            dx = s[i, 0] - s[j, 0] 
            dy = s[i, 1] - s[j, 1] 
            dz = s[i, 2] - s[j, 2] 

            dist_sq = dx*dx + dy*dy + dz*dz
            contact_threshold_sq = (R[i] + R[j]) ** 2
            if dist_sq > contact_threshold_sq * 1.01:
                continue
            if dist_sq < 1e-24:
                continue

            dist = np.sqrt(dist_sq)
            h_ij = R[i] + R[j] - dist
            if h_ij <= 0.0:
                # reset tangential history when not in contact
                tang_hist[i, j, 0] = 0.0
                tang_hist[i, j, 1] = 0.0
                tang_hist[i, j, 2] = 0.0
                tang_hist[j, i, 0] = 0.0
                tang_hist[j, i, 1] = 0.0
                tang_hist[j, i, 2] = 0.0
                continue

            inv_dist = 1.0 / dist
            r_hat_x = dx * inv_dist
            r_hat_y = dy * inv_dist
            r_hat_z = dz * inv_dist

            R_eff = (R[i] * R[j]) / (R[i] + R[j])

            # velocities (walls hv box_velocity)
            if i < n_falling:
                v_i_x = v[i, 0]
                v_i_y = v[i, 1]
                v_i_z = v[i, 2]
            else:
                v_i_x = box_velocity[0]
                v_i_y = box_velocity[1]
                v_i_z = box_velocity[2]

            if j < n_falling:
                v_j_x = v[j, 0]
                v_j_y = v[j, 1]
                v_j_z = v[j, 2]
            else:
                v_j_x = box_velocity[0]
                v_j_y = box_velocity[1]
                v_j_z = box_velocity[2]

            v_rel_x = v_i_x - v_j_x
            v_rel_y = v_i_y - v_j_y
            v_rel_z = v_i_z - v_j_z

            v_rel_normal = v_rel_x * r_hat_x + v_rel_y * r_hat_y + v_rel_z * r_hat_z

            root_term = np.sqrt(R_eff * h_ij)
            f_elastic_mag = (2.0/3.0) * E_tilde * np.sqrt(R_eff) * (h_ij ** 1.5)
            f_viscous_mag = -gamma_n[i] * root_term * v_rel_normal

            f_n_mag = f_elastic_mag + f_viscous_mag

            fnx = f_n_mag * r_hat_x
            fny = f_n_mag * r_hat_y
            fnz = f_n_mag * r_hat_z

            # tangential relative velocity
            vtx = v_rel_x - v_rel_normal * r_hat_x
            vty = v_rel_y - v_rel_normal * r_hat_y
            vtz = v_rel_z - v_rel_normal * r_hat_z

            #  tangential spring
            xi_x = tang_hist[i, j, 0] + vtx * t_step
            xi_y = tang_hist[i, j, 1] + vty * t_step
            xi_z = tang_hist[i, j, 2] + vtz * t_step

            ft_x = -k_t * xi_x
            ft_y = -k_t * xi_y
            ft_z = -k_t * xi_z

            ft_mag = np.sqrt(ft_x*ft_x + ft_y*ft_y + ft_z*ft_z)
            fn_norm = fnx * r_hat_x + fny * r_hat_y + fnz * r_hat_z
            if fn_norm < 0.0:
                fn_norm = 0.0
            max_ft = mu_t * fn_norm

            if ft_mag > max_ft and ft_mag > 1e-16:
                scale = max_ft / ft_mag
                ft_x *= scale
                ft_y *= scale
                ft_z *= scale
                xi_x = -ft_x / k_t
                xi_y = -ft_y / k_t
                xi_z = -ft_z / k_t

            # remove normal component from ft
            dot_ft_n = ft_x * r_hat_x + ft_y * r_hat_y + ft_z * r_hat_z
            ft_x -= dot_ft_n * r_hat_x
            ft_y -= dot_ft_n * r_hat_y
            ft_z -= dot_ft_n * r_hat_z

            tang_hist[i, j, 0] = xi_x
            tang_hist[i, j, 1] = xi_y
            tang_hist[i, j, 2] = xi_z

            tang_hist[j, i, 0] = -xi_x
            tang_hist[j, i, 1] = -xi_y
            tang_hist[j, i, 2] = -xi_z

            # rolling resistance ~ mu_r * Fn opposite v_t
            vt_norm = np.sqrt(vtx*vtx + vty*vty + vtz*vtz)
            frx = 0.0
            fry = 0.0
            frz = 0.0
            if vt_norm > 1e-12 and fn_norm > 0.0:
                fr_mag = mu_r * fn_norm
                inv_vt = 1.0 / vt_norm
                frx = -fr_mag * vtx * inv_vt
                fry = -fr_mag * vty * inv_vt
                frz = -fr_mag * vtz * inv_vt

            fx = fnx + ft_x + frx
            fy = fny + ft_y + fry
            fz = fnz + ft_z + frz

            if i < n_falling:
                F_contact[i, 0] += fx
                F_contact[i, 1] += fy
                F_contact[i, 2] += fz
            if j < n_falling:
                F_contact[j, 0] -= fx
                F_contact[j, 1] -= fy
                F_contact[j, 2] -= fz

    return F_contact

def get_forces_optimised(s, v, R, m, gamma_n, E_tilde, n_falling, box_velocity):
    n = len(s)
    F_total = np.zeros((n, 3))

    # Gravity
    F_total[:n_falling, 1] = m[:n_falling] * g[1]

    # Air drag
    F_total[:n_falling] -= 6.0 * np.pi * Mu_air * R[:n_falling, np.newaxis] * v[:n_falling]

    # Contact forces --> Numba
    F_contact = get_forces_numba(
        s, v, R, gamma_n, E_tilde,
        n_falling, box_velocity,
        k_t, mu_t, mu_r,
        t_step, tangential_history
    )
    F_total += F_contact

    return F_total




# === SIMULATION LOOP === 

def run_simulation():
    print("\n" + "-"*60)
    print("GRANULAR CONVECTION SIMULATION")
    print("-"*60)

    s_current = s.copy()
    v_current = v.copy()
    s_history = [s_current.copy()]

    time = 0.0
    time_history = [0.0]
    box_velocity = oscil_config.get_velocity(time)
    F = get_forces_optimised(s_current, v_current, R, m, gamma_n, E_tilde, n_falling, box_velocity)
    a_current = F / m[:, np.newaxis]
    last_saved_time = 0.0

    frame_counter = 1
    n_steps = int(simulation_duration / t_step)

    import time as time_module
    start_time = time_module.time()
    last_print = start_time

    print("\nRunning simulation...")

    for step in range(1, n_steps):
        time = step * t_step

        box_disp = oscil_config.get_displacement(time)
        box_velocity = oscil_config.get_velocity(time)

        current_time = time_module.time()
        if current_time - last_print >= 10.0:
            elapsed = current_time - start_time
            progress = 100 * step / n_steps
            steps_per_sec = step / elapsed
            eta = (n_steps - step) / steps_per_sec
            print(f"  {progress:.1f}% | {elapsed:.0f}s elapsed | {eta:.0f}s remaining")
            last_print = current_time

        s_new = s_current.copy()
        s_new[:n_falling] = s_current[:n_falling] + v_current[:n_falling] * t_step + \
                            0.5 * a_current[:n_falling] * t_step**2

        s_new[n_falling:] = box_positions_initial + box_disp

        F_new = get_forces_optimised(s_new, v_current, R, m, gamma_n, E_tilde, n_falling, box_velocity)
        a_new = F_new / m[:, np.newaxis]

        v_new = v_current.copy()
        v_new[:n_falling] = v_current[:n_falling] + 0.5 * (a_current[:n_falling] + a_new[:n_falling]) * t_step

        s_current = s_new
        v_current = v_new
        a_current = a_new

        if step % save_every_n_steps == 0:
            s_history.append(s_current.copy())
            time_history.append(time)
            frame_counter += 1

        if time - last_saved_time >= 20.0:
            np.savez(
                "generated_values.npz",
                s_history=np.array(s_history),
                n_frames=frame_counter,
                R=R,
                n_falling=n_falling,
                time_history=time_history,
                oscillation_amplitude_x=oscil_config.amplitude_x,
                oscillation_amplitude_y=oscil_config.amplitude_y,
                oscillation_frequency_x=oscil_config.frequency_x,
                oscillation_frequency_y=oscil_config.frequency_y,
                oscillation_phase_x=oscil_config.phase_x,
                oscillation_phase_y=oscil_config.phase_y,
                oscillation_enable_x=oscil_config.enable_x,
                oscillation_enable_y=oscil_config.enable_y
            )
            last_saved_time = time

    np.savez(
        "generated_values.npz",
        s_history=np.array(s_history),
        n_frames=frame_counter,
        R=R,
        n_falling=n_falling,
        time_history=time_history,
        oscillation_amplitude_x=oscil_config.amplitude_x,
        oscillation_amplitude_y=oscil_config.amplitude_y,
        oscillation_frequency_x=oscil_config.frequency_x,
        oscillation_frequency_y=oscil_config.frequency_y,
        oscillation_phase_x=oscil_config.phase_x,
        oscillation_phase_y=oscil_config.phase_y,
        oscillation_enable_x=oscil_config.enable_x,
        oscillation_enable_y=oscil_config.enable_y,
        t_step = t_step,
        simulation_duration = simulation_duration, 
        display_fps = display_fps
    )

    total_time = time_module.time() - start_time
    print(f"\n{'-'*60}")
    print(f"Generated {frame_counter} frames in {total_time}s")
    print(f"{'-'*60}\n")

    return frame_counter, s_history

if __name__ == "__main__":
    n_frames, s_history = run_simulation()
