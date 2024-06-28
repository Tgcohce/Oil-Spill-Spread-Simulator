import numpy as np
import matplotlib.pyplot as plt
from parser import *

# seed for reproducibility
np.random.seed(42)

# Constants
alpha = 0.035  # Wind influence coefficient
g = 9.81  # Acceleration, gravity

# data
weather_data = {
    'avg_air_temp': 14.6,
    'sea_surface_temp': 11.9,
    'baro_press': 102.2,
    'peak_wind_speed': 11.3,  # knots
    'avg_wind_speed': 9.5,    # knots
    'wind_direction': 253.4   # degrees
}

wave_data = {
    'max_wave_height': 0.6,    # meters
    'sig_wave_height': 0.4,    # meters
    'wave_direction': 141.4,   # degrees
    'peak_wave_period': 8.3,   # seconds
    'wave_spread_avg': 38      # degrees
}

current_data = {
    'current_velocity': 0.1,   # knots
    'current_direction': 353.4 # degrees
}

oil_properties = {
    'viscosity': 0.89,  # Pa.s
    'density': 900,  # kg/m^3
    'evaporation_rate': 0.001  # Example evaporation rate
}


# tidal cycle and direction
tide_data = [
    {'time': 1.0, 'height': 0.35, 'direction': 315},  # Low tide
    {'time': 7.07, 'height': 1.48, 'direction': 135},  # High tide
    {'time': 12.99, 'height': 0.63, 'direction': 315},  # Low tide
    {'time': 18.57, 'height': 1.69, 'direction': 135},  # High tide
]

# parameters
time_steps = 3000
num_particles = 5000
time_step_hours = 0.16667  # Hours per simulation step (10 minutes)
salinity = 28.49  # PSU
temperature = 12  # °C

# ship paths (list of dictionaries for each ship)
ship_paths = [
    {'path': [(-63.57, 44.65), (-63.55, 44.66)], 'speed': 10, 'width': 0.1},  # Example ship path
    # Add more ships as needed
]

# particle positions
initial_position = [-63.57, 44.65]  # Longitude, Latitude of Halifax Harbor
particles = initial_position + 0.1 * np.random.randn(num_particles, 2)  # Random spread around the initial position
initial_particles = particles.copy()  # Store initial positions for direction arrows

# function to compute Stokes drift velocity
def stokes_drift_velocity(H, T):
    lambda_wavelength = (g * T ** 2) / (2 * np.pi)
    return (np.pi * H ** 2) / (T * lambda_wavelength)

# convert degrees to radians
def deg_to_rad(degrees):
    return np.deg2rad(degrees)

# calculate ship-induced velocity field
def ship_velocity_field(ship_paths, particles, time_step_hours):
    ship_effect = np.zeros_like(particles)
    for ship in ship_paths:
        path = np.array(ship['path'])
        speed = ship['speed']
        width = ship['width']
        for segment_start, segment_end in zip(path[:-1], path[1:]):
            direction = np.array(segment_end) - np.array(segment_start)
            distance = np.linalg.norm(direction)
            direction = direction / distance
            ship_position = np.array(segment_start) + speed * time_step_hours * direction
            distances = np.linalg.norm(particles - ship_position, axis=1)
            influence = np.exp(-distances**2 / (2 * width**2))
            ship_effect += influence[:, None] * speed * direction
    return ship_effect

def update_positions(particles, wind_speed, wind_direction, wave_height, wave_period, wave_direction, current_speed, current_direction, oil_properties, time_step_hours, salinity, temperature, ship_paths, tide_data, current_time):
    wind_effect = alpha * wind_speed * np.array([
        np.cos(deg_to_rad(wind_direction)),
        np.sin(deg_to_rad(wind_direction))
    ]) * time_step_hours

    stokes_drift = stokes_drift_velocity(wave_height, wave_period)
    wave_effect = stokes_drift * np.array([
        np.cos(deg_to_rad(wave_direction)),
        np.sin(deg_to_rad(wave_direction))
    ]) * time_step_hours

    current_effect = current_speed * np.array([
        np.cos(deg_to_rad(current_direction)),
        np.sin(deg_to_rad(current_direction))
    ]) * time_step_hours

    # tidal effect
    tide_effect = np.zeros(2)
    for tide in tide_data:
        if current_time % 24 >= tide['time'] and current_time % 24 < tide['time'] + 6:
            tide_direction = tide['direction']
            tide_amplitude = tide['height']
            tide_effect = tide_amplitude * np.array([
                np.cos(deg_to_rad(tide_direction)),
                np.sin(deg_to_rad(tide_direction))
            ]) * time_step_hours
            break

    # Calculate water density based on salinity and temperature
    rho_0 = 1000  # density of pure water in kg/m^3
    S_0 = 35  # salinity in PSU
    T_0 = 15  # temperature in °C
    beta_s = 0.8  # salinity expansion coefficient in kg/m^3 per PSU
    beta_t = 0.2  # thermal expansion coefficient in kg/m^3 per °C

    rho_water = rho_0 + beta_s * (salinity - S_0) + beta_t * (T_0 - temperature)

    rho_oil = 900  # density of oil in kg/m^3

    # Buoyancy effect
    buoyancy_effect = (rho_water - rho_oil) / rho_water

    ship_effect = ship_velocity_field(ship_paths, particles, time_step_hours)

    total_effect = (wind_effect + wave_effect + current_effect + tide_effect + ship_effect) * buoyancy_effect

    particles += total_effect

    return particles

def apply_evaporation(particles, evaporation_rate, time_step_hours):
    num_evaporated_particles = int(evaporation_rate * len(particles) * time_step_hours)
    if num_evaporated_particles > 0:
        particles = np.delete(particles, np.random.choice(len(particles), num_evaporated_particles, replace=False), axis=0)
    return particles

# Simulation loop
for t in range(time_steps):
    current_time = t * time_step_hours
    particles = update_positions(
        particles, weather_data['avg_wind_speed'], weather_data['wind_direction'], wave_data['sig_wave_height'], wave_data['peak_wave_period'], wave_data['wave_direction'],
        current_data['current_velocity'], current_data['current_direction'], oil_properties, time_step_hours,
        salinity, temperature, ship_paths, tide_data, current_time
    )
    particles = apply_evaporation(particles, oil_properties['evaporation_rate'], time_step_hours)

# Calculate coordinate ranges for the end points of the particles
longitude_range = (particles[:, 0].min(), particles[:, 0].max())
latitude_range = (particles[:, 1].min(), particles[:, 1].max())

# Calculate number of initial and final particles
initial_particles_count = num_particles
final_particles_count = len(particles)

# Calculate time passed
total_time_passed_hours = time_steps * time_step_hours

# summary statistics
print(f"Longitude Range: {longitude_range}")
print(f"Latitude Range: {latitude_range}")
print(f"Initial Particles Count: {initial_particles_count}")
print(f"Final Particles Count: {final_particles_count}")
print(f"Total Time Passed: {total_time_passed_hours} hours")
print(f"Evaporation Rate: {oil_properties['evaporation_rate']}")

# Plot results
fig, ax = plt.subplots(2, 2, figsize=(15, 12))

# Scatter plot of final positions
ax[0, 0].scatter(particles[:, 0], particles[:, 1], alpha=0.5, s=5, label='Final Positions')
ax[0, 0].set_title('Final Positions of Particles')
ax[0, 0].set_xlabel('Longitude')
ax[0, 0].set_ylabel('Latitude')
ax[0, 0].legend()
ax[0, 0].grid(True)

# Quiver plot for direction
ax[0, 1].scatter(particles[:, 0], particles[:, 1], alpha=0.5, s=5, label='Final Positions')
for i in range(0, initial_particles_count, initial_particles_count // 100):  # Reduce number of arrows
    if i < len(particles):  # index is within current particle count
        ax[0, 1].arrow(initial_particles[i, 0], initial_particles[i, 1],
                       particles[i, 0] - initial_particles[i, 0],
                       particles[i, 1] - initial_particles[i, 1],
                       head_width=0.01, head_length=0.02, fc='r', ec='r')
ax[0, 1].set_title('Direction of Movement')
ax[0, 1].set_xlabel('Longitude')
ax[0, 1].set_ylabel('Latitude')
ax[0, 1].legend()
ax[0, 1].grid(True)

# Histogram of distances traveled
distances = np.linalg.norm(particles - initial_particles[:len(particles)], axis=1)
ax[1, 0].hist(distances, bins=50, alpha=0.75, edgecolor='black')
ax[1, 0].set_title('Histogram of Distances Traveled')
ax[1, 0].set_xlabel('Distance')
ax[1, 0].set_ylabel('Frequency')

# Density plot
hb = ax[1, 1].hexbin(particles[:, 0], particles[:, 1], gridsize=50, cmap='viridis', mincnt=1)
cb = fig.colorbar(hb, ax=ax[1, 1])
cb.set_label('Density')
ax[1, 1].set_title('Density Plot of Final Positions')
ax[1, 1].set_xlabel('Longitude')
ax[1, 1].set_ylabel('Latitude')

plt.tight_layout()
plt.savefig('oil_spill_analysis_halifax_harbor.png')
plt.show()
