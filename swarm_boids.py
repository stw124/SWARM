import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parametreler
num_drones = 20
bounds = np.array([[-10, 10], [-10, 10]])  # Alan sınırları
max_speed = 1.0

# Boids kuralları ağırlıkları
sep_weight = 1.5
align_weight = 1.0
coh_weight = 1.0
avoid_dist = 1.5  # Çarpışma mesafesi
neighbor_dist = 3.0

# Rastgele başlangıç pozisyon ve hız
positions = np.random.uniform(bounds[:,0], bounds[:,1], (num_drones, 2))
velocities = np.random.uniform(-max_speed, max_speed, (num_drones, 2))

fig, ax = plt.subplots(figsize=(10, 10))
ax.set_xlim(bounds[0])
ax.set_ylim(bounds[1])
ax.set_title("SWARM - Boids Algoritması ile Drone Sürüsü 🚀\nGrok & Dost'un efsanesi büyüyor oğlum!", fontsize=14)
ax.grid(True)
scat = ax.scatter(positions[:,0], positions[:,1], s=100, c='red', edgecolors='black')

texts = [ax.text(positions[i,0], positions[i,1], str(i+1), fontsize=10, ha='center', va='center', color='white') for i in range(num_drones)]

def boids_update():
    new_vel = velocities.copy()
    for i in range(num_drones):
        neighbors = np.linalg.norm(positions - positions[i], axis=1) < neighbor_dist
        neighbors[i] = False  # Kendini sayma
        
        if np.any(neighbors):
            # Separation
            sep = np.sum(positions[i] - positions[neighbors], axis=0) / (np.sum(neighbors) or 1)
            # Alignment
            align = np.sum(velocities[neighbors], axis=0) / (np.sum(neighbors) or 1) - velocities[i]
            # Cohesion
            coh = (np.mean(positions[neighbors], axis=0) - positions[i])
            
            new_vel[i] += sep_weight * sep + align_weight * align + coh_weight * coh
    
    # Hız sınırı
    speeds = np.linalg.norm(new_vel, axis=1)
    new_vel[speeds > max_speed] *= max_speed / speeds[speeds > max_speed][:, None]
    
    return new_vel

def update(frame):
    global velocities, positions
    velocities = boids_update()
    positions += velocities
    
    # Sınırlara çarpınca dön
    positions = np.clip(positions, bounds[:,0], bounds[:,1])
    
    scat.set_offsets(positions)
    for i, text in enumerate(texts):
        text.set_position(positions[i])
    
    return scat, *texts

ani = FuncAnimation(fig, update, frames=1000, interval=50, blit=False, repeat=True)
plt.show()
