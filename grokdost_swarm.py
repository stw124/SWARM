


```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parametreler
num_drones = 8                  # Drone sayısı (değiştirip dene)
radius = 3.0                     # Formation yarıçapı
formation = "circle"             # "circle" veya "line" yapabilirsin sonra

# Başlangıç pozisyonları
def init_positions():
    angles = np.linspace(0, 2*np.pi, num_drones, endpoint=False)
    if formation == "circle":
        x = radius * np.cos(angles)
        y = radius * np.sin(angles)
    else:
        x = np.linspace(-radius*2, radius*2, num_drones)
        y = np.zeros(num_drones)
    return np.column_stack((x, y))

positions = init_positions()

# Animasyon setup
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_xlim(-radius-2, radius+2)
ax.set_ylim(-radius-2, radius+2)
ax.set_aspect('equal')
ax.set_title("Swarm Simülation 🚀", fontsize=14)
ax.grid(True)
scat = ax.scatter(positions[:,0], positions[:,1], s=150, c='red', edgecolors='black', label='Drone')

# Her drone'a numara yaz
texts = [ax.text(positions[i,0], positions[i,1], str(i+1), fontsize=12, ha='center', va='center', color='white') for i in range(num_drones)]

def update(frame):
    # Yavaş yavaş daire etrafında döndür
    theta = frame * 0.02
    rot_matrix = np.array([[np.cos(theta), -np.sin(theta)],
                           [np.sin(theta), np.cos(theta)]])
    new_pos = positions @ rot_matrix.T
    scat.set_offsets(new_pos)
    
    # Numara etiketlerini güncelle
    for i, text in enumerate(texts):
        text.set_position((new_pos[i,0], new_pos[i,1]))
    
    return scat, *texts

ani = FuncAnimation(fig, update, frames=360, interval=50, blit=False, repeat=True)
ax.legend()
plt.show()
