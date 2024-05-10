import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def update(frameNum, img, grid):
   
    neighbor_count = (
        np.roll(grid, 1, axis=0) + np.roll(grid, -1, axis=0) +
        np.roll(grid, 1, axis=1) + np.roll(grid, -1, axis=1) +
        np.roll(np.roll(grid, 1, axis=0), 1, axis=1) +
        np.roll(np.roll(grid, 1, axis=0), -1, axis=1) +
        np.roll(np.roll(grid, -1, axis=0), 1, axis=1) +
        np.roll(np.roll(grid, -1, axis=0), -1, axis=1)
    )
    
   
    newGrid = np.where((grid == 1) & ((neighbor_count < 2) | (neighbor_count > 3)), 0, grid)
    newGrid = np.where((grid == 0) & (neighbor_count == 3), 1, newGrid)
    
  
    img.set_data(newGrid)
    grid[:] = newGrid[:]
    return img


N = 300
generations = 100


grid = np.random.randint(0,2,size=(N,N))


fig, ax = plt.subplots()
img = ax.imshow(grid, cmap="gray",interpolation='nearest')
ani = animation.FuncAnimation(fig, update, fargs=(img, grid),
                              frames=generations,
                              interval=50,
                              save_count=50)

plt.show()