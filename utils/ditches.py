from collections import deque
import numpy as np

class DitchMap:
    def __init__(self, max_length = 20, max_depth = 20, max_grad = 5, ditches=10, resolution=512):
        self.num_ditches = ditches
        self.max_length = max_length
        self.max_depth = max_depth
        self.resolution = resolution
        self.height_map = np.zeros((resolution, resolution))
        self.max_grad = max_grad
        self.generate_ditches()

    def generate_ditches(self):
        x_coords = np.random.choice(self.resolution, self.num_ditches, replace=False)
        y_coords = np.random.choice(self.resolution, self.num_ditches, replace=False)
        lengths = np.random.randint(5, self.max_length+1, self.num_ditches)
        depths = np.random.randint(3, self.max_depth+1, self.num_ditches)
        directions = np.random.randint(0, 8, self.num_ditches)
        self.grads = 0.2 + (self.max_grad - 0.2) * np.random.rand(self.num_ditches, 100)

        q = deque()
        visited = set()
        for i in range(self.num_ditches):
            x = x_coords[i]
            y = y_coords[i]
            for _ in range(lengths[i]):
                modifier = np.random.randint(-1, 2)
                direction = (directions[i] + modifier) % 8
                if direction == 0:
                    x += 1
                elif direction == 1:
                    x += 1
                    y += 1
                elif direction == 2:
                    y += 1
                elif direction == 3:
                    x -= 1
                    y += 1
                elif direction == 4:
                    x -= 1
                elif direction == 5:
                    x -= 1
                    y -= 1
                elif direction == 6:
                    y -= 1
                elif direction == 7:
                    x += 1
                    y -= 1    
                if self.in_range(x, y):
                    self.height_map[x, y] = -depths[i]
                    q.append((x, y, i, 0))
                    visited.add((x, y))
                else:
                    break
        self.errode(q, visited)

    def errode(self, q: deque, visited: set):
        while q:
            x, y, d, k = q.popleft()
            new_height = self.height_map[x, y] + self.grads[d, k%100]
            for i in range(-1, 2):
                for j in range(-1, 2):
                    if (x+i,y+j) in visited or (i == 0 and j == 0):
                        continue
                    if self.in_range(x+i, y+j):
                        if self.height_map[x+i, y+j] - self.height_map[x, y] > self.grads[d, k%10]:
                            self.height_map[x+i, y+j] = new_height
                            q.append((x+i, y+j, d, k+1))
                            visited.add((x+i, y+j))

    def in_range(self, x: int, y: int):
        return 0 <= x < self.resolution and 0 <= y < self.resolution