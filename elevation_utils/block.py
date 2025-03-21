import numpy as np

block = np.pad(np.array([[0.5, 0.5], [0.5, 0.5]]), pad_width=2, mode='constant', constant_values=0)
print(block)

np.save('block.npy', block)