from scipy.io import loadmat
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os
from MAN import MAN_T
import torch

data = loadmat("striped_ip.mat")
print(data.keys())

data = data["striped_ip"]
print(data.shape)
print(data.shape[2])

plt.imshow(data[:, :, 102], cmap="gray")
plt.show()

# for i in range(data.shape[2]):
#     slice = data[:, :, i]
#     slice = ((slice - np.min(slice)) / (np.max(slice) - np.min(slice)) * 255).astype(np.uint8)

#     image = Image.fromarray(slice).convert("L")

#     image.save(os.path.join("indian_pines", "ip_slice_%s.png" %i))