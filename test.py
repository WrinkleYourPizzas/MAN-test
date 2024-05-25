from scipy.io import loadmat, savemat
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import h5py
import numpy as np
import os
import torch

def resize(data):
    data = torch.from_numpy(data).type(torch.float)
    resize = torch.nn.functional.interpolate(data.unsqueeze(0).unsqueeze(0), size=(64, 64, 31), mode='trilinear', align_corners=False)
    resize = resize.squeeze(0).squeeze(0)
    resize = resize.numpy()
    print(resize.shape)

    return resize

# data = loadmat("striped_ip.mat")
striped = resize(loadmat("striped_ip.mat")["striped_ip"])
# original = resize(loadmat("Indian_pines_corrected.mat")["indian_pines_corrected"])
# gt = loadmat("Indian_pines_gt.mat")["indian_pines_gt"]
icvl = h5py.File("sat_0406-1107.mat")
print(icvl.keys())
# print(icvl["bands"].shape)
print(icvl["rad"].shape)
print(striped.shape)

# test = loadmat("sample_hs_im.mat")
# print(test.keys())
# print(test["rgb"].shape)
# print(test["bands"].shape)
# print(test["rad"].shape)

# data = data["striped_ip"]
# print(data.shape)
# print(data.shape[2])

result = resize(np.array(torch.from_numpy(np.array(icvl["rad"])).permute(1,2,0)))
# plt.imshow(result[:, :, 20])
# plt.show()

# for i in range(data.shape[2]):
#     slice = data[:, :, i]
#     slice = ((slice - np.min(slice)) / (np.max(slice) - np.min(slice)) * 255).astype(np.uint8)

#     image = Image.fromarray(slice).convert("L")

#     image.save(os.path.join("indian_pines", "ip_slice_%s.png" %i))

icvl = {}
icvl["input"] = result
icvl["gt"] = result
print(icvl.keys())
savemat("icvl_64_64_31.mat", icvl)

# data["input"] = striped[:, :, :31]
# # data["gt"] = original[:, :, :31]
# data["gt"] = gt
# savemat("alt_indian_pines.mat", data)