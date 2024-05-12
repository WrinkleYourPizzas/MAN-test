from scipy.io import loadmat
import matplotlib.pyplot as plt
from MAN import MAN, man_l, man_s, man_m
import torch
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((64, 64)),
])

# net = MAN(1, 16, 5, [1,3])
# net = man()
net = man_m()
net.load_state_dict(torch.load("man_m_gaussian.pth")["net"])
net.to("cuda")
net.eval()

data = loadmat("striped_ip.mat")
data = torch.from_numpy(data["striped_ip"]).type(torch.float)
data = data.permute(2, 0, 1)

input_data = transform(data)
data = input_data.reshape(1, 1, input_data.shape[0], input_data.shape[1], input_data.shape[2]).to("cuda")

output = net(data)

output_image = torch.reshape(output[0, 0, 30, :, :], (64,64)).to("cpu")
output_image = output_image.detach().numpy()

f, axarr = plt.subplots(2,)
axarr[0].imshow(input_data[30, :, :], cmap="gray")
axarr[1].imshow(output_image, cmap="gray")
plt.show()
