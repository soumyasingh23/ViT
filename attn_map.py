import numpy as np
data = np.load('2D_turbulence.npy')

data = np.expand_dims(data, axis = 1)


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
import numpy as np
from model import ViT

num_images = 100
# idxs = torch.randint(0, len(data) - 2, (num_images,))
idxs = range(100)


# In[69]:

#
# x = data[idxs]
# y = data[idxs+1]

x = data[:-1]
y = data[1:]


# In[70]:


input_data = torch.tensor(x, dtype=torch.float32)
target_data = torch.tensor(y, dtype=torch.float32)


# In[71]:


dataset = TensorDataset(input_data, target_data)
print(dataset[0][0].shape)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)


# In[72]:


dataset[9][1].shape


# In[73]:


device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(device)

attentions = []


# In[78]:


model = ViT(
    image_size=64,
    patch_size=4,
    output_size=64,
    dim=1024,
    depth=6,
    heads=16,
    mlp_dim=2048,
    channels=1,
    dropout=0.1,
    emb_dropout=0.1
).to(device)
model.load_state_dict(torch.load("model_attn.pt"))
model.eval()
print(model)


# In[80]:


def get_attention(model, input, output):
    attentions.append(output.detach().cpu())
import re
pattern = r"transformer.layers\.\d+\.\d+\.dropout"
# Add forward hook
for name, module in model.named_modules():
    # print(name)
    if re.match(pattern, name):
        print("true")
        module.register_forward_hook(get_attention)


# In[81]:


ims = input_data
ims = ims.to(device)


# In[82]:


# for x, y in dataloader:
#     x = x.to(device)
#     model(
model(ims)
# Handle residuals
attentions = [(torch.eye(att.size(-1)) + att)/(torch.eye(att.size(-1)) + att).sum(dim=-1).unsqueeze(-1) for att in attentions]


# In[85]:


attentions[0].shape


# In[92]:


ims[0].shape


# In[101]:


result = torch.max(attentions[0], dim=1)[0]
# Max or mean both are fine
for i in range(1, 6):
    att = torch.max(attentions[i], dim=1)[0]
    result = torch.matmul(att, result)

masks = result
print(masks.shape)
masks = masks[:, 0, 1:]
print(masks.shape)
import cv2
import os
for i in range(num_images):
    im_input = torch.permute(ims[i].detach().cpu(), (1, 2, 0)).numpy()
    print(im_input.shape)
    # im_input = im_input[:, :, [2, 1, 0]]
    im_input = (im_input+1)/2 * 255
    mask = masks[i].reshape((16, 16)).numpy()
    print(mask.shape)
    
    mask = mask/np.max(mask)
    
    mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_LINEAR)[..., None]
    if not os.path.exists('/output'):
        os.mkdir('output')
    cv2.imwrite('output/input_{}.png'.format(i), im_input)
    cv2.imwrite('output/overlay_{}.png'.format(i), im_input*mask)
    # cv2.imwrite('output/input_{}.png'.format(i), target_data[i])

