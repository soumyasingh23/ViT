#!/usr/bin/env python
# coding: utf-8

# In[26]:


import numpy as np
data = np.load('../dns_large.npy')
data.shape


# In[27]:


data = data[:, ::4, ::4]
data.shape


# In[28]:


data = np.expand_dims(data, axis = 1)
data.shape


# In[29]:


x = data[:-1]
y = data[1:]
x.shape


# In[30]:


y.shape


# In[31]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
import numpy as np
from model import ViT
from tqdm import tqdm


# In[32]:


input_data = torch.tensor(x, dtype=torch.float32)
target_data = torch.tensor(y, dtype=torch.float32)


# In[33]:


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


# In[34]:


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


# In[35]:


criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)


# In[36]:


dataset = TensorDataset(input_data, target_data)
print(dataset[0][0].shape)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)


# In[37]:


num_epochs = 200
y_true = []
y_pred = []

for epoch in range(num_epochs):
    for batch, (x, y) in enumerate(tqdm(dataloader)):
        optimizer.zero_grad()
        x = x.to(device)
        y = y.to(device)

        # print(x.shape)
        # print(y.shape)
        outputs = model(x)
        # print("out",outputs.shape) 

        loss = criterion(outputs, y)
        if epoch == num_epochs-1:
            y_true.append(y)
            y_pred.append(outputs)
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')


# In[38]:


y_true = torch.vstack(y_true)
y_true.shape


# In[39]:


y_pred = torch.vstack(y_pred)
# y_pred.shape


# In[40]:


y_pred.shape

import os
import matplotlib.pyplot as plt

true_dir = 'data/true'
pred_dir = 'data/pred'
os.makedirs(true_dir, exist_ok=True)
os.makedirs(pred_dir, exist_ok=True)

for i in range(y_pred.shape[0]):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Save predicted image
    pred_path = os.path.join(pred_dir, f'predicted_{i}.png')
    plt.imsave(pred_path, (y_pred[i].cpu().detach().numpy()).squeeze(), cmap='gray')

    # Save true image
    true_path = os.path.join(true_dir, f'true_{i}.png')
    plt.imsave(true_path, (y_true[i].cpu().detach().numpy()).squeeze(), cmap='gray')

    plt.close(fig)


# In[42]:


torch.save(model.state_dict(), "model_attn.pt")

