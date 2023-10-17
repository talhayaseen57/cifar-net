"""
    Evaluate the model
"""
import torch
import torchvision
# import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
# import torch.nn as nn
# import torch.nn.functional as F
import data_setup
import model_setup
from pathlib import Path            


trainloader, testloader = data_setup.getTrainTestLoaders()

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

dataiter = iter(testloader)
images, labels = next(dataiter)


#
def imshow(img , caption=""):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(caption)
    plt.show()
    
    save_path = Path("images")
    save_path.mkdir(parents=True, exist_ok=True)
    # save_path=str(save_path/'predicted.png')
    plt.savefig('images/predicted.png')
    plt.close()
    
model_path = 'models/cifar_net.pth'
net = model_setup.Net()
net.load_state_dict(torch.load(model_path))


outputs = net(images)
_, predicted = torch.max(outputs, 1)

caption = 'Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'  for j in range(4))
imshow(torchvision.utils.make_grid(images), caption)
print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))