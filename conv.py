import torch
from torch import nn
import matplotlib.pyplot as plt
from torchvision import transforms, datasets


# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {device} device")


class conv(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
        nn.Linear(28*28, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, 10)
        )
        
               
    def load(self,data_dir):
        '''Loading data into the model'''
        data_dir = r'C:/Users/Taylor/Documents/DataSets/train'
        #resizing the images to be processed
        transform = transforms.Compose(
        [transforms.Resize(255),
        transforms.CenterCrop(224),
        transforms.ToTensor()])
        #apply the transform to the training data
        dataSet = datasets.ImageFolder(data_dir,transform=transform)
        #load this dataset into the right format to be loaded 
        dataLoader = torch.utils.data.DataLoader(dataSet,batch_size=32,shuffle=True)
        #iterate through
        images, labels = next(iter(dataLoader))
        print('Number of samples: ', len(images))
        image = images[2][0]
        #Showing the images, plt.show() is redundant but required
        plt.imshow(image, cmap='gray')
        plt.show()
        print(labels)
        
        

model = conv().to(device)
print(model)
data_dir = r'C:/Users/Taylor/Documents/DataSets/train'
model.load(data_dir)
