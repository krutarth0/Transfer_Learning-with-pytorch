import torch
from torchvision import transforms 
from cnn import model_ft
from PIL import Image

real=Image.open('ant.jpg')

preprocess=transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

Image=torch.FloatTensor(preprocess(real).unsqueeze_(0))
model_ft.to('cpu')
output=model_ft(Image)
_,preds = torch.max(output, 1)
print("the chances for it to be an ant is ",output[0][0])
print("the chances for it to be a bee is ", output[0][1])
