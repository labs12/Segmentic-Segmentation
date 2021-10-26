import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data
import model
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob


images=glob.glob('./CamVid/train/*.png')
images2=glob.glob('./CamVid/val/*.png')
filename=[os.path.basename(x) for x in glob.glob('./CamVid/train/*.png')]
filename=[os.path.basename(x) for x in glob.glob('./CamVid/val/*.png')]

# targets=glob.glob('./CamVid/trainannot/*.png')
targets=glob.glob('./CamVid/trainannot_changed/*.png')
targets2=glob.glob('./CamVid/valannot_changed/*.png')

if(torch.cuda.is_available()):
    seg=model.Segnet().cuda()
else:
    seg=model.Segnet()

if os.path.isfile('model.pth'):
    seg.load_state_dict(torch.load('model.pth'))


epoch=100
optimizer=optim.SGD(seg.parameters(),lr=0.1,momentum=0.9)
prev_loss=float('inf')
if(torch.cuda.is_available()):
    criterion=nn.CrossEntropyLoss().cuda()
else:
    criterion = nn.CrossEntropyLoss()

#original annotation
# for l in range(len(images)):
#     image1=cv2.imread(images[l])
#     image1=cv2.resize(image1,(224,224))
#     images[l]=image1.transpose((2,1,0))
#     target = cv2.imread(targets[l], cv2.COLOR_BGR2GRAY)
#     target = cv2.resize(target, (224, 224))

#changed annotation
for l in range(len(images)):
    image1=cv2.imread(images[l])
    image1=cv2.resize(image1,(224,224))
    images[l]=image1.transpose((2,0,1))

    target = cv2.imread(targets[l], cv2.COLOR_BGR2GRAY)
    target = cv2.resize(target, (224, 224))
    target= target.transpose((2, 0, 1))
    targets[l] =target[0]
for m in range(len(images2)):
    image2 = cv2.imread(images2[m])
    image2 = cv2.resize(image2, (224, 224))
    images2[m] = image2.transpose((2, 0, 1))

    target2 = cv2.imread(targets2[m], cv2.COLOR_BGR2GRAY)
    target2 = cv2.resize(target2, (224, 224))
    target2 = target2.transpose((2, 0, 1))
    targets2[m] = target2[0]

images=torch.tensor(images)
targets=torch.tensor(targets)
dt=data.TensorDataset(images,targets)
train_loader=data.DataLoader(dt, batch_size=12, shuffle=True)

images2=torch.tensor(images2)
targets2=torch.tensor(targets2)
vt=data.TensorDataset(images2,targets2)
val_loader=data.DataLoader(dt,batch_size=12,shuffle=False)

for n in range(epoch):
    train_loss_sum=0.0
    val_loss_sum=0.0
    for input,target in train_loader:

        if (torch.cuda.is_available()):
            input=torch.tensor(input,dtype=torch.float32).cuda()
        else:
            input = torch.tensor(input, dtype=torch.float32)
        input = torch.autograd.Variable(input)

        if (torch.cuda.is_available()):
            target = torch.tensor(target, dtype=torch.int64).cuda()
        else:
            target = torch.tensor(target, dtype=torch.int64)

        # target = torch.unsqueeze(target, 0)
        target = torch.autograd.Variable(target)

        optimizer.zero_grad()

        # input=torch.unsqueeze(input,dim=0)
        output,sm=seg(input)

        loss=criterion(output,target)

        loss.backward()
        optimizer.step()
        train_loss_sum+=loss

    with torch.no_grad():
        for input,target in val_loader:
            if (torch.cuda.is_available()):
                input = torch.tensor(input, dtype=torch.float32).cuda()
            else:
                input = torch.tensor(input, dtype=torch.float32)
            input = torch.autograd.Variable(input)

            if (torch.cuda.is_available()):
                target = torch.tensor(target, dtype=torch.int64).cuda()
            else:
                target = torch.tensor(target, dtype=torch.int64)
            target = torch.autograd.Variable(target)
            output, sm = seg(input)
            loss2 = criterion(output, target)
            val_loss_sum+=loss2

    # loss_sum=loss_sum/len(images)
    print(f'epoch : {n+1} train_loss : {train_loss_sum} val_loss : {val_loss_sum}')
    if(val_loss_sum<prev_loss):
        prev_loss=val_loss_sum
        torch.save(seg.state_dict(),'model.pth')