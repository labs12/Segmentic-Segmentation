import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import model
import old_model
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

images=glob.glob('./CamVid/test/*.png')
filename=[os.path.basename(x) for x in glob.glob('./CamVid/test/*.png')]
# targets=glob.glob('./CamVid/testannot/*.png')
targets=glob.glob('./CamVid/testannot_changed/*.png')

# images=glob.glob('./CamVid/train/*.png')
# filename=[os.path.basename(x) for x in glob.glob('./CamVid/train/*.png')]
# # targets=glob.glob('./CamVid/trainannot/*.png')
# targets=glob.glob('./CamVid/trainannot_changed/*.png')

# images=glob.glob('./CamVid/val/*.png')
# filename=[os.path.basename(x) for x in glob.glob('./CamVid/val/*.png')]
# # targets=glob.glob('./CamVid/valannot/*.png')
# targets=glob.glob('./CamVid/valannot_changed/*.png')

# images=glob.glob('./DJI_0027/*.jpg')
# filename=[os.path.basename(x) for x in glob.glob('./DJI_0027/*.jpg')]

if(torch.cuda.is_available()):
    seg=old_model.Segnet().cuda()
else:
    seg=old_model.Segnet()

if os.path.isfile('model9.pth'):
    seg.load_state_dict(torch.load('model9.pth'))

size=(480,360)
size2=(3840,2160)
size3=(224,224)
# cmap=[[128, 128, 128],
#       [128, 0, 0],
#       [192, 192, 128],
#       [128, 64, 128],
#       [0, 0, 192],
#       [128, 128, 0],
#       [192, 128, 128],
#       [64, 64, 128],
#       [64, 0, 128],
#       [64, 64, 0],
#       [0, 128, 192],
#       [0, 0, 0]]

cmap2=[[64, 0, 128],
       [128, 64, 128],
       [0, 0, 192],
       [128, 128, 0],
       [64, 64, 0],
       [128, 0, 0],
       [192, 192, 128],
       [0,0,0]]

# if(torch.cuda.is_available()):
#     criterion=nn.CrossEntropyLoss().cuda()
# else:
#     criterion = nn.CrossEntropyLoss()


for l in range(len(images)):
    image1 = cv2.imread(images[l])
    image1 = cv2.resize(image1, (224, 224))
    input = image1.transpose((2, 0, 1))

    if (torch.cuda.is_available()):
        input = torch.tensor(input, dtype=torch.float32).cuda()
    else:
        input = torch.tensor(input, dtype=torch.float32)

    # target = cv2.imread(targets[l], cv2.COLOR_BGR2GRAY)
    # target = cv2.resize(target, (224, 224))
    #
    # #changed
    # target = target.transpose((2, 0, 1))
    # target=target[0]
    #
    # if (torch.cuda.is_available()):
    #     target = torch.tensor(target, dtype=torch.int64).cuda()
    # else:
    #     target = torch.tensor(target, dtype=torch.int64)
    # target = torch.unsqueeze(target, 0)


    input = torch.unsqueeze(input, dim=0)
    output, sm = seg(input)
    # loss = criterion(output, target)
    # print(loss)
    sm=torch.squeeze(sm)
    sm=sm.argmax(axis=0)
    sm=sm.cpu().detach().numpy()
    sm=np.array(sm,dtype=object)

    new=np.zeros((224,224,3),dtype='uint8')
    width = sm.shape[0]
    height = sm.shape[1]
    for i in range(width):
        for j in range(height):
            # new[i][j] = cmap[sm[i][j]]
            new[i][j] = cmap2[sm[i][j]]
            iou_array= sm[i][j]
    new=cv2.resize(new,size,interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(os.path.join('./CamVid/Prediction/test',filename[l]),new)

    # cv2.imwrite(os.path.join('./CamVid/Prediction/train', filename[l]), new)

    # cv2.imwrite(os.path.join('./CamVid/Prediction/val', filename[l]), new)

    # cv2.imwrite(os.path.join('./DJI_Test',filename[l]),new)