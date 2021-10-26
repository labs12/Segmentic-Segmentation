import numpy as np
import torch
import cv2
import glob
import os

gts=glob.glob('./CamVid/testtarget_changed/*.png')
iou_array=glob.glob('./CamVid/Prediction/test/model9/*.png')

# gts=glob.glob('./CamVid/traintarget_changed/*.png')
# iou_array=glob.glob('./CamVid/Prediction/train/*.png')

# gts=glob.glob('./CamVid/valtarget_changed/*.png')
# iou_array=glob.glob('./CamVid/Prediction/val/*.png')

# gts=glob.glob('./CamVid/real_annotation_changed/*.png')
# iou_array=glob.glob('./DJI_Test/*.jpg')

def iou_mean(pred, target, n_classes = 7):

    ious = []
    iousSum = 0
    pred = torch.from_numpy(pred)
    pred = pred.view(-1)

    target = np.array(target)
    target = torch.from_numpy(target)
    target = target.view(-1)


    for cls in range(0, n_classes):
      pred_inds = pred == cls

      target_inds = target == cls

      intersection = (pred_inds[target_inds]).long().sum().data.cpu().item()

      union = pred_inds.long().sum().data.cpu().item() + target_inds.long().sum().data.cpu().item() - intersection

      if union == 0:
        ious.append(float('nan'))
        n_classes=n_classes-1
      else:
        ious.append(float(intersection) / float(max(union, 1)))
        iousSum += float(intersection) / float(max(union, 1))
    return iousSum/n_classes

# for i in range(len(gts)):
#     pred=cv2.imread(iou_array[i])
#     gt=cv2.imread(gts[i])
#     pred=pred.transpose((2,0,1))
#     gt=gt.transpose((2,0,1))
#     pred=pred[0]
#     gt=gt[0]
#     print(iou_mean(pred,gt,7))
change_anno2=[0,1,2,3,4,5,6,7]

cmap2=[[64, 0, 128],
       [128, 64, 128],
       [0, 0, 192],
       [128, 128, 0],
       [64, 64, 0],
       [128, 0, 0],
       [192, 192, 128],
       [0,0,0]]

cmap2=np.array(cmap2)

images_miou=0

size=(360,480)
size2=(2160,3840)

for l in range(len(gts)):
    pred=cv2.imread(iou_array[l])
    gt=cv2.imread(gts[l])

    pred2=np.zeros(size)
    gt2=np.zeros(size)

    for i in range(pred.shape[0]):
        for j in range(pred.shape[1]):
            if((pred[i][j]==cmap2[0]).all()):
                pred2[i][j]=0
            elif ((pred[i][j] == cmap2[1]).all()):
                pred2[i][j] = 1
            elif ((pred[i][j] == cmap2[2]).all()):
                pred2[i][j] = 2
            elif ((pred[i][j] == cmap2[3]).all()):
                pred2[i][j] = 3
            elif ((pred[i][j] == cmap2[4]).all()):
                pred2[i][j] = 4
            elif ((pred[i][j] == cmap2[5]).all()):
                pred2[i][j] = 5
            elif ((pred[i][j] == cmap2[6]).all()):
                pred2[i][j] = 6
            elif ((pred[i][j] == cmap2[7]).all()):
                pred2[i][j] = 7
            if ((gt[i][j] == cmap2[0]).all()):
                gt2[i][j] = 0
            elif ((gt[i][j] == cmap2[1]).all()):
                gt2[i][j] = 1
            elif ((gt[i][j] == cmap2[2]).all()):
                gt2[i][j] = 2
            elif ((gt[i][j] == cmap2[3]).all()):
                gt2[i][j] = 3
            elif ((gt[i][j] == cmap2[4]).all()):
                gt2[i][j] = 4
            elif ((gt[i][j] == cmap2[5]).all()):
                gt2[i][j] = 5
            elif ((gt[i][j] == cmap2[6]).all()):
                gt2[i][j] = 6
            elif ((gt[i][j] == cmap2[7]).all()):
                gt2[i][j] = 7
    print(f'{l} : {iou_mean(pred2,gt2)}')
    images_miou+=iou_mean(pred2,gt2)
print('---')
print(images_miou/len(gts))