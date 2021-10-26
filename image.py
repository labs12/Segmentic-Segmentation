import cv2
import numpy as np
import glob
import os

#train
# images=glob.glob('./CamVid/trainannot/*.png')
# filename=[os.path.basename(x) for x in glob.glob('./CamVid/trainannot/*.png')]

#train_changed
# images=glob.glob('./CamVid/trainannot_changed/*.png')
# filename=[os.path.basename(x) for x in glob.glob('./CamVid/trainannot_changed/*.png')]

#test
# images=glob.glob('./CamVid/testannot/*.png')
# filename=[os.path.basename(x) for x in glob.glob('./CamVid/testannot/*.png')]

#test_changed
# images=glob.glob('./CamVid/testannot_changed/*.png')
# filename=[os.path.basename(x) for x in glob.glob('./CamVid/testannot_changed/*.png')]

#val
# images=glob.glob('./CamVid/valannot/*.png')
# filename=[os.path.basename(x) for x in glob.glob('./CamVid/valannot/*.png')]

#val_changed
# images=glob.glob('./CamVid/valannot_changed/*.png')
# filename=[os.path.basename(x) for x in glob.glob('./CamVid/valannot_changed/*.png')]


#cmap : sky0,building1,column_pole2,road3,sidewalk4,Tree5,SignSymbol6,Fence7,Car8,Pedestrian9,Bicyclist10,Void11
#change_anno : car0,road1,pavement2,tree3,pedestrian4,building5,other6,void7
cmap=[[128, 128, 128],
      [128, 0, 0],
      [192, 192, 128],
      [128, 64, 128],
      [0, 0, 192],
      [128, 128, 0],
      [192, 128, 128],
      [64, 64, 128],
      [64, 0, 128],
      [64, 64, 0],
      [0, 128, 192],
      [0, 0, 0]]

change_anno=[[0,0,0],
             [1,1,1],
             [2,2,2],
             [3,3,3],
             [4,4,4],
             [5,5,5],
             [6,6,6],
             [7,7,7]]
change_anno2=[0,1,2,3,4,5,6,7]

cmap2=[[64, 0, 128],
       [128, 64, 128],
       [0, 0, 192],
       [128, 128, 0],
       [64, 64, 0],
       [128, 0, 0],
       [192, 192, 128],
       [0,0,0]]


#cmap_original_annotation
# for k in range(len(images)):
#     image = cv2.imread(images[k])
#     width = image.shape[0]
#     height = image.shape[1]
#     for i in range(width):
#         for j in range(height):
#             image[i][j] = cmap[image[i][j][0]]
#
#     #train
#     # cv2.imwrite(os.path.join('./CamVid/traintarget', filename[k]), image)
#     #test
#     # cv2.imwrite(os.path.join('./CamVid/testtarget', filename[k]), image)
#     #val
#     cv2.imwrite(os.path.join('./CamVid/valtarget', filename[k]), image)

#change_anno
# for k in range(len(images)):
#     image = cv2.imread(images[k])
#     width = image.shape[0]
#     height = image.shape[1]
#     for i in range(width):
#         for j in range(height):
#             k1=image[i][j][0]
#             if(k1==0 or k1==11):
#                 image[i][j]=change_anno2[7]
#             elif(k1==1):
#                 image[i][j]=change_anno2[5]
#             elif(k1==2 or k1==6 or k1==7 or k1==10):
#                 image[i][j] = change_anno2[6]
#             elif(k1==3):
#                 image[i][j]=change_anno2[1]
#             elif (k1 == 4):
#                 image[i][j] = change_anno2[2]
#             elif (k1 == 5):
#                 image[i][j] = change_anno2[3]
#             elif (k1 == 8):
#                 image[i][j] = change_anno2[0]
#             elif(k1==9):
#                 image[i][j]=change_anno2[4]
#
#     #train
#     # cv2.imwrite(os.path.join('./CamVid/trainannot_changed', filename[k]), image)
#     #test
#     # cv2.imwrite(os.path.join('./CamVid/testannot_changed', filename[k]), image)
#     #val
#     cv2.imwrite(os.path.join('./CamVid/valannot_changed', filename[k]), image)





#cmap2
# for k in range(len(images)):
#     image = cv2.imread(images[k])
#     width = image.shape[0]
#     height = image.shape[1]
#     for i in range(width):
#         for j in range(height):
#             image[i][j] = cmap2[image[i][j][0]]

    #train
    # cv2.imwrite(os.path.join('./CamVid/traintarget_changed', filename[k]), image)
    #test
    # cv2.imwrite(os.path.join('./CamVid/testtarget_changed', filename[k]), image)
    #val
    # cv2.imwrite(os.path.join('./CamVid/valtarget_changed', filename[k]), image)

