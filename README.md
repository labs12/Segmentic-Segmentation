# Segmentic-Segmentation
This file contains pytorch implementation of SegNet and Bayesian Bayesian SegNet
## Motivation
Semantic segmentation is one of the key technology for the self-driving car. Accordingly, with the advancement of CNNs SegNet has recieved high performance in semantic segmentation. In SegNet smooth label prediction can be achieved using forward evaluation only. Also visualization of feature activation effect is easy and it does not require any additional cues like depth, video frames or post processings.
![image](https://user-images.githubusercontent.com/66351304/138885461-fb625533-6d40-40b5-b7ef-711fa76fb12a.png)

## We used SegNet as a base model for Autonomous Vechicle where:
- Input : We used CamVid dataset for training
- Output : Masking objects from input image/video
- Cross-Entropy was used as loss function with SGD optimization
- Changed the total number of classes to 7 
- mIoU was used for performance test
## Test Results
- Got 29.2% of accuracy after 300 epoch 
- Adding droupout(0.5) helped to increase accuracy to 38.3%
### [Test Result Without Droupout]
![NDO_300](https://user-images.githubusercontent.com/66351304/138886989-c3bf378f-9d10-46bd-88fe-c20f82657346.gif)
### [Test Result after adding Droupout]
![DO_300](https://user-images.githubusercontent.com/66351304/138887107-d1ced902-ffd0-49e9-910c-e994ec4b9434.gif)
### [Test Resut after adding Droupout and Ground Truth ]
![DO-GT](https://user-images.githubusercontent.com/66351304/138887163-f3b79951-4bb5-4d38-864b-3ff04353e8a0.gif)
### [Prediction with real datasets(Captured by drone)]
![real-prediction](https://user-images.githubusercontent.com/66351304/138887429-a3ce0188-51c8-4cfb-8710-dc9199e46140.gif)

## References 
- https://ai-pool.com/m/segnet-1567616679
- http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/
- https://www.kaggle.com/carlolepelaars/camvid
- http://mi.eng.cam.ac.uk/~agk34/demo_segnet/demo.php#code
