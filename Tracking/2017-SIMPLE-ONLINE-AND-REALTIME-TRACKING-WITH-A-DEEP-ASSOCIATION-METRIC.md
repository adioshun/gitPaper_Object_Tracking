https://arxiv.org/pdf/1703.07402.pdf

https://github.com/nwojke/deep_sort





















---

# [DeepSORT: Deep Learning to Track Custom Objects in a Video](https://nanonets.com/blog/object-tracking-deepsort/)

> blog, by Shishira R Maiya, 2019.07

## 1. Introduction
## 2. Single object tracking
## 3. Multiple object tracking
## 4. Object detection vs Object Tracking
## 5. Challenges
## 6. Traditional Methods

### 6.1 Meanshift

Meanshift or Mode seeking is a popular algorithm, which is mainly used in **clustering** and other related unsupervised problems. 

It is similar to **K-Means**, but replaces the simple centroid technique of calculating the cluster centers with a weighted average that gives importance to points that are closer to the mean. 

The goal of the algorithm is to find all the modes in the given **data distribution**. Also, this algorithm does not require an optimum “K” value like K-Means. More info on this can be found [here](http://www.chioka.in/meanshift-algorithm-for-the-rest-of-us-python/).

> [Meanshift Algorithm for the Rest of Us (Python)](http://www.chioka.in/meanshift-algorithm-for-the-rest-of-us-python/)

Suppose we have a detection for an object in the frame and we extract certain features from the detection (colour, texture, histogram etc). 

By applying the meanshift algorithm, we have a general idea of where the mode of the distribution of features lies in the current state. 

Now when we have the next frame, where this distribution has changed due to the movement of the object in the frame, the meanshift algo looks for the new largest mode and hence tracks the object.

### 6.2 Optical flow

> [Introduction to Motion Estimation with Optical Flow](https://nanonets.com/blog/optical-flow/)

This method differs from the above two methods, as we do not necessarily use features extracted from the detected object. Instead, the object is tracked using the spatio-temporal image **brightness variations** at a pixel level.

Here we focus on obtaining a displacement vector for the object to be tracked across the frames. 

Tracking with optical flow rests on three important **assumptions**:

- Brightness consistency: Brightness around a small region is assumed to remain nearly constant, although the location of the region might change.
- Spatial coherence: Neighboring points in the scene typically belong to the same surface and hence typically have similar motions
- Temporal persistence: Motion of a patch has a gradual change.
- Limited motion: Points do not move very far or in a haphazard manner.

Once these criteria are satisfied, we use something called the **Lucas-Kanade** method to obtain an equation for the velocity of certain points to be tracked (usually these are easily detected features). 

> [Implementing Lucas-Kanade Optical Flow algorithm in Python](https://sandipanweb.wordpress.com/2018/02/25/implementing-lucas-kanade-optical-flow-algorithm-in-python/)

Using the equation and some prediction techniques, a given object can be tracked throughout the video.

For more info on Optical flow, refer [here](https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html).



## 7. Kalman Filters

In almost any engineering problem that involves prediction in a temporal or time series sense, be it computer vision, guidance, navigation or even economics, “Kalman Filter” is the go to algorithm.

### 7.1 Core idea

The core idea of a Kalman filter is to use the available detections and previous predictions to arrive at a best guess of the current state, while keeping the possibility of errors in the process.

> [Tutorial: Kalman Filter with MATLAB example part1](https://www.youtube.com/watch?v=FkCT_LV9Syk): youtube
> [Tutorial: The Kalman Filter](http://web.mit.edu/kirtley/kirtley/binlustuff/literature/control/Kalman%20filter.pdf): pdf, 8page

### 7.2 Why kalman works

Kalman filter works best for **linear systems** with **Gaussian processes** involved. 

In our case the tracks hardly leave the linear realm and also, most processes and even noise in fall into the Gaussian realm. So, the problem is suited for the use of Kalman filters.


## 8. Deep Learning based Approaches

### 8.1 Deep Regression Networks (ECCV, 2016) 

> [GOTURN??](https://davheld.github.io/GOTURN/GOTURN.html)

One of the early methods that used deep learning, for **single object tracking**. 

A model is trained on a dataset consisting of videos with labelled target frames. 

The objective of the model is to simply track a given object from the given image crop.

To achieve this, they use a two-frame CNN architecture which uses both the current and the previous frame to accurately regress on to the object.

![](https://lh3.googleusercontent.com/zSyuq3tQrBHsBQdM8hcpwKzulGQkeqvxb6VphdZIrsIRScE0fRIYc1F4sxZ3ikCeZHw0NqRth84dJjypj9A6WNk4kh0Adny-XPpkp0_8xgVys3KUDHEl1j0Qe_eXDK5JKV2FK8NQ)

As shown in the figure, we take the crop from the previous frame based on the predictions and define a “Search region” in the current frame based on that crop. 

Now the network is trained to regress for the object in this search region

The network architecture is **simple with CNN’s followed by Fully connected layers** that directly give us the bounding box coordinates.

### 8.2 ROLO - Recurrent Yolo (ISCAS 2016)

> https://arxiv.org/pdf/1607.05781v1.pdf

An elegant method to track objects using deep learning. 

Slight modifications to **YOLO detector** and attaching a **recurrent LSTM** unit at the end, helps in tracking objects by capturing the spatio-temporal features.

![](https://lh5.googleusercontent.com/fuXhhAjjpola_hnkmWcxcZ_Q4JUWLYgwZObvMqLuSCwttAusb49t9S4Bbr7CP-flq_01v1M8_l_dJ4fOxSXtmlDeSkYEw2ebLEjZG5tpmTVPX35s00oMlhyBRJcAcG_WEPGJZbPu)

As shown above, the architecture is quite simple. 
- The Detections from YOLO (bounding boxes) are concatenated with the feature vector from a CNN based feature extractor (We can either re-use the YOLO backend or use a specialised feature extractor). 
- Now, this concatenated feature vector, which represents most of the spatial information related to the current object, along with the information on previous state is passed onto the LSTM cell.
- The output of the cell, now accounts for both spatial and temporal information. 

This simple trick of using CNN’s for feature extraction and LSTM’s for bounding box predictions gave high improvements to tracking challenges.

## 9. Deep SORT

The most popular and one of the most widely used, elegant object tracking framework is Deep SORT, an extension to SORT (Simple Real time Tracker). 

We shall go through the concepts introduced in brief and delve into the implementation. 

Let us take a close look at the moving parts in this paper.

### 9.1 The Kalman filter

Our friend from above, Kalman filter is a crucial component in deep SORT. 

Our state contains 8 variables; (u,v,a,h,u’,v’,a’,h’) where 
- (u,v) are centres of the bounding boxes, 
- a is the aspect ratio and 
- h, the height of the image. 
- The other variables are the respective velocities of the variables.

As we discussed previously, the variables have only absolute position and velocity factors, since we are assuming a simple linear velocity model. 

The Kalman filter helps us factor in the noise in detection and uses prior state in predicting a good fit for bounding boxes.

For each detection, we create a “Track”, that has all the necessary state information. 

It also has a parameter to track and delete tracks that had their last successful detection long back, as those objects would have left the scene.

Also, to eliminate duplicate tracks, there is a minimum number of detections threshold for the first few frames.

### 9.2 The assignment problem

Now that we have the new bounding boxes tracked from the Kalman filter, the next problem lies in associating new detections with the new predictions. 

Since they are processed independently, we have no idea on how to associate track\_i with incoming detection\_k.

[중요 구성요소] To solve this, we need 2 things: 
- A distance metric to quantify the association and 
- an efficient algorithm to associate the data.

#### A. The distance metric 

The authors decided to use the **squared Mahalanobis distance** (effective metric when dealing with distributions)  to incorporate the uncertainties from the Kalman filter. 

> 마할라노비스 거리 : '평균과의 거리가 표준편차의 몇 배' 인지 나타내는 값 [[참고]](https://rfriend.tistory.com/199)


Thresholding this distance can give us a very good idea on the actual associations. 

This metric is more accurate than say, euclidean distance as we are effectively measuring distance between 2 distributions (remember that everything is distribution under Kalman!)

#### B. The efficient algorithm 

In this case, we use the standard Hungarian algorithm, which is very effective and a simple data association problem. I won’t delve into it’s details. More on this can be found here.

Lucky for us, it is a single line import on sklearn !





## 10. Code Review
## 11. Conclusion