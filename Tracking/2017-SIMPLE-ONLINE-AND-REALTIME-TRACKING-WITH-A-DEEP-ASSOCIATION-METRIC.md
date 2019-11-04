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


## 8. Deep Learning based Approaches
## 9. Deep SORT
## 10. Code Review
## 11. Conclusion