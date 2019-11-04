# [Integrated detection and tracking for ADAS using deep neural network](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8695310&tag=1)


> 탐색기는 SSD변형 + 추적기, 완전한 end-to-end는 아님것 같음

탐지, 추적, 인지가 가능한 통합 솔루션 제안 `The recent advancements in computer vision technology have ensured that it has an increasingly important position in intelligent transportation.This paper proposes an integral system, including object detection and tracking, to recognize multiple objects in dynamic and complex real-world scenes. `

기반은 SSD방식의 SqueezeNet 활용. 탐지기는 추적기 이후에 수행된다. 추적기는 CNN기반 여러 Feature+ motion information+shape information들을 Fuse한다. `A backbone network of the single shot multi-box detector (SSD) is implemented using an improved SqueezeNet for performance improvement. The object detector is followed by an online object tracker that fuses multiple information features, including the appearance feature extracted by CNNs, motion information, and shape information. `

Both the detector and tracker can well balance accuracy and processing time. The proposed system shows acceptable performance, especially the detector demonstrates the best performance among real-time models on the KITTI test benchmark.

## 1. Introduction


딥러닝 기반 탐지, 추적 기술이 많이 발전 했지만 실시간성이 부족하다. `With the advancements in computer vision technology, especially the deep neural network used for object detection and tracking, it now occupies a prominent position in autonomous vehicles and advanced driver assistance system (ADAS). Hence, it is more crucial than ever to detect and track objects in dynamic and complex real-world scenarios robustly and in real-time. However, the majority of the pre-existing methods are still limited due to slow processing time.`

탐지기에서, CNN기반 탐지기는 2가지로 분류 된다. **two-stage** 와 **one-stage** 전자는 높은 정확도를 보이지만 속도가 느리다. 후자는  **multi-feature map**기법을 적용하여 상대적으로 속도가 빠르다. 하지만 아직 실생활에 적용하기에는 느리다. `Real-time detection and tracking are facing many new challenges, especially with the introduction of convolutional neural network (CNN)-based methods. CNN-based methods have impressive accuracy. However, it is very time-consuming. Fortunately, with the development of optimization of CNN, real-time processing is now possible. A CNN-based detector can be divided into two categories: two-stage and one-stage. RCNN family (RCNN [1], Fast RCNN [2], and Faster RCNN [3]) is a two-stage algorithm that outperforms numerous other detection algorithms in terms of accuracy. However, these methods need more computational cost that leads to the consumption of much more processing time. From the one-stage detector perspective, single shot multi-box detector (SSD) [4] is proposed by considering the multi-feature map, which has a faster processing time than any other methods of the RCNN family. However, even though the one-stage methods can achieve a faster processing time, they still cannot satisfy the requirement of autonomous driving that needs real-time processing in an embedded system such as ADAS.`

추적기에서, **tracking-by-detection**기법이 최근 많이 사용된다. 이 방식은 탐지기의 성능에 크게 좌우 된다. 기존 연구들은 다음과 같다. `On the other hand, in order to cope with sources of uncertainty problem for tracking, tracking-by-detection methods have become increasingly popular with the recent progress in object detection. These methods involve a continuous application of a detector in individual frames and associating detections across frames, which means that the tracking performance is heavily affected by the detection results.`
-  Yu et al. [5] explore high-performance detection and deep learning-based appearance features for online multiple object tracking. The network they use for tracking is similar to GoogLeNet, and the cosine distance is used for measuring the appearance affinity. Compared with the public detector provided by MOT Challenge [6], the false positive (FP) and false negative (FN) decrease by 50% using their own detector. 
- Chen et al. [7] exploit features from multiple convolutional layers, in which the top layer is used to formulate a category-level classifier, and the lower layer is used to identify instances from one category. They also carry out experiments with different detectors including their detector and the public detector provided by MOT Challenge[8]. The results show that using their detectors can enhance tracking accuracy by 14.5%, as compared to the public detector, which indicates that the detector is critical to improving the tracking accuracy.

```
[5] F. Yu, W. Li, Q. Li, Y. Liu, X. Shi, and J. Yan, “POI: multiple object tracking with high performance detection and appearance feature,” Proc. the European Conference on Computer Vision, Oct. 2016, pp:36-42.
[7] L. Chen, H. Ai, C. Shang, Z. Zhuang, and B. Bai, “Online multi-object tracking with convolutional neural networks,” Proc. IEEE International Conference on Image Processing, Sep. 2017, pp:645-649.
```

본 논문의 주 역할은 다음과 같다. `This paper makes the following contributions:`
- 탐지기와 추적기를 통합 하였다. ` (1) We design a system that incorporates deep learning-based object detection and tracking. The system employs a detector to get detection results, and then multiple object tracker is used for associating the detection results with the tracklets.`
- CNN을 이용하여 탐지기와 추적기의 성능을 향상 시켰다. ` (2) CNN is used to improve the accuracy of both the detector and the tracker. `
- 실시간성도 고려 하였다. `(3) Even though the deep neural network is utilized to enhance the accuracy, both detector and tracker can balance processing time and accuracy, even on an embedded board for the autonomous driving system.`

논문의 구성은 아래와 같다.` The remainder of this paper is organized into three sections.`
- Section 2 describes the overall system framework with both detection and tracking algorithms. 
- Section 3 presents a quantitative evaluation for both detector and tracker on KITTI dataset [9] and MOT Challenge, respectively. 
- Section 4 concludes the paper with a summary and outlook.


## 2. Proposed method

### 2.1. Integrated system

제안 방식은 탐지기와 추적기로 구성되어 있다. `With the aim of locating the object precisely and identifying the object accurately, the whole system is made up of two parts: object detector and object tracker. Figure 1 illustrates the main components of the system.`

![](https://i.imgur.com/lsoJgdm.png)


### 2.2 Object detector

탐지기는 기본은 SSD를 사용하였다. 본 논문에서는 SSD의 속도 향상을 초점을 맞추었다. `The detector proposed in this paper is based on SSD. We mainly focus on optimizing the backbone network of SSD because the original VGGNet needs more computational cost.`

제안 방식의 네트워크는 아래와 같다.  제안방식은 SquezeeNet와 비슷하지만 성능향상을 위해 몇가지를 수정 하였다.  `The proposed backbone network architecture is shown in figure 2. It can be seen that the proposed backbone network is very similar to the SquezeeNet, where some changes are made to improve the performance.`

![](https://i.imgur.com/S6fveAB.png)

The changes are introduced as follows. 

#### 가. 변경점 #1 : Batch normalization

Batch normalization (BN) is used to avoid gradient explosion problem. It can be seen in figure 3(a), where the blue arrow is the expected path. Specifically, this figure means that if there is no BN before or after convolution layers, it leads to a gradient explosion even though the weight changes a little. While, the model can be converged following an expected path with BN, as shown in figure 3(b). To this point, the network can converge very fast.

#### 나. 변경점 #2 : parameter sharing technique
 
To further improve the performance of the original SqueezeNet, a parameter sharing technique is utilized in the fire module to solve the vanishing gradient problem in the training stage, as shown in figure 4. The original fire model in the left should include 1×1 and 3×3 Conv. However, the improved model showed in the right removes 1×1 Conv and share the previous 1×1 Conv with 3×3 Conv to solve the vanishing gradient problem and save processing time.

#### 다. 변경점 #3 

Pre-activation allows batch normalization to maximize the normalization effect and improve the processing time by reducing memory access.

#### 라. 변경점 #4 : small-sized maxpooling filter 

To increase representation capability and solve the spatial information loss problem, small-sized (2×2) maxpooling filter is used. When the 2×2 maxpooling filter is used, the small object information can be maintained. In addition, many experiments are done to fix the postion of maxpooling filters to enhance the accuracy.


변경후 비교는 아래와 같다. `The comparison is conducted after these changes are made, which is shown in Table 1. It can be seen that the improved SqueezeNet has the smallest size to satisfy the requirement of an autonomous driving embedded system like ADAS. The improved model is evaluated in section 3.`

![](https://i.imgur.com/4MH27G2.png)


### 2.3. Object tracker

CNN기반 appearance model  + Motion prediction + shape information 을 기반으로 유사도 행렬(affinity matrix)을 생성한다. DA는 이 유사도 행렬을 기반으로 이루어 진다. DA후 correlation filter[10]를 통해 계산된 신뢰값을 기반으로 bject appearance model을 업데이트 한다. `In order to identify each on-road object, an online multiple object tracking algorithm is proposed. It uses CNN for appearance model expression and then combines with motion prediction and shape information to construct an affinity matrix. Data association is implemented based on the affinity matrix. After data association, each object appearance model is updated based on the confidence score calculated by a correlation filter[10].`

> [https://elecs.tistory.com/167](https://elecs.tistory.com/167), [https://elecs.tistory.com/169](https://elecs.tistory.com/169)

추적기의 전체 동작 과정은 아래 그림과 같다. `The overview of the proposed tracker is shown in figure 5. `
![](https://i.imgur.com/ADsD0qj.png)
The steps are as follows: 
- 탐지기를 활용하여 현 프레임에서의 BBox 생성 `firstly, the detected bounding boxes are obtained by the detector at the current frame t;`
- CNN기반  appearance feature추출, 추출된 값을 모션과 외형 정보와 결합 `secondly, each appearance feature of the detected result is extracted by CNN model and combined with motion and shape information.`
- DA 수행하여 추적 결과 획득 ` Thirdly, the data association is executed to get the final tracking result.`
- Finally, each tracklet is updated based on the associated result, and the appearance model for tracklet is selectively updated based on the confidence score.

#### A. Affinity matrix 계산법 
Multi-information ensemble: To construct an affinity matrix for data association, we calculate an affinity between tracklets and detections that includes appearance, shape, and motion information. 

Details of the affinity calculation are as follows:
![](https://i.imgur.com/xtCNnJn.png)

where A_mot, A_shape ,and A_appear indicate motion, shape, and appearance affinity between detection and tracklet, respectively. 

- (X, Y ) is the center coordinate of the bounding box. 
- (W, H) is the width and height of the bounding box, 
- and x is the feature extracted by CNN.

For the **appearance information**, 
- we first extract output features for each detection result using forward propagation and then compute their L2 distance from each tracklet appearance model. 
- We use the ResNet-50 architecture by discarding the last layer and adding two fully connected layers. 
- The first fully connected layer has 1024 units, followed by ReLU. 
- The second one goes down to 128 units which are the final embedding dimension. 
- The network is trained by MARS [11] dataset which is a large person reidentification dataset and VeRi [12] which is a large vehicle reidentification dataset. 

The **shape information** is 
- calculated by their width and height. 

The **motion information** is 
- calculated by width, height, and location predicted by **Kalman filter**.


#### B. Data association

Pairwise association is performed while assigning detections to existing tracklets. When m tracklets and n detections are given, we calculate a score matrix S_{m×n} defined as:

![](https://i.imgur.com/ZcKdjQg.png)

where A(trk_i, det_j ) is calculated by Eq. (4). 

![](https://i.imgur.com/4EZFlqV.png)

We determine the association result between tracklets and detections using **Hungarian algorithm**.

> [Youtube](https://www.youtube.com/watch?v=CldH2y9eMBw), [http://www.hungarianalgorithm.com/hungarianalgorithm.php](http://www.hungarianalgorithm.com/hungarianalgorithm.php)

#### C. Appearance model update

기존 CNN기반 추적기는  **백프로파게이션** 방법을 이용하여 appearance model을 업데이트 한다. `Best CNN model for multiple object tracking requires a well-discriminating internal category. To this end, the majority of CNN-based trackers update appearance model by backpropagation. `

기존 방법은 느리기 때문에 제안 방식에서는  correlation filter로 계산된 **confidence score**를 이용하여 tracklets appearance models을 업데이트 한다. `However, it is very time-consuming. In order to improve the robustness of the tracker and considering processing time, we update tracklets appearance models based on the confidence score which is calculated by the correlation filter`

The **confidence score δ** can be calculated as 

![](https://i.imgur.com/rr92roP.png)

- where F^{−1} denotes the inverse discrete Fourier transform. 
-  `O` is the element-wise product operator, and 
- ˆx means in the frequency domain.

The **appearance model** is updated as follows:

![](https://i.imgur.com/sjJ7BYH.png)

- where η is a hyperparameter and 
- t is the frame index. 
- δ_u is a threshold.

If the confidence score is higher than that threshold, the appearance model is updated based on the association result. Otherwise, we assume it is occluded, and the appearance model is not updated to keep its robustness. In addition, because it works in frequency domain it can save processing time compared with backpropagation.

## 3. Experimental results
