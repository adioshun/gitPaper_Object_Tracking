|논문명 |Online learning for human classification in 3D LiDAR-based tracking |
| --- | --- |
| 저자\(소속\) | \(\) |
| 학회/년도 | 2017, [논문](http://webpages.lincoln.ac.uk/nbellotto/doc/Yan2017.pdf) |
| Citation ID / 키워드 | |
| 데이터셋(센서)/모델 | |
| 관련연구||
| 참고 | [홈페이지](http://lcas.lincoln.ac.uk/wp/), [저자](https://yzrobot.github.io/), [Youtube](https://www.youtube.com/watch?v=bjztHV9rC-0) |
| 코드 ||


연구 History

|년도|1st 저자|논문명|코드|
|-|-|-|-|
|2007|Nicola Bellotto|[PEOPLE TRACKING WITH A MOBILE ROBOT: A COMPARISON OF KALMAN AND PARTICLE FILTERS](http://www.robots.ox.ac.uk/~nick/doc/Bellotto2007b.pdf)||
|2010|Nicola Bellotto|[Computationally Efficient Solutions for Tracking People with a Mobile Robot: an Experimental Evaluation of Bayesian Filters](http://webpages.lincoln.ac.uk/nbellotto/doc/Bellotto2010.pdf)||
|2017|Zhi Yan|[Online learning for human classification in 3D LiDAR-based tracking](http://webpages.lincoln.ac.uk/nbellotto/doc/Yan2017.pdf)|[github](https://github.com/yzrobot/online_learning) |
|2018|Zhi Yan|[Multisensor Online Transfer Learning for 3D LiDAR-based Human Detection with a Mobile Robot](https://arxiv.org/pdf/1801.04137.pdf)|[Github](https://github.com/LCAS/online_learning/tree/multisensor)|



## 2. RELATED WORK

휴먼 탐지/트래킹 관련 연구 `Human detection and tracking have been widely studied in recent years.`

- 대부분의 연구는 **RGB-D 카메라** 를 이용하였으며, 제약적인 탐지 범위와 FoV를 가지고 있다. `Many popular approaches are based on RGB-D cameras [12], [13], although these have limited range and field of view. `

- Lidar가 대안이 될수 있지만, 포인트클라우드의 적은 정보로 사람을 인지 하는 것이다. `3D LiDARs can be an alternative, but one of the main challenges working with these sensors is the difficulty of recognizing humans using only the relatively low information they provide. `


### 2.1 그룹화

- 가능한 접근법은 그룹화 하는것이다. `A possible approach to detect humans is by clustering point clouds in depth images or 3D laser scans. `

- For example,
  - Rusu [14] presented a straightforward but computationally expensive method based on Euclidean distance.
  - Bogoslavskyi and Stachniss [15] proposed a faster approach, although the computational efficiency limits the clustering precision.

```
[14] R. B. Rusu, “Semantic 3D object maps for everyday manipulation in human living environments,” Ph.D. dissertation, Computer Science department, Technische Universitaet Muenchen, Germany, 2009.
[15] I. Bogoslavskyi and C. Stachniss, “Fast range image-based segmentation of sparse 3d laser scans for online operation,” in Proceedings of the 2016 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2016, pp. 163–169.
```

### 2.2 분류기 학습 (Offline)

-일반적인 접근법은 분류기를 학습시켜 사용하는 것이다. `A very common approach is to use an offline trained classifier for human detection. `


- For example,
  - 7개의 특징값을 SVM을 이용하여 학습 `Navarro-Serment et.al. [1], introduced seven features for human classification and trained an SVM classifier based on these features. `
  - 2개의 특징값을 추가 `Kidono et al. [3] proposed two additional features considering the 3D human shape and the clothing material (i.e. using the reflected laser beam intensities), showing significant classification improvements. ``
  - 샘플링방식으로 향상 `Li et al. [7] implemented instead a resampling algorithm in order to improve the quality of the geometric features proposed by the former authors.`
  - 탑다운 분류기 & 다운업 탐지기 사용 `Spinello et al. [4] combined a top-down classifier based on volumetric features and a bottom-up detector, to reduce false positives for distant persons tracked in 3D LiDAR scans.`
  - 슬라이딩 윈도우 이용 `Wang and Posner [8] applied a sliding window approach to 3D point data for object detection, including humans.`
    - They divided the space enclosed by a 3D bounding box into sparse feature grids,
    - then trained a linear SVM classifier based on six features related to the occupancy of the cells,
    - the distribution of points within them, and the reflectance of these points.

```
[1] L. E. Navarro-Serment, C. Mertz, and M. Hebert, “Pedestrian detection and tracking using three-dimensional ladar data,” in Proceedings of the 7th Conference on Field and Service Robotics (FSR), 2009, pp. 103–112.
[3] K. Kidono, T. Miyasaka, A. Watanabe, T. Naito, and J. Miura, “Pedestrian recognition using high-definition LIDAR,” in Proceedings of the 2011 IEEE Intelligent Vehicles Symposium (IV), 2011, pp. 405–410.
[7] K. Li, X. Wang, Y. Xu, and J. Wang, “Density enhancement-based long-range pedestrian detection using 3-d range data,” IEEE Transactions on Intelligent Transportation Systems, vol. 17, pp. 1368–1380, 2016.
[4] L. Spinello, M. Luber, and K. O. Arras, “Tracking people in 3d using a bottom-up top-down detector,” in Proceedings of the 2011 IEEE International Conference on Robotics and Automation (ICRA), 2011, pp. 1304–1310.
[8] D. Z. Wang and I. Posner, “Voting for voting in online point cloud object detection,” in Proceedings of Robotics: Science and Systems, 2015.
```

- 학습기반 방식의 제약 : The problem with offline methods, though, is that the classifier needs to be manually retrained every time for new environments.


### 2.3 Tracking

- 위 방법들은 Lidar데이터에서 학습된 분류기를 통해서 사람을 탐지 하는 방법 들이다. `The above solutions rely on pre-trained classifiers to detect humans from the most recent LiDAR scan. `

- tracking을 통해서 사람 탐지 하는것은 많이 연구 되진 않았다. `Only a few methods have been proposed that use tracking to boost human detection. `

- for example,
  - 확장 칼만필터를 이용하여 탐지력 높임 `Shackleton et al. [2],  employed an Extended Kalman Filter to estimate the position of a target and assist human detection in the next LiDAR scan. `
  - 준-지도 학습 방법을 이용 `Teichman et al. [5] presented a semi-supervised learning method for track classification. `
    - Their method requires a large set of labeled background objects (i.e. no pedestrians) to trains classifiers offline,
    - which showed good performances for track classification but not for object recognition.

```
[2] J. Shackleton, B. V. Voorst, and J. A. Hesch, “Tracking people with a 360-degree lidar,” in Proceedings of the Seventh IEEE International Conference on Advanced Video and Signal Based Surveillance (AVSS), 2010, pp. 420–426.
[5] A. Teichman and S. Thrun, “Tracking-based semi-supervised learning,” International Journal of Robotics Research (IJRR), vol. 31, no. 7, pp. 804–818, 2012.
```

### 2.4 annotation-free methods

-깊이 카메라와 다르게 데이터 셋이 별로 없다. `Besides datasets collected with RGB-D cameras [12], [13],[16], there are a few 3D LiDAR datasets available to the scientific community for outdoor scenarios [4], [6], [9], [17], but not with annotated data for human tracking in large indoor environments, like the one presented here.`


- Some authors proposed annotation-free methods.
  - 비지도 학습 방식 3D--> 2D 투영 `Deuge et al. [6] introduced an unsupervised feature learning approach for outdoor object classification by projecting 3D LiDAR scans into 2D depth images. `
  - 움직임 정보를 기반으로 `Dewan et al. [18] proposed a model-free approach for detecting and tracking dynamic objects, which relies only on motion cues. `

```
[6] M. D. Deuge, A. Quadros, C. Hung, and B. Douillard, “Unsupervised feature learning for classification of outdoor 3d scans,” in Proceedings of the Australasian Conference on Robotics and Automation (ACRA), 2013.
[18] A. Dewan, T. Caselitz, G. D. Tipaldi, and W. Burgard, “Motion-based detection and tracking in 3D LiDAR scans,” in Proceedings of the 2016 IEEE International Conference on Robotics and Automation (ICRA), 2016, pp. 4508–4513.
```

하지만 이 방법들은 별로 좋지 않다 `These methods, however, are either not very accurate or unsuitable for slow and static pedestrians.`

annotation free방식은 최근 트랜드에 맞지 않아 보인다. `It is clear that there remains a large gap between the state of the art and what would be required for an annotation free, high-reliability human classification implementation that works with 3D LiDAR scans. `

본 연구에서는 **Tracking**과 **Online Learning**을 합쳐서 사람 분류 정확도를 향상 시키고자 한다. `Our work helps to close this gap by demonstrating that human classification performance can be improved by combining tracking and online learning with a mobile robot in highly dynamic environments.`


## 3.  GENERAL FRAMEWORK


제안 방식은 3개의 모듈로 구성되어 있다. 
- a 3D LiDAR point cluster detector : 전체 스캔에서 클러스터 추출 
- a multi-target tracker : 각 클러스터의 위치와 속도 계산,  trajectories생성 
- a human classifier and a sample generator : 사람 여부 분류 
`Our learning framework is based on four main components: a 3D LiDAR point cluster detector, a multi-target tracker, a human classifier and a sample generator (see Fig. 2). At each iteration, a 3D LiDAR scan (i.e. 3D point cloud) is first segmented into clusters. The position and velocity of these clusters are estimated in real-time by a multi-target tracking system, which outputs the trajectories of all the clusters. At the same time, a classifier identifies the type of cluster, i.e. human or non-human. `

처음에는 지도기반 학습 방식을 사용한다. 학습량이 적어도 이후 단계에서 샘플이 증가 하면서 추가 학습이 이루어 진다. `At first, the classifier has to be initialised by supervised training with labeled clusters of human subjects. The initial training set can be very small though (e.g. one sample), as more samples will be incrementally added and used for retraining the classifier in future iterations. `

분류기는 false positive와 false negativ두 종류의 에러를 출력 한다. 에러는 독립된 두 전문가 시스템(P-expert와 N-expert)에 의해서 결정 된다. 에러의 종류에 따라서 샘플 생성기는 다음 단계에서 사용할 새 학습 데이터를 생성한다. `The classifier can make two types of errors: false positive and false negative. Based on the estimation of the error type by two independent “experts”, i.e. a positive P-expert and a negative one N-expert, which cross-check the output of the classifier with that of the tracker, the sample generator produces new training data for the next iteration. `

좀더 자세히는 P-expert는  false negatives 를 positive 샘플로 만들고, N-expert는 false positive를 negative 샘플을 만든다. 충분한 샘플수가 생성되면 재 학습을 수행 한다. 이러한 학습은 지정된 목표점에 도달 하면 멈춘다. `the P-expert converts false negatives into positive samples, while the N-expert converts false positives into negative samples. When there are enough new samples, the classifier is re-trained. The process typically iterates until convergence (i.e. no more false positives and false negatives) or some other stopping criterion is reached.`

![](https://i.imgur.com/j5ZQhfD.png)

제안 방식은 기존 [11]과 비교 하여 하기 세가지가 다르다. `Our system, however, differs from the previous work [11] in three key aspects, `
- [1] namely the frequency of the training process,
- [2] the independence of the tracker from the classifier, 
- [3] and the implementation of the experts. 

```
[11] Z. Kalal, K. Mikolajczyk, and J. Matas, “Tracking-learning-detection,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 34, pp. 1409–1422, 2012.
```
구체적으로 `In particular,`
- [1] 프레임단위로 학습하지 않고 배치 형식을 따른다. ` rather than instance-incremental training (i.e. frame-by-frame training), our system relies on less frequent batch-incremental training [19] (i.e. gathering samples in batches to train classifiers), collecting new data online as the robot moves in the environment. `
- [2] 분류기의 성능이 P-N expert와 추적기의 성능에 영향을 받기 때문에 추적기는 분류기와 독립되도록 설계 하였다. `Also, while the performance of the human classifier depends on the reliability of the P-N experts and the tracker, the latter is independent from and completely unaffected by the classification performance. `
- [3] Finally, the implementation of our experts can deal with more than one target and therefore generate new training samples from multiple detections, speeding up the online training process.

```
[19] J. Read, A. Bifet, B. Pfahringer, and G. Holmes, “Batch-incremental versus instance-incremental learning in dynamic and evolving data,” in Proceedings of the Eleventh International Symposium on Intelligent Data Analysis (IDA 2012), 2012, pp. 313–323.
```

제안 시스템은 특정 조건에서 동작한다. 이러한 조건이 실생활에서 항상 만족하지는 않는다. 하지만, 조건이란 존재 하는 대상인 사람이  많은 환경에서는 유용하다. `Note that under particular conditions, the stability of our training process is guaranteed as per [11]. The assumption here is that the number of correct samples generated by the N-expert is greater than the number of errors of the P-expert, and conversely that the correct samples of the Pexpert outnumber the errors of the N-expert. Although in the real world these assumptions are not always met, the stability of our system is simplified by the fact that we operate in environments where the vast majority of moving targets are humans and occasional errors are corrected online.`

## 4. 3D LIDAR-BASED TRACKING

Key components of this system include the efficient 3D LiDAR point cluster detector and the robust multi-target tracker. This section provides details about both.

### 4.1 Cluster Detector

입력 : 3D LiDAR 스캔 `The input of this module is a 3D LiDAR scan,`

#### A. 바닦제거 

z값을 지정하여 바닦 필터링. 바닦이 평면하다는 가정하에 수행 `The first step of the cluster detection is to remove the ground plane by keeping only the points pi with z_i . This is necessary in order to remove from object clusters points that belong to the floor, accepting the fact that small parts of the object bottom could be removed as well. Note that this simple but efficient solution works well only for relatively flat ground, which is one of the assumptions in our scenarios.`

#### B. Euclidean Distance 클러스터링 

Point clusters are then extracted from the point cloud P ∗ , based on the Euclidean distance between points in 3D space

![](https://i.imgur.com/OctuJJ5.png)

##### 가. 거리에 따라 d 변경 
거리에 따라 distance threshold(d*)는 바뀌어야 함. 따라서 제안에서는 adaptive 방식을 사용함 `We therefore propose an adaptive method to determine d ∗ according to different scan ranges, that can be formulated as`

$$
d^* = 2 * r * tan \frac{Θ}{2}
$$
- r is the scan range of the 3D LiDAR and 
- Θ is the fixed vertical angular resolution

![](https://i.imgur.com/rjaJ2V9.png)

위 공식에 따라 9m의 대상의 d는 0.314이다. `9 m, the minimum threshold should be d ∗ = 0.314 m`


##### 나. 

문제점 : Clustering points in 3D space, however, can be computationally intensive. The computational load is proportional to the desired coverage area: the longer the maximum range, the higher the value of d ∗ , and therefore the number of point clouds that can be considered as clusters. In addition, the larger the area, the more likely it is that indeed new clusters will appear within it. 

해결책 : 공간을 센서를 기준으로 nested circular region으로 나눔 To face this challenge, we propose to divide the space into nested circular regions centred at the sensor (see Fig. 4), like wave fronts propagating from a point source, where different distance thresholds are applied.

> 추후 다시 확인 


### 4.2 Multi-target Tracker

추적에는 UKF를 사용하고 DA에는 NN를 사용하였다. `Cluster tracking is performed using Unscented Kalman Filter (UKF) and Nearest Neighbour (NN) data association methods, which have already been proved to perform efficiently in previous systems [10], [16].`

```
[10] N. Bellotto and H. Hu, “Computationally efficient solutions for tracking people with a mobile robot: an experimental evaluation of bayesian filters,” Autonomous Robots, vol. 28, pp. 425–438, 2010.
[16] T. Linder, S. Breuers, B. Leibe, and K. O. Arras, “On multi-modal people tracking from mobile platforms in very crowded and dynamic environments,” in Proceedings of the 2016 IEEE International Conference on Robotics and Automation (ICRA), 2016, pp. 5512–5519.
```

추적은 2D정보를 이용하였다. 3D 정보(높이)는 현재는 사용하지않았다. (추후 연구 예정)` Tracking is performed in 2D, assuming people move on a plane, and without taking into account the 3D cluster size, which is left to future extensions of our work.`

#### A. Estimation 단계 `The estimation consists of two steps.`

- In the first step, the following 2D constant velocity model is used to predict the target state at time tk given the previous state at tk−1

- In the second step, if one or more new observations are available from the cluster detector, the predicted states are updated using a 2D polar observation model

For the sake of simplicity, in the above equations, noises and transformations between robot and world frames of reference are omitted. However, it is worth noting that, from our experience, the choice of the (non-linear) polar observation model, rather than a simpler (linear) Cartesian model, as in [20], is important for the good performance of long range tracking. This applies independently of the robot sensor used, as in virtually all of them, the resolution of the detection decreases with the distance of the target. In particular, the polar coordinates better represent the actual functioning of the LiDAR sensor, so its angular and range noises are more accurately modelled. This leads also to the UKF adoption, since it is known to perform better than Extended Kalman Filters (EKF) in the case of non-linear models [10]. Finally, the NN data association takes care of multiple cluster observations in order to update, in parallel, multiple UKFs (i.e. one for each tracked target).

## 5. ONLINE LEARNING FOR HUMAN CLASSIFICATION

온라인 학습은 새 클러스터 샘플을 선택/축적 하여 반복적으로 재 학습하면서 이루어 진다. `Online learning is performed iteratively by selecting and accumulating a pre-defined number of new cluster samples while the robot moves and/or tracks people, and re-training a classifier using old and new samples. `

Details of the process are presented next.

### 5.1 Human Classifier

분류기는 SVM을 사용하였다. `A Support Vector Machine (SVM) [21] is used for human classification, which is known to be effective in non-linear cases and has shown to work well experimentally in 3D LiDAR-based human detection [1], [3]. `

6개 특징 61D 정보를 학습에 사용 하였다. `Six features with a total of 61 dimensions are extracted from the clusters for human classification, as shown in Table I.`

![](https://i.imgur.com/HI0QJqi.png)

The set of feature values of each sample Cj forms a vector fj = (f1, . . . , f6). 
- Features from f1 to f4 were introduced by [1], 
- while features f5 and f6 were proposed by [3]. 

```
[1] L. E. Navarro-Serment, C. Mertz, and M. Hebert, “Pedestrian detection and tracking using three-dimensional ladar data,” in Proceedings of the 7th Conference on Field and Service Robotics (FSR), 2009, pp. 103– 112.
[3] K. Kidono, T. Miyasaka, A. Watanabe, T. Naito, and J. Miura, “Pedestrian recognition using high-definition LIDAR,” in Proceedings of the 2011 IEEE Intelligent Vehicles Symposium (IV), 2011, pp. 405– 410.
```

We discard the other three features, i.e. the so-called “geometric features”, presented in [1], because of their relatively low classification performance [3] and the heavy computational load observed in our experiments, which make them unsuitable for realtime tracking.

실험결과 위 6개 특징은 초기 학습을 서있는 사람만으로 수행하였어도 앉은 사람, 서있는 사람에게 모두 유용하였다. `We also observed that our classifier, based on this set features, can typically identify both standing and sitting people, even after being initially trained with samples of walking people only.`


분류기의 학습 데이터 분류는 1:1이다(Positivie:Negative), 데이터는 [-1,1]로 scaled되었다. **Gaussian Radial Basis Function kernel**를 이용하여 확률 값을 출력 하게 되어있다. ` A binary classifier is trained for human classification (i.e. human or non-human) at each iteration, based on the above features, using LIBSVM [22]. The ratio of positive to negative training samples is set to 1 : 1, and all data are scaled to [−1, 1], generating probability outputs and using a Gaussian Radial Basis Function kernel [23]. `

```
[22] C.-C. Chang and C.-J. Lin, “LIBSVM: A library for support vector machines,” ACM Transactions on Intelligent Systems and Technology, vol. 2, pp. 1–27, 2011.
[23] S. S. Keerthi and C.-J. Lin, “Asymptotic behaviors of support vector machines with gaussian kernel,” Neural Computation, vol. 15, no. 7, pp. 1667–1689, 2003.
```

Since LIBSVM does not currently support incremental learning, our system stores all the training samples accumulated from the beginning and retrains the entire classifier at each new iteration. The framework, however, also allows for other classifiers and learning algorithms.

### 5.2 Sample Generator

두개의 독립된 **experts**를 이용하여 새 학습 데이터를 만드는 방법이 적용 되어 있다. ` An approach based on two independent positive and negative experts is adopted for generating new training samples.`

 At each time step, 
 - the P-expert analyses all the new cluster samples classified as negative, identifies those that are more likely to be wrong (i.e. false negatives) and adds them to the training set as positive samples. 
 - The N-expert instead analyses samples classified as positive, extracts the wrong ones (i.e. false positives) and adds them to the set of negative samples for the next training iteration. 

P-expert는 분류기의 일반성(**generality**)을 향상 시키고, N-expert는 분류기의 구별성(**discriminability**)을 향상 시킨다. `The P-expert increases the classifier’s generality, while the N-expert increases the classifier’s discriminability. `

사전에 정의된 수 만큼의 학습 데이터가 수집되면 재 학습이 이루어 진다. 학습 절차는 사전에 지정되 값에 도달될때까지 반복 된다. `Once a pre-defined number of new samples is collected, the augmented training set is used to re-train the classifier. This learning process iterates until convergence or other stopping criterion, such as maximum training set size.`

#### A. P-expert 

추적기의 경로(**trajectories**)정보에 기반 한다. 이 아이디어는 비사람으로 분류 되었지만 **사람과 비슷한 경로**를 가진 물체가 있다면 positive sample로 취급 하는 것이다. ` The P-expert is based on the tracker’s trajectories. The idea is that clusters classified as non-human (negative) but belonging to a human-like trajectory in which at least one cluster has been classified as human (positive), will be considered as false negatives and added to the training set as positive samples.`

판단의 근거가 되는 **사람과 비슷한 경로**의 조건은 아래와 같다. `In our system, a human-like trajectory satisfies the following two conditions:`
- 1) the target moves a minimum distance r p min within a given time interval K ∆t:
- 2) the target’s velocity is non-zero but also not faster than a person’s preferred walking speed of 1.4 m/s

> 이후 추가 내용 확인 필요 

#### B. N-expert

False positive를 negative 샘플로 변경한다. 이 아이디어는 아무리 가만히 서있는 사람이라도 조금의 움직임/크기변화는 있을것이고리는 가정하에 수행 된다. ` The N-expert converts false positives into new negative samples. We assume that people are not completely static, and there will still be some small changes in the clusters’ shape and/or position even though they are just standing or sitting.`

센서의 높은 accuracy특성(~3cm)으로 이를 탐지 할수 있으며  고정된 물체의 낮은 위치 변화량은 아래의 공식으로 식별 가능하다. `Taking advantage of the 3D LiDAR’s high accuracy, these static objects with low position variances can be identified by the following conditions:`

> 이후 추가 내용 확인 필요 

## 6. Experiments 

Velodyne VLP-16 3D LiDAR


## 7. CONCLUSIONS



