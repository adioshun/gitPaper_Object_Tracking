
# [Multisensor Online Transfer Learning for 3D LiDAR-based Human Detection with a Mobile Robot](https://arxiv.org/pdf/1801.04137.pdf)

> 2018, [깃허브(ROS, C++)](https://github.com/LCAS/online_learning/tree/multisensor), [실내로봇청소 프로젝트](http://www.flobot.eu/)


Abstract— Human detection and tracking is an essential task for service robots, where the combined use of multiple sensors has potential advantages that are yet to be fully exploited. In this paper, we introduce a framework allowing a robot to learn a new 3D LiDAR-based human classifier from other sensors over time, taking advantage of a multisensor tracking system. The main innovation is the use of different detectors for existing sensors (i.e. RGB-D camera, 2D LiDAR) to train, online, a new 3D LiDAR-based human classifier based on a new “trajectory probability”. 

Our framework uses this probability to check whether new detection belongs to a human trajectory, estimated by different sensors and/or detectors, and to learn a human classifier in a semi-supervised fashion. The framework has been implemented and tested on a real-world dataset collected by a mobile robot. We present experiments illustrating that our system is able to effectively learn from different sensors and from the environment, and that the performance of the 3D LiDAR-based human classification improves with the number of sensors/detectors used.

## I. INTRODUCTION

사람 탐지는 로봇의 이동에 중요 하다. 멀티 센서를 사용하면 탐색 반경도 넒어 지고, 더 풍부한 정보로 추적 성능을 올릴수 있다. `Human detection and tracking is an important task in service robotics, where knowledge of human motion properties such as position, velocity and direction can be used to improve the behavior of the robot, for example to improve its collision avoidance and adapt its velocity to that of the surrounding people. Using multiple sensors to track people has advantages over a single one. The most obvious one is that multiple sensors can often do the task with a wider field of view and thus track more people within a larger range [1], [2]. Another advantage is that multiple sensors providing redundant information can increase tracking accuracy and reliability [3], [4], [5], [6].`

다양한 센서(3D/2D Lidar, RGBD카메라)는 고유의 장단점을 가지고 있다. `Different sensors have different properties. The 3D LiDAR in our robot platform (Fig. 1) has 16 scan channels, 360◦ horizontal and 30◦ vertical fields of view, and up to 100 m range. However, this sensor provides only sparse point clouds, from which human detection can be very difficult because some useful features, such as color and texture, are missing. 2D LiDARs have obviously similar problems, with further limitations due the availability of a single scan channel and reduced field of view. However, these sensors are also cheaper than the previous, and have been used in mobile robotics long enough to stimulate the creation of many human detection algorithms [7], [8]. RGB-D cameras, instead, can detect humans more reliably but only within short range and limited field of view [9], [10].`

특정 환경에서 특정 센서(RGB-D)를 위한 사람 탐지 알고리즘은 많이 연구 되었다. 하지만 3D Lidar를 위한 연구는 많이 이루어 지지 않았다. 학습 데이터가 부족한것도 이유이다.  `In the literature, several algorithms already exist which can reliably detect human under particular conditions and with specific sensors (e.g. close range RGB-D detection). Other sensors, however, are not yet so popular to benefit from good human detection software (e.g. 3D LiDAR). In some cases, there are simply not enough datasets with such sensors to learn robust human classifiers for many realworld applications. `

In this paper, therefore, we wish to train a 3D LiDAR-based human classifier in a semi-supervised way by learning from existing RGB-D and 2D LiDAR-based detectors. 

좋은 탐지 성능은 좋은 추적기 성능을 이루어 내지만 본 논문에서는 **탐지**쪽에 초점을 맞추었다. 추적은 다음 논문에서 다루기로 한다. `Although better human detectors ultimately lead to better people tracking systems, here we focus on the first part only and leave the second for future work.`


 일반적으로 데이터 수집 및 학습은 오프라인으로 이루어 진다. 따라서 작업 비용 및 잠재적 휴먼 에러가 있다. 제안 방식에서는 시-공간적 정보를 이용하여서 추적기 서브시스템에서 온라인 학습을 수행한다. 이 추적기 서브 시스템은 사전에 학습된 2D LiDAR & RGB-D기반 사람 탐지기이다.  `Typically, data collection and training of the classifier are done offline, with obvious labor cost and potential human errors. In the proposed transfer learning framework, instead, a 3D LiDAR-based human classifier is trained online while exploiting spatio-temporal information from the tracking sub-system, which uses static (i.e. pre-trained) human detectors for the 2D LiDAR and RGB-D sensors. `

이 방식은 세미-지도 학습 방식을 이용하여 새 3D LIDAR가 추적되는 사람의 경로로 부터 학습 한다. `This framework enables a new sensor to learn from trajectories of the tracked people, `each one with an associated confidence, or probability, of being human-generated, and fusing different model-based (labeled) and model-free (unlabeled) detections according to a semi-supervised learning scheme [11]. 

```
[11] X. Zhu and A. B. Goldberg, Introduction to Semi-Supervised Learning. Morgan & Claypool, 2009.
```

기존 방식과 달리 제안 방식은 학습 데이터가 불필요 하고 온라인 학습을 진행 한다. `In contrast to previous approaches [12], [13], our solution does not need any hand-labeled data, performing online learning completely from scratch. Besides reducing the burden of data annotation, this feature makes our system easily adaptable to the environment where the robot is deployed.`

```
[12] Z. Yan, T. Duckett, and N. Bellotto, “Online learning for human classification in 3d lidar-based tracking,” in Proc. of IROS, 2017, pp. 864–871. 
[13] A. Teichman and S. Thrun, “Tracking-based semi-supervised learning,” International Journal of Robotics Research, vol. 31, no. 7, pp. 804–818, 2012.
```

본 논문의 기여 `The contributions of this paper can be summarized as follows:`
- 1) we propose an online transfer learning framework for multisensor people tracking based on a new **trajectory probability**, which takes into account both sensor independence (in the detection) and multisensor interaction (in the trajectory estimation); 
- 2) we present an **experimental evaluation** of our system for 3D LiDAR-based human classification with a mobile robot on a real-world dataset using different sensor combinations.

본 논문의 구성 `The remainder of the paper is organized as follows: `
- Section II provides an overview of relevant literature in human detection and tracking. 
- Section III presents our solution framework for online transfer learning. 
- Section IV describes the application of the proposed framework to the problem of 3D LiDAR-based human classification. 
- Section V illustrates the experimental results for different sensor configurations. 
- Finally, Section VI concludes the paper summarizing the contributions and suggesting future research work.

## II. RELATED WORK

추적관련 대부분의 시스템은 **베이지안 방법**을 사용한다. `The problem of multitarget and multisensor tracking has been extensively studied during the past few decades. Most of present systems are based on Bayesian methods [14],`
- which compute an estimate of the correspondence between **features detected in the sensor** data and the **different humans** to be tracked. 

로봇에서는 멀티 센서가 사용된다. `Regarding robotic applications, multiple sensors can be deployed in single- or multi-robot systems [15], while the former is the concern of this paper.`

카메라와 2D LIDAR기반 기존 연구들 `RGB/RGB-D camera plus 2D LiDAR is the most frequently used combination in the literature. `
- [3] presented two different methods for mobile robot tracking and following of a fast-moving person in an outdoor environment. The robot was equipped with an omnidirectional camera and a 2D LiDAR. 
- [16] presented an integrated system to detect and track people and cars in outdoor scenarios, based on the information retrieved from a camera and a 2D LiDAR on an autonomous car. 
- [2] introduced a people tracking system for mobile robots in very crowded and dynamic environments. Their system was evaluated with a robot equipped with two RGB-D cameras, a stereo camera and two 2D LiDARs.

저자의 이전 연구 `Our previous work with the aforementioned sensor combination includes [8] and [17]. `
- [8] The former presented a human tracking system for mobile service robots in populated environments, 
- [17] while the latter extended this system to a fully integrated perception pipeline for people detection and tracking in close vicinity to the robot. 
- 기존 제안 방식은 2D LIDAR로 다리를 탐지 하고 카메라로 상체를 탐지후에 UKF를 이용하였다. `The proposed tracking system tracks people by detecting legs extracted from a 2D LiDAR and fusing this with the faces or the upper-bodies detected with a camera using a sequential implementation of the Unscented Kalman Filter (UKF).`

```
[8] N. Bellotto and H. Hu, “Multisensor-based human detection and tracking for mobile service robots,” IEEE Transactions on Systems, Man, and Cybernetics – Part B, vol. 39, no. 1, pp. 167–181, 2009.
[17] C. Dondrup, N. Bellotto, F. Jovan, and M. Hanheide, “Real-time multisensor people tracking for human-robot spatial interaction,” in ICRA Workshop on Machine Learning for Social Robotics, 2015.
```

3D LIDAR를 이용한 연구들 `The combination with 3D LiDAR is increasing with the development of the 3D LiDAR technology. Taking advantage of its high accuracy, `
- [4] developed an algorithm to align 3D LiDAR data with high-resolution camera images obtained from five cameras, in order to accurately track moving vehicles. 
- Other reported results include [18] and [19], which mainly focused on **pedestrian detection** rather than tracking. 
- In addition, earlier work presented multitarget tracking with a mobile robot equipped with two 2D LiDARs, respectively located at the front and back [1]. 
- Thus the robot can have a 360◦ horizontal field of view, where each scan of these two sensors covers the whole surrounding of the robot at an angular resolution of 1◦ .

추적에 머신러닝을 이용한는것은 특별한 장점이 있다. `The use of machine learning algorithms for tracking has particular advantages. `
- [13]은 본 연구와 가장 비슷하며 EM알고리즘을 이용하여 반-지도학습으로 Track 분류를 하였다. `The closest work to ours is [13], where the authors proposed a semi-supervised learning approach to the problem of track classification in 3D LiDAR data, based on Expectation-Maximization (EM) algorithm. `
- 이와 반대로 본 논문에서는 위 방식과 달리 전혀 학습데이터가 필요 하지 않는 방식을 사용하였다. `In contrast to our approach, their learning procedure needs a small set of seed tracks and a large set of background tracks, that need to be manually or semi-manually labeled at first, whereas we do not need any hand-labeled data.`

(아마도) 멀티 센서를 이용한 온라인 학습 추적 시스템은 최초의 논문이다. `To our knowledge, no existing work in the robotics field explicitly exploits information from multisensor-based tracking to implement transfer learning between different sensors as in this paper. Our work combines the advantages of multiple sensors with the efficiency of semi-supervised learning, and integrates them into an single online framework applied to 3D LiDAR-based human detection.`

## III. ONLINE TRANSFER LEARNING

![](https://i.imgur.com/vgArSJu.png)

제안 방식은 4개의 모듈로 이루어져 있다. `An overview of our solution framework for online transfer learning can be seen in Fig. 2. It contains four main components: `
- static detectors denoted by Ds, 
- dynamic detectors Dd, 
- a target tracker T and 
- a label generator G. 

이해를 위해 위 순서로 설명 하곘다. `In order to facilitate the explanation, we present each component following the sequence of an entire iteration, starting with human detection.`

### 3.1 A. Human Detection and Tracking

Ds can detect humans with offline-trained or heuristic detectors, typically with high confidence, while Dd acquires this ability through the online framework. 

Both detectors provide **new observations** for T and their corresponding **probabilities** for G. 

Ds provides labeled detections, while Dd provides both labeled and unlabeled ones. Here we assume the initial training set is substituted, instead, by a transfer learning process between the initial Ds and the final Dd.

The tracking process T 
1. gathers the observations, 
2. fuses them and 
3. generates human motion estimates. 

Both moving and stationary targets are tracked. For the latter, the trajectory length is supposed to be null or at least very small. 

T associates human detections from different sensors to the same corresponding estimates, linking Ds and Dd detections and therefore making the transfer learning possible. 

In order to enable this on a mobile robot with multiple sensors, T should: 
- a) be robust to sensor noise and partial occlusions; 
- b) fuse multisensor data; 
- c) be able to deal with multiple targets simultaneously; 
- d) cope with noise introduced by robot motion.

### 3.2 B. Transfer Learning

The label generator G fuses the information coming from Ds, Dd and T , then generates training labels for Dd. 

A **trajectory probability** is measured by G, based on Bayes’ theorem. The idea is to measure the likelihood that a trajectory belongs to a human, which is defined as follow. 

Given an objectness proposal x_i and its category label y_i , P(yi |xi , dj ) denote the predictive probability that sample x_i is a human observed by detector dj ∈ D (D = Ds ∪ Dd) at time t. 

For a whole trajectory of detections XT {xi} and its category label YT , the predictive probability of the whole trajectory P(YT |XT , D) is computed by integrating the observations of the different detectors according to the following formula:

...

### 3.3 C. Convergence of the Learning Process

...

---

## IV. APPLICATION TO MULTISENSOR HUMAN DETECTION

In this section, we present an **online transfer learning** for human classification in 3D LiDAR scans using the robot shown in Fig. 1. 

The sensor configuration resembles the one adopted for an industrial floor washing robot developed by the EU project FLOBOT and, besides a 3D LiDAR on the top, includes an RGB-D camera and a 2D LiDAR mounted on the front. 

We describe in the following paragraphs how to use the state-of-the-art detectors to train a 3D LiDAR-based human detector online, instead of training it offline using manually-labeled samples.

![](https://i.imgur.com/G1iwdvS.png)

The detailed block diagram of our implementation can be seen in Fig. 3. 

At each iteration, **3D LiDAR** scans are 
- first segmented into point clusters. 
- The 2D position and velocity of these clusters are estimated in real-time by a multitarget tracking system, which outputs the trajectories of all the clusters. 
- At the same time, a classifier is trained to classify the clusters as human or not, assigning a normalized confidence value to each of them. 
	- This confidence is the predictive probability P(yi |xi , dj ) for the 3D LiDAR based detector, which is needed for the calculation of the trajectory probability in Eq.1-3. 
	- The classifier is initialized and retrained online. 
	- The trajectories and the probabilities are sent to a label generator, which generates the training labels for the next iteration.

The upper-body detector [10] and the leg detector [7], respectively based on the **RGB-D camera** and the **2D LiDAR**, are the static detectors Ds. 
- Both enable human tracking by sending the position of the detections. 
- In addition, they provide the corresponding probabilities P(yi |xi , dj ) (i.e. normalized detection confidence) to the label generator. 

The combination of 3D-LiDAR-based cluster detector and the human classifier, instead, constitutes the dynamic detector Dd that we want retrain online. 

For an intuitive understanding of the various detectors and their outputs, please refer to the example in Fig. 4. The following paragraphs describe each module in detail.

![](https://i.imgur.com/2L9fMTP.png)
```
Fig. 4. A screenshot of our multisensor-based detection and tracking system in action. 
- The sparse colored dots represent the laser beams with reflected intensity from the 3D LiDAR. 
- The white dots indicate the laser beams from the 2D LiDAR. 
- The colored point clouds are RGB images projected on depth data of the RGB-D camera. 
- The robot is at the center of the 3D LiDAR beam rings. 
- The numbers are the tracking IDs and the colored lines represent the people trajectories generated by the tracker. 
- For example, the person with tracking ID 49 has been 
	- detected by the RGB-D upper-body detector (green cube), 
	- the 2D LiDAR leg detector (green circle), 
	- and the 3D LiDAR cluster detector (blue bounding box).
```

### A. Upper Body Detector and Leg Detector

The upper-body detector identifies upper-bodies (shoulders and head) in 2D range (depth) images, taking advantage of a pre-defined template. The confidence of the detection is inversely proportional to the observation range. 

The leg detector detects legs in 2D LiDAR scans based on 14 features, including the number of beams, circularity, radius, mean curvature, mean speed, and more. 
- Its detection performance is limited by moving and crowd people. 

As the upperbody detector and leg detector are not probabilistic methods and for the sake of simplifying mathematical conversion, a **probability of 0.5** is assigned if x_i is detected as a human.

> 사람으로 탐지시 확률을 고정값 0.5로 지정 한다. 

### B. Cluster Detector and Human Classifier

The 3D LiDAR-based cluster detector and the human classifier are originally from our recent work [12], while the former has been incorporated in different problems [23], [24].

```
[12] Z. Yan, T. Duckett, and N. Bellotto, “Online learning for human classification in 3d lidar-based tracking,” in Proc. of IROS, 2017, pp. 864–871
[23] L. Sun, Z. Yan, S. M. Mellado, M. Hanheide, and T. Duckett, “3DOF pedestrian trajectory prediction learned from long-term autonomous mobile robot deployment data,” in In Proceedings of ICRA, Brisbane, Australia, May 2018.
[24] L. Sun, Z. Yan, A. Zaganidis, C. Zhao, and T. Duckett, “Recurrentoctomap: Learning state-based map refinement for long-term semantic mapping with 3d-lidar data,” IEEE Robotics and Automation Letters, 2018.
```

 As input of this module, a 3D LiDAR scan is 
 - 거리기반 클러스터링 알고리즘 적용 `first properly segmented into different clusters using an adaptive clustering approach. The latter enables to use different optimal thresholds for point cloud clustering according to the scan ranges.`
 - 6개의 Feature이용 SVM 학습 및 분류 `Then, a Support Vector Machine (SVM)-based classifier [25] with six features (a total of 61 dimensions) is trained online. `
	 - These features are selected to meet robots’ requirements for real-time and online computing performance. For more implementation details, please refer to [12].

SVM분류기에 대한 추가 설명 
- In our approach (based on LIBSVM [26]), the uncalibrated error function of SVM is squashed into a logistic function (here is the sigmoid function) to get the predictive probability P(yi |xi , dj ) (used in Eq. 3). 
- To be more specific, a binary classifier (i.e. human or non-human) is trained at each iteration. 
- The ratio of positive to negative training samples is set to 1 : 1, and all data are scaled to [−1, 1], generating probability outputs and using a Gaussian Radial Basis Function kernel [27]. 

하지만 LibSVM은 온라인 학습을 지원 하지 않는다. `Since LIBSVM does not currently support incremental learning, `the system stores all the training samples accumulated from the beginning and retrains the entire classifier at each new iteration. The solution framework, however, also allows for other classifiers and learning algorithms.

### C. Bayesian Tracker

People tracking is performed by a robust multisensor-multitarget Bayesian tracker, which has been widely used and described in previous works [2], [8], [12], [17], [28]. 

The estimation consists of two steps. 
- In the first step, a constant velocity model is used to predict the target state at time t given the previous state at t − 1. 
- In the second step, if one or more new observations are available from the detectors, the predicted states are updated using a Cartesian or a Polar observation model, depending on the type of the sensor. 

An efficient implementation of **Nearest Neighbor** data association is finally included to resolve ambiguities and assign each person the correct detections, in case more than one are simultaneously generated by the same or different sensors.

As reported in [2], [17], [28], our tracker fulfills the requirements listed in Sec. III-A. We refer the reader to these publications for further details.

```
[2] T. Linder, S. Breuers, B. Leibe, and K. O. Arras, “On multi-modal people tracking from mobile platforms in very crowded and dynamic environments,” in Proc. of ICRA, 2016, pp. 5512–5519.
[17] C. Dondrup, N. Bellotto, F. Jovan, and M. Hanheide, “Real-time multisensor people tracking for human-robot spatial interaction,” in ICRA Workshop on Machine Learning for Social Robotics, 2015
[28] N. Bellotto and H. Hu, “Computationally efficient solutions for tracking people with a mobile robot: an experimental evaluation of bayesian filters,” Autonomous Robots, vol. 28, pp. 425–438, 2010.
```

### D. Label Generator

The positive training labels X+ are generated according to Eq. 4, while the negatives X− are generated based on a volume filter:

![](https://i.imgur.com/U1G7uAh.png)

where Wi , Di , Hi are the width, depth and height of a 3D cluster xi . 

The idea is that clusters without a pre-defined human-like volumetric model will be considered as negative samples for the next training iteration. 

In our application, the dynamic classifier was trained from scratch without any manually-labeled initial sets. 

As the validation set is not available, the maximum number of iterations was used as halting criteria.

## V. EVALUATION

## VI. CONCLUSION

In this paper, we presented a framework for online transfer learning, applied to 3D LiDAR-based human classification, taking advantage of multisensor-based tracking. 

The framework, which relies on the computation of human trajectory probabilities, enables a robot to learn a new human classifier over time with the help of existing human detectors. 

To this end, we proposed a semi-supervised learning method, which fuses both model-based (labeled) and model-free (unlabeled) detections from different sensors. 

A very promising feature of the proposed solution is that the new human classifier can be learned directly from the deployment environment, thus removing the dependence on pre-annotated data. 

The experimental results, based on a real-world dataset, demonstrated the efficiency of our system. 

ROS코드로 공개 되어 있다. `The proposed framework has been fully implemented into ROS with a high level of modularity. The software and the dataset are publicly available to the research community, with the intention to perform objective and systematic comparisons between the recognition capabilities of different robots. Moreover, our framework is easy to extend to other sensors and moving objects, such as cars, bicycles and animals.`

개선이 필요한 사항 `Despite these encouraging results, there are several aspects which could be improved. `
- For example, the AP of the online learned classifier is still relatively low, due to the complexity of the environment recorded in our dataset. This can be further improved by using a more advanced model for negative sample generation. 
- In addition, it remains to be verified how a new human detector, based on the online trained classifier, will affect the stability of the system and its tracking performance.





