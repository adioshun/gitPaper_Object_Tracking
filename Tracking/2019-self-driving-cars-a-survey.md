# Self-Driving Cars: A Survey

https://arxiv.org/abs/1901.04407

아래 전체 중에서 Tracking 부분만 정리 [[전체는 여기]](https://legacy.gitbook.com/book/adioshun/deep_drive/edit#/edit/master/paper2018-survey-self-driving-cars.md?_k=fmgtwp)참고 

```
Summary
I. Introduction......................................................................2

II. Overview of the Architecture of Self-Driving Cars.....3

III. Perception ....................................................................4
A. Localization..............................................................4
    1) LIDAR-Based Localization .....................................5
    2) LIDAR plus Camera-Based Localization ................6
    3) Camera-Based Localization .....................................6
B. Offline Obstacle Mapping........................................7
    1) Discrete Space Metric Representations....................7
    2) Continuous Space Metric Representations...............8
C. Road Mapping..........................................................8
    1) Road Map Representation........................................8
    2) Road Map Creation ..................................................9
D. Moving Objects Tracking ......................................10
    1) Traditional Based MOT .........................................10
    2) Model Based MOT.................................................10
    3) Stereo Vision Based MOT .....................................11
    4) Grid Map Based MOT ...........................................11
    5) Sensor Fusion Based MOT ....................................11
    6) Deep Learning Based MOT ...................................12
E. Traffic Signalization Detection and Recognition.......12
    1) Traffic Light Detection and Recognition ...............12
    2) Traffic Sign Detection and Recognition.................13
    3) Pavement Marking Detection and Recognition......14

IV. Decision Making........................................................14
A. Route Planning.......................................................14
1) Goal-Directed Techniques .....................................15
2) Separator-Based Techniques..................................15
3) Hierarchical Techniques........................................15
4) Bounded-Hop Techniques .....................................15
5) Combinations.........................................................16
B. Motion Planning ....................................................16
1) Path Planning.........................................................16
2) Trajectory Planning ...............................................17
C. Control...................................................................19
1) Path Tracking Methods..........................................19
2) Hardware Actuation Control Methods...................20

V. Architecture of the UFES’s Car “IARA”...................20

VI. Self-Driving Cars under Development in the Industry
References .............................................................................24
```

---

### 3.4 Moving Objects Tracking

MOT는 자차 주변 물체 탐지 및 추적에 관련된 기능이다. `The Moving Objects Tracking (MOT) subsystem (also known as Detection and Tracking Multiple Objects - DATMO) is responsible for detecting and tracking the pose of moving obstacles in the environment around the self-driving car. This subsystem is essential to enable autonomous vehicles to take decisions and to avoid collision with potentially moving objects (e.g., other vehicles and pedestrians). `

주변 물체의 위치 는 센서등을 이용하여 매 순간 예측 된다. `Moving obstacles’ positions over time are usually estimated from data captured by ranging sensors, such as LIDAR and RADAR, or stereo cameras. Images from monocular cameras are useful to provide rich appearance information, which can be explored to improve moving obstacle hypotheses.`


센서의 불확실성을 보완 하기 위해 여러 필터들의 도움을 받는다. `To cope with uncertainty of sensor measurements, Bayes filters (e.g., Kalman and particle filter) are employed for state prediction. `

많은 관련연구 중에서 최근 10년의 연구 결과를 6개의 분류로 나누어 보았다. `Various methods for MOT have been proposed in the literature. Here, we present the most recent and relevant ones published in the last ten years. For earlier works, readers are referred to Petrovskaya et al. [PET12], Bernini et al. [BER14], and Girão et al. [GIR16]. Methods for MOT can be mainly categorized into six classes: traditional, model based, stereo vision based, grid map based, sensor fusion based, and deep learning based.`


#### A. Traditional Based MOT

전통적인 MOT는 3단계로 되어 있다. `Traditional MOT methods follow three main steps: `
- data segmentation, 
- data association, and 
- filtering [PET12]. 

```
[PET12] A. Petrovskaya, M. Perrollaz, L. Oliveira, L. Spinello, R. Triebel, A. Makris, J. D. Yoder, C. Laugier, U. Nunes, and P. Bessiere, “Awareness of road scene participants for autonomous driving”, in Handbook of Intelligent Vehicles, London: Springer, pp. 1383–1432, 2012.
```

In the data segmentation step, 
- sensor data are segmented using clustering or pattern recognition techniques. 

In the data association step, 
- segments of data are associated with targets (moving obstacles) using data association techniques. 

In the filtering phase, 
- for each target, a position is estimated by taking the **geometric mean** of the data assigned to the target. 

**Position estimates** are usually updated by Kalman or particle filters. 

Amaral et al. [AMA15] propose a traditional method for detection and tracking of moving vehicles using 3D LIDAR sensor. - 
- The 3D LIDAR point cloud is segmented into clusters of points using the **Euclidean distance**. 
- Obstacles (clusters) observed in the current scan sensor are associated with obstacles observed in previous scans using a **nearest neighbor** algorithm. 
- States of obstacles are estimated using a **particle filter** algorithm. 
- Obstacles with velocity above a given threshold are considered moving vehicles. 

```
[AMA15] E. Amaral, C. Badue, T. Oliveira-Santos, A. F. De Souza, “Detecção e Rastreamento de Veículos em Movimento para Automóveis Robóticos Autônomos”, Simpósio Brasileiro de Automação Inteligente, 2015, pp. 801–806.
```

[ZHA13]은 바운딩박스 정보를 이용하여서 차량인지 아닌지를 구분 하였다. `Zhang et al. [ZHA13] build a cube bounding box for each cluster and use box dimensions for distinguishing whether a cluster is a vehicle or not. `
- Data association is solved by an optimization algorithm. 
- A **Multiple Hypothesis Tracking** (MHT) algorithm is employed to mitigate association errors. 

```
[ZHA13] L. Zhang, Q. Li, M. Li, Q. Mao, and A. Nüchter, “Multiple vehicle-like target tracking based on the Velodyne LIDAR”, in IFAC Proceedings Volumes, vol. 46, no. 10, pp. 126–131, 2013.
```

[HWA16]는 이미지 정보를 이용해서 Lidar의 불필요한 부분을 제거 하였다. `Hwang et al. [HWA16] use images captured by a monocular camera to filter out 3D LIDAR points that do not belong to moving objects (pedestrians, cyclists, and vehicles). `
- Once filtered, object tracking is performed based on a segment matching technique using features extracted from images and 3D points.

```
[HWA16] S. Hwang, N. Kim, Y. Choi, S. Lee, and I. S. Kweon, “Fast multiple objects detection and tracking fusing color camera and 3d LIDAR for intelligent vehicles”, International Conference on Ubiquitous Robots and Ambient Intelligence, 2016, pp. 234–239
```

#### B. Model Based MOT 

모델 기반 방식은 센서 데이터에서 바로 센서의 물리 모델과 탐지 물체의 geometric모델을 이용하여서 바로 추론 작업을 진행 한다. 그리고 non-parametric filters를 사용한다. `Model-based methods directly infer from sensor data using physical models of sensors and geometric models of objects, and employing non-parametric filters (e.g., particle filters) [PET12]. `

군집화와 association작업은 불필요 하다.  `Data segmentation and association steps are not required,`
- because geometric object models associate data to targets. 


Petrovskaya and Thrun [PET09] present the model based method for detection and tracking of moving vehicles adopted by the self-driving car “Junior” [MON08]. 
- Moving vehicle hypotheses are detected using differences over LIDAR data between consecutive scans. 
- Instead of separating data segmentation and association steps, new sensor data are incorporated by updating the state of each vehicle target, which comprises vehicle pose and geometry. 
- This is achieved by a hybrid formulation that combines Kalman filter and RaoBlackwellized Particle Filter (RBPF). 


The work of Petrovskaya and Thrun [PET09] was revised by He et al. [HE16] that propose to combine RBPF with Scaling Series Particle Filter (SSPF) for geometry fitting and for motion estimate throughout the entire tracking process. 
- The geometry became a tracked variable, which means that its previous state is also used to predict the current state. 


Vu and Aycard [VU09] propose a model-based MOT method that aims at finding the most likely set of tracks (trajectories) of moving obstacles, given laser measurements over a sliding window of time. 
- A track is a sequence of object shapes (L-shape, I-shape and mass point) produced over time by an object satisfying the constraints of a measurement model and motion model from frame to frame. 
- Due to the high computational complexity of such a scheme, they employ a Data Driven Markov chain Monte Carlo (DD-MCMC) technique that enables traversing efficiently in the solution space to find the optimal solution. 
- DD-MCMC is designed to sample the probability distribution of a set of tracks, given the set of observations within a time interval. 
- At each iteration, DD-MCMC samples a new state (set of tracks) from the current state following a proposal distribution.
- The new candidate state is accepted with a given probability. 
- To provide initial proposals for the DD-MCMC, dynamic segments are detected from laser measurements that fall into free or unexplored regions of an occupancy grid map and moving obstacle hypotheses are generated by fitting predefined object models to dynamic segments. 


Wang et al. [WAN15] adopt a similar method to the model-based one, but they do not assume prior categories for moving objects.
- A Bayes filter is responsible for joint estimation of the pose of the sensor, geometry of static local background, and dynamics and geometry of objects. 
- Geometry information includes boundary points obtained with a 2D LIDAR. 
- Basically, the system operates by iteratively updating tracked states and associating new measurements to current targets. 

Hierarchical data association works in two levels. 
- In the first level, new observations (i.e., cluster of points) are matched against current dynamic or static targets. 
- In the second level, boundary points of obstacles are updated.

#### C. Stereo Vision Based MOT

Stereo vision based methods rely on color and depth information provided by stereo pairs of images for detecting and tracking moving obstacles in the environment. 


Ess et al. [ESS10] propose a method for obstacle detection and recognition that uses only synchronized video from a forwardlooking stereo camera. The focus of their work is obstacle tracking based on the per-frame output of pedestrian and car detectors. For obstacle detection, they employ a Support Vector Machine (SVM) classifier with Histogram of Oriented Gradients (HOG) features for categorizing each image region as obstacle or non-obstacle. For obstacle tracking, they apply a hypothesize-and-verify strategy for fitting a set of trajectories to the potentially detected obstacles, such that these trajectories together have a high posterior probability. The set of candidate trajectories is generated by Extended Kalman Filters (EKFs) initialized with obstacle detections. Finally, a model selection technique is used to retain only a minimal and conflict-free set of trajectories that explain past and present observations. 


Ziegler et al. [ZIE14a] describe the architecture of the modified Mercedes-Benz S-Class S500 “Bertha”, which drove autonomously on the historic BerthaBenz-Memorial-Route. For MOT, dense disparity images are reconstructed from stereo image pairs using Semi-Global Matching (SGM). All obstacles within the 3D environment are approximated by sets of thin and vertically oriented rectangles called super-pixels or stixels. Stixels are tracked over time using a Kalman filter. Finally, stixels are segmented into static background and moving obstacles using spatial, shape, and motion constraints. The spatio-temporal analysis is complemented by an appearance-based detection and recognition scheme, which exploits category-specific (pedestrian and vehicle) models and increases the robustness of the visual perception. The real-time recognition consists of three main phases: Region Of Interest (ROI) generation, obstacle classification, and object tracking. 


Chen et al. [CHEN17] compute a disparity map from a stereo image pair using a semi-global matching algorithm. Assisted by disparity maps, boundaries in the image segmentation produced by simple linear iterative clustering are classified into coplanar, hinge, and occlusion. Moving points are obtained during egomotion estimation by a modified Random Sample Consensus (RANSAC) algorithm. Finally, moving obstacles are extracted by merging super-pixels according to boundary types and their movements.

#### D. Grid Map Based MOT

이 방식은 **occupancy grid map**을 생성하는 것에서 부터 시작 한다. `Grid map based methods start by constructing an occupancy grid map of the dynamic environment [PET12]. `

Map 생성 작업은 다음과 같다. `The map construction step is followed by`
- data segmentation, 
- data association, 
- and filtering steps in order to provide object level representation of the scene. 


[NGU12]는 **양안 카메라**기반 방식을 제안 하였다. `Nguyen et al. [NGU12] propose a grid-based method for detection and tracking of moving objects using stereo camera. The focus of their work is pedestrian detection and tracking. 3D points are reconstructed from a stereo image pair. An inverse sensor model is used to estimate the occupancy probability of each cell of the grid map based on the associated 3D points. A hierarchical segmentation method is employed to cluster grid cells into segments based on the regional distance between cells. Finally, an Interactive Multiple Model (IMM) method is applied to track moving obstacles. `


 [AZI14] 는 **Octree**기반 방식을 제안 하였다. `Azim and Aycard [AZI14] use an octree-based 3D local occupancy grid map that divides the environment into occupied, free, and unknown voxels. After construction of the local grid map, moving obstacles can be detected based on inconsistencies between observed free and occupied spaces in the local grid map. Dynamic voxels are clustered into moving objects, which are further divided into layers. Moving objects are classified into known categories (pedestrians, bikes, cars, or buses) using geometric features extracted from each layer. `


Ge et al. [GE17] leverage a 2.5D occupancy grid map to model static background and detect moving obstacles. A grid cell stores the average height of 3D points whose 2D projection falls into the cell space domain. Motion hypotheses are detected from discrepancies between the current grid and the background model.

#### E. Sensor Fusion Based MOT

센서 퓨전 기반 방식은 여러 센서의 정보를 혼합하여 인지 성능을 향상시킨 방식이다. `Sensor fusion-based methods fuse data from various kinds of sensors (e.g., LIDAR, RADAR, and camera) in order to explore their individual characteristics and improve environment perception. `


Darms et al. [DAR09] present the sensor fusion-based method for detection and tracking of moving vehicles adopted by the self-driving car “Boss” [URM08]. The MOT subsystem is divided into two layers. The sensor layer extracts features from sensor data that may be used to describe a moving obstacle hypothesis according to either a point model or a box model. The sensor layer also attempts to associate features with currently predicted hypotheses from the fusion layer. Features that cannot be associated to an existing hypothesis are used to generate new proposals. An observation is generated for each feature associated with a given hypothesis, encapsulating all information that is necessary to update the estimation of the hypothesis state. Based on proposals and observations provided by the sensor layer, the fusion layer selects the best tracking model for each hypothesis and estimates (or updates the estimation of) the hypothesis state using a Kalman Filter. 


Cho et al. [CHO14] describe the new MOT subsystem used by the new experimental autonomous vehicle of the Carnegie Mellon University. The previous MOT subsystem, presented by Darms et al. [DAR09], was extended for exploiting camera data, in order to identify categories of moving objects (e.g., car, pedestrian, and bicyclists) and to enhance measurements from automotive-grade active sensors, such as LIDARs and RADARs. 


Mertz et al. [MER13] use scan lines that can be directly obtained from 2D LIDARs, from the projection of 3D LIDARs onto a 2D plane, or from the fusion of multiple sensors (LADAR, RADAR, and camera). Scan lines are transformed into world coordinates and segmented. Line and corner features are extracted for each segment. Segments are associated with existing obstacles and kinematics of objects are updated using a Kalman filter. 


Byun et al. [BYU15] merge tracks of moving obstacles generated from multiple sensors, such as RADARs, 2D LIDARs, and a 3D LIDAR. 2D LIDAR data is projected onto a 2D plane and moving obstacles are tracked using Joint Probabilistic Data Association Filter (JPDAF). 3D LIDAR data is projected onto an image and partitioned into moving obstacles using a region growing algorithm. Finally, poses of tracks are estimated or updated using Iterative Closest Points (ICP) matching or image-based data association. 


Xu et al. [XU15] describe the context-aware tracking of moving obstacles for distance keeping used by the new experimental driverless car of the Carnegie Mellon University. Given the behavioral context, a ROI is generated in the road network. Candidate targets inside the ROI are found and projected into road coordinates. The distance-keeping target is obtained by associating all candidate targets from different sensors (LIDAR, RADAR, and camera). 


Xue et al. [XUE17] fuse LIDAR and camera data to improve the accuracy of pedestrian detection. They use prior knowledge of a pedestrian height to reduce false detections. They estimate the height of the pedestrian according to pinhole camera equation, which combines camera and LIDAR measurements.

#### F. Deep Learning Based MOT 

딥러닝 기반 방식은 위치, 모양, 추적에 뉴럴 네트워크를 이용하는 방식이다. `Deep learning based methods use deep neural networks for detecting positions and geometries of moving obstacles, and tracking their future states based on current camera data. `


[HUV15]는 카메라 이미지와 CNN응 이용하였다. `Huval et al. [HUV15] propose a neural-based method for detection of moving vehicles using the Overfeat Convolutional Neural Network (CNN) [SER13] and monocular input images with focus on real-time performance. CNN aims at predicting location and range distance (depth) of cars in the same driving direction of the ego-vehicle using only the rear view of them. `

Mutz et al. [MUT17] address moving obstacle tracking for a closely related application known as “follow the leader”, which is relevant mainly for convoys of autonomous vehicles. The tracking method is built on top of the Generic Object Tracking Using Regression Networks (GOTURN) [HEL16]. GOTURN is a pre-trained deep neural network capable of tracking generic objects without further training or object specific fine-tuning. Initially, GOTURN receives as input an image(입력으로 이미지) and a manually delimited bounding box of the leader vehicle. It is assumed that the object of interest is in the center of the bounding box. Subsequently, for every new image, GOTURN gives as output an estimate of the position and geometry (height and width) of the bounding box. The leader vehicle position is estimated using LIDAR points that fall inside the bounding box and are considered to be vehicle.

```
[MUT17] F. Mutz, V. Cardoso, T. Teixeira, L. F. R. Jesus, M. A. Gonçalves, R. Guidolini, J. Oliveira, C. Badue, and A. F. De Souza, “Following the leader using a tracking system based on pre-trained deep neural networks”, International Joint Conference on Neural Networks, 2017, pp. 4332–4339.
```

