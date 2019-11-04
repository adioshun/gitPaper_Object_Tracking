[3D-LIDAR Multi Object Tracking for Autonomous Driving](https://www.slideshare.net/adioshun/3dlidar-multi-object-tracking-for-autonomous-driving-111277160): A.S. Abdul Rachman,석사 학위 논문, 140page



# 

## 1. Introduction



### 1.3 Multi-Object-Tracking: Overview and State-of-Art

![](https://i.imgur.com/cMfis3G.png)

> tracking-by-detection 모델(탐지와 추적이 연속적으로 이루어짐)


#### On the detection stage, 

segmentation of the raw data is done to build basic feature of the scanned objects and to distinguish between dynamic and static objects. 

Subsequently, the segmented objects pose is then estimated based on either its outlier feature or fitting the measurement into known model[28]. 

At this stage, the raw measurement has already been translated into refined measurement with meaningful attributes (e.g. the static/dynamic classification and the objects pose). 

In another word, the object is successfully detected. 


Subsequently, the detected object is given to state estimation filter so that object kinematic states can be predicted according to a dynamic motion model. 

The purpose of the tracker is to effectively estimate the possible evolution of the detected object states even in the presence of measurement uncertainties, an optimal Bayesian filter such as Kalman Filter and Particle Filter are extensively used to address state uncertainties due to unmodelled dynamics of target evolution or noise acting on measurement.



## 2. data association (DA)

Next, a data association (DA) procedure is done by assigning detected object into a track with established filter state or newly started trajectory. 

The purpose of DA is to ensure the detected objects are localized while simultaneously maintaining their unique identity. 

At this stage uncertainties of the measurement due to finite sensor resolution and/or detection step imperfection may arise 

In order to address this issue, Bayesian probabilistic approaches are commonly used to handle DA process
- Joint Probabilistic Data Association Filter (JPDAF)[29] 
- Multiple Hypothesis Tracking (MHT)[30].


## 3. 

Finally, in practice a track management is required to cancel out spurious track based on specific criteria. 

Track management is responsible for handling existence and classification uncertainties based on a sensor-specific heuristic threshold. 

For example, track existence is defined to be true if 90% of its trajectory of track hypothesis is associated with a highly correlated measurement. 

The same also applies with class uncertainties; object class can be determined based thresholding of dimension, past motion, or visual cue (i.e. colour). 

Alternatively, these two uncertainties can also be assumed to be a random variable evolving according to Markovian
process. 

The Integrated Probabilistic Data Association[31] specifically model the existence of track as binary Markov process.

## 추가 팁 

In a typical urban scenario, sensor occlusions are unavoidable occurrences, to address this several works try to explicitly model occlusion[32, 33, 34] during detection and incorporated occlusion handling in their object tracking method. 

In the inter-process level, one notable approach as proposed by Himmelsbach[25] is the bottom-up/top-down approach which lets the tracker and detector exchange stored geometrical information to refine the detection parameter.

Alternatively, some other works have proposed a track-before-detect method that avoids explicit detection that takes sensor raw data as an input for tracking process[35].

> 여러 Lidar를 이용한 front-back/back-front 기법은 어떤가? 


---
## 2. Detection Fundamentals

  

### 2.1 Overview

  

### 2.2 Spatial Data Acquisition using 3D LIDAR

  

### 2.3 Segmentation

  

포인트 클라우드는 데이터 양이 많기 때문에 의미 있는 그룹으로 묶어 주는 작업이 선행되어야 한다. ` A large amount of point clouds data demand a high computational power to process, in addition, due to discontinuous nature of point cloud, it is useful to combine the geometric points into a semantically meaningful group.`

따라서 탐지 작업 전에 전처리로 불필요한 부분을 제거 해야 한다. ` Therefore, the raw measurement needs to be pre-processed to eliminate unnecessary element and reduce the dimensionality of possible target object before it passed to the detection process. `

세그멘테인은 도로, 커브등과 같이 추적이 불 필요한 오브젝트와 차량, 보행자 같은 유니크한 물체로 구분시킨다. `The segmentation process mainly deals with differentiating non-trackable objects such as terrain and kerb from unique objects of interest such as cars, cyclists and pedestrians.`

추적 방법에 따른 세그멘테이션 기법 분류 ` Segmentation method can be divided into two groups based on the underlying tracking scheme[28]: the grid-based and the object-based. `

-  그리드 기반 : The grid-based segmentation is mainly used for **track-before-detect** scheme, and 
- 오브젝트 기반 : the object-based is used mostly for the **track-by-detect** scheme. 

혼합 형태도 있다. Although it is important to note the relation is not exclusive. For example, Himmelsbach et al.[25] used grid-based approach in the pre-processing stage (specifically clustering) but later used object-based approach to perform tracking.


#### A. Grid-based

이 방식은 특정 그리드에 물체가 존재 확률을 나타내는 **global occupancy grid maps**에 의지 하고 있다. `The grid cell-based methods chiefly rely on the global occupancy grid maps to indicate the probability of an object existing in a specific grid (i.e. occupied). `

가장 일반적인 존재 확률 갱신 방법은 **Bayesian Occupancy Filter**을 이용하여서 이전 프레임과 현 프레임의 존재 여부를 비교 하는 것이다. `One common approach to update the probabilities is to compare the occupancy in current time frame k with occupancy in time frame k − 1 with the Bayesian Occupancy Filter[74]. `

파티클 필터가 사용되어 셀의 속도도 추출 한 연구도 있다. `In some implementation, Particle Filter is also used in used derive velocity in each grid[75, 76]. `

The particles with positions and velocities in a particular cell represent its velocity distribution and occupancy likelihood.

이웃셀과 속도가 비슷하다면 클러스터링 된다. When the neighbouring cells with similar speeds are clustered, then each cluster can be represented in the form of an oriented cuboid. 

장점 : 그리드 기반 추적의 결과는  간단한 탐지 절차와 덜 복잡한 DA절차를 가진다. `Grid-based tracking results in simpler detection process and less complicated data association process,`

단점 
- 가려짐등으로 이동 물체를 탐지 하지 못하게 되면 해당 영역은 Static Grid로 간주 된다. ` however, the disadvantage of grid-based representations is if a moving object cannot be detected (e.g. due to occlusion), the area will be mapped as a static grid by default if no occlusion handling is applied. `

- 또한,  Additionally, grid-based representations contain a lot of irrelevant details such as unreachable free space areas and only have limited ability to represent dynamic objects[36]. 

그리드 기반 탐지는 자율주행에는 적합하지 않다. `The latter part suggests grid-based detection is insufficient for the purpose of urban autonomous vehicle perception which require detailed representation of dynamic objects to model the complex environment. `

따라서, 본 논문에서는 **track-by-detect**와 **object-based ** 기법에 좀더 초점을 두어 살펴 본다. `Therefore, we shall focus on the track-by-detect scheme, and the object-based detection will be explored more in-depth.`
  
#### B. Object-based

Point Model을 이용하여 물체를 표현한다. `Object based segmentation on the other hand, uses the point model (or rather collections of bounded points) to describe objects. `

이전 방식과 달리 **pose estimation**와 **tracking filter**를 필요로 한다. `Unlike grid-based method, a separate pose estimation and tracking filter are required to derive dynamic attributes of the segmented object. `

세그멘테이션 절차는 비-추적 물체를 분리하는 지표면 제거절차와 클러스터링을 거쳐 진행 된다. `The segmentation is chiefly done with ground extraction to separate non-trackable object with objects of interest, followed by clustering to reduce the dimensionality of tracked objects. `

The two steps will be discussed in the following subsections. 

Note that the step-wise results of object-based segmentation processes can be seen in Figure. 2-5.

  ![](https://i.imgur.com/URg92zi.png)

  

##### 가. Ground Extraction

도면이 2차 평면이 아니기 때문에 포인트 클라우드는 장애물이 아닌 지면 정보도 포함 하고 있다. `Due to non-planar nature of roads, a point cloud coming from 3D Laser scanner also includes terrain information which is considered as non-obstacle (i.e. navigable) by the vehicle. `

운행 가능한 도면을 표시 해두는것은 중요 하다. `It is useful to semantically label the navigable terrain (hereby called ground) from the elevated point that might pose as an obstacle. `


지표면 제거는 Object Detection에서 중요한 절차 이다. `Ground extraction is an important preliminary process in object detection process. `

Lidar의 연산 부하는 무시 할수 없다. `As we are going to deal with a large number of raw LIDAR measurement, computation load must be factored during implementation. `

[77]에서는 도로 제거를 3분류 하였다. `Chen et al.[77]divide ground extraction process into three subgroups:`
- grid/cell-based methods, 
- line-based methods and 
- surface-based methods. 

[15]에서는 후자 2개를 scan-based method로 합쳤다. `Meanwhile, Rieken et al.[15] consider line-based and surface-based method as one big group called scan-based method. `

###### Grid-cell based method 
- divides the LIDAR data into **polar coordinate cells** and **channel**. 
- The method uses information of height and the radial distance between adjacent grid to deduct existence of obstacle when slope between cell cross certain threshold, i.e. slope-based method (see[12, 10]).

###### scan-based method 
- extracts a planar ground (i.e. flat world assumption) derived from specific criteria, 
- one of the approaches is to take the lowest z value and applying Random sample consensus (RANSAC) fitting to determine possible ground[78]. 


그리드 셀 기반 방식의 장점 : The advantage of grid cell-based method is that the ground contour is preserved and flat terrain is represented better. 

그리드 셀 기반 방식의 단점 : However, compared to the scan-based method it does not consider the value of neighbourhood channels, and thus may lead to inconsistency of elevation due to over-sensitivity to slope change. 

Ground measurement may also be incomplete due to occlusion by a large object.

하이드리브 방법 : `One notable approach which factored the occlusion is by Rieken et all.[14],`
- they combine of channel-wise ground classification with a grid-based representation of the ground height to cope with false spatial measurements and 
- also use inter-channel dependency to compensate for missing data.


  


##### 나. Clustering

포인트클라우는 연산 부하가 크다. `A large number of point clouds are computationally prohibitive if the detection and tracking are to be done over individual hits.` 

따라서 클러스터링 단계가 필요 하다. `Therefore, these massive point cloud is to be reduced into smaller clusters in which each of them is simply a combination of multiple, spatially close-range samples; the process is called clustering. `

클러스터링 방법론은 3가지로 나뉠수 있다. `Clustering can be either done in`
- 3D
- 2D (taking the top-view or XY plane) or 
- 2.5D[79] which retain some z-axis information such as elevation in occupancy grids.

###### 2D clustering

2D clustering offers computationally simpler operation. 

[80]에서 2D에 Connected Component클러스터링을 적용 하였다. [14]에서는 이를 3D에 적용 하였다. `Rubio et al.[80] presented a 2D clustering based on Connected Component Clustering which has shown to be implementable in real-time due to its low complexity, this method is also used in 3D object tracking method by Rieken et al.[14].`

일부 2D 추적 기법들도 사용 가능 하다. `Some literature in 2D object tracking[25, 45, 81] have shown that this approach is often sufficient in the application of object tracking. `

그러나 나무 밑에 있는 사람 처럼 중첩된 환경에서는 하나로 인식 하기에 조심 해야 한다. `However, care should be taken as vertically stacked objects (e.g. pedestrian under a tree) will be merged into one single cluster, which might be undesirable depending on the vehicle navigation strategy.`

###### 3D Clustering 

-3D clustering offers high fidelity object cluster that incorporates the vertical (z-axis) features.

Still, the resulting data and the computational effort required is some magnitude larger than its 2D counterpart. 

Compared to 2D clustering, there are fewer works which explicitly deal with 3D clustering for object tracking with LIDAR data. 

[63]에서는 RNN을 이용하였다. `Klasing et. al.[63] proposed a 3D clustering method based on Radially Bounded Nearest Neighbor (RNN), `

[82]에서는 DBSCAN을 이용하였다. `and more recently Hwang, et. al[82] using DBSCAN (Density-Based Spatial Clustering of Applications with Noise) to employ full 3D clustering. `

실시간성을 만족 하기 위해서 2D, 2.5D 기법이 좀더 선호 된다. `Considering real-time requirement and significant number of points involved, 2 or 2.5D clustering is more preferred[83] owing to the fact vehicle onboard computer likely to have limited computational power.`

### 2.4 Pose Estimation

물체의 경로나 방향을 알기위해서는 [자세추정]이 진행 되어야 한다. `Subsequently, in order to extract usable information in term of the object trajectory and orientation, the pose estimation needs to be done. `

물체의 자세(pose)란 다음의 것들을 의미 한다. `Object pose is a broad term that may include the`
- dimension, 
- orientation (heading), 
- velocity and 
- acceleration of such objects. 

분류 `Pose estimation generally can be grouped into model-based or feature-based method. `
- 모델 기반 : 센서 수집값과 알고 있는 도형 모델 연결 `Model-based pose estimation aims to match raw measurement into a known geometric model, `
- 특징 기반 : 물체의 특징 정로보 추론 `while feature-based pose estimation deduces object shape from a sets of feature.`


#### A. Model-based

##### 가. 모양과 관련하여 

모델 기반 방식은 최적화 기법을 이용하여 반복하여 차량을 cuboid등에 맞춘다. `Model-based pose estimation uses optimization-based iteration to fit vehicle into cuboid or rectangle representation. `

The cuboid object is parametrized and the most probable vehicle pose from the segment points are iterated. 

In order to fit clusters of points into a model, edge-like features are to be extracted and "best-fit" method is utilised to fit it into a known model.


Barrois[84]
- A notable example is in Barrois[84] where the optimization problem is formulated as the minimization of the polar distance between the scene points obtained and the visible sides to compute the best vehicle pose.

Petrovskaya and Thurn[21]
- Petrovskaya and Thurn[21] use importance sampling scoring based on the fitting of measurement to a predetermined geometric model. 

Another interesting approach is by Morris et.al.[85] 
- whose matched filter takes view-dependent self-occlusion effects into account, 
- and utilise 4 rectangles to represent the inner and outer sides of the vehicle.

##### 나. 방향과 관련하여 

Another the major challenge in bounding box generation is the orientation estimation, 

일반적 방법은 : common approaches are by calculating the minimum area of clustered points[86], 

단점은, 가려진 부분이 있으면 성능이 않좋다. however in the presence of partial occlusion the results can be spurious in term of dimension and orientation accuracy, 

문제 해결 #1 : to tackle this problem Rieken, et al.[43] uses an L-, U- or I-like simple set of geometric classifier to derive most appropriate orientation. 

문제 해결 #2 : An alternative approach is to use convex-hull method to generate bounding box[87, 52]; the idea is to minimize the average distance between the
convex-hull points and fit a rectangle. (see Figure 2-6).



![](https://i.imgur.com/XcTXYMv.png)

Model-based pose estimation can also be combined with Bayesian probabilistic likelihood such as in the works of Vu and Aycard[88], Liu[64], and Nashashibi[89]. 

They explicitly modelled the possible occlusion area to estimate the vehicle dimension based on scan-line distance.

Liu[64] in particular uses "transitional region" between the inner-outer bounding box model object aiming to accommodate more measurement errors.


모델 기반의 단점은 
- 계산 시간이 큼 `Although model-based method offers optimal pose estimation, the major disadvantage of this method is the high computational time required, and this may not be suitable for realtime application. `
- 최적화 측면에서 Local Minima에 빠질 가능성이 있다. `Moreover, the optimisation problem may reach a solution at local minima depending on the initialization and results in a sub-optimal pose. `

그래서 특징 기반 방식이 선호 된다. `As a consequence, the feature-based pose estimation shall be preferred.`


#### B. Feature-based

edge특징을 이용하여 물체의 차원을 유추 할 수 있다. `Feature-based pose estimation, on the other hand, deduces the object dimension based on the edge features. `

For instance, 
- Darms et al.[90] extracted edge targets from raw measurement as a part of the box model to represent the vehicle’s pose. 
- Himmelsbach et. al.[25] used Random sample consensus (RANSAC) algorithm to fit the dominant line of the segment points with the orientations of the objects. 
- Luo et al.[45] uses a graph-based method to fit clustered scans into an arbitrary shape, although this approach does not provide orientation information as is.
- Another approach, as done by Ye et. al.[91] is to extract a sequence of points facing the sensor with the smallest radius among the others in similar observation azimuth. 
    - then these points were fitted into L-shaped polyline using iteration endpoint algorithm. 
- Mertz et al.[92] use corner-fitting method to iterate through set of edge points to deduce the possible corner points.
- Similarly, Tay et al.[93] use edge filtering to deduce a bounding box vertex by iterating the edge lines to the nearest end point. (Refer to Figure 2-7)


![](https://i.imgur.com/9IX0xMy.png)


- 머신러닝 이용 : Machine-learning AdaBoost based detection methods can be found in the work of Zhang[94] et al., 
    - they use positive training samples obtained from multiple viewpoints of the object to train the detector to find 3D Harr-like features from clustered points. 
    - The trained detector is then used to generate a voxelized box for detection result. 

- 딥러닝 이용 : More recently, Braun, et. al.[95] utilises sensor-fusion approach using Regional Convolutional Neural Network (R-CNN) to estimate object orientation based on the joint proposal of stereo camera as well as LIDAR data

특징 기반 탐지 기법은 정확도-계산시간의 트레이드 오프 관계 이다. 일부에서는 불안정한 센서 측정값에 너무 민감하다는 우려도 있다. `Feature-based detection offers good trade-off between accurate pose estimation and computational time. However, Liu[28] also asserted that these approaches are notably sensitive to unstable measurement.`


### 챕터 요약 

To summarise this chapter, the 3D LIDAR sensor is selected due to its ability to acquire surround spatial information with feasible computational cost. 

However, occlusion-aware detection method has to be used to utilise the full potential of LIDAR. 

In addition, a real-time requirement calls for efficient methods. 

Therefore, based on these two criteria, the **Slope-based channel classification** and **2D Connected Component Clustering** are to be used for segmentation process with embedded height. 

Meanwhile, the pose estimation shall utilise minimum area rectangle augmented with L-shape fitting and cluster height information to form a 3D Box. 

Both methods have been shown to yield fast but reasonably accurate detection result under urban environment[96, 15], which in turn is essential for mission-critical urban MOT task.

---

## 3. Tracking Fundamentals

### 3.1 Overview

![](https://i.imgur.com/EW4cpdY.png)

1. The result of detection is used to start a new track, or if existing track exists, the measurement is checked
if it statistically likely to be correct measurement through gating. 

2. Passing the gating is the prerequisite of association between measurement and tracks before being sent forward to state
estimator. 

This chapter shall cover the assumed modelling of sensor and target dynamics, used not only in state estimatio and prediction, but also Data Association. 

Accordingly, several classes of Bayesian filter will be introduced.


### 3.2 Sensor and Target Modelling

The LIDAR sensor is placed on moving ego-car, which is considered as the origin. 

Due to the measuring principle of rotating LIDAR sensor the measurement originally comes in polar coordinates (in the form of distance and bearing). 

However, Velodyne digital processing unit readily provides the sensor measurement on the Cartesian coordinate system, 

In this thesis, the latter coordinate system will be used to conform better with ego-vehicle navigation frame. 

The relation between the ego-car navigation frame and sensor measurement frame can be seen in Figure 3-2.


> ???

### 3.3 Object Tracking as A Filtering Problem

In this thesis, object tracking problem is modelled as a filtering problem in which the object states are to be estimated from noisy, corrupted or otherwise false measurements. 

The estimated states and assumed system dynamics are given in the previous section. 

The basic concept of Bayes filtering is introduced along with the filters which will be used during implementation.


#### A. Bayes Theorem in Object Tracking


#### B. Unscented Kalman Filter

#### C. Interacting Multiple Mode


#### D. Data Association Filter

칼만필터가 예측값의 최적화를 지원 하긴 하지만 `KF (or its variant) offers the ability to provide optimal estimates of an object track. `

그럼에도불구하고  센싱의 불확실성으로 인해서 추적 물체가 해당 물체가 맞는지 보장 할수 없다. `Notwithstanding, due to measurement uncertainty there is no guarantee that the tracked object is actually a relevant object, or even if it is an existing object in the first place. `

따라서 후속적인 분류및 확인 절차가 필요 하다. `Therefore, a subsequent classification and validation of the estimated track are simply necessary.`

Data Association (DA) is a process of associating the detection result into a tracking filter.

There are two classes of DA filter: 
- the deterministic filter and 
- the probabilistic filter. 


##### 가. NNF

- Representative of deterministic DA filter is Nearest Neighborhood Filter (NNF) algorithm 

- NNF updates each object with the closest measurement relative to the state. 

- NNF associates object with known track based on the shortest Euclidean or the Mahalanobis distance between the
measurement and track.

##### 나. PDAF

- The probabilistic DA filter that is very well-known in object tracking literature body is the
eponymous Probabilistic Data Association Filter (PDAF)[29]. 

- The PDAF perform a weighted update of the object state using all association hypotheses in order to avoid hard, possibly erroneous association decisions commonly encountered in the use of NNF algorithm. 

- The erroneous association is often found during the scenario in which multiple measurements is located close to each other (i.e. clutter) and results in single measurement being used to incorrectly update all other nearby objects.


> PDA is also one of the most computationally efficient tracking algorithms among clutter-aware tracker[97], for
instance when compared to MHT[117]. 


![](https://i.imgur.com/eJBHCLO.png)



### 3.4 Probabilistic Data Association Filter (단일)


### 3.5 JPDA: Tracking Multiple Target in Clutter (멀티)






---

## 4. MOT System Design and Implementation

### 4.1 Overview

챕터 구성 
- First, the system architecture will be given to give the reader a bird-view of the implementation.
- Subsequently, the development platform is presented to inform readers how the proposed framework is designed and implemented. 
- Finally, each of the building blocks in the system will be visited to investigate the individual underlying reasons and how the theory is implemented in practice.


### 4.2 System Architecture


주요 구성 요소 `The MOT framework is divided into two major components based on the functional objective:`
- (1) 탐지기: **Detector** which aims to produce meaningful, unambiguous segmented objects derived from raw LIDAR data and 
- (2) 추적기: **Tracker**, whose task is to assign and maintain dynamic attributes of detected objects across all time frame.



The component hierarchy and input-output flows can be seen in Figure 4-1. 

입력 : The input of the system is the raw data from 3D LIDAR scan in point cloud representation of environmental spatial data in Cartesian coordinate, 

출력 : while the output is a list of objects embedded with context aware attributes, namely the 
- bounding boxes, 
- trajectories and 
- static/dynamic classification.


탐지기 구성 3요소 `Detector consists of 3 sub-components which similarly, represent the subtask of detection process: `
- the ground removal, to eliminate object of non-interest and reduce dimensionality of raw data, 
- clustering, which segment the point clouds into collection of coherently grouped points (i.e. the object), and 
- bounding box fitting, which embed a uniform object dimension and general direction heading information to each cluster.

Tracker retrieves the list of bounding boxes and is responsible for keeping itself updated for the bounding box spatial and dimensional evolution on each time step change, handled by Position Tracker and Box Tracker sub-component, respectively. 

Note that the spatial evolution is expected to change according to motion model, while the box dimension should stays constant with changing heading. 

Both evolutions are perturbed by noise and uncertainties, therefore position tracker requires Bayesian filtering to reject the disturbance. 

Tracker also stores the output state of every track iteration in box history, in turn, the Box Tracker relies on past information from Box History to filter out noise before updating the bounding box.

다중 물체 추적(MOT)는 여러 구성 요소의 동작으로 작동한다. `The MOT tasks are largely sequential and inter-related, as each component relies on the output of preceding components to perform its task.`

To exploit this behaviour, the so-called bottom-up top-down approach[25] is used. 
- The Detector ("top" component) is to exchange information in a feedback-loop fashion with Tracker ("bottom" component) to reduce false detection. 
- Since track-by-detection paradigm is used, this approach also augments Tracker component task.

![](https://i.imgur.com/U0Ux5iU.png)


### 4.3 Development Framework and Methodology


C++ & Matlab 

PCL, Opencv 사용 



### 4.4 Detector 

탐지기는 추적기에게 전달 하기전 바운딩 박스 등 전처리를 의미한다. `The detector component is responsible for initial pre-processing and fitting of bounding boxes which later are to be passed to the tracker. `

Each step on detection process is going to be presented in the following subsections. 

Note that the parameters of the detector used in implementations can be found in Appendix C.



#### A. Ground Removal

수 많은 라이다 데이터 중에서 대부분은 지표면 값이며 이는 불필요 한 정보로 제거가 필요 하다. `The raw LIDAR data consists of approximately 3.0 × 10 5 points and a large majority of the points belongs to the ground plane which does not carry meaningful information for tracking purpose, the first step in measurement pre-processing is thus the ground removal.`

지표면 제거 방식은 아래와 같다. `In this process, the removal is done using`
- slope-based channel classification augmented with consistency check and median filtering. 

지표면이 제거된 Lidar 측정값은 **elevated points** 라고 불리운다. `The LIDAR measurement with the ground removed is called elevated points.`

지표면 제거를 위해 **slope-based channel classification[14]**가 사용되었는데 이는 가려진 물체를 처리 하는데 좋은 성능을 보인다. `To achieve this purpose slope-based channel classification is utilised following[14] which is shown to handle partial occlusion on LIDAR data efficiently. `

```
[14] J. Rieken, R. Matthaei, and M. Maurer, “Benefits of using explicit ground-plane information for grid-based urban environment modeling,” Fusion, pp. 2049–2056, 2015.
```

**slope based channel classification**은 연속된 포인트들의 높이값 차이를 비교하여 구획을 구분한다. `The slope based channel classification determines the ground height by compartmentalizing the LIDAR point clouds and comparing the difference of height (i.e. slope) of the successive compartment.`

![](https://i.imgur.com/d2VGFoC.png)

절차 `The procedure is as follows`: 
- 먼저 Lidar 스캔 영역을 NxM개의 셀로 나눈다. first, the raw LIDAR scans is divided into a polar grid with $$m_{bins} \× n_{channels} $$ -cells with minimum radius $$r_{min}$$ being the radius closest to ego vehicle and r max being the farthest radius also from ego-vehicle (see Figure 4-4). 
- 최소 반경과 최대 반경을 정한다. The minimum radius is the radius in which the reflection of the ego vehicle is no longer can be seen while the maximum radius is determined by the effective range of the sensor. 
- 아래 식을 이용하여 각 포인트 클라우드 데이터를 채널과 빈에 맵핑한다. Mapping of each point cloud data $$p_i = \{x_ i , y_ i , z_ i , I_ i\ }$$ (see Subsection 2-2-3) to each channel and bin is thus given as:

![](https://i.imgur.com/dn0Z7Vc.png)

높이 정보 z는 cell의 속성 정보로 맵핑 된다. `Note the height information of the point coming from z_i is stored as the mapped cell’s attribute.`

가장 낮은 z는 **prototype point**로 불리우며 'Local' 그라운드로 사용되어 셀에서 가장 낮은 포인트를 나타낸다. The lowest z i point, hereby called prototype point is to be used as ’local’ ground to determine the lowest possible point for all cells. 

절대 높이 **absolute (local) height** 정보는 가장 낮은 높이와 높은 높이의 차이 값으로 계산 된다. `Additionally, the absolute (local) height in the cell can be enumerated by computing the difference between lowest z i and the highest z i within the same cell.`

다음, 지표면 높이의 **threshold**를 구한다. Next, we define the interval $$ \[T_{hmin} T_{hmax} \]$$ of possible ground height h i for threshold-based classification. 

센서가 설치된 높이 정보는 사전에 알수 있기 때문에 이 정보도 활용한다. Since the sensor’s mounting height above ground level $$h_{sensor}$$ is known a priori, we can extrapolate that the closest point to the sensor must be situated above ground point.
- 따라서 , 센서 높이 정보는 t_{hmax}로 지표면으로 간주된다. Thus $$h_{sensor}$$ is then used as the $$T_{hmax}$$ for a point to be considered as ground. 

The cell information is filled from the **bins** which are closest to ego-vehicle to the farthest, if the **prototype point** of a cell lies inside the interval [T hmin T hmax ] then it is set as the h i of that cell. 

If the **prototype point** is higher than T hmax , then the ground level is set to be equal to that of $$h_{sensor}$$ .

각 셀들이 계산되고 나서 경사 m을 계산 하여 갑작스런 높이 변화가 있는지 체크 한다. `After all cells have been enumerated, a slope (simple Euclidean gradient) m between cells thus can be calculated to inspect if there is sudden height increase between cells. `

또한, 절대 높이 차이도 계산하여 지표면이 아닌 물체를 판단 하는데 사용된다. `In addition, absolute height difference is also computed as a secondary check to identify possible non-ground object residing in the neighbouring cell. `

경사면은 오랜지 색으로 표현 하였다. 일반적 지표면은 녹색이다. **elevated point **는 노란색으로 표시 하였다. The slopes are illustrated as an orange line in Figure 4-4, normal ground is represented by a green circle, and the elevated point is coloured as yellow 
- due to slope change and height difference with the previous cell exceeding a predefined threshold T slope and T hdif f .

비록 이 방법이 mooth terrain에는 성능이 좋으나 과속 방지턱이나 잔듸등 삐져나온 지표면은  elevated points로 인식 한다. `Although this approach is good for smooth terrain, a small, protruding terrain features, such as road bump and grass can still be classified as elevated points. `

문제 해결을 위해 **Consistency Check** 와 **Median Filtering[14]**를 추가 적용 하였다. `To tackle this Consistency Check and Median Filtering[14] are employed to further flatten the ground plane and yield better ground estimate. `

###### Consistency Check

Consistency Check is done by iterating non-ground cells which are flanked by non-ambiguous ground cells and then comparing the cells’ height consistency with the neighbouring cells. 

The cell height is compared against a predefined absolute height T f lat and its height difference with adjacent cells is compared to threshold T consistency . 

A value below thresholds indicates the cells should belong to the ground and thus they are to be re-classified accordingly.

###### Median filtering

Median filtering on the other hand deals with missing ground plane information (common due to occlusion), as the name implies, the height value of the missing cell is to be replaced with the median value of neighbouring cells. 

The polar grid is again iterated to identify ground cells which have missing information but is surrounded by ground cells, the tunable parameter of the filter is kernel (window) size s kernel which indicates the number of neighbouring cells involved.

At the last step, a tolerance value h tol is used during final classification to further smooth the transition between ground and elevated points in noisy measurement. 

The whole procedure is summarized in Algorithm 1 (Appendix B).

![](https://i.imgur.com/9WRG5rR.png)

#### B. Object Clustering

Connected Component Clustering는 2D 이미지용으로 개발 되었지만 3D에서도 적용 가능 하다. `Connected Component Clustering is utilised in order to distinguish each possible object in the elevated points. The connected component clustering is originally designed to find the connected region of 2D binary images. However, it is also applicable to LIDAR point cloud[80] since in urban situation traffic object does not stack vertically, and thus the top view of the LIDAR measurement can be treated as a 2D binary image. However, the height information is retained by deriving the difference between the highest and the lowest point in each cluster. The choice of using this approach allows the MOT system to perform in real-time while still preserving the height information of detected objects.`


Following two-pass row-by-row labelling[127] written in Algorithm 2 (Appendix B). 

이 방식은 **one pass**는 connectedness에 대한 임식 label할당시 사용되며 ** second pass**는 각 label을 유니크한 클러스터 ID로 변경한다. ` The approach uses one pass to assign the temporary label of ’connectedness’ and the second pass is to replace each label with the unique cluster ID.`

![](https://i.imgur.com/KSgmci3.png)

The XY plane of the elevated points is discretized into grids with m × n cells. 

The grid is assigned with 2 initial states, empty (0), occupied (-1) and assigned. 

Subsequently, a single cell in x, y location is picked as a central cell, and the clusterID counter is incremented by one.

Then all adjacent neighbouring cells (i.e. x − 1, y + 1, x, y + 1, x + 1, y + 1 x − 1, y, x + 1, y,x − 1, y − 1, x, y − 1, x + 1, y + 1) are checked for occupancy status and labelled with current cluster ID. 

This procedure is repeated for every and each x, y in the m × n grid, until all non-empty cluster has been assigned an ID.


#### C. Bounding Box Fitting

클러스터링 절차는 추적할 후보 물체들을 생성해 된다. `The clustering process produced candidate object to be tracked. `

물체에 대한 **intuitive semantic information** 정보 생성을 위해서 바운딩 박스를 활용한다. `In order to make intuitive semantic information about the object, bounding box is fitted to have uniform dimensional information and yaw direction. `

MAR을 적용하여 2D Box를 만들고, 높이 정보가 더해 지면 3D box를 만든다. `A Minimum Area Rectangle (MAR)[128] is applied to each clustered object which results in a 2D box, and when combined with height information retained in the clustering process, becomes a 3D bounding box (see Figure 4-7).`

MAR은 가려진 물체에는 성능이 좋지 않다. `The MAR approach is sufficient for most well-defined measurement of a target object, however it is not guaranteed to correctly enclose partially-occluded object (c.f. Figure 4-8). `

이를 해결 하기 위해 ** feature-based L-shape fitting**[91]이 제안 되었다. `To tackle this issue, a feature-based L-shape fitting as proposed by Ye[91] is used to deduce correct yaw orientation.`

![](https://i.imgur.com/5lGi5ec.png)

##### 가. L-shape fitting procedure 

동작과정 `The L-shape fitting procedure is done as follows: `

먼저 센서에서 반대로 가장 멀리 있는 점 x1, x2를 선택 한다. `first, we select two farthest outlier point x_1 and x_2 which lies on the opposite sides of the object facing the LIDAR sensor. `

선택된 두 점을 Line_d로 연결 한다. 직교선 L_o를 그린다. `A line L_d is then drawn between the two points, then an orthogonal line L o is projected from L d toward the available points.` 

**end-point fit**알고리즘을 이용하여 직교선 L_o와 최대 거리 d_max 와 90도 각도를 구한다. `The projected L_o with maximum distance d_max and angle close to 90 degrees is then obtained using iteration end-point fit algorithm[129]. `

직교 선에 연결된 점을 x3로 정한다. `The points connected to the orthogonal line then becomes the corner point x 3 . `

세 점을 이용하여 L-shaped polyline을 생성하여 완성 한다. `Closing a line between x 1 , x 2 and x 3 would form an L-shaped polyline. `

진행 방향(Orientation)은 가장 긴 직선을 이용하여 구한다. `The orientation of the bounding box is then determined by the longest line, as most traffic object (e.g. car, cyclist) is heading in parallel with its longest dimensional side. `

This procedure is illustrated in Figure 4-9.

![](https://i.imgur.com/QXRAIqo.png)

위 방식을 적용하려면 충분한 포인트들이 탐지 되어야 하며 탐지 대상이 cuboid형태여야 한다. `Note that this approach needs a sufficient amount of measurement points to be able to fit reliable line and is designed for a cuboid-like object. `

따라서 **L-shape fitting**은 자동차에는 적합하지만 가려진 물체나 사람이나, 자전거에는 적합 하지 않다. `Therefore, the L-shape fitting is only applied on a car-like object, which correspondingly also suffer from occlusion more than smaller objects such as cyclist and pedestrian.`

#### D. Rule-based Filter

탐지의 마지막 절차는 비 관심 영역인 벽, 나무, 빌딩 등을 제거 하는 것이다. `The last step of the detection process is to eliminate majority object of non-interest, such as a wall, bushes, building, and tree. `

치수 정보를 이용 하여 이를 진행 한다. `A dimensional thresholding is implemented to achieve this.`

추가적으로 길이, 넓이, 높이 정보와 길이와 높이 비율 도 활용 된다. `In addition to 3 standard dimensional sizes (length, width, height), the ratio between length and width is considered to remove disproportionately thin objects such as wall and side rails.`

또한, 포인트수와 밀집도 정보도 사용된다.(m^3당 포인트 수)  `Furthermore, amount of LIDAR points and its volumetric density (i.e. point per m 3 ) are also taken into account. `

하지만 conservative한 임계치값이 탐지 실패를 막기 위해 사용된다. 비슷한 dimensional profile을 가진 물체는 필터링 되지 않는다. `However, conservative thresholds are used in order to prevent missed detection; it is expected that objects which share a similar dimensional profile, such as pole and pedestrian, or car and large bush are not to be filtered. `

추가적으로 가려진 물체는 완벽한 바운딩 박스를 가지지 못한다는 점을 상기 해야 한다. `In addition, recall that occluded object will not have a full dimensional bounding box. `

그러므로, Thus, the ratio check is not applicable for over-segmented box-(es) which may belong to object of interest. 

따라서 Therefore, the ratio check is only applied for larger object to distinguish between thin wall-like object and fragment of a real object due to over-segmentation. 

룰기반 필터링 방법이 단점도 많지만 불필요한 물체를 제거 하는데 좋다. `The rule-based filter is not intended as a catch all measure for false positive, but it is intended to reduce significant number of non-object of interest passed to tracker components.`
