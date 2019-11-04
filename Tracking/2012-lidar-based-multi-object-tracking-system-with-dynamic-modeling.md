[LIDAR-BASED MULTI-OBJECT TRACKING SYSTEM WITH DYNAMIC MODELING](https://neu-gou.github.io/thesis_Mengran.pdf): 2012, 석사 학위 논문

DBSCAN  
DA : Greedy Nearest Neighbor \(GNN\)

## 1. Introduction

### 1.2 Multi-object tracking system

![](https://i.imgur.com/mBkE6k4.png)

#### A. detection

During the detection step,

* the raw data obtained by the sensor is captured by the system, 
* and the foreground data is partitioned from background data and separated into differ-
  ent segments. 

Once the foreground objects are detected and grouped, the problem of multi-object  
tracking becomes the problem of estimating the dynamic states of each object.

Generally, this estimation consists of two parts:

* Data Association and 
* Filtering 

### B. Data Association

Data association seeks to match the observations from the sensor to corresponding existing tracked objects.

### C. Data Filtering

Filtering is applied to improve the estimate of the state by combining recent observations with **models of object behavior**.

This model is not only used for data filtering, but also for predicting the motion of any temporarily occluded targets.

## 2. Related Work

### 2.1 Historical development of laser-based tracking system

the problem of detecting and tracking multi-object generally consists of three parts:

* Detection
* Filtering
* Data Association \[5\]. 

```
[5] Y. Bar-Shalom and T.E. Fortman, “Tracking and Data Association,” Academic Press, 1988
```

Detection addresses the problem of extracting the foreground from the raw data.

Filtering is applied to improve the estimate of the target state by combining the existing model and observation.

Data association deals with matching the observations from the sensor to existing tracked objects.

* This “matching” process for moving objects can be extremely hard for crowded scenes.

### 2.2 Moving objects detection for laser-based tracking system

#### A. Background subtraction

Most tracking systems apply a “background filter” first before processing the data

* Goal of a background filter : separate the foreground and background, and to remove noise

Background subtraction techniques have been widely used for detecting moving objects from static cameras \[23\].

```
[23] M. Piccardi, “Background subtraction techniques: A review,” IEEE International Conference on Systems, Man and Cybernetics, 2004, pp. 3099–3104
```

One approach, presented by Fod et,al., considers the different features provided by LIDAR within a background model based on range information \[24\].

```
[24] A. Fod, A. Howard, and M. J. Mataric, “Laser-based people tracking,” IEEE International Conference on Robotics and Automation, Washington DC, May, 2002, pp. 3024–3029
```

In order to simultaneously detect moving targets and maintain the position of stationary objects, an occupancy grid map is employed to detect moving objects \[25\].

```
[24] A. Fod, A. Howard, and M. J. Mataric, “Laser-based people tracking,” IEEE International Conference on Robotics and Automation, Washington DC, May, 2002, pp. 3024–3029
```

The grid-based map technique has been widely used in Simultaneous Localization and Mapping \(SLAM\) for representing complex outdoor environments \(Figure 2.1\) \[7, 35, 59\].

```
[7] C. C. Wang, “Simultaneous localization, mapping and moving object tracking,” PhD thesis, The Robotics Institute, Carnegie Mellon University, Pittsburgh, PA, Apr. 2004
[35] T. D. Vu, O. Aycard, and N. Appenrodt, “Online localization and mapping with moving object tracking in dynamic outdoor environments,” IEEE Intelligent Vehicles Symposium, Istanbul, Turkey, June 2007
[59] T. D. Vu and O. Aycard, “Laser-based detection and tracking moving objects using data-driven markov chain monte carlo,” IEEE International Conference on Robotics and Automation, Kobe, Japan, May 2009
```

Due to the few features extracted from LIDAR data, by building motion-map and stationary-map separately, Wang showed this approach is more robust than a feature-based approach \(Figure 2.1\) \[7\].

#### B. Segmentation

The data segmentation process seeks to

* divide the collected data points into distinct segments 
* such that points associated with the same object are grouped together. 

거리 기반 방식으로 세그멘테이션 `As a first criterion, a common method is to perform segmentation based on a distance threshold.`

* Examples : Wang et al., Maclanchlan and Mertz and Vu et al. \[28, 35, 66\].

한 물체의 연속적인 점들 도출 `As the next step, consecutive(연속적) points on the same object are sought(구하다).`

* 문제점 : This is challenging because of the limited angular resolution of the LIDAR sensor in this study. 
* 빈번한 position/angle 변경시 더 큰 문제 `Specifically, the distance between two consecutive scan points on the same surface can change dramatically depending on the varying position and angle of the surface relative to the sensor.`

```
[28] C. C. Wang, C. Thorpe, and S. Thrun, “Online simultaneous localization and mapping with detection and tracking of moving objects: Theory and results from a ground vehicle in crowded urban areas,” IEEE International Conference on Robotics and Automation, Vol. 1, Sept. 2003, pp. 842–849
[35] T. D. Vu, O. Aycard, and N. Appenrodt, “Online localization and mapping with moving object tracking in dynamic outdoor environments,” IEEE Intelligent Vehicles Symposium, Istanbul, Turkey, June 2007
[66] R. A. Maclachlan, “Tracking of moving objects from a moving vehicle using a scanning laser rangefinder,” IEEE Intelligent Transportation Systems Conference, Toronto, Canada, Sept. 2006
```

![](https://i.imgur.com/vSviMed.png)  
동일 물체라도 angle문제로 Distance가 서로 다르다.

해결책 \#1: Adaptive Distance Threshold `A possible solution to this problem has been suggested by Sparbert et al. and Mendes et al. who used an adaptive distance threshold to perform segmentation [29, 30].`

* Their methods are based on the distance between data point to the LIDAR. 
* In their method any two consecutive points $$r_k$$ and $$r_{k+1}$$ will be regarded as belonging to the same object if the distance between them $$r_{k,k+1}$$ fulfill the Equation 2.1: \(공식은 해당 논문 참고\)
* 위 해결책은 노이즈 포인트에 잘 대처 하지 못함 `Because the algorithm mentioned above is based on the consecutive beams, it is very sensitive to the noise point.`

```
[29] J. Sparbert, K. Dietmayer, and D. Streller, “Lane detection and street type classification using laser range images,” IEEE Intelligent Transportation System Conference, Oakland, CA, USA, Aug. 2001
[30] A. Mendes, L. C. Bento, and U. Nunes, “Multi-target detection and tracking with a laser-scanner,” IEEE Intelligent Vehicles Symposium, Parma, Italy, June 2004
```

해결책 \#2 : DBSCAN `Therefore, in this thesis, a density-based spatial clustering algorithm is applied for data segmentation. This clustering algorithm is a well-known data mining method named “Density-Based Spatial Clustering of Applications with Noise” (Figure 2.3) [31].`

> 자세한 DBSCAN의 설명 및 장점은 논문 참고   
> 본 논문에서 제안 하는 DBSCAN 파라미터
>
> * The minimum number of points required to form a cluster is set to 2. 
> * The distance is empirically chosen as 120cm.

```
[31] M. Ester, H. Kriegel, J. Sander, X. Xu, “A density-based algorithm for discovering clusters in large spatial databases with noise,” proc. 2nd Int. Conf. on Knowledge Discovery and Data Mining, Portland, OR, USA, 1996, pp. 226
```

#### C. Occlusion

![](https://i.imgur.com/j5LgoWY.png)

정의 : the shadow of one object may partially block the view of a second object, causing segmentation to try to classify the single second object as several different objects moving in unison \(Figure 2.4\).

해결책 \#1 : a geometric vehicle model that requires continuity between disjoint point segments \[47, 52, 81\].

해결책 \#2 : by performing image processing approaches before clustering step.

* the LIDAR image is processed as a 2D birds-eye-view image, 
* and afterwards image-processing methods are applied.
* eg \#1. Zhao and Thorpe used **Hough transformation** to extract the lines \[22\]
* eg \#2. Burke applied a **median filter** following PCA \[40\].

```
[47] A. Petrovskaya and S. Thrun, “Model based vehicle tracking in urban environments,” IEEE International Conference on Robotics and Automation, Workshop on Safe Navigation, Vol. 1, 2009, pp. 1–8
[52] T. Ogawa, H. Sakai, Y. Suzuki, K. Takagi, K. Morikawa, “Pedestrian detection and tracking using in-vehicle lidar for automotive application,” IEEE Intelligent Vehicles Symposium, Baden-Baden, Germany, June 2011
[81] O. Aycard, “Contribution to perception for intelligent vehicles,” PhD thesis, Universite de Grenoble, France, 2010
```

#### D. Classification

방법 \#1 : 물체별 속도로 구분 \(사람 5mph, 자동차 5mph+\)

방법 \#2 : laser-based multi-object tracking system has been studied extensively \[29, 30, 34, 36\]

```
[29] J. Sparbert, K. Dietmayer, and D. Streller, “Lane detection and street type classification using laser range images,” IEEE Intelligent Transportation System Conference, Oakland, CA, USA, Aug. 2001
[30] A. Mendes, L. C. Bento, and U. Nunes, “Multi-target detection and tracking with a laser-scanner,” IEEE Intelligent Vehicles Symposium, Parma, Italy, June 2004
[34] C. Premebida and U. Nunes, “A multi-target tracking and GMM-classifier for intelligent vehicles,” IEEE Intelligent Transportation Systems Conference, Toronto, Canada, Sept. 2006
[36] F. Nashashibi and A. Bargeton, “Laser-based vehicles tracking and classification using occlusion reasoning and confidence estimation,” IEEE Intelligent Vehicles Symposium, Eindhoven, The Netherlands, June 2008
```

방법 \#3 : Voting `Mendes, et al. uses a voting scheme presented by [37].`

* By considering all the hypotheses over time, an object is assigned with a class until the confidence level reaches a reasonable value \[30\]. 
* Although the features used are not discussed in detail, the results showed that voting classification approach can assign the right class after several frames \(Figure 2.5\).

```
[37] T. Deselaers, D. Keysers, R. Paredes, E. Vidal, and H. Ney, “Local representations for multi-object recognition,” DAGM 2003, Pattern Recognition, 25th DAGM Symp, pp. 305312, September 2003
```

### 2.3 Filtering methods for laser-based tracking system

목적 : Filtering is necessary to smooth the trajectory and to predict the vehicles pose state when the observation cannot be obtained directly.

방법 `To perform this filtering,`

* Bayesian based filters : the Kalman Filter, the Extended Kalman Filter, the particle filter 
* The IMM algorithm

#### A. Kalman Filter

#### B. Particle Filter

### 2.4 Data association approaches for laser-based tracking system

목적:  Data association seeks to match data points to a specific object

Lidar기반 DA의 문제점 : Because of the limited features provided by the distance sensor,  
accurate data association is very difficult, especially for crowded scenes.

가장 일반적 알고리즘 : Greedy Nearest Neighbor \(GNN\) filter \[5\]

* As the most intuitive approach to assign the nearest segment to the object
* 간담함, 직관적, 부하 적음, Reasonable ERROR , 많은 연구에서 활용 \[35, 38, 52\]

```
[5] Y. Bar-Shalom and T.E. Fortman, “Tracking and Data Association,” Academic Press, 1988
[35] T. D. Vu, O. Aycard, and N. Appenrodt, “Online localization and mapping with moving object tracking in dynamic outdoor environments,” IEEE Intelligent Vehicles Symposium, Istanbul, Turkey, June 2007
[38] F. Fayad and V. Cherfaoui, “Tracking objects using a laser scanner in driving situation based on modeling target shape,” IEEE Intelligent Vehicle Symposium, Istanbul, Turkey, June 2007
[52] T. Ogawa, H. Sakai, Y. Suzuki, K. Takagi, K. Morikawa, “Pedestrian detection and tracking using in-vehicle lidar for automotive application,” IEEE Intelligent Vehicles Symposium, Baden-Baden, Germany, June 2011
```

향상된 알고리즘\[53,54\] : advanced well-known Bayesian approaches are

* 1\) multiple hypothesis tracking \(MHT\) algorithm 
* 2\) joint probabilistic data association \(JPDA\)

```
[53] S. S. Blackman, “Multiple hypothesis tracking for multiple target tracking,” IEEE Aerospace and Electronic Systems Magazine, Vol. 19, No. 1, 2004, pp. 5–18
[54] Y. B. Shalom, T. Kirubarajan, and X. Lin, “ Probabilistic data association techniques for target tracking with applications to sonar, radar and EO sensors,” IEEE Aerospace and Electronic System Magazine, Vol. 20, No. 8, Aug. 2005, pp. 37–56
```

최근 알고리즘 : Recently, a batch of multi-object tracking systems have achieved notable success by applying Markov chain Monte Carlo data association \(MCMCDA\) \[55, 56, 57\].

* 기존 알고리즘 : Unlike other data-association methods that seek to maintain or test all possible data assignments, 
* MCMCDA uses Markov chain Monte Carlo sampling \[58\]. 
* 활용예 : Vu, et al,applied this data association for laser-based tracking systems and obtained positive results \[59\]

```
[55] S. Oh, S. Russell, and S. Sastry, “Markov chain monte carlo data association for general multiple-target tracking problems,” IEEE Conference on Decision and Control, Atlantis, Paradise Island, Bahamas, Dec. 2004
[56] T. Zhao, R. Nevatia, and B. Wu, “Segmentation and tracking of multiple humans in crowded environments,” IEEE Transactions on Pattern Analysis and Machine Intelligence, 2008
[57] S. Oh, S. Russell, and S. Sastry, “Markov chain monte carlo data association for multi-target tracking,” IEEE Transactions on Automatic Control, Vol. 54, No. 3, Mar. 2009
[58] R. Karlsson and F. Gustafsson, “Monte carlo data association for multiple target tracking,” Target Tracking: Algorithms and Applications, Vol. 1, Oct. 2001
[59] T. D. Vu and O. Aycard, “Laser-based detection and tracking moving objects using data-driven markov chain monte carlo,” IEEE International Conference on Robotics and Automation, Kobe, Japan, May 2009
```

#### A. Greedy Nearest Neighbor \(GNN\) Filter

간단하면서도 많이 쓰이는 방법은 GNN이다. `The simplest and probably the most widely applied data association approach is the Greedy Nearest Neighbor Filter.`

The NNF(Nearest Neighbor Filter??) only takes into account the prediction of the existing track from the last frame and the new observation obtained from the sensor. For each new data set, every segment is assigned to the nearest previous track after predicting motions for previous tracks from previous measurements to the current measurement. 

[61]에서는 Range image를 이용하여서 군중들 추적을 연구 하였다. 사람의 위치를 **general dynamic model**로 표현할수 없어 **greedy nearest neighbor filter**를 이용 하였다. `Prassler, et al. introduced a people tracking system for a crowded scene [61]. Considering that a person’s location cannot be easily characterized by a general dynamic model, the algorithm had to rely almost entirely on the greedy nearest neighbor filter to track people for consecutive range images. The result indicate that this system could track 5 to 30 moving objects in real-time (Figure 2.14).`

![](https://i.imgur.com/d1DfkrB.png)

대부분의 레이져 기반 차량 추적 시스템에서는 **Mahalanobis distance as the proximity metric for GNN**를 채택하고 있다. `Most laser-based vehicle tracking systems usually apply the Mahalanobis distance as the proximity metric for GNN. The Mahalanobis Distance was introduced by P.C.Mahalanobis in 1936. `

이 방법은 유클리드 거리 방식대비 geometric정보를 사용하기에 채택 되었다. `This measurement is used instead of Euclidean distance because it considers the geometric information implicit in:`
- that they primarily move forward and not side-to-side.

![](https://i.imgur.com/T2yhrF7.png)


차량은 앞뒤로 만 움직이며 긴 좌표(거리)방향이 필요 하므로 적합하다. `For the 2-dimensional data, since the covariance matrix represents the axis of the ellipse covering the distribution of the data, it can represent the direction and size of the entire data cluster. For points with the same mean value of the Euclidean distances to every point in the data cluster, the ones near the long axis direction have a small Mahalanobis distance versus the ones near the short axis direction. This property is very useful for finding the corresponding feature points of a vehicle since the vehicle always tends to move towards the long axis direction. `


두 Metrics에 대한 비교는 아래에 설명 되어 있다. `The difference between the two metrics is illustrated by the following example.`

![](https://i.imgur.com/t5iQ6TS.png)

위 그림에서 원은 과거의 측정값(Measuremet)이고 start는  현재의 **two probable locations**이다. `In Figure 2.15, the circle points are a measurement of a vehicle in the last frame, and the star points represent two probable locations of the vehicle in the current frame. `

유클리드 방식으로는 A의 값이 더 작다. 차량으로써는 갑자기 방향을 바꿀수 없기 때문에 B가 더 현실적이다. `If the Euclidean distance is used to measure the distance, the Euclidean distance between the probably positions and the data set are D_e(A) = 252.5 and D_e(B) = 361.4. However, as a vehicle, point B should have a higher probability of being the next cluster position than point A since it´ s nearly impossible for a vehicle to suddenly move laterally. `

마할라노비스 방식으로는 B가 더 작다. `If the Mahalanobis distance is applied, then the distances become D_m(A) = 77.9 and D_m(B) = 6.6,`
- which illustrates that the association along the long axis of a data cluster tends to generate trajectories consistent with expected vehicle motion. 

비슷하게, 만약 궤적을 알고 있다면 마할라노비스 거리 방식은 가중치를 줄수 있다. `Similarly, if the trajectory of a cluster is known, the Mahalanobis distance metric can use weightings aligned with the expected trajectory. `

따라서 마할라노비스 방식이 더 좋다. `Thus, the Mahalanobis distance better associates new measurements to a vehicle projected along its probable path, not to vehicles that happen to be close to the new measurement but whose paths are not nearby the measurement [38].`

본 논문에서는 허용할만한 에러를 보이고 간단한 구현 방식으로 GNN 필터를 사용 하였다. `In this thesis, for easiness implementation with acceptable error, GNN filter is applied as data association approach.`



#### B. Joint probabilistic data association \(JPDA\)

복합한 환경에서의 다중 물체 추적시 발생 가능한 association의 불확실성을 제거 하기 위해서는  JPDA가 모든 가능한 연관 확률 값을 고려하므로 유용하다. `To eliminate association ambiguity in complex scenes, especially for multi-objective tracking, JPDA is a data association algorithm that takes into account every possible association. `

이 방식은 센서로 탐지된 위치와 기존 Track간의 가능성을 **Bayesian estimate **을 계산하여 **hypothesis matrix **형태로 저장 한다. 이후 가장 높은 확률값을 가지고 있는것을 할당한다.  ` It computes the Bayesian estimate of the correspondence between segments detected by sensor and possible existing tracks and forms a hypothesis matrix including all possible associations. The assignments with highest probability are picked out. `

 [45, 62]에서는 최초로 JPDA를 적용하여 가능성을 보였다. `As an example of JPDA, Schulz applied sample-based JPDA in laser-based tracking system at first and showed its effectiveness for the multiple people tracking problem (Figure 2.16) [45, 62]. `

![](https://i.imgur.com/S3hU37H.png)


[46]에서는 JPDA를 수정한 방식을 제안 하였다. `By modifying JPDA to separate highly-correlated target-path combinations from poorly-correlated combinations, Frank, et al. proposed two extended JPDA approaches and tested them off-line (Figure 2.17) [46]. `

![](https://i.imgur.com/pH29bxg.png)


DARPA 2007 에서 JPDA가 이용되었다. `The robot in DARPA 2007 challenge, Junior, mentioned earlier in Section 2.2.2.2, also used this JPDA approach.`



#### C. Multiple Hypothesis Tracking \(MHT\)

### 2.5 Approaches for occlusion\(맞물림\) handling

사람이 많은 곳에서 서로 가려짐이 발생할경우 추적이 어렵게 된다.

The temporary occlusion of the objects may lead to mismatch when they return to view.

To maintain estimates of the tracks during occlusion, it is critical to have an estimate of the motion model of the objects during occlusion \[64\].

```
[64] R. Rosales and S. Sclaroff, “Improved tracking of multiple humans with trajectory prediction and occlusion modeling,” IEEE CVPR Workshop on the Interpretation of Visual Motion,1998
```

해결 방법 : 이전장에 언급한 filltering알고리즘에 기능을 추가 하여 해결 가능 `To some extent, this occlusion problem can be resolved using some of the previously-mentioned filtering methods necessary in laser-based tracking systems.`

* 움직임 예측 가능 : 칼만 필터 
* 움직임 예측이 불가능 함\(칼만필터기반\) : IMM\(Interactive Multiple Model\), 일반적으로 사용됨 
  * runs several Kalman Filters with different models parallel and merge the outputs to predict the positions \[22, 28, 42, 52\].
* 움직임 예측이 불가능 함\(확률 기반\) : 파티클 필터 
  * probability based model in a particle filter can deal with complex dynamic motion

#### A.Explicit model with Kalman Filter

#### B. Probability model with particle filter

#### C. Interacting Multiple Models \(IMM\)

#### D. Dynamic Model

## 3. System Setup

> 장비 구성 및 테스트 환경

## 4. Methodology

![](https://i.imgur.com/B6ndvLa.png)

1. Each frame of the LIDAR data is pre-processed to remove noise and to extract foreground information from background information.

2. Next, the data is separated into different segments and each segment is classified and labeled as either belonging to a vehicle or a pedestrian.

3. During the data association step, the system tries to match an object to the  
   nearest segment with the same label

   * If a new segment cannot be associated to a previously-existing object, it is stored for a short period. Such stored segments are marked as missing and deleted if no prior or subsequent matches are found for several frames.
   * If a segment is not found to match a new object, and the object persists for a specific time interval, it is marked as a new object and added to the object list for future tracking.

4. Once the association is completed, each object’s position is predicted and correlated with incoming data using Kalman filters.

5. Objects remaining within a small boundary for a long time interval are marked as stationary.

6. If any object is marked as missing during the association step, its new position will be estimated by the dynamic model.

7. Finally, the system obtains a new data frame and repeats the processing loop again.

### 4.1 Pre-processing

background subtraction is very useful

* to separate the foreground and background, and 
* to remove noise

an occupancy grid map \[25\] is employed to detect moving objects.

```
[25] A. Elfes, “Occupancy grids : a probabilistic framework for robot percpetion and navigation,” PhD thesis, Carnegie Mellon University, 1989
```

To form the occupancy map,

1. the field of the LIDAR view is separated into 40cm×40cm grids, which is empirically selected.

2. At any time frame,

   * if a grid is occupied by the segments detected by sensor, its corresponding value will be increased by 1 and 
   * if no segment detected in this gird, the value will be decreased by 1. 

3. After a reasonable time interval, the grids representing stationary obstacles will have a relatively high value and any segments in these grids will be regarded as stationary objects.

4. The stationary map is formed by all the grids with high enough value.

> In the experiment without temporary static objects, the map tends to be stable after 80 to 100 frames.

```python
# table after 80 to 100 frames.
mapUpdate(oldM ap, D)
addMap = same size to oldMap with all grids are 1

for each beam j with valid value in data set D
    the value of corresponding grid of addMap plus 1

for the value of each grid V_m of the addMap
    if V_m >= 1
        V_m = 1

newMap = oldMap + addMap
return newMap
```

### 4.2 Segmentation and classification

1. Once the noise is removed from the raw data,

2. LIDAR scan is separated into different segments and classified by the similar rule-based classification method applied by Nashashibi \[36\].

3. After classifying, every segment will be marked by feature points for the convenience of data association.

```
[36] F. Nashashibi and A. Bargeton, “Laser-based vehicles tracking and classification using occlusion reasoning and confidence estimation,” IEEE Intelligent Vehicles Symposium, Eindhoven, The Netherlands, June 2008
```

![](https://i.imgur.com/nZE9cMP.png)

#### A. Segmentation

DBSCAN사용

#### B. Partially occluded object detection

본 논문에서는 \[38, 36\]연구를 참고 하여, the partial occlusion can be detected by the distance information and classified as follows :

* Occluded one endpoint;
* Occluded both endpoints;
* Occluded middle part;

```
[38] F. Fayad and V. Cherfaoui, “Tracking objects using a laser scanner in driving situation based on modeling target shape,” IEEE Intelligent Vehicle Symposium, Istanbul, Turkey, June 2007
[36] F. Nashashibi and A. Bargeton, “Laser-based vehicles tracking and classification using occlusion reasoning and confidence estimation,” IEEE Intelligent Vehicles Symposium, Eindhoven, The Netherlands, June 2008
```

```python
# The pseudo-code of the occlusion detection is:
occlusionDetec(S, D)
for each segment i in S
    for each beam j in the 5 neighbor beams outside the endpoints of i
        if the distance of j is less the distance of corresponding endpoint
            D(j) = 1
return D
```

#### C. Object classification and feature extraction

룰 기반 분류 `Similar to the method mentioned by Nashashibi [36], a rules-based classification is performed to distinguish pedestrian and vehicles:`

* Segments with width less than 80cm are pedestrian;
* Segments with width larger than 80cm are vehicle candidates and will be fitted to **line shape** or **L shape**;
* L-shaped segments with both sides less than 80cm and no occlusion detection are vehicle.

#### D. Line and “L-shape” classification

The corner point is found by searching the distance between the points and the line formed by the two endpoints.

가장 멀리 있는 포인트가 코너 포인트가 됨`The farthest point will be regarded as the corner point.`

After that, the segment will be separated into two parts and the **weighted line fitting**  will be applied to each part.

If the angle between the two lines is

* less than 45 degree, this segment will be marked as **line**. 
* Otherwise, it will be marked as **L-shape**. 

When the number of the points of either part is less than 3,

* the segment will be marked as **line** since the information of this part is too vague to determine a feature.

![](https://i.imgur.com/2sIXbRf.png)

```python
LlDistinguish(i)
for each point n in segment i
    d(n) = distance between n to the line formed by two endpoints
mark the point with greatest d as potential corner point C
divided i into two sides A, B based on C
if sizeof (A) <= 3 or sizeof (B) <= 3
    mark i as a line
    return
else
    lineA = weightedlineFit(A)*
    lineB = weightedlineFit(B)*
    a = angle between lineA and lineB
    if a < 45
        mark i as a line
    else
        mark i as a L shape
(* this part is presented in following section)
```

#### E. Robust line fitting

#### F. Weighted line fitting

#### G. Corner fitting

#### H. Feature points calculation

The feature points are used to represent a vehicle depend on the shape and occlusion situation of the segment of points representing the vehicle.

##### 가. vehicle-classified object

For a vehicle-classified object, the object may be in the shape of a line, an “L”, or be uncertain.

For a line segment, the two endpoints can serve as feature points.

For this line representation, any occluded endpoints are deleted.

For an “L-shaped” object, the corner point will also be included as a feature.

If no feature point is detected, the mean point of the whole segment will be counted as the feature point.

##### 나. For pedestrian-classified objects

For pedestrian-classified objects, the feature point is represented by the mean point of the segment.

Figure 4.13 illustrates some examples of the feature points calculation.

![](https://i.imgur.com/8sAxXyf.png)

### 4.3 Data association

Greedy Nearest Neighbor \(GNN\) 알고리즘 사용

* the Mahalanobis distance is used to measure the distance instead of Euclidean
  distance 
* because it\(=Mahalanobis distance\) takes into account the distribution of the data

![](https://i.imgur.com/qw7nNeg.png)
![](https://i.imgur.com/s46YsoT.png)

#### A. GNN data association

![](https://i.imgur.com/TA6CjPP.png)

전 프레임에서 탐지된 물체는 새로 분류된 클러스터와 nearest match를 위해 비교 작업을 수행 한다. `Every existing object in the last frame is compared to all the new classified clusters to find the nearest match. `

이 알고리즘은 Miss match를 방지 하기 위해 Oldest Object에 우선권을 부여 한다. `The algorithm gives priority to the oldest objects first, which helps to eliminate the accidental miss matching caused by noise or wrong classification of dark objects (Figure 4.15).`

위 그림은 동일한 물체에 대한 두개의 연속적인 프레임이다. (빨간원) `Shown in Figure 4.15 are two consecutive frames of the same object (red circles).` 

검은 십자가는 차량의 4개 코너 특징점이다. `The black crossings are the four corner feature points of the vehicle.`

센서의 오차로 오른쪽 앵글의 두 모서리에 gap이 발생 하였다. `Since the error or accuracy of the sensor, there is gap between two edges of the right angle. `
- 첫 프레임에는 오류가 있지만 `In the first frame (shown at left), the gap is notable such that the DBSCAN regarded them as different clusters and association step marks the horizontal edge as an “adding” object (blue circles). `
- 두번째 프레임은 Older에게 할당한 우선순위로 오류가 수정 되었다. `In the new frame (shown at right), the object is tracked successfully again because the older objects have priority to match to the segments. `

The artificial “added” object is deleted since it does not last for enough time

#### B. Feature point association

특징점 기반 연동을 위해 물체의 새 postion과 speed가 계산 된다. `To form a feature point association, the new position and speed of the object`
- are calculated by determining the feature correspondence between measured data and previously associated object. 

본 연구에서는 두 종류의 특징점이 활용 되었다. `In this study, there are two kinds of feature points considered: `
- feature points representing the mean position of the segment, and corner figure points.

앞장에서 기술 하였듯이 **feature points**는 분류 단계에서 계산 된다. `As discussed in section 4.2.8, the feature points of the segment are computed during the classification step.`

물체는 feature points를 계속 유지 하며, 새 feature points가 생기면 추가 한다. `The object will keep the feature points of prior data after successful matching, and the object will add new feature points when the segment is associated with more information from new data. `

결과적으로 모든 물체는 4개의 모서리 feature points를 가지고 있어야 한다. `Eventually, every object should have four corner feature points unless the segments it matched cannot provide enough information.`

불충분한 정보를 가지게 되는 하나의 예로 차량에 가려지는 것이다. `One example of where there is insufficient information would be when the vehicle is scanned from the frontal direction such that the back of the vehicle is never visible. `

![](https://i.imgur.com/Z6gHubO.png)

Figure 4.16 is an example of data association of 20 consecutive frames. 

검은점은 그리드맵기법으로 배경 제거가 된것이다. 빨간점은 센서가 탐지한 것이고 검은 십자가는 차량의 feature point이다. `The black dots are background extracted by grid map approach, the red dots the segments obtained by sensor and the black crossing the feature points of the vehicle object.`

숫자는 순서로 추적기에서 작동으로 계산 된다.  정답값은 foreground의 normal maked 물체를 카운팅 하여서 계산 한다. `The number is calculated automatically by the tracking system, which represents the order of the object (including the noise object with short survival time and the obstacle on background). The true number of the moving objects can be computed by counting the normal marked objects on foreground.`

![](https://i.imgur.com/XttkmSv.png)
Figure 4.17 is another example of data association. Black star represents the feature point of the pedestrian object. It indicates the algorithm can deal with the vehicle object and pedestrian object at the same time. The feature statistics such as the mean and corner positions will obviously change as the object moves through the scanning field, and sometimes these changes are quite abrupt. To make the model of the vehicle stable, the length of the edge is computed by averaging all the corresponding history valid edge values. In other words, the feature “remembers” the extent of previous vehicle scans and uses this information to improve the expected mean and corner positions of the vehicle. 


![](https://i.imgur.com/nsVkVgS.png)
Figure 4.18 shows the lengths of two sides of the vehicle 15 in previous Figure 4.17 versus the time.


### 4.4 Model motion

Similar to Lim used in visual tracking system \[83\], a general time series dynamic model is applied in this work to predict the state of targets during occlusion.

```
83] Hwasup Lim, “Dynamic motion and appearance modeling for robust visual tracking, ” PhD thesis, The Pennsylvania State University, 2007
```

#### A. Dynamic modeling for motion model

In this thesis, the motions considered include at least

* constant velocity, 
* constant acceleration,
* geometric turning.

## 5. Experimental Results

## 6. Conclusion



