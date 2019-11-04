# A portable three-dimensional LIDARbased system for long-term and widearea people behavior measurement


https://journals.sagepub.com/doi/pdf/10.1177/1729881419841532


## 1. Introduction


## 2. Related work


## 3. System overview

## 4. Offline environmental mapping


## 5. Online people behavior measurement

### 5.1 Sensor localization

### 5.2 People detection and tracking

먼저 foreground points 추출을 위해서 배경을 제거 하였다. `We first remove the background points from an observed point cloud to extract the foreground points. `

이후, 복셀사이즈 0.5m로 occupancy grid map를 생성 하였다. `Then, we create an occupancy grid map with a certain voxel size (e.g.0.5 m) from the environmental map. `

센서 자세 추정을 통해 지도 좌표계로 입력된 포인트 클라우드를 변경 하였다. 이후 environmental map에 포함된 포인트를 배경으로 간주 하고 제거 하였다. `The input point cloud is transformed into the map coordinate according to the sensor pose estimated by UKF, and then each point at a voxel containing environmental map points is removed as the background. `

후보 사람 영역 클러스터링을 위해 유클리드 군집화 기법을 적용 하였다. `The Euclidean clustering is then applied to the foreground points to detect human candidate clusters.`

하지만, 근접된 사람들은 하나의 클러스터로 구분 되는 문제가 있다. `However, in case persons are close together, their clusters may be wrongly merged and are detected as a single cluster. `

문제 해결을 위해 **Haselich’s split merge clustering**알고리즘을 적용 하였다. `To deal with this problem, we employ Haselich’s split merge clustering algorithm.`[깃북 정리](https://legacy.gitbook.com/book/adioshun/paper-3d-object-detection-and-tracking/edit#/edit/master/Tracking/2014-confidence-based-pedestrian-tracking-in-unstructured-environments-using-3d-laser-distance-measurements.md?_k=4bfxq2)


```
Confidence-Based Pedestrian Tracking in Unstructured Environments Using 3D Laser Distance Measurements
```

**Haselich’s split merge clustering**알고리즘은 클러스터를 서브-클러스터로 특정 쓰레쉬홀드까지 **dp-means**를 이용하여 작게 나눈다. `The algorithm first divides a cluster into subclusters until each cluster gets smaller than a threshold (e.g. 0.45m) by using dp-means so that every cluster does not have points of different persons. `

만약 서브-클러스터 사이에 간격이 없다면 하나의 사람으로 간주되고 머징된다. `Then, if there is no gap between those subclusters, the clusters are considered to belong to a single person and remerged into one cluster. `

![](https://i.imgur.com/wtzwEPu.png)

Figure 10 shows an example of the detection results. 

위 방식을 통해 근접한 두 사람도 잘 나뉘게 된다. `The person clusters are correctly separated even when they are very close together thanks to the split and the remerge process`


탐지된 클러스터는 사람이 아닌것도 포함될수 있다. `The detected clusters may contain nonhuman clusters (i.e. false positives).`

사람인지 아닌지 분류 작업을 **Kidono**의 알고리듬을 이용하여 진행 한다[깃북정리](https://legacy.gitbook.com/book/adioshun/paper-3d-object-detection-and-tracking/edit#/edit/master/2011-pedestrian-recognition-using-high-definition-lidar.md?_k=1m2iwy)
. `To eliminate nonhuman clusters among detected clusters, we judge whether a cluster is a human or not by using a human classifier trained with slice features by Kidono et al. and Schapire and Singer.`

```
Pedestrian Recognition Using High-definition LIDAR
```

사람이 지표면위에 걸어 다닌다는 가정 하에 높이 정보 상관 없이 xy로 추적을 실시 한다. `Assuming that persons walk on the ground plane, we track persons on the XY plane without the height. `

추적을 위해서 아래 방법들을 사용 하였다. `We employ the combination of Kalman filter with the constant velocity model and global nearest neighbor data association to track persons. `
- Kalman filter
- constant velocity mode
- global nearest neighbor data association

제안된 기법은 잘 동작 한다. `The tracking scheme works well as long as the tracked persons are visible from the sensor and are correctly detected.`




---

# [hdl_people_tracking](https://github.com/koide3/hdl_people_tracking)



hdl_people_tracking is a ROS package for real-time people tracking using a 3D LIDAR. 
- 클러스터링 : It first performs Haselich's clustering technique to detect human candidate clusters, 
    - Confidence-Based Pedestrian Tracking in Unstructured Environments Using 3D Laser Distance Measurements
    - [깃북 정리](https://legacy.gitbook.com/book/adioshun/paper-3d-object-detection-and-tracking/edit#/edit/master/Tracking/2014-confidence-based-pedestrian-tracking-in-unstructured-environments-using-3d-laser-distance-measurements.md?_k=4bfxq2)
- 분류 : and then applies Kidono's person classifier to eliminate false detections. 
    - Pedestrian Recognition Using High-definition LIDAR
    - [깃북정리](https://legacy.gitbook.com/book/adioshun/paper-3d-object-detection-and-tracking/edit#/edit/master/2011-pedestrian-recognition-using-high-definition-lidar.md?_k=1m2iwy)
- 추적 : The detected clusters are tracked by using Kalman filter with a contant velocity model.


