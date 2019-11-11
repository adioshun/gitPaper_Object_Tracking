
# [Online learning for 3D LiDAR-based human detection: experimental analysis of point cloud clustering and classification methods](https://link.springer.com/epdf/10.1007/s10514-019-09883-y?author_access_token=bVLnE4rWjkyUnk8WopA0Lfe4RwlQNchNByi7wbcMAY6zHIY15ykgJsK70R8O7eQrMr2yHIZQSiyxe3OktHw_9R1puJtMefwAs4tGo2L7ytrEzPSDTxHtSdjXNYkRozK46fQM7ZPLOgSknycKxSoIsA%3D%3D)

> https://rdcu.be/bODuU

Abstract - This paper presents a system for online learning of human classifiers by mobile service robots using 3D LiDAR sensors, and its experimental evaluation in a large indoor public space. The learning framework requires a minimal set of labelled samples (e.g. one or several samples) to initialise a classifier. The classifier is then retrained iteratively during operation of the robot. New training samples are generated automatically using multi-target tracking and a pair of “experts” to estimate false negatives and false positives. Both classification and tracking utilise an efficient real-time clustering algorithm for segmentation of 3D point cloud data. We also introduce a new feature to improve human classification in sparse, long-range point clouds. We provide an extensive evaluation of our the framework using a 3D LiDAR dataset of people moving in a large indoor public space, which is made available to the research community. The experiments demonstrate the influence of the system components and improved classification of humans compared to the state-of-the-art. 

## 1 Introduction

사람탐지는 로봇에서 중요하다. 사람인지아닌지 구분하기 위하여 추적과 경로(tracking and trajectory)는 중요한 단서이다. `Human detection is a key feature of mobile service robots operating in public and domestic spaces. Besides obvious safety requirements, distinguishing between humans and inanimate objects provides extra information for the robot to plan and adapt its next movement in the environment. Typically, the robot observes the surrounding area through on-board sensors and estimates the location of all the relevant static and/or moving objects within range. In order to recognize whether these objects are humans or not, cues from tracking and trajectory analysis can be exploited for classification purposes.`

사람탐지 센서로의 RGB-D & 2D LIDAR의 제약  `Most human detection and tracking systems for service robots must deal with both hardware limitations and changing environments. RGB-D cameras can provide colour information and dense point clouds, but their sensing range is usually just a few meters and their field-of-view is usually less than 90◦ in both horizontal and vertical directions. A SICK-like 2D LiDAR is capable of sensing ranges of several tens of meters but it is difficult to extract useful information from the sparse distribution of points obtained.`

환경 변화가 큰 곳에서 특징 기반 분류기는 재학습이 필요 하다. 라벨링 작업에는 많은 비용이 든다. `In response to changing environments, a feature-based classifier usually needs to be re-tuned and often tediously retrained to achieve acceptable performance in new scenarios. An approach to eliminate this dilemma is to train a generalized classifier. However, the latter requires a large amount of labelled data, usually associated with high labor costs.`

3D LIDAR의 장단점 `The application of 3D LiDAR sensors in robotics and autonomous vehicles has grown dramatically in recent years, either used alone or in combination with other sensors, including their use for human detection and tracking. An important specification of this type of sensor is the ability to provide long-range and wide-angle laser scans. In addition, it produces point clouds that become more sparse as the distance increases, but which usually are very accurate and not affected by lighting conditions. `

사람탐지 센서로의 3D LIDAR의 문제점 `However, human detection in 3D LiDAR scans is still very challenging, `
- 초근접/원거리 탐지 불가 `especially when the person is too close or too far away from the robot. `
- 계산 부하 `Moreover, since increasing the sensing range increases also the area under consideration, and therefore the number of people potentially within it, 3D LiDAR-based human detection can be computationally very expensive. `
- 학습 데이터 `Finally, a large number of manually-annotated training data was usually required by previous methods to learn, offline, a human classifier`. 
	- Unfortunately, labelling 3D LiDAR point clouds is tedious work and prone to human error, in particular if many variations of human pose, shape and size need to be correctly classified. 
	- Offline manual annotation is also not very feasible for complex real-world scenarios and where the same system needs to be retrained for different operational environments.

본 논문의 2017년 논문의 확장판이다. 추적기능 기반 온라인 학습이 가능하다. `This paper extends our recent work (Yan et al. 2017) on human classification, in which we developed a framework for online learning to classify humans in 3D LiDAR scans, taking advantage of a suitable multi-target tracking system (Bellotto and Hu 2010) (see Fig. 1). `

```
Yan, Z., Duckett, T., & Bellotto, N. (2017). Online learning for human classification in 3D LiDAR-based tracking. In In Proceedings of the 2017 IEEE/RSJ international conference on intelligent robots and systems (IROS). Vancouver, Canada

Computationally efficient solutions for tracking people with a mobile robot: An experimental evaluation of bayesian filters N Bellotto, H Hu Autonomous Robots, 2010
```

온라인 러닝은 에러에 영향을 받기 쉽다. 따라서 2012의 논문을 참고 하여 N/P-expert를 도입 하였다. `Since online learning methods do not rely on explicit human supervision, they are affected by errors. In order to deal with the latter, and inspired by pre-vious solutions for tracking-learning-detection (Kalal et al. 2012), we proposed an online learning approach based on two types of “experts”, one converting false negatives into new positive training samples (P-expert), and another one correcting false positives to generate new negative samples (N-expert). We then showed an improvement in the performance of the human classifier by iteratively correcting its errors.`


기존 연구대비 새로운 점 `Compared to Yan et al. (2017), the current paper includes several new contributions.`
- Firstly, 분류기 성능 개선 :we improve the 3D LiDAR-based human classification with a new low-cost feature, coupling human profiles in point clouds with distance changes, hence enhancing the sensitivity of the classifier to samples far away from the robot. 
- Secondly, 데이터셋 비교 분석 : we provide a detailed comparison of our recent 3D LiDAR dataset to other popular ones for pedestrian detection as a guideline for researchers in this area. 
- Thirdly, 클러스터링 알고리즘 성능 평가 수행 : we perform a thorough evaluation of our clustering algorithm for 3D LiDAR point clouds, including a comparison with other state-of-the-art solutions, covering both precision and runtime performance. 
- Finally, 분류 성능 평가 수행 : we extend significantly the performance evaluation of our approach for human classification, including a detailed stability analysis of the online learning process.

논문 구성 `The remainder of this paper is organized as follows. `
- Section 2 gives an overview of related work, in particular on 3D point-cloud-based segmentation and human classification. 
- Then, we introduce our online framework in Sect. 3 and the relation between its modules, including those for human tracking and classification. 
- The former is presented in Sect. 4, including a detailed description of the clustering algorithm. 
- The actual learning process for human classification is explained in Sect. 5, which clarifies the link between the P–N experts and tracking. 
- Section 6 presents our publicly available 3D LiDAR dataset and a comparison to the existing ones, 
- while Sect. 7 provides a comprehensive experimental evaluation of the system performance, including clustering and human classification. 
- Finally, conclusions and future research are discussed in Sect. 8.

## 2 Related work

3D LIDAR기반 사람 탐지는 센서의 제약으로 어렵다. `Human detection and tracking for mobile service robots have been widely studied in recent years, including solutions for RGB and RGB-D cameras, as well as 2D  and 3D LiDARs. Although the latter provides accurate distance information, one of the main challenges working with LiDARs is the difficulty of recognizing humans from a relatively sparse set of points and without additional color information.`

전통적 추적을 위한 인지 시스템 파이프라인 `The traditional perception pipeline for object tracking consists of several stages, typically including`
- segmentation (e.g. clustering), 
- feature extraction, 
- classification, 
- data association, 
- and position/velocity estimation. 

딥러닝 기반 추적 기술 등장 `However, emerging methods like end-to-end learning (typically deep learning) provide alternative frameworks for tracking applications.`

기존 딥러닝 기반 추적 기술들 ` For example,`
- Dequaire et al. (2017) employed a recurrent neural network (RNN) to capture the environment state evolution with 3D LiDAR (the 3D point clouds were reduced to a 2D scan), training a model in an entirely unsupervised manner, and then using it to track cars, buses, pedestrians and cyclists from an autonomous car. 
- Zhou and Tuzel (2018) developed another end-to-end network that divides the 3D point cloud into a number of voxels. After random sampling and normalization of the points, several Voxel Feature Encoding (VFE) layers are used for each non-empty voxel to extract voxelwise features. This network is trained to learn an effective discriminative representation of objects with various geometries. 

딥러닝 기술이 좋지만, 본 논문의 온라인 학습 기반 방식에는 적용 하지 않았다. 여러 계산 부하로 실생활에는 적합하지 않기 때문이다. ` Despite encouraging results in this field, we have not integrated deep learning-based approaches into our online framework because such methods typically require considerable fine-tuning with manual intervention, longer training times, and special hardware requirements (e.g. GPU and power supply), making it difficult to meet the requirements of autonomous mobile robots in real-world applications.`

본 논문의 인지 시스템은 아래 아이디어들을 활용 하였다. `The framework described in this paper involves a multistage perception pipeline, where the most relevant methods for 3D point cloud segmentation utilise the scan line run (Zermas et al. 2017), depth information (Bogoslavskyi and Stachniss 2016), and Euclidean distance (Rusu 2009). `

```
Zermas, D., Izzat, I., & Papanikolopoulos, N. (2017). Fast segmentation of 3d point clouds: A paradigm on lidar data for autonomous vehicle applications. In Proceedings of the 2018 IEEE international conference on robotics and automation (ICRA). Singapore
Bogoslavskyi, I., & Stachniss, C. (2016) Fast range image-based segmentation of sparse 3d laser scans for online operation. In Proceedings of the 2016 IEEE/RSJ international conference on intelligent robots and systems (IROS) (pp. 163–169)
Rusu, R.B. (2009). Semantic 3D object maps for everyday manipulation in human living environments. Ph.D. thesis, Computer Science department, Technische Universitaet Muenchen, Germany

```
### 2.1 군집화 

The **run-based segmentation** includes two steps which
- 바닥제거 : first extracts the ground surface in an iterative fashion using deterministically assigned seed points, 
- 물체 군집화  : and then clusters the remaining non-ground points using a two-run connected component labelling technique from binary images. 

**Depth clustering** is a fast method with low computational demands, 
- which first transforms 3D LiDAR scans into 2D range images, 
- then performs the segmentation on the latter. 

The **Euclidean method**, instead, clusters points by calculating the distance between any two of them directly in the 3D space. 

위 3가지 군집화 방법에 대하여 성능 평가를 하였다. `Our clustering method is compared against these three alternatives in Sect. 7.1.`

### 2.2 분류 

가장 일반적인 방법은 학습데이터를 오프라인러닝한후 적용하는 것이다. `Regarding point-cloud-based human classification, a very common approach is to train a classifier offline, under human supervision, and then apply it to sensor data during robot operations. `

대표적 연구는 아래와 같다. `For example,`
- Navarro-Serment et al. (2009) introduced **seven features** for human classification and trained an SVM classifier based on these features. 
- Kidono et al. (2011) proposed **two additional features** considering the 3D human shape and the clothing material (i.e. using the reflected laser beam intensities), showing significant classification improvements. 
- Häselich et al. (2014) **implemented eight of the abovementioned features** for human detection in unstructured environments, discarding the intensity feature due to a lack of hardware support. 
- Li et al. (2016) **implemented instead a resampling algorithm** in order to improve the quality of the geometric features proposed by the former authors. 
- Wang and Posner (2015) applied a **sliding window approach to 3D point data for object detection**, including humans. 
	- They divided the space enclosed by a 3D bounding box into sparse feature grids, 
	- then trained an SVM classifier based on six features related to the occupancy of the cells, the distribution of points within them, and the reflectance of these points. 
- Spinello et al. (2011) combined a **top-down classifier**, based on volumetric features, and a **bottom-up detector** to reduce false positives for tracking distant persons in 3D LiDAR scans. 
- An alternative approach by Deuge et al. (2013) introduced an **unsupervised feature learning** approach for outdoor object classification by projecting 3D LiDAR scans into 2D depth images.

본 연구에서는 2D LIDAR를 사용하고 있다. 이와 관련된 연구들도 있다. `Although our paper focuses on a different sensor, the use of 2D LiDARs in human detection and tracking is worth mentioning. Often, this type of sensor is used for robot navigation, including localization and safe obstacle avoidance, and it is therefore installed on the lower front side of the mobile robot, close to the ground. Some researchers have exploited this configuration to perform human tracking by detecting human legs in 2D range data (Arras et al. 2007; Bellotto and Hu 2009; Luber and Arras 2013; Leigh et al. 2015; Linder et al. 2016; Beyer et al. 2018).`

오프라인 학습 방식은 로봇이 새로운 공간에 가면 파인튜닝이나 재 학습이 필요하다는 것이다. `The inconvenience with offline methods is that the pretrained classifier is not always effective when the robot moves to a different environment, and fine-tuning or retraining are typically required as a consequence.`

일부 연구에서는 적은양의 학습 데이터(`less annotation`)를 이용하는 방법이 제안 되었다. `To help relieve the user from these tedious tasks, some authors proposed methods requiring less annotation.`
-  For example, Teichman and Thrun (2012) presented a **semi-supervised learning** method for multi-object classification, which needs only a small set of hand-labelled seed object tracks for training the classifiers. However, it requires a large training set of background objects (i.e. with no pedestrians, cyclists, cars, etc.), provided by a human expert, in order to achieve good classification performance. 
- Other authors proposed **classifier-free methods**. Shackleton et al. (2010), for example, employed a surface matching technique for human detection and an **Extended Kalman Filter** to estimate the position of a human target and assist the detection in the next LiDAR scan. However, the method is suitable only for simple scenarios, and is further restricted to the case of moving humans. 
- Dewan et al. (2016) proposed a **classifier-free** approach to detect and **track** dynamic objects, which again relies on **motion cues** and is therefore not suitable for slow and static objects such as pedestrians.


적은양의 학습 데이터나 분류기 없이 동작 하는 방식이 제안 되었지만 학습데이터는 여전히 중요 하다. 아쉽게도 3D LIDAR 학습 데이터는 별로 없다. 이에 대한 비교 작업을 6장에서 진행 하였다. `Despite the need for low-annotation or classification-free approaches for human detection, good datasets are important for validation and comparison of new methods to previous ones. Unfortunately, unlike the abundance of datasets collected with RGB and RGB-D cameras, there are only a few 3D LiDAR datasets available to the scientific community, in particular for mobile service robots. Our dataset introduces a new combination of interesting properties, useful for 3D LiDAR-based human detection and tracking in large indoor environments, which are compared with the existing datasets in Sect. 6.`

최근 추세는 효율적인 **low-annotation** 방식을 사용하는 것이다. 우리의 제안 방식도 이런 추세를 따르고 있다. `To summarize, by looking at the current state-of-the-art, it is clear that an effective low-annotation method would be highly beneficial to detect humans in 3D LiDAR scans. Our work contributes to this need by demonstrating that human detection can be improved by combining tracking and online learning with a mobile robot, even in highly dynamic environments, and that such an approach provides comparable or superior results with respect to previous methods.`


## 3 General framework for online learning

![](https://i.imgur.com/hFpqWNB.png)

시스템 구성 `Our system consists of four main components:`
- a cluster detector for 3D LiDAR point clouds, 
- a multi-target tracker, 
- a human classifier and 
- a training sample generator (see Fig. 2). 

분류기는 최소한의 학습 데이터로 지도기반 학습을 시작한다. 추후 학습데이터는 추가 된다. `The classifier is initialised by supervised learning with a small set of human clusters. The size of this set could be as small as one, since more samples will be added incrementally and used for retraining in future iterations.`

온라인 학습 동작 과정 `The online process works as follows.`
- At each new iteration, a 3D LiDAR scan (i.e. 3D point cloud) is segmented by the **cluster detector**. 
- Positions and velocities of the clusters are estimated in real-time by a **multi-target tracking ** system. 
- These estimates are buffered in trajectory arrays together with their respective cluster observations. 
- At the same time, the classifier labels each cluster as ‘human’ or ‘non-human’.

샘플 생성기는 clusters, trajectories, labels 정보를 이용한다. 라벨 정보는 두 종류(false positive & false negative)의 에러에 영향을 미친다. 샘플생성기는 두개의 독립적인 EXPERT를 이용하여 이 에러를 보정 한다. EXPERT의 결과물에 따라 샘플생성기는 새 학습 데이터를 생성하여 지속적인 학습이 가능하다. `The sample generator exploits all the information about clusters, trajectories and labels. The latter are typically affected by two types of errors: false positive and false negative. The sample generator tries to correct them by using two independent “experts”, which cross-check the output of the classifier with that of the tracker. Based on the experts’ decisions, the sample generator produces new training data. In particular, the P-expert in Fig. 2 converts false negatives into positive samples, while the N-expert converts false positives into negative samples. When enough new samples have been generated, they are used to retrain the classifier, so the system can learn and improve from previous errors. The process and the experts are explained in detail in Sect. 5.2.`
- The P-expert converts false negatives into positive samples
- The N-expert converts false positives into negative samples.

온라인 학습 방식이 기존 TLD방식을 도입 하였지만 몇가지는 다른다. `Although conceptually similar to a previous tracking-learning-detection framework (Kalal et al. 2012), the proposed systems differs in three key aspects,`
- 차별점 #1 :  namely the independence of the tracker from the classifier, 
- 차별점 #2 :  the frequency of the training process, 
- 차별점 #3 : and the implementation of the experts. 

차별점 #1 : 특히, 분류기와 추적기가 독립적으로 동작 할수 있어 더 좋은 분류기로 교체도 가능하다. `In particular, while the performance of the human classifier depends on the reliability of the experts and the tracker, the latter is completely unaffected by the classification performance. This decoupling makes the system modular and potentially applicable to alternative classification methods. `

차별점 #2 : 기존 방식은 매 샘플단위로 재 학습을 하지만, 제안 방식은 배치 방식으로 재 학습을 진행 하여 **언더피팅** 문제를 해결 하였다. `Also, instead of retraining incrementally from single samples (i.e. frame-by-frame training), our system performs a less frequent batch-incremental training (Kalal et al. 2012) (i.e. gathering samples in batches to retrain the classifier after a certain period), collecting new data online as the robot moves in the environment. This feature can effectively prevent under-fitting in online learning. `

차별점 #3 : 구현된 Expert는 동시에 여러 타겟을 처리 할수 있으므로 속도면에서 좋다. `Finally, our implementation of the experts is specifically designed to deal with more than one target simultaneously, and therefore to generate new training samples from multiple detections, speeding up the learning process.`

```
[Read et al. 2012] Tracking-learning-detection Z Kalal, K Mikolajczyk, J Matas IEEE Transactions on Pattern Analysis and Machine Intelligence, 2012
```

## 4 Point cloud cluster detection and tracking

제안 시스템에서 두개의 모듈(탐지기/추적기)이 가장 중요 하다. 탐지기는 실시간으로 점군에서 물체를 구분해 낸다. 추적기는 실시간으로 위치/속도를 추론한다. 이때 탐지 물체가 사람/물체 인지는 고려 하지 않는다. `Two key components of the proposed system shown in Fig. 2 are the cluster detector and the multi-target tracker. The former detects, in real-time, clusters of point clouds from 3D LiDAR data. Their positions and velocities are estimated by the tracker, also in real-time, independently of whether the clusters belong to humans or not. Details about the two modules are provided next.`

### 4.1 Point cloud cluster detector

- 높이 정보를 이용하여 바닦제거
- 근거리/원거리별 다른 군집화 파라미터 사용 
- 사람보다 크거나 작은 물제 제거 

### 4.2 Multi-target tracker

추적시스템은 UKF + GNN을 이용하여 구현 되었다. `The multi-target tracker is based on an efficient Unscented Kalman Filter (UKF) implementation, using Global Nearest Neighbour (GNN) data association to deal with multiple clusters simultaneously (Bellotto and Hu 2009, 2010).`

2D 상에서 추적이 수행 되며, 3D 정보(높이??)는 고려 되지 않는다. `Human clusters are tracked on a 2D plane, corresponding to a flat floor, estimating horizontal coordinates and velocities with respect to a fixed world frame of reference. In the current implementation, the 3D cluster size is not taken into account,` although it could prove beneficial in more challenging tracking scenarios.

#### prediction step
The prediction step of the estimation is based on the following constant velocity model (Li and Jilkov 2003):

$$
\left\{\begin{array}{l}{x_{k}=x_{k-1}+\Delta t \dot{x}_{k-1}} \\ {\dot{x}_{k}=\dot{x}_{k-1}} \\ {y_{k}=y_{k-1}+\Delta t \dot{y}_{k-1}} \\ {\dot{y}_{k}=\dot{y}_{k-1}}\end{array}\right.
$$

- where x_k and y_k are the Cartesian **coordinates** of the target at time_tk, 
- xk yk are the respective **velocities**, 
- andΔt=tk−tk−1.
 

The position of a cluster is computed by projecting onto the (x, y) plane its centroid c_j , computed as follows:

$$
c_{j}=\frac{1}{\left|C_{j}\right|} \sum_{p_{i} \in C_{j}} p_{i}
$$

####  Update step

The update step of the estimation then uses a 2D polar observation model to represent the position of the cluster:

$$
\left\{\begin{array}{l}{\theta_{k}=\tan ^{-1}\left(y_{k} / x_{k}\right)} \\ {\gamma_{k}=\sqrt{x_{k}^{2}+y_{k}^{2}}}\end{array}\right.
$$

where θk and γk are the bearing and the distance, respectively, of the cluster from the sensor. 

Note that noises and coordinate transformations, including those relative to the robot motion, are omitted in the above equations for the sake of simplicity.

The choice of the above models and estimation technique is motivated by the type of sensor used. In particular, the polar observation model better represents the actual functioning of our LiDAR sensor (i.e. the raw measures are range values at regular angular intervals) and its noise properties. 

The **non-linearity** of this model leads therefore to the adoption of the **UKF**, which is known to perform better than a standard EKF (Julier and Uhlmann 2004; Bellotto and Hu 2010). 

>  polar observation model??이 UKF 사용과 관련이 있나?

이전 연구에서 제안 방식이 성능이 좋음을 증명 하였다. `Previous studies (Bellotto and Hu 2010; Linder et al. 2016) showed that our estimation framework is an effective and efficient solution to track multiple people with mobile robots. `


추적 시스템에 대한 더 자세한 내용은 아래 논문 참고 `More details about our track management solution (i.e. initialisation, maintenance, deletion) and possible application can be found in Bellotto and Hu (2009, 2010), Dondrup et al. (2015).`

Finally, the covariance matrices Q and R of the noises for the prediction and observation models, respectively, are the following (Li and Jilkov 2003):

$$
\mathbf{Q}=\left[\begin{array}{cccc}{\frac{\Delta t^{4}}{4} \sigma_{x}^{2} \frac{\Delta t^{3}}{2} \sigma_{x}^{2}} & {0} & {0} \\ {\frac{\Delta t^{3}}{2} \sigma_{x}^{2}} & {\Delta t^{2} \sigma_{x}^{2}} & {0} & {0} \\ {0} & {0} & {\frac{\Delta t^{4}}{4} \sigma_{y}^{2} \frac{\Delta t^{3}}{2} \sigma_{y}^{2}} \\ {0} & {0} & {\frac{\Delta t^{3}}{2} \sigma_{y}^{2} \Delta t^{2} \sigma_{y}^{2}}\end{array}\right] \mathbf{R}=\left[\begin{array}{cc}{\sigma_{\theta}^{2}} & {0} \\ {0} & {\sigma_{\gamma}^{2}}\end{array}\right]
$$

where the noise standard deviations σx , σ y , σθ and σγ were empirically determined to optimize the human tracking performance of our robot platform.

## 5 Online learning for human classification

클러스터된 물체는 추적기의 결과 정보와 함께 학습 데이터로 재 사용된다. `The point cloud clusters detected in Sect. 4.1 are analysed, in real-time, by a classifier distinguishing between humans and non-humans. Our online learning framework is an iterative process in which the classifier is periodically retrained, online, using old and new cluster detections, which are provided by the sample generator. The latter selects, based on tracking information, a pre-defined number of positive and negative clusters, and uses them to retrain the human classifier. The classification and sample generation processes are explained in detail in the following sub-sections.`

### 5.1 Human classifier

SVM을 이용하여 분류작업을 진행 하였다. 이 방식은 비선형에서도 동작 한다. `A standard Support Vector Machine (SVM) (Cortes and Vapnik 1995) is used for human classification. The SVM method has a solid theoretical foundation in mathematics, which is good at dealing with small data samples and therefore very suitable for our proposed online learning framework. Moreoever, this algorithm is known to work well in non-linear classification problems, and has already been applied successfully for 3D LiDAR-based human detection (Navarro-Serment et al. 2009; Kidono et al. 2011).`


학습을 위한 Features는 아래 표와 같다. 여러 연구에서 선별적으로 채택 하였다. `In order to train the SVM, we extract seven different features from each point cloud cluster, which are shown in Table 1. Features ( f 1 , . . . , f 7 ) were proposed by NavarroSerment et al. (2009). However, we discarded the last three because of their heavy computational cost and relatively low classification performance (Kidono et al. 2011), which make them unsuitable for real-time people tracking in large populated environments, and replaced them instead with three different features: f 8 , f 9 and f 10 . The former two, f 8 and f 9 were originally proposed by Kidono et al. (2011). The combination of these two features with ( f 1 , . . . , f 4 ) was shown to successfully characterise both standing and sitting people (Yan et al. 2017).`

![](https://i.imgur.com/kE1c1ql.png)

새 특징인 10은 **slice Distance**로 원거리의 sparse한 점군 처리를 위해 도입 되었다. `Feature f 10 , instead, is a new addition to our system, which aims at coupling the detection distance of the sensor with the 3D shape of humans, especially for long distance classification with a sparse point cloud. This new feature, called “slice distance”, is based on the 10 slices of Kidono et al. (2011) and is computed as follows:`

$$
f_{10}=\left\{\left\|c_{i}\right\|_{2} | c_{i}=\left(x_{i}, y_{i}, z_{i}\right) \in \mathbb{R}^{3}, i=1, \ldots, 10\right\}
$$

- where c_i is the centroid of each slice, which can be calculated as in Eq. (5). 

The purpose of the 10 slices was to extract the partial features of humans observed from a long distance, where the spatial resolution of the LiDAR’s point clouds decreases. 

The 3D points in the cluster are divided into 10 blocks, and the length and width of each block are computed as features (see Fig. 6).

![](https://i.imgur.com/LwMneJO.png)


The respective feature vector is the following: 

$$  f_{8}=\left\{L_{j}, W_{j} | j=1, \ldots, 10\right\} $$

7.3에서 위 특징을 이용한 성능 평가를 실시 하였다. `In Sect. 7.3 we will show that, by using our feature set, the classifier improves on the state-of-the-art.`

분류기 구현에 대한 정보들 `The full set of features from cluster C_j forms a vector ( f 1 , . . . , f 4 , f 8 , . . . , f 10 ), with 71 dimensions in total. At each iteration of the learning process, a binary SVM is trained to label human and non-human clusters based on these features. The software tool we use is LIBSVM (Chang and Lin 2011), setting the ratio of positive and negative training samples to 1:1, and scaling all the feature values within the interval [−1, 1]. The SVM uses a Gaussian Radial Basis Function kernel (Keerthi and Lin 2003) and outputs probabilities associated to the labels. Currently our software does not support incremental learning, so it stores all the training samples since the beginning and uses all of them to retrain the SVM at each new iteration. The training/retraining time is proportional to the number of samples. For our experimental configuration in Sect. 7, it takes from less than 1 millisecond to a few minutes. However, based on our released source code, one can easily decouple the training process from the online learning (e.g. by using independent threads) as needed, or fine-tune the k-fold cross validation (for finding optimal training parameters) to speed up the training process. Moreover, our framework for online learning allows for the implementation of different classifiers and training algorithms.`

### 5.2 Sample generator

분류기 재 학습을 위해서는 학습샘플이 필요 하다. 이는 샘플 생성기를 통해 수행되며 두개의 독립 모듈로 구성 되어 있다. `The training samples needed to retrain the human classifier are selected from the current cluster detections by the sample generator in Fig. 2. This is based on two independent modules: `
- a positive (P) expert and 
- a negative (N) expert. 

At each time step, 
- P모듈은 사람이 아닌 샘플중에서 잘못 분류된것을 찾아 사람으로 재 샘플링 한다. `the P-expert analyses the current clusters classified as non-humans to identify the potentially incorrect ones (i.e. false negatives). The latter are added to the training set as new positive samples. `
- N모듈은 사람인 샘플중에서 잘못 분류된것을 찾아 사람이 아님으로 재 샘플링 한다. `Conversely, the N-expert examines the current clusters classified as humans, identifies the wrong ones (i.e. false positives), and adds them to training set as new negative samples. `

이 작업은 충분한 수의 샘플데이터가 수집될떄 까지 반복된 후 재 학습용 데이터로 사용된다. `This process is repeated several times, until a sufficient number of new positive and negative samples have been collected. Then the human classifier is retrained with the augmented training set. `

연구 결과 `In practice,`
- P모듈은 일반화 성능을 올려 주고 ` the P-expert increases the generality of the classifier,`
- N모듈은 분별력 성능을 올려 준다. ` while the N-expert increases the discriminability. `

The implementation details are described in Algorithm 1.

![](https://i.imgur.com/SWy1pxW.png)


#### P모듈은 어떻게 오판 여부를 판단 하는가? 

P모듈은 추적 경로 정보를 사용한다. 한번 사람이라고 인식되고, 동일한 추적 물체일경우 사람으로 판단 한다. `The P-expert selects new positive samples based on the tracked trajectories of the detected clusters. In particular, clusters classified as non-human, but belonging to a trajectory where at least one cluster was classified as human, will be added to the training set as positive samples.`

The conditions to be satisfied by such new positive samples are as follows:

![](https://i.imgur.com/COKCneC.png)

The values of `K,r^p_{min},v^p_{min} and σ^p_{max}`used in our system were empirically tuned before the experiments. 

The last condition,in particular, is particularly useful to filter out clusters that,even if associated with “human-like” trajectories, have a high level of uncertainty because of sudden movements or the proximity of other clusters (see Fig.7).

![](https://i.imgur.com/pyx8ZcI.png)
```
[Fig. 7 ]
- Example of human-like trajectory samples, including one (redcrossed) filtered out because too uncertain. 
- The green dashed line is the target’s trajectory, while the blue dashed circles are the position’s uncertainties 
```

#### N모듈은 어떻게 오판 여부를 판단 하는가? 

N모듈은 사람은 **100% STATIC**하지 않다는 가정을 기반으로 한다.사람이 가만히 서있거나 앉아 있더라도, 사람의 형태나 중앙위치는 조금씩 변한다고 가정한한다. 아래의 조건을 만족하면 오판으로 인정한다. ` The N-expert analyses clusters classified as humans and selects those which are potential false positives, transforming them into new negative samples for future retraining. This selection is based on the assumption that humans are not completely static, as there are always some changes in a cluster’s shape and its centroid position, even if the person is simply standing or sitting. Taking advantage of the 3D LiDAR’s high accuracy, these static clusters (considered as negative samples) can be identified if they satisfy the following conditions:`


$$
r_{k} \leq r_{\max }^{n} \text { and } v_{k} \leq v_{\max }^{n} \text { and } \sigma_{x}^{2}+\sigma_{y}^{2} \leq\left(\sigma_{\max }^{n}\right)^{2}
$$

The parameters `r^n_{max},v^n_{max}, and σ^n_{max}`were determined empirically for the experiments. 

성능 평가는 7.2에서 수행 되었다. The performance of the P–N experts with respect to the stability of the online learning process is also discussed and evaluated in Sect.7.2.

## 6 System setup and dataset

## 7 Experimental results

### 7.1 Clustering performance

### 7.2 Stability analysis (온라인 학습 을 위해 필요 한듯 !!!!)

A stability analysis of the learning process is possible by considering the variations of false positives α and false negatives β generated by the human classifier (Kalal et al. 2012)

...

we use the following **performance metrics** for the **P–N experts**...

...

### 7.3 Classification performance

In this section we evaluate the performance of the 3D LiDAR based human classifier, 
- first by analysing the classification results of the SVM trained online and offline, 
- then by assessing the online trained SVM under uncertainty, 
- and finally by comparing the results with our new feature set against other state-of-the-art feature combinations.

#### 7.3.1 Online versus offline classification

#### 7.3.2 Online classification under uncertainty

#### 7.3.3 Feature sets comparison

## 8 Conclusions

본 논문은 온라인 학습 및 추적, 분류가 가능한 로봇에 대하여 다루고 있다. `This paper presented an improved online learning framework for human detection in 3D LiDAR scans, including an extensive evaluation of runtime performance, stability, and classification results. The framework relies on a multitarget tracking system with a real-time clustering algorithm and an efficient sample generator. It enables a mobile robot to learn autonomously from the environment what humans look like, which greatly reduces the need for tedious and time-consuming data annotation.`

성능도 좋다. `We showed that our adaptive clustering method is more precise than other state-of-the-art algorithms, while still maintaining a low computational cost suitable for most human tracking applications. The stability of the online learning process has been analysed in detail, and has been guaranteed in practice by the high-precision of our P–N expert modules in providing good training samples. Our experiments showed also that, thanks to an augmented set of efficient features, our human classifier performs better than other state-of-the-art solutions in terms of F-measure (accuracy), but with comparable precision and recall.`

본 연구의 코드는 오픈되었다. 다른 부분에 적용한 사례도 있다. `The whole system is implemented in ROS following a modular design. Both the software and the dataset used in our experiments are publicly available for research purposes. Although currently used for human detection and tracking, `
- our software could be extended to deal with other moving objects such as cars, bicycles or animals (Sun et al. 2018). 
- The 3D LiDAR-based cluster detection module could also be replaced by other detectors based on different sensors, such as RGB-D cameras and 2D LiDARs (Yan et al. 2018).

향후 연구 `Future extensions should include `
- **coupling the human classifier to the multi-target tracker**, so that improving the classification of human clusters also improves people tracking. 
- Moreover, although our solution enables human detection with mobile robots in dynamic environments, in the current paper it has been tested only on datasets recorded for relatively **short period** of times. Daily routines in public environments could be exploited by a service robot, for example, to collect negative background samples at night, when there are no moving objects, and positive human samples during the day.9 

최근 연구 트랜드 : Long-term operation and open-ended learning are therefore two promising directions for future research in this area (Vintr et al. 2019; Krajnik et al. 2017). 

향후 딥러닝을 위한 온라인 학습법에 대한 연구를 진행 중인다. `Future work should look at other classification methods such as deep neural networks, exploiting online learning to overcome the difficulty of collecting extensive training samples.`