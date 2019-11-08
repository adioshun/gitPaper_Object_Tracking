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


