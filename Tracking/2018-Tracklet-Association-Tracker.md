# [Tracklet Association Tracker](https://arxiv.org/pdf/1808.01562.pdf)

> 2018

기존의 추적기는 유사도 학습 + DA단계로 나누어 진다. 각 단계는 hand-craft가 필요 하여 Feature에서 직접 학습하여 수행 하는 목적을 달성 할수 없다. `Traditional multiple object tracking methods divide the task into two parts: affinity learning and data association. The separation of the task requires to define a hand-crafted training goal in affinity learning stage and a hand-crafted cost function of data association stage, which prevents the tracking goals from learning directly from the feature. `

본 논문에서는 데이터 기반 추적 기법인 **Tracklet Association Tracker (TAT)**를 제안 하였다. 제안 방식은 bi-level optimization formulation을 사용하여 특징 학습과 DA를 하나로 합쳤다. 따라서 association  결과를 특징 학습으로 바로 알수 있다. `In this paper, we present a new multiple object tracking (MOT) framework with data-driven association method, named as Tracklet Association Tracker (TAT). The framework aims at gluing feature learning and data association into a unity by a bi-level optimization formulation so that the association results can be directly learned from features. `

To boost the performance, we also adopt the popular hierarchical association and perform the necessary alignment and selection of raw detection responses. Our model trains over 20× faster than a similar approach, and achieves the stateof-the-art performance on both MOT2016 and MOT2017 benchmarks.

## 1. Introduction


MOT의 트렌드가 탐지기반 추적으로 바뀌면서 아래 두가지가 중요 요소로 뽑히고 있다. . `Multiple Object Tracking (MOT) is one of the most critical middle-level computer vision tasks with wide-range applications such as visual surveillance, sports events, and robotics. Owing to the great success of object detection techniques, detection based paradigm dominates the community of MOT. The critical components of the paradigm include an affinity model telling how likely two objects belong to a single identity, and a data association method that links objects across frames, based on their affinities, so as to form a complete trajectory for each identity.`
- affinity model: how likely two objects belong to a single identity
- DA : links objects across frames, based on their affinities

트랙렛기반 연결은 탐지기반 추적에 적합한 방법이다. `Tracklet-based association is a well-accepted approach in detection-based MOT [16, 36, 34, 31]. `

트랙렛기반 연결은 두가지 단계로 이루어져있다. `It is usually constructed by two stages: `
- 단계 1 : In stage I, we link detection responses in the adjacent frame using straightforward strategies to form short tracklets. 
- 단계 2 : In stage II, we mainly perform two tasks: 
	- 단계 2-a : extract much finer features from the tracklets, including temporal and spatial, appearance and motion data to construct a tracklet-level affinity model, 
	- 단계 2-b : and then perform graph-based association across all of them, and conduct necessary post-processing. 

탐지기반 연결보다 트랙렛기반 연결에는 두가지 장점이 있다. `There are two advantages of this approach, compared to associations on detection responses directly.`
-  With tracklet-based association, the number of connected components is significantly brought down so that investigating detection dependency across distant frames is computationally affordable. 
- Besides, it is capable of extracting high-level information, while reducing bounding box noises brought by bad detectors [16].

단계 1의 affinity model을 정의 하는데는 여러 방법들이 있다. `There are various ways to define the affinity model in stage I, `
- like bounding box intersection-over-union(IOU), 
- spatial-temporal distance, 
- appearance similarity, 
- etc. 

단계 2가 어려운 부분이다. `The harder part exists in stage II.`
- For the affinity model, traditional hand-crafted features or individually learned affinities do not work well [16, 37], due to the lack of data-driven properties in joint consideration of multiple correlated association choices. 
- For the association, it is regular to use a global optimization algorithm, such as linear programming or network flow, to link these short tracklets. 
- However, it is non-trivial to define a proper cost function for these approaches. 
- Earlier trackers use hand-crafted cost functions and perform an inference afterward. 
- Sometimes, they have to use grid search and empirical tuning to find the hyperparameters producing the best outcome.

딥러닝 기반 방식은 hand-craft방식보다 특징을 잘 추출 하여 성능이 좋다. 하지만 MOT는 **특징(Feature of Object)**에 큰 영향을 받지 않는다. 따라서 association method에 프레임간 연결성을 모델링 하여 추가 하는 작업이 필요 하다. 이렇게 함으로써 특징(Feature)과 연결(Association) 사이의 다리를 만들수 있다. `Recently, deep learning has shown its powerful learning capability in feature extraction. It outperforms almost all hand-crafted feature descriptors, such as HOG, SIFT, etc. Large-scale data provides nutrients for the learning of large models, and data-driven approaches are becoming rather important. However, for MOT, the ultimate goal, like MOTA, is not directly related to the features of objects. Thus, it is necessary to model the connectivity across frames into an association method, so that we can build a bridge from the features to the association goal, and perform learning using optimization method. `

Because network flow is an approach which can be solved in polynomial time, it has a great potential for data-driven learning comparing to other NP-hard formulations [31, 32, 33, 36]. 
- Schulter et al. [29] propose a network flow based novel framework to handle both tasks of stage II in an end-to-end fashion: 
	- 원리 : by back-propagating a constrained linear programming objective function. 
	- 장단점 : While the framework allows learning from detection features and auto-tuning of costs, the drawbacks are clear: 1) the large number of detection responses limit the expansion of window size; 2) the unbounded costs are easily diverging, and training is slow; 3) High-level motion information is not considered.

```
[29] S. Schulter, P. Vernaza, W. Choi, and M. Chandraker. Deep network flow for multi-object tracking. arXiv preprint arXiv:1706.08482, 2017.
```

본 논문에서는 [29]대비 3가지 주요 차이점을 가지는 TAT기법을 제안 하였다. `We propose Tracklet Association Tracker (TAT), an improved bi-level optimization framework compared to work of Schulter et al. [29] in three key aspects. `
- First, we use deep metric learning to extract the appearance embedding for each detection response; 
- Second, we introduce tracklet to the framework, not only accelerating the computation but also provides motion dependency. 
- Last but not the least, we adopt an approximate gradient that significantly improves the association model training process. 

By clarifying the boundary of cost values, the framework ensures convergence can always be achieved and includes all cost parameters into the end-to-end training process while retaining high accuracy.

본 논문의 기여는 아래와 같다. `All in all, our contributions include:`
- We introduce tracklet association into the bi-level optimization framework. By exploiting tracklets, our system improves the performance on long time occlusion. 
- We implement TAT, an approximate network flow learning approach that provides a more stable and faster(over 20×) solution of similar method [29]. The method achieves the state-of-the-art performance on MOT2016 and MOT2017 [22].
- We conduct comprehensive discussions on the impact of each component we introduce. Besides, we give a quantitative evaluation on the importance of alignment and noisy outlier removal, which shows both ancient and modern detectors can benefit from these strategies.

## 2. Related Work



