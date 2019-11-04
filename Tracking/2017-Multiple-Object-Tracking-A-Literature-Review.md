
# [Multiple Object Tracking: A Literature Review](https://arxiv.org/pdf/1409.7618.pdf)

  



여러 제안 들이 있지만 문제점은 존재 한다. `Although different kinds of approaches have been proposed to tackle this problem, it still remains challenging due to factors like`
- abrupt appearance changes and
- severe object occlusions.

본 연구에서는 위 문제들에 대한 리뷰를 하겠다. `In this work, we contribute the first comprehensive and most recent review on this problem. `

기여도는 아래와 같다. `The main contributions of this review are fourfold: `
- 1) Key aspects in a multiple object tracking system, including formulation, categorization, key principles, evaluation of an MOT are discussed.
- 2) Instead of enumerating individual works, we discuss existing approaches according to various aspects, in each of which methods are divided into different groups and each group is discussed in detail for the principles, advances
and drawbacks.
- 3) We examine experiments of existing publications and summarize results on popular datasets to provide quantitative comparisons. We also point to some interesting discoveries by analyzing these results.
- 4) We provide a discussion about issues of MOT research, as well as some interesting directions which could possibly become potential research effort in the future.

## 1. INTRODUCTION

MOT는 컴퓨터 비젼에서 중요한 역할을 담당한다. 추적 대상은 사람 부터 차량까지 다양하다. `Multiple Object Tracking (MOT), or Multiple Target Tracking (MTT), plays an important role in computer vision. The task of MOT is largely partitioned to locating multiple objects, maintaining their identities, and yielding their individual trajectories given an input video. Objects to track can be, for example, pedestrians on the street [1], [2],vehicles in the road [3], [4], sport players on the court [5], [6], [7], or groups of animals (birds [8], bats [9], ants [10], fish [11], [12], [13], cells [14], [15], etc.). `


멀티 오브젝트는 어떻게 보면 싱글 오브젝트로도 볼수 있다. `Multiple “objects” could also be viewed as different parts of a single object [16]. `

본 논문에서 추적 대상은 **사람**으로 잡았다. 이유는 다음과 같다. `In this review, we mainly focus on the research on pedestrian tracking. The underlying reasons for this specification are threefold. `
- 사람은 강체가 아니라 좋은 연구 대상이다. `First, compared to other common objects in our environment, pedestrians are typical nonrigid objects, which is an ideal example to study the MOT problem. `
- 사람 추적은 응용 서비스가 많다. `Second, videos of pedestrians arise in a huge number of practical applications, which further results in great commercial potential. `
- 대부분의 연구는 사람 추적에 대해 이루어 진다. `Third, according to all data collected for this review, at least 70% of current MOT research efforts are devoted to pedestrians.`

MOT는 다음 high-level task로 나누어(??grounds) 볼수 있다. `As a mid-level task in computer vision, multiple object tracking grounds high-level tasks such as `
- pose estimation [17],
- action recognition [18],
- and behavior analysis [19].

여러 응용 서비스 들도 많다. `It has numerous practical applications, such as visual surveillance [20], human computer interaction [21] and virtual reality [22]. These practical requirements have sparked enormous interest in this topic.`

SOT & MOT목적
- SOT
- sophisticated appearance models and/or motion models 설계
- scale changes, outof-plane rotations and illumination variations 문제 해결
- MOT
- 위 문제 외 추가 2가지
- 시간 마다 변하는 물체 **수**
- 물체 ID

```
Compared with Single Object Tracking (SOT), which primarily focuses on designing sophisticated appearance models and/or motion models to deal with challenging factors such as scale changes, outof-plane rotations and illumination variations, multiple object tracking additionally requires two tasks to be solved:
- determining the number of objects, which typically varies over time, and
- maintaining their identities.
```

MOT에서 주요 이슈는 `Apart from the common challenges in both SOT and MOT, further key issues that complicate MOT include among others:`
- 1) frequent occlusions,
- 2) initialization and termination of tracks,
- 3) similar appearance, and
- 4) interactions among multiple objects.

여러 연구가 진행 되었지만 다양한 복잡성으로 새연구원들이 접근하기 어렵다. `In order to deal with all these issues, a wide range of solutions have been proposed in the past decades. These solutions concentrate on different aspects of an MOT system, making it difficult for MOT researchers, especially newcomers, to gain a comprehensive understanding of this problem. `

그래서 본 논문을 작성 하였다. `Therefore, in this work we provide a review to discuss the various aspects of the multiple object tracking problem.`

### 1.1 Differences from Other Related Reviews

기존의 여러 리뷰 논문을 분류 하면 아래와 같다. `To the best of our knowledge, there has not been any comprehensive literature review on the topic of multiple object tracking. However, there have been some other reviews related to multiple object tracking, which are listed in Table 1. We group these surveys into three sets and highlight the differences from ours as follows.`

![](https://i.imgur.com/JszNpgL.png)

리뷰논문들의 첫 분류 : The **first set** [19], [20], [21], [23], [24] discusses tracking as an individual part while this work specifically discusses various aspects of MOT. For example, object tracking is discussed as a step in the procedure of high-level tasks such as crowd modeling [19], [23], [24]. Similarly, in [21] and [20], object tracking is reviewed as a part of a system for behavior recognition [21] or video surveillance [20].


리뷰논문들의 두번째 분류 :The **second set** [25], [26], [27], [28] is dedicated to general visual tracking techniques [25], [26], [27] or some special issues such as appearance models in visual tracking [28]. Their reviewing scope is wider than ours; ours on the contrary is more comprehensive and focused on multiple object tracking.


리뷰논문들의 세번째 분류 :The **third set** [29], [30] introduces and discusses benchmarks on general visual tracking [29] and on specific multiple object tracking [30]. Their attention is laid on experimental studies rather than literature reviews.



### 1.2 Contributions

본 논문의 기여와 구성 요소는 아래와 같다. `We provide the first comprehensive review on the MOT problem to the computer vision community, which we believe is helpful to understand this problem, its main challenges, pitfalls, and the state of the art. The main contributions of this review are summarized as follows:`

- 2.1절에서는 문제점을 정의 하고 2.2절에서는 기술 분류 하였다. `We derive a unified formulation of the MOT problem which consolidates most of the existing MOT methods (Section 2.1), and two different ways to categorize MOT methods (Section 2.2).`

- 3장에서는 구성요소에 대하여 기술 하였다.`We investigate different key components involved in an MOT system, each of which is further divided into different aspects and discussed in detail regarding its principles, advances, and drawbacks (Section 3).`

- 4장에서는 실험 결과와 데이터셋을 기술 하였다. `Experimental results on popular datasets regarding different approaches are presented, which makes future experimental comparison convenient. By investigating the provided results, some interesting observations and findings are revealed (Section 4).`

- 5장에서는 미해결 문제와 연구 방향을 기술 하였다. `By summarizing the MOT review, we unveil existing issues of MOT research. Furthermore, open problems are discussed to identify potential future research directions (Section 5).`


최근 연구 물들을 중심으로 하였다. `Note that this work is mainly dedicated to reviewing recent literature on the advances in multiple object tracking. As mentioned above, we also present experimental results on publicly available datasets excepted from existing publications to provide a quantitative view on the state-of-the-art MOT methods. `

활용된 벤치마킹 [30]이다. `For standardized benchmarking of multiple object tracking we kindly refer the readers to the recent work MOTChallenge by Leal-Taixe´ et al. [30].`


### 1.3 Organization of This Review



### 1.4 Denotations

---

## 2 MOT PROBLEM

MOT의 문제점을 수학적으로 기술 하였다. We first endeavor to give a general mathematical formulation of MOT. We then discuss its possible categorizations based on different aspects.`

### 2.1 Problem Formulation


### 2.2 MOT Categorization

다음 3가지 criteria에 따라서 분류 하였다. `It is difficult to classify one particular MOT method into a distinct category by a universal criterion. Admitting this, it is thus feasible to group MOT methods by multiple criteria. In the following we attempt to conduct this according to three criteria:`
- a) initialization method,
- b) processing mode, and
- c) type of output.

위 3가지를 선택한 이유는 process를 설명하기 직관적이기 때문이다. `The reason we choose these three criteria is that this naturally follows the way of processing a task, i.e., `
- how the task is initialized,
- how it is processed and
- what type of result is obtained.

In the following, each of the criteria along with its corresponding categorization is represented.


#### 2.2.1 Initialization Method

물체가 초기화 되는 방식에 따라 나눌수 있다. `Most existing MOT works can be grouped into two sets [49], depending on how objects are initialized: `
- Detection-Based Tracking (DBT) and
- Detection-Free Tracking (DFT).

![](https://i.imgur.com/0XXPHuV.png)

##### Detection-Based Tracking

이 방식은 먼저 탐지를 하고 궤도에 연결하는 방식이다. `As shown in Figure 1 (top), objects are first detected and then linked into trajectories.`

**tracking-by-detection**라고도 불리운다. `This strategy is also commonly referred to as “tracking-by-detection”. `

흐름은 `Given a sequence, `
- 각 프레임마다 후보군에 대한 물체 탐지 또는 모션 탐지가 수행 된다. `type-specific object detection or motion detection (based on background modeling) [50],[51] is applied in each frame to obtain object hypotheses, `
- 이후 탐지 후보를 궤도에 연결하는 추적 작업이 수행 된다. `then (sequential or batch) tracking is conducted to link detection hypotheses into trajectories. `

두가지 이슈가 있다. `There are two issues worth noting.`
- 탐지기 학습이 선행 되기에 학습 대상에 의존적이다. `First, since the object detector is trained in advance, the majority of DBT focuses on specific kinds of targets, such as pedestrians, vehicles or faces. `
- 탐지기 성능에 의존성이 크다. `Second, the performance of DBT highly depends on the performance of the employed object detector.`

##### Detection-Free Tracking.

직접 물체의 수를 지정 해야 한다. 이후프레임에서 해당 물체의 위치를 localized 해준다. `As shown in Figure 1 (bottom), DFT [52], [53], [54], [55] requires manual initialization of a fixed number of objects in the first frame, then localizes these objects in subsequent frames.`


DBT방식 많이 사용된다. DFT는 학습이 불필요한 장점이 있다. `DBT is more popular because new objects are discovered and disappearing objects are terminated automatically. DFT cannot deal with the case that objects appear. However, it is free of pre-trained object detectors.`

Table 3 lists the major differences between DBT and DFT.

![](https://i.imgur.com/R6WV5DN.png)

#### 2.2.2 Processing Mode

현재 프레임을 처리 할때 이후 프레임 정보를 사용 하느냐 아니냐에 따라 온라인/오프라인 방식으로 나뉜다. `MOT can also be categorized into online tracking and offline tracking. The difference is whether or not observations from future frames are utilized when handling the current frame. `


온라인은 **인과방식(casual)**이라도도 불리운다. 추적 기법은 과거의 정보만을 이용하기 때문이다. 반면 오프라인(=배치트래킹)은 과거와 미래 정보 모두를 이용한다. `Online, also called causal, tracking methods only rely on the past information available up to the current frame, while offline, or batch tracking approaches employ observations both in the past and in the future.`

![](https://i.imgur.com/iB7kKat.png)

##### Online tracking.

In online tracking [52], [53], [54], [56], [57], the image sequence is handled in a step-wise manner, thus online tracking is also named as sequential tracking.

An illustration is shown in Figure 2 (top), with three objects (different circles) a, b, and c. The green arrows represent observations in the past. The results are represented by the object’s location and its ID. Based on the up-to-time observations, trajectories are produced on the fly.

##### Offline tracking.

Offline tracking [1], [46], [47], [51], [58], [59], [60], [61], [62], [63] utilizes a batch of frames to process the data.

As shown in Figure 2 (bottom), observations from all the frames are required to be obtained in advance and are analyzed jointly to estimate the final output.

계산 부하의 문제가 있어 나누거나 게층 구조로 처리 하기도 한다. `Note that, due to computational and memory limitation, it is not always possible to handle all the frames at once. An alternative solution is to split the data into shorter video clips, and infer the results hierarchically or sequentially for each batch. `


Table 4 lists the differences between the two processing modes.

![](https://i.imgur.com/1pMV8Bp.png)

#### 2.2.3 Type of Output

결과물의 무작위성(`randomness`)에 따라 MOT방식은 경정론적 또는 확률적 방식으로 나뉜다. `This criterion classifies MOT methods into deterministic ones and probabilistic ones, depending on the randomness of output. `

Deterministic방식은 반복횟수에 상관 없이 동일한 결과 값이 나온다. `The output of deterministic tracking is constant when running the methods multiple times. `

probabilistic방식은 반복시마다 다른 결과가 나온다. `While output results are different in different running trials of probabilistic tracking methods. `

이 두 가지 방법의 차이는 섹션 2.1에서 언급한 최적화 방법에서 비롯된다`The difference between these two types of methods results from the optimization methods adopted as mentioned in Section 2.1.`


#### 2.2.4 Discussion

탐지기 적용 여부에 따라 DBT/DFT가 나뉜다. `The difference between DBT and DFT is whether a detection model is adopted (DBT) or not (DFT). `

탐지값 사용방식에 따라 온라인/오프라인이나뉜다.`The key to differentiate online and offline tracking is the way they process observations. `

대부분 DFT는 온라인 방식과 같아 보인다. 하지만 예외도 있다. `Readers may question whether DFT is identical to online tracking because it seems DFT always processes observations sequentially. This is true in most cases although some exceptions exist. `

Orderless tracking [64] is an example. It is DFT and simultaneously processes observations in an orderless way. Though it is for single object tracking, it can also be applied for MOT, and thus DFT can also be applied in a batch mode.

Another vagueness may rise between DBT and offline tracking, as in DBT tracklets or detection responses are usually associated in a batch way. Note that there are also sequential DBT which conducts association between previously obtained trajectories and new detection responses [8], [31], [65].

The categories presented above in Section 2.2.1, 2.2.2 and 2.2.3 are three possible ways to classify MOT methods, while there may be others.

Notably, specific solutions for sport scenarios [5], [6], aerial scenes [44], [66], generic objects [8], [65], [67], [68], etc. exist and we suggest the readers refer to the respective publications.

By providing these three criteria described above, it is convenient for one to tag a specific method with the combination of the categorization label. This would help one to understand a specific approach easier.

---

## 3 MOT COMPONENTS

MOT의 구성 요소를 다루고 있다. `In this section, we represent the primary components of an MOT approach. As mentioned above, the goal of MOT is to discover multiple objects in individual frames and recover the identity information across continuous frames, i.e., trajectory, from a given sequence. `

MOT개발시 2가지를 고려 해야 한다. `When developing MOT approaches, two major issues should be considered. `
- 물체들간의 유사성을 측정(`Measure`) 하는 방법. `One is how to measure similarity between objects in frames, `
- 유사성을 기반으로 ID값을 찾는 법 `the other one is how to recover the identity information based on the similarity measurement between objects across frames. `

Roughly speaking,
- 유사성 측정은 외형, 움직인, 상호작용, 가려짐의 **Modeling** 작업이다. `the first issue involves the modeling of appearance, motion, interaction, exclusion, and occlusion. `
- ID 찾는것은 **Inference**문제 이다. `The second one involves with the inference problem.`

We review recent progress regarding both items as the following

### 3.1 Appearance Model

외관은 관련성(`affinity`) 계산하는데 중요한 요소 이다. `Appearance is an important cue for affinity computation in MOT. `

그러나 SOT의 경우 배경에서 **sophisticated appearance model**을 생성하는것에 초점을 두고 있는 반면 대부분의 MOT는 외형 모델을 중요하게 생각 하지 않는다. `However, different from single object tracking, which primarily focuses on constructing a sophisticated appearance model to discriminate object from background, most MOT methods do not consider appearance modeling as the core component, although it can be an important one. `

외형 모델은 두가지 구성 요소로 이루어져 있다. `Technically, an appearance model includes two components: `
- visual representation and
- statistical measuring.

**Visual representation**은 물체의 시각적 특성을 특징 정보를 이용하여 기술 하는것이다. ` Visual representation describes the visual characteristics of an object using some features, either based on a single cue or multiple cues. `

**Statistical measuring**은 다른 특정에서 유사성을 계산 하는 것이다. `Statistical measuring, on the other hand, is the computation of similarity between different observations.`

#### 3.1.1 Visual Representation

##### Local features.

##### Region features.

바운딩 박스에서 추출 된다. `Compared with local features, region features are extracted from a wider range (e.g. a bounding box). `

We illustrate them as three types:
- a) zero-order type,
- b) first-order type and
- c) up-to-second-order type

##### Others.

Besides local and region features, there are some other kinds of representation.
- Taking depth as an example, it is typically used to refine detection hypotheses [71], [84], [85], [86], [87].
- The Probabilistic Occupancy Map (POM) [42], [88] is employed to estimate how likely an object would occur in a specific grid cell.
- One more example is gait(걸음걸이) feature, which is unique for individual persons [62].

##### Discussion

#### 3.1.2 Statistical Measuring

This step is closely related to the section above.

Based on visual representation, statistical measure computes the affinity between two observations.

While some approaches solely rely on one kind of cue, others are built on multiple cues.

##### Single cue.

단일 cue를 이용하여 외형을 모델링 하는 방법은 **거리 정보를 유사도로 변환**하거나 **직접 유사성을 계산** 하는 방법이 있다. `Modeling appearance using single cue is either transforming distance into similarity or directly calculating the affinity. `

For example, the **Normalized Cross Correlation (NCC)** is usually adopted to calculate the affinity between two counterparts based on the representation of raw pixel template mentioned above [2], [69], [80], [90].

Speaking of color histogram, Bhattacharyya distance B (·, ·) is used to compute the distance between two color histograms ci and cj . The distance is transformed into similarity S like S (Ti , Tj ) = exp (−B (ci , cj )) [31], [36], [58], [62], [63], [91] or fit the distance to Gaussian distributions like [38]. Transformation of dissimilarity into likelihood is also applied to the representation of covariance matrix [61]. Besides these typical models, bag-of-words model [92] is employed based on point feature representation [33].

##### Multi cue

다중 cue를 사용하여서 외형 모델을 좀더 강인하게 만들수 있다. `Different kinds of cues can complement each other to make the appearance model more robust. `

그러나 정보를 융합하는 방법을 결정하는 것은 쉬운 일이 아니다. `However, it not trivial to decide how to fuse the information from multiple cues. `

퓨젼 방법론을 5가지로 분류 해 보았다. `Regarding this, we summarize multicue based appearance models according to five kinds of fusion strategies: `Boosting, Concatenating, Summation, Product, and Cascading (see also Table 5).

![](https://i.imgur.com/ac3uuAc.png)

###### Boosting.
The strategy of Boosting usually selects a portion of features from a feature pool sequentially via a Boosting based algorithm.

For example, from color histogram, HOG and covariance matrix descriptor, AdaBoost, RealBoost, and a HybridBoost algorithm are respectively employed to choose the most representative features to discriminate pairs of tracklets of the same object from those of different objects in [60], [49] and [40].

###### Concatenation.
Different kinds of features can be concatenated for computation. In [46], color, HOG and optical flow are concatenated for appearance modeling.

###### Summation.
This strategy takes affinity values from different features and balance these values with weights [71], [93], [94].

###### Product.
Differently from the strategy above, values are multiplied to produce the integrated affinity [33], [51], [95], [96]. Note that, independence assumption is usually made when applying this strategy.

###### Cascading.
This is a cascade manner of using various types of visual representation, either to narrow the search space [87] or model appearance in a coarseto-fine way [77].



### 3.2 Motion Model

모션 모델은 물체의 동적 움직임을 capture하는 것이다. `The motion model captures the dynamic behavior of an object. `

모션 모델은 다음 프레임에서의 물체 위치를 예측 하여 탐색 부하를 줄여 준다. `It estimates the potential position of objects in the future frames, thereby reducing the search space. `

대부분의 경우 물체는 천천히 움직인다고 가정한다. `In most cases, objects are assumed to move smoothly in the world and therefore in the image space (except for abrupt motions).`

본 논문에서는 선형 모델과 비선형 모델을 살펴 보겠다. ` We will discuss linear motion model and non-linear motion model in the following.`


#### 3.2.1 Linear Motion Model

가장 인기 있는 모델이다. `This is by far the most popular model [32], [97], [98]. `

고정 속도라는 가정하에 이 모델을 만든다. 모델 생성하는데는 3가지 방법이 있다. `A constant velocity assumption [32] is made in this model. Based on this assumption, there are three different ways to construct the model.`


##### 가. Velocity smoothness

물체에 속도값을 강제 적용하여 속도가 서서히 변하게 한다. Velocity smoothness is modeled by enforcing the velocity values of an object in successive frames to change smoothly.

In [45], it is implemented as a cost term,

![](https://i.imgur.com/ylOKsT9.png)

where the summation is conducted over N frames and M trajectories/objects.

##### 나. Position smoothness

이 방식은 탐지 위치와 예측 위치의 불 일치에 강제성을 준다. `Position smoothness directly forces the discrepancy between the observed position and estimated position. `

###### Let us take [31] as an example.

Considering a temporal gap ∆t between tail of tracklet T_j and head of tracklet T_j , the smoothness is modeled by fitting the estimated position to a Gaussian distribution with the observed position as center.

In the stage of estimation, both forward motion and backward motion are considered. Thus, the affinity considering linear motion model is,

![](https://i.imgur.com/4DqFfio.png)

where “F” and “B” means forward and backward direction.

###### A similar strategy is adopted by Yang et al. [59].

The displacement between observed position and estimated position ∆p is fit to a Gaussian distribution with zero center.

###### Other examples of this strategy are [1], [7], [58], [59], [60], [99].

##### 다. Acceleration smoothness.

[99]에서는 위치/속도외에 가속도도 고려 되었다. `Besides considering position and velocity smoothness, acceleration is taken into account [99]. `

The probability distribution of motion of a state {^s_k} at time k given the observation tracklet {o_k} is modeled as,

![](https://i.imgur.com/5mGYfe0.png)

where
- v_k is the velocity,
- a_k is the acceleration,
- N is a zero-mean Gaussian distribution.

#### 3.2.2 Non-linear Motion Model

대부분의 경우 선형 모델로 표현 된다. `The linear motion model is commonly used to explain the object’s dynamics. `

그러나 더 정확한 모델 표현을 위해 비선형이 필요 하다. `However, there are some cases which the linear motion model cannot deal with. To this end, nonlinear motion models are proposed to produce more accurate motion affinity between tracklets.`

![](https://i.imgur.com/4gHocGj.png)

For instance, Yang et al. [47] employ a non-linear motion model to handle the situation that targets may move freely. Given two tracklets T1 and T2 which belong to the same target in Figure 4(a), the linear motion model [59] would produce a low probability to link them. Alternatively, employing the non-linear motion model, the gap between the tail of tracklet T1 and the head of tracklet T2 could be reasonably explained by a tracklet T0 ∈ S, where S is the set of support tracklets.

As shown in Figure 4(b), T0 matches the tail of T1 and the head of T2. Then the real path to bridge T1 and T2 is estimated based on T0, and the affinity between T1 and T2 is computed similarly as described in Section 3.2.1.

### 3.3 Interaction Mode (=mutual motion model)

물체와 물체들사이의 영향력을 표현한 것이다. `Interaction model, also known as mutual motion model, captures the influence of an object on other objects. `

In the crowd scenery, an object would experience some “force” from other agents and objects.

For instance, when a pedestrian is walking on the street, he would adjust his speed, direction and destination, in order to avoid collisions with others.

Another example is when a crowd of people walk across a street, each of them follows other people and guides others at the same time.

In fact, these are examples of two typical interaction models known as the social force models [100] and the crowd motion pattern models [101].

#### 3.3.1 Social Force Models

#### 3.3.2 Crowd Motion Pattern Models

### 3.4 Exclusion Model

Exclusion is a constraint employed to avoid physical collisions when seeking a solution to the MOT problem.

이는 물리 공간에서 두개의 물체가 동일한 곳에 있을수 없다는 사실에 기반 한다. `It arises from the fact that two distinct objects cannot occupy the same physical space in the real world. `

Given multiple detection responses and multiple trajectory hypotheses, generally there are two constraints. The first one is the socalled detection-level exclusion [105], i.e., two different detection responses in the same frame cannot be assigned to the same target. The second one is the so-called trajectory-level exclusion, i.e., two trajectories cannot be infinitely close to each other.

### 3.5 Occlusion Handling

맞물림은 MOT에서 가장 큰 문제이다. `Occlusion is perhaps the most critical challenge in MOT. `

맞물림으로 인해 ID변경이나 궤도 단절이 발생 한다. `It is a primary cause for ID switches or fragmentation of trajectories. `

In order to handle occlusion, various kinds of strategies have been proposed.

#### 3.5.1 Part-to-whol

#### 3.5.2 Hypothesize-and-tes

#### 3.5.3 Buffer-and-recove

#### 3.5.4 Others

### 3.6 Inference

#### 3.6.1 Probabilistic Inference

이 방식의 물체의 상태를 **distribution with uncertainty**로 표현한다. ` Approaches based on probabilistic inference typically represent states of objects as a distribution with uncertainty. `

추적기의 목표는 이전 측정 값들에 probability reasoning기법을 적용하여 target state의 probabilistic distribution를 예측 하는 것이다. `The goal of a tracking algorithm is to estimate the probabilistic distribution of target state by a variety of probability reasoning methods based on existing observations. `

이 방식은 과거의 측정 정보들만 필요로 하기 때문에 일종의 온라인 추적기이다. `This kind of approaches typically requires only the existing, i.e. past and present observations, thus they are especially appropriate for the task of online tracking. `

존재 하는 측정치만 예측에 사용되므로 자연적으로 마코프 속성.....As only the existing observations are employed for estimation, it is naturally to impose the assumption of Markov property in the objects state sequence.

This assumption includes two aspects, recalling the formula in Section 2.1.
- First, the current object state only depends on the previous states. Further, it only depends on the very last state if the first-order Markov property is imposed,
- Second, the observation of an object is only related to its state corresponding to this observation. In other words, the observations are conditionally independent

위 두 관점은 **동적 모델**과 **관측 모델**에 관련이 있다. `These two aspects are related to the Dynamic Model and the Observation Model, respectively. `

동적 모델은 **tracking strategy**에 해당한다. 관측 모델은 물체 상태에 관련있는 관측값을 제공한다. `The dynamic model corresponds to the tracking strategy, while the observation model provides observation measurements concerning object states.`

예측 (`Prediction step`)단계에서 과거 관측치를 기반으로 현재 상태를 예측 한다. `The predict step is to estimate the current state based on all the previous observations.`

좀더 자세히는 현 상태의 **posterior probability distribution **은 동적 모델로 합쳐진 이전 물체 상태로 예측 된다. `More specifically, the posterior probability distribution of the current state is estimated by integrating in the space of the last object state via the dynamic model. `

업데이트(`update step`) 단계는 관측 모델하에 얻어진 측정값을 기반으로 **posterior probability distribution **를 업데이트 하는 것이다. `The update step is to update the posterior probability distribution of states based on the obtained measurements under the observation model.`


According to the equations, states of objects can be estimated by iteratively conducting the prediction and updating steps.

However, in practice, the object state distribution cannot be represented without simplifying assumptions, thus there is no analytical solution to computing the integral of the state distribution.

Additionally, for multiple objects, the dimension of the sets of states is very large, which makes the integration even more difficult, requiring the derivation for approximate solutions.

대표적인 **probabilistic inference models**들 ` Various kinds of probabilistic inference models haves been applied to multi-object tracking [36], [95], [117], [118], such as Kalman filter [35], [37], Extended Kalman filter [34]and Particle filter [32], [33], [52], [93], [119], [120], [121], [122].`

##### 가. Kalman filter.

선형/정규 분포 `In the case of a linear system and Gaussian-distributed object states, the Kalman filter [37] is proved to be the optimal estimator. It has been applied in [35].`


##### 나. Extended Kalman filter.

비선형 `To include the non-linear case, the extended Kalman filter is one possible solution. It approximates the non-linear system by a Taylor expansion [34].`

##### 다. Particle filter.

**Monte Carlo sampling** 최근 가장 인기 있는 기법 이다. `Monte Carlo sampling based models have also become popular in tracking, especially after the introduction of the particle filter [10], [32], [33], [52], [93], [119], [120], [121]. `

This strategy models the underlying distribution by a set of weighted particles, thereby allowing to drop any assumptions about the distribution itself [32], [33], [36], [93].


#### 3.6.2 Deterministic Optimization


이 방식은 MAP를 찾는 것이다. `As opposed to the probabilistic inference methods, approaches based on deterministic optimization aim to find the maximum a posteriori (MAP) solution to MOT. `

이렇게 하기 위해 데이터 연동/Target state 추론은 최적화 문제로 변하게 된다. `To that end, the task of inferring data association, the target states or both, is typically cast as an optimization problem. `

이 방식은 **오프라인 추적기**에 좀더 적합하다. `Approaches within this framework are more suitable for the task of offline tracking `
- because observations from all the frames or at least a time window are required to be available in advance.

Given observations (usually detection hypotheses) from all the frames, these types of methods endeavor to globally associate observations belonging to an identical object into a trajectory.

중요 문제는 어떻게 최적의 association을 하는가 이다. `The key issue is how to find the optimal association.`

주요 기법들은 아래와 같다. ` Some popular and well-studied approaches are detailed in the following.`

##### 가. Bipartite graph matching

##### 나. Dynamic Programming.

##### 다. Min-cost max-flow network flow.

##### 라. Conditional random field.

##### 마. MWIS

#### 3.6.3 Discussion

현실에서는 확률 기반 접근 방식 보다는 결정론적 최적화 방식이 더 많이 사용된다. `In practice, deterministic optimization or energy minimization is employed more popularly compared with probabilistic approaches. `

> 왜? 오프라인 추적기인데???

비록, 확률기반이 직관적이고, 완벽한 솔루션이지만 대부분 **추론**이 어렵다. Although the probabilistic approaches provide a more intuitive and complete solution to the problem, they are usually difficult to infer.

반면, 최적화 기반은 주어진 시간동안 꽤 괜찮은 결과를 뽑아 낸다. `On the contrary, energy minimization could obtain a “good enough” solution in a reasonable time.`

---


---

  

MOT는 오랜 기간 연구되었지만 아직 부족하다. 본 논문을 통해 리뷰를 하려 한다. `Multiple Object Tracking (MOT) is an important computer vision problem which has gained increasing attention due to its academic and commercial potential. Although different kinds of approaches have been proposed to tackle this problem, it still remains challenging due to factors like abrupt appearance changes and severe object occlusions. In this work, we contribute the first comprehensive and most recent review on this problem. We inspect the recent advances in various aspects and propose some interesting directions for future research. To the best of our knowledge, there has not been any extensive review on this topic in the community. We endeavor to provide a thorough review on the development of this problem in recent decades.`

  

The main contributions of this review are fourfold:

- 1) Key aspects in a multiple object tracking system, including formulation, categorization, key principles, evaluation of an MOT are discussed.

- 2) Instead of enumerating individual works, we discuss existing approaches according to various aspects, in each of which methods are divided into different groups and each group is discussed in detail for the principles, advances

and drawbacks.

- 3) We examine experiments of existing publications and summarize results on popular datasets to provide quantitative comparisons. We also point to some interesting discoveries by analyzing these results.

- 4) We provide a discussion about issues of MOT research, as well as some interesting directions which could possibly become potential research effort in the future.

  

  

## 1 INTRODUCTION

  

MOT에서 대상은 사람, 차량, 운동선수등이 될수 있다. `Multiple Object Tracking (MOT), or Multiple Target Tracking (MTT), plays an important role in computer vision. The task of MOT is largely partitioned to locating multiple objects, maintaining their identities, and yielding their individual trajectories given an input video. Objects to track can be, for example, pedestrians on the street [1], [2],vehicles in the road [3], [4], sport players on the court [5], [6], [7], or groups of animals (birds [8], bats [9], ants [10],fish [11], [12], [13], cells [14], [15], etc.). Multiple “objects” could also be viewed as different parts of a single object[16]. `

  

본 논문에서는 보행자 추적에 중점을 두었다. 그 이유는 아래와 같다. `In this review, we mainly focus on the research on pedestrian tracking. The underlying reasons for this specification are threefold.`

- First, compared to other common objects in our environment, pedestrians are typical nonrigid objects, which is an ideal example to study the MOT problem.

- Second, videos of pedestrians arise in a huge number of practical applications, which further results in great commercial potential.

- Third, according to all data collected for this review, at least 70% of current MOT research efforts are devoted to pedestrians.

  

  

As a mid-level task in computer vision, multiple object tracking grounds high-level tasks such as pose estimation [17], action recognition [18], and behavior analysis [19].

  

It has numerous practical applications, such as visual surveillance [20], human computer interaction [21] and virtual reality [22]. These practical requirements have sparked enormous interest in this topic.

  

Compared with Single Object Tracking (SOT), which primarily focuses on designing sophisticated appearance models and/or motion models to deal with challenging factors such as scale changes, out of-plane rotations and illumination variations, multiple object tracking additionally requires two tasks to be solved:

- determining the number of objects, which typically varies over time,

- and maintaining their identities.

  

Apart from the common challenges in both SOT and MOT, further key issues that complicate MOT include among others:

- 1) frequent occlusions,

- 2) initialization and termination of tracks,

- 3) similar appearance, and

- 4) interactions among multiple objects.

  

In order to deal with all these issues, a wide range of solutions have been proposed in the past decades. These solutions concentrate on different aspects of an MOT system, making it difficult for MOT researchers, especially newcomers, to gain a comprehensive understanding of this problem. Therefore, in this work we provide a review to discuss the various aspects of the multiple object tracking problem.

  

### 1.1 Differences from Other Related Reviews

  

다른 리뷰 논문들 요약

  

### 1.2 Contributions

  

### 1.3 Organization of This Review

  

### 1.4 Denotations

  

---

  

## 2 MOT PROBLEM

  

### 2.1 Problem Formulation

  

### 2.2 MOT Categorization

  

#### 2.2.1 Initialization Method

  

Most existing MOT works can be grouped into two sets [49], depending on how objects are initialized:

- Detection-Based Tracking (DBT) and

- Detection-Free Tracking (DFT).

  

##### Detection-Based Tracking (=tracking-by-detection)

  

As shown in Figure 1 (top), objects are first detected and then linked into trajectories. This strategy is also commonly referred to as “tracking-by-detection”.

  

Given a sequence,

- type-specific object detection or motion detection (based on background modeling) [50], [51] is applied in each frame to obtain object hypotheses,

- then (sequential or batch) tracking is conducted to link detection hypotheses into trajectories.

  

There are two issues worth noting.

- First, since the object detector is trained in advance, the majority of DBT focuses on specific kinds of targets, such as pedestrians, vehicles or faces.

- Second, the performance of DBT highly depends on the performance of the employed object detector

  

  

##### Detection-Free Tracking

  

As shown in Figure 1 (bottom), DFT [52], [53], [54], [55] requires manual initialization of a fixed number of objects in the first frame, then localizes these objects in subsequent frames.

  

DBT is more popular because new objects are discovered and disappearing objects are terminated automatically. DFT cannot deal with the case that objects appear. However, it is free of pre-trained object detectors.

  

![](https://i.imgur.com/T6epzVr.png)

  

Table 3 lists the major differences between DBT and DFT.

  

#### 2.2.2 Processing Mode

  

MOT can also be categorized into online tracking and offline tracking.

  

The difference is whether or not observations from future frames are utilized when handling the current frame.

- Online, also called causal, tracking methods only rely on the past information available up to the current frame,

- while offline, or batch tracking approaches employ observations both in the past and in the future.

  

##### Online tracking (=sequential tracking)

  

In online tracking [52], [53], [54], [56], [57], the image sequence is handled in a step-wise manner, thus online tracking is also named as sequential tracking.

  

An illustration is shown in Figure 2 (top),

- with three objects (different circles) a, b, and c.

- The green arrows represent observations in the past.

  

The results are represented by the object’s location and its ID. Based on the up-to-time observations, trajectories are produced on the fly.

  

  

##### Offline tracking

  

Offline tracking [1], [46], [47], [51], [58], [59], [60], [61], [62], [63] utilizes a batch of frames to process the data.

  

As shown in Figure 2 (bottom), observations from all the frames are required to be obtained in advance and are analyzed jointly to estimate the final output.

  

Note that, due to computational and memory limitation, it is not always possible to handle all the frames at once.

  

An alternative solution is to split the data into shorter video clips, and infer the results hierarchically or sequentially for each batch.

  

  

  

![](https://i.imgur.com/W1oNuc5.png)

  

Table 4 lists the differences between the two processing modes

  

  

#### 2.2.3 Type of Output

  

This criterion classifies MOT methods into deterministic ones and probabilistic ones, depending on the **randomness of output**.

- The output of deterministic tracking is constant when running the methods multiple times.

- While output results are different in different running trials of probabilistic tracking methods.

  

  

The difference between these two types of methods results from the optimization methods adopted as mentioned in Section 2.1.

  

  

#### 2.2.4 Discussion

  

The difference between DBT and DFT is whether **a detection model** is adopted (DBT) or not (DFT).

  

The key to differentiate online and offline tracking is the way they process observations.

  

Readers may question whether DFT is identical to online tracking because it seems DFT always processes observations sequentially.

- This is true in most cases although some exceptions exist.

- Orderless tracking [64] is an example.

- It is DFT and simultaneously processes observations in an orderless way. Though it is for single object tracking, it can also be applied for MOT, and thus DFT can also be applied in a batch mode.

  

  

Another vagueness may rise between DBT and offline tracking, as in DBT tracklets or detection responses are usually associated in a batch way.

  

Note that there are also sequential DBT which conducts association between previously obtained trajectories and new detection responses [8], [31], [65].

  

The categories presented above in Section 2.2.1, 2.2.2 and 2.2.3 are three possible ways to classify MOT methods, while there may be others.

  

Notably, specific solutions for sport scenarios [5], [6], aerial scenes [44], [66], generic objects [8], [65], [67], [68], etc. exist and we suggest the readers refer to the respective publications.

  

By providing these three criteria described above, it is convenient for one to tag a specific method with the combination of the categorization label.

  

This would help one to understand a specific approach easier.

  

---

  

## 3 MOT COMPONENTS

  

  

MOT의 목적은 연속된 프레임에서 각 물체를 추출 하고, 물체 식별자를 부여 하는 것이다. `In this section, we represent the primary components of an MOT approach. As mentioned above, the goal of MOT is to discover multiple objects in individual frames and recover the identity information across continuous frames, i.e., trajectory, from a given sequence. `

  

MOT 개발시 두가지 요소를 고려 해야 한다. `When developing MOT approaches, two major issues should be considered.`

- 프레임간 물체의 **유사도**를 어떻게 측정할 것인가? `One is how to measure similarity between objects in frames, `

- 프레임간 물체의 유사도를 기반으로 어떻게 **식별자**를 부여할 것인가? `the other one is how to recover the identity information based on the similarity measurement between objects across frames. `

  

Roughly speaking,

- 유사도는 외형, 움직임등의 **모델링**에 관련된 일이다. `the first issue involves the modeling of appearance, motion, interaction, exclusion, and occlusion. `

- 식별자 부여는 **추론**에 관련된 일이다. `The second one involves with the inference problem. `

  

  

We review recent progress regarding both items as the following.

  

  

### 3.1 Appearance Model

  

  

### 3.2 Motion Model

  

#### 3.2.1 Linear Motion Model

  

#### 3.2.2 Non-linear Motion Model

  

  

### 3.3 Interaction Model (=mutual motion model)

  

Interaction model, also known as mutual motion model, captures the influence of an object on other objects.

  

In the crowd scenery, an object would experience some “force” from other agents and objects.

- For instance, when a pedestrian is walking on the street, he would adjust his speed, direction and destination, in order to avoid collisions with others.

  

Another example is when a crowd of people walk across a street, each of them follows other people and guides others at the same time.

  

In fact, these are examples of two typical interaction models known as

- the social force models [100] and

- the crowd motion pattern models [101].

  

#### 3.3.1 Social Force Models

  

#### 3.3.2 Crowd Motion Pattern Models.

  

  

### 3.4 Exclusion Model

  

Exclusion is a constraint employed to avoid physical collisions when seeking a solution to the MOT problem.

  

이 가정은 물리적 환경에서 한 공간에 두 물체가 존재 할수 없기 때문이다. `It arises from the fact that two distinct objects cannot occupy the same physical space in the real world. `

  

Given multiple detection responses and multiple trajectory hypotheses,generally there are two constraints.

- 탐지 단계에서의 배제 : 두개의 탐지값은 동일한 대상에 할당 되지 못한다. `The first one is the so called detection-level exclusion [105], i.e., two different detection responses in the same frame cannot be assigned to the same target. `

- 추적 단계에서의 배제 : 두 경로는 일치 할수 없다. `The second one is the so-called trajectory-level exclusion, i.e., two trajectories cannot be infinitely close to each other.`

  

#### 3.4.1 Detection-level Exclusion Modeling

  

Different approaches are adopted to model the detection level exclusion. Basically, there are “soft” and “hard” models.

  

##### “Soft” modeling.

  

##### “Hard” modeling

  

  

#### 3.4.2 Trajectory-level Exclusion Modelin

  

  

### 3.5 Occlusion Handling

  

![](https://i.imgur.com/t2reTSu.png)

  

**가려짐**은 MOT에서 가장 큰 문제점이다. 이로 인해서 ID 교환이나 경로 단절 등의 문제가 발생 한다. 다양한 해결 방법이 제안 되었다. `Occlusion is perhaps the most critical challenge in MOT.It is a primary cause for ID switches or fragmentation of trajectories. In order to handle occlusion, various kinds of strategies have been proposed`

  

  

#### 3.5.1 Part-to-whole

  

가려짐이 발생 하여도 일부분은 보인다는 가정하에 진행 된다. 보이는 일부분으로 전체 모습을 추론 하여 동작 한다. `This strategy is built on the assumption that a part of the object is still visible when an occlusion happens. This assumption holds in most cases. Based on this assumption, approaches adopting this strategy observe and utilize the visible part to infer the state of the whole object. `

  

The popular way is dividing a holistic object (like a bounding box) into several parts and computing affinity based on individual parts. If an occlusion happens, affinities regarding occluded parts should be low. Tracker would be aware of this and adopt only the un-occluded parts for estimation. Specifically, parts are derived by dividing objects into grids uniformly [52], or fitting multiple parts into a specific kind of object like human, e.g. 15 non-overlap parts as in [49], and parts detected from the DPM detector [110] in [77], [111].

  

Based on these individual parts, observations of the occluded parts are ignored. For instance, part-wise appearance model is constructed in [52]. Reconstructed error is used to determine which part is occluded or not. The appearance model of the holistic object is selectively updated by only updating the unoccluded parts. This is the “hard” way of ignoring the occluded part, while there is a “soft” way in [49].

  

Specifically, the affinity concerning two tracklets j and k is computed as P where f is feature, i is the index of parts.

  

The weights are learned according to the occlusion relationship of parts. In [77], human body part association is conducted to recover the part trajectory and further assists whole object trajectory recovery. “Part-to-whole” strategy is also applied in tracking based on feature point clustering, which assumes feature points with similar motion should belong to the same object. As long as some parts of an object are visible, the clustering of feature point trajectories will work [62], [68], [112].

  

#### 3.5.2 Hypothesize-and-test

  

  

This strategy sidesteps challenges from occlusion by hypothesizing proposals and testing the proposals according to observations at hand. As the name indicates, this strategy is composed of two steps, hypothesize and test.

  

##### Hypothesize.

  

Zhang et al. [38] generate occlusion hypotheses based on the occludable pair of observations, which are close and with similar scale.

  

  

##### Test

  

The hypotheses would be employed for MOT when they are ready. Let us revisit the two approaches described above.

  

In [38], the hypothesized observations along with the original ones are given as input to the cost-flow framework and MAP is conducted to obtain the optimal solution.

  

In [114] and [113], a multi-person detector is trained based on the detection hypotheses.

  

This detector greatly reduces the difficulty of detection in case of occlusion.

  

#### 3.5.3 Buffer-and-recover

  

This strategy buffers observations when occlusion happens and remembers states of objects before occlusion.

  

When occlusion ends, object states are recovered based on the buffered observations and the stored states before occlusion.

  

Mitzel et al. [71] keep a trajectory alive for up to 15 frames when occlusion happens, and extrapolates the position to grow the dormant trajectory through occlusion.

  

In case the object reappears, the track is triggered again and the identity is maintained. This idea is followed in [34]. Observation mode is activated when the tracking state becomes ambiguous due to occlusion [115].

  

As soon a enough observations are obtained, hypotheses are generated to explain the observations. This could also be treated as “buffer-and-recover” strategy

  

#### 3.5.4 Others

  

The strategies described above may not cover all the tactics explored in the community.

  

For example, Andriyenko et al. [116] represent targets as Gaussian distributions in image space and explicitly model pair-wise occlusion ratios between all targets as part of a differentiable energy

function.

  

In general, it is non-trivial to distinctly separate or categorize various approaches to occlusion modeling, and in some cases, multiple strategies are used in combination.

  

---

  

## 3.6 Inference

  

> 프레임간 물체의 유사도를 기반으로 어떻게 **식별자**를 부여할 것인가?

  

### 3.6.1 Probabilistic Inference

  

Approaches based on probabilistic inference typically represent states of objects as a distribution with uncertainty.

  

추적 알고리즘의 목적은 탐지된 결과물들을 기반으로 물체 상태의 확률적 분포를 추론 하는것이다. `The goal of a tracking algorithm is to estimate the probabilistic distribution of target state by a variety of probability reasoning methods based on existing observations. `

  

여기서 **탐지된 결과물(existing)**만 사용하므로 온라인 러닝 방식에 적합하다. `This kind of approaches typically requires only the existing, i.e. past and present observations, thus they are especially appropriate for the task of online tracking. `

  

추론을 위해 탐지된 결과물을 이용하기에 물체의 상태를 마코브 성향을 가진다고 가정한다. `As only the existing observations are employed for estimation, it is naturally to impose the assumption of Markov property in the objects state sequence. `

> 마르코프 성질은 과거와 현재 상태가 주어졌을 때의 미래 상태의 조건부 확률 분포가 과거 상태와는 독립적으로 현재 상태에 의해서만 결정된다는 것을 뜻한다

This assumption includes two aspects, recalling the formula in Section 2.1.
  
- First, the current object state only depends on the previous states. Further, it only depends on the very last state if the first-order Markov property is imposed, 

- Second, the observation of an object is only related to its state corresponding to this observation. In other words, the observations are conditionally independent

These two aspects are related to the **Dynamic Model** and the **Observation Model**, respectively. 
- The dynamic model corresponds to the tracking strategy, 
- while the observation model provides observation measurements concerning object states. 

The **predict step** is to estimate the current state based on all the previous observations. More specifically, the posterior probability distribution of the current state is estimated by integrating in the space of the last object state via the dynamic model. 

The **update step** is to update the posterior probability distribution of states based on the obtained measurements under the observation model.

According to the equations, states of objects can be estimated by iteratively conducting the prediction and updating steps. However, in practice, the object state distribution cannot be represented without simplifying assumptions, thus there is no analytical solution to computing the integral of the state distribution. 

Additionally, for multiple objects, the dimension of the sets of states is very large, which makes the integration even more difficult, requiring the derivation for approximate solutions.

Various kinds of probabilistic inference models haves been applied to multi-object tracking [36], [95], [117], [118], such as 
- Kalman filter [35], [37], 
- Extended Kalman filter [34] and 
- Particle filter [32], [33], [52], [93], [119], [120], [121], [122].

#### Kalman filter

  In the case of a linear system and Gaussian-distributed object states, the Kalman filter [37] is proved to be the optimal estimator. It has been applied in [35].  

#### Extended Kalman filter

  To include the non-linear case, the extended Kalman filter is one possible solution. It approximates the non-linear system by a Taylor expansion [34].  

#### Particle filter

Monte Carlo sampling based models have also become popular in tracking, especially after the introduction of the particle filter [10], [32], [33], [52], [93], [119], [120], [121]. 

This strategy models the underlying distribution by a set of weighted particles, thereby allowing to drop any assumptions about the distribution itself [32], [33], [36], [93].

  
### 3.6.2 Deterministic Optimization

As opposed to the probabilistic inference methods, approaches based on deterministic optimization aim to find the **maximum a posteriori (MAP)** solution to MOT. 

To that end, the task of inferring data association, the target states or both, is typically cast as an optimization problem. 

이 방식은 **오프라인 추적**에 적합하다. `Approaches within this framework are more suitable for the task of offline tracking because observations from all the frames or at least a time window are required to be available in advance. `

Given observations (usually detection hypotheses) from all the frames, these types of methods endeavor to globally associate observations belonging to an identical object into a trajectory. 

The key issue is how to find the **optimal association**. Some popular and well-studied approaches are detailed in the following.

#### Bipartite graph matching

By modeling the MOT problem as bipartite graph matching, two disjoint sets of graph nodes could be existing trajectories and new detections in online tracking or two sets of tracklets in offline tracking. 

Weights among nodes are modeled as affinities between trajectories and detections. Then either a greedy bipartite assignment algorithm [32], [111], [123] or the optimal Hungarian algorithm [31], [39], [58], [66], [124] are employed to determine the matching between nodes in the two sets.
  

#### Dynamic Programming

  Extend dynamic programming [125], linear programming [126], [127], [128], quadratic boolean programming [129], K-shortest paths [18], [42], set cover [130] and subgraph multicut [131], [132] are adopted to solve the association problem among detections or tracklets.

#### Min-cost max-flow network flow

  Network flow is a directed graph where each edge has a certain capacity. For MOT, nodes in the graph are detection responses or tracklets. Flow is modeled as an indicator to link two nodes or not. To meet the flow balance requirement, a source node and a sink node corresponding to the start and the end of a trajectory are added to the graph (see Figure 6). 
 
 One trajectory corresponds to one flow path in the graph. The total flow transited from the source node to the sink node equals to the number of trajectories, and the cost of transition is the negative log-likelihood of all the association hypotheses. Note that the globally optimal solution can be obtained in polynomial time, e.g. using the push-relabel algorithm. This model is exceptionally popular and has been widely adopted [18], [38], [41], [43], [90], [133].

#### Conditional random field

The conditional random field model is adopted to handle the MOT problem in [1], [59], [105], [134]. Defining a graph G = (V, E) where V is the set of nodes and E is the set of edges, low-level tracklets are given as input to the graph. Each node in the graph represents observations [105] or pairs of tracklets [59], and a label is predicted to indicate which track the observations belongs to or whether to link the tracklets.
  

#### MWIS

  The maximum-weight independent set (MWIS) is the heaviest subset of non-adjacent nodes of an attributed graph. As in the CRF model described above, nodes in the attribute graph represent pairs of tracklets in successive frames, weights of nodes represent the affinity of the tracklet pair, and the edge is connected if two tracklets share the same detection. Given this graph, the data association is modeled as the MWIS problem [46], [97].

  

  

### 3.6.3 Discussion

  

현실에서는 확률적 방법보다 결정론적 방법이 더 많이 사용된다. `In practice, deterministic optimization or energy minimization is employed more popularly compared with probabilistic approaches. `

  

비록 확률적 방법이 좀더 직관적이고 완벽한 해결책으로 보이지만 대부분의 경우 infer가 제대로 수행되지 않는다. `Although the probabilistic approaches provide a more intuitive and complete solution to the problem, they are usually difficult to infer. `

  

반면에 **energy minimization**는 제한된 시간동안 나쁘지 않은 결과를 도출 해낸다. `On the contrary, energy minimization could obtain a “good enough” solution in a reasonable time.`


