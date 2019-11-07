# [The Probabilistic Data Association Filter]( http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.212.383&rep=rep1&type=pdf)

## 1. 개요  

DA불확실성은 센서의 측정값의 출처(origin)가 불확실할때 발생한다.  `Data association uncertainty occurs when remote sensing devices, such as radar, sonar, or electro-optical devices, yield measurements whose origin is uncertain,` that is, not necessarily the target of interest. 

이런 불확실성은 타겟의 신호가 약하거나, 잡음이 있거나, 여러개의 타겟이 목적 타겟 주변에 있을때 발생 한다. `This uncertainty occurs when the signal from the target is weak and, to detect it, the detection threshold has to be lowered, in which case background signals and sensor noise may also be detected, yielding clutter or spurious measurements. This situation can also occur when several targets are present in the same neighborhood. `

추적 필터가 이런 잘못된 측정값을 사용하게 되면 추정에러(`estimation error`)가 커져 결국 트랙을 놓치게 된다. `Using spurious measurements in a tracking filter leads to divergence of the estimation error and, thus, track loss. `

Consequently, 
- 먼저, 관심 타겟의 상태정보 업데이를 위해 추적 필터(eg. KF, EKF)에서 사용할 측정치를 선별 하는 작업이 중요하다. `the first problem is to select measurements to be used to update the state of the target of interest in the tracking filter, which can be a Kalman filter (KF) or an extended KF (EKF). `
- 다음, DA불확실성을 고려 하여 필터를 수정해야 할지 결정 해야 한다. `The second problem is to determine whether the filter has to be modified to account for this data association uncertainty. `

목적은 타겟 상태와 연동 불확실성에 대한 MMSE estimation을 구하는 것이다. `The goal is to obtain the minimum mean square error (MMSE) estimate of the target state and the associated uncertainty.`

[문서 구성] 추적기술의 연구 분야는 많다. 본 튜토리얼에서는 아래의 내용을 다룬다. `The literature on tracking targets in the presence of data association uncertainty is extensive [1]–[4]. This tutorial starts with `
- an illustration of the data association uncertainty stemming from the ambiguity as to which measurements are appropriate for use in the tracking filter. 
- Then we present a brief discussion on the optimal state-estimation algorithm in the MMSE sense in the presence of data association uncertainty and how other approaches stand in relationship to the optimum. 
	- The optimal estimator consists of the recursive computation of the conditional probability density function (pdf) of the state. 
	- The conditions under which this pdf is a sufficient statistic in the presence of data association uncertainty are detailed. 
- We then discuss several practical algorithms that carry out estimation and data association, namely, 
	- the probabilistic data association filter (PDAF), 
	- joint PDAF (JPDAF), 
	- mixture reduction PDAF (MXPDAF), 
	- particle filter (PF), and 
	- multiple hypothesis tracker (MHT), 
	- together with the approximations made by these algorithms. 
- PDAF and JPDAF are then described in detail along with two illustrative examples that show the reduced track loss probability of PDAF compared to KF in the presence of false measurements. 
- Finally, several real-world applications of PDAF and JPDAF are discussed, together with some lessons learned from the selection and design of tracking and data association algorithms.

  

## 2. THE DATA ASSOCIATION PROBLEM IN TARGET TRACKING

In target-tracking applications, the signal-detection process yields measurements from which the measurements to be incorporated into the target state estimator are selected. In a radar, the reflected signal from the target of interest is sought within a time interval determined by the anticipated range of the target when it reflects the energy transmitted by the radar. 

Gate의 범위가 설정 되고, 이 GATE안에있는 탐지값들이 관심 타겟으로  할당 된다.` A range gate is set up, and the detections within this gate can be associated with the target of interest. `

이런 측정값에는 여러 종류가 있다. `These measurements can consist of range, azimuth, elevation, or direction cosines, possibly also range rate for radar or active sonar; bearing, possibly also frequency, time difference of arrival, and frequency difference for passive sonar; line-of-sight angles or direction cosines for optical sensors. `

이후 다차원 Gate가 설정되어 타겟에서의 신호를 탐지 하게 된다. 이런 절차를 통해 광범위한 탐색 범위를 줄일수 있다. ` A multidimensional gate is then set up for detecting the signal from the target. This procedure avoids searching for the signal from the target of interest in the entire measurement space.`

However, a measurement in the gate, while not guaranteed to have originated from the target associated with the gate, is a valid association candidate, and such a gate is called a **validation region**.

유효 범위(`validation region`) 타겟 측정 값이 높은 확률로 범위 안에 들어 옴을 보장해준다. `The validation region is set up to guarantee that the target measurement falls in it with high probability, called the gate probability, based on the statistical characterization of the predicted measurement. `

Gate안에 하나 이상의 탐지값이 있다면 할당(`association`) 불확실성 문제에 빠진다. 말그대로 어떤 측정값을 타겟에서 발생한것인지 결정 해서 Track 업데이트를 수행 해야 한다. `If more than one detection appears in the gate, then we face an association uncertainty, namely, we must determine which measurement is target originated and thus to be used to update the track, which consists of the state estimate and covariance or, more generally, the sufficient statistic for this target. `

> update the track : the state estimate and covariance 

Measurements outside the validation region can be ignored because they are too far from the predicted measurement and thus unlikely to have originated from the target of interest. This situation occurs when the gate probability (the probability that the target-originated measurement falls in the gate) is close to unity and the statistical model used to obtain the gate is correct.

### 2.1 A Single Target in Clutter
 
 The problem of tracking a single target in clutter arises when several measurements occur in the validation region. The set of validated measurements consists of the correct measurement, if detected in this region, and the spurious measurements, which are clutter or false-alarm originated. In air traffic control systems, with cooperative targets, each measurement also includes a target identifier, called the squawk number. If the identifier is perfectly reliable, there is no data association uncertainty. However, a potentially hostile target is not expected to be cooperative, and then data association uncertainty is a problem.

![](https://i.imgur.com/4BopeAO.png)
```
[FIGURE 1] Several measurements z_i in the validation region of a single target. 
- The validation region is an ellipse centered at the predicted measurement z^ 1 . 
- Any of the shown measurements could have originated from the target or none if the target is not detected.
```

위 그림은 다수의 유효 측정값이 발생하는 경우이다. 2D 유효 범위는 타원형이고 중앙에 예측된 탐지값 z^ 가 있다. `A situation with several validated measurements is depicted in Figure 1. The two-dimensional validation region in Figure 1 is an ellipse centered at the predicted measurement z^.`

유효 범위가 타원인 이유와 생성 파라미터 : The elliptical shape of the validation region is a consequence of the assumption that the error in the target’s predicted measurement, that is, the innovation, is Gaussian. The parameters of the ellipse are determined by the covariance matrix S of the innovation.

유효 범위내 측정값들은 타겟에서 발생한것일수도 있다. 하지만 이중 하나만 실제 값이다. `All the measurements in the validation region may have originated from the target of interest, even though at most one is the true one.`

Consequently, the set of possible association events are the following:
- z1 originated for the target, and z2 and z3 are spurious; 
- z2 originated for the target, and z1 and z3 are spurious; 
- z3 originated for the target, and z2 and z1 are spurious; 
- all are spurious. 

The fact that these association events are mutually exclusive and exhaustive allows the use of the total probability theorem for obtaining the state estimate in the presence of data association uncertainty. 

Under the assumption that there is a single target, the spurious measurements constitute a random interference. A typical model for such false measurements is that they are uniformly spatially distributed and **independent** across time. This model corresponds to residual clutter. The constant clutter, if any, is assumed to have been removed.

### 2.2 Multiple Targets in Clutter

When several targets as well as clutter or false alarms are present in the same neighborhood, the data association becomes more complicated. 

![](https://i.imgur.com/ArKFZgo.png)
```
[FIGURE 2] Two targets with a measurement z2 in the intersection of their validation regions.
- The validation regions are the ellipses centered at the predicted measurements z^ 1 and z^ 2 . 
- Each of the measurements in the validation region of one of targets could have originated from the corresponding target or from clutter. 
- The measurement z2 in the intersection of the validation regions could have originated from either target or both.
```

Figure 2 illustrates this case, where the predicted measurements for the two targets are denoted as z^ 1 and z^ 2 . 

가능한 예측은 아래와 같다. `In Figure 2 three measurement origins, or association events, are possible,`
- z1 from target 1 or clutter; 
- z2 from either target 1 or target 2 or clutter; 
- and z4 and z5 from target 2 or clutter. 

However, if z2 originated from target 2 then it is likely that z1 originated from target 1. 

This situation illustrates the **interdependence** of the associations in a situation where a persistent interference from a neighboring target is present in addition to random interference or clutter. 

In this case, **joint association** events must be considered.

하지만 더 복잡한 상황이 발생 할수도 있다. Z2가 한곳에서 온것이 아니라 양쪽의 모두의 탐지 값이 겹치(merging)는 경우라면 어떨까. `A more complicated situation can arise as follows. Up to this point each measurement is assumed to originate from either one of the targets or from clutter. However, in view of the fact that any signal processing system has an inherent finite resolution capability, an additional possibility has to be considered, that z2 could be the result of the merging of the detections from the two targets, namely, that this measurement is an unresolved one. The unresolved measurement constitutes a fourth origin hypothesis for a measurement that lies in the intersection of two validation regions.`


The above discussion illustrates the difficulty of associating measurements to tracks. The full problem, as discussed below, consists of associating measurements at each time instant, updating the track sufficient statistic, and propagating it to the next time instant.

## 3. STATE ESTIMATION IN THE PRESENCE OF DATA ASSOCIATION UNCERTAINTY

When estimating the states of several targets using measurements with uncertain origin, it is not known which measurement originated from which target or from clutter. 

The goal is to obtain the MMSE estimate of the vector x, of known dimension, which might be the state of a single target or a stacked vector consisting of the states of several targets. 

DA불확실성을 가지는 물체 추적 기법 분류 `The approaches to target tracking in the presence of data association uncertainty can be classified into several categories, discussed below.`

### 3.1 Pure MMSE approach

The pure MMSE approach to tracking and data association is obtained using the smoothing property of expectations (see [5, Sec. 1.4.12]). In other words, the conditional mean of the state is obtained by averaging over all the association events 

### 3.2 MMSE-MAP Approach

The MMSE-MAP approach, instead of enumerating and summing over all the association events, selects the one with highest posterior, or maximum a posteriori (MAP), probability

### 3.3 MMSE-ML Approach 

The MMSE-ML approach does not assume priors for the association events and relies on the maximum likelihood (ML) approach to select the event,

### 3.4 Heuristic Approaches

간단한 휴리스틱 방법도 있다. 가장 간단한 방법으로는 **Mahalanobis** 거리 정보를 이용하는 것이다. `In addition to the above approaches, some simpler heuristic techniques are available. The simplest technique relies on the Mahalanobis distance metric, which is the square of the norm of the error with respect to its covariance, `
- see [5, Sec. 5.4], for sequential measurement-to-track association 
- and [3, Sec. 8.4], for track-to-track association. 

이방식은 NN규칙을 이용하여 각 측정치를 Association한다. `This approach associates each measurement with a track based on the nearest neighbor (NN) rule, which is a local or greedy selection [1], [4]. `

The same criterion can be used in a global cost function, which is minimized by an assignment algorithm using binary optimization, which can be the Jonker-Volgenant-Castanon algorithm ( JVC) [6] or auction [7]. 

These approaches result in the local and global nearest neighbor standard filter (NNSF) approaches, respectively. 

This filter is designated as standard because it assumes the assignment as perfect, without accounting for the possibility that it might be erroneous.


## 4. ESTIMATION AND DATA ASSOCIATION IN NONLINEAR DYNAMIC STOCHASTIC SYSTEMS

### 4.1 The Model



### 4.2 The Optimal Estimator for the Pure MMSE Approach


### 4.3 The Conditional Density of the State as the Information State


### 4.4 PRACTICAL ESTIMATORS

실용적인 DA를 위한 Estimator로는 PDAF와 JPDAF가 있다. `Two of the practical estimators that handle data association are PDAF for a single target in clutter and JPDAF for multiple targets. `

These estimators are based on the pure MMSE approach, but they are suboptimal. 

The suboptimality of PDAF follows from the fact that it approximates the conditional pdf of the target’s state at every stage as a Gaussian with moments matched to the mixture as in (5). 

The exact pdf under the linear-Gaussian assumption for the target and its measurement model, and with uniformly and independently distributed clutter (false measurements), is a Gaussian mixture with an exponentially growing number of terms. This mixture follows directly from (22).

Similarly,  JPDAF [3, Sec. 6.2] approximates each target state as an independent random variable with Gaussian pdf. 

- 확장 #1 : coupled JPDAF 가 좀더 최적화된 방법이다. The coupled JPDAF [3, Sec. 6.2.7] is closer to the optimum by approximating the joint pdf of the targets’ state vectors as a Gaussian. The effectiveness of this singleGaussian approximation of the Gaussian mixture is illustrated in [3] and by the fielded systems that utilize it, as described in the sequel.

- 확장 #2 : The approach of [9] is to improve on the PDAF by keeping a limited number of terms in the Gaussian mixture, resulting in the mixture PDAF (MXPDAF). 

- 확장 #3 : The particle filter approach is used in [10] to approximate the conditional-state pdf using a weighted sum of delta functions (particles). The resulting JPDA is shown to handle multiple maneuvering targets in clutter. The key is a suitable control that keeps the number of particles limited while approximating the state’s mixture pdf well enough.

The **MHT**, which makes hard association decisions over multiple frames of measurements, that is, on sequences of association hypotheses, 두가지 버젼이 있다. `has two versions`: 
- i) the hypothesis oriented MHT (HOMHT) [11] (see also [3]), which uses the MMSE-MAP approach, and 
- ii) the track oriented MHT (TOMHT) [12], [1], [4], which uses the MMSE-ML approach. For details, see [13].

두 버젼의 MHT는 공통적으로 가설이 증가하는 특징을 가지고 있다. 이 가설들은 Pruning작업이 필요 하다. Pruning작업은 ** sliding window**나 **낮은 probability/likelihood hypotheses**를 버리는 방식으로 수행 된다. `A common feature of both MHT algorithms is the exponentially increasing number of hypotheses, which necessitates pruning. This pruning is accomplished by a sliding window implementation as well as discarding low probability/likelihood hypotheses.`

The MHT uses a Gaussian posterior of the target states at the rear end of its sliding window, conditioned on the chosen hypothesis up to that point in time (MAP for the HOMHT, ML for the TOMHT), while it ignores all other association hypotheses. 

This use of a Gaussian posterior implicitly assumes a dominant hypothesis, namely, that the data association uncertainties from the past have been resolved. However, within its window, the MHT can carry a large number of association hypotheses, whose probabilities or likelihoods are recomputed, and the hypotheses are propagated in time as the window slides forward. 

Another advantage of the MHT is the fact that, by making hard decision associations, this method can use the measurements unassociated with existing tracks to start new tracks.


## 5. THE PDAF


### 5.1 Overview of PDA

PDA 알고리즘은 추적하는 물체와의 **association**확률을 계산한다. `The PDA algorithm calculates the association probabilities to the target being tracked for each validated measurement at the current time. `

이 확률값은 PDAF 추적 알고리즘에서 사용된다. `This probabilistic or Bayesian information is used in the PDAF tracking algorithm, which accounts for the measurement origin uncertainty. `

추적 물체의 상태나 측정식이 선형이라고 가정되면 PDAF는 KF를 비선형이면 EFK를 사용한다. `Since the state and measurement equations are assumed to be linear, the resulting PDAF algorithm is based on KF. If the state or measurement equations are nonlinear, then PDAF is based on EKF.`



### 5.2 Assumptions

7가지 

### 5. 3 Outline of the Algorithm

  

![](https://i.imgur.com/wt0zRo7.png)

  

위 그림은 PDAF의 순서도 이다. 몇가지 추가 모듈을 빼고는 칼만필터와 유사 하다. `Figure 3 summarizes one cycle of a PDAF, which is similar to KF with the following additional features: `

- 1) PDAF는 유효한 측정치를 선별하는 단계가 있음  `A PDAF has a selection procedure for the validated measurements at the current time.`

- 2) 매 측정마다. `For each such measurement, `

- **association probability**이 계산 되어 가중치 계산에 활용된다. `an association probability is computed for use as the weighting of this measurement in the combined innovation. `

- 결과는 상태 추정값 update에 활용된다. `The resulting combined innovation is used in the update of the state estimate;`

- this computation conforms to property P2 of the pure MMSE estimator even though P2 is conditioned on satisfying P1 exactly;

- nevertheless, P2 is still used for the sake of simplicity when P1 is satisfied approximately.

- 3) The final updated state covariance accounts for the measurement origin uncertainty.

  

The stages of the algorithm are presented next.

  
#### A. Prediction

#### B. Measurement Validation

#### C. Data Association

#### D. State Estimation

---

## 6. THE JOINT PROBABILISTIC DATA ASSOCIATION FILTER (JPDAF)

The joint probabilistic data association (JPDA) approach is the extension of the PDA. 

The following are the assumptions of the JPDA:
- 1) 타겟의 갯수를 알아야 한다. `The number of established targets in the clutter is known. `
- 2) Measurements from one target can fall in the validation region of a neighboring target. 
	- This situation can happen over several sampling times, and acts as a persistent interference. 
- 3) The past of the system is summarized by an approximate sufficient statistic consisting of state estimates, which are given by approximate conditional means, along with covariances for each target. 
- 4) The states are assumed to be Gaussian distributed with means and covariances according to the above approximate sufficient statistic. 
- 5) Each target has a dynamic and a measurement model as in (27), (28). 
	- The models for the various targets do not have to be identical.

PDAF models all incorrect measurements as random interference with uniform spatial distribution. 방해가 집중되면 성능이 급속도로 나빠진다. `The performance of PDAF degrades significantly when a neighboring target gives rise to persistent interference. `

동작 과정 `JPDAF consists of the following steps`:
- 1) The **measurement-to-target association probabilities** are computed jointly across the targets. 
- 2) In view of the assumption that a sufficient statistic is available, the association probabilities are computed only for the latest set of measurements. 
	- This approach conforms to the results from the section “The Optimal Estimator for the Pure MMSE Approach.” 
- 3) The state estimation is done 
	- **Decoupled** : either separately for each target as in PDAF, that is, in a decoupled manner, resulting in JPDAF,
	-  **Coupled** : or in a coupled manner using a stacked state vector, resulting in JPDACF.


The key feature of the JPDA is that it evaluates the **conditional probabilities** of the following joint association events ....(생략)


### 6.1 The Parametric and Nonparametric JPDA

두가지 버젼이 있다. `As in the case of the PDA, the JPDA has two versions, according to the model used for the pmf  of the number of false measurements.`


### 6.2 The State Estimation

The state-estimation or filtering algorithm can be carried out in two ways. 
- Assuming that the states of the targets conditioned on the past observations are mutually independent, the problem becomes one of decoupled estimation for the targets under consideration, that is, JPDAF. In this case the marginal association probabilities are needed. These marginal probabilities are obtained from the joint probabilities by summing over all joint events in which the marginal event of interest occurs.

The state-estimation equations are then decoupled among the targets and exactly the same as in PDAF, presented in (39)–(44).

### 6.3 JPDAF with Coupling

> 개선 방안 

The more realistic alternative assumption that the states of the targets, given the past, are correlated yields the JPDA coupled filter (JPDACF), which performs coupled estimation for the targets under consideration. 

The JPDA is developed assuming that, conditioned on the past, the target states and, thus, the target-originated measurements, are independent. 

Consequently, the joint association is followed by decoupled filtering of the targets’ states, which simplifies the resulting algorithm.

## 7. A SIMPLE EXAMPLE OF TRACKING IN CLUTTER



## 8. A MISSILE TRACKING EXAMPLE WITH SPACE-BASED SENSORS

## 9. CONCLUSIONS

DA는 타겟 track의  state estimate를 업데이트하기 위한 적합한 측정치는 선별하는 작업이며 필수 적이다. ` The measurement selection for updating the state estimate of a target’s track, known as data association, is essential for good performance in the presence of spurious measurements or clutter. `

분류 : A classification of tracking and data association approaches has been presented, 
- as a pure MMSE approach, which amounts to a soft decision, 
- and single best-hypothesis approach, which amounts to a hard decision.

It has been shown that the optimal state estimator in the presence of data association uncertainty consists of the computation of the conditional pdf of the state x(k) given all information available at time k, namely, the prior information about the initial state, the intervening known inputs, and the sets of measurements through time k. 

It has also been pointed out that if the exact conditional pdf, which is a mixture, is available, then its recursion requires only the probabilities of the most recent association events. 

The conditions under which this result holds, namely, whiteness of the noise, detection and clutter processes, were presented.


PDAF/JPDAF은 **DA**와 **state estimation**를 수행하여 이 예시에 대하여 설명 하였다. `The PDAF and JPDAF algorithms, which carry out data association and state estimation in clutter, have been described. A simple example was given to illustrate how the clutter and occasional missed detections can lead to track loss for a standard tracking filter, and how PDAF can keep the target in track under such circumstances. A simulated spacebased surveillance example showed, using Monte Carlo runs, that PDAF can track a target in a level of clutter in which the NNSF loses the target with high probability. `

The main approximation for both these algorithms is the **single Gaussian** used as the state’s conditional pdf. 

The fact that PDAF and JPDAF do not need to recompute past associations is a consequence of the assumption that the information state is available, even though the assumption of a Gaussian information state is only an approximate one. 

이 방법의 성능이 좋은건 실 생활에 적용되는걸로 확인 할수 있다. `The fact that this approximation is quite good is proven by the numerous successful real-world systems that use it. `

반면 MHT는 **hypothesis tree**를 이용하여 더 좋은 성능을 보이지만 복잡성과 자원 부하가 크다. `The MHT, on the other hand, uses an implicit finite mixture through its hypothesis tree, which is a better approximation of the exact information state. The complexity of the MHT in terms of computation time, memory, and code length/debugging is orders of magnitude higher than that of the PDAF/JPDAF.`

The numerous applications of the PDAF/JPDAF illustrated in “Real-World Applications of PDAF and JPDAF” show the potential pitfalls of using sophisticated algorithms for tracking in difficult environments as well as how to overcome them. The effect of finite sensor resolution can be a more severe problem than the data association [26] and deserves special attention.



-end-