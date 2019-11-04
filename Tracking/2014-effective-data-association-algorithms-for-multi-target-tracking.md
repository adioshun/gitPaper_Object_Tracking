# [Effective Data Association Algorithms for Multitarget Tracking](https://macsphere.mcmaster.ca/bitstream/11375/16272/2/thesis%20-%20Biruk%20Habtemariam.pdf)

## 2. Background

---

### 2.1 System Model


#### 2.1.1 Target Dynamics


비행기 같은 physical systems의 dynamics는 **state-space model**로 모델링 가능 하다. `The dynamics of physical systems such as airplanes, vessels, ballistic targets, etc., can be modeled using a state-space model [14][18][93]. `

With state-space modeling, a moving target can be represented as the transition in the state of a target driven by a process noise.

##### A. nonlinear model



##### B. linear time invariant system


#### 2.1.2 Observation Model


---

### 2.2 TBD vs DBT

센서 값에서 타겟의 상태를 추정 하는 방법은 2가지가 있다. `estimate the target state from the observation data`
- TBD
- DBT 


#### A. Track-Before-Detect (TBD)

-Raw신호를 직접 처리 하여 Track의 패턴을 찾는 방법 `One method to estimate the target state from the observation data is to directly process the raw signal and search for track patterns. `

- 신호가 약하고 배경 노이즈와 구분이 어려운 SNR시나리오에 적합 `This approach is effective for low Signal-to-NoiseRatio (SNR) scenario where signal from a target is weak and indistinguishable from the background noise [94]. `

- TBD methods use the entire measurement set of a sensor’s resolution cells and integrate tentative targets over multiple frames [95]. 

- 계산 부하가 크다. `As a result, TBD methods are computationally demanding in most cases.`


#### B. Detect-Before-Track

- 탐지 정보를 활용 `Another approach is to apply a threshold to the raw signal from the radar and extract the measurements (also referred as detection, contacts and radar plots in the literature) [14]. `

- 추출된 Measurement에서 track과 데이터 연동 작업을 수행 `Based on the extracted measurements, new tracks can be initialized or already initialized tracks can be associated to measurements and their states be updated using the associated measurements [18]. `

- As detection precedes the tracking process, this methods are collectively referred as Detect-Before-Track.

---

### 2.3 Data Association

DBT의 첫 단계는 radar에서 들어오는 raw 신호에 임계치를 적용 하는 것이다. `As discussed in the previous section, the initial step in Detect-Before-Track methods is to apply a threshold to the raw signal received from the radar. `

일반적인 임계치 방법은 **CFAR**이다. `The common thresholding technique is the Constant False Alarm Rate (CFAR) method [31]. `

이 방식은 적응형 방식으로 데이터에 따라 값이 다르다. `The CFAR is an adaptive thresholding approach, in which a constant false alarm rate can be achieved by applying a threshold level determined by sliding window neighbourhood resolution cells averaging technique [72]. As a result, the threshold level can go up and down from one resolution cell to another depending on the local clutter situation.`

Applying thresholds results in a discrete measurement set either from target or clutter distributed in the entire measurement space. 

따라서 측정값이 tart에서 온것인지, clutter에서 온것인지 측적값과 track을 연결하는 작업이 필요 하다. `Hence, a measurement-to-track data association is required to determine if a measurement is from a target or clutter [14]. `


대부분의 DA기술들은 후보를 줄이기 위해 **게이팅 **기법을 사용 한다. `Most data association techniques involve gating techniques in order to reduce the number of feasible measurement-to-track association. `


만약 이전 Track의 state와 covariance를 알고 있다면 **validation gate**를 생성 할수 있다. `If a track’s previous state and covariance is known, a measurement validation gate can be constructed around the predicted track position. `

간단한 방법은 gate요구 사항을 만족하는 영역을 설정 하는 방법이다. `The simplest method is to specify a regular region that will satisfy the gate requirement [18]. `

복작합 방법은 **n-dimensional ellipse**을 설정한 후 아래(=본문) 요구 사항을 만족 하는것을 선택 하는 것이다. `A more effective approach is to define an n-dimensional ellipse around the predicted track position and choose measurements that satisfy the condition`

> 자세한 공식 및 설명은 본문 참고 

![](https://i.imgur.com/H0I3muF.png)

위 본문의 공식의 thresholding과 gating 후에는 후보 measurement가 선별 된다. `Thresholding and gating yield the potential measurement candidates to be associated with a track.`

위 그림을 예로 4개의 측정치가 **predicted track positions** 주변에 있고, 이중 3개가 validation region안에 포함 되어 있다. `Referring back to Figure 2.2, there are four measurements in the vicinity of predicted track positions and three of them are in the validation region.`

 
가장 간단한 one-to-one 측정치와 트랙간의 연동 방법은 `The simplest data association techniques based on one-to-one measurement-to-track matching are`
-  the Nearest Neighbor Filter (NNF) and 
- Strongest Neighbor Filter (SNF) [18]. 

The NNF associates a track with the measurement closest to the predicted measurement among the validated measurements while the SNF associates the measurement with the strongest intensity (assuming amplitude information is
available). 


이 간단한 방법들은 반사 신호가 강하고 false alarm rate가 낮은 상황에서는 좋은 성능을 보인다. `These data association techniques are computationally efficient and perform reasonably in a scenario where the target return is very strong and the false alarm rate is low.`


그러나 아래 상황에서는 성능이 않좋다. `However, with degraded target observability, dense clutter and closely-spaced targets, such approaches begin to fall short [8] to resolve the measurement origin uncertainty. `
- degraded target observability, 
- dense clutter and 
- closely-spaced targets

해결책은 베이지안 기반 연관 기법이 좋다. `Under such conditions, a more practical approach to deal with measurement origin uncertainty to applying Bayesian association techniques.`

#### 2.3.1 Probabilistic Data Association

The Probabilistic Data Association Filter (PDAF) [15] [14], also referred as the all-neighbors data association filter [19], implements a Bayesian approach for data association. 

In PDAF, weights are assigned to the measurements based on probabilistic inference made on the number of measurements and location of the measurements relative to the predicted track state [14]. 

For example, for the scenario in Figure 2.2 each validated measurement, i.e., {z1(k), z2(k), z3(k)}, is assigned a weight in contrast to choosing a single measurement as in NNF and SNF methods. 

Track state is updated with the innovations from each measurements are combined according to the assigned weights. 

Whenever there are multiple targets close to each other, joint association events can be considered in order to resolve from which target a measurement is originated uncertainty as in Joint Probabilistic Data Association Filter (JPDA) [23][20]. 

The PDAF as well as its multitarget variant, JPDAF, assume that track/tracks are already uninitialized. 

Tracks can be initialized, for example, with two-point track initialization method [14]. 

Furthermore, in a more robust approach, target existence models can be incorporated into the PDA framework as in Integrated Probabilistic Data Association Filter (JIPDA) [28] and similarly to the JIPDA framework as in Joint Integrated Probabilistic Data Association Filter (JIPDA) [57]. 

With target existence model, the JIPDA handles time varying number of targets and track management tasks such as track initiation, confirmation and deletion.



#### 2.3.2 Multiple Hypothesis Testing

In the Multiple Hypothesis Testing (MHT) [19][78] approach a hypothesis will be generated and tested with the received measurements in the current scan or frame. 

For a given measurement the hypotheses could be the measurement is originated from one of initialized tracks, or is originated from a new target or is a false alarm [18]. 

A Bayesian approach will be used to compute the probabilities of each hypothesis. 

The valid hypotheses derived from sequences of measurements are evaluated and propagated over time, each of them generating a set of new hypotheses at every sample time k. 

단점 : The major drawback in implementing the MHT algorithm for practical applications is the exponential growth in the number of the assignment hypotheses as time of a scan and number of measurement increases. 

This leads to the development of several hypothesis pruning, hypothesis merging and gating techniques [18][29]. 


#### 2.3.3 Frame Based Assignment

Frame based assignment algorithms formulate the measurement-to-track assignment as a **global cost minimization problem**.

At a given time step, only the current frame (e.g. 2D assignment) or the current frame and previous frames (e.g. multiframe
assignment) can be considered for the association.

##### 가. 2D Assignment (eg. Hungarian algorithm)

In 2D assignment, at a time a single frame is used to associate the detection with tracks.

![](https://i.imgur.com/7rdl96c.png)
` z : 측정값 / x : Tracks `

The cost of measurement-to-track association is determined by the **negative loglikelihood ratio** of target-originated measurement likelihood to false alarm density [14].

...

For example, the Hungarian algorithm [48] and Jonker-Volgenant-Castanon (JVC) [40] algorithm solve the measurement-to-track association problem

...



##### 나. Multiframe Assignment

Multiframe assignment, also called multidimensional assignment, extends the 2D problem into optimization of assignment from sequence of frames [76][91] as shown in Figure 2.4.

![](https://i.imgur.com/ER9c9iK.png)

The multiframe assignment algorithm determins the most likely set of frames such that each measurements is assigned to one and only one track, or declared as false alarm, and each track receives at most one detection from each frame used for association. 

The multiframe assignment improves the association accuracy compared to the 2D assignment at the cost of increased computation and latency corresponding to the number of frames used in the association.

The main challenge in associating data from three or more sequence of frames is that the resulting optimization problem is NP-hard. 
- This issue is addressed by using a Lagrangian relaxation-based methods are used to successively solve the association
problem as a series of 2-D assignments [62][63][64][67]. 

Accordingly, the constraints in (2.20) are relaxed one set at a time there by solving the resulting subproblem iteratively and then reconstructing a feasible solution for the original multidimensional discrete optimization problem.


---

### 2.4 Filtering

Once a track is associated to a measurement, filtering methods can be used in order to estimate the current state of target. 

If no measurement is associated with a track, the track will be updated with the predicted state [18]. 

There are various filtering methods to estimate the current state of the target based on the associated measurement. 

One of the early filtering techniques is α−β filters [43] that use a fixed tracking coefficients.

#### 2.4.1 Kalman Filter


#### 2.4.2 Extended Kalman Filter


#### 2.4.3 Unscented Kalaman Filter


#### 2.4.4 Interactive Multiple Model


#### 2.4.5 Particle Filter


---


### 2.5 Random Finite Set Methods

The Probability Hypothesis Density (PHD) filter [10][11][89] is a Bayesian multitarget tracking estimator initially proposed in [56]. 

The PHD filter is developed based on the 
- Random Finite Set (RFS) theory, 
- point processes, and 
- Finite Set Statistics (FISST).

특징 : It estimates all the targets states at once, as a multitarget state, projected on the single-target space.

The PHD filter has been shown an effective way of tracking a time-varying multiple number of targets that avoids model-data association problems [56]. 

변형/개선  
- A Gaussian mixture implementation of PHD filter (GM-PHD) is presented in [89]. 
- For nonlinear measurements, the Sequential Monte Carlo (SMC) implementation of the PHD filter is presented in [10].


---

## 3. Multiple Detection Target Tracking



> 제안 아이디어 부분 인듯 : 하나의 물체가 어떠한 이유로 중복 탐지 결과를 생성 할경우 

### one detection per one target 

대부분의 탐지 기반 추적 기법들은 타겟은 최대 한개의 detection만 유발 한다고 가정 하고 있다. `Most detection-based target tracking algorithms assume that a target generates at most one detection per scan with probability of detection less than unity. `

이 경우 데이터 연동 문제는 탐지 측적 소스에 대한 불확실성 뿐이다. `In this case, the data association uncertainty is only the measurement origin uncertainty [14] [92]. `

그러므로 measurement중에서 하나만 타겟에서 나온것이고 나머지는 잘못된 것이다. `Thus, given a set of measurements in a scan, at most one of them can originate from the target and the rest have to be false alarms. `

이러한 가정으로 인해서 one-to-one 탐지값과 타겟 연도문제는 최적화나 enumeration문제로 해결 된다. `This basic assumption results in the formulation of one-to-one measurement-to-track association as an optimization or enumeration problem.`

예를 들어 PDA나 JPDA의 경우 ` For example, in the Probabilistic Data Association (PDA) filter[1][15][44][92] and its multitarget version, the Joint Probabilistic Data Association (JPDA) filter [2][20][57][71], presented in Chapter 2, `
- weights are assigned to measurements based on a Bayesian assumption that only one of the measurements is from the target and the rest are false alarms. 

다른 예로 MHT의 경우 Similarly, in the Multiple Hypothesis Tracker (MHT) [19][45][49][69] hypotheses are generated based on one-to-one measurement-to-track association. 

This assumption extends to the Multiframe Assignment (MFA) algorithm [76][91] since the measurement-to-track association is evaluated as one-to one combinatorial optimization in the best global hypothesis. 

위 두경우 모두 one-to-one 기본 fundamental이다. `In all these cases, the one-to-one assumption is fundamental for the correct measurement-to-track associations and accurate target state estimation.`



### Multiple detection per one target 


그러나 Target은 여러개의 Detection을 생성 할수도 있다.  `However, a target can generate multiple detection in a scan due to, for example, multipath propagation or extended nature of the target with a high resolution radar. `

When multiple detection from the same target fall within the association gate, the PDAF and its multitarget version, the JPDAF, tend to apportion the association probabilities, but still with the fundamental assumption that only one of them is correct. 

When the measurements are not close to one other, as in the case of multipath detection, the PDAF and JPDAF initialize multiple tracks for the same target. 

The MHT algorithm tends to generate multiple tracks to handle the additional measurements from the same target due to the basic assumption that at most one measurement originated from each target. 

Thus, an algorithm that explicitly considers multiple detection from the same target in a scan needs to be developed so that all useful information in the received measurements about the target is processed with the correct assumption. 

The presence of multiple detection per target per scan increases the complexity of a tracking algorithm due to uncertainty in the number of target-originated measurements, which can vary from time to time, in addition to the measurement origin uncertainty. 

However, estimation accuracy can be improved and the number of false tracks can be reduced using the correct assumption with multiple-detection.


> 이후 생략 필요시 참고 

### 3.1 Multiple-Detection Pattern


### 3.2 MD-PDAF and MD-JPDAF

