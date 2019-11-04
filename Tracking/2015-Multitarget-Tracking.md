# [Multitarget Tracking]([http://ba-ngu.vo-au.com/vo/VMBCOMV_MTT_WEEE15.pdf](http://ba-ngu.vo-au.com/vo/VMBCOMV_MTT_WEEE15.pdf))


## Abstract

MTT는 약 50년전부터 연구되던 학문이다. `Multitarget tracking (MTT) refers to the problem of jointly estimating the number of targets and their states or trajectories from noisy sensor measurements. MTT has a long history spanning over 50 years, with a plethora of applications in many fields of study. ` 많이 사용되는 알고리즘은 다음과 같다. `While numerous techniques have been developed, the three most widely used approaches to MTT are the joint probabilistic data association filter (JPDAF), multiple hypothesis tracking (MHT), and random finite set (RFS).`
- The JPDAF and MHT have been widely used for more than two decades, 
- while the random finite set (RFS) based MTT algorithms have received a great deal of attention during the last decade. 

In this article, we provide an overview of MTT and succinct summaries of popular state-of-the-art MTT algorithms.

## I. INTRODUCTION

MTT는 오랜기간 여러 분야에서 사용되고 있다. `In a multitarget scenario the number of targets and their trajectories vary with time due to targets appearing and disappearing. For example, the location, velocity and bearing of commercial planes at an airport, ships in a harbour, or pedestrians on the street. multitarget tracking (MTT) refers to the problem of jointly estimating the number of targets and their trajectories from sensor data. Driven by aerospace applications in the 1960’s, MTT has a long history spanning over 50 years. During the last decade, advances in MTT techniques, along with sensing and computing technologies, have opened up numerous research venues as well as application areas. Today, MTT has found applications in diverse disciplines, including, air traffic control, surveillance, defence, space applications, oceanography, autonomous vehicles and robotics, remote sensing, computer vision, and biomedical research, see for example the texts [10], [15], [22], [73], [86], [96], [99], [144]. The goal of this article is to discuss the challenges in MTT and present the state-of-the-art techniques.`

MTT는 여러 노이지등의 챌리지들을 고려 하여햐 한다. `In this article we only consider the standard setting where sensor measurements at each instance have been preprocessed into a set of points or detections. The multitarget tracker receives a random number of measurements due to detection uncertainty and false alarms (FAs). Consequently, apart from process and measurement noises, the multitarget tracker has to contend with much more complex sources of uncertainty, such as measurement origin uncertainty, false alarm, missed detection, and births and deaths of targets. Moreover, in the multi-sensor setting, a multitarget tracker needs to process measurements from multiple heterogeneous sensors such as radar, sonar, electro-optical, infrared, camera, unattended ground sensor etc.`

MTT에 대한 많은 알고리즘이 있지만 본 논문에서는 **JPDAF, MHT and RFS**에 중점을 두어 살펴 보겠다. `A number of MTT algorithms are used at present in various tracking applications, with the most popular being the joint probabilistic data association filter (JPDAF) [10], multiple hypothesis tracking (MHT) [15], and random finite set (RFS) based multitarget filters [86], [96]. This article focuses on summarizing JPDAF, MHT and RFS as the three main approaches to MTT. The JPDAF and MHT approaches are very well established and make up the bulk of the multitarget tracking literature, while the RFS approach is an emerging paradigm. `


대부분의 알고리듬들은 (single-target) filtering후 Data Association을 수행 하는 형태로 이루어져 있다. `JPDAF and MHT as well as many traditional MTT solutions, are formulated via data association followed by (single-target) filtering.`

DA는 탐지 값을 Track이나 FA로 분류 한다. `Data association refers to the partitioning of the measurements into potential tracks and false alarms `
Filtering은 탐지값들을 기반으로 state를 추론한다. `while filtering is used to estimate the state of the target given its measurement history (note that algorithms that operate on pre-detection signals do not involve data association). `

RFS가 다른점은 **DA작업 없이** 최적 추론을 통해 state를 추론 한다. `The distinguishing feature of the RFS approach is that, instead of focusing on the data association problem, the RFS formulation directly seeks both optimal and suboptimal estimates of the multitarget state. Indeed some RFS-based algorithms do not require data association at all.`

논문의 구성 
- We begin by reviewing the fundamental principles of Bayesian estimation and summarizing some of the commonly used (single-target) filters for tracking in Section II. 
- Section III presents some background on the MTT problem and describes the main challenges, setting the scene for the rest of the article. 
- The JPDAF, MHT, and RFS approaches to MTT are presented in chronological order of developments in Sections IV, V, and VI respectively, with JPDAF being the earliest and RFS being the most recent. 
- Nonetheless, Sections IV, V, and VI can be read independently from each other.


## II. BAYESIAN DYNAMIC STATE ESTIMATION

During the last two decades significant progress has been made in **nonlinear filtering**. This section provides a brief overview of the Bayesian paradigm for nonlinear filtering.

### A. Bayesian Estimation

### B. The Bayes Recursion

### C. The Kalman Filter 

The Kalman filter (KF) is a closed form solution to the **Bayes (filtering) recursion** for linear Gaussian models [1], [8], [59], [64], [69], [126].

### D. The Gaussian Sum Filter

The Gaussian sum filter is a generalization of the Kalman filter to Gaussian mixture models [143].

### E. The Particle Filter

The particle or sequential Monte Carlo (SMC) method is a class of approximate numerical solutions to the Bayes recursion that are applicable to nonlinear non-Gaussian dynamic and observation models. 

The basis of the particle method is the use of random samples (particles) to approximate probability distributions of interest [4], [21], [44]–[46], [55], [126].


### F. Filtering Algorithms for Maneuvering Targets

The filtering algorithms discussed previously use a single dynamic model and hence are known as single-model filters. 

The motion of a maneuvering target involves multiple dynamic models. 
- For example, an aircraft can fly with a nearly constant velocity motion, accelerated/decelerated motion, and coordinated turn [8], [10]. 

The multiple model approach is an effective filtering algorithm for maneuvering targets in which the continuous kinematic state and discrete mode or model are estimated. 
- This class of problems are known as **jump Markov** or **hybrid state estimation** problems.

---

## III. MULTITARGET TRACKING

This section provides some background on the MTT **problem** and the main **challenges**, setting the scene for the rest of the article.

### A. Multitarget Systems

Driven by aerospace applications, MTT was originally developed for tracking targets from radar measurements. Fig. 1 shows a typical scenario describing the measurements by a radar in which five true targets are present in the radar dwell volume (the volume of the measurement space sensed by a sensor at a scan time) and six measurements are collected by the radar. We see from Fig. 1 that three target-originated measurements and three false alarms (FAs) are generated, one target is not detected by the radar, and two closely spaced targets are not resolved. This type of information regarding the nature and origin of measurements is not known for real radar measurements due to measurement origin uncertainty. At each discrete dwell/scan time tj , a set of noisy radar measurements with measurement origin uncertainty is sent to a tracker, as shown in Fig. 2.


In a general multitarget system, not only do the states of the targets vary with time, but the number of targets also changes due to targets appearing and disappearing as illustrated in Fig. 3. The targets are observed by a sensor (or sensors) such as radar, sonar, electro-optical, infrared, camera etc. The sensor signals at each time step are preprocessed into a set of points or detections. It is important to note that existing targets may not be detected and that FAs (due to clutter) may occur. As a result, at each time step the multitarget observation is a set of detections, only some of which are generated by targets and there is no information on which targets generated which detections (see Fig. 3).

- Standard multitarget transition model
- standard multitarget observation model

### B. The MTT Problem

The objective of MTT is to jointly estimate, at each observation time, the number of targets and their trajectories from sensor data. Even at a conceptual level, MTT is a non-trivial extension of single-target tracking. Indeed MTT is far more complex in both theory and practice.

The concept of estimation error between a reference quantity and its estimated values plays a fundamental role in any estimation problem. 

In (single-target) filtering the system state is a vector and the notion of state estimation error is taken for granted. For example, the EAP estimator minimizes the expected squared Euclidean distance ||ˆx − x|| 2 between the estimated state vector ˆx and true state vector x. However the concept of Euclidean distance is not suitable for the multitarget case. To see this consider the scenario depicted in Fig. 4. Suppose that the multitarget state is formed by stacking individual states into a single vector with the ground truth represented by X and the estimate represented by Xˆ. The estimate is correct but the Euclidean distance is ||Xˆ −X|| = 2. Moreover, when the estimated number of targets is different from the true number the Euclidean distance is not defined.

Central to Bayesian state estimation is the concept of Bayes risk/optimality [70], [128]. A Bayes optimal solution is not simply one that invokes Bayes rule. Criteria for optimality for the single-target case such as the squared Euclidean distance is not appropriate. In addition, the concept of consistency (of an estimator) cannot be taken for granted since it is not clear what is the notion of convergence in the multitarget realm.

From a practical point of view, MTT is not a simple extension of classical (single-target) filtering. Even for the simple special case with exactly one target in the scene, classical filtering methods (described in Section II) cannot be directly applied due to false detection, missed detection, and measurement origin uncertainty. 

The simplest solution is the **nearest neighbor (NN) filter** which applies the Bayes filter to the measurement that is closest to the predicted measurement [7], [10], [15]. 

A more sophisticated yet intuitively appealing solution is the **Probabilistic Data Association filter (PDAF)** which applies the Bayes filter to the average of all measurements weighted according to their association probabilities [7], [10]. 

The solution based on enumerating association hypotheses, proposed in [141], coincides with the Bayes optimal filter in the presence of false detections, missed detections, and measurement origin uncertainty proposed in [158]. In the multitarget setting, even for the special case where all targets are detected and no false detections occur, classical filtering methods are not directly applicable since there is no information on which target has generated which measurements.

#### global nearest neighbor (GNN) tracker

The simplest multitarget filter is the global nearest neighbor (GNN) tracker, an extension of the NN filter to the **multiple target** case. 

The GNN tracker searches for the unique joint association of measurements to targets that minimizes/maximizes a total cost, such as a total distance or likelihood. 

The GNN filter then performs standard Bayes filtering for each target using these associated measurements directly. 

Although the GNN scheme is intuitively appealing and simple to implement, it is susceptible to track loss and consequently exhibits poor performance when targets are not well separated [15].

#### JPDA 

The JPDAF [7], [10] is an extension of the PDAF to a fixed and known **number of targets**. 

The JPDAF uses joint association events and joint association probabilities in order to avoid conflicting measurement to track assignments in the presence of multiple targets. 

The complexity of the calculation for joint association probabilities grows exponentially with the number of targets and the number of measurements. 

Several approximation approaches have been proposed such as the deterministic strategies in [130], [131], [103], [61], [12], [169], [132] and the Markov Chain Monte Carlo (MCMC) based strategies in [112]. 

Moreover, since the basic JPDAF can only accommodate a fixed and known number of targets, several novel extensions have been proposed to accommodate an unknown and time varying number of targets, 
- such as the joint integrated PDAF (JIPDAF) [109] along with an efficient implementation [110], 
- and automatic track formation (ATF) [6]. 

Further detail on the JPDAF is given in Section IV.

#### MHT 

MHT [123], [76], [15], [16], [10], [101] is a deferred decision approach to data association based MTT. 

At each observation time, the MHT algorithm attempts to propagate and maintain a set of association hypotheses with high posterior probability or track score. 

When a new set of measurements arrives, a new set of hypotheses is created from the existing hypotheses and their posterior probabilities or track scores are updated using Bayes rule. 

In this way, the MHT approach inherently handles initiation and termination of tracks, and hence accommodates an unknown and time-varying number of targets. 

Based on the best hypothesis, a standard Bayes (or Kalman when the models are linear Gaussian) filter can be used on the measurements in each track to estimate the trajectories of individual targets. 

The total number of possible hypotheses increases exponentially with time and heuristic pruning/merging of hypotheses is performed to reduce computational requirements. 

Further details on the MHT approach is given in Section V.



## IV. JOINT PROBABILISTIC DATA ASSOCIATION FILTER

A. Overview
B. The Key Feature of the JPDAF
C. The Feasible Joint Events
D. Evaluation of the Joint Probabilities
E. The Parametric and Nonparametric JPDAF
F. The State Estimation
G. A Modification of the JPDAF: Coupled Filtering
H. Extension
I. The JPDAF — Summa

## V. MULTIPLE HYPOTHESIS TRACKI

A. Single Hypothesis and Multiple Hypothesis Tracking
B. Types of MHT Algorithms
C. Tree Based TOMHT
D. Non-tree Based TOMHT
E. Track Filtering
F. Applications of MHT
G. Future Work

## VI. THE RANDOM FINITE SET APPROACH
A. Random Finite Set
B. Multitarget State Space Model
C. Multitarget Bayes Recursion
D. The PHD Filter
E. The Cardinalized PHD Filter
F. The Generalized Labeled Multi-Bernoulli Tracker


