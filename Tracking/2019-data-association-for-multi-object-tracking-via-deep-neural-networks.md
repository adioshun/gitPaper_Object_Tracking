# Data Association for Multi-Object Tracking via Deep Neural Networks

[https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6387419/pdf/sensors-19-00559.pdf](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6387419/pdf/sensors-19-00559.pdf)

 딥러닝의 발전으로 *tracking-by-detection*이 추적 기술에서 많이 활용 되었다. 이 방식은 DA가 중요하다. 본 논문에서는 딥러닝 기반의 DA에 대하여 제안 하였다. `With recent advances in object detection, the tracking-by-detection method has become mainstream for multi-object tracking in computer vision. The tracking-by-detection scheme necessarily has to resolve a problem of data association between existing tracks and newly received detections at each frame. In this paper, we propose a new deep neural network (DNN) architecture that can solve the data association problem with a variable number of both tracks and detections including false positives. `

The proposed network consists of two parts: encoder and decoder. 
- The encoder is the fully connected network with several layers that take bounding boxes of both detection and track-history as inputs. 
- The outputs of the encoder are sequentially fed into the decoder which is composed of the bi-directional Long Short-Term Memory (LSTM) networks with a projection layer. 
- The final output of the proposed network is an association matrix that reflects matching scores between tracks and detections. 

학습을 위해서는 스탠포드대 SDD를 사용하였다. 좋은 성능을 보였다. `To train the network, we generate training samples using the annotation of Stanford Drone Dataset (SDD). The experiment results show that the proposed network achieves considerably high recall and precision rate as the binary classifier for the assignment tasks. We apply our network to track multiple objects on real-world datasets and evaluate the tracking performance. The performance of our tracker outperforms previous works based on DNN and comparable to other state-of-the-art methods.`

## 1. Introduction

TBD(Track-by-Detection)이 추적 분야의 메인으로 떠오르고 있다. TBD는 DA를 사용해야 한다. 따라서 DA의 결과물은 연속적인 탐지물과 그 ID값이다. `Multi-object tracking is of great importance in computer vision for many applications including visual surveillance [1], robotics [2], and biomedical data analysis [3]. Although it has been extensively studied for decades, its practical usage for a real-world environment is still limited. Modern advances in object detection algorithms [4–8] in computer vision make the track-by-detection approach become the mainstream of multi-object tracking (MOT). MOT with track-by-detection necessarily exploits data association between existing tracks and new detections at each frame so that it forms trajectories of multiple objects. Thus, data association results produce sequences of detections with unique identities.`

DA를 해결하기 위한 많은 제안들이 있었다. :
- graph partitioning problem, 
- network flow-based methods, 
- exploit the appearance of object, 
- JPDA, 
- MHT, 
- stochastic filtering approaches 

`Many algorithms have been developed to solve data association problem in MOT. Several research works reformulated the problem as a graph partitioning problem and solved it using either binary integer programming or minimum cliques optimization [9–11]. Another group of recent research works uses network flow-based methods [12–14] that solve the problem by finding flows in their network. In addition, many tracking methods exploit the appearance of object to discriminate between objects [15–17]. There are also conventional approaches such as joint probabilistic data association (JPDA) [18,19] and multiple hypothesis tracking (MHT) [20–22] as well as stochastic filtering approaches [17,23].`

(최근에는 딥러닝 기반 방식이 제안되고 있으며) [24]에서는  LSTM기반 DA방식을 제안 하였다.  In [24], Milan et al. proposed data-driven approximations of the data association problem under recurrent neural network approach using Long Short Term Memory (LSTM) that approximates the marginal distributions of a linear assignment problem. They tested their method with simulated scenarios and showed that their method outperformed the JPDA [25] based methods. 
- 제약 : However, a limitation of their work is that it can process and produce fixed size of input and output. 

제안 방식에서는 *bi-directional LSTM*을 이용하여 문제 해결을 하였다. `In contrast, we propose a new method based on a bi-directional LSTM that sequentially processes inputs so that it is able to handle arbitrary-size data association problems. `

제안 방식의 구성 : The proposed network is comprised of two parts: encoder and decoder. 
- The encoder is a fully connected network with several layers that learns a feature representation of inputs (the position and size of detection bounding box). 
- The decoder is a bi-directional LSTM that can deal with the input sequence of variable size and help to learn from such data.

매 탐지시마다 네트워크 입력으로 두셋의 데이터를 확보하게 된다. (탐지값 & Tracks)`As new detection responses are received at every frame, we have two sets (i.e., a set of detections and a set of existing tracks) to arrange an input to the network. `

[그림 1b]처럼 탐지값(박스)와 track(화살표)는 합쳐진 형태로 입력으로 사용된다. `Then, the input of our network is formed by concatenating a detection with an existing track as illustrated in Figure 1b. `

현 프레임에서의 모든 가능한 *detection-to-track*  쌍 과 각 탐지에 대한 False alarm은 연쇄적 입력으로 형성되어 사용된다. `All possible pairs of detection-to-track at current frame and false alarm for each detection compose a sequence of inputs (a batch of training set). `

연쇄적인 값은 엔코더에서 사용된다. 엔코더의 구조는 아래와 같다. The sequence is consumed by the encoder. Each encoder is a fully connected network with several layers and produces encoded vectors that are sequentially used as inputs to the decoder of our network. The decoder is a bi-directional LSTM with a projection layer solving the association problem for each input in the sequence. 

학습결과로 각 입력이 긍정인지 부정인지 분류하여 association 결과로 출력한다. `Specifically, at the training time, it outputs a sequence of association results by classifying each input into either positive or negative assignment,` 테스트 결과로는 각 입력의 association quality를 점수로 출력한다. `while at the test time, it outputs a sequence of scores by measuring the quality of the association for each input.` 시퀀스들은 association (score) matrix로 재표현된다. `The sequence is reshaped to form an association (score) matrix. `

그림 1은 네트워크 구조와 입력을 나타내고 있다. 자세한 내용은 다음 장에 기술 되어 있다. `In Figure 1, an example of the training samples and architecture of the proposed network are described. We show the input pairs (rectangles and arrows) in Figure 1b to clearly specify the data flows. The proposed network is trained using generated samples, by using the ground-truth annotation of a Stanford Drone Dataset (SDD) [1]. We detailed training process in Section 3.2. Finally, the proposed network for data association is used for MOT. The detailed explanation of the MOT algorithm is given in Section 4.2.`


![](https://i.imgur.com/YUT8dq2.png)

```
Figure 1. This figure illustrates an example of training sample and model architecture of the proposed deep neural network: 
- (a) rectangles are detection bounding boxes in current video frame and each curved arrow represents an existing track; 
- (b) each encoder takes bounding boxes of both detection and history of existing track. 
- Then, the decoder reads encoded vectors one by one to generate the association matrix which is fed into the loss layer. 
- The rectangles and arrows in bracket refer to the input pairs in training sample. The same symbols in parentheses after the encoder show their origin (this figure is best viewed in color).
```

본 논문의 기여는 아래와 같다. `Contributions of this paper are as follows:`
- (1) We propose a new deep neural network that can the solve association problem with arbitrary-sized inputs; and 
- (2) we tested the proposed MOT algorithm based on the deigned deep neural network with the real-world datasets, e.g., SDD [1] and MOTChallenge [26]. 
	- The proposed network solves data association problems at every frame while it simultaneously produces trajectories. 
	- We argue that the result achieves an accuracy comparable to previous works that are similar to ours i.e., data association methods based on deep neural networks which do not exploit the appearance cue. 
- (3) The experiment demonstrates that the proposed network achieves considerably high accuracy as the binary classifier for the assignment tasks. 

본 논문의 구조는 아래와 같다. `The remainder of this paper is organized as follows.` 
- In Section 2, we review relevant previous works. 
- The detailed explanation of the proposed method is given in Section 3. 
- In Section 4, we state the implementation details and report the experiment results. 
- Finally, we conclude this paper in Section 5


## 2. Related Works

MOT알고리즘은 크게 온라인과 오프라인으로 구분 할수 있다. 학문적으로는 오프라인 방식이 좋은 성능을 보여 많이 사용된다. `MOT algorithms are largely classified into two categories: an offline method and online method. In literature, the offline method is getting popular due to superior performance compared to the online method. `

오프라인 방식은 연속적인 프레임을 입력으로 받는다. DA를 아래 알고리즘들로 해결한다. `The offline method takes a sequence of frames as its input. Then, data association for a batch of frames is solved by various optimization algorithms, e.g.,`
- network flow [13], 
- shortest path [12,14], 
- linear programming (LP) [27], and 
- conditional random field (CRF) [28]. 

그러나 실시간 시스템에 적용은 어렵다 `However, delayed outputs and complexity of the NP-hard (non-deterministic polynomial-time) problem limit its application for real-time requirements.`

반면에 확률적 필터링에 기반한 JPDA나 MHT가 최근 좋은 성능으로 호응을 얻고 있다. `On the other hand, the conventional approaches based on stochastic filtering such as JPDA [18] and MHT [21] are recently revisited and produce good results due to the good detection quality. `
- Rezatofighi et al. [25] propose an efficient approximation of JPDA to relax the combinatorial complexity in data association. 
- Kim et al. [21] demonstrate that the MHT framework can be extended to include online learned appearance models, resulting in performance gains

The solution of data association problem described above is obtained by optimizing the objective function. Accordingly, it is required to define an explicit model (e.g., appearance model and motion model) to compute the objective function. 

Our work is inspired by a series of deep neural network based detection and tracking of multi-objects [24,29–31] for the design of objective function. 

Hosang et al. [29] proposed a learning based non-maximum suppression using a convolutional neural network. The designed network takes bounding boxes of detection responses as input and output exactly one high scoring detection per object. The loss function penalizes double detections for one object during the training procedure. They proposed the GossipNet (Gnet) that jointly processes neighbouring detections so the network has necessary information to report whether an object was detected multiple times. 

Vinyals et al. [30] propose Pointer Network (Ptr-Net) that provides solutions for three different combinatorial optimisation problems (e.g., convex hull, Delaunay triangulation and the traveling salesman problem). Variable sized inputs are allowed in Ptr-Net. 

Milan et al. [31] present the end-to-end learning approach for online multi-object tracking using recurrent neural network (RNN). They test their method on real world dataset, MOTChallenge [27], but the performance is inferior to other existing methods. In addition, one drawback of their method is that objects are tracked independently ignoring interactions among objects because they compute the state estimation and data association for one object at a time. 

제안 방식과 유사한 선행 연구는 [24]이다. `The closely related work with ours is [24].` In [24], the solution of combinatorial problems (e.g., marginalisation for data association, feature point matching and the traveling salesman problem) is approximated with an LSTM network. 그러나 이 방식은 고정된 사이즈의 입력과출력이 있는 단점이 있다. `However, their method has one important limitation that it works only on the fixed input and output size.`

In practice, the size of data association problem varies with respect to the number of detections and objects that change over time. To handle this issue, we consider a data association problem for a sequence of inputs (a batch) and propose a network to process it sequentially. 따라서 제안 방식에서는 다양하 크기에 대하여 대처 할수 있다. `Hence, our method can learn to solve the data association problem with variable size.` Furthermore, we designed our network to consider the context of a sequence when it outputs an association score by using the bi-directional LSTM to exploit future and past information [32].

## 3. Problem Formulation
