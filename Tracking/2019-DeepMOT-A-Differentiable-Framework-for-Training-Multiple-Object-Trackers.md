
# [DeepMOT: A Differentiable Framework for Training Multiple Object Trackers](https://arxiv.org/pdf/1906.06618.pdf)


추적기법의 평가 항목은 **MOTA**와 **MOTP**이다. 이 두 항목에 기반한 추적기를 직접 최적화 하는것은 어렵다. 이유근 이 방식은 헝가리언 알고리즘에 기반하고 있고  헝가리언 알고리즘은 미분 불가능하기 때문이다. `Multiple Object Tracking accuracy and precision (MOTA and MOTP) are two standard and widely-used metrics to assess the quality of multiple object trackers. They are specifically designed to encode the challenges and difficulties of tracking multiple objects. To directly optimize a tracker based on MOTA and MOTP is difficult, since both the metrics are strongly rely on the Hungarian algorithm, which are non-differentiable. 
`

We propose a differentiable proxy for the MOTA and MOTP, thus allowing to train a deep multiple-object tracker by directly optimizing (a proxy of) the standard MOT metrics. The proposed approximation is based on a bidirectional recurrent network that inputs the object-to-hypothesis distance matrix and outputs the optimal hypothesis-to-object association, thus emulating the Hungarian algorithm. Followed by a differentiable module, the estimated association is used to compute the MOTA and MOTP. The experimental study demonstrates the benefits of this differentiable framework on two recent deep trackers over the MOT17 dataset. Moreover, the code is publicly available from https://gitlab.inria.fr/yixu/ deepmot.


## 1. Introduction

Object tracking is one of the core scientific challenges of computer vision. In the recent past, thanks to the advances of neural networks, great progress has been achieved in object tracking [42, 28, 46, 27, 11, 10]. Most of the research in visual tracking deals with single object tracking (SOT), where the main difficulties are (i) the properly modeling of the target dynamics and (ii) the learning of a robust appearance model. In addition to the challenges of the singleobject tracking, the complexity of multiple object tracking is further characterized by the data-to-track assignment problem [5]. Indeed, when dealing with tracking of multiple objects, one needs to associate data points (i.e. detections) to each of the tracks. Data association is essential not only at inference time, where the detections must be associated to different tracks, but also at evaluation time (training and performance evalution), where the inferred tracks have to be associated to the ground-truth. We will rather focus in the latter. Furthermore, these assignment problems are combinatorial and global. Combinatorial, because there is a number of assignment possibilities exponentially growing with the number of elements to be associated (tracks) and polynomially growing with the number of elements to be associated to (ground-truth objects). Global, because the decision has to be taken, considering the distance between all inferred tracks and all ground-truth objects. Moreover, both the number of inferred tracks and ground-truth objects vary over time, and therefore any assignment method must be able to cope with input distance matrices of variable size.

Traditionally, this generic assignment problem has been addressed with the Hungarian or Munkres algorithm [22]. In the case of MOT, a modified version of the Hungarian algorithm is used to assign tracks to ground-truth objects [6]. On one side, there can be a different number of tracks and ground-truth objects. On the other side, when the distance between a track and a ground-truth object exceeds a threshold, we should avoid this assignment (see [6] for more details). Therefore, not all tracks will be associated to a ground-truth object, and not all ground-truth objects will have an associated track. In an extreme case, there could be no associations at all at a given frame. What the standard and the modified Hungarian algorithms have in common is that the optimal assignment cannot be expressed as a differentiable function of the input distance matrix. This is problematic when one wishes to use MOTA or MOTP as training loss for a deep multiple object tracker. Current deep trackers are trained with ad-hoc differentiable losses that have little or nothing to do with MOTA or MOTP.

In this paper, we introduce a differentiable operator that approximates the Hungarian algorithm like in [6]. To that aim, we use recurrent neural networks, and we call it deep Hungarian network (DHN). Once trained, the DHN can be used in two different ways. 
- Firstly, to provide an approximation of the optimal track-to-ground-truth assignment, that is differentiable w.r.t. the distance matrix (and thus to approximate MOTA and MOTP to directly train deep MOT). 
- Secondly, to convert any fully trainable deep singleobject tracker, into a deep multiple object tracker. Indeed, the tracks can be inferred from a purely MOT method, from several SOT methods running in parallel, or from any combination of MOT and SOT methods working in parallel.

The rest of the paper is structured as follows. We first discuss works related to ours. We then present the structure of the overall pipeline, including the deep Hungarian network (DHN) and the operators to compute the MOTA and MOTP metrics. After, we evaluate the usefulness of the proposed pipeline in several ways. First, the advantages of the proposed training framework compared with standard losses are shown and discussed. Second, we show that thanks to the proposed training framework, a single-object tracker can achieve state-of-the-art multiple object onlinetracking performance.

## 2. Related Work

### 2.1. Single Object Tracking

SOT는 식별모델 학습을 통해서 물체를 추적 하고, 배경으로 부터 분리 하는것에 초점을 두고 있다. `Single object tracking (SOT) methods [11, 10] aim to learn discriminative models to track one object and separate it from the background. `

[15]에서는 물체의 모션과 외형을 학습하는 간단한 회귀네트워크를 사용하였다. `A simple feed-forward regression network learning a generic relationship between object motion and appearance was proposed in [15]. `

```
[15] D. Held, S. Thrun, and S. Savarese. Learning to track at 100fps with deep regression networks. In European Conference Computer Vision (ECCV), 2016.
```

 siamese trackers는 최근에 좋은 성능을 보이고 있다. `Moreover, siamese trackers [42, 28, 46, 27] have recently achieved state-of-the-art performance. `
 - 오리지널 : The original siamese tracker was proposed in [42], which is based on a correlation filter learner. It is able to extract deep features that are tightly coupled with the correlation filter. 
 - 확장 모델 : An extension of the siamese tracker using region proposal networks (RPN) was introduced in [28]. It employs the siamese subnetwork for feature extraction and the RPN to perform classification and regression. 

```
[42] J. Valmadre, L. Bertinetto, J. Henriques, A. Vedaldi, and P. H. Torr. End-to-end representation learning for correlation filter based tracking. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 2805–2813, 2017
[28] B. Li, J. Yan, W. Wu, Z. Zhu, and X. Hu. High performance visual tracking with siamese region proposal network. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 8971–8980, 2018.
[46] Z. Zhu, Q. Wang, B. Li, W. Wu, J. Yan, and W. Hu.Distractor-aware siamese networks for visual object tracking. In Proceedings of the European Conference on Computer Vision (ECCV), pages 101–117, 2018.
[27] B. Li, W. Wu, Q. Wang, F. Zhang, J. Xing, and J. Yan. Siamrpn++: Evolution of siamese visual tracking with very deep networks. arXiv preprint arXiv:1812.11703, 2018.
```

최근의  SOT 트래커들은 end-to-end로 동작하지만 다중 물체 추적으로 넘어 가면 성능이 좋지 않다. `Most of recent SOT trackers can be trained end-to-end, but their performance significantly drops when applied directly to MOT.`

### 2.2. Multiple Object Tracking 

MOT 는 tracking-by-detection 패러다임을 많이 따른다. MOT는  **data association** 문제 해결에 초점을 두고 있다. `Multiple object tracking often follows the tracking-by-detection paradigm. Unlike single object tracking, the goal of a MOT tracker is to solve the data association problem. Standard benchmarks are proposed in [25, 31] for pedestrians tracking.` 미래 정보를 이용하느냐에 따라서 MOT는 온라인과 오프라인 추적으로 나눌수 있다. `Based on whether the algorithms use future information, MOT methods can be split into online and offline tracking.`

```
[25] L. Leal-Taixe, A. Milan, I. Reid, S. Roth, and K. Schindler. ´MOTChallenge 2015: Towards a benchmark for multitarget tracking. arXiv:1504.01942 [cs], Apr. 2015. arXiv:1504.01942.
[31] A. Milan, L. Leal-Taixe, I. Reid, S. Roth, and K. Schindler. ´MOT16: A benchmark for multi-object tracking. arXiv:1603.00831 [cs], Mar. 2016. arXiv: 1603.00831.
```

#### A. Offline MOT

- [39] formulates the multi-person tracking by a multi-cut problem and use a pair-wise feature which is robust to occlusions. 

```
[39] S. Tang, B. Andres, M. Andriluka, and B. Schiele. Multiperson tracking by multicut and deep matching. In European Conference on Computer Vision, pages 100–111. Springer,2016.
```

- Person re-identification methods have been combined into a tracking framework in [40]. 
```
[40] S. Tang, M. Andriluka, B. Andres, and B. Schiele. Multiple people tracking by lifted multicut and person reidentification. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 3539–3548, 2017.
```


- Moreover, [17] solves the problem by co-clustering the low-level feature point motions from optical flow and the high-level bounding-box trajectories. 
```
[17] M. Keuper, S. Tang, B. Andres, T. Brox, and B. Schiele.Motion segmentation & multiple object tracking by correlation co-clustering. IEEE transactions on pattern analysis and machine intelligence, 2018.
```


- Quadruplet convolutional neural networks are used in [38]. They perform metric learning for target appearances together with the temporal adjacencies, which is then used for data association. 
```
[38] J. Son, M. Baek, M. Cho, and B. Han. Multi-object tracking with quadruplet convolutional neural networks. In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 5620–5629, 2017.
```


- In addition, [19] proposes a bilinear LSTM to learn the long-term appearance models, where the memory and the input have a linear relationship. 
```
[19] C. Kim, F. Li, and J. M. Rehg. Multi-object tracking with neural gating using bilinear lstm. In Proceedings of the European Conference on Computer Vision (ECCV), pages 200–215, 2018.
```


- An iterative multiple hypothesis tracking (MHT) is proposed in [37], which includes the prior association information from the previous frames. 
```
[37] H. Sheng, J. Chen, Y. Zhang, W. Ke, Z. Xiong, and J. Yu. Iterative multiple hypothesis tracking with tracklet-level association. IEEE Transactions on Circuits and Systems for Video Technology, 2018.
```


- [16] fuses both the head and full-body detection into one tracking framework. 
```
[16] R. Henschel, L. Leal-Taixe, D. Cremers, and B. Rosenhahn. ´Fusion of head and full-body detectors for multi-object tracking. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops, pages 1428–1437,2018.
```


- Another approach is proposed in [24], which introduces a Siamese network. It encodes both the appearance information from the RGB image and the motion information from the optical-flow map. The obtained features are then processed by a linear programming based tracker.
```
[24] L. Leal-Taixe, C. Canton-Ferrer, and K. Schindler. Learning by tracking: Siamese cnn for robust target association.In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops, pages 33–40, 2016. 
```


#### B. Online MOT

- [1, 4] formulate the problem in a probabilistic framework, then use a **variational expectation maximization** algorithm to find the track solution. 
```
[1] S. Ba, X. Alameda-Pineda, A. Xompero, and R. Horaud. An on-line variational bayesian model for multi-person tracking from cluttered scenes. Computer Vision and Image Understanding, 153:64–76, 2016.
[4] Y. Ban, S. Ba, X. Alameda-Pineda, and R. Horaud. Tracking multiple persons based on a variational bayesian model. In European Conference on Computer Vision, pages 52–67. Springer, 2016.
```


- Moreover, [9] proposes an aggregated local flow descriptor which encodes the relative motion pattern and then performs tracking. 
```
[9] W. Choi. Near-online multi-target tracking with aggregated local flow descriptor. In Proceedings of the IEEE international conference on computer vision, pages 3029–3037, 2015.
```


- Another solution is proposed in [43], which uses **Markov Decision Processes** and **reinforcement learning** for the best data association. 
```
[43] Y. Xiang, A. Alahi, and S. Savarese. Learning to track: Online multi-object tracking by decision making. In Proceedings of the IEEE international conference on computer vision, pages 4705–4713, 2015.
```


- Alternatively, [35] presents a framework based on Recurrent Neural Networks (RNN). 
	- The dynamics of the appearance change, motion, and people interactions are modeled independently by a RNN. 
	- Then different information are fused together with a convolutional network. 
```
[35] A. Sadeghian, A. Alahi, and S. Savarese. Tracking the untrackable: Learning to track multiple cues with long-term dependencies. In Proceedings of the IEEE International Conference on Computer Vision, pages 300–311, 2017.
```


- Besides, a model with dual matching attention networks is introduced by [45], which uses both spatial and temporal attention mechanisms. 
```
[45] J. Zhu, H. Yang, N. Liu, M. Kim, W. Zhang, and M.-H. Yang. Online multi-object tracking with dual matching attention networks. In Proceedings of the European Conference on Computer Vision (ECCV), pages 366–382, 2018.
```


- The estimation of the detection confidence is realized in [2], where the detections with different scores are then clustered into different classes and they are processed separately. 
```
[2] S.-H. Bae and K.-J. Yoon. Confidence-based data association and discriminative deep appearance learning for robust online multi-object tracking. IEEE transactions on pattern analysis and machine intelligence, 40(3):595–610, 2018.
```


- [32] proposes a recurrent neural network (RNN) based method for both motion dynamics and detection-to-track data association and further explored it to NP-hard problems [33]. 
```
[32] A. Milan, S. H. Rezatofighi, A. Dick, I. Reid, and K. Schindler. Online multi-target tracking using recurrent neural networks. In Thirty-First AAAI Conference on Artificial Intelligence, 2017.
```


- Finally, in [36] an optical-flow based approach is proposed.
```
[36] S. Schulter, P. Vernaza, W. Choi, and M. Chandraker. Deep network flow for multi-object tracking. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pages 6951–6960, 2017.
```


### 2.3. Single vs. Multiple Object Trackers

end-to-end입장에서 SOT는 어느정도 연구가 되었지만, MOT는 아니다. SOT의 end-to-end를 MOT에 적용하는것도 쉽지 않다. `Multiple object tracking frameworks have the clear advantage to be able to address the MOT problem, but most of the existing trackers are not trainable end-to-end. On the contrary, the literature on end-to-end trainable single object trackers is much more mature, but adapting to multiple object tracking is still unclear and tracker-dependent. `

본 논문에서는 Hungarian algorithm과 유사한 RNN을 제안 한다. `In this paper, we propose a deep proxy for the standard MOT metrics: a deep recurrent network that emulates the Hungarian algorithm.`

흥미로운 점은 제안 방식을 이용하면 SOT의 end-to-end방식을 MOT에 적용 할수 있게 한다. `Interestingly, this allows us to directly use – and train – end-to-end trainable single object trackers, for multiple object tracking. Therefore, our formulation allows us to take full advantage of the maturity of the single object tracking literature and to train these trackers in an end-to-end fashion, to tackle the multiple object tracking problem.`

## 3. Problem Formulation

The objective of MOT is to predict the trajectories of all objects at each time step, including their bounding boxes and associated identities.
- First, the trajectory should be precise in the sense that each bounding box should enclose its associated object well. 
- Second, the trajectories should be accurate, meaning that only real objects and all of them should be captured (no clutter objects or missed objects) by the bounding boxes; and each trajectory should always have a unique and consistent identity through time. These properties are respectively measured by MOTP and MOTA.

MOT에 딥러닝을 적용하는 방법으로 움직임, 외형, 상호 동작에 대한 강건한 모델 생성에 초점을 맞추었다. `With the emergence of deep learning, people address the MOT problem with (deep) neural networks, mainly by constructing robust models that capture information about motion, appearance and/or object interactions [35, 45]. `

그러나 end-to-end 학습입장에서 적합한 loss function에 대한 연구는 진행 되지 않았다. 본 논문에서는 MOTA,MOT성능 인자를 제안 하였다. `However, an end-to-end training framework with a dedicated loss function for MOT remains undiscovered. As our first contribution, we propose a differentiable approximation of the two common MOT performance metrics, MOTA and MOTP, so that any trainable MOT method can be optimised to maximise these (now differentiable) metrics.`

As an intermediate component for calculating the MOT metrics, the Hungarian algorithm of O(n 3 ) time complexity give an approximated solution to the linear sum assignment problem (LSAP) by minimizing the sum distance of assigning ground-truth bounding boxes to the predicted ones. Given the algorithmic nature of this operation, no gradient of the metrics with respect to the predicted bounding boxes can be computed. And therefore the standard MOTA and MOTP cannot be used as optimisation criteria. To solve this problem, we propose a Deep Hungarian Network, or DHN, which can be seen as a differentiable function that approximates the Hungarian algorithm. In this way, the full pipeline is differentiable and the gradient of the (approximated) MOT loss can be back-propagated to any trainable MOT methods.

## 4. Methodology


