# [The Probabilistic Data Association Filter](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.212.383&rep=rep1&type=pdf)

IEEE CONTROL SYSTEMS MAGAZINE » DECEMBER 2009


# PDA

## 1. Overview of PDA

The PDA algorithm calculates the association probabilities to the target being tracked for each validated measurement
at the current time. 

This probabilistic or Bayesian information is used in the PDAF tracking algorithm, which accounts for the measurement origin uncertainty. 

Since the state and measurement equations are assumed to be linear, the resulting PDAF algorithm is based on KF. 

If the state or measurement equations are nonlinear, then PDAF is based on EKF. 

## 2. Assumptions

> 7가지 

## 3. Outline of the Algorithm 

![](https://i.imgur.com/wt0zRo7.png)

위 그림은 PDAF의 순서도 이다. 몇가지 추가 모듈을 빼고는 칼만필터와 유사 하다. `Figure 3 summarizes one cycle of a PDAF, which is similar to KF with the following additional features:  `
- 1) 현 순간의 **validated measurements**을 선택 단계 있음 `A PDAF has a selection procedure for the validated measurements at the current time.`
- 2) 매 측정마다. `For each such measurement, `
    - **association probability**이 계산 되어 가중치 계산에 활용된다. `an association probability is computed for use as the weighting of this measurement in the combined innovation. `
    - 결과는 상태 추정값 update에 활용된다. `The resulting combined innovation is used in the update of the state estimate;`
    - this computation conforms to property P2 of the pure MMSE estimator even though P2 is conditioned on satisfying P1 exactly; 
    - nevertheless, P2 is still used for the sake of simplicity when P1 is satisfied approximately.
- 3) The final updated state covariance accounts for the measurement origin uncertainty.

The stages of the algorithm are presented next. 


### 3.1 Prediction


> 상세 설명 추후 확인 필요 

---

# THE JOINT PROBABILISTIC DATA ASSOCIATION FILTER 