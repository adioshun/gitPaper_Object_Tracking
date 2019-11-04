# [자율 주행 차량의 다중 물체 추적 알고리즘](http://www.dbpia.co.kr/journal/articleDetail?nodeId=NODE06357975)

> 이재준, 한국기술교육대학교, 2015, Multi-object Tracking Algorithm for Autonomous Vehicle

군집화시키고 각각의 군집들 을 추적\(Tracking\)하면 물체의 크기, 속도 등과 같이 더 많은 정보를 추출하는 것이 가능하다

본 논문에서 소개하는 물체 추적 알고리즘의 구성

* 데이터 군집화\(Data Clustering\)
* 데이터 관계 분석\(Data Association\)
* 상태 추정\(State Estimation\)
* 데이터 정리\(Data Arrangement\)

### 2.1 데이터 군집화\(Data Clustering\)

우선 데이터를 시간 순서대로 나열하여 생성된 2D Depth Image 내에서 지역 탐색을 이용하여 빠르게 군집화\[1\]를 수행한다

False separations 을 제거하기 위하여 군집들을 병합하는 단계를 추가로 수행

* 병합과정은 전역 탐색으로 이루어진다

```
[1] J. B. Trevor, S. Gedikli, R. B. Rusu, H. I. Christensen ,“Efficient Organized Point Cloud Segmentation with Connected Components,”
```

### 2.2 데이터 관계 분석\(Data Association\)

데이터 관계 분석은 **과거의 데이터**를 통해 **예측한 데이터**와 **새롭게 관측된 데이터**의 관계를 분석하는 단계이다.

관계 분석 과정에는 2 가지 중요한요소가 있다.

* 특징\(Feature\)의 선택
  * 일반적으로 위치 특징이 주로 특징 벡터로 사용되며 높이, 길이, 너비 등이 추가적인 특징으로 사용된다.
  * 하지만 LIDAR 의 특성상 때때로 길이, 너비등과 같은 특징은 움직이는 물체들에서 심하게 변동하는 특성을 보이기 때문에 위치와 높이만을 사용하였다.
* 할당\(Assignment\) 전략
  * 모든 군집과 추적 물체조합의 거리의 합을 근사적으로 최소화시키는 GNN\(Global Nearest Neighbor\)의 근사\(Approximation\)방식을 사용하였다.
  * 할당 전략은 군집-추적물체 테이블에서 가장 낮은 거리의 조합부터 차례대로 할당해 나가는 전략\[2\]을 취하였다.
  * 헝가리언 알고리즘을 적용 하기도 한다. [\[상세\]](https://cafe.naver.com/opencv/50555)

```
[2] F. Bourgeois and J. C. Lassalle, “An Extension of the Munkres Algorithm for the Assignment Problem to Rectangular Matrices.” Communications of the ACM, vol. 14, no. 12, p. 802, December, 1971.
```

[\[추천\] Introduction to Data Association](http://www.cse.psu.edu/~rtc12/CSE598C/datassocPart1.pdf): ppt, CSE598C Fall, 2012, Bob Collins, CSE, PSU

### 2.3 상태 추정\(State Estimation\)

상태 추정 과정은 데이터 관계 분석을 통해 얻은 대응 관계를 이용하여 측정값\(군집\)으로 그에대응하는 추적 물체의 상태를 갱신하는 단계이다.

상태 추정은 **등속 선형 운동 모델**을 가정하여 수행되었고 상태 추정 알고리즘으로는 **칼만 필터**를 사용하였다\[4\].

상태 벡터는 위치와 속도만으로 구성된다.

### 2.4 데이터 정리\(Data Arrangement\)

데이터 정리 과정에서는 불필요한 데이터의 삭제와 새로운 데이터의 등록이 이루어진다.

데이터의 등록은 새로운 물체가 나타나는 즉시 실행되지만 삭제는 시간 지연을 두고 실행되어 잡음, 가려짐\(occlusion\)등에 의해 물체가 센서의 시야에서 사라지는 현상에 더 강인하게 동작할 수 있도록 하였다.

하지만 때때로 이런 지연전략이 추적 물체 목록을 비대하게 만들 수 있기 때문에 상대적으로 관측 횟수가 적은 물체에는 더 작은 시간 지연을 부여하여 불필요한 리소스 소모를 줄였다.
