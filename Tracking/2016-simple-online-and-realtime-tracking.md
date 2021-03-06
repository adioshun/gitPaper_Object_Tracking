|논문명 |Simple online and realtime tracking|
| --- | --- |
| 저자\(소속\) | Bewley, Alex and Ge, Zongyuan and Ott, Lionel and Ramos, Fabio and Upcroft, Ben \(\) |
| 학회/년도 | 2016, [논문](https://arxiv.org/abs/1602.00763) |
| Citation ID / 키워드 | |
| 데이터셋(센서)/모델 |25 FPS(맥북 프로) |
| 관련연구||
| 참고 | [홈페이지](https://motchallenge.net/tracker/SORT), [Youtube](https://motchallenge.net/movies/ETH-Linthescher-SORT.mp4), [[추천]한글정리](https://jjeamin.github.io/paper/2019/04/25/sort/) |
| 코드 | [python/C++](https://github.com/abewley/sort), [Tracking-with-darkflow](https://github.com/bendidi/Tracking-with-darkflow), [Experimenting with SORT(Python)](https://github.com/ZidanMusk/experimenting-with-sort) |


|년도|1st 저자|논문명|코드|
|-|-|-|-|
|2016|Alex Bewley|[Simple Online and Realtime Tracking](https://arxiv.org/abs/1602.00763)|[깃허브](https://github.com/abewley/sort)|
|2017|Wojke|Simple Online and Realtime Tracking with a Deep Association Metric|[깃허브](https://github.com/nwojke/deep_sort)|
|2018|Wojke|Deep Cosine Metric Learning for Person Re-identification|-|




---

# [SORT(SIMPLE ONLINE AND REALTIME TRACKING)](https://jjeamin.github.io/paper/2019/04/25/sort/)

고전적이지만 매우 효율적인 두가지 방법인 Kalman ﬁlter,Hungarian method이 각각 tracking 문제의 움직임 예측과 데이터 연관요소를 처리하는데 사용된다.

이 논문에서 제안한 방법은
- detection의 주요 구성요소
- object의 상태를 미래의 frame에게 전달
- 현재 detection을 기존 object와 연결하고 tracking된 object 상태


> 간략한 코드 설명 포함 


---

# [Tracking Things in Object Detection Videos](https://lab.moovel.com/blog/tracking-things-in-object-detection-videos#3a-sort--simple-online-and-realtime-tracking)

> [깃허브](https://github.com/tdurand/node-moving-things-tracker)

## 1. Our problem

YOLO는 물체는 탐지 하지만 ID를 부여 하지 못함

## 2. First “intuitive” algorithm

```cpp

currentlyTrackedObjects = []
For each frame:
// 1. Try to match the currently tracked object with the detections of this frame
For each detections:
doesMatchExistingTrackedObject(detection, currentlyTrackedObjects)
// if it matches, update the trackedObject position
matchedTrackedObject.update(detection)
// 2. Assign unmatched detections to new objects
currentlyTrackedObjects.add(new trackedObject(detection))
// 3. Clean up unmatched tracked objects
For each currentlyTrackedObjects:
if isUnmatched(trackedObject):
trackedObject.remove()
```

`doesMatchExistingTrackedObject()`function :compare two detections, how to determine if they are tracking the same object ?

- `distance() function`: 두 탐지 대상의 위치를 비교하여 상대 거리를 계산 한다. `which compare two detections positions (current detections and candidate for next frame) and determine their relative distance. `
- 충분히 가까운 거리이며 같은 물체로 판별 한다. `If they are considered close enough, we can match them.`


```cpp
function distance(item1, item2) {
// compute euclidian distance between centers
euclidianDistance = computeEuclidianDistance(item1, item2)
if (euclidianDistance > DISTANCE_LIMIT):
// Do not match
else:
// Potential match
}
```

- 성능 및 제약 : This early implementation was already pretty good and matching correctly ~80% of the detections, but still had lots of re-assignments
- (when we lose track of the object and we assign it a new id even if it is the same object).

- 해결방안 : At that point we had some ideas on how to improve it:
1. By keeping a memory of the unmatched item for a few frames and avoid removing them directly
- (sometimes the detection algorithms miss the object for a few frames)
2. By predicting the position on the new frame with a velocity vector
3. By improving this distance function

### 2.1 Keep unmatched object in memory


바로 삭제 하는것이 아니라, 몇 프레임정도 기다렸다가 삭제 한다. `We first integrated the idea of keeping in memory the unmatched items for a few frames which is simply wait a few frames before removing it from the tracked items.`

```cpp

if isUnmatched(trackedObject):
trackedObject.unmatchedThisFrame()

## unmatchedThisFrame() 함수
function unmatchedThisFrame() {
nbUnmatchedFrame++
if nbUnmatchedFrames > 5:
//Effectively delete the item
this.remove()
}

```

이 방식을 통해 놓친 대상에 대한 회복 및 ID 재할당을 방지 한다. `This made the tracker more resilient to missing detections from YOLO and avoided some reassignments,`
- 그러나 효율적 적용을 위해서는 다음 위치를 예측 할수 있어야 한다. `but in order to have it more effective, we needed also to predict the next position.`

### 2.2 Predict position by computing the velocity vector

The idea behind this to be able to predict the next position of the tracked object if it is missing in the next frame, so it moves to its “theorical” position and will be more likely to be re-matched on the next frame.

```cpp

if isUnmatched(trackedObject):
trackedObject.predictNextPosition()
trackedObject.unmatchedThisFrame()
And the predictNextPosition() function looks like this

function predictNextPosition() {
x: trackedObject.x + velocityVector.dx,
y: trackedObject.y + velocityVector.dy
}

```

### 2.3 Improving the distance function

현 시점에서는 개선 방법이 떠오르지 않음 CV 연구를 통해 도출 예정 `At this point we didn’t have much clue on how to improve it, we felt the need to review some computer vision literature about tracking to make sure we were not missing some good ideas.`


## 3. Literature review


We found some useful papers for our problem statement. Check them out here:

- [High-Speed Tracking-by-Detection Without Using Image Information (Bochinski, Eiselein and Sikora) ](http://elvera.nue.tu-berlin.de/files/1517Bochinski2017.pdf)

- [Simple Online and Realtime Tracking (Bewley, Zongyuan Ge, Ott, Ramos, Upcroft)](http://arxiv.org/abs/1602.00763)

- [OpenCV Matching Faces Over Time (Dan Shiffman)](http://shiffman.net/general/2011/04/26/opencv-matching-faces-over-time/)


### 3.1 General things about tracking

We've identified two types of tracking:
- 단일 물체 추적(single object tracking): you choose one thing on the video and track it across all the frames
- 다중 물체 추적(multiple object tracking,MOT): track multiple object across all frames


And for each of them, there is two possibilities:
- 비실실간(Not real time tracking): algorithms that run on an existing video slower than real time
- (ie, takes more than a frame time to compute the next state of the tracker)
- 실시간(Real time tracking): the tracking algorithms that can run in real time
- (necessary for self-driving cars, robot ... )


We noticed that
- 실시간 알고리즘은 the real time algorithms use only the **detections inputs** (done by YOLO for ex),
- 비실시간 알고리즘은 the non real time trackers use information from the **image frames** to get more data to do the tracking.

> 대부분의 연구는 단일 물체 추적에 초점을 두고 있음 `Also most of the papers and projects published are focusing on single object tracking and not MOT`


Historically there was almost no algorithm working only on the detections output only as the detections weren't as good / fast as the recent progress of neural network based detector such as YOLO, and they needed to get more data from the image frame to do their job. But now this is changing and the tracking algorithms get simpler and faster as the detections are better. This technique is also called doing **tracking by detections**.

> “Due to recent progress in object detection, tracking-by-detection has become the leading paradigm in multiple object tracking.” [[출처:SORT deep paper]](https://arxiv.org/pdf/1703.07402.pdf)

본 문서에서는 tracking-by-detection기법을 적용하지 않았다. `That said there are maybe tracking approaches using image data that lead to better tracking results than just "tracking by detections" , but we haven’t looked into them as they will be more complex to run and may likely not run in real time: for example color based tracking, particle tracking ...`

### 3.2 Our problem set


> 주요 내용 없음


### 3.3 Benchmarking existing solutions

[MOT 추적 벤치마킹](https://motchallenge.net/)
: A challenge exists for researcher to compare their tracking algorithm, and it is specifically designed for Multiple object tracking:

많은 알고리즘 중에 하기 두가지 특징을 가지는 알고리즘을 분석 대상으로 삼았다. `There are plenty of algorithm, but we benchmarked two of them that had the following criterias:`
- run at more the 25 FPS
- open source implementation done in Python / C++


#### A. SORT : Simple Online and Realtime Tracking

https://adioshun.gitbooks.io/object-tracking/content/2dtracking/2016-sort-simpleonlinerealtimetracking.html



#### B. IOU Tracker:



https://adioshun.gitbooks.io/object-tracking/content/2dtracking/2017-iou-tracker.md




## 4. Finalizing implementation

유클리드 거리 방법 대신 IOU를 사용하여 성능 향상 가능

Based on the previous learning, we simply integrated to the tracker the `distance()` function of the **IOU** paper instead of reasoning on euclidean distances, this led to lots of improvements and making the tracker much more reliable.



## 5. Limitation and ideas for improvement

- 트럭에 특화됨 it was mostly tested on tracking cars, and could be over optimized for this use case and perform badly on other use cases.

- 카메라가 고정된 환경 Also it was only tested on fixed camera viewpoint.

- 개선 방안 To improve it further,
- 칼만필터 기반 예측 it could be a good idea to work on the prediction framework, by integrating Kalman filters,
- 탐지 신뢰도 점수 활용 and also integrate the confidence on the detection given by YOLO which isn’t used at the moment.


## 6. Conclusion

We didn’t take the time to improve the tracker further as it was good enough for our use case, but there is plenty of room to do so.

In this post we simply wanted to share our learning process on tracking things, and we hope it gives you enough context to be able to fork the project and customize it for you use case.






---
### 동작 방식 How does it work ?

기본 동작 : it compares a frame with the next using dimensions like position of the bbox, size of the bbox and compute a velocity vector. 


It does have novelties compared to our approach:

- It uses Kalman filters to compute the velocity factor: 
    - Kalman filter is essentially doing some math to smooth the velocity/direction computation by comparing the predicted state and the real detection given by YOLO. 
    - (and I think it smooth out also the size of the bounding box of the predictions)


- Its uses an assignment cost matrix 
    - that is computed as the intersection-over-union (IOU) distance between each detection and all predicted bounding boxes from the existing targets (which is putting all the dimensions in a normalized matrix). 
    - Then the best match is computed using the Hungarian Algorithm, which is a way to fastly compute lots of matrices …


- It also handles the score of the detections (how confident YOLO is of that detection). 
    - the tracker could choose between two close detections based on that.


### 제약 Limitations

Does not handle re-entering: 
- that means that if the tracker loose track of something, when the tracker gets the object back, it will give it a new id, which is bad for us as for the game it means that the masking is lost…
    - The velocity computation is not based on several frames: We've found out that with our algorithm it was better to compute velocity model based on the average of few frames back


### How does it perform on our use case of tracking cars ?

Out of the box not that great. The main problem being that there is high number of identity switches (as it does not handle re-entering). But it does perform better for some cases where our tracker is losing tracking.

Also, and this is true for all trackers of the MOT benchmark, the are optimized for persons, not cars, we didn't try with persons as we didn't shoot footage of persons yet, but we can hope that it performs way better than our algorithm for this.





