# Moving Object Tracking of Vehicle Detection

> http://docplayer.net/16497156-Moving-object-tracking-of-vehicle-detection-a-concise-review.html

## 5.1 Region-Based Tracking Methods

- 영역을 기반으로 물체를 추적한다. `In these  methods,  the  region of  the  moving  objects are  tracked  and  used  for  tracking  the vehicles. `

- 영역은 배경제거 기법등을 통해 추출 된다. `These regions are segmented using the subtracting process between the input frame image  and  prior  stored  background  image.`

- This  model  worked  on  series  of  traffic  scenes recorded  by  a  stable  camera  for  automobiles  monocular  images  and  provided  position  and speed  knowledge  for  each  vehicle as  long  as  it  is  visible.  

- The  processing  algorithms  of  this model represented by three levels: 
  - raw images, 
  - region level, and 
  - vehicle level.

## 5.2 Contour Tracking Methods

- 차량의 윤곽선 정보를 이용한다. `These  methods  depend  on  contours  (the  boundaries  of  vehicle)which  are  updated dynamically  in  successive  images  of  vehicle  in Tracking  Vehicle  Process  [36].`

- 영역 기반 방식보다 성능이 좋다. `These methods provide more efficient descriptions of objects than Region-Based Methods and have been  successfully  applied  to  practice.  `

- But  objects  occlusion and  automatic  initialization  of tracking are difficult to handle and tracking precision is limited by a lack of precision in the location of the contour.

## 5.3 3D Model-Based Tracking Methods

A  vehicle  anisotropic  distance  measurement  achieved  through  the  3D geometric  shape  of vehicles.  

A  new  3D  model-based  vehicle  detection  and  depiction  framework  is  based  on  a probabilistic  boundary  feature  grouping,which is used  for  vehicle  detection  and  tracking process [37].

In this paper, the occlusion of vehicles detection process uses a 3D solid cuboid form  with  up  to  six  vertices,  and  this  cuboid is used  to fit  any  different  types  and  sizes  of vehicle  images   by   changing   the   vertices   for   a   best   fit.  

Therefore,   vehicle   detection, segmentation   and   tracking can   be   achieved   efficiently   due   to   changes   in   the   region proportion, prototype width and height with consideration to previous images. 

## 5.4 Feature-Based Tracking Methods 

The  particular  vehicles  are  detected,  segmented  and  tracked  in  image  sequence  by assembling, bunching and approximating the 3D world coordinates of vehicle's feature points. 

An iterative  and  distinguishable  framework  based on  edge  points  as  features  is  used in similarity  process,  these  features  represents  a  large  region  of  set  of  features forms  a  strong depiction  for  object  classes.  

This proposed  framework  showed  a  good  performance for vehicle classification in surveillance videos[38].

A linearity feature technique is a  proposed line-based  shade  method  which  uses  line groups  to remove  all  undesirable  shades  and properly under takes the occlusion resulting from shades. 

## 5.5 Color and Pattern-Based Tracking Methods

This  technique  is  used  to analyze color  of image  series of  traffic  supervision  views [39]. 

Through the  practical  experiments,  this  system proven to work well under  several  weather situations,and it  is  insensitive  to light  variations.

This model-based  system is  used for real-time  traffic  supervision for continuous visual  tracing and classification  of vehicles  for  busy multi-lane highway scene[40].




