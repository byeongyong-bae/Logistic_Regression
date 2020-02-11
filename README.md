로지스틱 회귀분석   
------------------   
### 개요   
지리정보와 기상정보를 이용하여 침수 여부 예측   
침수 여부라는 0과 1의 클래스를 활용하기 위해 logistic regression기법 활용   
scikit-learn의 패키지를 활용하는 것이 아닌 코딩을 통해 기법 구축   

### logistic function   
로지스틱 회귀는 결과값에 특정한 함수 로지스틱을 적용한다.   
![logistic_function](https://user-images.githubusercontent.com/59756209/74232057-58eb0e00-4d0b-11ea-9941-00d0202cd38c.PNG)   
logistic function을 결과값에 적용하여 [0,1]의 값을 갖게 하여, 해당 데이터가 feature에 대한 y = True가 나올 conditional probability로 표현된다.   
logsitic function은 딥러닝의 sigmoid함수라고도 불린다.   

### likelihood   
로지스틱은 least square방식이 아니라 maximum liklihood로써 적합을 한다.   
그러므로 로지스틱은 회귀분석과 잘리 정규성가정, 등분산성가정이 필요없다.   
하지만, error들이 서로 독립적이어야하고 다중공선성은 없어야한다.   
likleihood는 단순하게 해당 target을 맞출수 있을 가장 그럴듯한 계수이다.   
고정된 계수에서 특정 target에 대해 로지스틱 모형이 추정할 확률은 다음과 같다.   
![likelihood1](https://user-images.githubusercontent.com/59756209/74232904-07dc1980-4d0d-11ea-9ca2-586325b68874.PNG)   
이를 단순히 게수에 대해 표현하면, 계수에 대한 likelihood가 되어 해당 likelihood를 maximize하는 계수를 구할 수 있다.   
또한 언더피팅방지, 계산상의 이유로 log를 사용하여 log_likelihood라고 표현하기도 한다.   
![likelihood2](https://user-images.githubusercontent.com/59756209/74233079-5ee1ee80-4d0d-11ea-9f8e-4ad0611c396f.PNG)   
머신러닝 기법의 기본가정과 같은 데이터가 서로 독립이라고 가정한다면, 위에서 구한 likelihood를 모든 target들에 대해서 cumproduct를 해서 전체 데이터에 대한 likelihood와, 그 전체 데이터의 likeihood를 maximize하는 계수를 구할 수 있을 것이다.   
즉, 모든 데이터 target에 대해 구한 최종적인 목적함수 log likelihood는 다음과 같다.   
![likelihood3](https://user-images.githubusercontent.com/59756209/74233371-dd3e9080-4d0d-11ea-9215-422f8190ea0f.PNG)   
해당 최종식을 gradient descent를 통해 최적화를 할 것이다. 그러기 위해선 각 계수에 대해 gradient를 구해야한다.   
![likelihood4](https://user-images.githubusercontent.com/59756209/74233830-cd737c00-4d0e-11ea-8de7-e6e568b86b9c.PNG)   
즉, 특정 데이터 x에 대해 계수 gradient는 
  

### 참고   
1. http://www.win-vector.com/blog/2011/09/the-simpler-derivation-of-logistic-regression/   
2. http://www.columbia.edu/~so33/SusDev/Lecture_10.pdf   
3. https://onlinecourses.science.psu.edu/stat414/node/191/   
4. http://gnujoow.github.io/ml/2016/01/29/ML3-Logistic-Regression/   
5. https://godongyoung.github.io/



