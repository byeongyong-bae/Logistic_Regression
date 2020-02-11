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
