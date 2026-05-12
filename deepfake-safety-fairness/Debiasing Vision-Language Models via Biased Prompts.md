# Debiasing Vision-Language Models via Biased Prompts

## 1. Introduction

### 문제 정의 및 연구 목표

본 논문은 텍스트 임베딩 공간에서 편향 방향(Bias Direction)을 제거함으로써 모델의 공정성을 높이는 디바이어싱 방식을 제안한다. 특히 분류 모델(Classification)과 생성 모델(Generative)의 서로 다른 특성을 고려하여, 각 모델의 목적에 맞는 최적화 목표를 설정하고 이를 달성하기 위한 방법론을 제시한다.

---

## 2. Optimization Goals for Fairness
<img width="2037" height="1143" alt="image" src="https://github.com/user-attachments/assets/25b2a2ec-3d53-44d6-a1b2-5791f5ab9f2d" />

### 2.1. 분류 모델 (Classification Models)

분류 모델에서는 기존 연구된 **집단 강인성(Group Robustness)** 프레임워크를 차용한다.

* **집단(Group)**: 목표 레이블(Target Label)과 허위 속성(Spurious Attribute)의 조합으로 정의된다.
* **목표**: 분류기가 최악 집단 오류(Worst-group Error)와 평균 오류 사이의 격차를 최소화하도록 최적화한다.

### 2.2. 생성 모델 (Generative Models)

생성 모델은 하나의 프롬프트에 대해 다양한 결과가 존재할 수 있어 목표 클래스가 명확하지 않다. 따라서 **통계적 동등성(Statistical Parity)** 개념을 사용한다.

* **측정**: 허위 속성을 예측하는 분류기를 사용하여 생성 분포의 불일치를 측정한다.
* **목표**: 생성된 이미지 중 각 허위 속성의 비율과 이상적인 균일 분포 간의 **$L_2$ norm**을 정의하고 이를 최소화한다.

---

## 3. Debiasing Classification Models

### 3.1. Projection-based Mitigation
<img width="2036" height="1141" alt="image" src="https://github.com/user-attachments/assets/aefc5320-4090-4528-96c7-47c7b8baff0d" />

먼저 "a photo of a male", "a photo of a female"과 같은 프롬프트를 통해 허위 속성을 정의하고, 이를 행렬 $A$로 구성한다. 이후 직교 사영 행렬(Orthogonal Projection Matrix)을 통해 텍스트 임베딩에서 해당 허위 방향 성분을 제거한다.

### 3.2. Calibration
<img width="2039" height="1145" alt="image" src="https://github.com/user-attachments/assets/666999e6-5eb5-4ff6-a31d-00b337858c14" />

프롬프트만으로 정의된 편향 방향의 노이즈를 해결하기 위해 다음과 같은 두 가지 항으로 구성된 보정 손실(Calibration Loss)을 추가한다.

1. **유지 항**: 보정된 행렬 $P$가 기존 사영 행렬 $P_0$에서 너무 멀어지지 않도록 제어한다.
2. **정렬 항**: 허위 속성만 다르고 의미는 동일한 양성 쌍(Positive Pair)에 대해, 사영 이후 임베딩 간 거리가 최소가 되도록 유도한다.

> **SVD(특이값 분해) 관점의 해석**
> 이 최적화는 Closed-form으로 풀리며, 허위 속성 변화에 민감한(특잇값이 큰) 방향 성분들이 역행렬 연산에 의해 억제된다. 결과적으로 원래 의미를 보존하면서도 편향만 효과적으로 제거할 수 있다.

---

## 4. Debiasing Generative Models
<img width="2036" height="1144" alt="image" src="https://github.com/user-attachments/assets/6dcc658f-f5df-437e-8f1c-c35958c003a2" />

### 4.1. Equalization Loss
균등화 손실은 임베딩 $z$ 자체를 직접 최적화하는 수식으로, 생성 모델의 디바이어싱에 활용된다.

* 원래 임베딩과 최적화된 임베딩 사이의 거리 최소화.
* 편향이 제거된 임베딩 $z$가 양성 쌍($z_i, z_j$) 각각과의 내적 차이를 최소화하도록 유도.
* 이 손실의 최적해는 앞서 구한 분류 모델의 보정 행렬과 동일한 구조를 가진다.

### 4.2.Calibration Matrix
생성 모델은 정답 정의가 다르므로 특정 클래스에 매몰되지 않는 **범용적인 보정 행렬**을 사용한다.

* 여러 직업군에 대해 허위 속성만 다른 양성 프롬프트 쌍을 사용해 범용 행렬을 구성한다.
* 특정 클래스에 의존하지 않아 일반화 성능이 뛰어나다.
* **이미지 품질 보존**: 성별 정보를 완전히 삭제하는 대신 남녀 비율이 균형을 이루도록 유도하기 위해, 기본 사영(Projection) 대신 보정 행렬만 적용한다.

---

## 5. Conclusion

본 논문은 분류와 생성이라는 서로 다른 태스크에 맞춤화된 공정성 목표를 설정하고, 텍스트 임베딩 공간 내에서의 사영 및 보정 메커니즘을 통해 효과적으로 편향을 제거하는 방법론을 확립하였다.
