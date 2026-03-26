# LoRA: Low-Rank Adaptation of Large Language Models

## Introduction
<img width="1719" height="962" alt="image" src="https://github.com/user-attachments/assets/7400e416-fbc5-45c1-b49d-35c63fb9f7c5" />

LoRA는 사전학습된 가중치는 고정한 채, 트랜스포머 각 계층에 저랭크 행렬을 삽입하여
다운스트림 태스크를 위한 학습 파라미터 수를 줄이는 방법이다.

기존 full fine-tuning 방식은 모델을 사전학습 가중치 $\Phi_0$ 로 초기화한 뒤,
각 다운스트림 태스크마다 서로 다른 파라미터 집합 $\Delta\Phi$ 를 학습한다.
문제는 $\Delta\Phi$ 의 차원이 $\Phi_0$ 와 동일하기 때문에,
태스크마다 모델 전체 크기에 해당하는 파라미터를 재학습해야 하는 비효율성이 있다.

---

## Method
<img width="1722" height="965" alt="image" src="https://github.com/user-attachments/assets/00a44665-391b-490b-84b4-07526c1d14c6" />


LoRA는 $\Delta\Phi = \Delta\Phi(\Theta)$ 를 훨씬 더 작은 파라미터 집합 $\Theta$ 로 인코딩하여,
$\Theta$ 에 대한 최적화 문제로 전환한다.

각 가중치 행렬의 업데이트를 저랭크 분해로 표현하면 다음과 같다.

$$W_0 + \Delta W = W_0 + BA$$

학습 과정에서 $W_0$ 는 고정되고, $A$ 와 $B$ 만 학습된다.

**추론 시 장점**: $W = W_0 + BA$ 를 미리 병합하여 저장한 뒤 $Wx$ 를 한 번만 계산하므로,
추가적인 추론 지연이 없다.

**태스크 전환 시 장점**: 기존 $BA$ 를 제거하고 새로운 $B'A'$ 만 삽입하면 되므로,
메모리 오버헤드가 거의 없다.
