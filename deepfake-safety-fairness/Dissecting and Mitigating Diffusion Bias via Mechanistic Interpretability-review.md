# Dissecting and Mitigating Diffusion Bias via Mechanistic Interpretability
## Introduction
<img width="1370" height="768" alt="image" src="https://github.com/user-attachments/assets/26305c09-69bf-4dee-9c1d-9f0d5efcc59d" />


확산 모델의 편향 문제를 위해서 파인튜닝하는 방식이나 프롬프트 수정, 어텐션 가중치를 조작 등 생성 과정 자체를 건드리는 방식은 모델 내부 메커니즘을 충분히 반영하지 못해 비용 문제도 있지만 품질 저하나 과도한 보정 문제가 발생한다.

그래서 이 논문은 모델 내부 메커니즘에 초점을 두고, 편향을 생성하는 패턴 자체를 규명하려는 논문이다. 이를 위해 특정 뉴런들의 집합적 활성화 패턴이 편향된 개념 생성을 유도한다고 보고, 확산 모델의 은닉 상태를 분석해 이 패턴을 **Bias Feature**로 정의한다. 이후 이를 직접 조작해 편향을 제어하는 **DIFFLENS**를 제안했다.

---

## Methodology — DIFFLENS
<img width="1370" height="768" alt="image" src="https://github.com/user-attachments/assets/3004f531-59c1-4f9d-9176-f4b5798989da" />


DIFFLENS는 다의적 뉴런을 분리하는 과정을 먼저 하고, Bias Feature를 식별한 뒤에 이를 가지고 조작을 수행하는 **세 단계**로 구성된다. 

---

### Step 1. 다의적 뉴런 분리
<img width="1374" height="770" alt="image" src="https://github.com/user-attachments/assets/fd5e3001-2dbc-4271-868a-fe2f204019c7" />


Step 1은 하나의 뉴런이 여러 개념에 동시에 반응하는 다의성(Polysemanticity) 문제를 해결하기 위해, **k-Sparse Autoencoder(k-SAE)** 를 사용해 은닉 상태를 더 큰 차원의, 즉 희소한 의미 공간으로 변환한다. 다음으로 상위 k개만 활성화해서 실제로 의미 있는 특징들만 fired된다. 따라서 변환된 공간에서의 각 특징은 단일한 의미(Monosemantic)를 가지기 때문에, 특정 속성에 해당하는 특징을 정밀하게 위치시킬 수 있다. 

- 은닉 상태 $h \in \mathbb{R}^n$ 을 차원이 훨씬 큰 희소 공간 $s \in \mathbb{R}^m$ $(m \gg n)$ 으로 매핑한다.
- 인코딩: $s = \text{TopK}(W_\text{enc}(h - b_\text{pre}))$ — 상위 k개 특징만 활성화(fired), 나머지는 0(unfired)
- 디코딩: $\hat{h} = W_\text{dec} \cdot s + b_\text{pre}$
- 학습 목표: 재구성 오차 최소화 $\mathcal{L}(h) = \| h - \hat{h} \|_2^2$

기존 방법처럼 원래 은닉 공간 전체를 직접 편집하면 다의성 때문에 관련 없는 속성까지 의도치 않게 변경되는 문제가 생기는데, 이를 근본적으로 해결한다. 학습은 원본 $h$와 복원된 $\hat{h}$의 차이, 즉 재구성 오차를 최소화하도록 이루어진다.

> k-SAE의 목적은 "하나의 특징 = 하나의 의미"를 만드는 것이다. 만약 모든 특징을 다 활성화하면 $m$개의 특징이 전부 동시에 켜져 있어 결국 원래 은닉 공간 $h$와 다를 게 없고, 여전히 여러 개념이 뒤섞인 상태가 된다. **희소성이 없으면 분리도 없다.**

---

### Step 2. Bias Feature 식별
<img width="1376" height="774" alt="image" src="https://github.com/user-attachments/assets/f87c724a-34c0-4d84-80ab-e2b247db39e2" />


편향 생성에 기여하는 특징을 찾기 위해 **그래디언트 기반 Attribution**을 사용한다.

먼저 편향 측정 함수 $F$를 정의하는데, 이는 희소 의미 공간의 특징 벡터 $s$가 주어졌을 때 분류기가 특정 클래스(예: "male", "female")로 예측할 확률을 출력한다. 그 다음 각 특징 $s_i$가 이 예측에 얼마나 기여하는지를 계산하고, 이 과정을 반복해 최종적으로 기여도가 가장 높은 상위 $\tau$개의 특징만 선택하여 최종적인 bias feature 집합으로 정의한다. 

- 편향 측정 함수: $F_x(s) = \Pr(y \mid s)$ — $y$는 "male", "female" 등 클래스
- 특징 $s_i$의 기여도: $S(s_i; x) = (s_i - s_i') \cdot \int_{\alpha=0}^{1} \frac{\partial F_x(s' + \alpha(s-s'))}{\partial s_i} d\alpha$
- 데이터셋 전체 집계: $S(s_i; X) = \sum_{j=1}^{N} S(s_i; x_j)$
- 기여도 상위 $\tau$개 특징 선택 → Bias Feature 집합 $A = \{i_1, \ldots, i_\tau\}$ 정의

이 과정은 특정 확산 모델에 대해 **한 번만 수행**하면 된다. 이 집합이 확산 모델 내부에서 편향 생성에 가장 직접적으로 관여하는 의미 특징들이다.

---

### Step 3. Bias Feature 개입
<img width="1377" height="773" alt="image" src="https://github.com/user-attachments/assets/cbce468d-e193-47b7-929a-dda14dd579c7" />


식별된 Bias Feature를 억제하거나 증폭시켜 편향 수준을 제어한다. 방법은 **Scaling**과 **Adding** 두 가지다.

- **Scaling:** $s_i \leftarrow \beta s_i$
- **Adding:** $s_i \leftarrow s_i + \beta$ (모든 $i \in A$, $\beta \in \mathbb{R}$)

$\beta$ 값을 통해 특정 방향의 비율을 조정한다. 따라서 DIFFLENS 방식은 전체 임베딩이 아니라 선별한 feature 집합에만 개입하기 때문에, 목표하지 않은 속성(미소, 안경 등)이 훼손되지 않는 장점이 있다.
