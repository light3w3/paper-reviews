<img width="2154" height="1209" alt="image" src="https://github.com/user-attachments/assets/4972e219-343a-45e2-a847-6f6f41be37a7" /># AIM-Fair: Improving Fairness via Selective Fine-Tuning with Synthetic Data

## 1. Introduction

### 문제 상황
<img width="2154" height="1210" alt="image" src="https://github.com/user-attachments/assets/573f3c7f-66cc-4947-af08-547aa2f460b1" />


모델은 학습 데이터가 특정 성별이나 인종에 치우쳐 있을 경우, 소수 집단에 대해 낮은 성능을 보이는 문제가 있다.

이러한 공정성 문제를 개선하기 위해 최근에는 **생성 모델을 활용한 합성 데이터로 파인튜닝하는 방식**이 많이 사용되고 있다.

아래 그림에서 Y축은 전체 정확도, X축은 공정성을 나타낸다.
**합성 데이터만 사용한 모델(주황색)** 은 공정성은 개선되지만 정확도가 크게 떨어진다.
<img width="2154" height="1209" alt="image" src="https://github.com/user-attachments/assets/04ed0b4b-6691-4970-a46f-f5c481a100d1" />

그 이유는 다음 두 가지다.

1. 합성 데이터의 품질과 다양성이 낮다.
2. 실제 데이터와 합성 데이터 간의 도메인 차이(domain gap)와 편향 차이(bias gap)가 동시에 존재한다.

또한, 합성 데이터 생성은 주로 실제 이미지에 대한 편집을 통해 진행된다.
예를 들어 남성 데이터를 여성 데이터로 수정하려면 각 데이터가 어떤 그룹에 속하는지에 대한 레이블이 필요하기 때문에, 높은 주석 비용 문제가 존재한다.

따라서 본 논문은 아래의 세 가지 방식을 통해 **정확도와 공정성을 동시에 개선**하는 것을 목표로 한다.

---

## 2. Related Work

### 2.1. 모델 편향 완화 방법
<img width="2157" height="1210" alt="image" src="https://github.com/user-attachments/assets/2c331ae0-56df-4975-bd8e-9ee850444e5f" />


공정성 향상 방법은 크게 **전처리, 학습 중 처리, 후처리** 세 가지로 나뉜다.

- **전처리**: 데이터 분포 자체를 수정하는 방식
- **학습 중 처리**: 최적화 과정에서 모델이 성능과 공정성을 함께 고려하도록 하는 방식
- **후처리**: 모델 출력에 보정이나 임계값 조정을 적용하여 그룹 간 성능 격차를 줄이는 방식

본 논문에서 제안하는 방식은 비편향 합성 데이터를 활용한다는 점에서 **전처리 패러다임**에 속한다.

---

### 2.2. 모델 파인튜닝
<img width="2150" height="1206" alt="image" src="https://github.com/user-attachments/assets/e8aba66a-68f2-4e64-97f0-d1fb8512c90f" />


합성 데이터를 사용한 파인튜닝 시, 분포가 다른 데이터에 적응하기 위해 일반적으로는 사전 학습된 모델의 마지막 몇 개 레이어만 업데이트하는 방식이 사용된다.

또한 과적합을 방지하기 위해 초기 레이어를 고정한 뒤 점진적으로 해제하거나, 레이어별로 다른 학습률을 적용하는 방법들도 존재한다.

이러한 흐름에서 일부 레이어만 선택적으로 업데이트하는 **Surgical Fine-tuning**이 제안되었다.
최근 연구에서는 레이어마다 전체 성능에 기여하는 정도가 다르다는 점이 밝혀졌으며,
본 논문은 이를 더 세밀하게 적용하여 **레이어 단위가 아닌 파라미터 단위에서 선택적으로 업데이트하는 방식**을 제안한다.

---

## 3. Method
<img width="2155" height="1210" alt="image" src="https://github.com/user-attachments/assets/3f1159fa-f6c9-48e4-a083-c54d804c5852" />


AIM-Fair는 다음 세 단계로 구성된다.

1. 균형 잡힌 합성 데이터 생성
2. 선택적으로 업데이트할 파라미터를 결정하는 마스크 생성
3. 두 결과를 결합하여 파인튜닝 수행

---

### 3.1. Contextual Synthetic Data Generation (CSDG)
<img width="2152" height="1208" alt="image" src="https://github.com/user-attachments/assets/344b1b1c-0a4f-4ac4-96e3-9659ee25c878" />

T2I 모델로 데이터를 생성할 때, `"portrait face photo of smiling male"` 과 같이 속성을 직접 명시하는 단순한 프롬프트를 사용하면 전형적인 이미지만 생성되고, 나이나 헤어스타일처럼 프롬프트에 포함되지 않은 속성에 대해 의도하지 않은 편향이 발생할 수 있다.

이를 해결하기 위해 **GPT-4를 활용한 문맥 기반 프롬프트 생성**을 제안한다.
GPT-4에 태스크, 타겟 속성, 보호 속성뿐만 아니라 얼굴 특징, 헤어스타일, 눈 모양, 머리 각도와 같은 다양한 세부 묘사까지 함께 입력하여 더 풍부하고 다양한 프롬프트를 생성한다.
이렇게 생성된 프롬프트를 Stable Diffusion에 전달하여 이미지를 생성함으로써, plain 프롬프트에 비해 더 다양하고 세밀한 합성 데이터를 얻을 수 있다.

데이터셋별 GPT-4 instruction은 다음과 같다.
<img width="2493" height="1397" alt="image" src="https://github.com/user-attachments/assets/4333c358-d07d-4f17-8ceb-1db7b7017cb8" />

- **CelebA**: CelebA 데이터셋에서 제공하는 속성 정보를 세부 묘사 속성으로 활용한다.
- **UTKFace**: 연령 분포가 넓기 때문에, 연령에 대한 지시문을 별도로 포함한다.

이러한 지시문을 통해 균형 잡힌 데이터셋을 구축한다.

---

### 3.2. Selective Mask Generation
<img width="2490" height="1398" alt="image" src="https://github.com/user-attachments/assets/341f9979-ad7b-4210-81ce-7f48b0916eee" />


앞선 CSDG로 생성한 균형 잡힌 합성 데이터를 사용해 모델 전체를 파인튜닝하면 오히려 전체 성능이 떨어진다.
그 이유는 실제 데이터와 합성 데이터 간에 두 가지 차이가 동시에 존재하기 때문이다.

- **도메인 차이(domain gap)**: 합성 이미지와 실제 이미지 간의 시각적 품질 차이
- **편향 차이(bias gap)**: 합성 데이터는 균형 있게 구성되었지만, 사전 학습 시 사용한 실제 데이터는 불균형하기 때문에 발생하는 차이

비편향 합성 데이터로 파인튜닝을 하게 되면, 공정성 개선에 필요한 변화뿐 아니라 도메인 차이에 반응하는 불필요한 변화까지 함께 학습되어 전체 성능이 떨어지게 된다.

또한 특정 레이어를 고정한 채로 학습을 진행했을 때 성능 차이가 발생한다는 점에서, 모든 레이어가 동일하게 중요하지 않고 레이어별로 민감도가 다르다는 것을 확인할 수 있다.
<img width="2494" height="1401" alt="image" src="https://github.com/user-attachments/assets/b2bf7f24-e1a7-4dd9-b793-841fd23d10f3" />

따라서 본 논문에서는 **도메인 변화에는 둔감하면서, 공정성에는 민감한 파라미터**를 찾기 위해 다음과 같이 세 가지 데이터셋을 구성한다.

| 데이터셋 | 도메인 | 편향 분포 |
|---|---|---|
| $D_R$ | 실제 (real) | 편향됨 (biased) |
| $D_{S1}$ | 합성 (synthetic) | 편향됨 (biased) |
| $D_{S2}$ | 합성 (synthetic) | 균형됨 (unbiased) |
<img width="2497" height="1400" alt="image" src="https://github.com/user-attachments/assets/034bf0b8-23cb-4854-80af-24fc4faadd96" />

각 데이터셋에 대해 그래디언트를 계산하고, 두 쌍의 그래디언트 차이를 구한다.

- **$\Delta_1 = \nabla(D_R) - \nabla(D_{S1})$**: $D_R$과 $D_{S1}$은 편향 분포가 같고 도메인만 다르므로, 이 차이는 **도메인에 민감한 파라미터**를 나타낸다.
- **$\Delta_2 = \nabla(D_{S1}) - \nabla(D_{S2})$**: $D_{S1}$과 $D_{S2}$는 같은 합성 도메인이지만 편향 분포가 다르므로, 이 차이는 **공정성에 민감한 파라미터**를 나타낸다.

$\Delta_1$은 오름차순, $\Delta_2$는 내림차순으로 정렬하여 각각 상위 k개를 선택하고, **두 집합의 교집합**을 최종 마스크로 사용한다.
이렇게 선별된 파라미터에 대해서만 마스크를 적용한다.

---

### 3.3. Selective Fine-Tuning
<img width="2497" height="1403" alt="image" src="https://github.com/user-attachments/assets/6225e110-4fe7-4558-a19d-366ce2b6b0c0" />


Selective Fine-Tuning은 앞선 단계에서 얻은 균형 잡힌 합성 데이터 $D_{S2}$와 selective mask $M$을 결합하여 파인튜닝을 수행하는 단계다.

기존 파인튜닝과 동일하게 그래디언트를 계산하되, 마스크 $M$을 원소별(element-wise)로 곱하여 선택된 파라미터만 업데이트한다. 이 마스크는 전체 파인튜닝 과정 동안 고정되어 적용된다.

**정리하면**, AIM-Fair는 CSDG로 다양한 합성 데이터를 생성하고, selective mask로 업데이트할 파라미터를 제한함으로써 도메인 차이는 최소화하면서 공정성은 최대화하는 파인튜닝을 수행하는 방식이다.

---

## 4. Experiment

### 데이터셋 및 평가 지표
<img width="2497" height="1402" alt="image" src="https://github.com/user-attachments/assets/cf79f992-b8f8-4d9c-8706-e69dab7f1d80" />


사용한 데이터셋은 다음과 같다.

| 데이터셋 | 보호 속성 (Protected) | 타겟 속성 (Target) |
|---|---|---|
| CelebA | Male, Young | Smiling, Young |
| UTKFace | Ethnicity | Gender |

평가 지표는 다음 5가지를 사용한다.
<img width="2494" height="1398" alt="image" src="https://github.com/user-attachments/assets/6245237d-773a-41f9-9615-52fdfbbb47d0" />

- **ACC**: 전체 정확도
- **Group ACC**: 그룹별 정확도
- **EO (Equalized Odds)**: 보호 그룹 간 TPR/FPR 차이를 측정하는 공정성 지표. 낮을수록 공정함.
- **WST (Worst-group accuracy)**: 최악 그룹 정확도
- **STD**: 그룹 간 표준편차

> **그룹 정확도 산출 방식**
> 1. 타겟 속성 기준으로 먼저 분리 (t = 0, t = 1)
> 2. 각 타겟 그룹 내에서 보호 속성 기준으로 다시 분리 (P = 0, P = 1)
> 3. 결과적으로 (t=0, P=0), (t=0, P=1), (t=1, P=0), (t=1, P=1)의 4개 그룹 각각에 대해 정확도를 계산

---

### Table 1 & 2: SOTA 방법과의 비교
<img width="2500" height="1405" alt="image" src="https://github.com/user-attachments/assets/5bbfd355-fdc1-46cd-a086-7158032f2f16" />


비교 모델은 기본 모델인 ERM과 디바이어싱 모델 6개로 구성된다.

생성 기반 방법인 DiGA와 AIM-Fair의 차이는 다음과 같다.
- **DiGA**: 기존 실제 이미지를 편집하는 방식으로 합성 데이터를 생성
- **AIM-Fair**: 이미지를 처음부터 새롭게 생성

결과를 보면, ERM은 ACC는 높지만 공정성 관련 지표의 성능이 낮다.
DiGA는 기존 디바이어싱 방법 중 가장 균형 잡힌 성능을 보이며, AIM-Fair는 정확도와 공정성이 가장 균형적이면서 좋은 성능을 달성한다.

오른쪽 테이블은 사전 학습 데이터인 실제 데이터 크기를 각각 50%, 25%, 10%로 줄였을 때의 성능을 비교한 실험이다.
10% 설정에서의 ACC를 제외하고는 AIM-Fair가 가장 좋은 성능을 보이며, 학습 데이터가 부족한 상황에서도 안정적인 성능을 유지한다.

---

### Table 3: Selective Fine-Tuning 효과 검증
<img width="2495" height="1404" alt="image" src="https://github.com/user-attachments/assets/df913d50-1e25-4272-a365-617de722c9f6" />


각 비교 방법의 특성은 다음과 같다.

| 방법 | 특징 | 한계 |
|---|---|---|
| 합성 데이터만 사용 | 공정성 개선 | domain gap으로 정확도 크게 하락 |
| 실제 + 합성 데이터 혼합 | 정확도 유지 | 편향 데이터 비율이 높아 공정성 개선 효과 희석 |
| Data Repairing | 데이터 분포 균형화 | domain shift로 공정성 개선 효과 미미 |
| Linear Probe | 마지막 FC 레이어만 학습 | 기존 feature가 편향되어 있어 개선 효과 제한적 |
| Fully Fine-Tuning | 공정성 크게 개선 | 정확도 하락 |
| **AIM-Fair (제안)** | 선택적 파라미터만 업데이트 | **정확도와 공정성 동시 달성** |

---

### Table 4: 다양한 Partial Fine-Tuning 방식과의 비교
<img width="2492" height="1405" alt="image" src="https://github.com/user-attachments/assets/f115082f-0a5e-4abb-b725-8521812716b9" />


| 방법 | 특징 |
|---|---|
| Random Selection | 40~85% 비율로 무작위 선택. 랜덤성으로 인해 성능이 일관되지 않음. |
| Sub-Tuning (block 단위) | 하나의 블록만 학습 시 공정성 부족, 하나만 고정 시 정확도 하락. 블록 단위로는 균형 달성이 어려움. |
| Cosine Similarity 기반 | gradient의 **방향**을 기준으로 파라미터 선택 |
| **AIM-Fair (제안)** | gradient의 **절댓값 차이**를 기준으로 파라미터 선택. 전반적인 균형에서 가장 좋은 성능. |

> **Cosine Similarity vs. Absolute Difference**
> 방향이 같더라도 크기가 크게 다르면 모델이 데이터 분포 변화에 민감하게 반응하고 있는 상황이다.
> Cosine Similarity는 이런 경우를 잘 구분하지 못하지만, Absolute Difference는 이를 직접 반영할 수 있다.
> 따라서 "어떤 파라미터가 데이터 분포 변화에 민감한가?"라는 질문에는 **변화량(magnitude)** 을 기준으로 하는 Absolute 방식이 더 직관적이고 효과적이다.

---

### Table 5: 다양한 프롬프트 방식 비교
<img width="2501" height="1403" alt="image" src="https://github.com/user-attachments/assets/0220b234-968b-48c2-8a71-813aa3033813" />


Plain Prompt를 기준으로 비교하면, Contextual Prompts를 사용하고 프롬프트 개수를 늘릴수록 성능이 점진적으로 향상된다.
머리 방향과 각도까지 포함한 **Contextual + Head Poses** 방식이 데이터 다양성을 극대화하여 가장 좋은 성능을 보인다.

---

### Ablation: Top-k, Bias Ratio, 데이터 규모
<img width="2499" height="1405" alt="image" src="https://github.com/user-attachments/assets/08ffe783-93a0-4d23-8184-817f1025c6e4" />

**Top-k 분석**
- $k = 55$일 때 정확도와 공정성 간의 가장 좋은 균형을 달성한다.

**Bias Ratio 분석**
- CSDG의 편향 합성 데이터($D_{S1}$)의 편향 비율을 변화시켰을 때, 비율이 극단적일수록 공정성 지표가 개선된다.
- 정확도 측면에서는 minority:majority = 4:6일 때 89.23으로 가장 높다.
- 실제 데이터($D_R$)는 90:10으로 고정되어 있으므로, Bias Ratio가 1:9로 갈수록 $D_R$과 $D_{S1}$의 편향 방향이 반대이면서 차이가 극명해진다.

**합성 데이터 비율 분석**
- 균형 잡힌 합성 데이터($D_{S2}$)와 실제 데이터의 비율이 **1:1일 때 공정성이 가장 높게** 나타난다.

---

## 5. Conclusion
<img width="2496" height="1402" alt="image" src="https://github.com/user-attachments/assets/02cb1646-6b53-4fca-a1ed-8f6ea08f55c8" />

AIM-Fair는 다음 두 가지 핵심 기여를 통해 모델의 공정성과 정확도를 동시에 향상시키는 방법을 제안한다.

1. **LLM 기반 합성 데이터 생성(CSDG)**: GPT-4를 활용한 문맥 기반 프롬프트로 데이터 품질과 다양성을 확보한다.
2. **Selective Fine-Tuning**: 공정성에 민감하면서 도메인 변화에는 둔감한 파라미터만 선택적으로 업데이트한다.
