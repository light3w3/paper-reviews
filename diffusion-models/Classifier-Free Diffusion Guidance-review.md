# Classifier-Free Diffusion Guidance (CFG)
## Introduction
기존의 classifier guidance는 생성 모델에 외부 분류기를 추가해서, 샘플 품질과 다양성 사이의 trade-off를 조절하는 방법이다.
그러나 이 방식은 분류기를 별도로 학습해야 해서 파이프라인을 복잡하게 만들고,
사전학습된 분류기를 그대로 활용하기 어렵다는 한계가 있다.
또한 샘플링 과정에서 분류기 gradient를 사용하기 때문에,
FID나 IS와 같은 분류기 기반 지표의 향상이 실제 품질 개선 때문인지,
아니면 adversarial effect 때문인지 명확하지 않다는 문제도 있다.

본 논문은 분류기를 사용하지 않고 생성 모델만으로 가이던스를 수행하는
**Classifier-Free Guidance(CFG)** 를 제안한다.

**Main Contributions:**
1. 외부 분류기 없이 조건부·비조건부 모델을 **단일 신경망으로 공동 학습**하는 방법을 제안한다.
2. 분류기 기울기를 사용하지 않아 **adversarial effect 문제를 원천적으로 제거**하였다.

---
## Method
### 1. Classifier Guidance
<img width="1721" height="961" alt="image" src="https://github.com/user-attachments/assets/50dc82fa-81fe-404f-9445-79027f84ca4b" />

CFG를 이해하기 위해 먼저 Classifier Guidance를 살펴본다.
Classifier guidance는 조건부 확산 모델의 스코어에 분류기 기울기를 선형 결합하는 방식으로,
truncation과 유사한 효과를 얻는다.

여기서 $w$는 가이던스 강도를 조절하는 하이퍼파라미터로,
$w$를 조정함으로써 품질과 다양성 사이의 trade-off를 조절할 수 있다.
$w > 0$이면 Inception Score는 증가하지만, 그 대신 샘플 다양성은 감소한다.

Classifier Guidance에서는 이 스코어를 바탕으로 분포에서 샘플링을 진행한다.
이 분포는 원래 확산 모델이 정의하는 분포에 분류기의 confidence term을 곱한 형태이다.
이때 원래 확산 모델의 분포를 변형해 다시 식에 대입하면,
조건부 스코어에 가중치 $w$로 가이던스를 적용하는 것과
비조건부 모델에 가중치 $(w+1)$을 가이던스로 적용하는 것이 이론적으로 동일한 효과를 낸다.

### 2. Classifier-Free Guidance
<img width="1720" height="968" alt="image" src="https://github.com/user-attachments/assets/8e0e0c89-79e8-4d76-afe5-025a8e915ddf" />
<img width="1722" height="965" alt="image" src="https://github.com/user-attachments/assets/9ba07b9f-2e9a-463b-a9b3-aef0523aa8e0" />
<img width="1722" height="968" alt="image" src="https://github.com/user-attachments/assets/788ece72-25cb-404c-88d5-ae38ced7c670" />

CFG는 위에서 도출한 식을 바탕으로 분류기에 해당하는 항을 제거하는 방향으로 스코어 식을 유도한다.
유도 방식은 베이즈 룰을 적용하고 로그 미분하여 스코어 형태로 정리하는 방식이다.

**학습 방법:**
- 조건부 모델과 비조건부 모델을 함께 학습한다.
- 일정 확률 $p_{\text{uncond}}$로 비조건부 모델의 조건 $c$를 null로 치환하는 방식을 사용해,
  단일 신경망으로 조건부와 비조건부 모델을 동시에 학습한다.

**샘플링 방법:**
- 공동 학습된 두 모델의 스코어 추정치를 선형 결합한다.
- 이 결합 가중치인 $w$를 조절함으로써 가이던스를 수행한다.
- 식에 분류기 기울기에 해당하는 항이 없기 때문에,
  이미지 분류기에 대한 기울기 기반 적대적 공격으로 해석할 수 없다.
