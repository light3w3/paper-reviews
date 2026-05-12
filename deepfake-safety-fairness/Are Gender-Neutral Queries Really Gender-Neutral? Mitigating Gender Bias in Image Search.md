# Are Gender-Neutral Queries Really Gender-Neutral? Mitigating Gender Bias in Image Search

## 1. Introduction

일반적인 이미지 검색 모델은 텍스트 쿼리와 이미지 간의 코사인 유사도 점수 $S(v, c)$를 계산하고, 유사도 점수가 높은 상위 $K$개의 이미지를 반환한다.

이때 발생하는 문제는 "a person is cooking"처럼 성별이 명시되지 않은 쿼리를 입력했을 때 특정 성별의 이미지가 훨씬 많이 검색되는 현상이다. 본 논문은 이러한 이미지 검색에서의 젠더 편향 문제를 다루며, 이를 정량적으로 측정하는 방식과 디바이어싱(Debiasing)하는 두 가지 방법을 제안한다.

---

## 2. Gender Bias Quantification
<img width="2034" height="1144" alt="image" src="https://github.com/user-attachments/assets/4f53b434-22ea-47f0-b96e-e05145d439b6" />


### 젠더 속성 정의 및 지표 산출

편향을 정량화하기 위해 이미지의 젠더 속성을 다음과 같이 세 가지로 정의한다.

* **Male**: 남성만 등장하는 경우
* **Female**: 여성만 등장하는 경우
* **Neutral**: 그 외의 경우

검색된 상위 $K$개의 이미지 집합에 대해 Male과 Female의 개수를 수집하고, 그 차이를 기반으로 젠더 편향 지표 $\Delta$를 정의한다. (단, 사람이 없는 쿼리로 인해 분모가 0이 되는 경우는 0으로 처리한다.)

**[젠더 편향 지표 $\Delta$의 해석]**

| Condition | Interpretation |
| --- | --- |
| $\Delta = 0$ | Balanced (No Bias) |
| $\Delta > 0$ | Male Bias |
| $\Delta < 0$ | Female Bias |

논문에서는 이 지표를 통해 MS-COCO와 Flickr30K 데이터셋에서 최신 모델들이 약 **70% 수준의 남성 편향**을 보임을 확인하였다.

---

## 3. Mitigation Methods

### 3.1. In-processing: FairSample

<img width="2038" height="1148" alt="image" src="https://github.com/user-attachments/assets/e6cebed8-c279-48ed-bc54-d13b6a04e0c3" />

특화 모델(Specialized Model)은 대조 학습(Contrastive Learning)을 통해 이미지와 텍스트의 유사도를 학습하는데, 이 과정에서 음성 샘플(Negative Samples)이 데이터 분포를 크게 반영한다. 즉, 남성 데이터가 많으면 음성 샘플도 남성 위주로 선택되어 모델이 편향되게 학습된다.

**[FairSample의 작동 방식]**

1. 미니배치 내 이미지를 성별 기준으로 분류한다.
2. **성별 중립 쿼리**: 음성 샘플을 남성과 여성 집단에서 각각 50%씩 균등하게 샘플링한다.
3. **성별 명시 쿼리**: 기존의 샘플링 방식을 유지한다.
4. **성능 균형**: 공정성 개선 시 Recall이 저하되는 문제를 해결하기 위해, 기존 손실 함수와 $\alpha$ 값으로 결합하여 최적의 균형을 맞춘다.

---

### 3.2. Post-processing: Feature Clipping

<img width="2039" height="1148" alt="image" src="https://github.com/user-attachments/assets/9168a766-a95c-4895-869e-354d354db324" />

CLIP과 같은 범용 모델(Foundation Model)은 재학습이 어렵기 때문에, 임베딩 벡터에서 성별 정보를 담고 있는 차원을 직접 제거하는 방식을 사용한다.

**[Feature Clipping 단계]**

1. 이미지 임베딩 벡터의 각 차원에 대해 성별 정보와의 상호 정보량(Mutual Information)을 계산한다.
2. 값이 큰 차원부터 순차적으로 선택하여 성별 정보와 강하게 연관된 특징 차원 집합 $Z$를 구성한다.
3. 이미지 임베딩에서 $Z$에 해당하는 차원을 제거(Clipping)한다.
4. **텍스트 임베딩**: 이미지에서 계산된 집합 $Z$를 동일하게 적용하여 같은 차원을 제거한다.
5. 수정된 두 임베딩의 코사인 유사도를 계산하여 검색을 수행한다.

---

## 5. Conclusion

본 논문은 이미지 검색 시스템에서 발생하는 성별 편향을 $\Delta$ 지표로 정량화하였으며, 학습 단계(FairSample)와 추론 단계(Feature Clipping) 각각에서 적용 가능한 디바이어싱 방법론을 제시하여 공정성과 검색 성능 사이의 균형을 유지하도록 하였다.
