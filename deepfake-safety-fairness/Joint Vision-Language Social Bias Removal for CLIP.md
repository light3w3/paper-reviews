# Joint Vision-Language Social Bias Removal for CLIP

## Introduction
<img width="1541" height="864" alt="image" src="https://github.com/user-attachments/assets/e0e6a044-5725-4e14-b381-48a8a709840c" />


이 논문은 CLIP에서 디바이어싱하는 방식을 다룬 논문이다.

기존 CLIP 디바이어싱 방법들은 bias를 줄이는 과정에서 **V-L 정렬 능력이 크게 저하되는 over-debiasing 문제**를 겪는다.

이에 대해 논문은 먼저 CLIP에 존재하는 사회적 편향을 분석했는데, 그 결과 이미지와 텍스트 두 모달리티 모두에서 편향이 존재하고 그 **분포가 서로 다름**을 확인하였다.

그런데 대부분의 기존 방법들은 **한 모달리티만 디바이어싱**하거나, 두 데이터를 함께 다루더라도 분포 차이를 고려하지 않고 디바이어싱한다. 따라서 한쪽을 기준으로 편향을 제거할 경우 다른 쪽에서도 동일한 차원이 제거되었는지에 대한 신뢰성이 떨어진다.

따라서 이 논문은 이미지와 텍스트 간 편향을 **먼저 정렬한 뒤**, 이를 **함께 제거**하는 방식을 제안하여 디바이어싱 과정에서도 **V-L 정렬을 유지할 수 있는 프레임워크**를 제안한다.

**Main Contributions:**
1. CLIP의 이미지·텍스트 두 모달리티 모두에서 편향이 존재하며, 그 **분포가 서로 다름**을 분석하였다.
2. 두 모달리티 간 편향을 **먼저 정렬(Dual-Bias Alignment)한 뒤 함께 제거(Counterfactual Debiasing)** 하는 2단계 프레임워크를 제안한다.
3. 학습된 BA 모듈을 기존 CLIP에 **plug-and-play 방식으로 적용**하여, 추가적인 학습 없이 디바이어싱된 임베딩을 바로 얻을 수 있다.

---

## Method

제안하는 방법은 크게 **Bias Alignment**와 **Counterfactual Debiasing**의 두 단계로 구성된다.

---

### 1. Step 1. Dual-Bias Alignment — 편향 분포 정렬
<img width="1539" height="862" alt="image" src="https://github.com/user-attachments/assets/af1621db-c532-4e99-947b-f598381d0888" />


첫 번째 단계는 **두 모달리티 간의 편향을 정렬**하는 단계이다.

이 논문은 CLIP의 임베딩이 **bias 정보와 bias-neutral 정보의 합으로 분해**될 수 있다는 기존 연구를 전제로 한다.

모델은 먼저 하나의 triplet을 입력으로 받는다.

- **t**: biased text
- **t′**: bias attribute만 변경된 counterfactual text
- **v**: biased image

이 triplet 구성의 이유는 다음과 같다:

| 입력 조합 | 확인 가능 | 확인 불가 |
|-----------|-----------|-----------|
| t, t′ | bias 변화 확인 | alignment 확인 불가 |
| t, v | alignment 확인 | bias 제거 여부 불가 |

→ 따라서 두 조건(**bias 제거 + V-L 정렬 유지**)을 동시에 만족하기 위해 **triplet**을 사용한다.

이 triplet을 CLIP encoder에 넣어 각각의 임베딩을 얻고, 편향을 정렬하기 위한 **BA 모듈**에 넣어 각 임베딩에서 bias 성분을 추출한다.

이때 추출된 bias를 MSE나 코사인 유사도로 직접 정렬하면 **배경 정보 손실이나 특징 다양성 감소**가 일어날 수 있다. 그 이유는 다음과 같다:

- **강제 동일화(Over-constraining)**: MSE / Cosine은 두 벡터를 거의 동일하게 만들려 하지만, 이미지와 텍스트의 bias는 표현 방식이 달라도 의미만 같으면 되므로 불필요한 정보까지 함께 눌린다.
- **bias + background의 혼재**: 실제 임베딩에서 bias 성분과 semantic 성분은 완전히 분리되어 있지 않아, 거리 기반 정렬 시 background 정보까지 함께 변형된다.
- **다양성 감소(Representation Collapse)**: Cosine/MSE로 계속 맞추면 서로 다른 샘플들도 비슷한 방향으로 몰려 feature diversity가 감소한다.

따라서 정렬 시 **무빙 큐(Moving Queue)** 메커니즘을 사용한다. 최근 M개의 텍스트와 이미지 임베딩을 큐에 저장해두고, 현재의 bias embedding이 이 큐 전체와 얼마나 유사한지를 기반으로 편향 **확률 분포**를 구성한다. BA 모듈을 통해 얻은 두 편향 분포 사이의 **KL divergence를 최소화**함으로써 두 모달리티의 편향 분포를 정렬한다.

---

### 2. Step 2. Counterfactual Debiasing
<img width="1542" height="866" alt="image" src="https://github.com/user-attachments/assets/7410d7d8-0828-4735-823a-abcfb13d23ef" />

bias가 정렬된 이후 두 번째 단계로 debiasing을 수행한다.

디바이어싱이 제대로 이루어졌다면, bias 속성만 다른 t와 t′의 neutral 표현은 같아져야 하므로 두 임베딩이 가까워져야 한다. 그러나 임베딩 사이의 거리만 좁혀버리면 V-L 정렬이 깨질 수 있으므로, loss는 **편향 제거와 V-L 정렬 유지를 동시에 고려**하여 설계한다.

**텍스트 디바이어싱:** 원본 텍스트 임베딩을 기준으로 이미지 큐와 대조한 유사도를 교사 신호로 사용하고, t와 t′를 **Bernoulli sampling**으로 stochastic하게 선택하여 이 둘이 같아질 수 있도록 디바이어싱을 수행한다.

**이미지 디바이어싱:** 이미지는 텍스트와 달리 counterfactual 버전을 만드는 것이 불가능하다. 따라서 텍스트 큐와의 대조를 통해 디바이어싱된 이미지 임베딩이 원본과 동일한 **V-L 정렬을 유지하는지를 기준**으로 학습한다.

<img width="1541" height="866" alt="image" src="https://github.com/user-attachments/assets/a02c99bc-1b06-4b7d-8fe2-9dacaf98a35b" />

전체 학습은 **Bias Alignment loss와 Debiasing loss를 함께 최적화**하는 방식으로 이루어진다:

$$L = \alpha \cdot L_{cd} + (1 - \alpha) \cdot L_{ba}$$

추론 단계에서는 학습된 BA 모듈을 기존 CLIP에 **plug-and-play 방식으로 적용**하여, 추가적인 학습 없이도 디바이어싱된 임베딩을 바로 얻을 수 있다:

$$\bar{\psi}(v) = g(v) - BA(g(v))$$

---

## Experiments

### 데이터셋 및 평가 지표
<img width="1328" height="746" alt="image" src="https://github.com/user-attachments/assets/bc349eba-e2ce-4211-a5cd-7937d44d4840" />


학습 데이터셋으로는 **FairFace**와 **UTKFace**를 사용하였으며, 테스트 시에는 전신 이미지 기반의 **FACET** 데이터셋을 추가로 활용하였다. 평가는 학습 데이터셋과 동일한 in-domain 셋과, 나머지 데이터셋으로 구성된 out-of-domain 셋으로 나뉜다.
<img width="1328" height="747" alt="image" src="https://github.com/user-attachments/assets/1f94c86f-995b-422e-b912-7f7ab9c543ef" />

**공정성 지표:**
- **mean MaxSkew@k**: 검색 결과 상위 k개에서 특정 그룹이 불균형하게 나타나는 정도를 측정한다. 여러 그룹(성별, 인종 등) 중 가장 불공평한 경우를 대표값으로 삼고, 여러 쿼리에 대해 평균을 낸다. 값이 작을수록 공정하다.
- **mean NDKL@k**: MaxSkew와 유사하나 최댓값이 아닌 전체 평균적인 불공평함을 측정하는 보완 지표이다.

**V-L 정렬 지표:**
- **ImageNet Top-1 / Top-5 정확도**: zero-shot 분류 성능으로, Top-1은 1순위 예측이 맞는 비율, Top-5는 상위 5개 안에 정답이 포함되는 비율이다.
- **Flickr-1K Recall@5**: 텍스트→이미지(TR) 및 이미지→텍스트(IR) 검색에서 상위 5개 안에 정답이 포함되는 비율이다.

**통합 지표:**
- **ABLE**: 공정성과 V-L 정렬 성능을 통합 평가하기 위해 논문에서 새롭게 제안한 지표로, F1-score 메커니즘을 차용하여 ImageNet Top-1 accuracy와 exp(−MaxSkew@k)를 결합한 형태이다. 값이 클수록 우수하다.

---

### Baselines

- **CLIP-clip**: 이미지 임베딩의 각 차원과 성별·인종 같은 속성 라벨 사이의 상호정보량(MI)을 계산하고, MI가 높은 차원을 이미지·텍스트 임베딩 **양쪽에서** 제거한다. 두 모달리티를 동시에 다루지만 모달 간 편향 분포 정렬은 고려하지 않으며, MI 계산을 위해 속성 라벨이 붙은 학습 데이터가 필요하다.
- **Biased-prompts**: "a photo of a woman", "a photo of a man" 같은 편향된 프롬프트들의 임베딩 차이로 편향 방향을 정의하고, 이를 모아 projection matrix를 구성한 뒤 텍스트 쿼리에서 해당 방향 성분을 제거한다. **텍스트 모달리티에만** 적용되며, 별도의 학습 데이터 없이 프롬프트 몇 개만으로 동작한다.

---

### Results
<img width="1330" height="751" alt="image" src="https://github.com/user-attachments/assets/2ba6e2d9-efcf-494e-9725-8a255bc1e716" />
<img width="1327" height="747" alt="image" src="https://github.com/user-attachments/assets/968cdab1-1154-4dd6-b820-1ab379aec998" />

**Table 1 / Table 2 (FairFace / UTKFace 학습 결과):** 기존 방법들은 편향 감소에는 효과적이지만 V-L 정렬 성능이 크게 저하되어 ABLE 점수가 낮게 나타났다. 반면 제안 방법은 V-L 정렬 성능 저하를 1% 미만으로 유지하면서 편향을 효과적으로 제거하여 가장 높은 ABLE 점수를 기록하였으며, in-domain과 out-of-domain 모두에서 우수한 일반화 성능을 보였다. UTKFace로 학습한 Table 2에서도 동일한 경향이 확인된다.
<img width="1328" height="749" alt="image" src="https://github.com/user-attachments/assets/31e5186a-345e-4dc2-9fcc-4156444cb2e3" />

**t-SNE 시각화:** BA 모듈 적용 후 편향 임베딩이 군집화되는 것을 확인할 수 있으며, 이는 편향 성분이 명확하게 분리됨을 시사한다.
<img width="1329" height="748" alt="image" src="https://github.com/user-attachments/assets/160831cb-8621-423b-8083-9ea5b48e8ef5" />

**Ablation Study:** counterfactual debiasing loss를 제거하면 디바이어싱 성능이 크게 저하되고, bias alignment를 제거하면 디바이어싱과 정렬 성능이 모두 감소한다. 두 요소를 함께 사용할 때 가장 좋은 균형을 달성한다.
<img width="1330" height="747" alt="image" src="https://github.com/user-attachments/assets/065d67fb-a547-48ad-8eae-59603ff92a80" />

**다중 편향 제거 실험:** 기존 방법들(CLIP-clip, Biased-prompts)은 성별·연령·인종의 세 가지 편향을 동시에 제거하는 설정을 검증하지 않았다. 본 논문에서 해당 실험을 추가로 진행한 결과, 제안 방법은 다중 편향을 동시에 효과적으로 제거하면서도 V-L 정렬 능력을 유지하는 것을 확인하였다.
