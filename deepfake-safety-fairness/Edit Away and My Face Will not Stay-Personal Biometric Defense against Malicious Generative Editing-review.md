# Edit Away and My Face Will not Stay: Personal Biometric Defense against Malicious Generative Editing
## Introduction
<img width="2284" height="1280" alt="image" src="https://github.com/user-attachments/assets/2e7f7e82-ac96-40cf-a230-464bee38ccbe" />


FaceLock 논문은 Diffusion 기반 이미지 편집 기술에서 deepfake defense를 다룬 논문이다. 기존 방어법(PhotoGuard, EditShield 등)은 **편집 효과 자체를 상쇄(cancel off)** 하는 adversarial perturbation을 생성하는 방식이었지만, 편집 프롬프트가 너무 다양하기 때문에 하나의 노이즈로 모든 프롬프트를 막는 것이 본질적으로 불가능한 문제가 있다.

이 논문은 **편집을 막는 것이 아니라, 편집 후 결과물에서 원본 인물의 얼굴(생체 정보)을 파괴하는 방향**으로 접근하는 **FaceLock**을 제안한다. 따라서 프롬프트와 상관없이 방어가 안정적으로 동작한다.

논문의 4가지 주요 기여는 다음과 같다.

- (i) **새로운 방향 제시**: 편집을 막는 게 아니라 편집 후 생체 정보를 파괴하는 방향
- (ii) **FaceLock 알고리즘**: 얼굴 인식 모델(CVLFACE) + feature embedding 손실을 결합한 새로운 방법
- (iii) **기존 평가 지표의 허점 폭로**: CLIP score, SSIM, PSNR이 얼마나 쉽게 속는지 분석
- (iv) **실험으로 우수성 입증**: 다양한 편집 프롬프트 및 purification 공격에 강인함

---

## Method
<img width="2285" height="1281" alt="image" src="https://github.com/user-attachments/assets/72d9a08c-1d92-43bd-862d-89ba93f83941" />


논문에서는 성공적인 이미지 편집의 조건을 두 가지로 정의하고, 둘 중 하나라도 만족하지 못하면 편집이 실패한 것으로 정의한다.

| 조건 | 설명 |
|---|---|
| ❶ Prompt Fidelity (프롬프트 충실도) | 편집 지시를 정확히 반영했는가 |
| ❷ Image Integrity (이미지 무결성) | 편집과 무관한 요소(얼굴, 포즈 등)는 그대로인가 |

기존 방어 방법들은 편집 자체를 제대로 수행하지 못하게 하는 방식이므로 **Prompt Fidelity를 훼손하는 방식**이고, **FaceLock은 편집 후 결과물에서 원본 인물의 생체 정보를 파괴하기 때문에 Image Integrity를 망가뜨리는 데 초점을 둔 방식**이다.

이를 위해 FaceLock은 두 가지 손실을 결합한 objective function을 최대화하는 perturbation $\delta$를 구한다.

- **FR Loss(Face Recognition Loss)**: CVLFace라는 얼굴 인식 모델을 활용하여, perturbation이 적용된 이미지를 편집했을 때 원본 얼굴과의 얼굴 유사도가 낮아지도록 유도한다.
- **FE Loss(Feature Embedding Loss)**: AlexNet이나 VGG 같은 CNN의 고수준 feature를 활용하여 편집 결과가 원본과 지각적으로 크게 달라지도록 만든다.

두 손실을 결합해서 육안으로는 거의 구분되지 않는 미세한 노이즈만으로 얼굴 정체성을 파괴한다.

---

### 설계 과정: 4단계 발전

FaceLock은 한 번에 나온 것이 아니라 실패한 시도들을 거쳐 발전했다.

**Design I — CVL** (실패): CVLFACE를 속이는 노이즈를 만들어 편집 모델에 입력한다. Diffusion 모델의 샘플링 과정이 노이즈를 자연스럽게 **정화(purify)** 해버려 FR 점수 0.901로 거의 효과가 없다.

**Design II — CVL-D** (부분 성공): Diffusion의 인코더/디코더를 통과한 결과물에 대해 직접 얼굴 유사도를 최소화한다. $\delta = \arg\max\ f_{FR}(D(E(x+\delta)), x)$. FR 점수 0.273으로 크게 낮아졌지만, **시각적으로는 여전히 원본처럼 보인다**.

**Design III — CVL-DP** (실패): 얼굴 영역 픽셀 차이를 강제하는 손실을 추가한다. $\delta = \arg\max\ f_{FR}(D(E(x+\delta)), x) + \lambda\|\delta \odot m\|_2$. **색상 변화만** 생기고 얼굴 구조 자체는 유지된다. FR 점수 오히려 0.658로 악화된다.

**FaceLock** (성공): 픽셀 수준이 아닌 **feature 수준**에서 차이를 만드는 것으로 전환한다. 사전학습된 CNN(AlexNet, VGG 등)으로 추출한 **고수준 feature embedding 거리**를 최대화한다.

$$\delta = \arg\max\ f_{FR}(D(E(x+\delta)), x) + \lambda \cdot f_{FE}(D(E(x+\delta)), x)$$


| 방법 | FR 점수 | 결과 |
|---|---|---|
| No Protection | 0.972 | 방어 없음 |
| Design I (CVL) | 0.901 | 실패 |
| Design II (CVL-D) | 0.273 | 부분 성공 |
| Design III (CVL-DP) | 0.658 | 실패 (악화) |
| **FaceLock** | **0.093** | **성공** |

---

### 기존 평가 지표의 함정
<img width="2283" height="1276" alt="image" src="https://github.com/user-attachments/assets/5accaafd-b226-4242-a3b5-71b8c13d7756" />


논문에서는 기존에 널리 사용되는 평가 지표들이 실제 편집 품질을 정확히 반영하지 못한다는 점을 지적한다.

**CLIP Score**는 프롬프트와 관련된 요소가 많이 반영될수록 높은 점수를 부여한다. 이로 인해 **over-editing을 선호하는 문제**가 발생하며, 프롬프트에 대해 잘 편집된 경우보다 얼굴이 완전히 다른 사람이 되었더라도 "pink hair"와 같은 속성이 강하게 반영되면 더 높은 CLIP Score를 받게 된다.

**SSIM과 PSNR**은 방어 없이 편집된 이미지를 기준으로 픽셀 수준의 차이를 측정한다. 실제로 방어를 성공한 경우보다 색상 정도만 크게 달라진 경우에 더 낮은 SSIM 점수, 즉 더 큰 방어 성공으로 잘못 평가하게 된다.

이러한 문제를 지적하면서 논문에서는 픽셀 수준이 아닌 **semantic feature 기반으로 유사도를 평가**하는 새 지표를 제시한다.

- **LPIPS**: 사전학습 신경망의 feature 차이로 유사도를 측정한다. 픽셀 색상에 속지 않으므로 SSIM/PSNR의 대안 지표로 제안된다.
- **FR Score**: CVLFACE로 편집 전후 얼굴 유사도를 직접 측정한다. 낮을수록 방어 성공을 의미하며, Image Integrity 측정을 위한 지표다.
> CLIP Score의 문제는 "over-editing을 선호한다"는 것인데, FR Score로 Image Integrity를 별도로 측정하게 되면 설령 CLIP Score가 높게 나오더라도 FR Score로 정체성이 파괴됐음을 잡아낼 수 있다. 결과적으로 CLIP Score만 단독으로 쓸 때의 맹점이 FR Score 도입으로 보완되는 구조다.
