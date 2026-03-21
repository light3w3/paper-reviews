# IDProtector: Protecting Portraits Against Encoder-based ID-Preserving Generation

## Introduction
<img width="1960" height="1099" alt="image" src="https://github.com/user-attachments/assets/82cd6aa3-65ed-41de-9bc3-e782364e658b" />


기존의 초상권 보호 방법(Glaze, AdvDM, AntiDreamBooth 등)은 파인튜닝 기반 공격만을 타겟으로 설계되었다. 그러나 PhotoMaker, IP-Adapter, InstantID 등 인코더 기반 정체성 보존 생성 모델은 사진 단 1장만으로도 특정 인물의 얼굴을 합성할 수 있어 심각한 프라이버시 위협이 된다. 본 논문은 이에 대한 최초의 방어 방법으로 **IDProtector**를 제안한다.

**Main Contributions:**

1. 인코더 기반 생성 모델에 대한 **최초의 범용 적대적 보호 방법**을 제안한다. CLIP 비전 인코더와 ArcFace 얼굴 인코더를 동시에 공격함으로써 다양한 모델에 대해 보편적 방어가 가능하다.
2. ViT 기반 **노이즈 인코더**를 학습시켜, PGD 방식 대비 수 분에서 수 초 이하로 보호 이미지를 생성하는 **효율적인 파이프라인**을 구성한다.
3. 아핀 변환 증강을 통해 JPEG 압축·리사이즈 등 실제 환경의 이미지 변환에도 적대적 효과가 유지되는 **강건성**을 확보한다.

---

## Method
<img width="1964" height="1104" alt="image" src="https://github.com/user-attachments/assets/8f6e4f7b-9c27-4566-91a2-5c7b3a24b67f" />


### 1. Noise Generation via ViT-based Encoder
<img width="1963" height="1107" alt="image" src="https://github.com/user-attachments/assets/91086e8c-8187-4d8d-a1c9-aa2eae724b76" />


보호 대상 이미지 $x$가 입력되면 224×224로 리사이즈된 후 ViT-S/8 기반 노이즈 인코더 $E_\theta$에 입력된다. 이때 RGB 3채널에 InsightFace로 생성한 얼굴 위치 마스크를 4번째 채널로 추가하여, 모델이 학습 초기부터 얼굴 위치를 인식할 수 있도록 한다.

생성된 perturbation은 원본 크기로 리사이즈되어 이미지에 더해지고, 보호된 이미지 $x + \delta$가 인코더 기반 생성 파이프라인으로 전달된다.

---

### 2. Embedding Disruption via Adversarial Loss
<img width="1961" height="1105" alt="image" src="https://github.com/user-attachments/assets/302251b3-1b98-4482-ad03-df7ee0427864" />


현재 인코더 기반 생성 모델은 CLIP 비전 인코더(PhotoMaker, IP-Adapter)와 얼굴 전용 ArcFace 인코더(InstantID, PhotoMaker)가 공존한다. 생성 모델은 특징 임베딩을 통해서만 얼굴 정보를 얻으므로, 임베딩을 교란하는 것이 핵심 공격 포인트가 된다.

| 모델 | 공격 포인트 |
|------|------------|
| InstantID | ArcFace 임베딩 |
| IP-Adapter, PhotoMaker | CLIP 출력 특징 |
| IP-Adapter-Plus | CLIP 마지막에서 두 번째 레이어 이전 특징 |

공격 대상 임베딩 선택은 세 가지 원칙을 따른다. 첫째, 모든 정보 흐름이 최소 하나의 공격 대상 임베딩을 반드시 통과하도록 한다. 둘째, 역전파 경로를 단축하기 위해 가능한 네트워크 초기 레이어의 특징을 선택한다. 셋째, 의미 정보가 밀집된 임베딩을 우선 대상으로 한다.

원본 임베딩 $e_i$와 보호된 이미지의 임베딩 $e_i'$ 사이의 코사인 유사도를 모든 모델에 대해 가중 합산한 것을 적대적 손실 $L_{adv}$로 정의한다. 여기에 perturbation의 크기를 제한하기 위해 $\ell_1$ 정규화와 $\epsilon$-ball(9/255) 초과 페널티를 포함한 $L_{reg}$를 합산하여 최종 손실을 구성한다.

$$L = L_{adv} + L_{reg}$$

학습은 총 4종의 모델에 대해 동시에 수행되며, curriculum learning을 통해 3단계에 걸쳐 $\alpha_1 \sim \alpha_4$의 가중치를 조정하면서 균형을 맞춘다.

---

### 3. Robustness via Affine Augmentation
<img width="1970" height="1108" alt="image" src="https://github.com/user-attachments/assets/64eae274-ae5a-47f6-b10e-09ff1ea8a1f0" />


보호된 이미지는 소셜미디어 업로드 과정에서 JPEG 압축이나 리사이즈가 발생하며, InstantID의 경우 모델 내부에서도 얼굴 정렬을 위한 아핀 변환이 수행되어 perturbation이 손상될 수 있다.

이를 해결하기 위해 학습 시 아핀 변환 행렬에 가우시안 노이즈를 추가한다.

$$A' = A + \mathcal{N}(0, 0.03I)$$

이 증강은 InstantID branch에만 적용되지만, 이미지 레벨의 기하학적 변환에 강한 noise를 학습하는 과정에서 CLIP branch의 강건성도 함께 향상되는 부가 효과가 있다.
