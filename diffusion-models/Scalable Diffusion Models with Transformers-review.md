# Scalable Diffusion Models with Transformers (DiT)
## Introduction
기존 Diffusion 모델은 U-Net을 백본으로 사용하는 방식이 주류였다.
본 논문은 이를 트랜스포머 아키텍처로 대체한 **Diffusion Transformer(DiT)** 를 제안하며,
트랜스포머의 확장성이 Diffusion 모델 성능 향상과 강한 상관관계를 가짐을 실증적으로 보인다.

**Main Contributions:**
1. U-Net 대신 트랜스포머 백본을 사용하는 **DiT Architecture**를 제안하며,
   adaLN-Zero 블록을 통해 학습 안정성을 확보하였다.
2. Gflops를 기준으로 모델 확장성과 성능(FID) 간의 **강한 상관관계**를 실험적으로 규명하였다.

---
## Method
### 1. DiT Architecture
<img width="1722" height="972" alt="image" src="https://github.com/user-attachments/assets/4f35700a-f243-4e93-b376-2c7c387d79dc" />

입력 latent 표현을 patchify해서 각 패치를 선형 임베딩한 뒤 d차원의 토큰 시퀀스 T로 변환한다.
이 토큰들은 N개의 **adaLN-Zero 트랜스포머 블록**을 통과하는데,
이 블록은 잔차 블록을 항등 함수로 초기화하면 학습이 안정적이라는 ResNet에서의 관찰에 기반한 설계이다.

**adaLN-Zero 블록 설계:**
- 조건으로부터 LayerNorm의 스케일 $\gamma$와 바이어스 $\beta$를 회귀한다.
- 각 잔차 연결 직전에 적용되는 차원별 스케일 파라미터 $\alpha$까지 조건으로부터 함께 회귀한다.
- MLP가 모든 $\alpha$를 0으로 초기화하여, 초기 상태에서 전체 블록이 항등 함수처럼 동작하도록 한다.

마지막 DiT 블록 이후에는 토큰들을 노이즈 예측과 대각 공분산 예측으로 선형 디코딩하고,
이를 원래 공간 배치로 재배열하여 최종 출력으로 사용한다.

---
### 2. Correlation between Gflops and Model Performance
<img width="1722" height="965" alt="image" src="https://github.com/user-attachments/assets/161a9d1e-c7c6-4dfe-a2fb-c3f5e30102b5" />
<img width="1722" height="967" alt="image" src="https://github.com/user-attachments/assets/7f9cfded-465d-4ad3-9855-b100eba421c1" />


논문에서는 Gflops 기준으로 모델의 확장성을 실험하였다.
패치 크기를 고정한 실험과 모델 크기를 고정한 실험 두 가지로 나누어 분석한다.

트랜스포머의 깊이와 너비를 늘리거나 입력 토큰 수를 증가시켜 Gflops를 확대한 DiT 모델일수록,
일관되게 더 낮은 FID를 달성한다.
모델 크기를 고정한 경우, 총 파라미터 수는 사실상 변하지 않고 Gflops만 증가하는데도 성능이 향상되어,
**Gflops 확장 자체가 성능 향상과 직결됨**을 확인할 수 있다.

FID-50k를 모델 Gflops에 대해 시각화했을 때,
서로 다른 DiT 구성이더라도 총 Gflops가 유사하면 FID 값 또한 유사하게 나타난다.
이를 통해 모델의 성능이 Gflops와 강한 상관관계를 가진다는 것을 알 수 있다.
