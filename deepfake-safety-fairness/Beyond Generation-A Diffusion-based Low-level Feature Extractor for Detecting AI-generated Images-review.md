# Beyond Generation: A Diffusion-based Low-level Feature Extractor for Detecting AI-generated Images

## Introduction
<img width="1721" height="968" alt="image" src="https://github.com/user-attachments/assets/384bcabc-a8d8-4519-abb9-676d20defc49" />

기존 AI 생성 이미지 탐지 방법들은 생성 과정에서 발생하는 아티팩트를 학습하기 때문에,
새로운 생성 모델이 등장할 때마다 탐지 성능이 저하되는 한계가 있다.

본 논문은 이를 해결하기 위해 **저수준 특징 추출기(low-level feature extractor)** 와
**One-Class Classification 프레임워크**를 결합한 탐지 방법을 제안한다.
생성 모델의 종류와 무관하게 real distribution에서 벗어나는 이미지를 AI 생성물로 판별하는 것이 핵심 기여점이다.

**Main Contributions:**
1. 픽셀 수준의 미세한 차이를 감지하는 저수준 특징 추출기를 설계하며,
   서로 다른 출처의 이미지들이 특징 공간에서 구별되는 분포를 형성하도록 한다.
2. AI 생성 이미지 탐지를 **One-Class Classification** 문제로 정식화하여,
   실제 이미지의 특징 분포에서 벗어나는 이미지를 AI 생성물로 식별한다.

---

## Method

<img width="1723" height="967" alt="image" src="https://github.com/user-attachments/assets/37efa217-052a-4c63-9236-b68de7de2671" />


### 1. Feature Extractor & Pretext Task

생성 모델이 발전함에 따라 AI 이미지와 실제 이미지의 고수준 특징은 점점 구별하기 어려워지고 있다.
반면, upsampling convolution의 spectral aliasing 특성으로 인해
생성 모델은 실제 이미지의 **저수준 특징까지는 흉내 내기 어렵다.**

아래 t-SNE 시각화에서 확인할 수 있듯이, CLIP의 특징 공간에서는
실제 이미지와 여러 생성 모델의 출력이 구분되지 않는다.
반면 제안하는 방식은 실제 이미지와 각 모델의 출처별로 분리된 클러스터가 형성된다.


저수준 특징 추출기의 학습은 **pretext task**로 이루어진다.
실제 이미지에 서로 다른 강도의 노이즈를 추가한 뒤,
사전 학습된 Stable Diffusion 모델로 디노이징한 이미지들을 생성한다.
여기서 확산 모델은 이미지 생성 도구가 아닌 **디노이징 도구**로 활용된다.
추출기의 목표는 원본 이미지와 디노이징된 이미지를 구별하는 것으로,
이 과정을 통해 추출기가 미세한 픽셀 수준 차이에 민감하게 반응하도록 학습된다.


---

### 2. One-Class Classification

학습된 특징 추출기를 탐지에 활용하는 방식은 **One-Class Classification**, 즉 anomaly detection 방식으로 진행된다.

실제 이미지들의 특징 분포를 **Gaussian Mixture Model(GMM)** 로 추정하고,
의심되는 이미지의 log-likelihood score가 설정 임계값 이하이면 AI 생성물로 판별한다.

이 방식의 핵심 장점은 새로운 생성 모델이 등장하더라도
**real distribution 자체는 변하지 않기 때문에** 탐지 성능이 유지된다는 점이다.
