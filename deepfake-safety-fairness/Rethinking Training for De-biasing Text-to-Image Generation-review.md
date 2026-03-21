# Rethinking Training for De-biasing Text-to-Image Generation: Unlocking the Potential of Stable Diffusion

## Introduction
<img width="1047" height="590" alt="image" src="https://github.com/user-attachments/assets/0abbec16-f558-499a-8720-dfa74107b13c" />

기존 T2I 생성 모델의 디바이어싱 방식은 대부분 추가 학습(fine-tuning)을 통해 이루어졌다.
이에 대한 대안으로 추가 학습 없이 편향을 완화하는 FairDiffusion이 제안된 바 있으나,
image–text alignment가 깨지는 문제가 존재했다.

본 논문은 이를 해결하기 위해 **weak guidance** 기반의 디바이어싱 방법을 제안한다.
추가 학습 없이도 편향을 효과적으로 감소시키는 것이 핵심 기여점이다.

**Main Contributions:**
1. 소수 속성(minority attributes)과 연관된 초기 노이즈들이 무작위로 산재하지 않고
   특정 **"minority regions"** 을 형성함을 발견하였다.
2. 제안한 **mode test**를 통해 minority regions를 식별하고,
   무작위 노이즈가 해당 영역으로 유도되도록 weak guidance를 적용함으로써
   편향을 효율적으로 감소시키는 방법을 제안한다.

---

## Method
<img width="1048" height="588" alt="image" src="https://github.com/user-attachments/assets/b489e62c-ba3f-4fbc-9ee0-d981a5ddf105" />

### 1. Mode Test for Exploring Minority Regions

디바이어싱을 위해 추가 학습이 반드시 필요한가에 대한 의문에서 출발한다.
Diffusion 모델에서 minority attribute와 연관된 초기 노이즈의 분포를 분석하기 위해
**mode test**를 수행하였다.

**Mode test 절차:**
1. 소수 속성이 명시된 프롬프트로 minority 이미지를 생성한다.
2. 해당 이미지에 노이즈를 추가해 초기 노이즈 공간으로 되돌린다.
3. 속성 중립 프롬프트(예: *"A photo of a CEO"*)로 다시 디노이징한다.

추가된 노이즈는 가우시안 분포에서 샘플링되므로 원래 이미지가 위치했던
노이즈 영역 근처에 존재할 가능성이 높다.
따라서 속성 중립 프롬프트로 디노이징했음에도 minority attribute가 재현된다면,
이는 해당 속성과 관련된 노이즈가 **특정 영역에 밀집되어 있음**을 시사한다.

이 분석을 통해, minority regions를 식별하고 weak guidance를 적용하면
추가 학습 없이도 디바이어싱이 가능하다는 핵심 아이디어를 도출하였다.

---

### 2. Weak Guidance
<img width="1049" height="593" alt="image" src="https://github.com/user-attachments/assets/70a442e1-1bf9-4054-80fd-58cc43459073" />


먼저 논문에서는 가이던스를 약하게 하면 bias가 감소하는지를 먼저 실험하였다.
CFG scale 감소 및 text condition embedding에 노이즈를 추가하는 방식을 시도한 결과,
bias는 감소하였으나 **image–text alignment가 깨지는 문제**가 동반되었다.

이를 해결하기 위해 본 논문은 **attribute direction 기반의 weak guidance**를 제안한다.

**방법:**
- Attribute direction을 텍스트 임베딩에 더하여,
  확산 과정이 특정 속성 방향으로 약하게 유도되도록 한다.
- 원래 프롬프트의 의미 보존을 위해, 모든 토큰 위치에 더하는 것이 아니라
  **마스크를 활용해 EOS 이후의 padding token 위치에만 선택적으로 추가**한다.

이를 통해 text alignment를 유지하면서도 효과적인 디바이어싱이 가능함을 보인다.
