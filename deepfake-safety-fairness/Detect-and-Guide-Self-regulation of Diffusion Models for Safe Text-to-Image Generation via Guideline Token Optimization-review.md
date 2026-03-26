# DAG: Detect-and-Guide for Safe Text-to-Image Generation

## Introduction
<img width="1720" height="964" alt="image" src="https://github.com/user-attachments/assets/d586a5de-91bd-4126-af0c-d1d1ff9424a2" />

최근 재학습 없이 유해 생성을 제어하기 위한 post-hoc 개입 방법으로
**unlearning-based**와 **guiding-based** 방식이 주로 사용된다.
그러나 Unlearning은 blacklist shortcut 문제가 있고,
Guiding 방식은 전역적 가이던스로 인해 mode shift를 유발할 수 있다.

본 논문은 이러한 한계를 해결하기 위해 **Detect-and-Guide(DAG)** 방식을 제안한다.
최적화된 토큰의 교차 어텐션 맵(CAM)을 이용해 유해 개념을 탐지한 뒤,
적응적 강도와 편집 영역을 갖는 안전 가이던스를 적용하여 비안전한 생성을 억제하는 것이 핵심 기여점이다.

**DAG는 안전 생성을 위해 두 가지 핵심 연산으로 구성된다:**
1. **(1) Guideline Detection**: 소규모 데이터셋으로 가이드라인 토큰 임베딩을 최적화하여
   특정 유해 개념과 정렬시키고, CAM을 통해 픽셀 수준의 유해 영역을 탐지한다.
2. **(2) Safe Self-regulation**: 탐지된 영역에 한정하여 적응적 스케일 맵 기반의
   안전 가이던스를 적용함으로써 불필요한 전역적 변형과 mode shift를 방지한다.

---

## Method

### 1. Guideline Detection
<img width="1721" height="965" alt="image" src="https://github.com/user-attachments/assets/64020640-1a8a-451d-82ad-d7fc25720075" />


노이즈가 포함된 잠재 표현 $z_t$와 프롬프트 조건 임베딩 $c_p$를 U-Net에 순전파시키고,
크로스 어텐션 계층의 모든 hidden state $h_t^l$을 캐시한다.

기존 크로스 어텐션에서는 $h_t^l$을 쿼리로, 프롬프트 임베딩 $c_p$를 key·value로 사용한다.
그러나 CLIP 텍스트 인코더의 self-attention 구조로 인해 $c_p$에는
개별 토큰 정보뿐 아니라 문맥적 의미가 혼합되어, 유해 개념과 무관한 배경까지
어텐션이 확장되는 **cross-attention leakage** 문제가 발생한다.
또한 'nude'와 같은 단어는 의미 범위가 넓어, 실제 제거 대상이 아닌 객체까지
누드 개념으로 잘못 탐지될 수 있다.

이를 해결하기 위해 DAG는 $c_p$ 대신 **가이드라인 임베딩 $c$** 를 사용하고,
모델 파라미터를 고정한 채 $h_t^l$과 $c$ 사이의 크로스 어텐션 연산을 수행한다.

**가이드라인 토큰 $c$ 최적화 절차:**
1. LLM으로 누드 개념을 포함한 텍스트-이미지 프롬프트를 생성하고,
   T2I diffusion 모델로 safe/unsafe 이미지로 구성된 소규모 데이터셋을 구축한다.
2. Grounded-SAM으로 누드 영역의 GT mask를 생성한다.
   긍정·부정 라벨을 함께 사용해 unsexual한 부위를 제외하고 실제 누드 영역만 남기며,
   safe 이미지는 누드 개념이 없으므로 zero mask로 설정한다.
3. 캐시된 $h_t^l$과 현재 토큰 $c$에 대해 크로스 어텐션 맵 $\text{CAM}(h_t^l[i], c)$을
   계산하고 이미지 크기로 보간하여 detection map을 얻는다.
4. detection map이 GT mask와 일치하도록 픽셀 단위 cross-entropy 손실을 정의하고,
   이를 최소화하는 방향으로 가이드라인 토큰 임베딩 $c$를 최적화한다.
   이때 **(a) background leakage loss**와 **(b) negative sample loss**를 함께 도입하여
   정밀한 CAM 추출이 가능하도록 한다.

샘플링 과정에서 최적화된 토큰 임베딩 $c^*$는 각 타임스텝마다
유해 개념을 탐지하면서 다양한 시나리오에 대해 일반화 가능성과 개념 특이성을 유지한다.

---

### 2. Safe Self-regulation
<img width="1723" height="960" alt="image" src="https://github.com/user-attachments/assets/833fd39a-6b4e-44f2-b4b6-5f5b5cc354d6" />


DAG는 CFG에서 착안한 safe guidance를 적용한다.
기본 구조는 CFG와 동일하되, 안전 조건 $c_s = \tau_\theta(p_s)$와
대응하는 노이즈 추정 $\epsilon^{c_s}$를 추가한다.
픽셀 수준 스케일 맵 $`\mathbf{S}_{c_s} \cdot \mathbf{M}_{c_s}`$를 통해
현재 샘플을 인접한 safe 모드 방향으로 조정한다.

기존 guiding 기반 방식은 가이던스가 전역적으로 적용되어 유해 개념과 무관한 배경까지
불필요하게 변형되고, 샘플링 초기 강한 가이던스 적용 시 큰 mode shift가 발생한다.

이를 해결하기 위해 DAG는 앞서 얻은 detection map $\hat{A}(c^*)$을 게이트로 사용하여,
**unsafe 개념이 높은 신뢰도로 탐지된 영역에만 탐지 신뢰도에 비례하여 가이던스를 적용**한다.
이를 위해 safe 스케일 맵 $\mathbf{S}_{c_s}$를 두 가지 요소로 구성한다.
<img width="1723" height="966" alt="image" src="https://github.com/user-attachments/assets/d65fb1b7-abda-466d-b701-bdb0ea383e46" />


- **면적 기반 스케일러**: confidence 0.5 이상으로 탐지된 unsafe 영역의 크기를 반영하여
  전체 가이던스 강도를 조절한다.
- **픽셀별 강도 스케일러**: 어텐션 맵의 confidence 값을 픽셀 단위로 재스케일링하여
  이에 비례하게 가이던스 강도를 조절한다.

confidence 0.01 미만인 픽셀에는 가이던스가 작동하지 않도록 게이트를 적용하여
유해 개념과 완전히 무관한 영역을 보존한다.

결과적으로 샘플링 초기 단계에서는 CAM confidence가 낮아 가이던스가 자연스럽게 약하게 작동하여
과도한 mode shift를 방지하고, unsafe 개념이 명확해질수록 가이던스가 점진적으로 강해진다.
이 방식은 모델 파라미터를 변경하지 않고 노이즈 예측값의 선형 재조합만으로
unsafe 성분을 공간적으로 억제하는 적응적 가이던스 구조를 구현한다.
