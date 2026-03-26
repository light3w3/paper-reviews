# Implicit Bias Injection Attacks against Text-to-Image Diffusion Models
## Introduction
<img width="1375" height="772" alt="image" src="https://github.com/user-attachments/assets/636dc5d5-2d6c-44ff-8d11-7bae68960dc6" />


두 번째 논문은 피부색이나 성별처럼 눈에 바로 드러나는 명시적 편향이 아니라, **암묵적 편향**을 다룬 논문이다.

암묵적 편향이란, 같은 프롬프트라도 사용자에 따라 서로 다른 분위기의 이미지를 생성하는 것으로, 예를 들어 개발도상국 사용자에게는 어둡고 지친 이미지를, 선진국 사용자에게는 밝고 자신감 있는 이미지를 생성하는 경우다. 이러한 암묵적 편향은 표정, 자세, 배경 같은 미묘한 요소로 나타나기 때문에 탐지가 어렵고, 다양한 프롬프트에 걸쳐 일관되게 나타나는 문제가 있다.

이를 해결하기 위해 논문에서는 프롬프트 임베딩 공간에서 편향 방향 벡터를 계산하고 사용자별로 동적으로 주입하는 **IBI-Attack**을 제안한다. 주요 기여는 다음과 같다.

- (i) 명시적 시각 특징 없이 다양한 의미 형태로 표현되는 새로운 암묵적 편향 유형 제안
- (ii) 텍스트 임베딩 공간의 편향 방향 벡터가 다양한 의미 표현을 인코딩하고 일반화됨을 발견
- (iii) 다양한 입력에 적응적으로 편향을 표현하는 모듈 제안

---

## Methodology — IBI-Attacks

IBI-Attacks는 세 단계로 구성된다.

- Step 1. 편향 방향 벡터 계산 (Directional Vector Generation)
- Step 2. 적응형 특징 선택 (Adaptive Feature Selection)
- Step 3. 사용자 추론 (User Inference)

---

### Step 1. 편향 방향 벡터 계산
<img width="1377" height="760" alt="image" src="https://github.com/user-attachments/assets/f9452d23-0505-4e03-9d98-42d16078c7c2" />


첫 단계인 Directional Vector Generation은 텍스트 임베딩 공간에서 편향의 방향을 나타내는 벡터를 구하는 단계다.

먼저 LLM을 활용해 두 가지 문장 집합을 생성한다. 하나는 중립 문장 집합 $X_{neu}$이고, 다른 하나는 중립적인 문장의 명사 앞에 편향적인 형용사만 추가한 편향 문장 집합 $X_{bias}$다.

> 문장 구조를 바꾸지 않는 이유는 편향과 무관한 정보가 임베딩에 섞이는 것을 막기 위해서다.

이렇게 두 문장 집합을 생성한 뒤 이를 각각 텍스트 인코더 $\varphi(\cdot)$에 통과시켜 임베딩으로 변환한다.

$$v_i^{neu} = \varphi(x_i^{neu}), \quad v_i^{bias} = \varphi(x_i^{bias})$$

그리고 두 집합 간 평균 차이를 편향 방향 벡터로 정의한다.

$$v^{diff} = \frac{1}{N} \sum_{i=1}^{N} (v_i^{bias} - v_i^{neu})$$

논문에서는 실험적으로 이 평균 벡터 하나가 표정, 자세, 배경 등 다양한 편향 표현을 포함하고 있고, 동시에 여러 프롬프트에 더해도 그대로 일반화가 가능함을 보인다.

---

### Step 2. 적응형 특징 선택
<img width="1375" height="776" alt="image" src="https://github.com/user-attachments/assets/226bc81b-e114-4208-9d5d-057d11117985" />


Step 2는 Adaptive Feature Selection 단계로, Step 1에서 얻은 고정 벡터를 그대로 더하면 프롬프트마다 편향이 과하거나 부족하게 주입되는 문제가 발생한다. 따라서 사용자 입력에 따라 편향을 조절하는 동적 모듈을 설계하는 단계에 해당한다.

텍스트 임베딩은 $D \times L$의 2차원 구조를 가지며, $D$는 임베딩 차원(각 토큰의 의미 측면), $L$은 토큰 차원(어떤 단어인지)이다. 피처 선택 모듈은 이 두 축 방향으로 번갈아 어텐션을 계산하는데, 먼저 토큰 차원을 평균 내어 문장 전체의 특성을 파악한 뒤 이를 기반으로 임베딩 방향 가중치를 결정하고, 반대로 임베딩 차원을 평균 내어 토큰 방향 가중치를 조정한다.

정리하자면 한 방향을 압축해서 전체 맥락을 파악하고, 그 정보로 반대 방향을 얼마나 강조할지 결정하는 방식이다. 이 과정을 통해 조정된 편향 벡터와 최종 편향 임베딩이 계산된다.

$$\tilde{v}^{diff} = \text{MLP}_\theta(\text{Avg}(v^{user})) \odot v^{diff}$$

$$\tilde{v}^{bias} = v^{user} + \tilde{v}^{diff}$$

이 모듈의 학습은 Step 1에서 만든 프롬프트 $(v_i^{neu}, v_i^{bias})$ 쌍을 재활용해서, 중립 임베딩에서 편향 임베딩 방향으로 이동할 수 있도록 아래 loss 함수로 학습한다.

$$\text{Loss} = \frac{1}{N} \sum_{i=1}^{N} \| v_i^{diff} - \text{MLP}_\theta(\text{Avg}(v_i^{neu})) \odot v^{diff} \|^2$$

---

### Step 3. 사용자 추론
<img width="1379" height="774" alt="image" src="https://github.com/user-attachments/assets/4791d4df-32a4-41a8-88c8-a60836397a61" />


마지막으로 학습이 완료된 적응형 모듈을 사전학습된 확산 모델의 텍스트 인코더 바로 뒤에 플러그인 형태로 삽입한다. 전체 아키텍처에서 위쪽은 정상 생성 경로(Clean Route), 아래쪽은 편향 주입 경로(Bias Injection)에 해당한다.

User Embedding이 생성된 직후, Adaptive Feature Selection Module이 개입하여 편향을 주입하고, 이렇게 생성된 Biased Embedding이 확산 모델로 전달되어 최종적으로 편향된 이미지가 생성된다.
