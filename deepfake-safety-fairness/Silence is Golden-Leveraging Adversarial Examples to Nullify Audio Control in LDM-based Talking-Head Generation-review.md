# Silencer: Protecting Portrait from Talking-Head Video Generation

## Introduction
<img width="1719" height="968" alt="image" src="https://github.com/user-attachments/assets/8b62bcb3-9922-4a8e-8fdb-abf36f4314ef" />

기존 LDM 기반 Talking-Head 방어 방식은 대부분 이미지 편집이나 스타일 모방 방지용으로 설계된
AdvDM, PhotoGuard, Mist 등의 기법을 활용하였다.
그러나 이 방법들은 ReferenceNet 구조를 사용하는 Talking-Head 모델에서
얼굴 이미지를 완전히 파괴하지 않는 이상 음성 신호가 여전히 입술 움직임을 제어할 수 있다.
또한 JPEG 압축, DiffPure, GrIDPure 등의 purification 기법으로 perturbation이 제거되는 한계가 있다.

본 논문은 이를 해결하기 위해 **Silencer**를 제안한다.
영상이 생성되더라도 **입이 움직이지 않고 정적인 상태를 유지하도록** 하는 perturbation 설계 방식이 핵심 기여점이다.

**기존 방어법이 Talking-Head에 실패하는 이유:**
1. **음성 신호 차단 불가**: ReferenceNet 구조 때문에 얼굴을 완전히 가리지 않으면 음성이 여전히 입술을 움직임
2. **Purification 취약**: JPEG 압축, DiffPure, GrIDPure 등으로 보호 노이즈를 제거하면 방어 효과가 사라짐

---

## Method

Silencer는 두 단계(two-stage)로 구성된다.

### Stage I: Nullifying Loss
<img width="1720" height="967" alt="image" src="https://github.com/user-attachments/assets/127bb13d-1d3e-47bf-bf9c-8e670074cf5b" />


현재 adversarial image $p^{n-1}$를 VAE Encoder에 통과시켜 잠재 표현 $\hat{z}_0$을 얻는다.
이 $\hat{z}_0$는 ReferenceNet에 입력되어 얼굴 외형 특징을 추출하는 데 사용되고,
동시에 랜덤 노이즈 $\epsilon$을 더해 $\hat{z}_t$를 만들어 U-Net에 입력한다.
U-Net은 ReferenceNet에서 받은 얼굴 특징과 음성 $a$를 함께 참고하여 예측 노이즈를 출력한다.

Silencer의 핵심 아이디어는 **원본 사진 $p$ 자체를 Ground Truth로 설정**하여,
어떤 음성 $a$가 들어와도 생성 결과가 원본 사진처럼 유지되도록 강제하는 것이다.

$$L_N = \mathbb{E}_t \mathbb{E}_{\mathcal{E}(p),\, p,\, a_i,\, \epsilon} \left[ \| \epsilon - \epsilon_\theta(\hat{z}_t,\, t,\, p,\, a_i) \|_2^2 \right]$$

여기서 $\hat{z}_0 = \mathcal{E}(p)$, 즉 원본 사진의 잠재 표현을 사용한다.
이 손실을 PGD **gradient descent** 방향으로 낮추면 모델의 목표 지점이 원본 사진으로 고정되어,
생성된 영상이 실제로 말하는 영상과 점점 멀어지게 된다.
이를 $n$번 반복하여 최종 adversarial image $p^n$을 얻는다.

**기존 접근법과의 비교:**
- **Semantic Loss (AdvDM)**: 학습 손실을 교란하는 방식이나, ground truth 프레임이 필요하다.
  Talking-Head에서는 어떤 사람이 어떤 말을 할지 미리 알 수 없어 적용이 불가능하다.
- **Texture Loss (PhotoGuard)**: ground truth가 필요 없지만 음성 신호를 직접 차단하지 못한다.
  얼굴을 완전히 가리지 않으면 오디오가 여전히 작동한다.

> **타임스텝 실험**: 1000개 타임스텝 중 **[200, 300] 구간이 최적**으로,
> 동기화 보호(Sync 4.26)와 사진 품질 보존(I-PSNR 32.34) 사이의 균형이 가장 좋다.

---

### Stage II: Anti-purification
<img width="1720" height="966" alt="image" src="https://github.com/user-attachments/assets/9e08e7f5-fe45-4453-bfc0-5d403074bc1e" />


Stage I에서 만든 $p^n$은 이미지 공간에서 PGD로 최적화된 단순한 구조이기 때문에,
DiffPure, GrIDPure 같은 purification 기법으로 perturbation이 쉽게 제거된다.

이를 해결하기 위해 **LDM의 잠재 공간(latent space)에서 직접 perturbation을 최적화**한다.
원본 사진 $p_0$를 DDIM Inversion을 통해 잠재 벡터 $p_t$로 변환한 뒤,
그 잠재 벡터 자체를 최적화하여 새로운 adversarial image $p'_0$를 얻는다.
잠재 공간의 perturbation은 이미지 공간의 것과 구조가 다르기 때문에 기존 purification 기법이 효과적으로 작동하지 않는다.

$$L_{AP} = \lambda_1 L_N + \lambda_2 \| \mathcal{E}(p'_0) - \mathcal{E}(p^n) \|_2^2$$

- $L_N$: 음소거 효과 유지
- $\lambda_2$ 항: Stage I 결과 $p^n$을 가이드로 활용하여 얼굴 정체성 변형 방지

**기존 방법(ACA)과의 차이**: ACA는 skip gradient로 잠재 벡터를 최적화했지만
Talking-Head 모델에서 얼굴 정체성이 크게 망가지는 문제가 있었다.
Silencer는 Stage I의 adversarial 결과 $p^n$을 가이드로 활용함으로써 이를 해결한다.

**얼굴 마스크 전략**: 처음 $s=100$번의 반복은 전체 이미지에 대해 최적화를 진행하고,
이후부터는 얼굴 영역을 마스크로 제외한다.
이를 통해 purification 저항성은 높이면서 얼굴 형태를 자연스럽게 보존한다.
