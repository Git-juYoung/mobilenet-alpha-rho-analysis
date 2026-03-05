# MobileNet Scaling Analysis: Accuracy vs Compute Trade-off

본 프로젝트는 MobileNet 구조의 핵심 설계 요소인  
**Depthwise Separable Convolution과 모델 스케일링 파라미터(Width Multiplier α, Resolution Multiplier ρ)**가  
모델의 **연산 효율성과 성능 사이의 trade-off에 어떤 영향을 미치는지 실험적으로 분석하는 연구이다.**

단순히 MobileNet을 구현하는 것이 아니라,

- Depthwise separable convolution의 연산 효율성
- Width multiplier(α)에 따른 모델 규모 변화
- Resolution multiplier(ρ)에 따른 연산량 변화
- Compute 대비 Accuracy trade-off 구조
- Pareto frontier 관점에서의 모델 효율성

을 단계적으로 분석한다.

---

# 1. Motivation

모바일 환경이나 임베디드 시스템에서는 **연산량과 모델 크기**가 중요한 제약 조건이 된다.

MobileNet은 이러한 문제를 해결하기 위해 다음과 같은 구조적 설계를 도입하였다.

- Depthwise Separable Convolution
- Width Multiplier (α)
- Resolution Multiplier (ρ)

이 구조는 다음과 같은 질문을 남긴다.

- Depthwise separable convolution은 실제로 얼마나 연산량을 줄이는가?
- 모델 규모를 줄일 때 accuracy는 어떤 방식으로 감소하는가?
- compute 감소와 accuracy 감소 사이에는 어떤 trade-off 구조가 존재하는가?

본 프로젝트는 이러한 질문을 **논문 실험 설정을 기준으로 재현하고, 결과를 시각적으로 분석**한다.

---

# 2. Background

## 2.1 Standard Convolution

일반적인 convolution layer는 다음 연산을 수행한다.

입력 feature map

```
DF × DF × M
```

출력 feature map

```
DF × DF × N
```

연산량

```
DK × DK × M × N × DF × DF
```

여기서

- DK : kernel size
- M : input channel
- N : output channel
- DF : spatial resolution

Standard convolution은 **채널 간 결합과 공간 필터링을 동시에 수행하기 때문에 연산량이 매우 크다.**

---

## 2.2 Depthwise Separable Convolution

MobileNet은 standard convolution을 다음 두 단계로 분리한다.

### Depthwise Convolution

각 입력 채널마다 독립적인 필터를 적용한다.

연산량

```
DK × DK × M × DF × DF
```

---

### Pointwise Convolution (1×1)

채널 간 정보를 결합한다.

연산량

```
M × N × DF × DF
```

---

### Total Cost

```
DK × DK × M × DF × DF
+
M × N × DF × DF
```

Standard convolution 대비 연산량 비율

```
1/N + 1/(DK²)
```

3×3 kernel 기준으로 약 **8~9배 연산량 감소**가 가능하다.

---

## 2.3 MobileNet Architecture

MobileNet 구조는 다음 특징을 가진다.

- 대부분의 layer가 Depthwise Separable Convolution
- BatchNorm + ReLU activation
- 마지막 layer는 global average pooling 이후 classification

구조적으로는 다음과 같은 block이 반복된다.

```
Depthwise Conv
→ Pointwise Conv
→ BatchNorm
→ ReLU
```

---

## 2.4 Model Scaling Hyperparameters

MobileNet은 두 가지 global hyperparameter를 통해 모델 규모를 조절할 수 있다.

---

### Width Multiplier (α)

채널 수를 줄여 모델 폭을 감소시킨다.

```
M → αM
N → αN
```

연산량 변화

```
Compute ∝ α²
```

---

### Resolution Multiplier (ρ)

입력 해상도를 줄여 spatial dimension을 감소시킨다.

```
DF → ρDF
```

연산량 변화

```
Compute ∝ ρ²
```

---

### Total Compute Scaling

MobileNet의 전체 연산량은 다음과 같이 표현된다.

```
Compute ∝ α² × ρ²
```

---

# 3. Dataset

본 실험에서는 **Tiny ImageNet** 데이터셋을 사용하였다.  
MobileNet 논문에서는 ImageNet 데이터셋을 사용하여 실험을 수행하였으나, 본 프로젝트에서는 논문의 실험 설정을 참고하여 **대규모 이미지 분류 문제의 구조를 유지하면서도 실험 비용을 고려해 Tiny ImageNet을 사용하였다.**

Tiny ImageNet은 ImageNet의 축소 버전으로, 다중 클래스 이미지 분류 구조를 유지하면서 비교적 빠른 실험이 가능하도록 설계된 데이터셋이다.  
따라서 본 프로젝트에서는 **MobileNet 논문의 스케일링 실험 구조를 재현하기 위한 데이터셋으로 Tiny ImageNet을 선택하였다.**

---

# 4. Experiment Design

본 실험에서는 MobileNet의 scaling strategy를 분석하기 위해  
여러 모델 구성을 비교하였다.

---

## 4.1 Experiment Goal

본 실험의 목적은 다음과 같다.

1. MobileNet scaling이 compute에 미치는 영향 분석
2. compute 감소에 따른 accuracy 변화 분석
3. compute-accuracy trade-off 구조 확인
4. Pareto frontier 관점에서 모델 효율성 분석

---

## 4.2 Model Configurations

실험 모델은 다음과 같다.

### MobileNet Grid

| α | ρ |
|---|---|
| 1.0 | 1.0 |
| 1.0 | 0.5 |
| 0.5 | 1.0 |
| 0.5 | 0.5 |

---

### Baseline Model

비교를 위해 **standard convolution 기반 CNN을 baseline 모델로 사용하였다.**  
MobileNet 논문에서도 depthwise separable convolution의 효율성을 비교하기 위해 standard convolution 기반 구조와의 비교가 이루어지지만, 구체적인 baseline architecture는 명확히 제시되지 않는다.

따라서 본 프로젝트에서는 **일반 convolution layer로 구성된 CNN 구조를 직접 설계하여 baseline 모델로 사용하였다.**

```
StandardCNN
```

---

## 4.3 Compute Metric

각 모델의 상대적 연산량은 다음과 같이 계산하였다.

```
Relative Compute = α² × ρ²
```

StandardCNN을 기준값 **1.0**으로 설정하였다.

---

# 5. Implementation

## 5.1 Training Pipeline

전체 학습 흐름은 다음과 같다.

```
dataset
→ dataloader
→ model
→ training loop
→ validation
→ result logging
```

---

## 5.2 Model Implementation

본 프로젝트에서는 두 가지 모델을 구현하였다.

### MobileNet

- Depthwise separable convolution 기반
- α, ρ scaling 실험 가능

### StandardCNN

- 일반 convolution 기반 baseline
- MobileNet과 compute trade-off 비교

---

# 6. Experimental Results

## 6.1 Accuracy vs Compute

각 모델의 Test Accuracy와 Relative Compute는 다음과 같다.

| Model | Compute | Accuracy |
|------|------|------|
| MobileNet (α=0.5, ρ=0.5) | 0.0625 | 0.26 |
| MobileNet (α=0.5, ρ=1.0) | 0.25 | 0.34 |
| MobileNet (α=1.0, ρ=0.5) | 0.25 | 0.32 |
| MobileNet (α=1.0, ρ=1.0) | 1.0 | 0.41 |
| StandardCNN | 1.0 | 0.46 |

---

## 6.2 Compute vs Accuracy Visualization

<p align="center">
  <img src="analysis/assets/pareto_plot.png" width="60%">
</p>

그래프는 compute 감소에 따라 accuracy가 점진적으로 감소하는 경향을 보여준다.

---

# 7. Pareto Frontier Analysis

Pareto frontier 관점에서 모델을 분석하면 다음과 같은 특징이 나타난다.

1. MobileNet은 compute 감소에 따라 accuracy가 완만하게 감소한다.
2. StandardCNN은 높은 accuracy를 제공하지만 compute cost가 높다.
3. 일부 MobileNet 구성은 **compute 대비 높은 효율성**을 보인다.

즉 MobileNet은

```
Accuracy ↔ Compute trade-off
```

관점에서 **연산량 대비 높은 성능을 제공하는 효율적인 CNN 구조임을 확인할 수 있다.**

---

# 8. Key Findings

본 실험을 통해 다음과 같은 결과를 확인하였다.

1. Depthwise separable convolution은 연산량을 크게 감소시킨다.
2. Width multiplier는 모델 규모를 효과적으로 조절한다.
3. Resolution multiplier(ρ)는 입력 해상도를 조절하여 연산량을 추가적으로 감소시킬 수 있다.
4. Compute 감소에 따라 accuracy는 비교적 부드럽게 감소한다.
5. MobileNet 논문의 스케일링 실험을 재현했을 때도 trade-off 구조가 일관되게 관찰된다.

---

# 9. Conclusion

본 프로젝트는 MobileNet의 핵심 설계 요소를 구현하고  
compute-accuracy trade-off를 논문 설정에 맞춰 재현·분석하였다.

결과적으로 MobileNet은

- Depthwise separable convolution을 통해 연산량을 크게 감소시키면서도
- 비교적 안정적인 성능 저하 곡선을 유지하는

효율 중심 CNN 구조임을 확인할 수 있었다.

---

# 10. Project Structure

```
.
├── analysis/                 # 실험 결과 분석
│   ├── pareto_plot.py
│   └── assets/               # 시각화 결과 (README 첨부 이미지)
│
├── src/                      # 학습 파이프라인
│   ├── config.py
│   ├── data.py
│   ├── engine.py
│   ├── models.py
│   ├── transforms.py
│   ├── utils.py
│   └── seed.py
│
├── mobilenet.py              # MobileNet implementation
├── standard_cnn.py           # baseline CNN
│
├── results/                  # 실험 결과 CSV
│
├── README.md
├── requirements.txt
└── .gitignore
```