# 물리 기반 4구 당구 시뮬레이션 명세서 (Revised)

## 목차

1. [좌표계 및 기본 물리량](#1-좌표계-및-기본-물리량)
2. [공의 운동 상태 분류](#2-공의-운동-상태-분류)
3. [큐 타격 역학 (Cue Impact & Squirt)](#3-큐-타격-역학-cue-impact--squirt)
4. [바닥 접촉점 속도 분석](#4-바닥-접촉점-속도-분석)
5. [슬라이딩 마찰 (Sliding Friction)](#5-슬라이딩-마찰-sliding-friction)
6. [구름 구속 조건 (Rolling Constraint)](#6-구름-구속-조건-rolling-constraint)
7. [구름 마찰 (Rolling Friction)](#7-구름-마찰-rolling-friction)
8. [수직축 스핀 감쇠 (Pivot Friction)](#8-수직축-스핀-감쇠-pivot-friction)
9. [매세 효과 — 마찰력 모델 (Masse Physics)](#9-매세-효과--마찰력-모델-masse-physics)
10. [공-공 충돌 (Ball-Ball Collision & Throw)](#10-공-공-충돌-ball-ball-collision--throw)
11. [쿠션 충돌 — 3D 높이 반영 (Rail Collision)](#11-쿠션-충돌--3d-높이-반영-rail-collision)
12. [수직 운동 — 점프샷 (Vertical Motion)](#12-수직-운동--점프샷-vertical-motion)
13. [수치 적분 방법 (Numerical Integration)](#13-수치-적분-방법-numerical-integration)
14. [물리 파라미터 전체 목록](#14-물리-파라미터-전체-목록)

---

## 1. 좌표계 및 기본 물리량

### 1.1 좌표계 정의

```text
          +Y (위)
          │
          │        +Z (테이블 길이 방향, 앞)
          │      ╱
          │    ╱
          │  ╱
          └────────── +X (테이블 폭 방향, 오른쪽)
        원점 = 테이블 중앙, Y=0은 테이블 바닥면
```
### 1.2 공의 기본 물리 상수

| 상수 | 변수명 | 값 | 단위 | 설명 |
|---|---|---|---|---|
| 반지름 | `R` | 0.03275 | m | 65.5mm 공 기준 |
| 질량 | `m` | 0.21 | kg | 표준 중량 |
| 관성 모멘트 | `I` | $0.4 m R^2$ | kg·m² | $2/5 m R^2$ (균질 구) |
| 중력 가속도 | `g` | 9.8 | m/s² | |

> **중요**: 관성 모멘트 역수 $1/I = \frac{1}{0.4 m R^2} = \frac{2.5}{m R^2}$ 임을 명심할 것.

---

## 2. 공의 운동 상태 분류

- **STATIONARY**: 정지
- **SLIDING**: 미끄러짐 ($v_c \neq 0$)
- **ROLLING**: 순수 구름 ($v_c = 0$)

---

## 3. 큐 타격 역학 (Cue Impact & Squirt)

### 3.1 충격량(Impulse) 기반 모델

큐가 공을 타격하는 시간은 매우 짧으므로 힘($F$) 대신 충격량($J_{cue}$)을 사용한다. 사용자가 힘 $F$를 입력하면 접촉 시간 $\Delta t_{contact}$를 곱해 $J_{cue}$로 변환한다.

$$J_{cue} = F \cdot \Delta t_{contact}$$

### 3.2 스쿼트(Squirt/Deflection) 현상

오조준(English) 타격 시, 공은 큐의 진행 방향($\hat{\mathbf{d}}$)과 약간 다른 방향으로 튀어나간다(스쿼트).

타점 오프셋 $o_x$ (좌우)에 따른 편향각 $\theta_{sq}$:
$$\theta_{sq} = \text{sq\_factor} \cdot \arctan\left(\frac{o_x}{R}\right)$$
(여기서 `sq_factor`는 큐대의 특성 계수, 보통 음수. 즉, 오른쪽을 치면 공은 왼쪽으로 편향됨)

보정된 초기 속도 방향 $\hat{\mathbf{d}}'$:
$$\hat{\mathbf{d}}' = \text{RotateY}(\hat{\mathbf{d}}, \theta_{sq})$$

선속도 초기값:
$$\mathbf{v}_0 = \frac{J_{cue}}{m} \hat{\mathbf{d}}'$$

### 3.3 각속도 초기값 (타점 오프셋)

타점 $\mathbf{r}_{hit} = (o_x, o_y, o_z)$ (공 중심 기준 로컬 좌표). $o_z = -\sqrt{R^2 - o_x^2 - o_y^2}$.

토크에 의한 충격량:
$$\mathbf{L} = \mathbf{r}_{hit} \times (J_{cue} \hat{\mathbf{d}})$$

각속도 초기값:
$$\boldsymbol{\omega}_0 = \frac{\mathbf{L}}{I} = \frac{2.5}{m R^2} (\mathbf{r}_{hit} \times J_{cue} \hat{\mathbf{d}})$$

---

## 4. 바닥 접촉점 속도 분석

### 4.1 접촉점 속도

바닥 접촉점 $\mathbf{r}_c = (0, -R, 0)$에서의 미끄럼 속도:

$$\mathbf{v}_c = \mathbf{v} + \boldsymbol{\omega} \times \mathbf{r}_c$$

이를 성분별로 풀면:
$$v_{c,x} = v_x + \omega_z R$$
$$v_{c,z} = v_z - \omega_x R$$
$$v_{c,y} = 0$$

> **Note**: $\omega_y$ (수직축 회전)는 바닥 접촉점의 선속도에 영향을 주지 않는다 (접촉점이 회전축 위에 있음).

---

## 5. 슬라이딩 마찰 (Sliding Friction)

### 5.1 운동 방정식

미끄럼이 존재할 때($|\mathbf{v}_c| > 0$), 마찰력은 미끄럼 반대 방향으로 작용한다.

$$\mathbf{F}_{slide} = -\mu_{slide} \cdot m \cdot g \cdot \frac{\mathbf{v}_c}{|\mathbf{v}_c|}$$

선가속도:
$$\mathbf{a} = \frac{\mathbf{F}_{slide}}{m} = -\mu_{slide} \cdot g \cdot \hat{\mathbf{v}}_c$$

각가속도 (**계수 주의**):
$$\boldsymbol{\tau} = \mathbf{r}_c \times \mathbf{F}_{slide}$$
$$\frac{d\boldsymbol{\omega}}{dt} = \frac{\boldsymbol{\tau}}{I} = \frac{\mathbf{r}_c \times \mathbf{F}_{slide}}{0.4 m R^2}$$

이를 풀면:
$$\frac{d\boldsymbol{\omega}}{dt} = \frac{2.5}{m R^2} (-R \hat{\mathbf{j}} \times \mathbf{F}_{slide}) = \frac{2.5}{m R} (\hat{\mathbf{j}} \times \mathbf{F}_{slide} \text{의 반대})$$

$\mathbf{F}_{slide}$의 크기가 $\mu mg$ 이므로, 접선 방향 각가속도의 크기는:
$$|\alpha| = \frac{2.5 \cdot (\mu m g) \cdot R}{m R^2} = \frac{2.5 \cdot \mu \cdot g}{R}$$

> **보완**: 기존 문서에서 누락되었던 $2.5$ 계수가 포함됨. 이로 인해 공은 훨씬 빠르게 구름 상태(Rolling)로 전환됨.

---

## 6. 구름 구속 조건 (Rolling Constraint)

구름 상태($\mathbf{v}_c = 0$)로 강제 변환:
$$\omega_x = v_z / R, \quad \omega_z = -v_x / R$$

---

## 7. 구름 마찰 (Rolling Friction)

구름 저항은 속도 반대 방향의 힘과 토크로 작용한다.

$$\mathbf{F}_{roll} = -\mu_{roll} \cdot m \cdot g \cdot \hat{\mathbf{v}}$$

속도 감소:
$$\mathbf{v}_{new} = \mathbf{v} + \frac{\mathbf{F}_{roll}}{m} \Delta t$$

속도가 줄어든 만큼 구름 조건 유지를 위해 각속도도 비율에 맞춰 감소시킨다 (Resync).

---

## 8. 수직축 스핀 감쇠 (Pivot Friction)

### 8.1 보정된 물리 모델

접촉면이 점이 아닌 패치(Patch)이므로 $\omega_y$에 저항하는 피벗 토크가 발생한다. 균일한 압력 분포를 가정한 원형 패치의 유효 토크 반지름에 기반한 근사식은 다음과 같다.

$$\tau_{pivot} = -\text{sign}(\omega_y) \cdot \mu_{spin} \cdot m \cdot g \cdot R_{effective}$$

시뮬레이션 단순화를 위해 관성 모멘트 계수(2.5)를 고려한 각감속도는:

$$\frac{d\omega_y}{dt} = - \text{sign}(\omega_y) \cdot \frac{2.5 \cdot \mu_{spin} \cdot g}{R}$$

> **수정**: 기존 식 대비 2.5배 더 빠르게 감쇠함.

---

## 9. 매세 효과 — 마찰력 모델 (Masse Physics)

### 9.1 에너지 보존을 고려한 힘 모델

기존의 "회전 행렬" 방식은 속도(에너지)를 보존하므로 물리적으로 부정확하다. 매세는 바닥 마찰에 의해 경로가 휘는 것이므로, **속도에 수직한 방향의 마찰력**으로 모델링해야 한다.

$$\mathbf{F}_{masse} = C_{masse} \cdot (\hat{\mathbf{v}} \times \hat{\mathbf{y}}) \cdot \omega_y$$

- $C_{masse}$: 매세 효율 상수
- $\hat{\mathbf{v}} \times \hat{\mathbf{y}}$: 진행 방향의 오른쪽 수직 벡터

### 9.2 운동 방정식 적용

$$\mathbf{a}_{masse} = \frac{\mathbf{F}_{masse}}{m}$$
$$\mathbf{v}(t+\Delta t) = \mathbf{v}(t) + (\mathbf{a}_{slide} + \mathbf{a}_{masse}) \Delta t$$

이 방식은 경로를 휘게 만들면서 동시에 $\mathbf{F}_{slide}$와 벡터 합성을 통해 전체 에너지도 자연스럽게 감소시킨다.

---

## 10. 공-공 충돌 (Ball-Ball Collision & Throw)

### 10.1 충격량과 마찰 (Throw Effect)

두 공 충돌 시 법선(Normal) 충격량뿐만 아니라, **접선(Tangent) 마찰 충격량**도 계산해야 "Throw(스핀에 의해 적구가 엉뚱한 방향으로 튀는 현상)"를 구현할 수 있다.

1. **법선 벡터**: $\hat{\mathbf{n}} = (\mathbf{r}_2 - \mathbf{r}_1) / |\mathbf{r}_2 - \mathbf{r}_1|$
2. **접촉점 상대 속도**:
   $$\mathbf{v}_{rel} = (\mathbf{v}_1 + \boldsymbol{\omega}_1 \times R\hat{\mathbf{n}}) - (\mathbf{v}_2 + \boldsymbol{\omega}_2 \times (-R\hat{\mathbf{n}}))$$
3. **접선 상대 속도**:
   $$\mathbf{v}_{rel, t} = \mathbf{v}_{rel} - (\mathbf{v}_{rel} \cdot \hat{\mathbf{n}})\hat{\mathbf{n}}$$
4. **마찰 충격량 $J_t$**:
   쿠션 충돌과 동일한 논리로 Coulomb 마찰 적용.
   $$J_t = \min(\mu_{ball} |J_n|, \ |J_{stop}|)$$
   방향은 $-\hat{\mathbf{v}}_{rel, t}$.

이 $J_t$를 양쪽 공에 반대 방향으로 적용하여 속도와 각속도를 업데이트한다.

---

## 11. 쿠션 충돌 — 3D 높이 반영 (Rail Collision)

### 11.1 쿠션 높이 오프셋 (Rail Height)

실제 당구대 쿠션의 접촉점은 공의 적도(Y=0)보다 약간 위에 있다. 이로 인해 쿠션 충돌 시 수평 스핀($\omega_y$)뿐만 아니라 전진/후진 회전($\omega_x, \omega_z$)도 변한다.

- 쿠션 높이 $h_{rail} \approx 1.6 \sim 1.7 \times R$ (바닥 기준 37~38mm)
- 공 중심 기준 접촉 높이 $h_{offset} \approx 0.14 R$

접촉점 벡터 $\mathbf{r}_c$:
- 우쿠션(-X방향 반발): $(R \cos\phi, \ h_{offset}, \ 0)$ 여기서 $\sqrt{x^2+y^2}=R$ 이므로 $x = \sqrt{R^2 - h_{offset}^2}$
- 단순화: $\mathbf{r}_c = (\pm R_{xy}, \ h_{offset}, \ 0)$ 또는 $(0, \ h_{offset}, \ \pm R_{xy})$

### 11.2 물리적 효과

쿠션 접촉점이 $y > 0$ 이므로:
1. 입사하는 공의 탑스핀($\omega_x$)은 쿠션을 아래로 긁어내려 공이 뜨지 않게 한다.
2. 강한 타격 시 공이 쿠션에 의해 아래로 눌리는 힘(Downforce)을 받는다.
3. 잉글리시($\omega_y$)가 쿠션 마찰에 의해 $y$축 토크뿐만 아니라 $z$축 토크(Roll)로도 일부 변환된다.

### 11.3 업데이트 로직

1. **법선 충격량 $J_n$**: 기존과 동일 (반발계수 적용)
2. **접선 속도 및 마찰 $J_t$**:
   - 접촉점 속도 $\mathbf{v}_c = \mathbf{v} + \boldsymbol{\omega} \times \mathbf{r}_c$ 계산 (3D 벡터 연산 필수).
   - $Z$축 성분뿐만 아니라 $Y$축 성분(공이 들리려는 속도)도 고려 가능.
   - 마찰 충격량 $\mathbf{J}_t$ 계산 (Coulomb 마찰).
3. **상태 업데이트**:
   $$\Delta \mathbf{v} = \frac{\mathbf{J}_n + \mathbf{J}_t}{m}$$
   $$\Delta \boldsymbol{\omega} = \frac{(\mathbf{r}_c \times \mathbf{J}_n) + (\mathbf{r}_c \times \mathbf{J}_t)}{I}$$
   > **핵심**: $\mathbf{r}_c$의 Y성분 때문에 $\mathbf{J}_n$(법선충격량)도 토크를 발생시킨다. 이것이 쿠션의 "Rail Bite" 효과를 만든다.

---

## 12. 수직 운동 — 점프샷 (Vertical Motion)

(기존과 유사하나 쿠션 높이 반영)

- 공이 쿠션에 강하게 충돌할 때, 쿠션 형상(아래로 경사짐)이나 접촉 높이에 의해 하향력(Downward force)이 발생하여 점프볼이 억제되는 효과를 구현할 수 있다 (11.3의 $\mathbf{r}_c \times \mathbf{J}_n$ 항이 이를 처리함).
- 바닥 반발 계수 `TABLE_BOUNCE` 적용.

---

## 13. 수치 적분 방법 (Numerical Integration)

(기존과 동일하게 Sub-stepping 적용 명시적 오일러법 사용)

- 정확한 스핀/마찰 구현을 위해 $\Delta t \le 1\text{ms}$ 권장.

---

## 14. 물리 파라미터 전체 목록

| 파라미터 | 변수명 | 추천값 | 설명 |
|---|---|---|---|
| 공 반지름 | `RADIUS` | 0.03275 m | 4구/3쿠션 공용 (65.5mm) |
| 공 질량 | `MASS` | 0.21 kg | |
| 슬라이딩 마찰 | `MU_SLIDE` | 0.20 | 천과 공 사이 미끄럼 마찰 |
| 구름 마찰 | `MU_ROLL` | 0.015 | 구름 저항 (기존보다 약간 낮춤) |
| 스핀 마찰 | `MU_SPIN` | 0.04 | 제자리 회전 감쇠 |
| 쿠션 높이 오프셋 | `RAIL_H_OFFSET` | 0.005 m | 공 중심보다 5mm 위 (약 0.15R) |
| 쿠션 반발 계수 | `E_CUSHION` | 0.75 | |
| 쿠션 마찰 계수 | `MU_CUSHION` | 0.18 | |
| 공-공 반발 계수 | `E_BALL` | 0.94 | 상아구 특성 |
| 공-공 마찰 계수 | `MU_BALL` | 0.08 | Throw 효과용 마찰 |
| 스쿼트 계수 | `SQUIRT_FACTOR` | -0.02 | 오조준 시 편향 정도 |

---

### 수정 요약 (Revision Summary)

1.  **관성 모멘트(I)**: 모든 회전 가속도 식에 $2.5/mR^2$ 계수를 적용하여 회전 반응 속도 물리적 정합성 확보.
2.  **쿠션 높이(Rail Height)**: 접촉점을 공 중심보다 높게 설정하여 입체적인 쿠션 반사 및 회전 변화 구현.
3.  **큐 타격**: 힘(Force) 대신 충격량(Impulse) 사용 및 스쿼트(Squirt) 현상 추가.
4.  **매세(Masse)**: 에너지 보존 법칙을 위배하는 회전 행렬 대신, 마찰력 벡터 모델을 사용하여 속도 감소와 경로 휘어짐 동시 구현.
5.  **공 충돌**: 단순 스핀 전달이 아닌 마찰 충격량 모델을 도입하여 Throw 현상 구현.