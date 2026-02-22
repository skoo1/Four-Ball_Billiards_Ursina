# 4구 당구 시뮬레이터 — 물리 역학 완전 해설

> **대상 독자**: 대학교 학부 수준 (일반물리학, 고전역학 수강 완료 기준)
> **구현 파일**: `physics.py`
> **엔진**: Python + NumPy, 명시적 오일러(Explicit Euler) 수치 적분

---

## 목차

1. [좌표계 및 기본 물리량](#1-좌표계-및-기본-물리량)
2. [공의 운동 상태 분류](#2-공의-운동-상태-분류)
3. [큐 타격 역학 (Cue Impact)](#3-큐-타격-역학-cue-impact)
4. [바닥 접촉점 속도 분석](#4-바닥-접촉점-속도-분석)
5. [슬라이딩 마찰 (Sliding Friction)](#5-슬라이딩-마찰-sliding-friction)
6. [구름 구속 조건 (Rolling Constraint)](#6-구름-구속-조건-rolling-constraint)
7. [구름 마찰 (Rolling Friction)](#7-구름-마찰-rolling-friction)
8. [수직축 스핀 감쇠 (Pivot Friction)](#8-수직축-스핀-감쇠-pivot-friction)
9. [매세 효과 — 경로 곡선 (Masse / Curve Effect)](#9-매세-효과--경로-곡선-masse--curve-effect)
10. [공-공 충돌 (Ball-Ball Collision)](#10-공-공-충돌-ball-ball-collision)
11. [쿠션 충돌 (Cushion Collision)](#11-쿠션-충돌-cushion-collision)
12. [수직 운동 — 점프샷 (Vertical Motion)](#12-수직-운동--점프샷-vertical-motion)
13. [수치 적분 방법 (Numerical Integration)](#13-수치-적분-방법-numerical-integration)
14. [물리 파라미터 전체 목록](#14-물리-파라미터-전체-목록)
15. [단계별 시뮬레이션 루프 요약](#15-단계별-시뮬레이션-루프-요약)

---

## 1. 좌표계 및 기본 물리량

### 1.1 좌표계 정의

```
         +Y (위)
          │
          │       +Z (테이블 길이 방향, 앞)
          │      ╱
          │    ╱
          │  ╱
          └────────── +X (테이블 폭 방향, 오른쪽)
        원점 = 테이블 중앙
```

| 축 | 방향 | 범위 |
|---|---|---|
| X | 테이블 폭 방향 (오른쪽 +) | `[-TABLE_WIDTH/2, +TABLE_WIDTH/2]` |
| Y | 수직 (위 +) | 0 = 테이블 바닥면 |
| Z | 테이블 길이 방향 (앞 +) | `[-TABLE_LENGTH/2, +TABLE_LENGTH/2]` |

> **오른손 좌표계(Right-Hand Coordinate System)**: 오른손의 엄지=+X, 검지=+Y, 중지=+Z

### 1.2 각속도 부호 규칙 (Right-Hand Rule)

각속도 벡터 **ω** = (ω_x, ω_y, ω_z)에서, 각 성분의 양수 방향은 해당 축을 오른손 엄지로 가리킬 때 나머지 손가락이 감기는 방향:

| 성분 | 물리적 의미 | 양수일 때 |
|---|---|---|
| ω_x | X축 기준 회전 | 위→뒤 방향 (탑스핀 효과의 일부) |
| ω_y | Y축 기준 회전 | 위에서 볼 때 반시계 방향 (좌잉글리시) |
| ω_z | Z축 기준 회전 | 오른쪽→위 방향 |

### 1.3 공의 기본 물리 상수

| 상수 | 변수명 | 값 | 단위 | 설명 |
|---|---|---|---|---|
| 반지름 | `R` | 0.033 | m | 국제 4구 규격 |
| 질량 | `m` | 0.21 | kg | 상아구 기준 |
| 관성 모멘트 | `I` | 2/5·m·R² | kg·m² | 균질 구(solid sphere) |
| 중력 가속도 | `g` | 9.8 | m/s² | |

**관성 모멘트 유도**:

균질 고체 구의 관성 모멘트는 반지름 방향으로 적분하여 구한다:

$$I = \int r^2 \, dm = \frac{2}{5} m R^2$$

수치 대입:

$$I = \frac{2}{5} \times 0.21 \times 0.033^2 \approx 9.15 \times 10^{-5} \ \text{kg·m}^2$$

---

## 2. 공의 운동 상태 분류

공은 항상 세 가지 상태 중 하나에 있다:

```
STATIONARY ──── 속도/각속도가 임계값 이하 → 정지
SLIDING    ──── 접촉점에 미끄럼 존재 (v_contact ≠ 0)
ROLLING    ──── 순수 구름 (v_contact = 0)
```

**상태 전이 규칙**:

```
큐 타격 → SLIDING
SLIDING + 마찰 → vc가 0에 수렴 → ROLLING
ROLLING + 감속 → 속도 ≤ 임계값 → STATIONARY
충돌 후 → SLIDING (재분류)
```

임계값:

$$|v| < 10^{-5} \ \text{m/s}, \quad |\boldsymbol{\omega}| < 10^{-4} \ \text{rad/s}$$

---

## 3. 큐 타격 역학 (Cue Impact)

### 3.1 선형 속도 변화

큐가 방향 벡터 $\hat{\mathbf{d}}$ 로 힘 $F$ 를 가할 때:

$$\Delta \mathbf{v} = \frac{F}{m} \hat{\mathbf{d}}$$

이것은 운동량-충격량 정리의 직접 적용이다 ($\mathbf{J} = F \cdot \hat{\mathbf{d}}$, $\Delta\mathbf{v} = \mathbf{J}/m$).

### 3.2 각속도 변화 (타점 오프셋 있을 때)

큐가 공 정면 기준으로 오프셋 $(o_x, o_y)$ 에서 타격할 때:

**Step 1**: 접촉점의 z 좌표 (공 표면 위)

$$o_z = -\sqrt{R^2 - o_x^2 - o_y^2}$$

음수인 이유: 큐는 공의 앞면(-방향)에서 접근하므로 접촉점의 z 성분은 음수.

**Step 2**: 큐 방향에 맞게 로컬 좌표 → 월드 좌표 변환

큐의 진행 방향 $\hat{\mathbf{f}}$ (xz 평면에 투영, 정규화), 위 벡터 $\hat{\mathbf{u}} = (0,1,0)$, 오른쪽 벡터 $\hat{\mathbf{r}} = \hat{\mathbf{u}} \times \hat{\mathbf{f}}$:

$$\mathbf{c} = \hat{\mathbf{r}} \cdot o_x + \hat{\mathbf{u}} \cdot o_y + \hat{\mathbf{f}} \cdot o_z \quad \text{(월드 좌표의 접촉 벡터)}$$

**Step 3**: 토크와 각속도 변화

$$\boldsymbol{\tau} = \mathbf{c} \times \mathbf{F}, \quad \mathbf{F} = F \hat{\mathbf{d}}$$

$$\Delta \boldsymbol{\omega} = \frac{\boldsymbol{\tau}}{I} = \frac{\mathbf{c} \times (F\hat{\mathbf{d}})}{I}$$

**타점 효과 요약**:

| 타점 | $o_x$ | $o_y$ | 주요 효과 |
|---|---|---|---|
| 중앙 | 0 | 0 | 순수 슬라이딩 시작 |
| 상단 (Follow) | 0 | +값 | 탑스핀 (ω_x +) → 빨리 구름 전환 |
| 하단 (Draw) | 0 | -값 | 백스핀 (ω_x -) → 역행 후 구름 |
| 우측 (Right English) | +값 | 0 | 우잉글리시 (ω_y -) → 경로 곡선 |
| 좌측 (Left English) | -값 | 0 | 좌잉글리시 (ω_y +) → 경로 곡선 |

---

## 4. 바닥 접촉점 속도 분석

### 4.1 접촉점 위치

공이 바닥(Y=0) 위에 있을 때, 접촉점은 공 중심에서 정확히 아래:

$$\mathbf{r}_c = (0, -R, 0)$$

### 4.2 접촉점 표면 속도

강체의 회전운동에 의해 접촉점의 속도는:

$$\mathbf{v}_c = \mathbf{v} + \boldsymbol{\omega} \times \mathbf{r}_c$$

외적을 전개하면:

$$\boldsymbol{\omega} \times \mathbf{r}_c = (\omega_x, \omega_y, \omega_z) \times (0, -R, 0)$$

$$= \begin{vmatrix} \hat{i} & \hat{j} & \hat{k} \\ \omega_x & \omega_y & \omega_z \\ 0 & -R & 0 \end{vmatrix}$$

$$= \hat{i}(\omega_y \cdot 0 - \omega_z \cdot (-R)) - \hat{j}(\omega_x \cdot 0 - \omega_z \cdot 0) + \hat{k}(\omega_x \cdot (-R) - \omega_y \cdot 0)$$

$$= (\omega_z R, \; 0, \; -\omega_x R)$$

따라서 수평 접촉 속도 (y 성분 제외):

$$\boxed{v_{c,x} = v_x + \omega_z R, \qquad v_{c,z} = v_z - \omega_x R}$$

### 4.3 핵심 관찰: ω_y는 접촉 속도에 기여하지 않는다

위 결과에서 $\omega_y$ 가 나타나지 않는다. 이는 **접촉점이 y축(ω_y의 회전축) 위에 있기 때문**이다:

$$\omega_y \hat{j} \times (-R\hat{j}) = -\omega_y R (\hat{j} \times \hat{j}) = 0$$

이것이 수직축 스핀(ω_y)이 슬라이딩 마찰과 직접 결합하지 않는 이유이며, 별도의 피벗 마찰 모델이 필요한 이유이다 ([8절](#8-수직축-스핀-감쇠-pivot-friction) 참조).

---

## 5. 슬라이딩 마찰 (Sliding Friction)

### 5.1 마찰력

공이 SLIDING 상태일 때, 쿨롱(Coulomb) 마찰 법칙을 적용한다:

$$\mathbf{F}_{fric} = -\mu_{slide} \cdot m \cdot g \cdot \hat{\mathbf{v}}_c$$

여기서 $\hat{\mathbf{v}}_c = \mathbf{v}_c / |\mathbf{v}_c|$ 는 접촉 미끄럼 방향의 단위벡터. 마찰력은 항상 미끄럼을 **반대 방향**으로 제동한다.

### 5.2 선형 및 각운동방정식

$$\frac{d\mathbf{v}}{dt} = \frac{\mathbf{F}_{fric}}{m} = -\mu_{slide} \cdot g \cdot \hat{\mathbf{v}}_c$$

$$\frac{d\boldsymbol{\omega}}{dt} = \frac{\boldsymbol{\tau}}{I} = \frac{\mathbf{r}_c \times \mathbf{F}_{fric}}{I}$$

토크를 전개하면 ($\mathbf{r}_c = (0,-R,0)$, $\mathbf{F}_{fric} = (F_x, 0, F_z)$):

$$\boldsymbol{\tau} = \mathbf{r}_c \times \mathbf{F}_{fric} = (-R F_z, \; 0, \; R F_x)$$

따라서:

$$\frac{d\omega_x}{dt} = \frac{-R F_z}{I}, \qquad \frac{d\omega_y}{dt} = 0, \qquad \frac{d\omega_z}{dt} = \frac{R F_x}{I}$$

**중요**: 바닥 마찰 토크는 $\omega_y$ 를 변경하지 못한다. 이것은 앞서 보인 것과 일치한다.

### 5.3 제로 교차 감지 (Overshoot 방지)

오일러 적분의 문제점: 접촉 속도가 0에 가까울 때, 한 스텝에서 방향이 반전될 수 있다(over-correction). 이를 방지하기 위해:

1. 다음 스텝의 접촉 속도를 예측:
$$\mathbf{v}_c^{new} = (\mathbf{v} + \mathbf{a}\Delta t) + (\boldsymbol{\omega} + \boldsymbol{\alpha}\Delta t) \times \mathbf{r}_c$$

2. 방향이 반전되면 ($\mathbf{v}_c \cdot \mathbf{v}_c^{new} < 0$), 전환 시각 $t^*$ 를 찾아 부분 적분:

$$t^* = \arg\min_t |\mathbf{v}_c + t(\mathbf{v}_c^{new} - \mathbf{v}_c)| \approx \Delta t \cdot \frac{-\mathbf{v}_c \cdot \Delta\mathbf{v}_c}{|\Delta\mathbf{v}_c|^2}$$

3. $[0, t^*]$ 구간은 슬라이딩으로, $[t^*, \Delta t]$ 구간은 구름 마찰로 처리.

---

## 6. 구름 구속 조건 (Rolling Constraint)

### 6.1 순수 구름의 정의

바닥과의 접촉점에서 **미끄럼이 없음**:

$$\mathbf{v}_c = \mathbf{v} + \boldsymbol{\omega} \times \mathbf{r}_c = \mathbf{0}$$

수평 성분만 쓰면:

$$v_x + \omega_z R = 0$$
$$v_z - \omega_x R = 0$$

이를 풀면:

$$\boxed{\omega_x = \frac{v_z}{R}, \qquad \omega_z = -\frac{v_x}{R}}$$

### 6.2 부호 직관

- 공이 +Z 방향으로 이동 ($v_z > 0$): $\omega_x = +v_z/R > 0$
  → X축 기준 양의 회전 = 위→뒤 방향 = **탑스핀** ✓
- 공이 +X 방향으로 이동 ($v_x > 0$): $\omega_z = -v_x/R < 0$
  → Z축 기준 음의 회전 = 앞→위 방향에서 볼 때 시계 방향 ✓

### 6.3 코드에서의 구현

```python
def _snap_to_rolling(ball):
    R = ball.radius
    ball.angular_velocity[0] = ball.velocity[2] / R   # ωx = vz/R
    ball.angular_velocity[2] = -ball.velocity[0] / R  # ωz = -vx/R
    # ωy 는 건드리지 않음 (독립적으로 감쇠)
```

---

## 7. 구름 마찰 (Rolling Friction)

### 7.1 물리적 원인

이상적인 강체 이론에서는 순수 구름에 마찰이 없다. 실제로는:
- 공/천의 탄성 변형에 의한 **점소성 손실(viscoelastic loss)**
- 공이 천의 섬유를 파고드는 **침투 저항**
- 공기 저항

이 모든 효과를 하나의 구름 마찰 계수 $\mu_{roll}$ 로 통합 모델링한다.

### 7.2 운동방정식

$$a_{decel} = \mu_{roll} \cdot g$$

진행 방향의 반대로 일정한 크기의 감속이 적용된다:

$$\mathbf{v}(t + \Delta t) = \hat{\mathbf{v}} \cdot \max\left(0, |\mathbf{v}| - \mu_{roll} \cdot g \cdot \Delta t\right)$$

속도 업데이트 후 구름 조건 유지를 위해 `_snap_to_rolling()` 재적용.

---

## 8. 수직축 스핀 감쇠 (Pivot Friction)

### 8.1 왜 별도 처리가 필요한가?

[4.3절](#43-핵심-관찰-ω_y는-접촉-속도에-기여하지-않는다)에서 보인 것처럼, $\omega_y$ 는 바닥 접촉점의 선속도를 만들지 않는다. 따라서 슬라이딩 마찰 모델로는 $\omega_y$ 를 감쇠시킬 수 없다.

### 8.2 피벗 마찰의 물리적 원인

실제 공-바닥 접촉은 **점(point)이 아닌 작은 타원형 패치(contact patch)**다. 공이 $\omega_y$ 로 회전할 때, 패치 내의 각 점이 각기 다른 방향으로 미끄러진다:

```
패치 중심에서 반지름 r인 점의 접선 속도 = ωy · r
```

이 분포된 마찰력을 패치 전체에서 적분하면 $\omega_y$ 에 반대되는 토크가 생기며, 이를 **피벗 마찰(pivot/twist friction)** 이라 한다.

### 8.3 근사 모델

패치 크기가 공 반지름에 비해 매우 작다고 가정하면, 피벗 토크의 크기는:

$$|\tau_{pivot}| = \mu_{spin} \cdot m \cdot g \cdot R$$

이에 의한 각감속:

$$\left|\frac{d\omega_y}{dt}\right| = \frac{|\tau_{pivot}|}{I} \approx \frac{\mu_{spin} \cdot g}{R}$$

($I = \frac{2}{5}mR^2$ 를 대입하면 분모에 $R$이 남는다)

이산 적분에서:

$$\omega_y^{new} = \omega_y - \text{sign}(\omega_y) \cdot \frac{\mu_{spin} \cdot g}{R} \cdot \Delta t$$

$\omega_y = 0$ 을 지나치지 않도록 clamp 처리.

### 8.4 $\omega_y$ 는 구름 스핀($\omega_x$, $\omega_z$)으로 변환되지 않는다

정리: 바닥과의 상호작용에서 $\omega_y$ 의 역할:

| 현상 | 원인 | 관련 변수 |
|---|---|---|
| $\omega_y$ 감쇠 | 피벗 마찰 (`MU_SPIN`) | $\omega_y \to 0$ |
| 경로 곡선 | 매세 효과 (9절) | $\omega_y \to \Delta\mathbf{v}$ 방향 변화 |
| $\omega_x$, $\omega_z$ 변화 | **없음** (직접 결합 없음) | — |

---

## 9. 매세 효과 — 경로 곡선 (Masse / Curve Effect)

### 9.1 물리 현상

$\omega_y \neq 0$ (수직축 스핀)인 공이 전진할 때, 경로가 원호처럼 휘어진다. 이를 **매세(masse)** 또는 **커브(curve)** 효과라 한다.

실제 원인은 복잡하나 (클로스와의 상호작용, 비대칭 압력 분포), 시뮬레이터에서는 다음과 같이 근사한다:

### 9.2 수식

단위 시간당 속도 방향의 회전률(곡선율):

$$\Omega_{curve} = \frac{\omega_y \cdot \mu_{slide} \cdot g \cdot R}{|\mathbf{v}| + \varepsilon}$$

여기서 $\varepsilon = 0.1$ m/s는 저속에서의 수치 발산 방지용 상수.

한 스텝에서의 회전각:

$$\theta = \Omega_{curve} \cdot \Delta t$$

속도 벡터를 Y축 기준으로 $\theta$ 만큼 회전:

$$\begin{pmatrix} v_x' \\ v_z' \end{pmatrix} = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix} \begin{pmatrix} v_x \\ v_z \end{pmatrix}$$

### 9.3 직관적 해석

$\omega_y > 0$ (위에서 볼 때 반시계, 좌잉글리시):

- 공이 +Z 방향으로 달리면 경로가 좌측(−X)으로 휘어짐
- 쿠션 후 잉글리시 효과와 구별되는 현상

$\omega_y$ 가 클수록, 공 속도가 느릴수록 곡선 효과가 강해진다.

---

## 10. 공-공 충돌 (Ball-Ball Collision)

### 10.1 충돌 감지

두 공의 중심 간 거리가 반지름의 합 이하일 때 충돌:

$$|\mathbf{r}_1 - \mathbf{r}_2| \leq R_1 + R_2$$

### 10.2 충돌 법선 방향

$$\hat{\mathbf{n}} = \frac{\mathbf{r}_1 - \mathbf{r}_2}{|\mathbf{r}_1 - \mathbf{r}_2|}$$

(공 2에서 공 1 방향, 공 1 입장의 "밖을 향하는" 법선)

### 10.3 겹침 보정

충돌 감지 시점에는 이미 공이 겹쳐 있을 수 있다. 두 공을 동등하게 밀어낸다:

$$\delta = (R_1 + R_2) - |\mathbf{r}_1 - \mathbf{r}_2|$$

$$\mathbf{r}_1 \mathrel{+}= \hat{\mathbf{n}} \cdot \frac{\delta}{2}, \qquad \mathbf{r}_2 \mathrel{-}= \hat{\mathbf{n}} \cdot \frac{\delta}{2}$$

### 10.4 법선 방향 충돌 임펄스

충돌 계수(restitution) $e$ 를 적용한 비탄성 충돌:

반발 조건: 법선 방향 상대속도 $v_{rel,n} = (\mathbf{v}_1 - \mathbf{v}_2) \cdot \hat{\mathbf{n}} < 0$ (서로 접근 중)일 때만 적용.

$$j = \frac{-(1 + e) \cdot v_{rel,n}}{\dfrac{1}{m_1} + \dfrac{1}{m_2}}$$

속도 업데이트:

$$\mathbf{v}_1' = \mathbf{v}_1 + \frac{j \hat{\mathbf{n}}}{m_1}, \qquad \mathbf{v}_2' = \mathbf{v}_2 - \frac{j \hat{\mathbf{n}}}{m_2}$$

동질량 ($m_1 = m_2 = m$), 완전탄성 ($e = 1$) 극한에서:

$$j = -m \cdot v_{rel,n}, \quad \mathbf{v}_1' \leftrightarrow \mathbf{v}_2' \text{ (법선 성분 교환)}$$

### 10.5 각속도 전달 (접선 마찰)

충돌 후 두 공의 접촉면에서 접선 방향 상대 속도가 있으면, 소량의 각운동량이 전달된다 (단순화 모델):

$$\text{접선 속도} = (\mathbf{v}_1' - \mathbf{v}_2') - [(\mathbf{v}_1' - \mathbf{v}_2') \cdot \hat{\mathbf{n}}]\hat{\mathbf{n}}$$

전달량 스케일: $\xi = 0.1 \cdot |\mathbf{v}_{tang}| / R$

토크 축: $\hat{\mathbf{k}} = \hat{\mathbf{n}} \times \hat{\mathbf{v}}_{tang}$

$$\Delta\boldsymbol{\omega}_1 = -\hat{\mathbf{k}} \cdot \xi, \qquad \Delta\boldsymbol{\omega}_2 = +\hat{\mathbf{k}} \cdot \xi$$

> **주의**: 이 항은 물리적으로 정확한 충돌 마찰 모델이 아니라 간략화된 근사이다. 완전한 모델에는 공-공 접촉점에서의 법선 임펄스와 마찰 임펄스를 연립하는 과정이 필요하다.

---

## 11. 쿠션 충돌 (Cushion Collision)

### 11.1 개요

쿠션 충돌은 두 단계로 처리한다:
1. **법선 방향**: 반발계수로 속도 반전
2. **접선 방향**: Coulomb 마찰 임펄스 → 속도와 스핀 동시 업데이트

### 11.2 접촉점과 접선 방향 미끄럼 속도

각 쿠션에서의 접촉점 위치 벡터 $\mathbf{r}_c$ (공 중심 기준):

| 쿠션 | 방향 | $\mathbf{r}_c$ | 접선 축 | 미끄럼 속도 $v_{slip}$ |
|---|---|---|---|---|
| 우쿠션 (x_max) | −X로 반발 | $(+R, 0, 0)$ | Z | $v_z - \omega_y R$ |
| 좌쿠션 (x_min) | +X로 반발 | $(−R, 0, 0)$ | Z | $v_z + \omega_y R$ |
| 앞쿠션 (z_max) | −Z로 반발 | $(0, 0, +R)$ | X | $v_x + \omega_y R$ |
| 뒤쿠션 (z_min) | +Z로 반발 | $(0, 0, −R)$ | X | $v_x - \omega_y R$ |

**우쿠션 미끄럼 속도 유도** ($\mathbf{r}_c = (R,0,0)$):

$$\boldsymbol{\omega} \times \mathbf{r}_c = (\omega_x, \omega_y, \omega_z) \times (R, 0, 0)$$

$$= \begin{vmatrix} \hat{i} & \hat{j} & \hat{k} \\ \omega_x & \omega_y & \omega_z \\ R & 0 & 0 \end{vmatrix} = (0, \; \omega_z R, \; -\omega_y R)$$

접촉점 속도의 Z 성분 (접선 방향):

$$v_{c,z} = v_z + (-\omega_y R) = v_z - \omega_y R$$

$\omega_y > 0$ (좌잉글리시)이고 $v_z \approx 0$ 이면 $v_{slip} < 0$ → 마찰이 공에 −Z 방향 힘을 가함 → 공이 아래쪽으로 꺾임.

### 11.3 법선 충돌 임펄스

반발계수 $e$ 적용:

$$J_n = (1 + e) \cdot |v_n| \cdot m$$

$$v_n' = -e \cdot v_n$$

### 11.4 접선 방향 Coulomb 마찰 임펄스

**유효 접촉 질량(Effective Contact Mass)**:

접선 임펄스 $J_f$ 가 선속도와 각속도를 동시에 변화시키므로, 유효 관성은 병렬 합산:

$$\frac{1}{m_{eff}} = \frac{1}{m} + \frac{R^2}{I}$$

균질 구에서 ($I = \frac{2}{5}mR^2$):

$$\frac{R^2}{I} = \frac{R^2}{\frac{2}{5}mR^2} = \frac{5}{2m} = \frac{2.5}{m}$$

$$\frac{1}{m_{eff}} = \frac{1}{m} + \frac{2.5}{m} = \frac{3.5}{m} \implies m_{eff} = \frac{m}{3.5}$$

**Coulomb 마찰 판정**:

미끄럼을 완전히 제거하는 데 필요한 임펄스 (점착, sticking):

$$J_{stop} = -v_{slip} \cdot m_{eff} = -v_{slip} \cdot \frac{m}{3.5}$$

Coulomb 마찰 한계:

$$J_{max} = \mu_{cushion} \cdot J_n$$

최종 마찰 임펄스:

$$J_f = \begin{cases} J_{stop} & \text{if } |J_{stop}| \leq J_{max} \quad \text{(sticking)} \\ -\text{sign}(v_{slip}) \cdot J_{max} & \text{if } |J_{stop}| > J_{max} \quad \text{(sliding)} \end{cases}$$

### 11.5 속도와 각속도 업데이트

우쿠션을 예로 ($J_f$는 Z방향, $\mathbf{r}_c = (R,0,0)$):

$$\Delta v_z = \frac{J_f}{m}$$

토크:

$$\boldsymbol{\tau} = \mathbf{r}_c \times \mathbf{F}_f = (R, 0, 0) \times (0, 0, J_f) = (0, -R J_f, 0)$$

$$\Delta \omega_y = \frac{\tau_y}{I} = \frac{-R J_f}{I}$$

**4개 쿠션의 업데이트 규칙 요약**:

| 쿠션 | $\Delta v_{tang}$ | $\Delta \omega_y$ |
|---|---|---|
| 우쿠션 (x_max) | $+J_f / m$ (Z방향) | $-R J_f / I$ |
| 좌쿠션 (x_min) | $+J_f / m$ (Z방향) | $+R J_f / I$ |
| 앞쿠션 (z_max) | $+J_f / m$ (X방향) | $+R J_f / I$ |
| 뒤쿠션 (z_min) | $+J_f / m$ (X방향) | $-R J_f / I$ |

### 11.6 잉글리시(English) 효과의 물리적 해석

우잉글리시 ($\omega_y < 0$)를 걸고 우쿠션(x_max)에 수직으로 입사하는 경우 ($v_x > 0$, $v_z \approx 0$):

$$v_{slip} = v_z - \omega_y R = 0 - \omega_y R = -\omega_y R > 0$$

($\omega_y < 0$ 이므로 $-\omega_y R > 0$)

$$J_f < 0 \quad \text{(접선 방향 Z로 음의 임펄스)}$$

$$\Delta v_z < 0 \quad \text{→ 공이 아래쪽(−Z)으로 꺾임}$$

즉, 우잉글리시를 걸면 우쿠션 반사 후 공이 아래쪽으로 꺾이는 것이 올바른 물리이며, 시뮬레이터가 이를 정확히 재현한다.

---

## 12. 수직 운동 — 점프샷 (Vertical Motion)

### 12.1 공중 비행 (자유 낙하)

큐를 아래로 향해 타격하면 공이 Y 방향 속도를 얻어 공중에 뜬다:

$$\frac{dv_y}{dt} = -g$$

$$v_y(t + \Delta t) = v_y(t) - g \Delta t, \qquad y(t + \Delta t) = y(t) + v_y(t) \Delta t$$

### 12.2 착지 반발

공이 바닥에 닿을 때 ($y < 0$):

$$v_y^{new} = \begin{cases} -e_{table} \cdot v_y & \text{if } |v_y| > 0.05 \ \text{m/s} \\ 0 & \text{otherwise (안착)} \end{cases}$$

$e_{table}$ = `TABLE_BOUNCE_REST` = 0.35 (당구 천의 바닥 반발 계수).

### 12.3 공중일 때의 처리 변경

공중에 있는 동안 (`position[1] > 0`):
- 바닥 마찰 적용 안 함 (공이 바닥에 닿지 않았으므로)
- 매세 곡선 적용 안 함
- 쿠션 충돌은 여전히 적용 (공이 옆 쿠션에 부딪힐 수 있으므로)

---

## 13. 수치 적분 방법 (Numerical Integration)

### 13.1 명시적 오일러법 (Explicit Euler Method)

이 시뮬레이터는 가장 단순한 수치 적분인 명시적 오일러를 사용한다:

$$\mathbf{v}(t + \Delta t) = \mathbf{v}(t) + \mathbf{a}(t) \cdot \Delta t$$

$$\mathbf{x}(t + \Delta t) = \mathbf{x}(t) + \mathbf{v}(t) \cdot \Delta t$$

**장점**: 구현이 단순, 계산 속도 빠름
**단점**: 오차가 $O(\Delta t)$ (1차 방법), 스텝이 크면 에너지가 증가하거나 불안정

### 13.2 서브스텝 (Substep)

정확도를 높이기 위해 프레임당 4번의 서브스텝 적분:

```
각 프레임 (≈ 1/60 s):
  for _ in range(4):          ← SIM_SUBSTEPS = 4
    engine.update(balls, dt)  ← dt = 0.001 s
```

물리 시뮬레이션 실제 시간 스텝: **Δt = 0.001 s = 1 ms**

### 13.3 충돌 처리 순서 (Operations Order)

매 스텝:
```
1. 마찰력 적용 (속도, 각속도 업데이트)
2. 위치 업데이트 (v * dt)
3. 수직 운동 적용 (중력, 착지)
4. 공-공 충돌 감지 및 해소
5. 쿠션 충돌 감지 및 해소
6. 정지 조건 검사
```

### 13.4 수치 안정성 주의사항

| 문제 | 원인 | 대처 |
|---|---|---|
| 슬라이딩↔구름 진동 | Δt 너무 큼 | 제로교차 감지 (5.3절) |
| 공이 쿠션 통과 | 고속에서 위치 업데이트 후 검사 | 위치 교정 + 속도 반전 |
| 두 공이 겹침 누적 | 매 스텝 약간의 관통 | 겹침 보정 (10.3절) |

---

## 14. 물리 파라미터 전체 목록

| 파라미터 | 변수명 | 기본값 | 단위 | 물리적 의미 |
|---|---|---|---|---|
| 슬라이딩 마찰 계수 | `MU_SLIDE` | 0.20 | 무차원 | 미끄럼 마찰 강도. 슬라이딩→구름 전환 속도 결정 |
| 구름 마찰 계수 | `MU_ROLL` | 0.02 | 무차원 | 구름 감속 강도. 공이 멈추는 거리 결정 |
| 수직 스핀 감쇠 계수 | `MU_SPIN` | 0.05 | 무차원 | 피벗 마찰. 잉글리시 지속 시간 결정 |
| 쿠션 반발 계수 | `CUSHION_RESTITUTION` | 0.80 | 무차원 | 쿠션 충돌 후 법선 속도 비율 |
| 공-공 반발 계수 | `BALL_RESTITUTION` | 0.95 | 무차원 | 두 공 충돌 후 상대속도 비율 |
| 중력 가속도 | `GRAVITY` | 9.8 | m/s² | 마찰력 크기와 점프 궤적에 영향 |
| 쿠션 마찰 계수 | `MU_CUSHION` | 0.20 | 무차원 | 쿠션 접선 Coulomb 마찰. 잉글리시 쿠션 효과 결정 |
| 바닥 반발 계수 | `TABLE_BOUNCE_REST` | 0.35 | 무차원 | 점프샷 착지 반발 계수 |

### 파라미터 민감도 분석

**`MU_SLIDE`** 를 두 배로 늘리면:
- 슬라이딩 구간이 절반으로 줄어들고 구름 전환이 빨라진다
- 백스핀/탑스핀의 효과 구간이 짧아진다

**`MU_CUSHION`** 을 0으로 설정하면:
- 쿠션 충돌에서 마찰이 전혀 없음 → 잉글리시 쿠션 효과 없음
- $\omega_y$ 변화 없이 법선 반사만 발생

**`CUSHION_RESTITUTION`** vs **`BALL_RESTITUTION`**:
- 쿠션은 고무+천 구조로 에너지 손실이 크다 (0.8)
- 당구공은 경질 상아/수지로 에너지 손실이 매우 작다 (0.95)

---

## 15. 단계별 시뮬레이션 루프 요약

```
┌──────────────────────────────────────────────────────────┐
│  매 서브스텝 (Δt = 0.001 s)                              │
│                                                          │
│  For each ball:                                          │
│    if NOT airborne:                                      │
│      [5절] 슬라이딩 or 구름 마찰 적용                    │
│        → 접촉점 속도 v_c = v + ω×r_c 계산               │
│        → F_fric = -μ_slide·m·g·v̂_c                     │
│        → v, ω 업데이트                                   │
│        → 제로교차 감지 → ROLLING 스냅                    │
│      [8절] ω_y 피벗 마찰 감쇠                            │
│      [9절] 매세 곡선 효과 (속도 방향 회전)               │
│                                                          │
│    위치 업데이트: x += v·Δt                              │
│    [12절] 수직 운동 (중력, 착지 반발)                    │
│                                                          │
│  For each pair (i, j):                                   │
│    [10절] 공-공 충돌 감지 및 임펄스 해소                  │
│                                                          │
│  For each ball:                                          │
│    [11절] 쿠션 충돌 감지                                  │
│      → J_n 계산 → 법선 반발                              │
│      → v_slip = v_tang ± ω_y·R                          │
│      → J_f (Coulomb) → v_tang, ω_y 업데이트             │
│                                                          │
│    정지 조건 검사: |v|, |ω| < 임계값 → STATIONARY       │
└──────────────────────────────────────────────────────────┘
```

---

## 부록: 주요 물리량 단위 확인

| 식 | 단위 확인 |
|---|---|
| $F = \mu \cdot m \cdot g$ | `[무차원]·[kg]·[m/s²] = [N]` ✓ |
| $\Delta v = F/m \cdot \Delta t$ | `[N]/[kg]·[s] = [m/s]` ✓ |
| $\tau = r \times F$ | `[m]·[N] = [N·m]` ✓ |
| $\Delta \omega = \tau/I \cdot \Delta t$ | `[N·m]/[kg·m²]·[s] = [rad/s]` ✓ |
| $J_n = (1+e) \cdot |v_n| \cdot m$ | `[m/s]·[kg] = [kg·m/s] = [N·s]` ✓ |
| $\Delta v = J/m$ | `[N·s]/[kg] = [m/s]` ✓ |
| $m_{eff} = m/3.5$ | `[kg]` ✓ |

---

*문서 작성: KBilliards Physics Engine 기반*
*최종 수정: 2026-02-19*
*대응 코드: `physics.py` (commit e5deaaf)*
