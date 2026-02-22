# 3D 4구 당구 게임 명세서 v1.0 (Advanced Physics)

## 1. 프로젝트 개요 및 목표
- 목표: Python과 Ursina 엔진을 사용하여 리얼한 물리 법칙이 적용된 4구 당구 게임 개발.
- 핵심 요구사항:
  1. 외부 물리 엔진 미사용 (Custom Physics Engine 구현).
  2. **회전(Spin) 역학 구현**: 밀어치기, 끌어치기, 찍어치기(Masse) 구현 필수.
  3. **바닥 마찰 구현**: Sliding(미끄러짐)과 Rolling(구름) 상태의 전이 구현.
  4. **기술 시현**: 4구의 핵심 기술(우라, 하코, 짱꼴라 등)이 물리 법칙에 의해 자연스럽게 발생해야 함.

## 2. 데이터 구조 및 물리 상수

### 2.1. 상수 (Constants)
- `BALL_RADIUS`: 0.033 (m)
- `BALL_MASS`: 0.21 (kg)
- `MU_SLIDE`: 0.2 (미끄러짐 마찰 계수 - 초반 스피드)
- `MU_ROLL`: 0.02 (구름 마찰 계수 - 자연스러운 감속)
- `MU_SPIN`: 0.05 (제자리 회전 감속 계수)
- `GRAVITY`: 9.8
- `I` (관성 모멘트): (2/5) * BALL_MASS * BALL_RADIUS^2 (구의 관성 모멘트 공식)

### 2.2. Ball Class 확장
- `position`: Vector3 [x, y, z]
- `velocity`: Vector3 [vx, vy, vz] (선속도)
- **`angular_velocity`**: Vector3 [wx, wy, wz] (각속도 - 회전량)
  - x, z축 회전: 전진/후진 회전 (밀어치기/끌어치기 관장)
  - y축 회전: 좌우 회전 (히네리/English 관장)
- `state`: Enum (SLIDING, ROLLING, STATIONARY)

## 3. 핵심 물리 알고리즘 (Custom Physics)

### 3.1. 큐 타격 (Cue Impact) - 기술의 시작
큐가 공의 특정 지점을 쳤을 때 선속도와 각속도를 동시에 생성한다.
- 입력: 타격 힘(`F`), 타격점 오프셋(`offset` - 공 중심으로부터의 거리 [x, y])
- **공식:**
  - 선속도 변화: `delta_v = F / m`
  - 각속도 변화: `delta_w = (offset X F) / I` (외적 사용)
  - *예: 공의 하단을 치면(offset.y < 0), 역회전(Backspin) 각속도가 생성됨 -> 끌어치기 준비 완료.*

### 3.2. 바닥 마찰 및 이동 (Floor Interaction) - 가장 중요
매 프레임 공이 미끄러지는지 구르는지 판단하여 속도를 조절한다.
1. **접점 속도(Contact Velocity) 계산:**
   - 바닥과 닿는 지점의 공 표면 속도 `vc = v + (w X r)`
2. **상태 판별:**
   - `vc`가 0이 아니면 **SLIDING** 상태 (미끄러지는 중).
   - `vc`가 0에 근접하면 **ROLLING** 상태 (순수하게 구르는 중).
3. **마찰력 적용:**
   - **SLIDING 상태:** 운동 방향 반대로 `MU_SLIDE` 만큼 힘이 작용. 동시에 바닥 마찰 토크가 발생하여 `angular_velocity`를 변화시킴.
     - *핵심 효과:* 역회전(Backspin)이 걸린 공은 바닥 마찰에 의해 선속도가 점점 줄어들다가, 어느 순간 뒤로 가속하거나 멈춤 (Stop Shot/Draw Shot 구현 원리).
   - **ROLLING 상태:** `MU_ROLL` 만큼 서서히 감속.

### 3.3. 찍어치기 (Masse) 효과 근사 구현
- 큐를 세워서 쳤을 때(수직에 가까운 타격), 공에 수직 축 회전뿐만 아니라 바닥을 찍어누르는 힘에 의한 궤적 휘어짐 발생.
- 구현: `angular_velocity`의 회전축이 수직이 아닐 때, 이동 경로(Velocity 방향)를 회전 방향으로 매 프레임 조금씩 회전시키는 `Curving Force` 추가.

### 3.4. 쿠션 충돌과 회전 (Cushion & Spin)
- 입사각과 반사각 계산 시 회전량(English) 반영.
- 공식 보정: `반사각 = 입사각 + (k * y축_회전량)`
- *효과:* 회전을 많이 주고 쿠션을 맞추면 각도가 커지거나 작아짐 (접시, 짱꼴라 구현 원리).

## 4. 데모 및 검증 시나리오 (Demo Scenarios)
키보드 입력(1~5번)으로 다음 샷을 자동으로 세팅하고 발사하여 물리를 검증한다.

- **Scenario 1: 밀어치기 (Follow Shot)**
  - 세팅: 적구와 일직선 배치.
  - 타격: 수구 상단 타격.
  - 기대 결과: 수구가 적구를 맞춘 후, 멈추지 않고 적구를 따라 앞으로 전진함.
  
- **Scenario 2: 끌어치기 (Draw Shot)**
  - 세팅: 적구와 일직선 배치.
  - 타격: 수구 하단 타격 (강하게).
  - 기대 결과: 수구가 적구를 맞춘 후, 뒤로 되돌아옴.

- **Scenario 3: 죽여치기 (Stop Shot)**
  - 세팅: 적구와 일직선.
  - 타격: 수구 중심 하단 타격 (적당한 힘).
  - 기대 결과: 수구가 적구를 맞춘 직후 그 자리에 멈춤 (운동량 전이 완료).

- **Scenario 4: 1쿠션 빈쿠션 치기 (Bank Shot)**
  - 세팅: 수구 -> 쿠션 -> 적구 경로.
  - 타격: 무회전 입사 vs 회전 입사 비교.
  - 기대 결과: 회전 입사 시 반사각이 달라져야 함.

- **Scenario 5: 3쿠션 대회전 (Nejire)**
  - 세팅: 강한 타격으로 테이블을 크게 도는 경로.
  - 기대 결과: 마찰에 의해 속도가 줄어들지만, 쿠션 반발력으로 계속 돌아야 함.

## 5. 구현 단계 (Phase)
1. **Phase 1: Advanced Physics Class**: `Angular Velocity`, `Friction Model` 구현.
2. **Phase 2: Visual Debugging**: 공의 회전 방향을 볼 수 있게 공 텍스처(줄무늬 등) 적용.
3. **Phase 3: Shot Preset System**: 위 시나리오 1~5를 실행하는 함수 구현.