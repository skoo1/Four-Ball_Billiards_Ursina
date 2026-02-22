"""
Tests for Shot Preset System (Phase 3)
각 시나리오의 물리적 기대 결과를 검증한다.
"""

import pytest
from shot_presets import ShotPreset


class TestScenario1Follow:
    """밀어치기: 수구가 충돌점 너머로 전진해야 한다."""

    def test_cue_advances_past_collision_point(self):
        result = ShotPreset.scenario_1_follow()

        # 충돌 전 수구 출발점 z=-0.8, 적구 위치 z=-0.3
        # 밀어치기이므로 수구 peak_z 가 충돌점(z=-0.3)보다 전방이어야 함
        collision_z = -0.3
        assert result["peak_z_white"] > collision_z, (
            f"Follow shot: 수구 peak_z={result['peak_z_white']:.4f}가 충돌점 z={collision_z}보다 전방이어야 함"
        )

    def test_simulation_completes(self):
        result = ShotPreset.scenario_1_follow()
        assert result["elapsed"] < 10.0


class TestScenario2Draw:
    """끌어치기: 수구가 출발점 뒤로 복귀해야 한다."""

    def test_cue_returns_behind_start(self):
        result = ShotPreset.scenario_2_draw()
        cue = result["white"]

        start_z = -0.5
        assert cue.position[2] < start_z, (
            f"Draw shot: 수구 z={cue.position[2]:.4f}가 출발점 z={start_z}보다 뒤에 있어야 함"
        )

    def test_simulation_completes(self):
        result = ShotPreset.scenario_2_draw()
        assert result["elapsed"] < 10.0


class TestScenario3Stop:
    """죽여치기: 수구가 충돌점 근처에서 정지해야 한다."""

    def test_cue_stops_near_collision_point(self):
        result = ShotPreset.scenario_3_stop()
        cue = result["white"]

        # 적구 초기 위치(z=0.075) 근처에서 수구 정지
        collision_z = 0.075
        distance = abs(cue.position[2] - collision_z)
        assert distance < 0.15, (
            f"Stop shot: 수구 z={cue.position[2]:.4f}가 충돌점 z={collision_z} 근처(±0.15m)여야 함, "
            f"실제 거리={distance:.4f}"
        )

    def test_simulation_completes(self):
        result = ShotPreset.scenario_3_stop()
        assert result["elapsed"] < 10.0


class TestScenario4Bank:
    """빈쿠션: 회전 유무에 따른 반사 후 도달 위치 차이."""

    def test_bank_shot_reaches_far_side(self):
        result = ShotPreset.scenario_4_bank(english=0.0)
        cue = result["white"]
        assert cue.cushion_hits >= 1, "뱅크샷 수구는 최소 1회 쿠션 히트해야 함"

    def test_english_changes_final_position(self):
        result_no_english = ShotPreset.scenario_4_bank(english=0.0)
        result_with_english = ShotPreset.scenario_4_bank(english=0.015)

        pos_no = result_no_english["white"].position.copy()
        pos_en = result_with_english["white"].position.copy()

        diff = float(max(abs(pos_no[0] - pos_en[0]), abs(pos_no[2] - pos_en[2])))
        assert diff > 0.01, (
            f"English 유무에 따라 최종 위치가 달라야 함: 차이={diff:.4f}"
        )

    def test_simulation_completes(self):
        result = ShotPreset.scenario_4_bank()
        assert result["elapsed"] < 10.0


class TestScenario5Nejire:
    """3쿠션 대회전: 수구가 3회 이상 쿠션에 닿아야 한다."""

    def test_three_or_more_cushion_hits(self):
        result = ShotPreset.scenario_5_nejire()
        cue = result["white"]
        assert cue.cushion_hits >= 3, (
            f"3쿠션 대회전: 쿠션 히트 {cue.cushion_hits}회 — 최소 3회 필요"
        )

    def test_simulation_completes(self):
        result = ShotPreset.scenario_5_nejire()
        assert result["elapsed"] < 10.0
