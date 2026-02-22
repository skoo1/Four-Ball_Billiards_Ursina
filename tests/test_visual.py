"""
Visual setup tests — Phase 2
Ursina 렌더링 없이 setup_only(run=False) 결과를 검증한다.
"""

import numpy as np
import pytest
from physics import TABLE_WIDTH, TABLE_LENGTH, BALL_RADIUS
from shot_presets import ShotPreset


HW = TABLE_WIDTH / 2
HL = TABLE_LENGTH / 2


class TestSetupOnly:
    """run=False로 호출 시 공 배치 + apply_cue만 수행되는지 검증."""

    SCENARIOS = [
        ShotPreset.scenario_1_follow,
        ShotPreset.scenario_2_draw,
        ShotPreset.scenario_3_stop,
        lambda run: ShotPreset.scenario_4_bank(english=0.0, run=run),
        ShotPreset.scenario_5_nejire,
    ]

    @pytest.mark.parametrize("scenario_fn", SCENARIOS)
    def test_returns_four_balls(self, scenario_fn):
        result = scenario_fn(run=False)
        assert "balls" in result
        assert len(result["balls"]) == 4

    @pytest.mark.parametrize("scenario_fn", SCENARIOS)
    def test_elapsed_is_zero(self, scenario_fn):
        result = scenario_fn(run=False)
        assert result["elapsed"] == 0.0

    @pytest.mark.parametrize("scenario_fn", SCENARIOS)
    def test_cue_has_velocity(self, scenario_fn):
        """run=False에서 apply_cue는 수행되므로 수구에 속도가 있어야 한다."""
        result = scenario_fn(run=False)
        cue = result["white"]
        assert np.linalg.norm(cue.velocity) > 0

    @pytest.mark.parametrize("scenario_fn", SCENARIOS)
    def test_all_balls_within_table(self, scenario_fn):
        result = scenario_fn(run=False)
        for ball in result["balls"]:
            x, z = ball.position[0], ball.position[2]
            assert -HW - BALL_RADIUS <= x <= HW + BALL_RADIUS, \
                f"{ball.name} x={x} out of table"
            assert -HL - BALL_RADIUS <= z <= HL + BALL_RADIUS, \
                f"{ball.name} z={z} out of table"

    @pytest.mark.parametrize("scenario_fn", SCENARIOS)
    def test_ball_names_unique(self, scenario_fn):
        result = scenario_fn(run=False)
        names = [b.name for b in result["balls"]]
        assert len(names) == len(set(names))

    @pytest.mark.parametrize("scenario_fn", SCENARIOS)
    def test_engine_returned(self, scenario_fn):
        result = scenario_fn(run=False)
        assert result["engine"] is not None


class TestBackwardCompatibility:
    """run=True (default) 호출이 기존과 동일하게 동작하는지 검증."""

    def test_scenario_1_default_runs(self):
        result = ShotPreset.scenario_1_follow()
        assert result["elapsed"] > 0

    def test_scenario_2_default_runs(self):
        result = ShotPreset.scenario_2_draw()
        assert result["elapsed"] > 0

    def test_scenario_3_default_runs(self):
        result = ShotPreset.scenario_3_stop()
        assert result["elapsed"] > 0
