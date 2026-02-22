"""
Controller Tests — Headless simulation determinism and RL API verification.

강화 3: 강화학습 환경에서 동일한 입력 → 비트 수준 동일한 출력 보장.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from controller import BilliardsController


class TestDeterminism:
    """강화학습 환경의 결정론(determinism) 검증."""

    def test_headless_determinism_basic(self):
        """동일한 초기 조건 + 동일한 커맨드 → 비트 수준 동일한 결과."""
        ctrl1 = BilliardsController()
        ctrl1.reset()
        ctrl2 = BilliardsController()
        ctrl2.reset()

        res1 = ctrl1.simulate_shot("white", azimuth=30, tip=[0.3, 0.5], power=80)
        res2 = ctrl2.simulate_shot("white", azimuth=30, tip=[0.3, 0.5], power=80)

        np.testing.assert_array_equal(
            res1["obs"], res2["obs"],
            err_msg="Two identical simulations must produce bit-identical obs vectors"
        )
        assert res1["scored"] == res2["scored"]
        assert res1["cushion_hits"] == res2["cushion_hits"]
        assert res1["sim_time"] == res2["sim_time"]
        assert res1["reward"] == res2["reward"]
        assert res1["balls"] == res2["balls"]

    def test_determinism_with_side_spin(self):
        """사이드 스핀 포함 시에도 결정론 보장."""
        ctrl1 = BilliardsController()
        ctrl1.reset()
        ctrl2 = BilliardsController()
        ctrl2.reset()

        # 강한 사이드 스핀 + 대각선 방향
        res1 = ctrl1.simulate_shot("white", azimuth=45, tip=[0.8, -0.3], power=60)
        res2 = ctrl2.simulate_shot("white", azimuth=45, tip=[0.8, -0.3], power=60)

        np.testing.assert_array_equal(res1["obs"], res2["obs"])
        assert res1["balls"] == res2["balls"]

    def test_determinism_across_multiple_shots(self):
        """연속 2회 시뮬레이션에서도 결정론 유지."""
        ctrl1 = BilliardsController()
        ctrl1.reset()
        ctrl2 = BilliardsController()
        ctrl2.reset()

        # Shot 1
        r1a = ctrl1.simulate_shot("white", azimuth=0, power=50)
        r2a = ctrl2.simulate_shot("white", azimuth=0, power=50)
        np.testing.assert_array_equal(r1a["obs"], r2a["obs"])

        # Shot 2 (다른 파라미터)
        r1b = ctrl1.simulate_shot("yellow", azimuth=180, tip=[0.0, 0.8], power=70)
        r2b = ctrl2.simulate_shot("yellow", azimuth=180, tip=[0.0, 0.8], power=70)
        np.testing.assert_array_equal(r1b["obs"], r2b["obs"])
