"""Drive one lap of the centreline offline, without the simulator.

A kinematic bicycle model stands in for the car and a first-order lag stands
in for the motor, so this only checks the geometry and the wiring - that pure
pursuit follows the path all the way round, through the wrap at the end of the
CSV, and that the speed controller settles on the target. It says nothing
about grip or about how throttle really maps to speed in AutoDRIVE.

    python3 -m pytest race_ws/src/team_driver/test/test_offline_lap.py -s
"""

import math
import os

import numpy as np

from team_driver.pure_pursuit import PurePursuit, load_path
from team_driver.speed_controller import SpeedController

HERE = os.path.dirname(os.path.abspath(__file__))
PATH_FILE = os.path.join(HERE, '..', '..', '..', '..', 'maps',
                         'icra26_compete_centerline.csv')


def test_one_lap():
    path = load_path(PATH_FILE)
    pursuit = PurePursuit(path, wheelbase=0.324, max_steer_rad=0.5236,
                          lookahead_min=0.6, lookahead_gain=0.15, lookahead_max=1.5)
    speed_ctl = SpeedController(k_ff=0.05, kp=0.10, ki=0.05, integral_limit=0.10,
                                throttle_min=0.0, throttle_max=0.30,
                                overspeed_margin=1.0, brake_throttle=-0.10)

    # Spawn from maps/README.md: (0.80, 3.16) facing -y.
    x, y, yaw, v = 0.80, 3.16, -math.pi / 2, 0.0
    dt, v_target = 0.025, 1.5
    gain = 1.0 / 0.05          # pretend steady-state speed = throttle / k_ff
    travelled, worst = 0.0, 0.0
    lap_length = float(np.sum(pursuit.seg_len))

    for _ in range(int(120 / dt)):
        steering = pursuit.steer(x, y, yaw, v)
        throttle = speed_ctl.update(v_target, v, dt)
        v += (gain * throttle - v) * dt / 0.5      # 0.5 s motor lag
        delta = steering * 0.5236
        x += v * math.cos(yaw) * dt
        y += v * math.sin(yaw) * dt
        yaw += v / 0.324 * math.tan(delta) * dt
        travelled += v * dt
        worst = max(worst, float(np.min(np.linalg.norm(path - [x, y], axis=1))))
        if travelled > lap_length * 1.05:
            break

    print(f'\nlap {lap_length:.1f} m, travelled {travelled:.1f} m, '
          f'final speed {v:.2f} m/s, worst distance from path {worst:.3f} m')
    assert travelled > lap_length, 'did not complete a lap in 120 s'
    assert worst < 0.30, 'strayed more than 0.30 m from the centreline'
    assert abs(v - v_target) < 0.1, 'speed did not settle on the target'
