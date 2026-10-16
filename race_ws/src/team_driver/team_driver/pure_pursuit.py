"""Pure pursuit path tracking on a closed loop of waypoints.

Coulter, "Implementation of the Pure Pursuit Path Tracking Algorithm",
CMU-RI-TR-92-01. Find the point on the path a lookahead distance ahead of the
car, then steer along the circular arc through it:

    delta = atan(2 * L * sin(alpha) / ld)

L is the wheelbase, alpha the angle to the lookahead point in the car's frame
and ld the distance to it. The path is a closed lap, so every index is taken
modulo its length and the lookahead runs straight across the finish line.
"""

import math

import numpy as np


def load_path(path_file):
    """Read an `x, y` CSV (comments start with #) into an (N, 2) array."""
    points = np.loadtxt(path_file, delimiter=',', comments='#', usecols=(0, 1),
                        ndmin=2)
    # A closed loop written with its first point repeated at the end would
    # give a zero-length segment at the seam.
    if len(points) > 1 and np.allclose(points[0], points[-1]):
        points = points[:-1]
    if len(points) < 3:
        raise ValueError(f'{path_file}: need at least 3 points, got {len(points)}')
    return points


class PurePursuit:

    def __init__(self, path, wheelbase, max_steer_rad,
                 lookahead_min, lookahead_gain, lookahead_max,
                 search_back=5, search_ahead=40, reacquire_dist=1.0):
        self.path = np.asarray(path, dtype=np.float64)
        self.n = len(self.path)
        self.wheelbase = wheelbase
        self.max_steer_rad = max_steer_rad
        self.lookahead_min = lookahead_min
        self.lookahead_gain = lookahead_gain
        self.lookahead_max = lookahead_max

        # Nearest-point search is local to where we were last cycle, so a
        # stretch of track that doubles back close by can never be picked by
        # mistake. If the car is far from that window - first call, or the
        # referee teleporting it back to the grid - search the whole lap.
        self.search_back = search_back
        self.search_ahead = search_ahead
        self.reacquire_dist = reacquire_dist
        self.nearest_idx = None

        # Segment i runs from point i to point i+1 (wrapping).
        self.seg_len = np.linalg.norm(np.roll(self.path, -1, axis=0) - self.path, axis=1)

        # Kept for visualisation.
        self.lookahead_point = None

    def lookahead_distance(self, speed):
        ld = self.lookahead_min + self.lookahead_gain * max(speed, 0.0)
        return min(max(ld, self.lookahead_min), self.lookahead_max)

    def find_nearest(self, x, y):
        pos = np.array([x, y])
        if self.nearest_idx is not None:
            idx = (self.nearest_idx + np.arange(-self.search_back, self.search_ahead + 1)) % self.n
            dist = np.linalg.norm(self.path[idx] - pos, axis=1)
            best = int(np.argmin(dist))
            if dist[best] < self.reacquire_dist:
                self.nearest_idx = int(idx[best])
                return self.nearest_idx
        dist = np.linalg.norm(self.path - pos, axis=1)
        self.nearest_idx = int(np.argmin(dist))
        return self.nearest_idx

    def find_lookahead(self, start_idx, ld):
        """Walk forward along the path from start_idx by arc length ld."""
        remaining = ld
        i = start_idx
        for _ in range(self.n):
            if self.seg_len[i] >= remaining:
                t = remaining / self.seg_len[i] if self.seg_len[i] > 0 else 0.0
                return self.path[i] + t * (self.path[(i + 1) % self.n] - self.path[i])
            remaining -= self.seg_len[i]
            i = (i + 1) % self.n
        return self.path[start_idx]

    def steer(self, x, y, yaw, speed):
        """Return the normalised steering command in [-1, 1]."""
        idx = self.find_nearest(x, y)
        ld = self.lookahead_distance(speed)
        target = self.find_lookahead(idx, ld)
        self.lookahead_point = target

        dx, dy = target[0] - x, target[1] - y
        alpha = math.atan2(dy, dx) - yaw
        alpha = math.atan2(math.sin(alpha), math.cos(alpha))
        dist = max(math.hypot(dx, dy), 1e-3)

        delta = math.atan2(2.0 * self.wheelbase * math.sin(alpha), dist)
        return float(np.clip(delta / self.max_steer_rad, -1.0, 1.0))
