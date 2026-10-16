"""Turn a target speed into a normalised throttle.

AutoDRIVE takes torque, not speed (docs/04-algorithms.md section 4.1), so:

    throttle = k_ff * v_target  +  kp * error  +  ki * integral(error)

The feed-forward term carries most of the load: the throttle needed to hold a
speed is roughly proportional to it. The PI trim cleans up what k_ff gets
wrong.

Two things deliberately limit the output:

- Anti-windup. The integral term's contribution is clamped to
  +-integral_limit (in throttle units), and the integral stops growing while
  the output is saturated in the direction it would push.
- No braking for small overspeeds. A negative throttle locks the wheels and a
  locked wheel does not steer, so being a little fast just lifts off
  (throttle_min, normally 0.0). Only when the car is more than
  overspeed_margin too fast is brake_throttle allowed.
"""


class SpeedController:

    def __init__(self, k_ff, kp, ki, integral_limit,
                 throttle_min, throttle_max,
                 overspeed_margin, brake_throttle):
        self.k_ff = k_ff
        self.kp = kp
        self.ki = ki
        self.integral_limit = integral_limit
        self.throttle_min = throttle_min
        self.throttle_max = throttle_max
        self.overspeed_margin = overspeed_margin
        self.brake_throttle = brake_throttle
        self.integral = 0.0

    def reset(self):
        self.integral = 0.0

    def update(self, v_target, v_measured, dt):
        error = v_target - v_measured

        badly_fast = -error > self.overspeed_margin
        lower = self.brake_throttle if badly_fast else self.throttle_min

        integral = self.integral + error * dt
        if self.ki > 0.0:
            limit = self.integral_limit / self.ki
            integral = max(-limit, min(limit, integral))

        raw = self.k_ff * v_target + self.kp * error + self.ki * integral
        throttle = max(lower, min(self.throttle_max, raw))

        # Conditional integration: keep the new integral only if it is not
        # pushing further into a limit we are already clamped against.
        saturated_high = raw > self.throttle_max and error > 0.0
        saturated_low = raw < lower and error < 0.0
        if not (saturated_high or saturated_low):
            self.integral = integral

        return throttle
