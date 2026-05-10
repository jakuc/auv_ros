#!/usr/bin/env python3
"""
esc_driver.py — Prawdziwy sterownik ESC dla silników T200 (szkielet).

Na rzeczywistym robocie BlueROV2:
  - Odbiera komendy ciągów [N] z /auv/thruster_cmds
  - Konwertuje N → PWM [1100–1900 µs] przez krzywą osiągów T200
    (Blue Robotics T200-Public-Performance-Data-10-20V-September-2019)
  - Wysyła sygnał PWM do ESC przez:
      · Bezpośrednio: GPIO / UART (np. Raspberry Pi + PCA9685)
      · MAVLink: COMMAND_LONG / DO_MOTOR_TEST do Pixhawka

Interfejs ROS identyczny z sim_thruster_driver:
  sub: /auv/thruster_cmds  (std_msgs/Float64MultiArray, [N], 6 wartości)

Brak implementacji hardware — placeholder do uzupełnienia po podłączeniu ESC.
"""

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray


# Uproszczona liniowa aproksymacja krzywej T200 @ 16V:
#   ≈ 0 N przy PWM 1500 µs (stop)
#   ≈ +37 N przy PWM 1900 µs (pełna moc fwd)
#   ≈ -24 N przy PWM 1100 µs (pełna moc rev)
# Rzeczywista krzywa jest nieliniowa — użyj tabeli z pliku xlsx.
_PWM_STOP    = 1500   # µs
_PWM_FWD_MAX = 1900   # µs  →  +37 N
_PWM_REV_MAX = 1100   # µs  →  -24 N
_THRUST_FWD  = 37.0   # N
_THRUST_REV  = -24.0  # N


def thrust_to_pwm(thrust_n: float) -> int:
    """Przybliżona konwersja siła [N] → PWM [µs]."""
    if thrust_n >= 0:
        pwm = _PWM_STOP + (_PWM_FWD_MAX - _PWM_STOP) * (thrust_n / _THRUST_FWD)
    else:
        pwm = _PWM_STOP + (_PWM_STOP - _PWM_REV_MAX) * (thrust_n / _THRUST_REV)
    return int(np.clip(pwm, _PWM_REV_MAX, _PWM_FWD_MAX))


class EscDriverNode(Node):
    def __init__(self):
        super().__init__("esc_driver")
        self._sub = self.create_subscription(
            Float64MultiArray, "/auv/thruster_cmds", self._cb, 10)
        self.get_logger().info("EscDriver gotowy (placeholder — brak HW)")

    def _cb(self, msg: Float64MultiArray) -> None:
        thrusts = np.array(msg.data)
        pwm_values = [thrust_to_pwm(t) for t in thrusts]
        # TODO: wyślij pwm_values do ESC przez serial / MAVLink
        self.get_logger().debug(f"PWM: {pwm_values}")


def main(args=None):
    rclpy.init(args=args)
    node = EscDriverNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
