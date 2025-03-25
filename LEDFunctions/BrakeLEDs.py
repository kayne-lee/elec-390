"""How To Use

toggle_brake_lights()  # Turns ON both lights
toggle_brake_lights()  # Turns them OFF"""


#!/usr/bin/env python3
from robot_hat import PWM
import time

# Setup PWM for both brake lights
pwm_brake_left = PWM('P6')
pwm_brake_right = PWM('P7')

# Configure both PWM channels
for pwm in [pwm_brake_left, pwm_brake_right]:
    pwm.freq(50)
    pwm.prescaler(1)
    pwm.period(100)

# Track brake light state
brake_on = False

def toggle_brake_lights():
    """Toggle brake lights on/off with each call."""
    global brake_on
    brake_on = not brake_on
    brightness = 100 if brake_on else 0

    pwm_brake_left.pulse_width_percent(brightness)
    pwm_brake_right.pulse_width_percent(brightness)

    print(f"Brake lights {'ON' if brake_on else 'OFF'}")

def shutdown():
    """Turn off brake lights and clean up."""
    pwm_brake_left.pulse_width_percent(0)
    pwm_brake_right.pulse_width_percent(0)
    print("Brake lights turned off and system shut down.")

if __name__ == "__main__":
    print("Brake Light Controller is running.")
    print("Use toggle_brake_lights() to turn brake lights on/off.")
    print("Call shutdown() to turn everything off cleanly.")
