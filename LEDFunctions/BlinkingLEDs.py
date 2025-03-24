"""How to Use It
In your Python shell or script, once this is running:

signal_left()      # Left LED blinks
signal_right()     # Right LED blinks
stop_signaling()   # Turn off both
shutdown()         # Cleanup everything"""


#!/usr/bin/env python3
from robot_hat import PWM
import threading
import time

# Setup PWM
pwm_left = PWM('P4')   # Left signal
pwm_right = PWM('P5')  # Right signal

# PWM Configuration
for pwm in [pwm_left, pwm_right]:
    pwm.freq(50)
    pwm.prescaler(1)
    pwm.period(100)

# Threading events
exit_event = threading.Event()
signal_event = threading.Event()
signal_direction = {'left': False, 'right': False}  # Track which LED is blinking

def flash_led(pwm, direction):
    """Flashes the specified LED when its signal is active."""
    while not exit_event.is_set():
        signal_event.wait()  # Wait until signaling is active
        state = False
        while signal_event.is_set() and signal_direction[direction] and not exit_event.is_set():
            state = not state
            pwm.pulse_width_percent(100 if state else 0)
            time.sleep(0.25)
        pwm.pulse_width_percent(0)  # Ensure LED turns off when stopped

# Start threads for both LEDs, but they'll only flash when activated
left_thread = threading.Thread(target=flash_led, args=(pwm_left, 'left'))
right_thread = threading.Thread(target=flash_led, args=(pwm_right, 'right'))

left_thread.start()
right_thread.start()

# Control Functions
def signal_left():
    """Activate left turn signal."""
    print("Left signal ON")
    signal_direction['left'] = True
    signal_direction['right'] = False
    signal_event.set()

def signal_right():
    """Activate right turn signal."""
    print("Right signal ON")
    signal_direction['left'] = False
    signal_direction['right'] = True
    signal_event.set()

def stop_signaling():
    """Turn off all signals."""
    print("Signals OFF")
    signal_event.clear()
    signal_direction['left'] = False
    signal_direction['right'] = False
    pwm_left.pulse_width_percent(0)
    pwm_right.pulse_width_percent(0)

def shutdown():
    """Gracefully shuts down everything."""
    print("Shutting down turn signal controller...")
    stop_signaling()
    exit_event.set()
    left_thread.join()
    right_thread.join()
    print("System shut down successfully.")

if __name__ == "__main__":
    print("Turn Signal Controller is running.")
    print("Use signal_left(), signal_right(), stop_signaling(), and shutdown() to control the LEDs.")
