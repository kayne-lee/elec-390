#!/usr/bin/env python3
from robot_hat import PWM
import threading
import time

# Create PWM objects for channels P4 and P5.
pwm_p4 = PWM('P4')
pwm_p5 = PWM('P5')

# Configure both PWM channels.
pwm_p4.freq(50)
pwm_p5.freq(50)

pwm_p4.prescaler(1)
pwm_p5.prescaler(1)

pwm_p4.period(100)
pwm_p5.period(100)

# Create events to control blinking.
exit_event = threading.Event()
start_event = threading.Event()

def flash_led(pwm, interval):
    """Toggles the LED based on the start event signal."""
    while not exit_event.is_set():
        start_event.wait()  # Wait for start signal
        state = False
        while start_event.is_set() and not exit_event.is_set():
            state = not state
            pwm.pulse_width_percent(100 if state else 0)
            time.sleep(interval)
        pwm.pulse_width_percent(0)  # Ensure LED turns off when stopped

# Create threads for LEDs
thread_p4 = threading.Thread(target=flash_led, args=(pwm_p4, 0.25))
thread_p5 = threading.Thread(target=flash_led, args=(pwm_p5, 0.5))

# Start threads
thread_p4.start()
thread_p5.start()

def start_blinking():
    """Start blinking LEDs."""
    print("Starting LED blinking...")
    start_event.set()

def stop_blinking():
    """Stop blinking LEDs."""
    print("Stopping LED blinking...")
    start_event.clear()

def shutdown():
    """Gracefully stop all threads and turn off LEDs."""
    print("Shutting down LED controller...")
    stop_blinking()
    exit_event.set()
    thread_p4.join()
    thread_p5.join()
    pwm_p4.pulse_width_percent(0)
    pwm_p5.pulse_width_percent(0)
    print("Both LEDs are turned off.")

# Prevent script from exiting immediately when imported
if __name__ == "__main__":
    print("LED Controller is running. Call start_blinking(), stop_blinking(), and shutdown() from another script.")
