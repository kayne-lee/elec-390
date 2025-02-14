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

# Create an event to signal the threads to stop.
exit_event = threading.Event()

def flash_led(pwm, interval, exit_event):
    """
    Toggles the LED connected to the given PWM channel.
    Every 'interval' seconds, the duty cycle is toggled between 0% and 100%.
    The loop will exit when exit_event is set.
    """
    state = False  # Start with LED off.
    while not exit_event.is_set():
        state = not state  # Toggle state.
        if state:
            pwm.pulse_width_percent(100)  # Turn LED on.
        else:
            pwm.pulse_width_percent(0)    # Turn LED off.
        time.sleep(interval)

# Create threads for each LED flash routine.
thread_p4 = threading.Thread(target=flash_led, args=(pwm_p4, 0.25, exit_event))
thread_p5 = threading.Thread(target=flash_led, args=(pwm_p5, 0.5, exit_event))

# Start the flashing threads.
thread_p4.start()
thread_p5.start()

print("LED on P4 is flashing every 0.25 seconds and LED on P5 every 0.5 seconds.")
print("Press Ctrl+C to exit.")

try:
    # Keep the main thread alive while the LED threads run.
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\nKeyboardInterrupt received. Exiting gracefully...")
    exit_event.set()  # Signal the threads to exit.
    thread_p4.join()  # Wait for the threads to finish.
    thread_p5.join()
finally:
    # Ensure both LEDs are turned off.
    pwm_p4.pulse_width_percent(0)
    pwm_p5.pulse_width_percent(0)
    print("Both LEDs have been turned off.")