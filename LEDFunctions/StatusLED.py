#!/usr/bin/env python3
from robot_hat import PWM, Pin
import time

# Setup
led = PWM('P8')           # PWM8 - LED pin
led.freq(50)
led.prescaler(1)
led.period(100)

button = Pin('D0')        # Button pin
button.mode(Pin.IN)
button.pull(Pin.PULL_UP)  # Assumes button connects to GND when pressed

led_state = False  # Track whether LED is ON or OFF

print("Press the button to toggle the LED.")

try:
    while True:
        if button.value() == 0:  # Button pressed (active low)
            led_state = not led_state
            led.pulse_width_percent(100 if led_state else 0)
            print(f"LED {'ON' if led_state else 'OFF'}")
            # Debounce delay
            time.sleep(0.3)

        time.sleep(0.01)  # Short delay to prevent CPU overuse

except KeyboardInterrupt:
    print("Shutting down...")
    led.pulse_width_percent(0)
