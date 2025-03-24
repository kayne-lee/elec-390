#!/usr/bin/env python3
import time

# Import your signal and brake light control functions.
# Adjust the import paths if the modules are in separate files.
from BlinkingLEDs import signal_left, signal_right, stop_signaling, shutdown as signal_shutdown
from BrakeLEDs import toggle_brake_lights, shutdown as brake_shutdown

def test_vehicle_lighting():
    print("🚗 Starting vehicle lighting test...")

    print("\n🔁 Testing LEFT turn signal for 5 seconds:")
    signal_left()
    time.sleep(5)
    stop_signaling()

    time.sleep(1)

    print("\n🛑 Testing BRAKE lights toggle ON:")
    toggle_brake_lights()
    time.sleep(2)

    print("🛑 Testing BRAKE lights toggle OFF:")
    toggle_brake_lights()
    time.sleep(1)

    print("\n🔁 Testing RIGHT turn signal for 5 seconds:")
    signal_right()
    time.sleep(5)
    stop_signaling()

    time.sleep(1)

    print("\n🧹 Shutting down systems...")
    signal_shutdown()
    brake_shutdown()

    print("✅ Test complete.")

if __name__ == "__main__":
    test_vehicle_lighting()
