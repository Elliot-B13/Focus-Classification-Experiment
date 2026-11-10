"""
Connection test for NEUROSPEC MMBT-S + DSI-7.

Set the MMBT-S switch to SIMPLE mode.
Start a short recording in DSI-Streamer before sending markers.

This tests connectivity, not stimulus timing.
Do not paste this blocking pulse code into the task's display loop.
"""

import time
import serial
from serial.tools import list_ports

print("\nAvailable serial ports:\n")

ports = list(list_ports.comports())

for port in ports:
    print(f"{port.device}: {port.description}")

if not ports:
    raise SystemExit(
        "\nNo serial ports found. Check the USB connection "
        "and the MMBT-S driver."
    )

port_name = input(
    "\nEnter the MMBT-S port, for example COM4: "
).strip().upper()

if port_name not in {port.device.upper() for port in ports}:
    raise SystemExit("That port was not in the detected list.")

# The MMBT-S uses a virtual serial port at 9600 baud.
with serial.Serial(
    port=port_name,
    baudrate=9600,
    bytesize=serial.EIGHTBITS,
    parity=serial.PARITY_NONE,
    stopbits=serial.STOPBITS_ONE,
    timeout=1,
    write_timeout=1,
) as box:

    try:
        # Begin with all trigger lines low.
        box.write(bytes([0]))
        box.flush()
        time.sleep(0.1)

        input(
            "\nStart recording in DSI-Streamer, "
            "then press ENTER here."
        )

        # Repeated 1s verify that separate identical events register.
        codes = [1, 2, 4, 8, 15, 1, 1]

        for code in codes:
            box.write(bytes([code]))
            box.flush()

            time.sleep(0.050)  # Hold high for at least 50 ms.

            box.write(bytes([0]))
            box.flush()

            print(f"Sent code {code}")
            time.sleep(1.0)

        print("\nTest finished.")
        print("Expected nonzero event sequence:", codes)
        print("Stop and save the DSI-Streamer recording.")

    finally:
        # Reset the output even if the test is interrupted.
        box.write(bytes([0]))
        box.flush()