import numpy as np
from src.loops.loop_a import DiscrepancyMonitor

# 1. Create the monitor
monitor = DiscrepancyMonitor(epsilon=0.5)

# 2. Fake some data: predictions vs reality.
# Say the array is [50 time steps, 23 features]
prediction = np.zeros((50, 23)) 
reality    = np.zeros((50, 23)) + 1.0  # Reality was vastly different!

# 3. Ask Loop A what to do!
decision = monitor.check(prediction, reality)

print(f"Flag: {decision.flag}")                 # Outputs: 1 (High Entropy)
print(f"Broadcast rate: {decision.beacon_hz}")  # Outputs: 10.0 (Hz)