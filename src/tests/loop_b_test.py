import torch
from src.loops.loop_b import StabilityScorer, MACBiasMapper

scorer = StabilityScorer()
mapper = MACBiasMapper(min_wait_ms=1.0, max_wait_ms=100.0)

# 1. The Transformer outputs intent logits: [Maintain, Turn, Exit, Brake]
# Let's pretend the car is braking hard:
intent_logits = torch.tensor([10.0, -5.0, -5.0, -5.0]) 

# 2. Get the Stability Score
scored_state = scorer.score(intent_logits)
print(f"Probability of being stable: {scored_state.P_stable}") # Practically 0.0

# 3. Get the MAC timer!
timer_ms = mapper.map(scored_state.P_stable)
print(f"You must wait {timer_ms} ms before relaying!") # Outputs 100.0 ms