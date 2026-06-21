"""
Loop B — Selective Forwarding (Relay Meritocracy)
=================================================

When an emergency alert (e.g. "Hard Braking") propagates through the VANET, every
receiving vehicle must decide *whether* and *when* to relay it. Instead of a
random MAC backoff (which causes broadcast storms), Loop B biases the 802.11p
backoff timer using the vehicle's Transformer-derived **Stability Probability**
P_stable.

Flow
----
1. Vehicle receives alert.
2. Feeds its current context through the local Transformer → ``intent_logits``.
3. ``StabilityScorer.score(intent_logits)`` → ``P_stable``.
4. ``MACBiasMapper.map(P_stable)`` → ``T_wait_ms``.
5. OBU sets MAC backoff to ``T_wait_ms``.
   - Stable vehicle (P_stable ≈ 1.0) → ~1–5 ms → relays first.
   - Unstable vehicle (P_stable ≈ 0.2) → ~50–100 ms → hears first relay, cancels.

Intent Label Constants
----------------------
MAINTAIN = 0   (Maintain Lane)  ← treated as the "stable" class
TURN     = 1
EXIT     = 2
BRAKE    = 3   (emergency manoeuvre)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Intent class labels
# ──────────────────────────────────────────────────────────────────────────────

MAINTAIN = 0  # Stable: maintain current lane / cruise
TURN     = 1  # Moderate: planned turning manoeuvre
EXIT     = 2  # Moderate: leaving road / highway exit
BRAKE    = 3  # Unstable: emergency hard braking

INTENT_NAMES = {MAINTAIN: "MaintainLane", TURN: "Turn", EXIT: "Exit", BRAKE: "Brake"}

# Index of the "stable" class — used as P_stable
STABLE_CLASS_IDX = MAINTAIN


# ──────────────────────────────────────────────────────────────────────────────
# Data class for a single Loop-B decision
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class LoopBDecision:
    """Result of one StabilityScorer.score() call.

    Attributes
    ----------
    P_stable : float
        Softmax probability of the MAINTAIN_LANE intent class.
    predicted_intent : int
        Argmax intent class index.
    predicted_intent_name : str
        Human-readable intent label.
    T_wait_ms : float
        Recommended MAC backoff delay in milliseconds (raw, before clipping).
    """
    P_stable:              float
    predicted_intent:      int
    predicted_intent_name: str
    T_wait_ms:             float


# ──────────────────────────────────────────────────────────────────────────────
# StabilityScorer
# ──────────────────────────────────────────────────────────────────────────────

class StabilityScorer:
    """Convert Transformer intent logits → P_stable and raw T_wait.

    Parameters
    ----------
    constant : float
        MAC biasing constant C.  T_wait = C / P_stable (seconds).
    stable_class_idx : int
        Index of the "stable" intent class.  Default MAINTAIN (0).

    Examples
    --------
    >>> scorer = StabilityScorer(constant=0.1)
    >>> logits = torch.tensor([10.0, -1.0, -1.0, -1.0])  # strongly MAINTAIN
    >>> decision = scorer.score(logits)
    >>> decision.P_stable          # ≈ 0.999
    >>> decision.predicted_intent  # 0 (MAINTAIN)
    """

    def __init__(
        self,
        constant:         float = 0.1,
        stable_class_idx: int   = STABLE_CLASS_IDX,
    ) -> None:
        self.constant         = constant
        self.stable_class_idx = stable_class_idx

    # ------------------------------------------------------------------
    def score(self, intent_logits: torch.Tensor) -> LoopBDecision:
        """Compute P_stable and raw T_wait from raw intent logits.

        Parameters
        ----------
        intent_logits : torch.Tensor, shape (num_intents,) or (1, num_intents)
            Raw (unnormalised) logits from the Transformer's intent head.

        Returns
        -------
        LoopBDecision
            P_stable, predicted_intent, predicted_intent_name, T_wait_ms.
        """
        if intent_logits.dim() > 1:
            intent_logits = intent_logits.squeeze(0)

        probs = F.softmax(intent_logits.float(), dim=-1)
        P_stable         = float(probs[self.stable_class_idx].item())
        predicted_intent = int(torch.argmax(probs).item())
        T_wait_ms        = (self.constant / max(P_stable, 1e-3)) * 1000.0

        decision = LoopBDecision(
            P_stable              = P_stable,
            predicted_intent      = predicted_intent,
            predicted_intent_name = INTENT_NAMES.get(predicted_intent, "Unknown"),
            T_wait_ms             = T_wait_ms,
        )

        logger.debug(
            "LoopB | P_stable=%.4f | intent=%s | T_wait=%.1f ms",
            P_stable, decision.predicted_intent_name, T_wait_ms,
        )
        return decision

    # ------------------------------------------------------------------
    def score_batch(self, intent_logits: torch.Tensor) -> list[LoopBDecision]:
        """Score a full batch of logits.

        Parameters
        ----------
        intent_logits : torch.Tensor, shape (B, num_intents)

        Returns
        -------
        list[LoopBDecision], length B
        """
        return [self.score(intent_logits[i]) for i in range(len(intent_logits))]


# ──────────────────────────────────────────────────────────────────────────────
# MACBiasMapper
# ──────────────────────────────────────────────────────────────────────────────

class MACBiasMapper:
    """Clip and scale the raw T_wait into the configurable [min, max] ms window.

    Parameters
    ----------
    min_wait_ms : float
        Minimum backoff (ms). Fully stable vehicles get this. Default 1 ms.
    max_wait_ms : float
        Maximum backoff (ms). Highly unstable vehicles get this. Default 100 ms.
    constant : float
        Scaling constant C used in raw formula (C / P_stable * 1000).
    log_scale : bool
        If True, map P_stable through a log curve for smoother distribution.
        If False, use the raw C/P_stable clipped formula.

    Examples
    --------
    >>> mapper = MACBiasMapper(min_wait_ms=1, max_wait_ms=100)
    >>> mapper.map(P_stable=0.98)   # ≈ 1–5 ms
    >>> mapper.map(P_stable=0.20)   # ≈ 50–100 ms
    """

    def __init__(
        self,
        min_wait_ms: float = 1.0,
        max_wait_ms: float = 100.0,
        constant:    float = 0.1,
        log_scale:   bool  = True,
    ) -> None:
        self.min_wait_ms = min_wait_ms
        self.max_wait_ms = max_wait_ms
        self.constant    = constant
        self.log_scale   = log_scale

    def map(self, P_stable: float) -> float:
        """Map P_stable ∈ (0, 1] → T_wait_ms ∈ [min, max].

        Parameters
        ----------
        P_stable : float in (0, 1]
            Probability of the MAINTAIN_LANE intent class.

        Returns
        -------
        float
            Transmission wait time in milliseconds.
        """
        P_stable = max(float(P_stable), 1e-4)

        if self.log_scale:
            # Logarithmic: maps [0,1] → [max, min] smoothly
            # P=1 → min, P→0 → max
            t = self.min_wait_ms + (self.max_wait_ms - self.min_wait_ms) * (
                -math.log(P_stable) / math.log(1 / 1e-4)
            )
        else:
            # Raw C/P, then clip
            t = (self.constant / P_stable) * 1000.0

        return float(max(self.min_wait_ms, min(self.max_wait_ms, t)))

    def map_decision(self, decision: LoopBDecision) -> LoopBDecision:
        """Apply mapping to an existing LoopBDecision (updates T_wait_ms)."""
        from dataclasses import replace
        clipped = self.map(decision.P_stable)
        return replace(decision, T_wait_ms=clipped)
