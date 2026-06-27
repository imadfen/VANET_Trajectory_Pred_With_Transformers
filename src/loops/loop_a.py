

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)



ENTROPY_LOW  = 0  # Flag value: prediction matches reality, reduce beacon rate
ENTROPY_HIGH = 1  # Flag value: unexpected manoeuvre, trigger alert rate




@dataclass
class LoopADecision:
    """Result of one DiscrepancyMonitor.check() call.

    Attributes
    ----------
    flag : int
        0 = Low Entropy (stable), 1 = High Entropy (alert).
    residual : float
        Mean L2 norm between predicted and actual windows.
    beacon_hz : float
        Recommended BSM beaconing frequency in Hz.
    """
    flag:       int
    residual:   float
    beacon_hz:  float




class DiscrepancyMonitor:
    """Compare the Transformer's predicted future against the actual sensor path.

    The monitor is stateless — call :meth:`check` on every inference tick.

    Parameters
    ----------
    epsilon : float
        Residual threshold.  Residual < epsilon → Low Entropy (Flag=0).
    beacon_hz_low : float
        Beaconing frequency (Hz) when Flag=0.  Default 2 Hz.
    beacon_hz_high : float
        Beaconing frequency (Hz) when Flag=1.  Default 10 Hz.

    Examples
    --------
    >>> mon = DiscrepancyMonitor(epsilon=0.5)
    >>> pred   = np.zeros((50, 23), dtype=np.float32)
    >>> actual = np.zeros((50, 23), dtype=np.float32) + 0.1
    >>> decision = mon.check(pred, actual)
    >>> decision.flag       # 0 — Low Entropy
    0
    >>> decision.beacon_hz  # 2.0 Hz
    2.0
    """

    def __init__(
        self,
        epsilon:        float = 0.5,
        beacon_hz_low:  float = 2.0,
        beacon_hz_high: float = 10.0,
    ) -> None:
        self.epsilon        = epsilon
        self.beacon_hz_low  = beacon_hz_low
        self.beacon_hz_high = beacon_hz_high


    def check(
        self,
        predicted_window: np.ndarray,
        actual_window:    np.ndarray,
    ) -> LoopADecision:
        """Compare predicted vs actual trajectory windows.

        Parameters
        ----------
        predicted_window : np.ndarray, shape (seq_len, feat_dim)
            Future trajectory predicted by the Transformer at the *previous* tick.
        actual_window : np.ndarray, shape (seq_len, feat_dim)
            Actual sensor readings over the same window.

        Returns
        -------
        LoopADecision
            Contains flag (0/1), residual scalar, and recommended beacon_hz.

        Notes
        -----
        Residual is computed as the **mean L2 norm** across the sequence.
        After global standardization all features live in roughly [-3, 3],
        so a threshold of ε=0.5 is roughly half a standard deviation of error.
        """
        if predicted_window.shape != actual_window.shape:
            raise ValueError(
                f"Shape mismatch: predicted {predicted_window.shape} "
                f"vs actual {actual_window.shape}"
            )

        diff     = predicted_window.astype(np.float64) - actual_window.astype(np.float64)
        # Mean L2 norm over time steps (not normalised by feat_dim so ε is
        # directly interpretable in the standardised feature space)
        residual = float(np.mean(np.linalg.norm(diff, axis=-1)))

        if residual < self.epsilon:
            flag      = ENTROPY_LOW
            beacon_hz = self.beacon_hz_low
        else:
            flag      = ENTROPY_HIGH
            beacon_hz = self.beacon_hz_high

        logger.debug(
            "LoopA | residual=%.4f | ε=%.4f | flag=%d | beacon=%.1f Hz",
            residual, self.epsilon, flag, beacon_hz,
        )
        return LoopADecision(flag=flag, residual=residual, beacon_hz=beacon_hz)


    def check_batch(
        self,
        predicted_batch: np.ndarray,
        actual_batch:    np.ndarray,
    ) -> list[LoopADecision]:
        """Vectorised version for evaluating a full validation batch offline.

        Parameters
        ----------
        predicted_batch : np.ndarray, shape (B, seq_len, feat_dim)
        actual_batch    : np.ndarray, shape (B, seq_len, feat_dim)

        Returns
        -------
        list[LoopADecision], length B
        """
        return [
            self.check(predicted_batch[i], actual_batch[i])
            for i in range(len(predicted_batch))
        ]




class HallucinationEngine:
    """Generate hallucinated trajectories to fill gaps during ego suppression.

    When the ego vehicle is in Low Entropy (2 Hz) mode, receiving neighbours
    have up to 500 ms (= 1/2 Hz) without a fresh beacon. This engine lets
    a neighbour autoregressively project the last known ego state forward using
    its *own* local Transformer, so its situational awareness stays current.

    Parameters
    ----------
    encoder : TSTransformerEncoder
        A loaded (and normalised) Transformer encoder instance.
    feat_dim : int
        Number of features (default 23 for the VANET feature set).

    Notes
    -----
    This is currently a **stub**. Full deployment requires:
    - A live sensor interface to feed `context_window` in real time.
    - Inverse-normalisation so predicted values are back in metric space.
    - Integration with the OBU's networking stack for timing.
    """

    def __init__(self, encoder, feat_dim: int = 23) -> None:
        self.encoder  = encoder
        self.feat_dim = feat_dim

    def hallucinate(
        self,
        context_window: np.ndarray,
        n_future_steps: int = 5,
    ) -> np.ndarray:
        """Autoregressively predict `n_future_steps` beyond the context.

        Parameters
        ----------
        context_window : np.ndarray, shape (seq_len, feat_dim)
            Historical (normalised) trajectory of the ego vehicle.
        n_future_steps : int
            How many 100 ms steps to project forward.

        Returns
        -------
        hallucinated : np.ndarray, shape (n_future_steps, feat_dim)
            Predicted continuation of the ego trajectory.
        """
        import torch

        self.encoder.eval()
        buffer = context_window.copy()  # (seq_len, feat_dim)

        predictions = []
        with torch.no_grad():
            for _ in range(n_future_steps):
                x = torch.from_numpy(buffer).unsqueeze(0).float()        # (1, T, F)
                mask = torch.ones(1, buffer.shape[0], dtype=torch.bool)  # (1, T)
                recon, *_ = self.encoder(x, mask)
                next_step = recon[0, -1, :].cpu().numpy()  # last time-step prediction
                predictions.append(next_step)
                # Slide buffer forward
                buffer = np.vstack([buffer[1:], next_step])

        return np.stack(predictions, axis=0)  # (n_future_steps, feat_dim)
