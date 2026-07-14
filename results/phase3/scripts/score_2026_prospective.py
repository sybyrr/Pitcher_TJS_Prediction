"""CLI entry point for immutable frozen-state prospective scoring.

This replaces the 2026-07-13 delayed-shadow script that reproduced the model
by fitting on the historical cohort. Existing 2025/2026 archives are preserved;
all new snapshots load ``frozen_model_state.json`` and are append-only.
"""

from __future__ import annotations

from prospective_scoring import main


if __name__ == "__main__":
    main()
