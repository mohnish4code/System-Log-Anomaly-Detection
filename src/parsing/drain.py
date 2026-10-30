"""Thin wrapper around drain3 with masking rules suited to system logs."""
from drain3 import TemplateMiner
from drain3.masking import MaskingInstruction
from drain3.template_miner_config import TemplateMinerConfig

# Order matters: specific patterns must run before the generic number mask.
MASKS = [
    (r"blk_-?\d+", "BLK"),
    (r"(?:\d{1,3}\.){3}\d{1,3}(?::\d+)?", "IP"),
    (r"0x[0-9a-fA-F]+", "HEX"),
    (r"((?<=[^A-Za-z0-9])|^)([\-\+]?\d+)((?=[^A-Za-z0-9])|$)", "NUM"),
]


class LogParser:
    """Online Drain parser: each call assigns a line to a stable integer event ID."""

    def __init__(self, sim_th: float = 0.4, depth: int = 4, max_children: int = 100):
        config = TemplateMinerConfig()
        config.drain_sim_th = sim_th
        config.drain_depth = depth
        config.drain_max_children = max_children
        config.masking_instructions = [MaskingInstruction(p, name) for p, name in MASKS]
        config.profiling_enabled = False
        self.miner = TemplateMiner(config=config)

    def parse(self, content: str) -> int:
        # Skips TemplateMiner.add_log_message, which builds a result dict
        # (including the full template string) on every call.
        cluster, _ = self.miner.drain.add_log_message(self.miner.masker.mask(content))
        return cluster.cluster_id

    def match(self, content: str) -> int | None:
        """Look up a line without creating or updating templates (inference-time use)."""
        cluster = self.miner.match(content)
        return None if cluster is None else cluster.cluster_id

    def templates(self) -> dict[int, tuple[str, int]]:
        return {c.cluster_id: (c.get_template(), c.size) for c in self.miner.drain.clusters}
