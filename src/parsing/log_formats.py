"""Line-level parsers that split a raw log line into header fields and free-text content."""
import calendar
import re
from dataclasses import dataclass
from functools import lru_cache

BLOCK_RE = re.compile(r"blk_-?\d+")

# Rule baseline: a line "looks bad" if its level or its text says so.
ALERT_LEVELS = frozenset({"ERROR", "FATAL", "SEVERE", "FAILURE"})
KEYWORD_RE = re.compile(r"\b(?:error|fatal|fail(?:ed|ure|s)?|exception)\b", re.IGNORECASE)


@dataclass(slots=True)
class LogRecord:
    timestamp: int  # unix seconds
    level: str
    component: str
    content: str
    label: int = 0  # per-line ground truth when the dataset provides it (BGL)
    node: str = ""


def is_keyword_alert(level: str, content: str) -> bool:
    return level in ALERT_LEVELS or KEYWORD_RE.search(content) is not None


@lru_cache(maxsize=4096)
def _hdfs_epoch(date: str, time: str) -> int:
    # Dates are yymmdd / hhmmss; consecutive lines repeat them, hence the cache.
    return calendar.timegm((2000 + int(date[:2]), int(date[2:4]), int(date[4:6]),
                            int(time[:2]), int(time[2:4]), int(time[4:6]), 0, 0, 0))


_HDFS_RE = re.compile(r"^(\d{6}) (\d{6}) \d+ (\w+) ([^:]+): (.*)$")


def parse_hdfs_line(line: str) -> LogRecord | None:
    """081109 203615 148 INFO dfs.DataNode$PacketResponder: Received block blk_... of size ..."""
    m = _HDFS_RE.match(line.rstrip("\r\n"))
    if m is None:
        return None
    date, time, level, component, content = m.groups()
    return LogRecord(_hdfs_epoch(date, time), level, component, content)


def parse_bgl_line(line: str) -> LogRecord | None:
    """- 1117838570 2005.06.03 R02-M1-N0-C:J12-U11 2005-06-03-15.42.50.675872 R02-M1-N0-C:J12-U11 RAS KERNEL INFO msg

    The first field is "-" for normal lines and an alert category otherwise.
    """
    parts = line.rstrip("\r\n").split(" ", 9)
    if len(parts) < 9 or not parts[1].isdigit():
        return None
    content = parts[9] if len(parts) == 10 else ""
    return LogRecord(
        timestamp=int(parts[1]),
        level=parts[8],
        component=parts[7],
        content=content,
        label=int(parts[0] != "-"),
        node=parts[3],
    )


LINE_PARSERS = {"hdfs": parse_hdfs_line, "bgl": parse_bgl_line}
