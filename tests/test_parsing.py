from src.evaluation.parsing_accuracy import grouping_accuracy
from src.parsing.drain import LogParser
from src.parsing.log_formats import is_keyword_alert, parse_bgl_line, parse_hdfs_line

HDFS_LINE = (
    "081109 203615 148 INFO dfs.DataNode$PacketResponder: Received block "
    "blk_38865049064139660 of size 67108864 from /10.251.42.84\n"
)
BGL_LINE = (
    "- 1117838570 2005.06.03 R02-M1-N0-C:J12-U11 2005-06-03-15.42.50.675872 "
    "R02-M1-N0-C:J12-U11 RAS KERNEL INFO instruction cache parity error corrected\n"
)
BGL_ALERT = (
    "KERNDTLB 1118536327 2005.06.11 R30-M0-N9-C:J16-U01 2005-06-11-17.32.07.581048 "
    "R30-M0-N9-C:J16-U01 RAS KERNEL FATAL data TLB error interrupt\n"
)


def test_parse_hdfs_line():
    rec = parse_hdfs_line(HDFS_LINE)
    assert rec.level == "INFO"
    assert rec.component == "dfs.DataNode$PacketResponder"
    assert rec.content.startswith("Received block blk_38865049064139660")
    assert rec.timestamp == 1226262975  # 2008-11-09 20:36:15 UTC


def test_parse_bgl_line_labels():
    normal, alert = parse_bgl_line(BGL_LINE), parse_bgl_line(BGL_ALERT)
    assert normal.label == 0 and alert.label == 1
    assert normal.timestamp == 1117838570
    assert alert.level == "FATAL"
    assert alert.content == "data TLB error interrupt"


def test_malformed_lines_are_rejected():
    assert parse_hdfs_line("garbage\n") is None
    assert parse_bgl_line("too short\n") is None


def test_keyword_rule():
    assert is_keyword_alert("FATAL", "anything")
    assert is_keyword_alert("INFO", "Exception in receiveBlock")
    assert not is_keyword_alert("INFO", "Received block blk_1 of size 5")


def test_drain_groups_lines_differing_only_in_variables():
    parser = LogParser()
    a = parser.parse("Received block blk_1 of size 67108864 from /10.251.42.84")
    b = parser.parse("Received block blk_-99 of size 3 from /10.0.0.1")
    c = parser.parse("Deleting block blk_1 file /mnt/hadoop/dfs/data/current/subdir/blk_1")
    assert a == b != c
    template = parser.templates()[a][0]
    assert "<BLK>" in template and "<IP>" in template
    assert parser.match("Received block blk_7 of size 1 from /1.2.3.4") == a
    assert parser.match("Something never seen before") is None


def test_parser_pickle_round_trip_keeps_templates():
    import pickle

    parser = LogParser(sim_th=0.5)
    a = parser.parse("Received block blk_1 of size 5 from /1.2.3.4")
    clone = pickle.loads(pickle.dumps(parser))
    assert clone._settings["sim_th"] == 0.5
    assert clone.templates() == parser.templates()
    assert clone.parse("Received block blk_2 of size 9 from /5.6.7.8") == a


def test_grouping_accuracy():
    assert grouping_accuracy(["A", "A", "B"], [1, 1, 2]) == 1.0
    # Predicted cluster 1 merges A and B: none of those lines are correct.
    assert grouping_accuracy(["A", "A", "B", "C"], [1, 1, 1, 2]) == 0.25
