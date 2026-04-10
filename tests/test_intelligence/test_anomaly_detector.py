"""Tests for anomaly detection."""

import polars as pl

from src.intelligence.anomaly_detector import detect_anomalies
from src.engine.models import SeverityLevel


class TestAnomalyDetector:
    def test_detects_z_score_outliers(self):
        values = [10, 11, 12, 10, 11, 13, 12, 10, 11, 1000]
        df = pl.DataFrame({"value": values})
        report = detect_anomalies(df, dataset_name="test", z_threshold=3.0)
        assert len(report.anomalies) > 0
        assert any(a.value == "1000" for a in report.anomalies)

    def test_detects_negative_in_positive_column(self):
        df = pl.DataFrame({"price": [10.0, 20.0, -5.0, 30.0]})
        report = detect_anomalies(df, dataset_name="test")
        assert any(a.value == "-5.0" for a in report.anomalies)

    def test_no_anomalies_in_clean_data(self):
        df = pl.DataFrame({"value": list(range(100))})
        report = detect_anomalies(df, dataset_name="test")
        # Clean sequential data should have few/no anomalies
        critical = [a for a in report.anomalies if a.severity == SeverityLevel.CRITICAL]
        assert len(critical) == 0

    def test_report_counts(self):
        df = pl.DataFrame({"price": [10, 20, -5, 30, -10, 99999]})
        report = detect_anomalies(df, dataset_name="test")
        assert report.total_rows == 6
        assert report.critical_count + report.warning_count + report.info_count == len(report.anomalies)

    def test_empty_dataframe(self):
        df = pl.DataFrame({"a": []}).cast({"a": pl.Float64})
        report = detect_anomalies(df, dataset_name="test")
        assert len(report.anomalies) == 0
