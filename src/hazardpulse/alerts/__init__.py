"""HazardPulse alerting subsystem (rule engine + sinks)."""
from hazardpulse.alerts.manager import (
    AlertManager,
    AlertRule,
    Alert,
    AlertSink,
    FileSink,
    SlackSink,
    WebhookSink,
    critical_eq_rule,
    severe_to_rule,
    hu_ri_rule,
)

__all__ = [
    "AlertManager",
    "AlertRule",
    "Alert",
    "AlertSink",
    "FileSink",
    "SlackSink",
    "WebhookSink",
    "critical_eq_rule",
    "severe_to_rule",
    "hu_ri_rule",
]
