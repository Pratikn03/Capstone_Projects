"""
Monitoring: Alert Notification Helpers.

This module provides utilities for sending alerts when monitoring thresholds
are exceeded. Alerts can be sent to various destinations:

- Slack/Discord webhooks (JSON POST)
- PagerDuty (incident creation)
- Email (via SMTP, not implemented here)
- Custom webhook endpoints

Alert Triggers:
    - Model drift detected (KS test p-value below threshold)
    - Data quality degradation (missing values spike)
    - Performance degradation (RMSE exceeds baseline)
    - System health issues (API latency spikes)

Retry Logic:
    Alerts are critical for operational awareness. The send_webhook function
    implements exponential backoff retries to handle transient network issues.

Usage:
    >>> from gridpulse.monitoring.alerts import send_webhook
    >>> send_webhook(
    ...     url="https://hooks.slack.com/services/...",
    ...     payload={"text": "Model drift detected on load_mw!"}
    ... )

Configuration:
    Webhook URLs should be stored securely (environment variables or secrets
    manager), not hardcoded. See configs/monitoring.yaml for alert thresholds.
"""
from __future__ import annotations

import logging
from typing import Mapping

from gridpulse.utils.net import get_session


def send_webhook(
    url: str, 
    payload: Mapping[str, object], 
    retries: int = 3, 
    backoff: float = 0.5
) -> None:
    """
    Send a JSON payload to a webhook endpoint with automatic retries.
    
    This function is designed for alerting purposes where delivery is
    important but not latency-critical. Failed attempts are logged
    and retried with exponential backoff.
    
    Args:
        url: Webhook URL (Slack, Discord, custom endpoint)
        payload: Dictionary to be JSON-encoded and POSTed
        retries: Number of retry attempts (default: 3)
        backoff: Initial backoff delay in seconds (doubles each retry)
        
    Raises:
        requests.HTTPError: If all retry attempts fail
        
    Example:
        >>> send_webhook(
        ...     "https://hooks.slack.com/services/XXX",
        ...     {"text": ":warning: Model drift detected!"}
        ... )
    """
    log = logging.getLogger(__name__)
    
    # Get a session with retry configuration
    session = get_session(retries=retries, backoff=backoff)
    
    try:
        resp = session.post(url, json=payload, timeout=15)
        resp.raise_for_status()
        log.info("Alert webhook sent successfully to %s", url)
    except Exception as exc:
        log.error("Alert webhook failed for %s: %s", url, exc)
        raise  # Re-raise so caller knows alert failed
