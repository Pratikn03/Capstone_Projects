"""
Safety: Network Resilience Watchdog for Grid Islanding Detection.

This module implements a watchdog timer that monitors connectivity to the
grid control system. If heartbeats are not received within the timeout period,
the system assumes a network partition has occurred and enters "island mode."

Why Islanding Detection Matters:
    In real grid-connected battery systems, losing communication with the
    central control system is a critical safety event. The battery must:
    
    1. Detect the loss of communication quickly
    2. Switch to safe local-only operation (island mode)
    3. Avoid actions that could destabilize the grid
    4. Resume normal operation when connectivity returns

Watchdog Behavior:
    - beat(): Called by external system to indicate healthy connectivity
    - If no beat() within timeout_seconds: Trigger island mode
    - When beat() resumes: Automatically reconnect to grid control

Threading Model:
    The watchdog runs in a background daemon thread, checking for heartbeat
    timeout every 5 seconds. This allows the main API to continue serving
    requests while monitoring runs independently.

Usage:
    >>> from gridpulse.safety.watchdog import SystemWatchdog
    >>> watchdog = SystemWatchdog(timeout_seconds=60)
    >>> watchdog.start()
    >>> # In your API health endpoint:
    >>> watchdog.beat()  # Call regularly to prevent island mode

Configuration:
    Set timeout via configs/serving.yaml or WATCHDOG_TIMEOUT_SECONDS env var.
    Default is 60 seconds - adjust based on expected network latency.

See Also:
    - bms.py: Battery Management System safety layer
    - services/api/health.py: Health endpoint that calls watchdog.beat()
"""
import time
from datetime import datetime
import threading
import logging

# Dedicated logger for watchdog events - critical for incident investigation
logger = logging.getLogger("watchdog")


class SystemWatchdog:
    """
    Monitors system connectivity and triggers island mode on timeout.
    
    This class implements a classic watchdog timer pattern: if the beat()
    method is not called within timeout_seconds, the system assumes
    connectivity has been lost and enters a safe island mode.
    
    Attributes:
        timeout_seconds: Maximum time between heartbeats before islanding
        is_islanded: True if currently in island (disconnected) mode
        last_heartbeat: Timestamp of most recent beat() call
    """
    
    def __init__(self, timeout_seconds: int = 60):
        """
        Initialize the watchdog with a timeout period.
        
        Args:
            timeout_seconds: Time without heartbeat before triggering island mode.
                            Default is 60 seconds (1 minute).
        """
        self.last_heartbeat = datetime.now()
        self.timeout_seconds = timeout_seconds
        self.is_islanded = False
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)

    def start(self) -> None:
        """Start the background monitoring thread."""
        if self._thread.is_alive():
            logger.info("Watchdog already running.")
            return
        if self._stop_event.is_set():
            self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        logger.info("Watchdog started. System is GRID-CONNECTED.")

    def stop(self) -> None:
        """Stop the background monitoring thread."""
        self._stop_event.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1)

    def beat(self) -> None:
        """Call this API endpoint to reset the timer."""
        self.last_heartbeat = datetime.now()
        if self.is_islanded:
            logger.info("Heartbeat restored. Reconnecting to GRID control.")
            self.is_islanded = False

    def _monitor_loop(self) -> None:
        while not self._stop_event.is_set():
            delta = (datetime.now() - self.last_heartbeat).total_seconds()

            if delta > self.timeout_seconds and not self.is_islanded:
                self.trigger_island_mode()

            time.sleep(5)

    def trigger_island_mode(self) -> None:
        """
        CRITICAL: Network loss detected.
        Revert to safe local logic (Self-Consumption).
        """
        self.is_islanded = True
        logger.warning(
            "COMM_LOSS DETECTED (%ss timeout). ENTERING ISLAND MODE.",
            self.timeout_seconds,
        )
        # Logic to clear current optimization schedule would go here

    def get_status(self) -> str:
        return "ISLANDED" if self.is_islanded else "CONNECTED"
