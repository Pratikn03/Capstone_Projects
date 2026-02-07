"""Network resilience watchdog for islanding behavior."""
import time
from datetime import datetime
import threading
import logging

logger = logging.getLogger("watchdog")


class SystemWatchdog:
    def __init__(self, timeout_seconds: int = 60):
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
