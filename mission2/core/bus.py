"""
Event bus for module communication.

Provides pub/sub pattern for loose coupling between modules.
Thread-safe and supports both sync and async event handling.
"""

import queue
import threading
from collections import defaultdict
from collections.abc import Callable

from .events import Event, EventType


class EventBus:
    """
    Simple pub/sub event bus for module communication.

    Thread-safe, supports both sync and async handlers.

    Usage:
        bus = EventBus()
        bus.subscribe(EventType.MOVE_MADE, my_handler)
        bus.publish(Event(type=EventType.MOVE_MADE, data=move))
    """

    def __init__(self) -> None:
        self._handlers: dict[EventType, list[Callable[[Event], None]]] = defaultdict(
            list
        )
        self._queue: queue.Queue[Event | None] = queue.Queue()
        self._lock = threading.Lock()
        self._running = False
        self._worker_thread: threading.Thread | None = None
        self._event_log: list[Event] = []
        self._log_enabled = True
        self._max_log_size = 100

    def subscribe(
        self, event_type: EventType, handler: Callable[[Event], None]
    ) -> None:
        """Register a handler for an event type."""
        with self._lock:
            if handler not in self._handlers[event_type]:
                self._handlers[event_type].append(handler)

    def unsubscribe(
        self, event_type: EventType, handler: Callable[[Event], None]
    ) -> None:
        """Remove a handler."""
        with self._lock:
            if handler in self._handlers[event_type]:
                self._handlers[event_type].remove(handler)

    def publish(self, event: Event) -> None:
        """Publish event asynchronously (queued for processing)."""
        if self._log_enabled:
            self._log_event(event)

        if self._running:
            self._queue.put(event)
        else:
            # If not running async, dispatch synchronously
            self._dispatch(event)

    def publish_sync(self, event: Event) -> None:
        """Publish event synchronously (immediate processing)."""
        if self._log_enabled:
            self._log_event(event)
        self._dispatch(event)

    def _dispatch(self, event: Event) -> None:
        """Dispatch event to all registered handlers."""
        with self._lock:
            handlers = self._handlers[event.type].copy()

        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                print(f"[EventBus] Handler error for {event.type}: {e}")

    def _log_event(self, event: Event) -> None:
        """Add event to log."""
        self._event_log.append(event)
        if len(self._event_log) > self._max_log_size:
            self._event_log.pop(0)

    def get_event_log(self, limit: int = 20) -> list[Event]:
        """Get recent events from log."""
        return self._event_log[-limit:]

    def clear_log(self) -> None:
        """Clear event log."""
        self._event_log.clear()

    def start(self) -> None:
        """Start async event processing."""
        if self._running:
            return

        self._running = True
        self._worker_thread = threading.Thread(target=self._process_loop, daemon=True)
        self._worker_thread.start()

    def stop(self) -> None:
        """Stop event processing."""
        if not self._running:
            return

        self._running = False
        self._queue.put(None)  # Sentinel to unblock
        if self._worker_thread:
            self._worker_thread.join(timeout=1.0)
            self._worker_thread = None

    def _process_loop(self) -> None:
        """Background event processing loop."""
        while self._running:
            try:
                event = self._queue.get(timeout=0.1)
                if event is None:
                    break
                self._dispatch(event)
            except queue.Empty:
                continue

    @property
    def is_running(self) -> bool:
        """Check if async processing is active."""
        return self._running


# ─────────────────────────────────────────────────────────────
# SINGLETON ACCESS
# ─────────────────────────────────────────────────────────────

_bus: EventBus | None = None


def get_event_bus() -> EventBus:
    """Get the global event bus singleton."""
    global _bus
    if _bus is None:
        _bus = EventBus()
    return _bus


def reset_event_bus() -> None:
    """Reset the event bus (for testing)."""
    global _bus
    if _bus is not None:
        _bus.stop()
    _bus = None
