"""Webhook dispatcher for rebuild events."""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError


class WebhookEvent(str, Enum):
    """Webhook event types."""

    REBUILD_STARTED = "rebuild.started"
    REBUILD_COMPLETED = "rebuild.completed"
    REBUILD_FAILED = "rebuild.failed"
    REBUILD_CANCELLED = "rebuild.cancelled"
    FILE_CHANGES_DETECTED = "file_changes.detected"
    INDEX_UPDATED = "index.updated"


@dataclass
class WebhookConfig:
    """Configuration for a webhook endpoint."""

    url: str
    events: list[WebhookEvent] = field(default_factory=list)
    secret: Optional[str] = None
    headers: dict = field(default_factory=dict)
    timeout: float = 10.0
    retries: int = 3
    retry_delay: float = 1.0
    enabled: bool = True

    def to_dict(self) -> dict:
        return {
            "url": self.url,
            "events": [e.value for e in self.events],
            "secret": "***" if self.secret else None,
            "headers": {k: v for k, v in self.headers.items() if k.lower() != "authorization"},
            "timeout": self.timeout,
            "retries": self.retries,
            "retry_delay": self.retry_delay,
            "enabled": self.enabled,
        }


@dataclass
class WebhookDelivery:
    """Record of a webhook delivery attempt."""

    event: WebhookEvent
    url: str
    payload: dict
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    success: bool = False
    status_code: Optional[int] = None
    error: Optional[str] = None
    attempt: int = 1

    def to_dict(self) -> dict:
        return {
            "event": self.event.value,
            "url": self.url,
            "timestamp": self.timestamp.isoformat(),
            "success": self.success,
            "status_code": self.status_code,
            "error": self.error,
            "attempt": self.attempt,
        }


class WebhookDispatcher:
    """Dispatches webhook events to configured endpoints."""

    def __init__(self, max_history: int = 1000):
        """Initialize webhook dispatcher.

        Args:
            max_history: Maximum delivery history to keep
        """
        self._webhooks: dict[str, WebhookConfig] = {}
        self._delivery_history: deque[WebhookDelivery] = deque(maxlen=max_history)
        self._lock = threading.RLock()

    def register(self, name: str, config: WebhookConfig) -> None:
        """Register a webhook endpoint.

        Args:
            name: Unique name for this webhook
            config: Webhook configuration
        """
        with self._lock:
            self._webhooks[name] = config

    def unregister(self, name: str) -> bool:
        """Unregister a webhook endpoint."""
        with self._lock:
            if name in self._webhooks:
                del self._webhooks[name]
                return True
            return False

    def get_webhook(self, name: str) -> Optional[WebhookConfig]:
        """Get webhook configuration."""
        with self._lock:
            return self._webhooks.get(name)

    def list_webhooks(self) -> list[dict]:
        """List all registered webhooks."""
        with self._lock:
            return [{"name": name, **config.to_dict()} for name, config in self._webhooks.items()]

    def dispatch(self, event: WebhookEvent, payload: dict) -> int:
        """Dispatch event to all matching webhooks.

        Args:
            event: Event type
            payload: Event payload

        Returns:
            Number of webhooks notified
        """
        with self._lock:
            matching = [(name, config) for name, config in self._webhooks.items()
                       if config.enabled and (not config.events or event in config.events)]

        if not matching:
            return 0

        count = 0
        for name, config in matching:
            try:
                self._deliver(name, config, event, payload)
                count += 1
            except Exception:
                # Delivery failures are recorded in history
                pass

        return count

    def _deliver(self, name: str, config: WebhookConfig, event: WebhookEvent, payload: dict) -> None:
        """Deliver webhook to a single endpoint with retries."""
        full_payload = {
            "event": event.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": payload,
        }

        headers = {
            "Content-Type": "application/json",
            "User-Agent": "TRACE-Webhook/1.0",
            **config.headers,
        }

        if config.secret:
            # Add HMAC signature for verification
            import hmac
            import hashlib
            body = json.dumps(full_payload).encode()
            signature = hmac.new(
                config.secret.encode(),
                body,
                hashlib.sha256
            ).hexdigest()
            headers["X-TRACE-Signature"] = f"sha256={signature}"

        last_error = None
        for attempt in range(1, config.retries + 1):
            try:
                req = Request(
                    config.url,
                    data=json.dumps(full_payload).encode(),
                    headers=headers,
                    method="POST",
                )

                with urlopen(req, timeout=config.timeout) as response:
                    status = response.getcode()
                    if 200 <= status < 300:
                        self._record_delivery(WebhookDelivery(
                            event=event,
                            url=config.url,
                            payload=full_payload,
                            success=True,
                            status_code=status,
                            attempt=attempt,
                        ))
                        return
                    else:
                        last_error = f"HTTP {status}"
            except HTTPError as e:
                last_error = f"HTTP {e.code}: {e.reason}"
            except URLError as e:
                last_error = f"URL Error: {e.reason}"
            except Exception as e:
                last_error = f"{type(e).__name__}: {e}"

            # Record failed attempt
            self._record_delivery(WebhookDelivery(
                event=event,
                url=config.url,
                payload=full_payload,
                success=False,
                error=last_error,
                attempt=attempt,
            ))

            # Wait before retry
            if attempt < config.retries:
                time.sleep(config.retry_delay * attempt)

        # All retries exhausted - final failure recorded above

    def _record_delivery(self, delivery: WebhookDelivery) -> None:
        """Record delivery attempt."""
        with self._lock:
            self._delivery_history.append(delivery)

    def get_delivery_history(
        self,
        event: Optional[WebhookEvent] = None,
        limit: int = 100,
    ) -> list[dict]:
        """Get delivery history."""
        with self._lock:
            history = list(self._delivery_history)
            if event:
                history = [d for d in history if d.event == event]
            history.sort(key=lambda d: d.timestamp, reverse=True)
            return [d.to_dict() for d in history[:limit]]

    def get_stats(self) -> dict:
        """Get dispatcher statistics."""
        with self._lock:
            total = len(self._delivery_history)
            successful = sum(1 for d in self._delivery_history if d.success)
            failed = total - successful
            return {
                "registered_webhooks": len(self._webhooks),
                "enabled_webhooks": sum(1 for c in self._webhooks.values() if c.enabled),
                "total_deliveries": total,
                "successful": successful,
                "failed": failed,
                "success_rate": successful / total if total > 0 else 0.0,
            }


# Global webhook dispatcher
_webhook_dispatcher: Optional[WebhookDispatcher] = None


def get_webhook_dispatcher() -> Optional[WebhookDispatcher]:
    """Get the global webhook dispatcher."""
    return _webhook_dispatcher


def set_webhook_dispatcher(dispatcher: WebhookDispatcher) -> None:
    """Set the global webhook dispatcher."""
    global _webhook_dispatcher
    _webhook_dispatcher = dispatcher