"""Adapter settings. Exposes a subset of the global settings."""

from dataclasses import dataclass

from core.settings import get_settings


@dataclass(frozen=True, slots=True)
class AdapterSettings:
    """Subset of runtime configuration needed by the Telegram adapter."""

    telegram_bot_token: str
    backend_url: str
    http_timeout_s: float
    task_poll_initial_s: float
    task_poll_max_s: float
    task_poll_factor: float
    task_poll_timeout_s: float
    request_id_header: str
    log_level: str


def load_adapter_settings() -> AdapterSettings:
    """Build `AdapterSettings` from the shared `Settings`."""
    s = get_settings()
    return AdapterSettings(
        telegram_bot_token=s.telegram_bot_token,
        backend_url=s.adapter_backend_url,
        http_timeout_s=s.adapter_http_timeout_s,
        task_poll_initial_s=s.adapter_task_poll_initial_s,
        task_poll_max_s=s.adapter_task_poll_max_s,
        task_poll_factor=s.adapter_task_poll_factor,
        task_poll_timeout_s=s.adapter_task_poll_timeout_s,
        request_id_header=s.adapter_request_id_header,
        log_level=s.adapter_log_level,
    )
