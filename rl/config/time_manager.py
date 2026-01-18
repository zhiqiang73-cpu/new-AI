"""
UTC time utilities for consistent timestamps across the system.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Union
import time


@dataclass
class TimeManager:
    timezone: timezone = timezone.utc
    display_format: str = "%Y-%m-%d %H:%M:%S UTC"

    def now(self) -> datetime:
        return datetime.now(self.timezone)

    def timestamp(self) -> float:
        return time.time()

    def timestamp_ms(self) -> int:
        return int(time.time() * 1000)

    def format_time(self, dt_or_ts: Union[datetime, float, int, str]) -> str:
        if isinstance(dt_or_ts, str):
            try:
                dt_or_ts = datetime.fromisoformat(dt_or_ts)
            except ValueError:
                return str(dt_or_ts)

        if isinstance(dt_or_ts, (int, float)):
            dt_or_ts = datetime.fromtimestamp(dt_or_ts, tz=self.timezone)

        if isinstance(dt_or_ts, datetime):
            if dt_or_ts.tzinfo is None:
                dt_or_ts = dt_or_ts.replace(tzinfo=self.timezone)
            return dt_or_ts.strftime(self.display_format)

        return str(dt_or_ts)

    def get_duration(
        self,
        start_time: Union[datetime, float],
        end_time: Union[datetime, float],
        unit: str = "seconds",
    ) -> float:
        if isinstance(start_time, (int, float)):
            start_time = datetime.fromtimestamp(start_time, tz=self.timezone)
        if isinstance(end_time, (int, float)):
            end_time = datetime.fromtimestamp(end_time, tz=self.timezone)

        seconds = (end_time - start_time).total_seconds()
        if unit == "minutes":
            return seconds / 60
        if unit == "hours":
            return seconds / 3600
        if unit == "days":
            return seconds / 86400
        return seconds


time_manager = TimeManager()


def now() -> datetime:
    return time_manager.now()


def timestamp() -> float:
    return time_manager.timestamp()


def format_time(dt_or_ts: Union[datetime, float, int, str]) -> str:
    return time_manager.format_time(dt_or_ts)


def get_duration(
    start_time: Union[datetime, float],
    end_time: Union[datetime, float],
    unit: str = "seconds",
) -> float:
    return time_manager.get_duration(start_time, end_time, unit)




