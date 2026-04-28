"""Compatibility shims for older bot code running on the newer Robocode API."""

from __future__ import annotations

import math
import threading
from typing import Any

from robocode_tank_royale.bot_api import events as events_module
from robocode_tank_royale.bot_api.base_bot_abc import BaseBotABC
from robocode_tank_royale.bot_api.bot import Bot
from robocode_tank_royale.bot_api.internal.base_bot_internals import BaseBotInternals
from robocode_tank_royale.bot_api.internal.bot_event_handlers import BotEventHandlers

_PATCHED = False


def _wrap_radians(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def _force_setattr(obj: Any, name: str, value: Any) -> None:
    try:
        object.__setattr__(obj, name, value)
    except Exception:
        try:
            setattr(obj, name, value)
        except Exception:
            pass


def _enemy_key(bot_id: Any) -> str:
    return str(bot_id)


def _normalize_bullet(bot: Bot, bullet: Any) -> None:
    if bullet is None:
        return
    if hasattr(bullet, "bearing"):
        return
    dx = float(getattr(bullet, "x", 0.0)) - float(bot.x)
    dy = float(getattr(bullet, "y", 0.0)) - float(bot.y)
    bearing = _wrap_radians(math.atan2(dx, dy) - math.radians(float(bot.direction)))
    _force_setattr(bullet, "bearing", bearing)


def _normalize_scanned_bot_event(bot: Bot, event: Any) -> None:
    dx = float(event.x) - float(bot.x)
    dy = float(event.y) - float(bot.y)
    distance = math.hypot(dx, dy)
    absolute_bearing = math.atan2(dx, dy)
    bearing = _wrap_radians(absolute_bearing - math.radians(float(bot.direction)))

    _force_setattr(event, "name", _enemy_key(getattr(event, "scanned_bot_id", "unknown")))
    _force_setattr(event, "distance", distance)
    _force_setattr(event, "bearing", bearing)
    _force_setattr(event, "velocity", float(getattr(event, "speed", 0.0)))

    direction_degrees = float(getattr(event, "direction_degrees", getattr(event, "direction", 0.0)))
    _force_setattr(event, "direction_degrees", direction_degrees)
    _force_setattr(event, "direction", math.radians(direction_degrees))


def _normalize_bot_death_event(event: Any) -> None:
    legacy_name = _enemy_key(getattr(event, "victim_id", "unknown"))
    _force_setattr(event, "name", legacy_name)
    _force_setattr(event, "victim_name", legacy_name)


def _normalize_hit_bot_event(event: Any) -> None:
    legacy_name = _enemy_key(getattr(event, "victim_id", "unknown"))
    _force_setattr(event, "name", legacy_name)
    _force_setattr(event, "victim_name", legacy_name)
    _force_setattr(event, "is_my_fault", bool(getattr(event, "rammed", False)))


def _normalize_bullet_hit_bot_event(event: Any) -> None:
    legacy_name = _enemy_key(getattr(event, "victim_id", "unknown"))
    _force_setattr(event, "name", legacy_name)
    _force_setattr(event, "victim_name", legacy_name)


def _normalize_event(bot: Any, event: Any) -> None:
    if bot is None or event is None:
        return

    if isinstance(event, events_module.ScannedBotEvent):
        _normalize_scanned_bot_event(bot, event)
    elif isinstance(event, events_module.BotDeathEvent):
        _normalize_bot_death_event(event)
    elif isinstance(event, events_module.HitBotEvent):
        _normalize_hit_bot_event(event)
    elif isinstance(event, events_module.BulletHitBotEvent):
        _normalize_bullet_hit_bot_event(event)
    elif isinstance(event, events_module.HitByBulletEvent):
        _normalize_bullet(bot, getattr(event, "bullet", None))


def _patch_event_aliases() -> None:
    if not hasattr(events_module, "HitRobotEvent"):
        events_module.HitRobotEvent = events_module.HitBotEvent
    if not hasattr(events_module, "BulletHitEvent"):
        events_module.BulletHitEvent = events_module.BulletHitBotEvent


def _patch_bot_helpers() -> None:
    def execute(self: Bot) -> None:
        self.go()

    def set_adjust_gun_for_robot_turn(self: Bot, adjust: bool) -> None:
        self.adjust_gun_for_body_turn = adjust

    def set_adjust_radar_for_robot_turn(self: Bot, adjust: bool) -> None:
        self.adjust_radar_for_body_turn = adjust

    def set_adjust_radar_for_gun_turn(self: Bot, adjust: bool) -> None:
        self.adjust_radar_for_gun_turn = adjust

    def set_max_velocity(self: Bot, value: float) -> None:
        self.max_speed = value

    def set_max_turn_rate(self: Bot, value: float) -> None:
        self.max_turn_rate = value

    def set_max_gun_turn_rate(self: Bot, value: float) -> None:
        self.max_gun_turn_rate = value

    def set_max_radar_turn_rate(self: Bot, value: float) -> None:
        self.max_radar_turn_rate = value

    Bot.execute = execute
    Bot.set_adjust_gun_for_robot_turn = set_adjust_gun_for_robot_turn
    Bot.set_adjust_radar_for_robot_turn = set_adjust_radar_for_robot_turn
    Bot.set_adjust_radar_for_gun_turn = set_adjust_radar_for_gun_turn
    Bot.set_max_velocity = set_max_velocity
    Bot.set_max_turn_rate = set_max_turn_rate
    Bot.set_max_gun_turn_rate = set_max_gun_turn_rate
    Bot.set_max_radar_turn_rate = set_max_radar_turn_rate
    Bot.time = property(lambda self: self.turn_number)
    Bot.others = property(lambda self: self.enemy_count)
    Bot.velocity = property(lambda self: self.speed)

    original_turn_radar_left = Bot.turn_radar_left
    original_turn_radar_right = Bot.turn_radar_right

    def turn_radar_left(self: Bot, degrees: float) -> None:
        if math.isinf(degrees):
            self.set_turn_radar_left(degrees)
            return
        original_turn_radar_left(self, degrees)

    def turn_radar_right(self: Bot, degrees: float) -> None:
        if math.isinf(degrees):
            self.set_turn_radar_right(degrees)
            return
        original_turn_radar_right(self, degrees)

    Bot.turn_radar_left = turn_radar_left
    Bot.turn_radar_right = turn_radar_right


def _patch_legacy_event_fallbacks() -> None:
    original_on_bot_death = BaseBotABC.on_bot_death
    original_on_hit_bot = BaseBotABC.on_hit_bot

    def on_bot_death(self: BaseBotABC, bot_death_event: Any) -> None:
        legacy_handler = getattr(type(self), "on_robot_death", None)
        if legacy_handler is not None:
            return legacy_handler(self, bot_death_event)
        return original_on_bot_death(self, bot_death_event)

    def on_hit_bot(self: BaseBotABC, bot_hit_bot_event: Any) -> None:
        legacy_handler = getattr(type(self), "on_hit_robot", None)
        if legacy_handler is not None:
            return legacy_handler(self, bot_hit_bot_event)
        return original_on_hit_bot(self, bot_hit_bot_event)

    BaseBotABC.on_bot_death = on_bot_death
    BaseBotABC.on_hit_bot = on_hit_bot


def _patch_stop_thread() -> None:
    original_stop_thread = BaseBotInternals.stop_thread

    def stop_thread(self: BaseBotInternals) -> None:
        if not self.is_running():
            return

        self.set_running(False)
        with self._next_turn_condition:
            self._next_turn_condition.notify_all()

        if self.thread is None:
            return
        if self.thread is threading.current_thread():
            self.thread = None
            return
        original_stop_thread(self)

    BaseBotInternals.stop_thread = stop_thread


def _patch_bot_event_handlers() -> None:
    original_init = BotEventHandlers.__init__
    original_fire_event = BotEventHandlers.fire_event

    def __init__(self: BotEventHandlers, base_bot: Any) -> None:
        original_init(self, base_bot)
        self._compat_base_bot = base_bot

    def fire_event(self: BotEventHandlers, event: Any) -> None:
        _normalize_event(getattr(self, "_compat_base_bot", None), event)
        original_fire_event(self, event)

    BotEventHandlers.__init__ = __init__
    BotEventHandlers.fire_event = fire_event


def apply() -> None:
    global _PATCHED
    if _PATCHED:
        return
    _patch_event_aliases()
    _patch_bot_helpers()
    _patch_legacy_event_fallbacks()
    _patch_stop_thread()
    _patch_bot_event_handlers()
    _PATCHED = True


apply()
