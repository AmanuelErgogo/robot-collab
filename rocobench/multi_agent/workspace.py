"""Conservative workspace-zone mapping for Phase 7."""

from typing import Any, Tuple


ZONE_LEFT = "left"
ZONE_RIGHT = "right"
ZONE_CENTER = "center"
ZONE_BIN_LEFT = "bin_left"
ZONE_BIN_RIGHT = "bin_right"
ZONE_HANDOFF = "handoff"
ZONE_UNKNOWN = "unknown"


class WorkspaceMapper(object):
    def robot_zone(self, agent_name: str) -> str:
        if agent_name == "Alice":
            return ZONE_LEFT
        if agent_name == "Bob":
            return ZONE_RIGHT
        return ZONE_UNKNOWN

    def target_zone(self, target_name: str) -> str:
        target = str(target_name)
        if "left" in target:
            return ZONE_BIN_LEFT
        if "right" in target:
            return ZONE_BIN_RIGHT
        if "middle" in target or "center" in target:
            return ZONE_CENTER
        return ZONE_UNKNOWN

    def object_zone(self, object_name: str, env: Any = None) -> str:
        del env
        name = str(object_name)
        # Conservative static defaults for PackGrocery initial layouts. If a
        # caller has richer geometry, it should provide a custom mapper.
        if name in ("apple", "milk", "bread"):
            return ZONE_LEFT
        if name in ("banana", "soda_can", "cereal"):
            return ZONE_RIGHT
        return ZONE_UNKNOWN

    def zones_for_put(self, agent_name: str, object_name: str, target_name: str, env: Any = None) -> Tuple[str, ...]:
        zones = [
            self.robot_zone(agent_name),
            self.object_zone(object_name, env=env),
            self.target_zone(target_name),
        ]
        if ZONE_UNKNOWN in zones:
            return (ZONE_UNKNOWN,)
        # Crossing from one side to the opposite bin is treated as center/handoff
        # and is not admitted concurrently by default.
        if self.robot_zone(agent_name) == ZONE_LEFT and self.target_zone(target_name) == ZONE_BIN_RIGHT:
            zones.append(ZONE_CENTER)
        if self.robot_zone(agent_name) == ZONE_RIGHT and self.target_zone(target_name) == ZONE_BIN_LEFT:
            zones.append(ZONE_CENTER)
        return tuple(sorted(set(zones)))
