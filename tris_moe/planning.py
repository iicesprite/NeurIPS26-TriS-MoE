"""Ordered weight targets and projection groups for one decoder layer."""

from dataclasses import dataclass

from .model_adapters import extract_expert_info, is_regular_expert, is_shared_expert


@dataclass(frozen=True)
class WeightTarget:
    name: str
    module: object
    role: str
    projection: str = None
    expert_index: int = None

    @classmethod
    def identify(cls, name, module):
        if is_shared_expert(name):
            return cls(name, module, "shared")
        if is_regular_expert(name):
            expert_index, projection = extract_expert_info(name)
            return cls(name, module, "factorized", projection, expert_index)
        return cls(name, module, "direct")


@dataclass(frozen=True)
class ProjectionGroup:
    name: str
    members: tuple

    def ordered_members(self):
        """Use expert IDs for stacking, calibration hooks, and quantization."""
        return tuple(sorted(self.members, key=lambda target: target.expert_index))

    def copy_weights(self):
        return [target.module.weight.data.clone() for target in self.ordered_members()]


@dataclass(frozen=True)
class LayerQuantizationPlan:
    direct_targets: tuple
    projections: tuple

    @classmethod
    def from_modules(cls, modules):
        targets = tuple(WeightTarget.identify(name, module) for name, module in modules.items())
        direct = tuple(target for target in targets if target.role == "direct")
        shared = tuple(target for target in targets if target.role == "shared")
        indexed = {}
        for target in targets:
            if target.role == "factorized":
                indexed.setdefault(target.projection, {})[target.expert_index] = target
        projections = tuple(
            ProjectionGroup(name, tuple(members.values())) for name, members in indexed.items()
        )
        # Keep direct and shared modules in discovery order, followed by each
        # projection in discovery order and its experts in ascending ID order.
        return cls(direct + shared, projections)

    def as_legacy_groups(self):
        """Expose the established three-dictionary helper result."""
        normal = {t.name: t.module for t in self.direct_targets if t.role == "direct"}
        shared = {t.name: t.module for t in self.direct_targets if t.role == "shared"}
        experts = {
            group.name: {t.expert_index: (t.name, t.module) for t in group.members}
            for group in self.projections
        }
        return normal, shared, experts
