"""Validated spatial decomposition plans for the optimized Wentian runtime."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from wentian.runtime.topology import RuntimeTopology

HEIGHT_ALIGNMENT = 12
WIDTH_ALIGNMENT = 24


def _partition_bounds(
    length: int,
    alignment: int,
    partition_count: int,
) -> tuple[tuple[int, int], ...]:
    if length <= 0 or alignment <= 0 or partition_count <= 0:
        raise ValueError("length, alignment, and partition_count must be positive")
    aligned_units = length // alignment
    if partition_count > aligned_units:
        raise ValueError(
            f"cannot split {length} positions into {partition_count} non-empty "
            f"partitions aligned to {alignment}"
        )

    base_units, remainder = divmod(aligned_units, partition_count)
    bounds = []
    consumed_units = 0
    for partition in range(partition_count):
        units = base_units + (1 if partition < remainder else 0)
        start = consumed_units * alignment
        stop = (
            min((consumed_units + units) * alignment, length)
            if partition < partition_count - 1
            else length
        )
        bounds.append((start, stop))
        consumed_units += units
    return tuple(bounds)


@dataclass(frozen=True)
class StagePlan:
    """Shape and ownership bounds for one transformer stage."""

    name: str
    shape: tuple[int, int, int, int, int]
    height_bounds: tuple[tuple[int, int], ...]
    width_bounds: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class RuntimePlan:
    """Complete spatial plan derived from one runtime topology."""

    topology: RuntimeTopology
    stages: tuple[StagePlan, ...]

    @classmethod
    def from_topology(cls, topology: RuntimeTopology) -> RuntimePlan:
        d0_height = _partition_bounds(
            181,
            HEIGHT_ALIGNMENT,
            topology.height_partitions,
        )
        d1_height = [(start // 2, stop // 2) for start, stop in d0_height]
        d1_height[-1] = (d0_height[-1][0] // 2, 91)
        u1_height = _partition_bounds(
            182,
            HEIGHT_ALIGNMENT,
            topology.height_partitions,
        )

        d0_width = _partition_bounds(
            360,
            WIDTH_ALIGNMENT,
            topology.width_partitions,
        )
        d1_width = tuple((start // 2, stop // 2) for start, stop in d0_width)

        return cls(
            topology=topology,
            stages=(
                StagePlan("d0", (1, 8, 181, 360, 256), d0_height, d0_width),
                StagePlan("d1", (1, 8, 91, 180, 512), tuple(d1_height), d1_width),
                StagePlan("u0", (1, 8, 91, 180, 512), tuple(d1_height), d1_width),
                StagePlan("u1", (1, 8, 182, 360, 256), u1_height, d0_width),
            ),
        )

    @property
    def stage_names(self) -> tuple[str, ...]:
        return tuple(stage.name for stage in self.stages)

    def stage(self, name: str) -> StagePlan:
        try:
            return next(stage for stage in self.stages if stage.name == name)
        except StopIteration as exc:
            raise KeyError(f"unknown stage: {name}") from exc

    def stage_index(self, name: str) -> int:
        try:
            return self.stage_names.index(name)
        except ValueError as exc:
            raise KeyError(f"unknown stage: {name}") from exc


def model_stage_blocks(model: Any) -> dict[str, Sequence[Any]]:
    """Return the checkpoint architecture's transformer blocks by stage name."""
    return {
        "d0": model.down_blocks[0].blocks,
        "d1": model.down_blocks[1].blocks,
        "u0": model.up_blocks[0].blocks,
        "u1": model.up_blocks[1].blocks,
    }


@dataclass
class _FlowEdge:
    target: int
    reverse: int
    capacity: int
    cost: int


def _add_flow_edge(graph: list[list[_FlowEdge]], source: int, target: int, cost: int) -> _FlowEdge:
    forward = _FlowEdge(target, len(graph[target]), 1, cost)
    backward = _FlowEdge(source, len(graph[source]), 0, -cost)
    graph[source].append(forward)
    graph[target].append(backward)
    return forward


def _globally_balanced_owners(
    indices: Sequence[int],
    reachable_by: Mapping[int, set[int]],
    overlaps: Mapping[int, Mapping[int, int]],
    partition_count: int,
    window_size: int,
) -> dict[int, int]:
    """Minimize load variance, preserve remainder order, then maximize locality."""
    source = 0
    window_offset = 1
    partition_offset = window_offset + len(indices)
    sink = partition_offset + partition_count
    graph: list[list[_FlowEdge]] = [[] for _ in range(sink + 1)]
    owner_edges: dict[int, list[tuple[int, _FlowEdge]]] = {}
    locality_budget = len(indices) * window_size
    remainder_weight = locality_budget + 1
    remainder_budget = len(indices) * max(0, partition_count - 1) * remainder_weight
    balance_weight = remainder_budget + locality_budget + 1
    base_load = len(indices) // partition_count

    for offset, index in enumerate(indices):
        window_node = window_offset + offset
        _add_flow_edge(graph, source, window_node, 0)
        owner_edges[index] = []
        for partition in sorted(reachable_by[index]):
            locality_penalty = window_size - overlaps[index][partition]
            edge = _add_flow_edge(
                graph,
                window_node,
                partition_offset + partition,
                locality_penalty,
            )
            owner_edges[index].append((partition, edge))

    for partition in range(partition_count):
        partition_node = partition_offset + partition
        for slot in range(len(indices)):
            remainder_cost = partition * remainder_weight if slot >= base_load else 0
            _add_flow_edge(
                graph,
                partition_node,
                sink,
                slot * balance_weight + remainder_cost,
            )

    for _ in indices:
        distances = [float("inf")] * len(graph)
        parents: list[tuple[int, int] | None] = [None] * len(graph)
        distances[source] = 0
        for _ in range(len(graph) - 1):
            changed = False
            for node, edges in enumerate(graph):
                if distances[node] == float("inf"):
                    continue
                for edge_index, edge in enumerate(edges):
                    candidate = distances[node] + edge.cost
                    if edge.capacity and candidate < distances[edge.target]:
                        distances[edge.target] = candidate
                        parents[edge.target] = (node, edge_index)
                        changed = True
            if not changed:
                break
        if parents[sink] is None:
            raise RuntimeError("could not assign every window to a reachable partition")

        node = sink
        while node != source:
            parent, edge_index = parents[node]
            edge = graph[parent][edge_index]
            edge.capacity -= 1
            graph[node][edge.reverse].capacity += 1
            node = parent

    owners = {}
    for index, edges in owner_edges.items():
        used = [partition for partition, edge in edges if edge.capacity == 0]
        if len(used) != 1:
            raise RuntimeError(f"window {index} has {len(used)} owners")
        owners[index] = used[0]
    return owners


def locality_balanced_owners(
    window_maps: Sequence[Sequence[int]],
    window_size: int,
    shift: int,
    length: int,
    bounds: Sequence[tuple[int, int]],
) -> dict[int, int]:
    """Balance window counts while maximizing overlap with each owner's tile."""
    if not bounds:
        raise ValueError("at least one partition is required")
    if len(window_maps) != len(bounds):
        raise ValueError("window maps and partition bounds must have the same length")
    if window_size <= 0 or length <= 0 or length % window_size:
        raise ValueError("length must be a positive multiple of window_size")

    reachable_by: dict[int, set[int]] = {}
    for partition, indices in enumerate(window_maps):
        for index in indices:
            reachable_by.setdefault(index, set()).add(partition)

    expected_indices = set(range(length // window_size))
    actual_indices = set(reachable_by)
    if actual_indices != expected_indices:
        missing = sorted(expected_indices - actual_indices)
        unexpected = sorted(actual_indices - expected_indices)
        raise ValueError(
            f"window maps do not cover the global window grid; "
            f"missing={missing}, unexpected={unexpected}"
        )

    indices = sorted(reachable_by)
    overlaps = {}
    for index, candidates in reachable_by.items():
        positions = [
            (index * window_size + offset + shift) % length for offset in range(window_size)
        ]
        overlaps[index] = {
            candidate: sum(
                bounds[candidate][0] <= position < bounds[candidate][1]
                or candidate == len(bounds) - 1
                and position >= bounds[candidate][0]
                for position in positions
            )
            for candidate in candidates
        }

    return _globally_balanced_owners(
        indices,
        reachable_by,
        overlaps,
        len(bounds),
        window_size,
    )
