"""Resolve precision-specific runtime topology from one validated interface."""

from __future__ import annotations

import os
from collections.abc import Collection, Mapping
from dataclasses import dataclass

from wentian.settings import SUPPORTED_PRECISIONS

FP64_NODE31_CPU_DOMAINS = (0, 1, 2, 3, 6, 7, 8, 10, 11, 12, 13, 14, 15, 5, 9)
DEFAULT_CPU_DOMAIN_CORES = 38
DEFAULT_HALO_HEIGHT = 6
DEFAULT_HALO_WIDTH = 12


def _integer(environment: Mapping[str, str], name: str, default: int) -> int:
    return int(environment.get(name, default))


def _domain_list(value: str) -> tuple[int, ...]:
    return tuple(int(item) for item in value.split(",") if item.strip())


def _cpu_map(value: str) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(int(cpu) for cpu in group.split(",") if cpu.strip()) for group in value.split(";")
    )


@dataclass(frozen=True)
class CPUAssignment:
    """CPUs and physical NUMA domain assigned to one worker."""

    cpus: tuple[int, ...]
    domain: int


@dataclass(frozen=True)
class RuntimeTopology:
    """Complete worker layout selected by precision with optional environment overrides."""

    height_partitions: int
    width_partitions: int
    worker_threads: int
    main_threads: int
    cpu_domain_cores: int = DEFAULT_CPU_DOMAIN_CORES
    halo_height: int = DEFAULT_HALO_HEIGHT
    halo_width: int = DEFAULT_HALO_WIDTH
    cpu_domains: tuple[int, ...] = ()
    excluded_cpu_domains: frozenset[int] = frozenset()
    cpu_map: tuple[tuple[int, ...], ...] = ()

    def __post_init__(self) -> None:
        positive_fields = {
            "height_partitions": self.height_partitions,
            "width_partitions": self.width_partitions,
            "worker_threads": self.worker_threads,
            "main_threads": self.main_threads,
            "cpu_domain_cores": self.cpu_domain_cores,
            "halo_height": self.halo_height,
            "halo_width": self.halo_width,
        }
        invalid = [name for name, value in positive_fields.items() if value <= 0]
        if invalid:
            raise ValueError(f"{', '.join(invalid)} must be positive")
        if self.halo_height < DEFAULT_HALO_HEIGHT:
            raise ValueError(f"halo_height must be at least {DEFAULT_HALO_HEIGHT}")
        if self.halo_width < DEFAULT_HALO_WIDTH:
            raise ValueError(f"halo_width must be at least {DEFAULT_HALO_WIDTH}")

        configured_sources = sum(
            bool(value) for value in (self.cpu_domains, self.excluded_cpu_domains, self.cpu_map)
        )
        if configured_sources > 1:
            raise ValueError(
                "CPUMAP, WENTIAN_CPU_DOMAINS, and WENTIAN_EXCLUDED_CPU_DOMAINS "
                "are mutually exclusive"
            )
        if any(domain < 0 for domain in self.cpu_domains) or len(set(self.cpu_domains)) != len(
            self.cpu_domains
        ):
            raise ValueError("CPU domains must contain unique non-negative integers")
        if any(domain < 0 for domain in self.excluded_cpu_domains):
            raise ValueError("excluded CPU domains must be non-negative")
        if self.cpu_domains and len(self.cpu_domains) < self.worker_count:
            raise ValueError("WENTIAN_CPU_DOMAINS defines fewer domains than workers")
        if self.cpu_map and len(self.cpu_map) != self.worker_count:
            raise ValueError(f"CPUMAP must define exactly {self.worker_count} CPU groups")

        mapped_cpus = tuple(cpu for group in self.cpu_map for cpu in group)
        if self.cpu_map and any(not group for group in self.cpu_map):
            raise ValueError("CPUMAP must not contain empty CPU groups")
        if any(cpu < 0 for cpu in mapped_cpus) or len(set(mapped_cpus)) != len(mapped_cpus):
            raise ValueError("CPUMAP must contain unique non-negative CPU numbers")
        if not self.cpu_map and self.worker_threads > self.cpu_domain_cores:
            raise ValueError("worker_threads may not exceed cpu_domain_cores without CPUMAP")

    @property
    def worker_count(self) -> int:
        return self.height_partitions * self.width_partitions

    @classmethod
    def from_environment(
        cls,
        precision: str,
        environment: Mapping[str, str] | None = None,
    ) -> RuntimeTopology:
        environment = os.environ if environment is None else environment
        precision = precision.lower()
        if precision not in SUPPORTED_PRECISIONS:
            choices = ", ".join(SUPPORTED_PRECISIONS)
            raise ValueError(f"precision must be one of {choices}")

        fp64 = precision == "fp64"
        default_height = 3 if fp64 else 4
        default_width = 5 if fp64 else 4
        default_main_threads = 38 if fp64 else 64
        height_partitions = _integer(environment, "WPH", default_height)
        width_partitions = _integer(environment, "WPW", default_width)
        worker_threads = _integer(environment, "WNT", 38)
        main_threads = _integer(environment, "WNT_MAIN", default_main_threads)
        cpu_domain_cores = _integer(
            environment,
            "GD_HBM_DOMAIN_CORES",
            DEFAULT_CPU_DOMAIN_CORES,
        )
        halo_height = _integer(environment, "HALO_H", DEFAULT_HALO_HEIGHT)
        halo_width = _integer(environment, "HALO_W", DEFAULT_HALO_WIDTH)

        configured_domains = _domain_list(environment.get("WENTIAN_CPU_DOMAINS", ""))
        excluded_domains = frozenset(
            _domain_list(environment.get("WENTIAN_EXCLUDED_CPU_DOMAINS", ""))
        )
        configured_cpu_map = _cpu_map(environment["CPUMAP"]) if environment.get("CPUMAP") else ()
        configured_sources = sum(
            bool(value) for value in (configured_domains, excluded_domains, configured_cpu_map)
        )
        is_tuned_fp64_profile = (
            fp64
            and height_partitions == 3
            and width_partitions == 5
            and worker_threads == 38
            and cpu_domain_cores == DEFAULT_CPU_DOMAIN_CORES
        )
        if configured_sources == 0 and is_tuned_fp64_profile:
            configured_domains = FP64_NODE31_CPU_DOMAINS

        return cls(
            height_partitions=height_partitions,
            width_partitions=width_partitions,
            worker_threads=worker_threads,
            main_threads=main_threads,
            cpu_domain_cores=cpu_domain_cores,
            halo_height=halo_height,
            halo_width=halo_width,
            cpu_domains=configured_domains,
            excluded_cpu_domains=excluded_domains,
            cpu_map=configured_cpu_map,
        )

    def environment(self) -> dict[str, str]:
        """Return the canonical environment consumed by the optimized process."""
        return {
            "WPH": str(self.height_partitions),
            "WPW": str(self.width_partitions),
            "WNT": str(self.worker_threads),
            "WNT_MAIN": str(self.main_threads),
            "GD_HBM_DOMAIN_CORES": str(self.cpu_domain_cores),
            "HALO_H": str(self.halo_height),
            "HALO_W": str(self.halo_width),
            "WENTIAN_CPU_DOMAINS": ",".join(map(str, self.cpu_domains)),
            "WENTIAN_EXCLUDED_CPU_DOMAINS": ",".join(map(str, sorted(self.excluded_cpu_domains))),
            "CPUMAP": ";".join(",".join(map(str, group)) for group in self.cpu_map),
        }

    def cpu_assignments(self, available_cpus: Collection[int]) -> tuple[CPUAssignment, ...]:
        """Resolve worker CPU assignments against the process affinity mask."""
        available = set(available_cpus)
        if not available:
            raise RuntimeError("the process has no available CPUs")

        if self.cpu_map:
            assignments = []
            for group in self.cpu_map:
                domains = {cpu // self.cpu_domain_cores for cpu in group}
                if len(domains) != 1:
                    raise ValueError("each CPUMAP group must stay within one physical CPU domain")
                assignments.append(CPUAssignment(tuple(group), domains.pop()))
            assignments = tuple(assignments)
        else:
            domain_count = max(available) // self.cpu_domain_cores + 1
            complete_domains = [
                domain
                for domain in range(domain_count)
                if set(self._worker_cpus(domain)).issubset(available)
            ]
            domains = self.cpu_domains or tuple(
                domain for domain in complete_domains if domain not in self.excluded_cpu_domains
            )
            if len(domains) < self.worker_count:
                raise ValueError("available CPU domains leave fewer domains than workers")
            assignments = tuple(
                CPUAssignment(self._worker_cpus(domain), domain)
                for domain in domains[: self.worker_count]
            )

        if len(assignments) != self.worker_count:
            raise ValueError(f"CPU assignments must define exactly {self.worker_count} workers")
        assigned_cpus = [cpu for assignment in assignments for cpu in assignment.cpus]
        if len(set(assigned_cpus)) != len(assigned_cpus):
            raise ValueError("CPU assignments must not reuse CPUs across workers")
        unavailable_cpus = set(assigned_cpus) - available
        if unavailable_cpus:
            raise ValueError(
                f"CPU assignments include unavailable CPUs: {sorted(unavailable_cpus)}"
            )
        return assignments

    def _worker_cpus(self, domain: int) -> tuple[int, ...]:
        start = domain * self.cpu_domain_cores
        return tuple(range(start, start + self.worker_threads))
