"""Track LLM usage: tokens, cost, latency per provider and model."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

from rich.console import Console
from rich.table import Table


@dataclass
class LLMCallRecord:
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float
    latency_ms: float
    timestamp: dt.datetime = field(default_factory=lambda: dt.datetime.now(dt.timezone.utc))


class CostTracker:
    """Accumulates LLM usage statistics across a pipeline run."""

    def __init__(self) -> None:
        self.records: list[LLMCallRecord] = []

    def record(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
        latency_ms: float,
    ) -> LLMCallRecord:
        rec = LLMCallRecord(
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            latency_ms=latency_ms,
        )
        self.records.append(rec)
        return rec

    @property
    def total_calls(self) -> int:
        return len(self.records)

    @property
    def total_tokens(self) -> int:
        return sum(r.input_tokens + r.output_tokens for r in self.records)

    @property
    def total_cost(self) -> float:
        return sum(r.cost_usd for r in self.records)

    @property
    def avg_latency_ms(self) -> float:
        if not self.records:
            return 0.0
        return sum(r.latency_ms for r in self.records) / len(self.records)

    def by_provider(self) -> dict[str, list[LLMCallRecord]]:
        groups: dict[str, list[LLMCallRecord]] = {}
        for r in self.records:
            groups.setdefault(r.provider, []).append(r)
        return groups

    def display(self, console: Console | None = None) -> None:
        console = console or Console()
        table = Table(title="LLM Usage Summary", border_style="blue")
        table.add_column("Provider", style="cyan")
        table.add_column("Calls", justify="right")
        table.add_column("Input Tokens", justify="right")
        table.add_column("Output Tokens", justify="right")
        table.add_column("Total Cost", justify="right", style="green")
        table.add_column("Avg Latency", justify="right")

        for provider, records in self.by_provider().items():
            table.add_row(
                provider,
                str(len(records)),
                f"{sum(r.input_tokens for r in records):,}",
                f"{sum(r.output_tokens for r in records):,}",
                f"${sum(r.cost_usd for r in records):.4f}",
                f"{sum(r.latency_ms for r in records) / len(records):.0f}ms",
            )

        table.add_section()
        table.add_row(
            "[bold]TOTAL[/bold]",
            f"[bold]{self.total_calls}[/bold]",
            f"[bold]{sum(r.input_tokens for r in self.records):,}[/bold]",
            f"[bold]{sum(r.output_tokens for r in self.records):,}[/bold]",
            f"[bold green]${self.total_cost:.4f}[/bold green]",
            f"[bold]{self.avg_latency_ms:.0f}ms[/bold]",
        )
        console.print(table)
