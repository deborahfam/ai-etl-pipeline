"""Pipeline engine: DAG resolution and execution."""

from __future__ import annotations

from typing import Any

import polars as pl
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from src.engine.context import PipelineContext
from src.engine.models import StepResult, StepStatus, StepType
from src.engine.step import Step

console = Console()


class Pipeline:
    """Directed Acyclic Graph of Steps with automatic dependency resolution."""

    def __init__(
        self,
        name: str,
        description: str = "",
        config: dict[str, Any] | None = None,
    ) -> None:
        self.name = name
        self.description = description
        self._steps: dict[str, Step] = {}
        self._config = config or {}

    def add_step(self, step: Step) -> "Pipeline":
        self._steps[step.name] = step
        return self

    def add_steps(self, steps: list[Step]) -> "Pipeline":
        for step in steps:
            self.add_step(step)
        return self

    # -- DAG resolution ------------------------------------------------------

    def _resolve_order(self) -> list[Step]:
        """Topological sort of steps based on depends_on."""
        visited: set[str] = set()
        order: list[Step] = []

        def visit(name: str) -> None:
            if name in visited:
                return
            visited.add(name)
            step = self._steps[name]
            for dep in step.depends_on:
                if dep not in self._steps:
                    raise ValueError(
                        f"Step '{name}' depends on '{dep}', which is not registered."
                    )
                visit(dep)
            order.append(step)

        for name in self._steps:
            visit(name)

        return order

    # -- execution -----------------------------------------------------------

    def run(
        self,
        ctx: PipelineContext | None = None,
        llm: Any | None = None,
        show_progress: bool = True,
    ) -> PipelineContext:
        """Execute the full pipeline in dependency order."""
        if ctx is None:
            ctx = PipelineContext(pipeline_name=self.name, config=self._config)

        steps = self._resolve_order()
        ctx.start()

        # Track outputs by step name for chaining
        outputs: dict[str, Any] = {}
        current_df: pl.DataFrame | None = None

        if show_progress:
            console.print(
                Panel(
                    f"[bold cyan]{self.name}[/bold cyan]\n{self.description}",
                    title="Pipeline Starting",
                    border_style="cyan",
                )
            )

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            TimeElapsedColumn(),
            console=console,
            disable=not show_progress,
        ) as progress:
            task = progress.add_task("Running pipeline...", total=len(steps))

            for step in steps:
                progress.update(
                    task, description=f"[cyan]{step.step_type.value}[/cyan] {step.name}"
                )
                ctx.event_bus.emit(
                    "on_step_start", step_name=step.name, step_type=step.step_type
                )

                # Build kwargs for the step function
                kwargs: dict[str, Any] = {}
                sig_params = _get_func_params(step.func)
                if "ctx" in sig_params:
                    kwargs["ctx"] = ctx
                if "df" in sig_params and current_df is not None:
                    kwargs["df"] = current_df
                if "llm" in sig_params and llm is not None:
                    kwargs["llm"] = llm
                if "outputs" in sig_params:
                    kwargs["outputs"] = outputs

                try:
                    rows_before = len(current_df) if current_df is not None else 0
                    cols_before = (
                        set(current_df.columns) if current_df is not None else set()
                    )

                    output, result = step.execute(**kwargs)

                    # Update current dataframe if step returned one
                    if isinstance(output, pl.DataFrame):
                        result.rows_in = rows_before
                        result.rows_out = len(output)
                        new_cols = set(output.columns) - cols_before
                        removed_cols = cols_before - set(output.columns)
                        result.columns_added = sorted(new_cols)
                        result.columns_removed = sorted(removed_cols)
                        if new_cols:
                            ctx.record_lineage(step.name, list(new_cols))
                        ctx.take_snapshot(step.name, output)
                        current_df = output

                    outputs[step.name] = output
                    ctx.run.steps.append(result)
                    ctx.event_bus.emit(
                        "on_step_complete", step_name=step.name, result=result
                    )

                except Exception as exc:
                    result = StepResult(
                        step_name=step.name,
                        step_type=step.step_type,
                        status=StepStatus.FAILED,
                        error=str(exc),
                    )
                    ctx.run.steps.append(result)
                    ctx.event_bus.emit(
                        "on_step_error", step_name=step.name, error=exc
                    )
                    ctx.fail(exc)
                    if show_progress:
                        console.print(
                            f"  [red]FAILED[/red] {step.name}: {exc}"
                        )
                    raise

                progress.advance(task)

        ctx.complete()
        ctx.run.total_rows_processed = len(current_df) if current_df is not None else 0

        if show_progress:
            self._print_summary(ctx)

        return ctx

    # -- display -------------------------------------------------------------

    def _print_summary(self, ctx: PipelineContext) -> None:
        table = Table(title="Pipeline Results", border_style="green")
        table.add_column("Step", style="cyan")
        table.add_column("Type", style="magenta")
        table.add_column("Status")
        table.add_column("Duration", justify="right")
        table.add_column("Rows", justify="right")
        table.add_column("Columns +/-", justify="right")

        for step in ctx.run.steps:
            status_style = {
                StepStatus.COMPLETED: "[green]OK[/green]",
                StepStatus.FAILED: "[red]FAIL[/red]",
                StepStatus.SKIPPED: "[yellow]SKIP[/yellow]",
            }.get(step.status, str(step.status.value))

            cols = ""
            if step.columns_added or step.columns_removed:
                parts = []
                if step.columns_added:
                    parts.append(f"[green]+{len(step.columns_added)}[/green]")
                if step.columns_removed:
                    parts.append(f"[red]-{len(step.columns_removed)}[/red]")
                cols = " ".join(parts)

            table.add_row(
                step.step_name,
                step.step_type.value,
                status_style,
                step.duration_display,
                f"{step.rows_in}→{step.rows_out}" if step.rows_out else "-",
                cols or "-",
            )

        console.print(table)

        # LLM usage summary
        run = ctx.run
        if run.total_llm_calls > 0:
            console.print(
                Panel(
                    f"LLM Calls: [cyan]{run.total_llm_calls}[/cyan] | "
                    f"Tokens: [cyan]{run.total_tokens_used:,}[/cyan] | "
                    f"Cost: [green]${run.total_llm_cost_usd:.4f}[/green] | "
                    f"Duration: [yellow]{run.duration_seconds:.2f}s[/yellow]",
                    title="AI Usage",
                    border_style="blue",
                )
            )

    def __repr__(self) -> str:
        return f"Pipeline('{self.name}', steps={list(self._steps.keys())})"


def _get_func_params(func: Any) -> set[str]:
    """Get parameter names from a function, handling wrappers."""
    import inspect

    target = getattr(func, "__wrapped__", func)
    try:
        sig = inspect.signature(target)
        return set(sig.parameters.keys())
    except (ValueError, TypeError):
        return set()
