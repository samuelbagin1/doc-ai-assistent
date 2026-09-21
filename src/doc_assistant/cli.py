"""Interaktívne terminálové rozhranie pre otázky, súbory, citácie a metriky."""

from __future__ import annotations

import os
import shlex
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from qdrant_client import QdrantClient
from rich import box
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from doc_assistant.api_resilience import ModelCallGate
from doc_assistant.chunking import SectionAwareChunker
from doc_assistant.config import Settings
from doc_assistant.document_store import DocumentStore
from doc_assistant.domain import AssistantAnswer
from doc_assistant.jev_verifier import JevVerifier
from doc_assistant.observability import MetricsStore
from doc_assistant.providers import (
    LocalEmbeddings,
    OpenAIAnswerModel,
    OpenAIEmbeddings,
    OpenAIWebSearch,
)
from doc_assistant.services import DocumentService
from doc_assistant.vector_store import QdrantVectorStore
from doc_assistant.workflow import RAGWorkflow


class AssistantCLI:
    """Zobrazuje chatový loop, správu dokumentov, zdroje a prevádzkové metriky."""

    def __init__(
        self,
        *,
        settings: Settings,
        documents: DocumentService,
        workflow: RAGWorkflow,
        metrics: MetricsStore,
        console: Console,
    ) -> None:
        """Prijme konfiguráciu a aplikačné služby; pripraví konzolu a históriu."""
        self.settings = settings
        self.documents = documents
        self.workflow = workflow
        self.metrics = metrics
        self.console = console
        self.last_answer: AssistantAnswer | None = None
        self.last_interaction_id: str | None = None
        self.session = PromptSession(
            history=FileHistory(str(settings.data_dir / "cli_history")),
        )

    def run(self) -> None:
        """Číta terminálové vstupy až po /exit; nič nevracia."""
        self._banner()
        while True:
            try:
                raw = self.session.prompt("❯ ").strip()
            except (EOFError, KeyboardInterrupt):
                self.console.print("\n[dim]Ukončujem asistenta.[/dim]")
                return
            if not raw:
                continue
            if raw.startswith("/"):
                if self._command(raw):
                    return
                continue
            self._ask(raw)

    def _command(self, raw: str) -> bool:
        """Prijme slash príkaz, vykoná ho a vráti True iba pri ukončení aplikácie."""
        try:
            parts = shlex.split(raw)
        except ValueError as error:
            self._error(str(error))
            return False
        command = parts[0].lower()
        argument = " ".join(parts[1:]).strip()
        actions = {
            "/help": lambda: self._help(),
            "/add": lambda: self._add(argument),
            "/files": self._files,
            "/delete": lambda: self._delete(argument),
            "/sources": self._sources,
            "/metrics": self._metrics,
            "/status": self._status,
            "/clear": self.console.clear,
            "/good": lambda: self._feedback(1, argument),
            "/bad": lambda: self._feedback(-1, argument),
        }
        if command == "/exit":
            self.console.print("[dim]Dovidenia.[/dim]")
            return True
        action = actions.get(command)
        if action is None:
            self._error(f"Neznámy príkaz {escape(command)}. Použite /help.")
        else:
            try:
                action()
            except Exception as error:  # noqa: BLE001 - UI boundary keeps the chat loop alive
                self._error(str(error))
        return False

    def _ask(self, question: str) -> None:
        """Odošle otázku workflowu a zobrazí odpoveď, citácie a tokeny."""
        try:
            with self.console.status("[cyan]Hľadám a overujem podklady…[/cyan]", spinner="dots"):
                answer = self.workflow.ask(question)
            self.last_answer = answer
            self.last_interaction_id = self.metrics.record(
                question,
                answer,
                tenant_id=self.settings.tenant_id,
                user_id=self.settings.user_id,
            )
            color = "yellow" if answer.abstained else "cyan"
            title = "Zdržanie sa odpovede" if answer.abstained else "Odpoveď"
            self.console.print(
                Panel(
                    Text(answer.text),
                    title=f"[{color}]{title}[/{color}]",
                    subtitle=(
                        f"istota {answer.confidence:.0%} · {answer.route} · "
                        f"{answer.latency_ms / 1000:.2f} s · "
                        f"{answer.input_tokens + answer.output_tokens} tokenov"
                    ),
                    border_style=color,
                    padding=(1, 2),
                )
            )
            self._sources()
            self.console.print("[dim]Ohodnoťte odpoveď pomocou /good alebo /bad poznámka.[/dim]")
        except Exception as error:  # noqa: BLE001 - API/provider errors become actionable UI output
            self._error(f"Inferencia zlyhala: {error}")

    def _add(self, argument: str) -> None:
        """Prijme cestu k dokumentu alebo si ju vypýta a pridá ho do úložiska."""
        if not argument:
            argument = self.session.prompt("Cesta k dokumentu: ").strip()
        if not argument:
            return
        with self.console.status("[cyan]Spracúvam, chunkujem a indexujem dokument…[/cyan]"):
            record = self.documents.add(Path(argument))
        self.console.print(
            f"[green]✓[/green] {escape(record.filename)} · {record.chunk_count} chunkov · "
            f"ID [dim]{record.id[:8]}[/dim]"
        )

    def _files(self) -> None:
        """Zobrazí dokumenty a voliteľne vyžiada potvrdený výber na zmazanie."""
        records = self.documents.list()
        if not records:
            self.console.print("[dim]Nie sú uložené žiadne dokumenty. Použite /add cesta.[/dim]")
            return
        table = Table(title="Uložené dokumenty", box=box.SIMPLE_HEAD, header_style="bold cyan")
        table.add_column("#", justify="right", style="dim")
        table.add_column("Názov")
        table.add_column("ID", style="dim")
        table.add_column("Chunky", justify="right")
        table.add_column("Pridané")
        for index, record in enumerate(records, start=1):
            table.add_row(
                str(index),
                record.filename,
                record.id[:8],
                str(record.chunk_count),
                record.created_at[:16].replace("T", " "),
            )
        self.console.print(table)
        selection = self.session.prompt("Vymazať #/ID (Enter = späť): ").strip()
        if selection:
            if selection.isdigit() and 1 <= int(selection) <= len(records):
                document_id = records[int(selection) - 1].id
            else:
                matches = [record.id for record in records if record.id.startswith(selection)]
                if len(matches) != 1:
                    raise ValueError("Výber nie je jednoznačný.")
                document_id = matches[0]
            confirm = self.session.prompt("Naozaj odstrániť súbor aj jeho vektory? [y/N] ").lower()
            if confirm in {"y", "yes", "a", "áno", "ano"}:
                self._delete(document_id)

    def _delete(self, argument: str) -> None:
        """Prijme ID dokumentu a odstráni súbor aj jeho vektory."""
        if not argument:
            raise ValueError("Použitie: /delete ID alebo vyberte dokument cez /files.")
        records = self.documents.list()
        matches = [record.id for record in records if record.id.startswith(argument)]
        document_id = matches[0] if len(matches) == 1 else argument
        with self.console.status("[yellow]Odstraňujem súbor a vektory…[/yellow]"):
            record = self.documents.delete(document_id)
        self.console.print(f"[green]✓[/green] Odstránené: {escape(record.filename)}")

    def _sources(self) -> None:
        """Bez vstupu zobrazí citácie poslednej odpovede alebo dôvod abstencie."""
        if self.last_answer is None:
            self.console.print("[dim]Zatiaľ nie je dostupná žiadna odpoveď.[/dim]")
            return
        if not self.last_answer.sources:
            if self.last_answer.reason:
                self.console.print(f"[yellow]Dôvod:[/yellow] {escape(self.last_answer.reason)}")
            return
        table = Table(title="Zdroje", box=box.SIMPLE, show_header=False)
        table.add_column(style="cyan", no_wrap=True)
        table.add_column(overflow="fold")
        for index, source in enumerate(self.last_answer.sources, start=1):
            table.add_row(f"[{index}]", source.locator)
            if source.excerpt:
                table.add_row("", Text(source.excerpt, style="dim"))
        self.console.print(table)

    def _metrics(self) -> None:
        """Bez vstupu zobrazí agregované metriky aktuálneho tenantu."""
        value = self.metrics.summary(tenant_id=self.settings.tenant_id)
        table = Table(title="Prevádzkové metriky", box=box.ROUNDED)
        table.add_column("Otázky", justify="right")
        table.add_column("Abstencie", justify="right")
        table.add_column("Priem. čas", justify="right")
        table.add_column("Tokeny", justify="right")
        table.add_column("Negatívne", justify="right")
        rate = value.abstentions / value.questions if value.questions else 0
        table.add_row(
            str(value.questions),
            f"{value.abstentions} ({rate:.0%})",
            f"{value.average_latency_ms / 1000:.2f} s",
            f"{value.input_tokens + value.output_tokens:,}",
            str(value.negative_feedback),
        )
        self.console.print(table)

    def _feedback(self, value: int, note: str) -> None:
        """Prijme hodnotenie ±1 a poznámku; uloží ich k poslednej odpovedi."""
        if not self.last_interaction_id:
            raise ValueError("Najprv položte otázku.")
        self.metrics.feedback(self.last_interaction_id, value, note)
        self.console.print("[green]✓[/green] Ďakujem, hodnotenie bolo uložené.")

    def _status(self) -> None:
        """Bez vstupu zobrazí modely, počet dokumentov a konfiguráciu fallbacku."""
        self.console.print(
            Panel.fit(
                f"Dokumenty: [cyan]{len(self.documents.list())}[/cyan]\n"
                f"Embedding: [cyan]{escape(self.settings.embedding_provider)}[/cyan]\n"
                f"RAG model: [cyan]{escape(self.settings.openai_answer_model)}[/cyan]\n"
                f"Verifikátor: [cyan]{escape(self.settings.typesafe_model)}[/cyan]\n"
                f"Web model: [cyan]{escape(self.settings.openai_web_model)}[/cyan]\n"
                f"Web fallback: [cyan]{'zapnutý' if self.settings.web_search_enabled else 'vypnutý'}[/cyan]\n"
                f"Tenant: [cyan]{escape(self.settings.tenant_id)}[/cyan]",
                title="Stav",
                border_style="cyan",
            )
        )

    def _help(self) -> None:
        """Bez vstupu zobrazí tabuľku dostupných chatových príkazov."""
        commands = [
            ("/add CESTA", "pridá, skopíruje a zaindexuje dokument"),
            ("/files", "zobrazí dokumenty a umožní interaktívne mazanie"),
            ("/delete ID", "odstráni súbor aj jeho vektory"),
            ("/sources", "znovu zobrazí zdroje poslednej odpovede"),
            ("/good [poznámka]", "označí poslednú odpoveď ako správnu"),
            ("/bad [poznámka]", "zaradí poslednú odpoveď na kontrolu"),
            ("/metrics", "zobrazí čas, tokeny, abstencie a feedback"),
            ("/status", "zobrazí aktívnu konfiguráciu"),
            ("/clear", "vyčistí terminál"),
            ("/exit", "ukončí aplikáciu"),
        ]
        table = Table(title="Príkazy", box=box.SIMPLE_HEAD, header_style="bold cyan")
        table.add_column("Príkaz", style="cyan", no_wrap=True)
        table.add_column("Popis")
        for command, description in commands:
            table.add_row(command, description)
        self.console.print(table)

    def _banner(self) -> None:
        """Bez vstupu vykreslí úvodný panel a krátku nápovedu."""
        self.console.print(
            Panel.fit(
                "[bold]DOC·AI[/bold]\n[dim]Odpovede z dokumentov s overenými citáciami[/dim]",
                border_style="cyan",
                padding=(1, 4),
            )
        )
        self.console.print("[dim]Napíšte otázku alebo /help. Ukončenie: /exit[/dim]\n")

    def _error(self, message: str) -> None:
        """Prijme text chyby a zobrazí bezpečný červený panel."""
        self.console.print(Panel(escape(message), title="Chyba", border_style="red"))


def build_cli(settings: Settings, console: Console) -> AssistantCLI:
    """Prijme nastavenia a konzolu; prepojí adaptéry a vráti hotové CLI."""
    gate = ModelCallGate()
    if settings.embedding_provider == "openai":
        embeddings = OpenAIEmbeddings(settings.openai_embedding_model, gate=gate)
    else:
        embeddings = LocalEmbeddings(settings.local_embedding_model)
    qdrant = QdrantVectorStore(
        client=QdrantClient(path=str(settings.qdrant_path)),
        collection=settings.qdrant_collection,
        embeddings=embeddings,
    )
    documents = DocumentService(
        documents=DocumentStore(settings.files_dir, settings.data_dir),
        vectors=qdrant,
        chunker=SectionAwareChunker(settings.chunk_size, settings.chunk_overlap),
        tenant_id=settings.tenant_id,
        ocr_mode=settings.ocr_mode,
    )
    rag_model = OpenAIAnswerModel(
        model=settings.openai_answer_model,
        gate=gate,
    )
    verifier = JevVerifier(
        model=settings.typesafe_model,
        min_confidence=settings.min_jev_confidence,
        gate=gate,
    )
    web = (
        OpenAIWebSearch(settings.openai_web_model, gate=gate)
        if settings.web_search_enabled
        else None
    )
    workflow = RAGWorkflow(
        retriever=qdrant,
        model=rag_model,
        verifier=verifier,
        web_search=web,
        tenant_id=settings.tenant_id,
        top_k=settings.top_k,
        max_retrieval_attempts=settings.max_retrieval_attempts,
        min_retrieval_score=settings.min_retrieval_score,
        min_answer_confidence=settings.min_answer_confidence,
    )
    return AssistantCLI(
        settings=settings,
        documents=documents,
        workflow=workflow,
        metrics=MetricsStore(settings.data_dir / "metrics.sqlite3"),
        console=console,
    )


def main() -> None:
    """Načíta konfiguráciu a spustí chat; pri chybe ukončí proces s kódom 1."""
    console = Console()
    try:
        settings = Settings.load()
        required_keys = ["OPENAI_API_KEY", "TYPESAFE_API_KEY"]
        missing = [name for name in required_keys if not os.getenv(name)]
        if missing:
            raise RuntimeError(
                "Chýbajú API kľúče: " + ", ".join(missing) + ". Skopírujte .env.example do .env."
            )
        build_cli(settings, console).run()
    except Exception as error:
        console.print(Panel(escape(str(error)), title="Spustenie zlyhalo", border_style="red"))
        raise SystemExit(1) from error
