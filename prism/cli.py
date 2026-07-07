"""The ``prism`` command line: an interactive SQL shell over CSV data.

Run ``prism data/employees.csv`` to load a file and drop into a REPL, or pass
``-c "SELECT ..."`` to run a single statement and exit. Inside the shell,
statements end with ``;`` and a handful of dot-commands introspect and load
data:

    .tables            list registered tables
    .schema [table]    show column types
    .load PATH [name]  load a CSV file
    .timing on|off     toggle query timing
    .help              show this help
    .quit              exit

``EXPLAIN <query>`` prints the optimized plan instead of running the query.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from prism import __version__
from prism.engine import Database
from prism.format import render_schema, render_table
from prism.plan import PlanError
from prism.sql.lexer import LexError
from prism.sql.parser import ParseError

_HELP = """Commands:
  .tables            list registered tables
  .schema [table]    show column types for one or all tables
  .load PATH [name]  load a CSV file (name defaults to the file stem)
  .timing on|off     toggle query timing
  .help              show this help
  .quit              exit

End SQL statements with a semicolon. Prefix a query with EXPLAIN to see its plan.
"""


class Shell:
    """A small read-eval-print loop around a :class:`Database`."""

    def __init__(self, db: Database, timing: bool = True, out=None) -> None:  # type: ignore[no-untyped-def]
        self.db = db
        self.timing = timing
        self.out = out if out is not None else sys.stdout

    def run_statement(self, statement: str) -> None:
        """Execute one statement (SQL or dot-command) and print the result."""
        text = statement.strip().rstrip(";").strip()
        if not text:
            return
        if text.startswith("."):
            self._run_command(text)
            return
        if text.upper().startswith("EXPLAIN "):
            self._print(self.db.explain(text[len("EXPLAIN ") :]))
            return
        self._run_query(text)

    def _run_query(self, query: str) -> None:
        start = time.perf_counter()
        result = self.db.sql(query)
        elapsed = time.perf_counter() - start
        self._print(render_table(result))
        if self.timing:
            self._print(f"({elapsed * 1000:.1f} ms)")

    def _run_command(self, text: str) -> None:
        parts = text.split()
        command = parts[0].lower()
        args = parts[1:]

        if command in (".quit", ".exit"):
            raise _Exit
        if command == ".help":
            self._print(_HELP)
        elif command == ".tables":
            names = self.db.catalog.names()
            self._print("\n".join(sorted(names)) if names else "(no tables)")
        elif command == ".schema":
            self._show_schema(args)
        elif command == ".load":
            self._load(args)
        elif command == ".timing":
            self.timing = not args or args[0].lower() == "on"
            self._print(f"timing {'on' if self.timing else 'off'}")
        else:
            self._print(f"unknown command {command!r}; try .help")

    def _show_schema(self, args: list[str]) -> None:
        names = args if args else sorted(self.db.catalog.names())
        if not names:
            self._print("(no tables)")
            return
        blocks = []
        for name in names:
            if name not in self.db.catalog:
                blocks.append(f"unknown table {name!r}")
            else:
                blocks.append(render_schema(name, self.db.catalog.get(name)))
        self._print("\n".join(blocks))

    def _load(self, args: list[str]) -> None:
        if not args:
            self._print("usage: .load PATH [name]")
            return
        path = args[0]
        name = args[1] if len(args) > 1 else None
        table = self.db.load_csv(path, name=name)
        registered = name if name is not None else Path(path).stem
        self._print(f"loaded {registered!r} ({table.num_rows} rows, {table.num_columns} columns)")

    def _print(self, text: str) -> None:
        print(text, file=self.out)


class _Exit(Exception):
    """Signals a clean exit from the shell loop."""


def _repl(db: Database, timing: bool) -> int:
    shell = Shell(db, timing=timing)
    print(f"prism {__version__} - interactive SQL shell. Type .help or .quit.")
    buffer = ""
    while True:
        prompt = "prism> " if not buffer else "  ...> "
        try:
            line = input(prompt)
        except EOFError:
            print()
            return 0
        except KeyboardInterrupt:
            print("\n(interrupted)")
            buffer = ""
            continue

        buffer = f"{buffer}\n{line}" if buffer else line
        if line.strip().startswith(".") or buffer.rstrip().endswith(";"):
            try:
                shell.run_statement(buffer)
            except _Exit:
                return 0
            except (ParseError, LexError, PlanError, KeyError, TypeError, ValueError) as exc:
                print(f"error: {exc}", file=sys.stderr)
            except FileNotFoundError as exc:
                print(f"error: {exc}", file=sys.stderr)
            buffer = ""


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``prism`` command."""
    parser = argparse.ArgumentParser(
        prog="prism", description="A columnar SQL query engine over CSV data."
    )
    parser.add_argument("csv", nargs="*", help="CSV files to load (named by file stem)")
    parser.add_argument("-c", "--command", help="run a single SQL statement and exit")
    parser.add_argument("--no-timing", action="store_true", help="do not print query timing")
    parser.add_argument("--version", action="version", version=f"prism {__version__}")
    args = parser.parse_args(argv)

    db = Database()
    for path in args.csv:
        db.load_csv(path)

    if args.command:
        shell = Shell(db, timing=not args.no_timing)
        try:
            shell.run_statement(args.command)
        except (ParseError, LexError, PlanError, KeyError, TypeError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        return 0

    return _repl(db, timing=not args.no_timing)


if __name__ == "__main__":
    sys.exit(main())
