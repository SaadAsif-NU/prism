"""Enable ``python -m prism`` to launch the interactive SQL shell."""

from prism.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
