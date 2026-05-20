from __future__ import annotations

from ai_advocate.infrastructure.config import Settings
from ai_advocate.infrastructure.container import Container


def main() -> None:
    settings = Settings()
    container = Container(settings)
    result = container.ingest_use_case().run()
    print(
        f"Indexed {result.documents_indexed} documents / "
        f"{result.chunks_indexed} chunks into Chroma."
    )


if __name__ == "__main__":
    main()
