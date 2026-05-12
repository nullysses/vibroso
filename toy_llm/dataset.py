from __future__ import annotations

import json
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import torch

from toy_llm.config import TrainConfig
from toy_llm.tokenizer import Tokenizer


def load_text(path: str | Path) -> str:
    text_path = Path(path)
    if not text_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {text_path}")
    if not text_path.is_file():
        raise ValueError(f"Dataset path is not a file: {text_path}")
    try:
        text = text_path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"Dataset is not valid UTF-8: {text_path}") from exc
    if not text:
        raise ValueError(f"Dataset file is empty: {text_path}")
    return text


def parse_url_list(path: str | Path) -> list[str]:
    text = load_text(path)
    urls: list[str] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if not stripped.startswith(("http://", "https://")):
            raise ValueError(f"Expected URL on line {line_no} of {path}: {stripped!r}")
        urls.append(stripped)
    if not urls:
        raise ValueError(f"URL list contains no URLs: {path}")
    return urls


def parse_title_list(path: str | Path) -> list[str]:
    text = load_text(path)
    titles: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        titles.append(stripped)
    if not titles:
        raise ValueError(f"Title list contains no Wikipedia page titles: {path}")
    return titles


def _require_non_empty_string(row: dict[str, Any], key: str, line_no: int) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"Invalid instruction JSONL row at line {line_no}: "
            f"field {key!r} must be a non-empty string"
        )
    return value.strip()


def format_instruction_example(
    instruction: str,
    response: str,
    context: str | None = None,
) -> str:
    instruction = instruction.strip()
    response = response.strip()
    context = context.strip() if context else ""
    if context:
        return (
            "<|user|>\n"
            f"{instruction}\n\n"
            "Context:\n"
            f"{context}\n"
            "<|assistant|>\n"
            f"{response}\n"
            "<|end|>\n"
        )
    return (
        "<|user|>\n"
        f"{instruction}\n"
        "<|assistant|>\n"
        f"{response}\n"
        "<|end|>\n"
    )


def load_instruction_jsonl(path: str | Path) -> str:
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Instruction JSONL file not found: {dataset_path}")
    if not dataset_path.is_file():
        raise ValueError(f"Instruction JSONL path is not a file: {dataset_path}")

    examples: list[str] = []
    with dataset_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {dataset_path}:{line_no}: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise ValueError(
                    f"Invalid instruction JSONL row at line {line_no}: expected a JSON object"
                )

            instruction = _require_non_empty_string(row, "instruction", line_no)
            response = _require_non_empty_string(row, "response", line_no)
            context_value = row.get("context")
            if context_value is not None and not isinstance(context_value, str):
                raise ValueError(
                    f"Invalid instruction JSONL row at line {line_no}: "
                    "field 'context' must be a string when present"
                )
            examples.append(
                format_instruction_example(
                    instruction=instruction,
                    response=response,
                    context=context_value,
                )
            )

    if not examples:
        raise ValueError(f"Instruction JSONL file contains no examples: {dataset_path}")
    return "\n".join(examples)


class _ReadableHTMLParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript", "svg"}:
            self._skip_depth += 1
        if tag in {"p", "br", "div", "section", "article", "li", "h1", "h2", "h3"}:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript", "svg"} and self._skip_depth:
            self._skip_depth -= 1
        if tag in {"p", "li", "h1", "h2", "h3"}:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        stripped = " ".join(data.split())
        if stripped:
            self.parts.append(stripped + " ")

    def get_text(self) -> str:
        lines = [" ".join(line.split()) for line in "".join(self.parts).splitlines()]
        return "\n".join(line for line in lines if line)


def html_to_text(html: str) -> str:
    parser = _ReadableHTMLParser()
    parser.feed(html)
    parser.close()
    text = parser.get_text()
    if not text:
        raise ValueError("Fetched HTML did not contain readable text")
    return text


def fetch_url_text(url: str, timeout: float, user_agent: str) -> str:
    request = Request(url, headers={"User-Agent": user_agent})
    try:
        with urlopen(request, timeout=timeout) as response:
            content_type = response.headers.get("Content-Type", "")
            charset = response.headers.get_content_charset() or "utf-8"
            body = response.read()
    except HTTPError as exc:
        raise ValueError(f"Failed to fetch {url}: HTTP {exc.code}") from exc
    except URLError as exc:
        raise ValueError(f"Failed to fetch {url}: {exc.reason}") from exc

    text = body.decode(charset, errors="replace")
    if "html" in content_type.lower() or text.lstrip().startswith(("<!doctype", "<html", "<")):
        return html_to_text(text)
    return text


def load_url_list_corpus(
    path: str | Path,
    timeout: float,
    user_agent: str,
    cache_path: str | Path | None = None,
) -> str:
    urls = parse_url_list(path)
    documents: list[str] = []
    for i, url in enumerate(urls, start=1):
        print(f"fetching {i}/{len(urls)}: {url}")
        documents.append(fetch_url_text(url, timeout=timeout, user_agent=user_agent))
    corpus = "\n\n".join(documents)
    if not corpus.strip():
        raise ValueError("Fetched URL corpus is empty")
    if cache_path is not None:
        cache_file = Path(cache_path)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(corpus, encoding="utf-8")
        print(f"cached fetched corpus: {cache_file}")
    return corpus


def _make_wikipedia_client(language: str, user_agent: str):
    try:
        import wikipediaapi
    except ImportError as exc:
        raise ImportError(
            "Wikipedia title imports require the 'wikipedia-api' package. "
            "Install it with: python -m pip install wikipedia-api"
        ) from exc
    return wikipediaapi.Wikipedia(language=language, user_agent=user_agent)


def load_wikipedia_titles_corpus(
    path: str | Path,
    language: str,
    user_agent: str,
    cache_path: str | Path | None = None,
    wiki=None,
) -> str:
    titles = parse_title_list(path)
    wikipedia = wiki if wiki is not None else _make_wikipedia_client(language, user_agent)
    documents: list[str] = []
    missing_titles: list[str] = []
    for i, title in enumerate(titles, start=1):
        print(f"fetching Wikipedia page {i}/{len(titles)}: {title}")
        page = wikipedia.page(title)
        if hasattr(page, "exists") and not page.exists():
            missing_titles.append(title)
            continue
        text = getattr(page, "text", "")
        if text.strip():
            documents.append(text)
        else:
            missing_titles.append(title)

    if missing_titles:
        missing = ", ".join(repr(title) for title in missing_titles)
        raise ValueError(f"Wikipedia pages were missing or empty: {missing}")

    corpus = "\n\n".join(documents)
    if not corpus.strip():
        raise ValueError("Fetched Wikipedia corpus is empty")
    if cache_path is not None:
        cache_file = Path(cache_path)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(corpus, encoding="utf-8")
        print(f"cached fetched corpus: {cache_file}")
    return corpus


def load_corpus(config: TrainConfig) -> str:
    if config.dataset_kind == "text":
        return load_text(config.dataset_path)
    if config.dataset_kind == "instruction_jsonl":
        return load_instruction_jsonl(config.dataset_path)
    if config.corpus_cache_path:
        cache_file = Path(config.corpus_cache_path)
        if cache_file.exists():
            print(f"loading cached corpus: {cache_file}")
            return load_text(cache_file)
    if config.dataset_kind == "wikipedia_titles":
        return load_wikipedia_titles_corpus(
            config.dataset_path,
            language=config.wikipedia_language,
            user_agent=config.user_agent,
            cache_path=config.corpus_cache_path,
        )
    return load_url_list_corpus(
        config.dataset_path,
        timeout=config.fetch_timeout,
        user_agent=config.user_agent,
        cache_path=config.corpus_cache_path,
    )


@dataclass
class TextDataset:
    train_data: torch.Tensor
    val_data: torch.Tensor
    block_size: int
    batch_size: int
    device: torch.device

    @classmethod
    def from_text(
        cls,
        text: str,
        tokenizer: Tokenizer,
        train_split: float,
        block_size: int,
        batch_size: int,
        device: torch.device,
    ) -> "TextDataset":
        encoded = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        minimum_tokens = block_size + 2
        if encoded.numel() < minimum_tokens:
            raise ValueError(
                f"Dataset has {encoded.numel()} tokens; need at least {minimum_tokens} "
                "for the configured block_size"
            )
        n = int(train_split * len(encoded))
        train_data = encoded[:n]
        val_data = encoded[n:]
        if len(train_data) <= block_size:
            raise ValueError("Training split is too small for block_size")
        if len(val_data) <= block_size:
            raise ValueError("Validation split is too small for block_size")
        return cls(train_data, val_data, block_size, batch_size, device)

    def get_batch(self, split: str) -> tuple[torch.Tensor, torch.Tensor]:
        if split not in {"train", "val"}:
            raise ValueError("split must be 'train' or 'val'")
        data = self.train_data if split == "train" else self.val_data
        max_start = len(data) - self.block_size
        ix = torch.randint(max_start, (self.batch_size,))
        x = torch.stack([data[i : i + self.block_size] for i in ix])
        y = torch.stack([data[i + 1 : i + self.block_size + 1] for i in ix])
        return x.to(self.device), y.to(self.device)
