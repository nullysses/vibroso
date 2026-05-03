import torch
import pytest

from toy_llm.dataset import (
    TextDataset,
    html_to_text,
    load_wikipedia_titles_corpus,
    parse_title_list,
    parse_url_list,
)
from toy_llm.tokenizer import CharTokenizer


def test_train_validation_split_sizes():
    text = "0123456789" * 10
    tokenizer = CharTokenizer.from_text(text)
    dataset = TextDataset.from_text(
        text,
        tokenizer,
        train_split=0.8,
        block_size=4,
        batch_size=2,
        device=torch.device("cpu"),
    )
    assert len(dataset.train_data) == 80
    assert len(dataset.val_data) == 20


def test_batch_shapes_and_targets_are_shifted():
    text = "abcdefghijklmnopqrstuvwxyz" * 5
    tokenizer = CharTokenizer.from_text(text)
    dataset = TextDataset.from_text(
        text,
        tokenizer,
        train_split=0.9,
        block_size=8,
        batch_size=4,
        device=torch.device("cpu"),
    )
    x, y = dataset.get_batch("train")
    assert x.shape == (4, 8)
    assert y.shape == (4, 8)
    assert torch.equal(x[:, 1:], y[:, :-1])


def test_parse_url_list_ignores_comments_and_blank_lines(tmp_path):
    links = tmp_path / "links.txt"
    links.write_text(
        "\n# comment\nhttps://example.com/a\n\nhttps://example.com/b\n",
        encoding="utf-8",
    )
    assert parse_url_list(links) == ["https://example.com/a", "https://example.com/b"]


def test_parse_url_list_rejects_non_urls(tmp_path):
    links = tmp_path / "links.txt"
    links.write_text("not-a-url\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Expected URL"):
        parse_url_list(links)


def test_parse_title_list_keeps_wikipedia_titles(tmp_path):
    titles = tmp_path / "titles.txt"
    titles.write_text(
        "\n# comment\nArtificial intelligence\nTransformer (deep learning architecture)\n",
        encoding="utf-8",
    )
    assert parse_title_list(titles) == [
        "Artificial intelligence",
        "Transformer (deep learning architecture)",
    ]


def test_html_to_text_skips_script_and_style_content():
    html = """
    <html>
      <head><style>.hidden { color: red; }</style></head>
      <body>
        <h1>Title</h1>
        <script>alert("nope")</script>
        <p>Readable <b>article</b> text.</p>
      </body>
    </html>
    """
    text = html_to_text(html)
    assert "Title" in text
    assert "Readable article text." in text
    assert "alert" not in text
    assert "hidden" not in text


class _FakePage:
    def __init__(self, text: str, exists: bool = True):
        self.text = text
        self._exists = exists

    def exists(self):
        return self._exists


class _FakeWiki:
    def __init__(self, pages):
        self.pages = pages

    def page(self, title):
        return self.pages[title]


def test_load_wikipedia_titles_corpus_uses_page_text(tmp_path):
    titles = tmp_path / "titles.txt"
    titles.write_text("Artificial intelligence\nLanguage model\n", encoding="utf-8")
    cache = tmp_path / "wiki_cache.txt"
    wiki = _FakeWiki(
        {
            "Artificial intelligence": _FakePage("AI page text"),
            "Language model": _FakePage("LM page text"),
        }
    )

    corpus = load_wikipedia_titles_corpus(
        titles,
        language="en",
        user_agent="test",
        cache_path=cache,
        wiki=wiki,
    )

    assert corpus == "AI page text\n\nLM page text"
    assert cache.read_text(encoding="utf-8") == corpus


def test_load_wikipedia_titles_corpus_rejects_missing_pages(tmp_path):
    titles = tmp_path / "titles.txt"
    titles.write_text("Missing page\n", encoding="utf-8")
    wiki = _FakeWiki({"Missing page": _FakePage("", exists=False)})

    with pytest.raises(ValueError, match="missing or empty"):
        load_wikipedia_titles_corpus(titles, language="en", user_agent="test", wiki=wiki)
