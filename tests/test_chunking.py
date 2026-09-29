import json
from pathlib import Path

import pandas as pd
import pytest
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers

from src.chunking import CHUNK_DTYPES, ChunkingConfig, chunk_contracts, validate_chunks


@pytest.fixture
def tokenizer():
    vocab = {
        word: i
        for i, word in enumerate(
            [
                "[UNK]",
                "alpha",
                "beta",
                "gamma",
                "delta",
                ".",
                "cafe",
                "中",
                "文",
                "a",
                "##b",
                "##c",
                "##d",
                "##e",
                "b",
                "c",
                "d",
                "e",
            ]
        )
    }
    result = Tokenizer(models.WordPiece(vocab=vocab, unk_token="[UNK]"))
    result.normalizer = normalizers.BertNormalizer()
    result.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    return result


def contracts(text):
    return pd.DataFrame([{"contract": "example", "text": text}])


def test_budget_offsets_overlap_and_short_tail(tokenizer):
    source = contracts("  " + "alpha " * 13 + "\n")
    config = ChunkingConfig(max_tokens=6, overlap=2)
    chunks = chunk_contracts(source, "train", tokenizer=tokenizer, config=config)
    assert chunks.n_tokens.tolist() == [6, 6, 5]
    assert chunks.start.iloc[0] == 0
    assert chunks.end.iloc[-1] == len(source.text.iloc[0])
    assert chunks.start.iloc[1] < chunks.end.iloc[0]
    validate_chunks(source, chunks, "train", tokenizer=tokenizer, config=config)


@pytest.mark.parametrize("separator", ["\n\n", ". "])
def test_prefers_natural_boundary(tokenizer, separator):
    text = "alpha beta gamma delta" + separator + "alpha beta gamma delta alpha beta"
    config = ChunkingConfig(max_tokens=8, overlap=2)
    chunks = chunk_contracts(
        contracts(text), "train", tokenizer=tokenizer, config=config
    )
    assert chunks.text.iloc[0] == "alpha beta gamma delta" + separator


def test_paragraph_takes_precedence_over_later_sentence(tokenizer):
    text = "alpha beta gamma delta\n\nalpha. beta gamma delta alpha beta"
    chunks = chunk_contracts(
        contracts(text), "train", tokenizer=tokenizer, config=ChunkingConfig(8, 2)
    )
    assert chunks.text.iloc[0] == "alpha beta gamma delta\n\n"


@pytest.mark.parametrize("text", ["café 中文 alpha\n\nbeta " * 9, "abcde " * 9])
def test_unicode_and_wordpiece_slices(tokenizer, text):
    config = ChunkingConfig(7, 2)
    source = contracts(text)
    chunks = chunk_contracts(source, "train", tokenizer=tokenizer, config=config)
    validate_chunks(source, chunks, "train", tokenizer=tokenizer, config=config)
    assert all(row.text == text[row.start : row.end] for row in chunks.itertuples())


@pytest.mark.parametrize("text", ["", " \n\t "])
def test_empty_documents_have_typed_empty_output(tokenizer, text):
    source = contracts(text)
    chunks = chunk_contracts(source, "train", tokenizer=tokenizer)
    assert chunks.empty and list(chunks) == list(CHUNK_DTYPES)
    validate_chunks(source, chunks, "train", tokenizer=tokenizer)


def test_ids_and_order_are_stable_without_annotations(tokenizer):
    source = pd.DataFrame(
        [
            {"contract": "b", "text": "alpha beta"},
            {"contract": "a", "text": "gamma delta"},
        ]
    )
    first = chunk_contracts(source, "train", tokenizer=tokenizer)
    reordered = source.iloc[::-1].assign(annotations="ignored")
    pd.testing.assert_frame_equal(
        first, chunk_contracts(reordered, "train", tokenizer=tokenizer)
    )
    test_chunks = chunk_contracts(source, "test", tokenizer=tokenizer)
    assert set(first.chunk_id).isdisjoint(test_chunks.chunk_id)
    changed = chunk_contracts(source.assign(text="alpha"), "train", tokenizer=tokenizer)
    assert set(first.chunk_id).isdisjoint(changed.chunk_id)


@pytest.mark.parametrize(
    "max_tokens,overlap", [(0, 0), (511, 128), (4, 4), (4, -1), (4.0, 1)]
)
def test_invalid_configuration(max_tokens, overlap):
    with pytest.raises(ValueError):
        ChunkingConfig(max_tokens, overlap)


def test_rejects_duplicate_contracts_and_truncation(tokenizer):
    source = contracts("alpha")
    with pytest.raises(ValueError, match="unique"):
        chunk_contracts(pd.concat([source, source]), "train", tokenizer=tokenizer)
    tokenizer.enable_truncation(max_length=1)
    with pytest.raises(ValueError, match="truncation"):
        chunk_contracts(source, "train", tokenizer=tokenizer)


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("text", "changed", "source slice"),
        ("n_tokens", 100, "token count"),
        ("split", "test", "different split"),
        ("end", 1000, "offsets"),
        ("chunk_id", "wrong", "Chunk ID"),
    ],
)
def test_validation_rejects_corrupt_rows(tokenizer, field, value, message):
    source = contracts("alpha beta")
    chunks = chunk_contracts(source, "train", tokenizer=tokenizer)
    chunks.loc[0, field] = value
    with pytest.raises(ValueError, match=message):
        validate_chunks(source, chunks, "train", tokenizer=tokenizer)


def test_validation_rejects_missing_tail(tokenizer):
    source = contracts("alpha " * 10)
    config = ChunkingConfig(4, 1)
    chunks = chunk_contracts(source, "train", tokenizer=tokenizer, config=config)
    with pytest.raises(ValueError, match="document ending"):
        validate_chunks(
            source, chunks.iloc[:-1], "train", tokenizer=tokenizer, config=config
        )


def test_export_round_trip(tokenizer, tmp_path):
    from scripts.export_chunks import export_split

    text = "alpha beta gamma"
    raw = {
        "data": [
            {
                "title": "example",
                "paragraphs": [
                    {
                        "context": text,
                        "qas": [
                            {
                                "id": "example__Governing Law_0",
                                "question": "Governing law?",
                                "answers": [{"text": "beta", "answer_start": 6}],
                                "is_impossible": False,
                            }
                        ],
                    }
                ],
            }
        ]
    }
    (tmp_path / "train_separate_questions.json").write_text(json.dumps(raw))
    result = export_split(
        "train", tmp_path / "out", tokenizer=tokenizer, raw_dir=tmp_path
    )
    restored = pd.read_parquet(tmp_path / "out/train/chunks.parquet")
    assert restored.text.tolist() == [text]
    assert result["fully_contained_priority_spans"] == 1
    assert result["labeled"] is False
    saved = json.loads((tmp_path / "out/train/metadata.json").read_text())
    assert saved == result


def test_shared_example_has_full_partial_and_negative_cases():
    sample = json.loads(
        (Path(__file__).parent / "fixtures/chunking_example.json").read_text()
    )
    text = sample["contracts"][0]["text"]
    chunks, spans = sample["chunks"], sample["spans"]
    assert all(c["text"] == text[c["start"] : c["end"]] for c in chunks)
    assert all(s["text"] == text[s["start"] : s["end"]] for s in spans)

    def overlaps(c, s):
        return c["start"] < s["end"] and s["start"] < c["end"]

    def contains(c, s):
        return c["start"] <= s["start"] and s["end"] <= c["end"]

    assert any(contains(c, s) for c in chunks for s in spans)
    assert any(overlaps(c, s) and not contains(c, s) for c in chunks for s in spans)
    assert any(not any(overlaps(c, s) for s in spans) for c in chunks)


def test_retokenizes_wordpiece_at_overlap_boundary():
    words = ["[UNK]", "abcd", "##efg", "e", "##f", "##g", "alpha"]
    tokenizer = Tokenizer(
        models.WordPiece(vocab={w: i for i, w in enumerate(words)}, unk_token="[UNK]")
    )
    tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    source = contracts("alpha alpha abcdefg alpha alpha alpha alpha alpha")
    config = ChunkingConfig(5, 2)
    chunks = chunk_contracts(source, "train", tokenizer=tokenizer, config=config)
    assert chunks.text.iloc[1].startswith("efg")
    assert chunks.n_tokens.iloc[1] == 5
    validate_chunks(source, chunks, "train", tokenizer=tokenizer, config=config)


def test_zero_overlap_has_no_gaps(tokenizer):
    source = contracts("alpha " * 11)
    config = ChunkingConfig(4, 0)
    chunks = chunk_contracts(source, "train", tokenizer=tokenizer, config=config)
    assert chunks.text.str.cat() == source.text.iloc[0]
    validate_chunks(source, chunks, "train", tokenizer=tokenizer, config=config)
