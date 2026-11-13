from types import SimpleNamespace

import pytest

import rag_engine
from rag_engine import (
    BM25Index,
    Chunk,
    SearchResult,
    build_user_prompt,
    extract_chunks,
    extractive_answer,
)


def test_retrieval_ranks_matching_passage_and_builds_citation_prompt():
    chunks = [
        Chunk("ai-report.pdf", 4, 5, "AI tools reduce repetitive coding tasks."),
        Chunk("jobs.pdf", 1, 1, "Employment trends changed across several sectors."),
    ]
    results = BM25Index(chunks).search("coding tasks", limit=1)

    assert len(results) == 1
    assert results[0].chunk.source == "ai-report.pdf"
    prompt = build_user_prompt("What changes?", results)
    assert "[Source: ai-report.pdf, pp. 4-5]" in prompt
    assert "What changes?" in prompt


def test_chunking_keeps_overlap_and_validates_settings(monkeypatch):
    text = " ".join(f"word{number}" for number in range(240))
    monkeypatch.setattr(
        rag_engine,
        "PdfReader",
        lambda _: SimpleNamespace(pages=[SimpleNamespace(extract_text=lambda: text)]),
    )
    chunks = extract_chunks([("long.pdf", b"test")], chunk_tokens=100, overlap_tokens=20)

    assert len(chunks) > 1
    assert set(chunks[0].text.split()[-15:]) & set(chunks[1].text.split()[:20])
    with pytest.raises(ValueError, match="smaller than the chunk size"):
        extract_chunks([], chunk_tokens=100, overlap_tokens=100)


def test_empty_query_returns_no_results():
    index = BM25Index([Chunk("source.pdf", 1, 1, "A searchable sentence.")])

    assert index.search("   ") == []


def test_extractive_answer_returns_matching_sentence_with_citation():
    results = [
        SearchResult(
            Chunk(
                "enterprise-ai.pdf",
                5,
                5,
                "ChatGPT Enterprise users save an average of 40 minutes per active day.",
            ),
            3.0,
        )
    ]

    answer = extractive_answer(
        "How many minutes per active day do ChatGPT Enterprise users save?", results
    )

    assert answer == (
        "ChatGPT Enterprise users save an average of 40 minutes per active day. "
        "[enterprise-ai.pdf, p. 5]"
    )


def test_extractive_answer_abstains_without_matching_evidence_and_limits_sentences():
    assert extractive_answer("How many developers use AI daily?", []) is None

    matching = [
        SearchResult(
            Chunk(
                "report.pdf",
                1,
                2,
                "AI users save minutes. AI users save hours. AI users save time. AI users save work.",
            ),
            1.0,
        )
    ]
    answer = extractive_answer("AI users save", matching)

    assert answer is not None
    assert answer.count("[report.pdf") == 3


def test_extractive_answer_deduplicates_overlapping_chunks():
    results = [
        SearchResult(
            Chunk("report.pdf", 5, 7, "Enterprise users save 40 minutes per active day."),
            2.0,
        ),
        SearchResult(
            Chunk("report.pdf", 7, 9, "Enterprise users save 40 minutes per active day."),
            1.5,
        ),
    ]

    answer = extractive_answer("How many minutes per active day do enterprise users save?", results)

    assert answer == "Enterprise users save 40 minutes per active day. [report.pdf, pp. 5-7]"


def test_extractive_answer_deduplicates_repeated_numeric_claims():
    results = [
        SearchResult(
            Chunk("report.pdf", 5, 6, "Enterprise users save 40-60 minutes per active day."),
            2.0,
        ),
        SearchResult(
            Chunk("report.pdf", 7, 8, "Enterprise users report saving 40-60 minutes each active day."),
            1.5,
        ),
    ]

    answer = extractive_answer("How many minutes per active day do enterprise users save?", results)

    assert answer == "Enterprise users save 40-60 minutes per active day. [report.pdf, pp. 5-6]"


def test_extractive_answer_accepts_weak_term_overlap_with_citation():
    results = [
        SearchResult(
            Chunk("report.pdf", 1, 1, "Enterprise users describe the workplace as collaborative."),
            1.0,
        )
    ]

    answer = extractive_answer("How many minutes do enterprise users save per day?", results)

    assert answer == "Enterprise users describe the workplace as collaborative. [report.pdf, p. 1]"


def test_extractive_answer_falls_back_to_top_passage_for_broad_question():
    results = [
        SearchResult(
            Chunk("report.pdf", 2, 3, "AI adoption reshaped team workflows. The report compares several sectors."),
            2.0,
        )
    ]

    answer = extractive_answer("What are the findings?", results)

    assert answer == "AI adoption reshaped team workflows. [report.pdf, pp. 2-3]"


def test_broad_findings_question_prefers_source_bullet_points():
    results = [
        SearchResult(
            Chunk(
                "report.pdf",
                2,
                3,
                "Key findings\n"
                "• The survey found that early-career workers report more concern about displacement.\n"
                "• Respondents report productivity gains from taking on new tasks.\n"
                "• The largest speedups correspond with higher concern about job displacement.",
            ),
            2.0,
        )
    ]

    answer = extractive_answer("What are the main findings?", results)

    assert answer is not None
    assert "Key findings" not in answer
    assert answer.startswith(
        "The survey found that early-career workers report more concern about displacement."
    )
    assert answer.count("[report.pdf") == 3


def test_extractive_answer_keeps_daily_usage_percentage_sentence():
    results = [
        SearchResult(
            Chunk(
                "state-of-code-developer-survey-report.pdf",
                5,
                5,
                "72% of developers who have tried AI use it every day. "
                "72% of developers who have tried AI coding tools now use them every day. "
                "n=979 2027 2026 2025 2024 2023 65% 55% 42% 19% 6% 72% of developers who have tried AI use it every day.",
            ),
            10.0,
        )
    ]

    answer = extractive_answer(
        "In State of Code Developer Survey Report 2026, what percentage of developers who have tried AI coding tools use them every day?",
        results,
    )

    assert answer is not None
    assert "72%" in answer
    assert "every day" in answer.lower()


def test_open_ended_question_selects_takeaway_not_chart_or_survey_question():
    results = [
        SearchResult(
            Chunk(
                "report.pdf",
                27,
                29,
                "n=799 Review AI-generated code 7% 49% 13% 22% 9%. "
                "The takeaway While AI is changing how developers work, it has not reduced toil. "
                "How do you believe AI changed your time to complete issues?",
            ),
            3.0,
        )
    ]

    answer = extractive_answer("How has AI changed work?", results)

    assert answer == (
        "While AI is changing how developers work, it has not reduced toil. "
        "[report.pdf, pp. 27-29]"
    )