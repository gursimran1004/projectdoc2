from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from transformers import pipeline
except Exception:
    pipeline = None


@dataclass
class QAResult:
    answer: str
    relevant_passage: str
    confidence: float
    source_label: str
    top_matches: List[tuple[str, str, float]]


@dataclass
class MCQResult:
    selected_option: str
    confidence: float
    explanation: str


def split_section_into_chunks(
    text: str, source_label: str, chunk_size: int = 700, overlap: int = 120
) -> List[tuple[str, str]]:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return []

    chunks: List[tuple[str, str]] = []
    start = 0
    while start < len(cleaned):
        end = min(start + chunk_size, len(cleaned))
        chunks.append((cleaned[start:end], source_label))
        if end == len(cleaned):
            break
        start = end - overlap
    return chunks


def build_chunks(
    document_text: str,
    sections: List[tuple[str, str]] | None,
    chunk_size: int,
    chunk_overlap: int,
) -> List[tuple[str, str]]:
    if sections:
        all_chunks: List[tuple[str, str]] = []
        for section_text, section_label in sections:
            all_chunks.extend(
                split_section_into_chunks(
                    section_text,
                    section_label,
                    chunk_size=chunk_size,
                    overlap=chunk_overlap,
                )
            )
        if all_chunks:
            return all_chunks
    return split_section_into_chunks(
        document_text,
        "Document",
        chunk_size=chunk_size,
        overlap=chunk_overlap,
    )


def retrieve_top_chunks(
    query: str, chunks_with_source: List[tuple[str, str]], top_k: int = 3
) -> List[tuple[str, str, float]]:
    """Retrieve top K most relevant chunks using TF-IDF similarity."""
    chunks = [chunk for chunk, _ in chunks_with_source]
    if not chunks:
        return []
    if len(chunks) == 1:
        only_chunk, only_source = chunks_with_source[0]
        return [(only_chunk, only_source, 1.0)]

    try:
        vectorizer = TfidfVectorizer(stop_words="english", max_features=500)
        docs = chunks + [query]
        tfidf = vectorizer.fit_transform(docs)
        chunk_vectors = tfidf[:-1]
        query_vector = tfidf[-1]
        similarities = cosine_similarity(query_vector, chunk_vectors).flatten()
        
        # Ensure we don't request more than available
        k = min(top_k, len(chunks))
        top_indices = np.argsort(similarities)[::-1][:k]
        
        results = [
            (chunks_with_source[idx][0], chunks_with_source[idx][1], float(similarities[idx]))
            for idx in top_indices
            if float(similarities[idx]) > 0.0  # Filter out zero scores
        ]
        return results if results else [(chunks[0], chunks_with_source[0][1], 0.1)]
    except Exception as e:
        print(f"Error in retrieve_top_chunks: {e}")
        # Fallback: return first chunk
        return [(chunks[0], chunks_with_source[0][1], 0.1)]


class DocumentQA:
    def __init__(self) -> None:
        self._qa_pipeline = None
        self._sum_pipeline = None
        self._gen_pipeline = None

    def _get_qa_pipeline(self):
        if self._qa_pipeline is None and pipeline is not None:
            try:
                self._qa_pipeline = pipeline(
                    "question-answering", model="distilbert-base-cased-distilled-squad"
                )
            except Exception as e:
                print(f"QA Pipeline error: {e}")
                self._qa_pipeline = False
        return self._qa_pipeline

    def _get_sum_pipeline(self):
        if self._sum_pipeline is None and pipeline is not None:
            try:
                self._sum_pipeline = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
            except Exception as e:
                print(f"Summarization Pipeline error: {e}")
                self._sum_pipeline = False
        return self._sum_pipeline

    def _get_gen_pipeline(self):
        if self._gen_pipeline is None and pipeline is not None:
            try:
                self._gen_pipeline = pipeline("text2text-generation", model="google/flan-t5-base")
            except Exception as e:
                print(f"Generation Pipeline error: {e}")
                self._gen_pipeline = False
        return self._gen_pipeline

    @staticmethod
    def _clean_answer_text(text: str) -> str:
        """Clean and normalize answer text."""
        text = re.sub(r"\s+", " ", text).strip()
        # Join character-fragmented words such as "c o n s i s t s".
        text = re.sub(r"\b(?:[A-Za-z]\s+){3,}[A-Za-z]\b", lambda m: m.group(0).replace(" ", ""), text)
        text = DocumentQA._repair_split_words(text)
        # Remove common OCR punctuation spacing artifacts.
        text = re.sub(r"\s+([,.;:!?])", r"\1", text)
        return text

    @staticmethod
    def _looks_noisy(text: str) -> bool:
        """Check if answer text appears noisy/corrupted."""
        if not text.strip():
            return True
        words = text.split()
        if not words:
            return True
        short_tokens = sum(1 for w in words if len(w) <= 2)
        uppercase_tokens = sum(1 for w in words if w.isupper() and len(w) > 1)
        return (short_tokens / len(words)) > 0.35 or uppercase_tokens > 6

    @staticmethod
    def _clean_context_text(text: str) -> str:
        """Clean context passage text."""
        text = re.sub(r"\s+", " ", text).strip()
        text = re.sub(r"\b(?:[A-Za-z]\s+){3,}[A-Za-z]\b", lambda m: m.group(0).replace(" ", ""), text)
        return DocumentQA._repair_split_words(text)

    @staticmethod
    def _repair_split_words(text: str) -> str:
        """Repair OCR-split words like 'databa se' -> 'database'."""
        stop = {"in", "on", "of", "to", "is", "as", "at", "by", "an", "or", "if", "be", "we", "he", "it"}
        words = text.split(" ")
        merged: List[str] = []
        i = 0
        while i < len(words):
            if i < len(words) - 1:
                a = words[i]
                b = words[i + 1]
                if (
                    a.isalpha()
                    and b.isalpha()
                    and a.lower() not in stop
                    and b.lower() not in stop
                    and a.islower()
                    and b.islower()
                    and len(a) >= 3
                    and len(b) <= 3
                    and (len(a) + len(b)) >= 6
                ):
                    merged.append(a + b)
                    i += 2
                    continue
            merged.append(words[i])
            i += 1
        repaired = " ".join(merged)

        # Merge runs of tiny tokens e.g., "pro ce du re" -> "procedure".
        tokens = repaired.split(" ")
        out: List[str] = []
        k = 0
        while k < len(tokens):
            if tokens[k].isalpha() and len(tokens[k]) <= 2 and tokens[k].islower():
                m = k
                run: List[str] = []
                while m < len(tokens) and tokens[m].isalpha() and len(tokens[m]) <= 2 and tokens[m].islower():
                    run.append(tokens[m])
                    m += 1
                if len(run) >= 3 and sum(len(x) for x in run) >= 6:
                    out.append("".join(run))
                    k = m
                    continue
            out.append(tokens[k])
            k += 1
        return " ".join(out)

    def answer_query(
        self,
        query: str,
        document_text: str,
        sections: List[tuple[str, str]] | None = None,
        chunk_size: int = 700,
        chunk_overlap: int = 120,
        top_k: int = 3,
    ) -> QAResult:
        """Answer a query based on document content."""
        # Validate input
        if not query.strip() or not document_text.strip():
            return QAResult(
                answer="Please upload a document and ask a valid question.",
                relevant_passage="",
                confidence=0.0,
                source_label="N/A",
                top_matches=[],
            )
        
        chunks_with_source = build_chunks(
            document_text,
            sections,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        
        if not chunks_with_source:
            return QAResult(
                answer="Could not process the document. Please try again.",
                relevant_passage="",
                confidence=0.0,
                source_label="N/A",
                top_matches=[],
            )
        
        top_hits = retrieve_top_chunks(query, chunks_with_source, top_k=top_k)
        if not top_hits:
            return QAResult(
                answer="No relevant content found in the document.",
                relevant_passage="",
                confidence=0.0,
                source_label="N/A",
                top_matches=[],
            )

        best_chunk, source_label, sim_score = top_hits[0]
        
        # Try QA pipeline first
        qa_pipe = self._get_qa_pipeline()
        if qa_pipe and qa_pipe is not False:
            try:
                result = qa_pipe(question=query, context=best_chunk)
                answer = (result.get("answer") or "").strip()
                score = float(result.get("score", 0.0))
                if answer and not self._looks_noisy(answer):
                    return QAResult(
                        answer=self._clean_answer_text(answer),
                        relevant_passage=self._clean_context_text(best_chunk),
                        confidence=max(0.5, score),
                        source_label=source_label,
                        top_matches=top_hits,
                    )
            except Exception as e:
                print(f"QA Pipeline execution error: {e}")

        # Try generation pipeline
        gen_pipe = self._get_gen_pipeline()
        if gen_pipe and gen_pipe is not False:
            try:
                context = " ".join([chunk for chunk, _, _ in top_hits[:2]])
                prompt = (
                    "Answer the question using only the context provided. "
                    "Be concise and accurate.\n\n"
                    f"Question: {query}\n"
                    f"Context: {context}\n"
                    "Answer:"
                )
                gen = gen_pipe(prompt, max_new_tokens=100, do_sample=False)
                if gen and isinstance(gen, list) and len(gen) > 0:
                    gen_text = (gen[0].get("generated_text") or "").strip()
                    # Remove the prompt from generated text if present
                    if "Answer:" in gen_text:
                        gen_text = gen_text.split("Answer:")[-1].strip()
                    if gen_text and len(gen_text) > 5:
                        return QAResult(
                            answer=self._clean_answer_text(gen_text),
                            relevant_passage=self._clean_context_text(best_chunk),
                            confidence=max(0.5, sim_score),
                            source_label=source_label,
                            top_matches=top_hits,
                        )
            except Exception as e:
                print(f"Generation Pipeline error: {e}")

        # Fallback: pick the most relevant sentence
        candidate_sentences: List[str] = []
        for hit_chunk, _, _ in top_hits[: max(1, min(3, len(top_hits)))]:
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", hit_chunk) if s.strip()]
            candidate_sentences.extend(sentences)

        fallback_answer = best_chunk[:300].strip() + "..."
        if candidate_sentences:
            try:
                vectorizer = TfidfVectorizer(stop_words="english", max_features=200)
                sent_matrix = vectorizer.fit_transform(candidate_sentences + [query])
                sims = cosine_similarity(sent_matrix[-1], sent_matrix[:-1]).flatten()
                best_sent_idx = int(np.argmax(sims))
                fallback_answer = candidate_sentences[best_sent_idx]
            except Exception as e:
                print(f"Sentence selection error: {e}")
                fallback_answer = candidate_sentences[0] if candidate_sentences else fallback_answer

        return QAResult(
            answer=self._clean_answer_text(fallback_answer),
            relevant_passage=self._clean_context_text(best_chunk),
            confidence=max(0.3, sim_score),
            source_label=source_label,
            top_matches=top_hits,
        )

    def summarize(self, document_text: str) -> str:
        """Generate a summary of the document."""
        if not document_text.strip():
            return "No document content to summarize."
        
        short_text = document_text[:3000]
        sum_pipe = self._get_sum_pipeline()
        if sum_pipe and sum_pipe is not False:
            try:
                summary = sum_pipe(short_text, max_length=150, min_length=50, do_sample=False)
                if summary and isinstance(summary, list) and len(summary) > 0:
                    text = summary[0].get("summary_text", "").strip()
                    if text:
                        return text
            except Exception as e:
                print(f"Summarization error: {e}")

        # Extractive fallback summary
        sentences = re.split(r"(?<=[.!?])\s+", short_text)
        top_sentences = [s.strip() for s in sentences[:5] if s.strip()]
        return " ".join(top_sentences) if top_sentences else "Summary unavailable."

    def solve_mcq(
        self,
        question: str,
        options: List[str],
        document_text: str,
        sections: List[tuple[str, str]] | None = None,
        chunk_size: int = 700,
        chunk_overlap: int = 120,
        top_k: int = 3,
    ) -> MCQResult:
        """Solve MCQ by finding best matching option."""
        valid_options = [opt.strip() for opt in options if opt.strip()]
        if not question.strip() or len(valid_options) < 2:
            return MCQResult(
                selected_option="N/A",
                confidence=0.0,
                explanation="Provide a question and at least two options.",
            )

        chunks_with_source = build_chunks(document_text, sections, chunk_size, chunk_overlap)
        if not chunks_with_source:
            return MCQResult(
                selected_option="N/A",
                confidence=0.0,
                explanation="Could not process the document.",
            )
        
        context_query = f"{question} {' '.join(valid_options)}"
        top_hits = retrieve_top_chunks(context_query, chunks_with_source, top_k=max(1, top_k))
        if not top_hits:
            return MCQResult("N/A", 0.0, "No relevant context found in document.")

        try:
            context = " ".join([chunk for chunk, _, _ in top_hits[:2]])
            vectorizer = TfidfVectorizer(stop_words="english", max_features=300)
            docs = valid_options + [f"{question} {context}"]
            matrix = vectorizer.fit_transform(docs)
            option_vecs = matrix[:-1]
            query_vec = matrix[-1]
            scores = cosine_similarity(query_vec, option_vecs).flatten()
            best_idx = int(np.argmax(scores))
            best_score = float(scores[best_idx])
            selected = valid_options[best_idx]

            src = top_hits[0][1]
            explanation = (
                f"Selected option '{selected}' based on highest similarity with document context. "
                f"Source: {src}"
            )
            return MCQResult(selected_option=selected, confidence=max(0.3, best_score), explanation=explanation)
        except Exception as e:
            print(f"MCQ solving error: {e}")
            return MCQResult(
                selected_option=valid_options[0],
                confidence=0.2,
                explanation=f"Error in analysis, selected first option as fallback.",
            )
