from __future__ import annotations

import gzip
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence, Tuple

SPECIAL_UNK = "<UNK>"
SPECIAL_BOS = "<BOS>"
SPECIAL_EOS = "<EOS>"

TOKEN_RE = re.compile(r"\w+(?:['’]\w+)*|[^\w\s]", re.UNICODE)
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+|\n+")


def tokenize(text: str) -> List[str]:
    return [m.group(0).lower() for m in TOKEN_RE.finditer(text)]


def split_sentences(text: str) -> List[List[str]]:
    parts = SENTENCE_RE.split(text.strip())
    sentences: List[List[str]] = []
    for part in parts:
        tokens = tokenize(part)
        if tokens:
            sentences.append(tokens)
    return sentences


def load_c4_json_gz(
    path: str | Path,
    max_docs: int | None = None,
) -> Iterator[str]:
    path = Path(path)
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_docs is not None and i >= max_docs:
                break
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text = obj.get("text", "")
            if isinstance(text, str) and text.strip():
                yield text


def load_corpus(
    path: str | Path,
    max_docs: int | None = None,
) -> List[List[str]]:
    corpus: List[List[str]] = []
    for text in load_c4_json_gz(path, max_docs=max_docs):
        corpus.extend(split_sentences(text))
    return corpus


def load_documents(
    path: str | Path,
    max_docs: int | None = None,
) -> List[List[List[str]]]:
    documents: List[List[List[str]]] = []
    for text in load_c4_json_gz(path, max_docs=max_docs):
        sentences = split_sentences(text)
        if sentences:
            documents.append(sentences)
    return documents


def flatten_documents(
    documents: Sequence[Sequence[Sequence[str]]],
) -> List[List[str]]:
    """Flatten document -> sentences into a sentence corpus."""
    return [list(sentence) for doc in documents for sentence in doc]


def split_documents(
    documents: Sequence[Sequence[Sequence[str]]],
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
) -> Tuple[List[List[List[str]]], List[List[List[str]]], List[List[List[str]]]]:
    n = len(documents)
    train_end = int(n * train_ratio)
    validation_end = train_end + int(n * validation_ratio)

    train = [list(doc) for doc in documents[:train_end]]
    validation = [list(doc) for doc in documents[train_end:validation_end]]
    test = [list(doc) for doc in documents[validation_end:]]
    return train, validation, test


class NGramLanguageModel:
    def __init__(
        self,
        n: int = 2,
        smoothing: str = "none",
        unk_cutoff: int = 0,
    ):
        if n not in (1, 2, 3):
            raise ValueError("n must be 1, 2, or 3")
        if smoothing not in ("none", "laplace"):
            raise ValueError("smoothing must be 'none' or 'laplace'")

        self.n = n
        self.smoothing = smoothing
        self.unk_cutoff = unk_cutoff

        self.vocab: set[str] = set()
        self.predictable_vocab: set[str] = set()
        self.ngram_counts: Counter[Tuple[str, ...]] = Counter()
        self.context_counts: Counter[Tuple[str, ...]] = Counter()
        self.total_target_tokens = 0
        self.fitted = False

    def build_vocabulary(
        self,
        corpus: Sequence[Sequence[str]],
    ) -> set[str]:
        counts = Counter(token for sent in corpus for token in sent)

        if self.unk_cutoff > 0:
            words = {
                token for token, count in counts.items()
                if count > self.unk_cutoff
            }
        else:
            words = set(counts)

        self.vocab = words | {SPECIAL_UNK, SPECIAL_EOS}
        # BOS is context-only and is never a prediction target.
        self.predictable_vocab = self.vocab.copy()
        return self.vocab

    def _normalize_sentence(
        self,
        sentence: Sequence[str],
    ) -> List[str]:
        result = []
        for token in sentence:
            token = token.lower()
            if token in self.vocab:
                result.append(token)
            else:
                result.append(SPECIAL_UNK)
        return result

    def _padded(
        self,
        sentence: Sequence[str],
    ) -> List[str]:
        sent = self._normalize_sentence(sentence)
        if self.n == 1:
            return sent + [SPECIAL_EOS]
        return [SPECIAL_BOS] * (self.n - 1) + sent + [SPECIAL_EOS]

    def count_ngrams(
        self,
        corpus: Sequence[Sequence[str]],
    ) -> Counter[Tuple[str, ...]]:
        self.ngram_counts.clear()
        self.context_counts.clear()
        self.total_target_tokens = 0

        for sentence in corpus:
            padded = self._padded(sentence)
            self.total_target_tokens += len(sentence) + 1  # +EOS

            for i in range(self.n - 1, len(padded)):
                ngram = tuple(padded[i - self.n + 1:i + 1])
                self.ngram_counts[ngram] += 1

                if self.n > 1:
                    context = ngram[:-1]
                    self.context_counts[context] += 1

        return self.ngram_counts

    def fit(
        self,
        corpus: Sequence[Sequence[str]],
    ) -> "NGramLanguageModel":
        self.build_vocabulary(corpus)
        self.count_ngrams(corpus)
        self.fitted = True
        return self

    def _prepare_context(
        self,
        context: Sequence[str],
    ) -> Tuple[str, ...]:
        if self.n == 1:
            return ()

        normalized = []
        for token in context:
            token = token.lower()
            normalized.append(token if token in self.vocab else SPECIAL_UNK)

        needed = self.n - 1
        return tuple(normalized[-needed:])

    def probability(
        self,
        context: Sequence[str],
        word: str,
    ) -> float:
        """
        Compute P(word | context).

        MLE:
            count(context, word) / count(context)

        Laplace:
            (count(context, word) + 1) /
            (count(context) + V)
        """
        if not self.fitted:
            raise RuntimeError("Call fit() before probability().")

        word = word.lower()
        if word not in self.vocab:
            word = SPECIAL_UNK

        if self.n == 1:
            numerator = self.ngram_counts.get((word,), 0)
            denominator = self.total_target_tokens
        else:
            ctx = self._prepare_context(context)
            numerator = self.ngram_counts.get(ctx + (word,), 0)
            denominator = self.context_counts.get(ctx, 0)

        if self.smoothing == "laplace":
            V = len(self.predictable_vocab)
            return (numerator + 1) / (denominator + V)

        if denominator == 0:
            return 0.0
        return numerator / denominator

    def sentence_log_probability(
        self,
        sentence: Sequence[str],
    ) -> float:
        """Return log P(sentence), including EOS."""
        padded = self._padded(sentence)

        if self.n == 1:
            targets = padded
            contexts = [()] * len(targets)
        else:
            targets = padded[self.n - 1:]
            contexts = [
                tuple(padded[i - self.n + 1:i])
                for i in range(self.n - 1, len(padded))
            ]

        total = 0.0
        for context, target in zip(contexts, targets):
            p = self.probability(context, target)
            if p <= 0:
                return float("-inf")
            total += math.log(p)
        return total

    def sentence_probability(
        self,
        sentence: Sequence[str],
    ) -> float:
        """Return P(sentence) from the log probability."""
        logp = self.sentence_log_probability(sentence)
        if math.isinf(logp) and logp < 0:
            return 0.0
        return math.exp(logp)

    def perplexity(
        self,
        corpus: Sequence[Sequence[str]],
    ) -> float:
        """Token-level perplexity, counting EOS as a target."""
        log_sum = 0.0
        target_count = 0

        for sentence in corpus:
            padded = self._padded(sentence)

            if self.n == 1:
                targets = padded
                contexts = [()] * len(targets)
            else:
                targets = padded[self.n - 1:]
                contexts = [
                    tuple(padded[i - self.n + 1:i])
                    for i in range(self.n - 1, len(padded))
                ]

            for context, target in zip(contexts, targets):
                p = self.probability(context, target)
                if p <= 0:
                    return float("inf")
                log_sum += math.log(p)
                target_count += 1

        if target_count == 0:
            return float("inf")
        return math.exp(-log_sum / target_count)

    def next_word_distribution(
        self,
        context: Sequence[str],
        top_k: int | None = None,
    ) -> List[Tuple[str, float]]:
        """Return the highest-probability next-word candidates."""
        pairs = []
        for word in sorted(self.predictable_vocab):
            p = self.probability(context, word)
            if p > 0:
                pairs.append((word, p))

        pairs.sort(key=lambda item: (-item[1], item[0]))
        return pairs if top_k is None else pairs[:top_k]

    def rank_candidates(
        self,
        context: Sequence[str],
        candidates: Sequence[str],
    ) -> List[Tuple[str, float]]:
        results = []

        for candidate in candidates:
            words = tokenize(candidate)
            current_context = list(context)
            score = 0.0

            for word in words:
                p = self.probability(current_context, word)
                if p <= 0:
                    score = float("-inf")
                    break
                score += math.log(p)
                current_context.append(word)

            results.append((candidate, score))

        results.sort(
            key=lambda item: (
                math.isinf(item[1]),
                -item[1] if not math.isinf(item[1]) else 0.0,
                item[0],
            )
        )
        return results

    def statistics(self) -> Dict[str, int]:
        return {
            "n": self.n,
            "vocabulary_size": len(self.vocab),
            "unique_ngrams": len(self.ngram_counts),
            "singleton_ngrams": sum(
                count == 1 for count in self.ngram_counts.values()
            ),
            "total_ngrams": sum(self.ngram_counts.values()),
        }


def corpus_statistics(
    corpus: Sequence[Sequence[str]],
) -> Dict[str, int]:
    unigram = Counter(token for sent in corpus for token in sent)
    bigram = Counter(
        (sent[i], sent[i + 1])
        for sent in corpus
        for i in range(len(sent) - 1)
    )
    trigram = Counter(
        (sent[i], sent[i + 1], sent[i + 2])
        for sent in corpus
        for i in range(len(sent) - 2)
    )

    vocabulary = set(unigram)

    return {
        "sentences": len(corpus),
        "tokens": sum(unigram.values()),
        "vocabulary_size": len(vocabulary),
        "unique_unigrams": len(unigram),
        "unique_bigrams": len(bigram),
        "unique_trigrams": len(trigram),
        "singleton_unigrams": sum(c == 1 for c in unigram.values()),
        "singleton_bigrams": sum(c == 1 for c in bigram.values()),
        "singleton_trigrams": sum(c == 1 for c in trigram.values()),
    }


def frequency_distribution(
    corpus: Sequence[Sequence[str]],
    n: int = 1,
) -> Counter:
    if n == 1:
        return Counter(token for sent in corpus for token in sent)
    if n == 2:
        return Counter(
            (sent[i], sent[i + 1])
            for sent in corpus
            for i in range(len(sent) - 1)
        )
    if n == 3:
        return Counter(
            (sent[i], sent[i + 1], sent[i + 2])
            for sent in corpus
            for i in range(len(sent) - 2)
        )
    raise ValueError("n must be 1, 2, or 3")


def count_unseen_ngrams(
    train_corpus: Sequence[Sequence[str]],
    eval_corpus: Sequence[Sequence[str]],
    n: int,
) -> int:
    train_ngrams = set(frequency_distribution(train_corpus, n))
    eval_ngrams = set(frequency_distribution(eval_corpus, n))
    return len(eval_ngrams - train_ngrams)

'''Demo'''
if __name__ == "__main__":
    demo = [
        ["language", "models", "predict", "words"],
        ["language", "models", "use", "context"],
    ]
    model = NGramLanguageModel(n=2, smoothing="laplace").fit(demo)
    print(model.probability(["language"], "models"))
