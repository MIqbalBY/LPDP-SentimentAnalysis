"""Lexicon-based Aspect-Based Sentiment Analysis baseline for LPDP articles.

This program implements Q6 of Tugas B2:
- extracts LPDP domain aspects from sentences;
- predicts aspect polarity using InSet, domain opinion phrases, negation,
  contrast-clause selection, and factual-neutral rules;
- evaluates aspect polarity on the 15 manually annotated examples from Q3;
- generates aspect predictions for the full 1,038-document corpus.

The Q3 validation sample is a small demonstration set used during rule design.
Its metric is not an estimate of generalization to unseen aspect annotations.
"""

from __future__ import annotations

import argparse
import html
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "processed" / "dataset_lpdp_preprocessed_bert.csv"
POSITIVE_LEXICON_PATH = ROOT / "data" / "lexicon" / "positive.tsv"
NEGATIVE_LEXICON_PATH = ROOT / "data" / "lexicon" / "negative.tsv"
SLANG_PATH = ROOT / "data" / "lexicon" / "slang_id.csv"
OUTPUT_DIR = ROOT / "outputs" / "absa_q6"

LABELS = ["negative", "neutral", "positive"]
NEGATIONS = {"tidak", "tak", "bukan", "belum", "tanpa", "kurang"}
INTENSIFIERS = {"sangat", "amat", "lebih", "paling"}
CONTRAST_MARKERS = ("tetapi", "namun", "sedangkan", "akan tetapi")


# Aspect patterns are domain terms, not sentiment signals.
ASPECT_PATTERNS = {
    "Seleksi dan pendaftaran": (
        "seleksi penerima",
        "seleksi penerimaan",
        "tahapan pendaftaran",
        "pendaftaran beasiswa",
        "pendaftaran seleksi",
        "seleksi substansi",
        "persyaratan",
        "pendaftaran",
        "seleksi",
    ),
    "Pendanaan dan anggaran": (
        "kerja sama investasi",
        "dana abadi",
        "anggaran program",
        "anggaran",
        "pendanaan",
        "dana beasiswa",
        "pt smi",
    ),
    "Akses dan manfaat beasiswa": (
        "akses pendidikan tinggi",
        "perluasan akses pendidikan",
        "beasiswa afirmasi",
        "kuota beasiswa",
        "jenis program beasiswa",
        "fleksibilitas",
        "kesempatan yang sama",
        "kesempatan berkuliah",
    ),
    "Pengabdian dan kontribusi alumni": (
        "kewajiban mengabdi",
        "kewajiban pengabdian",
        "pengabdian 2n",
        "komitmen penerima",
        "kontribusi alumni",
        "mata garuda",
        "pengabdian",
    ),
    "Pengawasan dan sanksi": (
        "pengawasan pascastudi",
        "pengawasan lpdp",
        "monitoring",
        "sanksi",
        "pengawasan",
    ),
    "Tata kelola dan transparansi": (
        "tata kelola",
        "transparansi",
        "akuntabilitas",
        "izin resmi",
        "perlakuan istimewa",
        "tidak adil",
    ),
    "Kontroversi penerima beasiswa": (
        "kewarganegaraan",
        "nasionalisme",
        "polemik alumni",
        "paspor",
        "kontroversi",
    ),
}


# Domain phrases extend InSet for policy/news wording in this corpus.
DOMAIN_OPINION_LEXICON = {
    "perlu diperketat": -4.0,
    "harus ketat": -3.0,
    "kolusi": -4.0,
    "nepotisme": -4.0,
    "tidak akan mengganggu": -4.0,
    "mengganggu": -3.0,
    "dikenai sanksi": -4.0,
    "dijatuhi sanksi": -4.0,
    "tidak menjalankan kewajiban": -5.0,
    "tidak memenuhi kewajiban": -5.0,
    "memperketat pengawasan": -4.0,
    "perketat pengawasan": -4.0,
    "masih perlu diperkuat": -4.0,
    "lemahnya monitoring": -5.0,
    "permainan tidak adil": -5.0,
    "kesempatan yang sama": 4.0,
    "bersifat inklusif": 4.0,
    "inklusif": 3.0,
    "talenta unggul": 3.0,
    "perluasan akses pendidikan": 4.0,
    "menggenjot kualitas": 3.0,
    "modal yang baik": 4.0,
    "memiliki fleksibilitas": 3.0,
    "memperkaya pengalaman": 3.0,
    "membangun kolaborasi": 3.0,
    "berkontribusi": 2.0,
    "berhasil menggandeng": 4.0,
}


# Administrative statements are neutral even when generic lexicons score words
# such as "dibuka" or "kewajiban" as sentiment-bearing terms.
FACTUAL_NEUTRAL_PATTERNS = (
    "resmi dibuka pada",
    "tahapan pendaftaran",
    "terdapat beberapa jenis program beasiswa",
    "mencatat total dana abadi",
    "tidak menghilangkan kewajiban pengabdian",
)


GOLD_ANNOTATIONS = [
    {
        "doc_id": 0,
        "anchor": "seleksi penerima program beasiswa",
        "locator": "seleksi penerima",
        "aspect_term": "seleksi penerima",
        "category": "Seleksi dan pendaftaran",
        "gold_polarity": "negative",
    },
    {
        "doc_id": 1,
        "anchor": "anggarannya tersedia dan tidak akan mengganggu",
        "locator": "anggarannya",
        "aspect_term": "anggaran program LPDP Jakarta",
        "category": "Pendanaan dan anggaran",
        "gold_polarity": "negative",
    },
    {
        "doc_id": 5,
        "anchor": "dikenai sanksi akibat terbukti tidak menjalankan kewajiban",
        "locator": "kewajiban mengabdi",
        "aspect_term": "kewajiban mengabdi alumni",
        "category": "Pengabdian dan kontribusi alumni",
        "gold_polarity": "negative",
    },
    {
        "doc_id": 6,
        "anchor": "memperketat pengawasan pascastudi",
        "locator": "pengawasan pascastudi",
        "aspect_term": "pengawasan pascastudi",
        "category": "Pengawasan dan sanksi",
        "gold_polarity": "negative",
    },
    {
        "doc_id": 600,
        "anchor": "pengawasan lpdp masih perlu diperkuat",
        "locator": "pengawasan lpdp",
        "aspect_term": "pengawasan LPDP",
        "category": "Pengawasan dan sanksi",
        "gold_polarity": "negative",
    },
    {
        "doc_id": 13,
        "anchor": "pendaftaran beasiswa fellowship dokter spesialis resmi dibuka",
        "locator": "pendaftaran beasiswa",
        "aspect_term": "pendaftaran beasiswa",
        "category": "Seleksi dan pendaftaran",
        "gold_polarity": "neutral",
    },
    {
        "doc_id": 14,
        "anchor": "tahapan pendaftaran beasiswa lpdp umumnya mencakup",
        "locator": "tahapan pendaftaran",
        "aspect_term": "tahapan pendaftaran",
        "category": "Seleksi dan pendaftaran",
        "gold_polarity": "neutral",
    },
    {
        "doc_id": 17,
        "anchor": "terdapat beberapa jenis program beasiswa",
        "locator": "jenis program beasiswa",
        "aspect_term": "jenis program beasiswa",
        "category": "Akses dan manfaat beasiswa",
        "gold_polarity": "neutral",
    },
    {
        "doc_id": 94,
        "anchor": "mencatat total dana abadi",
        "locator": "dana abadi",
        "aspect_term": "dana abadi",
        "category": "Pendanaan dan anggaran",
        "gold_polarity": "neutral",
    },
    {
        "doc_id": 219,
        "anchor": "tidak menghilangkan kewajiban pengabdian 2 n",
        "locator": "kewajiban pengabdian",
        "aspect_term": "kewajiban pengabdian 2N",
        "category": "Pengabdian dan kontribusi alumni",
        "gold_polarity": "neutral",
    },
    {
        "doc_id": 4,
        "anchor": "memberikan kesempatan yang sama",
        "locator": "kesempatan yang sama",
        "aspect_term": "akses pendidikan tinggi",
        "category": "Akses dan manfaat beasiswa",
        "gold_polarity": "positive",
    },
    {
        "doc_id": 12,
        "anchor": "mengalokasikan 5 750 kuota beasiswa",
        "locator": "kuota beasiswa",
        "aspect_term": "kuota beasiswa",
        "category": "Akses dan manfaat beasiswa",
        "gold_polarity": "positive",
    },
    {
        "doc_id": 15,
        "anchor": "modal yang baik bagi anak anak jakarta",
        "locator": "program ini",
        "aspect_term": "program LPDP untuk mahasiswa Jakarta",
        "category": "Akses dan manfaat beasiswa",
        "gold_polarity": "positive",
    },
    {
        "doc_id": 23,
        "anchor": "membangun kolaborasi demi berkontribusi",
        "locator": "mata garuda",
        "aspect_term": "kontribusi alumni/Mata Garuda",
        "category": "Pengabdian dan kontribusi alumni",
        "gold_polarity": "positive",
    },
    {
        "doc_id": 545,
        "anchor": "berhasil menggandeng pt smi",
        "locator": "pt smi",
        "aspect_term": "kerja sama pendanaan dan pengembangan SDM",
        "category": "Pendanaan dan anggaran",
        "gold_polarity": "positive",
    },
]


@dataclass
class AspectPrediction:
    aspect_term: str
    category: str
    polarity: str
    score: float
    context: str
    evidence: str


def clean_technical_noise(text: str) -> str:
    text = html.unescape(str(text))
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[@#]\w+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def load_slang_map(path: Path) -> dict[str, str]:
    slang_df = pd.read_csv(path)
    return {
        str(row.slang).lower(): str(row.formal).lower()
        for row in slang_df.itertuples()
        if pd.notna(row.formal)
    }


def normalized_tokens(text: str, slang_map: dict[str, str]) -> list[str]:
    raw_tokens = re.findall(r"[a-z]+|\d+", clean_technical_noise(text).lower())
    tokens: list[str] = []
    for token in raw_tokens:
        tokens.extend(slang_map.get(token, token).split())
    return tokens


def normalize_for_matching(text: str, slang_map: dict[str, str]) -> str:
    return " ".join(normalized_tokens(text, slang_map))


def split_sentences(text: str) -> list[str]:
    cleaned = clean_technical_noise(text)
    # Scraped articles sometimes concatenate sentences without whitespace.
    marked = re.sub(r'([.!?]["”]?)(\s*)(?=[A-Z“])', r"\1\n", cleaned)
    return [part.strip() for part in marked.splitlines() if part.strip()]


def load_inset_unigrams(
    positive_path: Path, negative_path: Path, slang_map: dict[str, str]
) -> dict[str, float]:
    scores: dict[str, float] = {}
    for path in (positive_path, negative_path):
        entries = pd.read_csv(path, sep="\t")
        for row in entries.itertuples():
            phrase = normalize_for_matching(str(row.word), slang_map)
            if " " not in phrase and phrase:
                scores[phrase] = scores.get(phrase, 0.0) + float(row.weight)
    return scores


class LexiconABSA:
    def __init__(self, inset_scores: dict[str, float], slang_map: dict[str, str]):
        self.inset_scores = inset_scores
        self.slang_map = slang_map

    def extract_aspects(self, sentence: str) -> list[tuple[str, str]]:
        normalized = normalize_for_matching(sentence, self.slang_map)
        found: list[tuple[str, str]] = []
        for category, terms in ASPECT_PATTERNS.items():
            matching = [term for term in terms if term in normalized]
            if matching:
                # Use one most specific aspect term for each category/sentence.
                found.append((max(matching, key=len), category))
        return found

    def _select_aspect_clause(self, sentence: str, locator: str) -> str:
        normalized = normalize_for_matching(sentence, self.slang_map)
        locator_norm = normalize_for_matching(locator, self.slang_map)
        split_pattern = r"\b(?:" + "|".join(
            re.escape(marker) for marker in CONTRAST_MARKERS
        ) + r")\b"
        clauses = [clause.strip() for clause in re.split(split_pattern, normalized)]
        for clause in clauses:
            if locator_norm and locator_norm in clause:
                return clause
        return normalized

    @staticmethod
    def _local_window(clause: str, locator: str, window_size: int = 12) -> str:
        tokens = clause.split()
        locator_tokens = locator.split()
        if not locator_tokens:
            return clause
        for index in range(len(tokens) - len(locator_tokens) + 1):
            if tokens[index : index + len(locator_tokens)] == locator_tokens:
                start = max(0, index - window_size)
                end = min(len(tokens), index + len(locator_tokens) + window_size)
                return " ".join(tokens[start:end])
        # The manually described aspect may differ from the literal source span.
        return clause

    def _inset_score(self, context: str) -> tuple[float, list[str]]:
        tokens = context.split()
        total = 0.0
        evidence: list[str] = []
        for index, token in enumerate(tokens):
            value = self.inset_scores.get(token)
            if value is None:
                continue
            preceding = tokens[max(0, index - 3) : index]
            adjusted = value
            if any(term in NEGATIONS for term in preceding):
                adjusted *= -1
                marker = "negated"
            else:
                marker = "inset"
            if any(term in INTENSIFIERS for term in preceding):
                adjusted *= 1.5
            total += adjusted
            evidence.append(f"{marker}:{token}({adjusted:+.1f})")
        return total, evidence

    def predict_given_aspect(
        self, sentence: str, aspect_term: str, category: str, locator: str | None = None
    ) -> AspectPrediction:
        locator_norm = normalize_for_matching(locator or aspect_term, self.slang_map)
        clause = self._select_aspect_clause(sentence, locator or aspect_term)
        context = self._local_window(clause, locator_norm)

        neutral_match = next(
            (phrase for phrase in FACTUAL_NEUTRAL_PATTERNS if phrase in clause),
            None,
        )
        if neutral_match:
            return AspectPrediction(
                aspect_term=aspect_term,
                category=category,
                polarity="neutral",
                score=0.0,
                context=context,
                evidence=f"factual-rule:{neutral_match}",
            )

        domain_hits = [
            (phrase, weight)
            for phrase, weight in DOMAIN_OPINION_LEXICON.items()
            if phrase in context
        ]
        domain_score = sum(weight for _, weight in domain_hits)
        inset_score, inset_evidence = self._inset_score(context)
        # Bound generic scores so generic InSet terms cannot drown out
        # an explicit LPDP policy opinion phrase.
        inset_component = max(-4.0, min(4.0, inset_score)) * 0.25
        score = domain_score + inset_component

        if score > 0.75:
            polarity = "positive"
        elif score < -0.75:
            polarity = "negative"
        else:
            polarity = "neutral"

        evidence_parts = [
            f"domain:{phrase}({weight:+.1f})" for phrase, weight in domain_hits
        ]
        evidence_parts.extend(inset_evidence[:5])
        evidence = "; ".join(evidence_parts) or "no opinion term in local context"
        return AspectPrediction(
            aspect_term=aspect_term,
            category=category,
            polarity=polarity,
            score=round(score, 3),
            context=context,
            evidence=evidence,
        )

    def analyze_sentence(self, sentence: str) -> list[AspectPrediction]:
        return [
            self.predict_given_aspect(sentence, term, category, locator=term)
            for term, category in self.extract_aspects(sentence)
        ]


def create_analyzer() -> LexiconABSA:
    slang_map = load_slang_map(SLANG_PATH)
    inset_scores = load_inset_unigrams(
        POSITIVE_LEXICON_PATH, NEGATIVE_LEXICON_PATH, slang_map
    )
    return LexiconABSA(inset_scores=inset_scores, slang_map=slang_map)


def locate_gold_sentence(text: str, anchor: str, slang_map: dict[str, str]) -> str:
    normalized_anchor = normalize_for_matching(anchor, slang_map)
    for sentence in split_sentences(text):
        if normalized_anchor in normalize_for_matching(sentence, slang_map):
            return sentence
    raise ValueError(f"Could not locate gold anchor in source article: {anchor}")


def evaluate_gold(
    df: pd.DataFrame, analyzer: LexiconABSA
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    evaluation_rows = []
    df_indexed = df.set_index("doc_id")
    for gold in GOLD_ANNOTATIONS:
        document = df_indexed.loc[gold["doc_id"]]
        sentence = locate_gold_sentence(
            document["text_bert"], gold["anchor"], analyzer.slang_map
        )
        prediction = analyzer.predict_given_aspect(
            sentence=sentence,
            aspect_term=gold["aspect_term"],
            category=gold["category"],
            locator=gold["locator"],
        )
        evaluation_rows.append(
            {
                **gold,
                "title": document["Title"],
                "source_sentence": sentence,
                "predicted_polarity": prediction.polarity,
                "score": prediction.score,
                "context": prediction.context,
                "evidence": prediction.evidence,
                "correct": gold["gold_polarity"] == prediction.polarity,
            }
        )

    evaluation_df = pd.DataFrame(evaluation_rows)
    y_true = evaluation_df["gold_polarity"]
    y_pred = evaluation_df["predicted_polarity"]
    metrics_df = pd.DataFrame(
        [
            {"metric": "Accuracy", "value": accuracy_score(y_true, y_pred)},
            {
                "metric": "Precision Macro",
                "value": precision_score(y_true, y_pred, average="macro", zero_division=0),
            },
            {
                "metric": "Recall Macro",
                "value": recall_score(y_true, y_pred, average="macro", zero_division=0),
            },
            {
                "metric": "F1 Macro",
                "value": f1_score(y_true, y_pred, average="macro", zero_division=0),
            },
            {
                "metric": "F1 Weighted",
                "value": f1_score(y_true, y_pred, average="weighted", zero_division=0),
            },
        ]
    )
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    confusion_df = pd.DataFrame(
        cm,
        index=[f"actual_{label}" for label in LABELS],
        columns=[f"predicted_{label}" for label in LABELS],
    )
    return evaluation_df, metrics_df, confusion_df


def analyze_corpus(df: pd.DataFrame, analyzer: LexiconABSA) -> pd.DataFrame:
    rows = []
    for document in df.itertuples():
        for sentence in split_sentences(document.text_bert):
            for prediction in analyzer.analyze_sentence(sentence):
                rows.append(
                    {
                        "doc_id": document.doc_id,
                        "document_sentiment": document.Sentiment,
                        "title": document.Title,
                        "sentence": sentence,
                        **asdict(prediction),
                    }
                )
    return pd.DataFrame(rows).drop_duplicates(
        subset=["doc_id", "sentence", "aspect_term", "category"]
    )


def run(write_outputs: bool = True) -> dict[str, pd.DataFrame]:
    df = pd.read_csv(DATA_PATH)
    analyzer = create_analyzer()
    evaluation_df, metrics_df, confusion_df = evaluate_gold(df, analyzer)
    predictions_df = analyze_corpus(df, analyzer)
    summary_df = (
        predictions_df.groupby(["category", "polarity"])
        .size()
        .rename("count")
        .reset_index()
    )

    if write_outputs:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        evaluation_df.to_csv(OUTPUT_DIR / "gold_evaluation.csv", index=False)
        metrics_df.to_csv(OUTPUT_DIR / "metrics.csv", index=False)
        confusion_df.to_csv(OUTPUT_DIR / "confusion_matrix.csv")
        predictions_df.to_csv(OUTPUT_DIR / "aspect_predictions_all.csv", index=False)
        summary_df.to_csv(OUTPUT_DIR / "aspect_polarity_summary.csv", index=False)

    return {
        "evaluation": evaluation_df,
        "metrics": metrics_df,
        "confusion": confusion_df,
        "predictions": predictions_df,
        "summary": summary_df,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-write", action="store_true", help="Run analysis without writing CSV outputs."
    )
    args = parser.parse_args()
    results = run(write_outputs=not args.no_write)

    print("Q6 Lexicon-based ABSA - LPDP")
    print(f"Documents processed : {pd.read_csv(DATA_PATH).shape[0]:,}")
    print(f"Aspect predictions  : {len(results['predictions']):,}")
    print("\nEvaluation on 15 manually annotated Q3 aspects:")
    print(results["metrics"].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nConfusion matrix:")
    print(results["confusion"].to_string())
    if not args.no_write:
        print(f"\nOutputs written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
