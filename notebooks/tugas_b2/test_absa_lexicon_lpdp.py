"""Behavior checks for the Q6 lexicon-based ABSA implementation."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from absa_lexicon_lpdp import create_analyzer, run


class LexiconABSATest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.analyzer = create_analyzer()

    def predict(self, sentence, aspect, category, locator=None):
        return self.analyzer.predict_given_aspect(
            sentence, aspect, category, locator=locator
        ).polarity

    def test_negative_policy_criticism_is_detected(self):
        label = self.predict(
            "Seleksi penerima perlu diperketat untuk mencegah kolusi dan nepotisme.",
            "seleksi penerima",
            "Seleksi dan pendaftaran",
        )
        self.assertEqual(label, "negative")

    def test_variant_perketat_pengawasan_is_negative(self):
        label = self.predict(
            "DPD minta perketat pengawasan pascastudi penerima LPDP luar negeri.",
            "pengawasan pascastudi",
            "Pengawasan dan sanksi",
        )
        self.assertEqual(label, "negative")

    def test_administrative_statement_is_neutral(self):
        label = self.predict(
            "Pendaftaran Beasiswa Fellowship resmi dibuka pada 1 Januari 2026.",
            "pendaftaran beasiswa",
            "Seleksi dan pendaftaran",
        )
        self.assertEqual(label, "neutral")

    def test_positive_access_statement_is_detected(self):
        label = self.predict(
            "LPDP memberikan kesempatan yang sama untuk mengakses pendidikan tinggi.",
            "akses pendidikan tinggi",
            "Akses dan manfaat beasiswa",
            locator="kesempatan yang sama",
        )
        self.assertEqual(label, "positive")

    def test_contrast_clause_keeps_different_aspect_polarities(self):
        sentence = (
            "Akses beasiswa memberikan kesempatan yang sama, tetapi "
            "pengawasan LPDP masih perlu diperkuat."
        )
        access = self.predict(
            sentence,
            "akses beasiswa",
            "Akses dan manfaat beasiswa",
            locator="kesempatan yang sama",
        )
        monitoring = self.predict(
            sentence,
            "pengawasan LPDP",
            "Pengawasan dan sanksi",
            locator="pengawasan lpdp",
        )
        self.assertEqual(access, "positive")
        self.assertEqual(monitoring, "negative")

    def test_q3_validation_annotations_are_all_locatable(self):
        results = run(write_outputs=False)
        self.assertEqual(len(results["evaluation"]), 15)
        self.assertFalse(results["evaluation"]["source_sentence"].isna().any())


if __name__ == "__main__":
    unittest.main()
