import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_dose_and_multiplicity import concentration, dose_match, holm, paired_summary


class ReviewStatisticsTests(unittest.TestCase):
    def test_holm_preserves_input_order_and_handles_ties(self):
        self.assertEqual(holm([.04, .01, .03]), [.06, .03, .06])
        self.assertEqual(holm([.02, .02, .001]), [.04, .04, .003])
        self.assertEqual(holm([.9, 1.]), [1., 1.])

    def test_missing_or_conflicting_doses_are_not_matches(self):
        dose = {(600., "μM")}
        self.assertFalse(dose_match(set(), set()))
        self.assertFalse(dose_match({None}, {None}))
        self.assertFalse(dose_match(dose, {(15., "μM")}))
        self.assertFalse(dose_match(dose, dose | {(15., "μM")}))

    def test_concentration_matching_is_unit_invariant_but_not_mass_to_molar(self):
        self.assertTrue(dose_match({concentration(1, "mM")}, {concentration(1000, "µM")}))
        self.assertTrue(dose_match({concentration(50, "ng/mL")}, {concentration(.05, "μg/mL")}))
        self.assertFalse(dose_match({concentration(1, "μM")}, {concentration(1, "μg/mL")}))
        self.assertIsNone(concentration(1, "μg/kg"))

    def test_sign_test_drops_exact_ties_and_handles_empty_pairs(self):
        rows = [{"time_minus_species_pp": d} for d in [1, 2, 0, 3]]
        result = paired_summary(rows)
        self.assertEqual(result["n_nonzero"], 3)
        self.assertEqual(result["p_two_sided_sign_nominal"], .25)
        self.assertIsNone(paired_summary([])["p_two_sided_sign_nominal"])


if __name__ == "__main__":
    unittest.main()
