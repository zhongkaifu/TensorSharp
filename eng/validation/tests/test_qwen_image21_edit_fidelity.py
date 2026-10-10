"""Region bookkeeping of the Qwen-Image edit fidelity measure."""
import importlib.util
import math
from pathlib import Path
import unittest

import numpy as np


MODULE = Path(__file__).resolve().parents[1] / "qwen-image21-edit-fidelity.py"
SPEC = importlib.util.spec_from_file_location("qwen_image21_edit_fidelity", MODULE)
FIDELITY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FIDELITY)


class EditFidelityTests(unittest.TestCase):
    def setUp(self):
        self.source = np.full((40, 40, 3), 100.0)
        self.changed = np.zeros((40, 40), dtype=bool)
        self.changed[10:20, 10:20] = True

    def test_a_change_confined_to_the_object_and_its_margin_keeps_the_rest_exact(self):
        edited = self.source.copy()
        edited[8:22, 8:22] = 30.0   # the object and two pixels around it
        record = FIDELITY.measure(self.source, edited, self.changed, margin=2)
        self.assertEqual(record["psnr_outside_changed_db"], math.inf)
        self.assertEqual(record["changed_mean_abs_diff"], 70.0)
        self.assertAlmostEqual(record["psnr_whole_db"], 10 * math.log10(255 ** 2 / (14 * 14 * 70 ** 2 / 1600)), places=2)
        self.assertEqual(record["unchanged_fraction"], round(1 - 14 * 14 / 1600, 4))

    def test_a_change_beyond_the_margin_counts_against_the_kept_area(self):
        edited = self.source.copy()
        edited[30:32, 30:32] = 110.0
        record = FIDELITY.measure(self.source, edited, self.changed, margin=2)
        kept = 1600 - 14 * 14
        self.assertAlmostEqual(record["psnr_outside_changed_db"], 10 * math.log10(255 ** 2 / (4 * 100 / kept)), places=2)
        self.assertEqual(record["changed_mean_abs_diff"], 0.0)

    def test_a_selection_counts_protected_pixels_and_measures_inside_it(self):
        selection = np.zeros((40, 40), dtype=bool)
        selection[5:25, 5:25] = True
        edited = self.source.copy()
        edited[6, 6] = 0.0          # inside the selection, outside the object's margin
        edited[30, 30] = 0.0        # protected
        record = FIDELITY.measure(self.source, edited, self.changed, margin=2, selection=selection)
        self.assertEqual(record["protected_pixels_changed"], 1)
        inside = int((selection & ~FIDELITY.grow(self.changed, 2)).sum())
        self.assertAlmostEqual(record["psnr_selection_outside_changed_db"],
                               10 * math.log10(255 ** 2 / (3 * 100 ** 2 / (3 * inside))), places=2)

    def test_a_picture_of_another_size_is_refused(self):
        with self.assertRaises(ValueError):
            FIDELITY.measure(self.source, np.zeros((20, 40, 3)), self.changed, margin=2)


if __name__ == "__main__":
    unittest.main()
