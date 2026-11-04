"""Keep the manuscript's manually typeset rho tables on the current figure data."""
import json
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'artifacts/biology/concordance'
FIELDS = ('spearman_all', 'spearman_cross', 'partial_taxdist')
P_FIELDS = ('p_all', 'p_cross', 'p_partial')


def displayed_p(text):
    """A printed p-value and half a unit of its last displayed decimal place."""
    if '×' in text:
        coefficient, power = (s.strip() for s in text.split('×'))
        scale = {'10⁻⁵': 1e-5, '10⁻⁴': 1e-4}[power]
    else:
        coefficient, scale = text, 1
    decimals = len(coefficient.split('.')[1]) if '.' in coefficient else 0
    return float(coefficient)*scale, 0.5*10**(-decimals)*scale


class RhoTablesTest(unittest.TestCase):
    def setUp(self):
        self.manuscript = (ROOT / 'docs/paper/preprint.md').read_text()
        self.current = json.loads((DATA / 'rho_cross_panel_both_estimators.json').read_text())

    def test_table1_uses_current_reliabilities(self):
        lines = re.findall(r'^\| (?:2|6|24) h \|.*$', self.manuscript, re.M)
        self.assertEqual(len(lines), 3)
        for line, source in zip(lines, self.current['reliability']):
            cells = [s.strip() for s in line.strip('|').split('|')]
            self.assertIn(cells[0].replace(' ', ''), source['label'])
            for printed, field in zip(cells[1:], FIELDS):
                self.assertAlmostEqual(float(printed), source[field], delta=0.0005)

    def test_table3_coefficients_and_p_values_share_the_figure_sources(self):
        sources = {
            'raw': [r for r in self.current['comparisons'] if r.get('kind') == 'cross_time'],
            'double-centred': json.loads((DATA / 'rho_double_centred.json').read_text())['double_centred'],
        }
        for kind, rows in sources.items():
            lines = re.findall(r'^\| '+kind+r' \|.*$', self.manuscript, re.M)
            self.assertEqual(len(lines), 3)
            for line, source in zip(lines, rows):
                cells = [s.strip() for s in line.strip('|').split('|')]
                self.assertIn(cells[1].replace(' h', 'h'), source['label'])
                for cell, field, p_field in zip(cells[2:], FIELDS, P_FIELDS):
                    match = re.fullmatch(r'([+\-]\d+\.\d+) \(p = (.+)\)', cell)
                    self.assertIsNotNone(match, cell)
                    self.assertAlmostEqual(float(match[1]), source[field], delta=0.0005)
                    value, tolerance = displayed_p(match[2])
                    self.assertAlmostEqual(value, source[p_field], delta=tolerance)


if __name__ == '__main__':
    unittest.main()
