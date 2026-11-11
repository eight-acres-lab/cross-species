#!/usr/bin/env python3
"""Generate Figures 1–6 from the paper's result records and editable artwork."""
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import platform
import re
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
import pymupdf
from PIL import Image, ImageDraw, ImageOps

from paper_figure_plots import build_figure_2, build_figure_3, build_figure_4, build_figure_6
from paper_figure_schematics import build_figure_1, build_figure_5

ROOT = Path(__file__).resolve().parents[1]
FIGURES = {
    1: ('Fig1_method', build_figure_1),
    2: ('Fig2_information', build_figure_2),
    3: ('Fig3_correspondence_transport', build_figure_3),
    4: ('Fig4_adaptation', build_figure_4),
    5: ('Fig5_augmentation_design', build_figure_5),
    6: ('Fig6_augmentation_results', build_figure_6),
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalize_svg(path):
    raw = re.sub(r'<!DOCTYPE[^>]*>\s*', '', path.read_text())
    ET.register_namespace('', 'http://www.w3.org/2000/svg')
    ET.register_namespace('xlink', 'http://www.w3.org/1999/xlink')
    root = ET.fromstring(raw)
    for element in root.iter():
        style = element.get('style', '')
        match = re.search(r'font-size:\s*([0-9.]+)px;?', style)
        if match:
            element.set('font-size', match[1])
            element.set('style', style[:match.start()] + style[match.end():])
        if re.fullmatch(r'rotate\(-?0(?:\.0+)? [^)]*\)', element.get('transform', '')):
            del element.attrib['transform']
    path.write_bytes(ET.tostring(root, encoding='utf-8', xml_declaration=True))


class FigureContext:
    def __init__(self, output):
        self.output = output
        self.source_data = output / 'source_data'
        self.source_data.mkdir(parents=True, exist_ok=True)
        self.inputs = {}

    def record(self, relative):
        path = ROOT / relative
        self.inputs[str(relative)] = digest(path)
        return path

    def asset(self, name):
        return self.record('figures/assets/' + name)

    def load_json(self, relative):
        return json.loads(self.record(relative).read_text())

    def load_csv(self, relative):
        return pd.read_csv(self.record(relative))

    def write_json(self, name, value):
        (self.source_data / name).write_text(json.dumps(value, indent=2) + '\n')

    def write_csv(self, name, frame):
        frame.to_csv(self.source_data / name, index=False)

    def save(self, figure, name, axes, png_dpi=450, pdf_dpi=600, portable_svg=False):
        figure.canvas.draw()
        positions = np.array([axis.get_position().bounds for axis in axes])
        assert np.ptp(positions[:, 2]) < 1e-8, name + ': panel widths differ'
        assert np.ptp(positions[:, 3]) < 1e-8, name + ': panel heights differ'
        target = self.output / name
        figure.savefig(target.with_suffix('.pdf'), dpi=pdf_dpi,
                       metadata={'CreationDate': None, 'ModDate': None})
        with plt.rc_context({'svg.hashsalt': 'cross-species-paper-figures'}):
            figure.savefig(target.with_suffix('.svg'), dpi=pdf_dpi, metadata={'Date': None})
        if portable_svg:
            normalize_svg(target.with_suffix('.svg'))
        figure.savefig(target.with_suffix('.png'), dpi=png_dpi)
        plt.close(figure)


def compare_pdf(actual, reference, dpi=144):
    with pymupdf.open(actual) as result, pymupdf.open(reference) as expected:
        assert len(result) == len(expected) == 1
        assert result[0].rect == expected[0].rect, actual.name + ': page dimensions differ'
        matrix = pymupdf.Matrix(dpi / 72, dpi / 72)
        a = result[0].get_pixmap(matrix=matrix, alpha=False)
        b = expected[0].get_pixmap(matrix=matrix, alpha=False)
        aa = np.frombuffer(a.samples, dtype=np.uint8).reshape(a.height, a.width, a.n)
        bb = np.frombuffer(b.samples, dtype=np.uint8).reshape(b.height, b.width, b.n)
        delta = np.abs(aa.astype(np.int16) - bb.astype(np.int16))
        return {
            'render_dpi': dpi, 'width_px': a.width, 'height_px': a.height,
            'changed_pixels': int(np.count_nonzero(np.any(delta != 0, axis=2))),
            'max_channel_difference': int(delta.max()),
            'mean_channel_difference': float(delta.mean()),
            'pixel_identical': bool(not np.any(delta)),
        }


def overview(output, selected):
    width, height = 760, 760
    sheet = Image.new('RGB', (width * 2, height * ((len(selected) + 1) // 2)), 'white')
    draw = ImageDraw.Draw(sheet)
    for index, number in enumerate(selected):
        name = FIGURES[number][0]
        with Image.open(output / (name + '.png')) as source:
            preview = ImageOps.contain(source.convert('RGB'), (width - 28, height - 56))
        x = (index % 2) * width
        y = (index // 2) * height
        draw.text((x + 16, y + 12), 'Figure ' + str(number), fill='black', font_size=24)
        sheet.paste(preview, (x + (width - preview.width) // 2, y + 46))
    sheet.save(output / 'figures_overview.png')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT / 'figures/generated')
    parser.add_argument('--figures', nargs='+', type=int, choices=FIGURES, default=list(FIGURES))
    parser.add_argument('--compare', action='store_true', help='Compare rendered PDFs with the manuscript figures')
    args = parser.parse_args()
    output = args.out.resolve()
    if output == ROOT or output in ROOT.parents:
        parser.error('Choose a dedicated figure output directory.')
    for reserved in (ROOT / 'figures/assets', ROOT / 'figures/reference', ROOT / 'figures/captions'):
        if output == reserved or reserved in output.parents or output in reserved.parents:
            parser.error('Choose an output directory outside figure inputs.')
    font_path = font_manager.findfont('Arial', fallback_to_default=False)
    output.mkdir(parents=True, exist_ok=True)
    ctx = FigureContext(output)
    references = ctx.load_json('figures/paper_figure_manifest.json') if args.compare else {}
    comparisons = {}
    selected = list(dict.fromkeys(args.figures))
    for number in selected:
        name, build = FIGURES[number]
        build(ctx)
        if args.compare:
            reference = ctx.record('figures/reference/' + name + '.pdf')
            comparisons[name] = compare_pdf(output / (name + '.pdf'), reference)
            expected = references['figures'][name]
            if digest(reference) != expected['pdf_sha256']:
                raise ValueError(name + ': manuscript reference hash differs')
            with Image.open(output / (name + '.png')) as preview:
                pixels = preview.convert('RGBA')
                comparisons[name]['png_width_px'] = pixels.width
                comparisons[name]['png_height_px'] = pixels.height
                comparisons[name]['png_pixels_sha256'] = hashlib.sha256(pixels.tobytes()).hexdigest()
                comparisons[name]['png_pixel_identical'] = (
                    comparisons[name]['png_pixels_sha256'] == expected['png_pixels_sha256']
                    and [pixels.width, pixels.height] == expected['png_size']
                )
        print('Generated ' + name, flush=True)
    overview(output, selected)
    with pymupdf.open() as combined:
        for number in selected:
            with pymupdf.open(output / (FIGURES[number][0] + '.pdf')) as source:
                combined.insert_pdf(source)
        combined.save(output / 'paper_figures.pdf', garbage=4, deflate=True, no_new_id=True)
    runtime = {'python': platform.python_version()}
    for package in ('numpy', 'pandas', 'matplotlib', 'Pillow', 'PyMuPDF', 'CairoSVG'):
        runtime[package] = importlib.metadata.version(package)
    report = {
        'figures': [FIGURES[number][0] for number in selected],
        'runtime': runtime, 'font': {'family': 'Arial', 'sha256': digest(font_path)},
        'inputs_sha256': ctx.inputs,
        'code_sha256': {str(path.relative_to(ROOT)): digest(path)
                        for path in sorted((ROOT / 'scripts').glob('*figure*.py'))},
        'outputs_sha256': {str(path.relative_to(output)): digest(path)
                           for path in sorted(output.rglob('*'))
                           if path.is_file() and path.name != 'figure_verification.json'},
        'manuscript_comparison': comparisons,
    }
    (output / 'figure_verification.json').write_text(json.dumps(report, indent=2) + '\n')
    if args.compare and not all(result['pixel_identical'] and result['png_pixel_identical']
                                for result in comparisons.values()):
        raise SystemExit('Figure comparison found rendering differences; see figure_verification.json')


if __name__ == '__main__':
    main()
