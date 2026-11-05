#!/usr/bin/env python3
"""Build an editable Word review copy from the canonical manuscript and current figures.

Requires Pandoc and python-docx. Equations use Pandoc's native Word math writer;
the five figures are embedded at their original 160 mm print width.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import tempfile
from zipfile import ZipFile

from docx import Document
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Mm, Pt, RGBColor
from lxml import etree

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / 'docs/paper'
FONT = 'Times New Roman'


def run(args, **kwargs):
    return subprocess.run(args, check=True, text=True, **kwargs)


def plain(nodes):
    parts = []
    for node in nodes:
        if node['t'] == 'Str':
            parts.append(node['c'])
        elif node['t'] in ('Space', 'SoftBreak', 'LineBreak'):
            parts.append(' ')
        elif node['t'] in ('Strong', 'Emph', 'Superscript', 'Subscript'):
            parts.append(plain(node['c']))
    return ''.join(parts)


def count_nodes(value, kind):
    if isinstance(value, dict):
        return int(value.get('t') == kind) + sum(count_nodes(v, kind) for v in value.values())
    if isinstance(value, list):
        return sum(count_nodes(v, kind) for v in value)
    return 0


def set_style(doc, name, size, *, bold=False, spacing=1.15, after=5, before=0):
    style = doc.styles[name] if name in doc.styles else doc.styles.add_style(name, WD_STYLE_TYPE.PARAGRAPH)
    style.font.name = FONT
    style.font.size = Pt(size)
    style.font.bold = bold
    style.font.italic = False
    style.font.color.rgb = RGBColor(0, 0, 0)
    style.font.underline = False
    if style.type == WD_STYLE_TYPE.PARAGRAPH:
        fmt = style.paragraph_format
        fmt.space_before, fmt.space_after = Pt(before), Pt(after)
        fmt.line_spacing = spacing
        fmt.widow_control = True
        fmt.first_line_indent = Mm(0)
    return style


def flag(parent, tag, value=None):
    element = parent.find(qn(tag))
    if element is None:
        element = OxmlElement(tag)
        parent.append(element)
    if value is not None:
        element.set(qn('w:val'), str(value))
    return element


def page_number(paragraph):
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    field = OxmlElement('w:fldSimple')
    field.set(qn('w:instr'), 'PAGE')
    r = OxmlElement('w:r')
    t = OxmlElement('w:t'); t.text = '1'
    r.append(t); field.append(r); paragraph._p.append(field)


def format_tables(doc):
    fractions = [[.16, .23, .27, .34], [.53, .16, .23, .08],
                 [.15, .16, .23, .23, .23], [.045, .32, .265, .37]]
    for number, (table, widths) in enumerate(zip(doc.tables, fractions), 1):
        table.autofit = False
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        pr = table._tbl.tblPr
        table_width = flag(pr, 'w:tblW'); table_width.set(qn('w:type'), 'dxa')
        table_width.set(qn('w:w'), str(round(Mm(160).twips)))
        borders = flag(pr, 'w:tblBorders')
        for name in ('top', 'left', 'bottom', 'right', 'insideH', 'insideV'):
            e = flag(borders, 'w:'+name, 'single')
            e.set(qn('w:sz'), '4'); e.set(qn('w:color'), 'D9D9D9')
        margins = flag(pr, 'w:tblCellMar')
        for name, value in [('top', 65), ('bottom', 65), ('left', 80), ('right', 80)]:
            e = flag(margins, 'w:'+name); e.set(qn('w:w'), str(value)); e.set(qn('w:type'), 'dxa')
        for col, frac in zip(table.columns, widths):
            col.width = Mm(160*frac)
        for ri, row in enumerate(table.rows):
            trpr = row._tr.get_or_add_trPr()
            flag(trpr, 'w:cantSplit')
            if ri == 0:
                flag(trpr, 'w:tblHeader')
            for ci, (cell, frac) in enumerate(zip(row.cells, widths)):
                cell.width = Mm(160*frac)
                cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
                shade = flag(cell._tc.get_or_add_tcPr(), 'w:shd', 'clear')
                shade.set(qn('w:fill'), 'EEF1F4' if ri == 0 else 'FFFFFF')
                for p in cell.paragraphs:
                    p.style = doc.styles['Review Table']
                    p.paragraph_format.keep_with_next = ri == 0
                    p.paragraph_format.keep_together = True
                    p.alignment = (WD_ALIGN_PARAGRAPH.CENTER if (number < 4 and ci > 0) or
                                   (number == 4 and ci == 0) else WD_ALIGN_PARAGRAPH.LEFT)
                    if ri == 0:
                        for r in p.runs:
                            r.bold = True
        following = table._tbl.getnext()
        if following is not None and following.tag == qn('w:p'):
            for p in doc.paragraphs:
                if p._p is following:
                    p.paragraph_format.space_before = Pt(6)
                    break


def normalize_math(doc):
    """Keep upright identifiers intact across Word and LibreOffice importers."""
    def upright(run):
        text = run.find(qn('m:t'))
        style = run.find(qn('m:rPr')+'/'+qn('m:sty'))
        return (run.tag == qn('m:r') and text is not None and (text.text or '').isalpha()
                and style is not None and style.get(qn('m:val')) == 'p')

    for p in doc.paragraphs:
        for math in p._p.xpath('.//m:oMath'):
            # Use the native overbar object: some importers render an overline
            # encoded as a generic accent as an acute accent instead.
            for accent in list(math.iter(qn('m:acc'))):
                char = accent.find(qn('m:accPr')+'/'+qn('m:chr'))
                if char is not None and char.get(qn('m:val')) in {'\u203e', '\u00af', '\u0304'}:
                    bar = OxmlElement('m:bar'); pr = OxmlElement('m:barPr')
                    pos = OxmlElement('m:pos'); pos.set(qn('m:val'), 'top')
                    pr.append(pos); bar.append(pr); bar.append(accent.find(qn('m:e')))
                    accent.getparent().replace(accent, bar)
            # Pandoc emits \\mathrm{rel} as three separate upright runs. Joining
            # them prevents LibreOffice from spacing them as three variables.
            for parent in list(math.iter()):
                children = list(parent)
                i = 0
                while i < len(children):
                    first = children[i]
                    if not upright(first):
                        i += 1
                        continue
                    j = i+1
                    while j < len(children) and upright(children[j]):
                        first.find(qn('m:t')).text += children[j].find(qn('m:t')).text
                        parent.remove(children[j]); j += 1
                    pr = first.find(qn('m:rPr'))
                    if pr.find(qn('m:nor')) is None:
                        pr.insert(0, OxmlElement('m:nor'))
                    i = j
            for r in math.iter(qn('m:r')):
                pr = r.find(qn('w:rPr'))
                if pr is None:
                    pr = OxmlElement('w:rPr')
                    r.insert(1 if r.find(qn('m:rPr')) is not None else 0, pr)
                size = '20' if p.style.name == 'Caption' else '22'
                flag(pr, 'w:sz', size); flag(pr, 'w:szCs', size)


def format_document(doc, title):
    set_style(doc, 'Normal', 11)
    for name in ('Body Text', 'First Paragraph', 'Compact'):
        set_style(doc, name, 11)
    set_style(doc, 'Title', 17, bold=True, spacing=1.05, after=10)
    set_style(doc, 'Heading 1', 13, bold=True, spacing=1.05, before=14, after=7)
    set_style(doc, 'Heading 2', 11, bold=True, spacing=1.05, before=12, after=6)
    set_style(doc, 'Caption', 10, spacing=1.05, after=7)
    set_style(doc, 'Review Front Matter', 10, spacing=1.05, after=5)
    set_style(doc, 'Review Abstract', 10.5, spacing=1.1, after=5)
    set_style(doc, 'Review Reference', 10.5, spacing=1.05, after=5)
    set_style(doc, 'Review Table', 10, spacing=1.05, after=0)
    set_style(doc, 'Header', 9, spacing=1, after=0)
    set_style(doc, 'Footer', 9, spacing=1, after=0)
    for name in ('Heading 1', 'Heading 2'):
        doc.styles[name].paragraph_format.keep_with_next = True
    for name in ('Verbatim Char', 'Source Code'):
        if name in doc.styles:
            doc.styles[name].font.name = 'Courier New'
            doc.styles[name].font.size = Pt(9)
    if 'Hyperlink' in doc.styles:
        doc.styles['Hyperlink'].font.color.rgb = RGBColor.from_string('174F70')
        doc.styles['Hyperlink'].font.underline = False

    for section in doc.sections:
        # Retain the manuscript's A4 geometry and 160 mm figure width.
        section.page_width, section.page_height = Mm(210), Mm(297)
        section.top_margin = section.bottom_margin = Mm(25)
        section.left_margin = section.right_margin = Mm(25)
        section.header_distance, section.footer_distance = Mm(12), Mm(12)
        section.different_first_page_header_footer = True
        header = section.header.paragraphs[0]
        header.text = 'Cross-species transcriptomic comparisons'
        header.style = doc.styles['Header']
        for footer in (section.footer, section.first_page_footer):
            p = footer.paragraphs[0]; p.style = doc.styles['Footer']; page_number(p)

    front, abstract, references = True, False, False
    figure_count = caption_count = 0
    for p in doc.paragraphs:
        text = p.text.strip()
        fmt = p.paragraph_format
        if p.style.name == 'Title':
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            continue
        if p.style.name.startswith('Heading'):
            front = False
            abstract = text == 'Abstract'
            references = text == 'References'
            fmt.keep_with_next = True
            if text in ('1. Introduction', 'References', 'Supplementary Information'):
                fmt.page_break_before = True
            continue
        if front:
            p.style = doc.styles['Review Front Matter']
            if text.startswith('Zihui Zhang'):
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                for r in p.runs:
                    r.font.size = Pt(11)
        elif abstract:
            p.style = doc.styles['Review Abstract']
        elif references:
            p.style = doc.styles['Review Reference']
            fmt.left_indent, fmt.first_line_indent = Mm(8), Mm(-8)
            fmt.keep_together = True
        if p._p.xpath('.//w:drawing'):
            figure_count += 1
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            fmt.keep_with_next = True
            fmt.keep_together = True
            fmt.space_before, fmt.space_after = Pt(6), Pt(6)
            fmt.line_spacing = 1
        elif re.match(r'^Figure [1-5]\.', text):
            caption_count += 1
            p.style = doc.styles['Caption']
            fmt.keep_together = True
            fmt.keep_with_next = False
        elif re.match(r'^Table [1-4]\.', text):
            p.style = doc.styles['Caption']
            fmt.keep_with_next = True
            fmt.keep_together = True
    assert figure_count == caption_count == 5
    assert len(doc.tables) == 4 and len(doc.inline_shapes) == 5
    for shape in doc.inline_shapes:
        ratio = shape.height / shape.width
        shape.width = Mm(160); shape.height = round(Mm(160)*ratio)
    format_tables(doc)
    normalize_math(doc)
    flag(doc.settings.element, 'w:updateFields', 'true')
    flag(doc.settings.element, 'w:doNotAutoCompressPictures')
    doc.core_properties.title = title
    doc.core_properties.author = 'Zihui Zhang; Wei Wang; Ziyan Zhou; Chen Peng; Zhi Wang'
    doc.core_properties.last_modified_by = ''
    doc.core_properties.comments = ''


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, default=PAPER/'preprint_review.docx')
    args = ap.parse_args()
    ast = json.loads(run(['pandoc', str(PAPER/'preprint.md'), '-f', 'markdown', '-t', 'json'],
                         capture_output=True).stdout)
    source_math_count = count_nodes(ast, 'Math')
    title_nodes = ast['blocks'][0]['c'][2]
    title = plain(title_nodes)
    ast['meta']['title'] = {'t': 'MetaInlines', 'c': title_nodes}
    ast['blocks'] = [b for b in ast['blocks'][1:] if b['t'] != 'HorizontalRule']
    for block in ast['blocks']:
        if block['t'] == 'Header':
            block['c'][0] -= 1
        elif block['t'] == 'Para' and block['c'][0]['t'] == 'Image':
            image = block['c'][0]
            image['c'][0][2].append(['width', '160mm'])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='preprint-word-') as temporary:
        intermediate = Path(temporary)/'converted.docx'
        run(['pandoc', '-f', 'json', '-t', 'docx', '--standalone',
             '--resource-path', str(PAPER), '-o', str(intermediate)], input=json.dumps(ast))
        doc = Document(intermediate)
        before = ''.join(e.text or '' for e in doc.element.iter()
                         if e.tag in {qn('w:t'), qn('m:t')})
        format_document(doc, title)
        after = ''.join(e.text or '' for e in doc.element.iter()
                        if e.tag in {qn('w:t'), qn('m:t')})
        assert before == after, 'Formatting changed manuscript text or equation characters.'
        doc.save(args.out)
    with ZipFile(args.out) as z:
        xml = etree.fromstring(z.read('word/document.xml'))
        ns = {'m': 'http://schemas.openxmlformats.org/officeDocument/2006/math',
              'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
        actual_math = len(xml.xpath('//m:oMath', namespaces=ns))
        assert actual_math == source_math_count, (actual_math, source_math_count)
        settings = etree.fromstring(z.read('word/settings.xml'))
        assert not settings.xpath('//w:documentProtection', namespaces=ns)
    print(f'Wrote {args.out}: 5 figures, 4 editable tables, {actual_math} native Word equations; editing unrestricted.')


if __name__ == '__main__':
    main()
