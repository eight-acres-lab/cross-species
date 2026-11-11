"""Method and animal-training schematics for Figures 1 and 5."""
from pathlib import Path
import copy
import io
import shutil
import xml.etree.ElementTree as ET

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, FancyBboxPatch, Rectangle, Arc
from PIL import Image

C = dict(human='#199B8D', animal='#4267A3', rose='#D86273', gold='#B49A44',
         ink='#454C54', pale='#E7EBF0', rabbit='#A78374')
STYLE = {'font.family':'Arial','font.size':7,'axes.labelsize':7,
         'xtick.labelsize':6.5,'ytick.labelsize':6.5,'axes.linewidth':.75,
         'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,
         'svg.fonttype':'none','savefig.facecolor':'white','figure.facecolor':'white'}
_CONTEXT = None


def build_figure_1(ctx):
    """Render the editable method diagram with separately positioned PDF text."""
    import cairosvg
    import pymupdf

    source = ctx.asset('Fig1_method.svg')
    tree = ET.fromstring(source.read_bytes())
    ns = '{http://www.w3.org/2000/svg}'
    artwork = copy.deepcopy(tree)
    labels = []
    for parent in artwork.iter():
        for child in list(parent):
            if child.tag == ns + 'text':
                labels.append(dict(child.attrib, text=''.join(child.itertext())))
                parent.remove(child)
    with plt.rc_context({**plt.rcParamsDefault, **STYLE}):
        background = pymupdf.open(
            stream=cairosvg.svg2pdf(bytestring=ET.tostring(artwork)), filetype='pdf'
        )
        figure = plt.figure(figsize=(180 / 25.4, 160 / 25.4))
        axis = figure.add_axes([0, 0, 1, 1])
        axis.set(xlim=(0, 180), ylim=(160, 0))
        axis.axis('off')
        for label in labels:
            axis.text(
                float(label['x']), float(label['y']), label['text'],
                fontsize=float(label['font-size']) * 72 / 25.4,
                fontfamily='Arial',
                fontweight='bold' if label.get('font-weight') == '700' else 'normal',
                color=label.get('fill', '#000000'), va='baseline',
                ha={'middle': 'center', 'start': 'left', 'end': 'right'}[
                    label.get('text-anchor', 'start')
                ],
            )
        buffer = io.BytesIO()
        figure.savefig(buffer, format='pdf', transparent=True,
                       metadata={'CreationDate': None, 'ModDate': None})
        plt.close(figure)
        text_pdf = pymupdf.open(stream=buffer.getvalue(), filetype='pdf')
        background[0].show_pdf_page(background[0].rect, text_pdf, 0)
        background.set_metadata({'creator': 'paper_figure_schematics.py',
                                 'title': 'Figure 1: Data and method relationships',
                                 'creationDate': '', 'modDate': ''})
        target = ctx.output / 'Fig1_method'
        background.save(target.with_suffix('.pdf'), garbage=4, deflate=True, no_new_id=True)
        background[0].get_pixmap(matrix=pymupdf.Matrix(450 / 72, 450 / 72)).save(
            target.with_suffix('.png')
        )
        shutil.copyfile(source, target.with_suffix('.svg'))
        text_pdf.close()
        background.close()
    ctx.write_json('Fig1_method.json', {
        'type': 'schematic', 'editable_source': 'figures/assets/Fig1_method.svg',
        'panels': ['response differences', 'kernel prediction', 'correspondence controls'],
        'width_mm': 180, 'height_mm': 160,
    })


def build_figure_5(ctx):
    """Draw the two animal-training experiments and their matched comparisons."""
    global _CONTEXT
    _CONTEXT = ctx
    source_root = 'docs/manuscript_rewrite_20261109/figures/three_stage/source_data/'
    cohorts = ctx.load_csv(source_root + 'Fig6_cohorts.csv')
    axes = ctx.load_csv(source_root + 'Fig6_gene_axes.csv')
    budgets = ctx.load_csv(source_root + 'Fig6_budgets.csv')
    assert cohorts['cells'].tolist() == [6169, 3563, 12136, 1644, 638]
    assert axes['observed_genes'].tolist() == [6357, 4633]
    assert (budgets['human_added_cells'] == budgets['animal_added_cells']).all()
    with plt.rc_context({**plt.rcParamsDefault, **STYLE}):
        figure, panels = plt.subplots(2, 1, figsize=(180 / 25.4, 127 / 25.4))
        figure.subplots_adjust(left=.025, right=.985, top=.96, bottom=.04, hspace=.20)
        for panel in panels:
            panel.set(xlim=(0, 172), ylim=(0, 56))
            panel.set_aspect('equal')
            panel.axis('off')
        lane(panels[0], 'iPSC')
        lane(panels[1], 'LPS')
        ctx.save(figure, 'Fig5_augmentation_design', list(panels),
                 png_dpi=450, pdf_dpi=450, portable_svg=True)
    ctx.write_csv('Fig5_cohorts.csv', cohorts)
    ctx.write_csv('Fig5_gene_axes.csv', axes)
    ctx.write_csv('Fig5_budgets.csv', budgets)


def tint(col,a=.18):
    from matplotlib.colors import to_rgb
    return tuple(1-a+a*x for x in to_rgb(col))

def text(ax,x,y,s,size=7,bold=False,ha='center',va='center'):
    return ax.text(x,y,s,fontsize=size,fontweight='bold' if bold else 'normal',ha=ha,va=va,color='#151A20')

def line(ax,xy,co=None,lw=.8):
    ax.plot(*np.asarray(xy).T,color=co or C['ink'],lw=lw,solid_capstyle='round')

def arrow(ax,p,q,co=None):
    ax.annotate('',q,p,arrowprops=dict(arrowstyle='-|>',lw=.85,color=co or C['ink'],shrinkA=0,shrinkB=0,mutation_scale=7))

def icon(ax,name,x,y,w,h):
    p = _CONTEXT.asset(name + '.png')
    im=Image.open(p).convert('RGBA');bbox=im.getbbox();im=im.crop(bbox)
    iw,ih=im.size;ww=min(w,h*iw/ih);hh=ww*ih/iw
    ax.imshow(im,extent=(x-ww/2,x+ww/2,y-hh/2,y+hh/2),zorder=3,interpolation='lanczos')

def cell(ax,x,y,col,r=.75,open_=False):
    ax.add_patch(Circle((x,y),r,facecolor='white' if open_ else tint(col,.28),edgecolor=col,lw=.65))
    ax.add_patch(Circle((x+.08,y),r*.28,facecolor=col,edgecolor='none'))

def pool(ax,x,y,col,n=6,open_=False):
    for i in range(n):cell(ax,x+(i%3)*2.0,y+(i//3)*2.0,col,open_=open_)

def dish(ax,x,y,col):
    ax.add_patch(Ellipse((x,y),10,4.8,facecolor=tint(col,.10),edgecolor='#87929B',lw=.7))
    for dx,dy in [(-2,-.4),(0,.7),(2,-.2)]:cell(ax,x+dx,y+dy,col,.7)

def assay(ax,x,y,label):
    dish(ax,x-6,y,C['ink']);dish(ax,x+6,y,C['rose'])
    text(ax,x-6,y-4.3,'NTC' if label!='LPS' else 'Control',6.5);text(ax,x+6,y-4.3,label,6.5)

def matrix(ax,x,y,col,rows=5,cols=3):
    for i in range(rows):
        for j in range(cols):ax.add_patch(Rectangle((x+j*1.65,y+i*1.65),1.3,1.3,facecolor=tint(col,.25 if j%2 else .65),edgecolor='none'))

def model(ax,x,y):
    for off in [2.0,1.0,0]:
        ax.add_patch(FancyBboxPatch((x-7+off,y-7+off),14,14,boxstyle='round,pad=.25,rounding_size=1',facecolor=tint(C['animal'],.05),edgecolor='#8897A7',lw=.8,zorder=2))
    for i in range(3):
        for j in range(3):ax.add_patch(Circle((x-3+i*3,y+2+j*1.5),.48,fc=C['animal'],ec='none',zorder=3))
    text(ax,x,y-2,'State',8,True)

def lock(ax,x,y):
    ax.add_patch(Arc((x,y+1),3,4,theta1=0,theta2=180,lw=.85,color=C['ink']))
    ax.add_patch(FancyBboxPatch((x-2,y-2),4,3.5,boxstyle='round,pad=.1,rounding_size=.4',fc='white',ec=C['ink'],lw=.8))
    ax.add_patch(Circle((x,y),.3,fc=C['ink']))

def prediction(ax,x,y):
    matrix(ax,x-7,y-4,C['rose']);matrix(ax,x+7,y-4,C['human'])
    text(ax,x-5,y-7,'Pred.',6.5);text(ax,x+9,y-7,'Obs.',6.5)

def lane(ax,task):
    macaque=task=='iPSC';species='Cynomolgus' if macaque else 'Rabbit'
    title='POU5F1 CRISPRi · iPSCs' if macaque else 'LPS · monocytes / phagocytes'
    text(ax,3,53,'a' if macaque else 'b',9,True,ha='left');text(ax,11,53,title,8,ha='left')
    pool(ax,9,38,C['human']);text(ax,26,40,'Human',7,ha='left');text(ax,26,35,'2 lines' if macaque else '10 donors',6.5,ha='left')
    icon(ax,'macaque_dbcls' if macaque else 'rabbit',13,17,18,21)
    text(ax,26,19,species,7,ha='left');text(ax,26,14,'2 lines' if macaque else '2 animals',6.5,ha='left')
    line(ax,[(46,16),(48,16),(48,39),(46,39)],'#A6ADB5');arrow(ax,(49,27),(54,27))
    assay(ax,65,29,'POU5F1↓' if macaque else 'LPS')
    text(ax,65,41,'Perturbation',7)
    arrow(ax,(77,29),(82,29));matrix(ax,85,25,C['animal']);matrix(ax,91,25,C['human'])
    text(ax,90,41,'scRNA-seq',7);text(ax,90,19,'6,357 genes' if macaque else '4,633 genes',6.5)
    arrow(ax,(98,29),(104,29));model(ax,114,29)
    text(ax,114,9,'H / H + equal additions',6.5)
    text(ax,114,3,'Human / M56 / M82 / A0' if macaque else 'Human / rabbit / A0',6.3)
    arrow(ax,(124,29),(137,29));prediction(ax,148,29)
    text(ax,150,43,'Held-out human',7)
    if macaque:
        text(ax,150,16,'Other human line',6.5);lock(ax,139,7);text(ax,155,7,'Frozen → exp3',6.5)
    else:
        text(ax,150,16,'Donor folds',6.5)
        for i in range(5):ax.add_patch(Rectangle((140+i*4,5),3,3,fc=C['rose'] if i==4 else '#DCE4EF',ec='none'))
