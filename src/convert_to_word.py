"""
Word Document Generator for Mbaegbu Chinedum Grace's Dissertation.
Converts the academic markdown text, tables, and generated figures into a
publication-grade Microsoft Word (.docx) document formatted to university standards.
"""

import os
import re
import docx
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

def set_cell_background(cell, fill_hex):
    tcPr = cell._tc.get_or_add_tcPr()
    tcPr.append(parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>'))

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{m}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

def create_styled_document():
    doc = Document()
    
    # 1-inch margins all around (Standard Nigerian Academic Formatting)
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)
        
    # Set default style to Times New Roman, 12pt, 1.5 line spacing
    style = doc.styles['Normal']
    font = style.font
    font.name = 'Times New Roman'
    font.size = Pt(12)
    font.color.rgb = RGBColor(0, 0, 0)
    style.paragraph_format.line_spacing = 1.5
    style.paragraph_format.space_after = Pt(6)
    
    return doc

def add_header_footer(doc):
    for section in doc.sections:
        footer = section.footer
        p = footer.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        p.text = "Miva Open University | Department of Data Science"
        p.runs[0].font.name = 'Times New Roman'
        p.runs[0].font.size = Pt(9)
        p.runs[0].font.italic = True
        p.runs[0].font.color.rgb = RGBColor(128, 128, 128)

def add_title_page(doc):
    p_inst = doc.add_paragraph()
    p_inst.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_inst = p_inst.add_run("MIVA OPEN UNIVERSITY, ABUJA, NIGERIA\nSCHOOL OF COMPUTING\nDEPARTMENT OF DATA SCIENCE\n\n\n")
    r_inst.bold = True
    r_inst.font.size = Pt(14)
    r_inst.font.name = 'Times New Roman'
    
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_title = p_title.add_run("CREDIT RISK ANALYSIS AND LOAN DEFAULT PREDICTION USING MACHINE LEARNING MODEL AND DATA VISUALIZATION\n\n\n")
    r_title.bold = True
    r_title.font.size = Pt(16)
    r_title.font.name = 'Times New Roman'
    r_title.font.color.rgb = RGBColor(139, 0, 0) # Deep Crimson
    
    p_by = doc.add_paragraph()
    p_by.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_by = p_by.add_run("BY:\n")
    r_by.font.size = Pt(12)
    r_name = p_by.add_run("MBAEGBU CHINEDUM GRACE\n")
    r_name.bold = True
    r_name.font.size = Pt(14)
    r_mat = p_by.add_run("MATRICULATION NUMBER: 2025/A/DSC/0058\n\n\n")
    r_mat.font.size = Pt(12)
    
    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_sub = p_sub.add_run(
        "A PROJECT SUBMITTED TO THE DEPARTMENT OF DATA SCIENCE, SCHOOL OF COMPUTING, "
        "MIVA OPEN UNIVERSITY ABUJA, NIGERIA.\n"
        "IN PARTIAL FULFILLMENT OF THE REQUIREMENTS FOR THE AWARD OF THE "
        "BACHELOR OF SCIENCE (HONOURS) DEGREE IN DATA SCIENCE\n\n\n"
    )
    r_sub.font.size = Pt(11)
    r_sub.font.italic = True
    
    p_date = doc.add_paragraph()
    p_date.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r_date = p_date.add_run("AUGUST, 2026")
    r_date.bold = True
    r_date.font.size = Pt(12)
    
    doc.add_page_break()

def parse_markdown_table(lines):
    rows = []
    for line in lines:
        line = line.strip()
        if not line or not line.startswith('|'):
            continue
        cells = [c.strip() for c in line.split('|')[1:-1]]
        if all(re.match(r'^:?-+:?$', c) for c in cells):
            continue # separator line
        rows.append(cells)
    return rows

def add_table_to_doc(doc, table_data):
    if not table_data:
        return
    n_rows = len(table_data)
    n_cols = len(table_data[0])
    
    table = doc.add_table(rows=n_rows, cols=n_cols)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = True
    
    for r_idx, row in enumerate(table_data):
        for c_idx, val in enumerate(row):
            if c_idx >= n_cols:
                continue
            cell = table.cell(r_idx, c_idx)
            cell.text = val
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            p.paragraph_format.line_spacing = 1.15
            p.paragraph_format.space_after = Pt(2)
            
            run = p.runs[0] if p.runs else p.add_run()
            run.font.name = 'Times New Roman'
            run.font.size = Pt(10)
            
            set_cell_margins(cell, top=120, bottom=120, left=150, right=150)
            
            if r_idx == 0: # Header
                run.bold = True
                run.font.color.rgb = RGBColor(255, 255, 255)
                set_cell_background(cell, "8B0000") # Crimson header
            else:
                if r_idx % 2 == 1:
                    set_cell_background(cell, "F9F9F9") # Zebra striping
                else:
                    set_cell_background(cell, "FFFFFF")
    
    p_sp = doc.add_paragraph()
    p_sp.paragraph_format.space_after = Pt(6)

def add_markdown_section(doc, md_text, results_dir="results"):
    lines = md_text.split('\n')
    i = 0
    in_table = False
    table_lines = []
    
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        
        # Check table
        if stripped.startswith('|'):
            table_lines.append(stripped)
            i += 1
            continue
        elif table_lines:
            t_data = parse_markdown_table(table_lines)
            add_table_to_doc(doc, t_data)
            table_lines = []
            
        if not stripped:
            i += 1
            continue
            
        # Headings
        if stripped.startswith('# '):
            text = stripped.lstrip('# ').strip()
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(18)
            p.paragraph_format.space_after = Pt(10)
            p.paragraph_format.keep_with_next = True
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = p.add_run(text)
            run.bold = True
            run.font.size = Pt(15)
            run.font.name = 'Times New Roman'
            run.font.color.rgb = RGBColor(139, 0, 0)
        elif stripped.startswith('## '):
            text = stripped.lstrip('# ').strip()
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(14)
            p.paragraph_format.space_after = Pt(6)
            p.paragraph_format.keep_with_next = True
            run = p.add_run(text)
            run.bold = True
            run.font.size = Pt(13)
            run.font.name = 'Times New Roman'
        elif stripped.startswith('### '):
            text = stripped.lstrip('# ').strip()
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(10)
            p.paragraph_format.space_after = Pt(4)
            p.paragraph_format.keep_with_next = True
            run = p.add_run(text)
            run.bold = True
            run.font.size = Pt(12)
            run.font.name = 'Times New Roman'
        elif stripped.startswith('#### '):
            text = stripped.lstrip('# ').strip()
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(8)
            p.paragraph_format.space_after = Pt(2)
            p.paragraph_format.keep_with_next = True
            run = p.add_run(text)
            run.bold = True
            run.font.italic = True
            run.font.size = Pt(11.5)
            run.font.name = 'Times New Roman'
        elif stripped == '---':
            # Separator / Page Break
            pass
        elif stripped.startswith('- ') or stripped.startswith('* '):
            # Bullet point
            clean_text = stripped[2:].strip()
            p = doc.add_paragraph(style='List Bullet')
            p.paragraph_format.line_spacing = 1.3
            p.paragraph_format.space_after = Pt(4)
            _add_formatted_runs(p, clean_text)
        elif re.match(r'^\d+\.\s', stripped):
            # Numbered list
            clean_text = re.sub(r'^\d+\.\s', '', stripped)
            p = doc.add_paragraph(style='List Number')
            p.paragraph_format.line_spacing = 1.3
            p.paragraph_format.space_after = Pt(4)
            _add_formatted_runs(p, clean_text)
        else:
            # Regular paragraph
            p = doc.add_paragraph()
            p.paragraph_format.line_spacing = 1.5
            p.paragraph_format.space_after = Pt(6)
            _add_formatted_runs(p, stripped)
            
        i += 1
        
    if table_lines:
        t_data = parse_markdown_table(table_lines)
        add_table_to_doc(doc, t_data)

def _add_formatted_runs(paragraph, text):
    """Parses inline bold (**text**), italics (*text*), and math ($text$)."""
    # Replace math markers with clean representations
    text = text.replace('$$', '').replace('$', '')
    
    parts = re.split(r'(\*\*.*?\*\*|\*.*?\*)', text)
    for part in parts:
        if not part:
            continue
        if part.startswith('**') and part.endswith('**'):
            r = paragraph.add_run(part[2:-2])
            r.bold = True
        elif part.startswith('*') and part.endswith('*'):
            r = paragraph.add_run(part[1:-1])
            r.italic = True
        else:
            r = paragraph.add_run(part)
        r.font.name = 'Times New Roman'

def embed_figures_into_chapter_4(doc, results_dir="results"):
    figures = [
        ("Figure 4.1: Receiver Operating Characteristic (ROC) & Precision-Recall Curves — Personal Loans", os.path.join(results_dir, "personal_roc_pr_curves.png")),
        ("Figure 4.2: Expected Financial Loss Curves vs Decision Threshold — Personal Loans", os.path.join(results_dir, "personal_cost_curve.png")),
        ("Figure 4.3: Receiver Operating Characteristic (ROC) & Precision-Recall Curves — Retail SME Loans", os.path.join(results_dir, "retail_roc_pr_curves.png")),
        ("Figure 4.4: Expected Financial Loss Curves vs Decision Threshold — Retail SME Loans", os.path.join(results_dir, "retail_cost_curve.png")),
        ("Figure 4.5: Global Feature Attribution (SHAP Beeswarm Plot) — Personal Loans", os.path.join(results_dir, "personal_shap_beeswarm.png")),
        ("Figure 4.6: Global Risk Driver Importance (Mean |SHAP|) — Personal Loans", os.path.join(results_dir, "personal_shap_summary.png")),
        ("Figure 4.7: Global Feature Attribution (SHAP Beeswarm Plot) — Retail SME Loans", os.path.join(results_dir, "retail_shap_beeswarm.png")),
        ("Figure 4.8: Global Risk Driver Importance (Mean |SHAP|) — Retail SME Loans", os.path.join(results_dir, "retail_shap_summary.png")),
    ]
    
    p_head = doc.add_paragraph()
    p_head.paragraph_format.space_before = Pt(16)
    r_h = p_head.add_run("4.9 EMPIRICAL EVALUATION CHARTS AND SHAP VISUALIZATIONS")
    r_h.bold = True
    r_h.font.size = Pt(13)
    r_h.font.name = 'Times New Roman'
    
    for caption, img_path in figures:
        if os.path.exists(img_path):
            p_img = doc.add_paragraph()
            p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p_img.paragraph_format.space_before = Pt(12)
            p_img.paragraph_format.space_after = Pt(4)
            p_img.paragraph_format.keep_with_next = True
            doc.add_picture(img_path, width=Inches(6.0))
            
            p_cap = doc.add_paragraph()
            p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p_cap.paragraph_format.space_after = Pt(14)
            r_cap = p_cap.add_run(caption)
            r_cap.font.name = 'Times New Roman'
            r_cap.font.size = Pt(10.5)
            r_cap.font.italic = True
            r_cap.font.color.rgb = RGBColor(60, 60, 60)

def generate_full_dissertation_docx():
    print("[*] Generating Full Complete Dissertation (.docx)...")
    doc = create_styled_document()
    add_header_footer(doc)
    add_title_page(doc)
    
    # Read chapter files in order
    chap_files = [
        "dissertation/Chapter_1_Introduction.md",
        "dissertation/Chapter_2_Literature_Review.md",
        "dissertation/Chapter_3_Methodology_System_Design.md",
        "dissertation/Chapter_4_Results_and_Discussion.md",
        "dissertation/Chapter_5_Conclusion_and_Recommendations.md"
    ]
    
    for idx, path in enumerate(chap_files, 1):
        if os.path.exists(path):
            print(f"    Processing {path}...")
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            add_markdown_section(doc, content)
            
            if idx == 4:
                # Embed the high-res generated figures directly in Chapter 4
                embed_figures_into_chapter_4(doc, results_dir="results")
                
            doc.add_page_break()
            
    out_path = "dissertation/MBAEGBU_CHINEDUM_GRACE_CREDIT_RISK_DISSERTATION.docx"
    doc.save(out_path)
    print(f"[OK] Full Dissertation saved to: {out_path}")
    return out_path

def generate_individual_chapter_docx(md_path, out_path, is_chapter_4=False):
    print(f"[*] Generating Chapter Word Document: {out_path}...")
    doc = create_styled_document()
    add_header_footer(doc)
    
    with open(md_path, 'r', encoding='utf-8') as f:
        content = f.read()
    add_markdown_section(doc, content)
    
    if is_chapter_4:
        embed_figures_into_chapter_4(doc, results_dir="results")
        
    doc.save(out_path)
    print(f"[OK] Saved: {out_path}")

if __name__ == "__main__":
    generate_full_dissertation_docx()
    generate_individual_chapter_docx("dissertation/Chapter_4_Results_and_Discussion.md", "dissertation/Chapter_4_Results_and_Discussion.docx", is_chapter_4=True)
    generate_individual_chapter_docx("dissertation/Chapter_5_Conclusion_and_Recommendations.md", "dissertation/Chapter_5_Conclusion_and_Recommendations.docx", is_chapter_4=False)
