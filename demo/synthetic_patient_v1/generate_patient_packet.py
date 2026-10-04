from __future__ import annotations

import hashlib, json, os, random, shutil, textwrap
from pathlib import Path
from typing import List, Tuple

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.pdfbase.acroform import AcroForm
from reportlab.graphics.barcode import code128
from reportlab.lib.utils import ImageReader
from PIL import Image, ImageOps, ImageEnhance, ImageFilter, ImageDraw
import fitz

OUT = Path(__file__).resolve().parent
# Regenerates the v4 PDFs and refreshes fixture SHA256 mappings in place.

W,H = letter
BLACK = colors.HexColor('#111111')
MID = colors.HexColor('#555555')
LIGHT = colors.HexColor('#bcbcbc')
PALE = colors.HexColor('#eeeeee')

# ---------- low-level drawing ----------
def set_font(c, name='Helvetica', size=8):
    c.setFont(name,size)
    c.setFillColor(BLACK)

def line(c, y, x1=36, x2=W-36, width=.5, color=colors.HexColor('#777777')):
    c.setStrokeColor(color); c.setLineWidth(width); c.line(x1,y,x2,y)

def wrap(c, text, x, y, width, font='Helvetica', size=8.2, leading=10.5, bold_prefix=None):
    # simple measured wrapping with explicit newlines
    c.setFont(font,size); c.setFillColor(BLACK)
    for para in str(text).split('\n'):
        if para == '': y -= leading; continue
        words=para.split(' '); cur=''
        lines=[]
        for word in words:
            test=(cur+' '+word).strip()
            if c.stringWidth(test,font,size) <= width:
                cur=test
            else:
                if cur: lines.append(cur)
                cur=word
        if cur: lines.append(cur)
        for ln in lines:
            c.drawString(x,y,ln); y-=leading
    return y

def centered(c, text, y, font='Helvetica-Bold', size=10):
    c.setFont(font,size); c.setFillColor(BLACK); c.drawCentredString(W/2,y,text)

def small_synth(c):
    c.saveState(); c.setFillColor(colors.HexColor('#777777')); c.setFont('Helvetica',5.2)
    c.drawString(34,15,'SYNTHETIC DEMO RECORD - fictional patient/institution - not for clinical use')
    c.restoreState()

def patient_strip(c, report_no, date, title=None):
    set_font(c,'Helvetica-Bold',8.2)
    c.drawString(38,H-42,'MERIDIAN VALE HEALTH SYSTEM')
    set_font(c,'Helvetica',6.4); c.setFillColor(MID)
    c.drawRightString(W-38,H-42,'PRINTED FROM ELECTRONIC MEDICAL RECORD')
    line(c,H-49,38,W-38,.55,BLACK)
    set_font(c,'Helvetica-Bold',7.2)
    c.drawString(38,H-63,'Patient: ROWAN, MAYA')
    c.drawString(230,H-63,'DOB: 11/02/1966')
    c.drawString(350,H-63,'MRN: MV000017')
    c.drawRightString(W-38,H-63,f'Date: {date}')
    set_font(c,'Helvetica',6.6)
    c.drawString(38,H-75,f'Report/Accession: {report_no}')
    if title:
        c.drawRightString(W-38,H-75,title)
    line(c,H-82,38,W-38,.3,colors.HexColor('#888888'))

def page_footer(c, page, total=None):
    set_font(c,'Helvetica',5.6); c.setFillColor(MID)
    c.drawRightString(W-36,15,f'Page {page}' + (f' of {total}' if total else ''))
    small_synth(c)

def section(c, label, y, x=40, underline=True):
    set_font(c,'Helvetica-Bold',8.3)
    c.drawString(x,y,label.upper())
    if underline:
        c.setStrokeColor(colors.HexColor('#999999')); c.setLineWidth(.25); c.line(x,y-2,W-40,y-2)
    return y-14

def kv_rows(c, rows:List[Tuple[str,str]], y, x=40, labelw=105, valw=390, lineh=13):
    for k,v in rows:
        set_font(c,'Helvetica-Bold',7.2); c.drawString(x,y,k)
        set_font(c,'Helvetica',7.3); c.drawString(x+labelw,y,v)
        y-=lineh
    return y

def barcode(c, value, x, y, width=160, height=24):
    bc=code128.Code128(value,barHeight=height,barWidth=.6,humanReadable=False)
    bc.drawOn(c,x,y)

# ---------- document constructors ----------
def pathology_doc(path, accession, date, specimen, diagnosis, biomarkers, clinical, gross, micro, comment, scanned=True, addendum=None):
    tmp=path.with_suffix('.base.pdf')
    c=canvas.Canvas(str(tmp),pagesize=letter)
    pages=2
    for pg in range(1,pages+1):
        # pathology department-style, monochrome, minimal branding
        set_font(c,'Helvetica-Bold',10.5); c.drawString(34,H-34,'MERIDIAN VALE HEALTH SYSTEM')
        set_font(c,'Helvetica-Bold',8.5); c.drawString(34,H-48,'DEPARTMENT OF PATHOLOGY AND LABORATORY MEDICINE')
        barcode(c, accession.replace('-',''), W-180,H-59,140,17)
        set_font(c,'Helvetica',5.7); c.drawRightString(W-38,H-66,accession)
        line(c,H-73,34,W-34,.8,BLACK)
        set_font(c,'Helvetica-Bold',11.2); c.drawString(34,H-90,'SURGICAL PATHOLOGY REPORT')
        set_font(c,'Helvetica',6.5); c.drawRightString(W-34,H-89,'FINAL')
        # patient table-like block
        y=H-108
        set_font(c,'Helvetica-Bold',6.6)
        c.drawString(36,y,'PATIENT'); c.drawString(190,y,'MRN'); c.drawString(300,y,'DOB'); c.drawString(400,y,'SEX')
        set_font(c,'Helvetica',7.1)
        c.drawString(36,y-11,'ROWAN, MAYA'); c.drawString(190,y-11,'MV000017'); c.drawString(300,y-11,'11/02/1966'); c.drawString(400,y-11,'F')
        line(c,y-17,34,W-34,.3,colors.HexColor('#777777'))
        y-=31
        y=kv_rows(c,[('Accession',accession),('Collected',date+' 09:18'),('Received',date+' 11:02'),('Specimen',specimen),('Ordering provider','Elena Morris, MD')],y,labelw=102,lineh=12)
        y-=5
        if pg==1:
            y=section(c,'Clinical history',y)
            y=wrap(c,clinical,42,y,W-84,size=7.7,leading=10)
            y-=5
            y=section(c,'Final diagnosis',y)
            set_font(c,'Helvetica-Bold',8.3)
            for ln in textwrap.wrap(diagnosis, 92): c.drawString(42,y,ln); y-=11
            y-=4
            y=section(c,'Breast biomarker studies',y)
            # lab-style table
            cols=[42,205,310,535]
            set_font(c,'Helvetica-Bold',6.8)
            c.drawString(cols[0],y,'TEST'); c.drawString(cols[1],y,'RESULT'); c.drawString(cols[2],y,'INTERPRETATION / DETAILS')
            line(c,y-4,42,550,.5,BLACK); y-=15
            for test,res,detail in biomarkers:
                set_font(c,'Helvetica',7.1); c.drawString(cols[0],y,test); c.drawString(cols[1],y,res)
                y=wrap(c,detail,cols[2],y,232,size=6.9,leading=8.8)
                y-=4
            y-=2
            y=section(c,'Comment',y)
            y=wrap(c,comment,42,y,W-84,size=7.5,leading=9.8)
            if addendum:
                y-=6; y=section(c,'Addendum',y)
                y=wrap(c,addendum,42,y,W-84,size=7.4,leading=9.6)
        else:
            y=section(c,'Gross description',y)
            y=wrap(c,gross,42,y,W-84,size=7.3,leading=9.6)
            y-=6; y=section(c,'Microscopic description',y)
            y=wrap(c,micro,42,y,W-84,size=7.3,leading=9.6)
            y-=7; y=section(c,'Pathologist',y)
            set_font(c,'Helvetica',7.1)
            c.drawString(42,y,'Anita Patel, MD'); y-=10
            c.drawString(42,y,'Electronically signed  '+date+' 16:41')
            y-=18
            set_font(c,'Helvetica',6.4); c.setFillColor(MID)
            c.drawString(42,y,'This report may contain results from tests performed at affiliated reference laboratories.')
        page_footer(c,pg,pages); c.showPage()
    c.save()
    if scanned:
        scanify_pdf(tmp,path,seed=int(accession[-3:].replace('A','1').replace('B','2')) if accession[-3:].isdigit() else 17, dpi=190, angle=(-0.22 if '341' in accession else 0.15), fax=False)
        tmp.unlink()
    else: tmp.replace(path)


def radiology_doc(path, accession, date, comparison, indication, findings, impression, radiologist='Daniel Cho, MD'):
    c=canvas.Canvas(str(path),pagesize=letter)
    patient_strip(c,accession,date,'RADIOLOGY')
    y=H-103
    set_font(c,'Helvetica-Bold',10.5); c.drawString(38,y,'CT CHEST ABDOMEN PELVIS W IV CONTRAST'); y-=15
    set_font(c,'Helvetica',6.5); c.setFillColor(MID)
    c.drawString(38,y,'Result status: Final'); c.drawString(170,y,'Performed: '+date+' 09:47'); c.drawString(360,y,'Accession: '+accession); y-=15
    line(c,y,38,W-38,.4,colors.HexColor('#777777')); y-=15
    y=section(c,'Narrative',y,38,False)
    set_font(c,'Helvetica-Bold',7.2); c.drawString(38,y,'CLINICAL INDICATION:'); set_font(c,'Helvetica',7.4); c.drawString(137,y,indication); y-=13
    set_font(c,'Helvetica-Bold',7.2); c.drawString(38,y,'COMPARISON:'); set_font(c,'Helvetica',7.4); c.drawString(137,y,comparison); y-=13
    set_font(c,'Helvetica-Bold',7.2); c.drawString(38,y,'TECHNIQUE:'); y-=10
    y=wrap(c,'Contrast-enhanced CT of the chest, abdomen, and pelvis. Multiplanar reformatted images were reviewed. Dose reduction techniques were utilized as appropriate.',48,y,W-92,size=7.2,leading=9.4)
    y-=4; set_font(c,'Helvetica-Bold',7.4); c.drawString(38,y,'FINDINGS:'); y-=12
    for organ,text in findings:
        set_font(c,'Helvetica-Bold',7.3); c.drawString(48,y,organ+':');
        y=wrap(c,text,102,y,W-145,size=7.2,leading=9.5); y-=2
    y-=4; set_font(c,'Helvetica-Bold',8.0); c.drawString(38,y,'IMPRESSION:'); y-=13
    for i,item in enumerate(impression,1):
        set_font(c,'Helvetica-Bold',7.6); c.drawString(48,y,f'{i}.')
        y=wrap(c,item,62,y,W-105,size=7.6,leading=10); y-=2
    y-=8
    set_font(c,'Helvetica',6.8); c.setFillColor(MID)
    c.drawString(38,y,f'Electronically signed by {radiologist} on {date} at 14:06')
    page_footer(c,1,1); c.showPage(); c.save()


def oncology_note(path, note_id, date, title, hpi, meds, assessment, plan, addendum=None, first=False):
    c=canvas.Canvas(str(path),pagesize=letter)
    # Mimics generic EHR note printout, intentionally plain
    set_font(c,'Helvetica-Bold',9.0); c.drawString(36,H-38,'MERIDIAN VALE CANCER CENTER')
    set_font(c,'Helvetica',6.4); c.setFillColor(MID); c.drawRightString(W-36,H-38,'Medical Oncology')
    line(c,H-45,36,W-36,.6,BLACK)
    set_font(c,'Helvetica-Bold',10.0); c.drawString(36,H-62,title)
    set_font(c,'Helvetica',6.5); c.drawRightString(W-36,H-62,'Note Status: Signed')
    y=H-80
    rows=[('Patient','Maya Rowan'),('MRN','MV000017'),('DOB','11/02/1966'),('Encounter Date',date),('Author','Elena Morris, MD'),('Encounter','Outpatient medical oncology')]
    # compact 2-col demographics like EHR print view
    set_font(c,'Helvetica',6.8)
    for i in range(0,len(rows),2):
        k1,v1=rows[i]; k2,v2=rows[i+1]
        c.setFont('Helvetica-Bold',6.6); c.drawString(36,y,k1+':'); c.setFont('Helvetica',6.8); c.drawString(95,y,v1)
        c.setFont('Helvetica-Bold',6.6); c.drawString(305,y,k2+':'); c.setFont('Helvetica',6.8); c.drawString(385,y,v2); y-=12
    line(c,y-2,36,W-36,.25,colors.HexColor('#888888')); y-=16
    y=section(c,'History of present illness',y,36,False); y=wrap(c,hpi,42,y,W-84,size=7.5,leading=9.7)
    y-=7; y=section(c,'Current cancer therapy',y,36,False)
    for m in meds:
        set_font(c,'Helvetica',7.4); c.drawString(47,y,'- '+m); y-=10
    y-=4; y=section(c,'Assessment',y,36,False); y=wrap(c,assessment,42,y,W-84,size=7.5,leading=9.7)
    y-=6; y=section(c,'Plan',y,36,False)
    for i,item in enumerate(plan,1):
        set_font(c,'Helvetica',7.4); c.drawString(48,y,f'{i}.'); y=wrap(c,item,62,y,W-105,size=7.4,leading=9.6); y-=2
    if addendum:
        y-=5; y=section(c,'Addendum',y,36,False); y=wrap(c,addendum,42,y,W-84,size=7.3,leading=9.5)
    y-=12
    set_font(c,'Helvetica',6.6); c.setFillColor(MID)
    c.drawString(36,y,'Electronically signed by Elena Morris, MD')
    c.drawString(280,y,'Filed: '+date+' 18:12')
    page_footer(c,1,1); c.showPage(); c.save()


def lab_doc(path):
    base=path.with_suffix('.base.pdf')
    c=canvas.Canvas(str(base),pagesize=letter)
    set_font(c,'Helvetica-Bold',9.3); c.drawString(32,H-34,'MERIDIAN VALE CLINICAL LABORATORY')
    set_font(c,'Helvetica',6.2); c.drawRightString(W-32,H-34,'FINAL RESULT')
    line(c,H-42,32,W-32,.7,BLACK)
    y=H-58
    # dense laboratory identity grid
    set_font(c,'Helvetica-Bold',6.3)
    c.drawString(34,y,'Patient:'); c.drawString(175,y,'MRN:'); c.drawString(290,y,'DOB:'); c.drawString(400,y,'Sex:')
    set_font(c,'Helvetica',6.8); c.drawString(75,y,'ROWAN, MAYA'); c.drawString(208,y,'MV000017'); c.drawString(325,y,'11/02/1966'); c.drawString(430,y,'F')
    y-=11
    set_font(c,'Helvetica-Bold',6.3); c.drawString(34,y,'Specimen:'); c.drawString(175,y,'Collected:'); c.drawString(350,y,'Received:')
    set_font(c,'Helvetica',6.8); c.drawString(82,y,'Whole blood (EDTA)'); c.drawString(225,y,'07/21/2025 08:04'); c.drawString(395,y,'07/21/2025 08:32')
    y-=11
    set_font(c,'Helvetica-Bold',6.3); c.drawString(34,y,'Ordering Provider:'); set_font(c,'Helvetica',6.8); c.drawString(118,y,'Elena Morris, MD')
    line(c,y-8,32,W-32,.35,colors.HexColor('#777777')); y-=25
    set_font(c,'Helvetica-Bold',8.0); c.drawString(34,y,'HEMATOLOGY / TUMOR MARKER'); y-=16
    cols=[34,260,335,380,445,540]
    headers=['TEST','RESULT','FLAG','UNITS','REFERENCE RANGE','STATUS']
    set_font(c,'Helvetica-Bold',6.2)
    for xx,hdr in zip(cols,headers): c.drawString(xx,y,hdr)
    line(c,y-4,32,W-32,.5,BLACK); y-=16
    rows=[
        ('WBC','2.2','L','10^3/uL','4.0 - 10.5','Final'),
        ('Absolute neutrophil count','0.8','L','10^3/uL','1.5 - 7.5','Final'),
        ('Hemoglobin','11.6','L','g/dL','12.0 - 16.0','Final'),
        ('Platelets','169','','10^3/uL','150 - 400','Final'),
        ('CA 15-3','29','','U/mL','0 - 30','Final'),
    ]
    for r in rows:
        set_font(c,'Helvetica',6.8)
        for xx,val in zip(cols,r): c.drawString(xx,y,val)
        line(c,y-4,32,W-32,.2,colors.HexColor('#b0b0b0')); y-=15
    y-=6
    set_font(c,'Helvetica-Bold',6.5); c.drawString(34,y,'Comment:'); y-=10
    y=wrap(c,'Neutropenia. Result verified on repeat analysis. Ordering clinician notified per laboratory policy.',44,y,W-88,size=6.7,leading=8.8)
    y-=10
    set_font(c,'Helvetica',6.1); c.setFillColor(MID)
    c.drawString(34,y,'Verified by: J. Kim, MT(ASCP)     07/21/2025 09:11')
    page_footer(c,1,1); c.showPage(); c.save()
    scanify_pdf(base,path,seed=621,dpi=170,angle=-0.38,fax=True); base.unlink()


def genomic_doc(path):
    c=canvas.Canvas(str(path),pagesize=letter)
    for pg in (1,2):
        # external-lab look, different institution and typography
        set_font(c,'Helvetica-Bold',12.0); c.drawString(34,H-38,'APEX MOLECULAR DIAGNOSTICS')
        set_font(c,'Helvetica',6.2); c.setFillColor(MID); c.drawString(34,H-49,'Oncology Genomics Laboratory')
        barcode(c,'AMD26009871',W-176,H-58,135,18)
        set_font(c,'Helvetica',5.8); c.drawRightString(W-35,H-66,'AMD26-009871')
        line(c,H-74,34,W-34,.7,BLACK)
        set_font(c,'Helvetica-Bold',10.0); c.drawString(34,H-92,'PLASMA COMPREHENSIVE GENOMIC PROFILING')
        set_font(c,'Helvetica',6.2); c.drawRightString(W-34,H-92,'FINAL REPORT')
        y=H-109
        y=kv_rows(c,[('Patient','Maya Rowan'),('MRN','MV000017'),('DOB','11/02/1966'),('Specimen','Peripheral blood / plasma cfDNA'),('Collected','05/02/2026 09:16'),('Report date','05/08/2026')],y,labelw=86,lineh=11)
        y-=4
        if pg==1:
            y=section(c,'Detected alteration',y,34,False)
            # intentionally bold boxed single finding, vendor-like
            c.setStrokeColor(BLACK); c.rect(34,y-64,W-68,62,fill=0,stroke=1)
            set_font(c,'Helvetica-Bold',7.0); c.drawString(44,y-14,'GENE / VARIANT'); c.drawString(280,y-14,'VARIANT ALLELE FRACTION');
            set_font(c,'Helvetica-Bold',13.0); c.drawString(44,y-40,'ESR1  p.D538G'); c.drawString(316,y-40,'3.8%')
            set_font(c,'Helvetica',7.2); c.drawString(44,y-54,'c.1613A>G'); y-=82
            y=section(c,'Interpretation',y,34,False)
            y=wrap(c,'A pathogenic ESR1 D538G alteration was detected in circulating tumor DNA. ESR1 ligand-binding-domain alterations can emerge in estrogen-receptor-positive metastatic breast cancer after aromatase inhibitor exposure. Interpret this result with the complete clinical history, pathology, imaging, current therapy, and applicable guidelines.',40,y,W-80,size=7.4,leading=9.8)
            y-=6; y=section(c,'Assay',y,34,False)
            y=wrap(c,'Hybrid-capture next-generation sequencing of plasma cell-free DNA. Reportable alteration classes include selected substitutions, insertions/deletions, rearrangements, and copy-number changes within validated regions.',40,y,W-80,size=7.2,leading=9.5)
            y-=6; y=section(c,'Limitations',y,34,False)
            y=wrap(c,'A negative result does not exclude tumor-associated alterations. Variant allele fraction is influenced by tumor shedding and total cell-free DNA. This report does not independently establish treatment eligibility.',40,y,W-80,size=7.2,leading=9.5)
        else:
            y=section(c,'Technical appendix',y,34,False)
            y=wrap(c,'Specimen quality: PASS. Targeted next-generation sequencing was performed on extracted plasma cell-free DNA. The assay is designed for somatic oncology profiling and is not a germline test.',40,y,W-80,size=7.3,leading=9.6)
            y-=8
            set_font(c,'Helvetica-Bold',6.5)
            for xx,hdr in zip([40,120,220,380,445],['GENE','VARIANT','CODING','VAF','CLASSIFICATION']): c.drawString(xx,y,hdr)
            line(c,y-4,38,W-38,.4,BLACK); y-=16
            set_font(c,'Helvetica',7.1)
            for xx,val in zip([40,120,220,380,445],['ESR1','p.D538G','c.1613A>G','3.8%','Pathogenic']): c.drawString(xx,y,val)
            y-=22
            y=section(c,'Method and limitations',y,34,False)
            y=wrap(c,'Alterations below the validated limit of detection may not be identified. Copy-number and rearrangement sensitivity varies by locus and specimen tumor fraction. Results should be interpreted by a qualified clinician in the context of other laboratory and clinical information.',40,y,W-80,size=7.3,leading=9.6)
        y-=20
        set_font(c,'Helvetica',6.6); c.setFillColor(MID)
        c.drawString(34,y,'Electronically signed: Priya Shah, PhD, FACMG - Laboratory Director')
        small_synth(c); c.drawRightString(W-34,15,f'Page {pg} of 2'); c.showPage()
    c.save()


def scanify_pdf(src:Path,dst:Path,seed=1,dpi=180,angle=.2,fax=False):
    random.seed(seed)
    doc=fitz.open(str(src))
    imgs=[]
    for i,p in enumerate(doc):
        pix=p.get_pixmap(matrix=fitz.Matrix(dpi/72,dpi/72),alpha=False)
        img=Image.frombytes('RGB',[pix.width,pix.height],pix.samples)
        img=ImageOps.grayscale(img)
        # emulate photocopy: slight paper tone, contrast and sharpness irregularity
        if fax:
            img=ImageEnhance.Contrast(img).enhance(1.23)
            img=ImageEnhance.Sharpness(img).enhance(0.75)
        else:
            img=ImageEnhance.Contrast(img).enhance(1.06)
        # low-level noise
        noise=Image.effect_noise(img.size, 4.0 if fax else 2.2)
        noise=ImageEnhance.Contrast(noise).enhance(.25)
        img=Image.blend(img,noise,0.035 if fax else 0.018)
        # subtle skew
        a=angle + (i*0.03)
        img=img.rotate(a,resample=Image.Resampling.BICUBIC,expand=False,fillcolor=255)
        d=ImageDraw.Draw(img)
        # left edge copier shadow and a couple faint feeder lines
        if fax:
            d.rectangle([0,0,5,img.height],fill=225)
            for _ in range(2):
                yy=random.randint(180,img.height-180)
                d.line([0,yy,img.width,yy],fill=240,width=1)
        # isolated dust specks
        for _ in range(45 if fax else 18):
            x=random.randrange(img.width); y=random.randrange(img.height)
            g=random.randint(170,225); r=random.choice([1,1,1,2]); d.ellipse([x-r,y-r,x+r,y+r],fill=g)
        imgs.append(img.convert('RGB'))
    doc.close()
    imgs[0].save(dst,save_all=True,append_images=imgs[1:],resolution=dpi,quality=88)

# ---- content ----
pathology_doc(OUT/'baseline/01_primary_breast_pathology.pdf','S25-00341','01/04/2025','Left breast, 10 o\'clock, ultrasound-guided core biopsy',
              "LEFT BREAST, 10 O'CLOCK, CORE BIOPSY: Invasive carcinoma of no special type (ductal), Nottingham grade 2. No definite lymphovascular invasion identified.",
              [('ER','Positive','95% tumor nuclei; strong intensity'),('PR','Positive','20% tumor nuclei; moderate intensity'),('HER2 IHC','Negative (1+)','Faint/incomplete membrane staining; no overexpression')],
              '59-year-old woman with left breast mass and suspicious liver lesions on staging imaging.',
              'Received in formalin labeled with patient identifiers and left breast 10:00 are six tan-white fibrofatty cores, 0.7-1.6 cm in length. Entirely submitted in A1-A2.',
              'Sections show infiltrative nests and cords of moderately differentiated malignant epithelial cells with duct formation in desmoplastic stroma. Nottingham score 6/9 (grade 2).',
              'Morphologic and immunophenotypic findings support primary invasive breast carcinoma. Correlate with imaging and clinical staging.')

pathology_doc(OUT/'baseline/02_metastatic_liver_pathology.pdf','S25-00619','01/15/2025','Liver, segment VIII mass, core biopsy',
              'LIVER MASS, CORE BIOPSY: Metastatic carcinoma consistent with breast primary.',
              [('ER','Positive','90% tumor nuclei; strong intensity'),('PR','Negative','<1% tumor nuclei'),('HER2 IHC','Equivocal (2+)','Weak to moderate complete membrane staining'),('HER2 ISH','Not amplified','HER2/CEP17 ratio 1.5; final HER2 interpretation: negative')],
              'Known left breast invasive carcinoma. Multiple hepatic lesions and sclerotic osseous lesions on staging CT. Biopsy requested for confirmation and receptor assessment.',
              'Received in formalin labeled liver mass are four tan-brown soft tissue cores, 0.8-1.4 cm in length. Entirely submitted in A1-A2.',
              'Core biopsies show metastatic adenocarcinoma involving hepatic parenchyma. Tumor cells are positive for GATA3 and pancytokeratin and negative for markers supporting a primary hepatic neoplasm.',
              'Findings support metastatic breast carcinoma. Receptor results on this metastatic specimen differ from the prior breast biopsy with respect to PR expression.',
              addendum='HER2 fluorescence in-situ hybridization: NOT AMPLIFIED. HER2/CEP17 ratio 1.5. Final HER2 status: negative.')

oncology_note(OUT/'baseline/03_oncology_treatment_start.pdf','ONC-250129-017','01/29/2025','Medical Oncology - New Patient / Treatment Planning Note',
              'Maya Rowan is seen after biopsy of a left breast primary and liver lesion confirmed metastatic breast carcinoma. Staging CT demonstrates liver and bone involvement. She reports mild fatigue but remains active and independent. No focal neurologic symptoms. ECOG performance status 1.',
              ['Letrozole 2.5 mg by mouth daily - planned start 02/03/2025','Ribociclib 600 mg by mouth daily, days 1-21 of a 28-day cycle - planned start 02/03/2025'],
              'De novo ER-positive/HER2-negative metastatic breast cancer involving liver and bone. Intent of systemic therapy is disease control. Baseline ECG and laboratory monitoring reviewed.',
              ['Begin letrozole plus ribociclib as above after baseline laboratory review.','CBC/CMP every 2 weeks for the first two cycles, then per protocol.','Restaging CT chest/abdomen/pelvis in approximately 12 weeks.','Call for fever, infection symptoms, severe diarrhea, rash, palpitations, or other concerning symptoms.'])

# radiology content
rad_specs=[
('04_baseline_ct.pdf','RAD25-012481','01/27/2025','None available.','Newly diagnosed breast carcinoma. Initial staging.',
 [('Lungs/Pleura','No suspicious pulmonary nodule. No pleural effusion.'),('Liver','Multiple hypoenhancing lesions. Dominant segment VIII lesion measures 3.1 cm. Segment IVa lesion measures 1.6 cm.'),('Lymph nodes','No pathologically enlarged thoracic, abdominal, or pelvic lymph nodes.'),('Bones','Sclerotic lesions at T8 and L2, suspicious for osseous metastases.')],
 ['Multiple hepatic lesions and sclerotic osseous lesions, suspicious for metastatic disease in the setting of known primary breast cancer.','No pulmonary metastatic disease.','The indeterminate baseline pattern will be followed on subsequent imaging.']),
('05_followup_ct_1.pdf','RAD25-028554','04/28/2025','CT 01/27/2025.','Metastatic breast cancer on systemic therapy. Restaging.',
 [('Lungs/Pleura','No suspicious pulmonary nodule or pleural effusion.'),('Liver','Dominant segment VIII lesion measures 2.3 cm, previously 3.1 cm. Segment IVa lesion measures 1.3 cm, previously 1.6 cm. No new hepatic lesion.'),('Lymph nodes','No new thoracic, abdominal, or pelvic adenopathy.'),('Bones','Sclerotic lesions at T8 and L2 are unchanged.')],
 ['Interval decrease in size of hepatic metastases.','Stable osseous metastatic disease.','Overall appearance remains consistent with stable disease on current therapy.']),
('08_followup_ct_2.pdf','RAD25-046912','07/30/2025','CT 04/28/2025.','Metastatic breast cancer on systemic therapy; restaging.',
 [('Lungs/Pleura','No suspicious pulmonary nodule, pleural effusion, or thoracic adenopathy.'),('Liver','Segment VIII metastasis measures 2.2 cm, previously 2.3 cm. Segment IVa lesion 1.2 cm, previously 1.3 cm. No new hepatic lesion.'),('Abdomen/Pelvis','No new visceral or nodal metastatic disease.'),('Bones','Stable sclerotic metastases at T8 and L2. No new osseous lesion or pathologic fracture.')],
 ['Stable hepatic metastatic disease without new liver lesion.','Stable osseous metastatic disease.','No new metastatic disease in the chest, abdomen, or pelvis.']),
('09_followup_ct_3.pdf','RAD25-064105','10/29/2025','CT 07/30/2025.','Metastatic breast cancer; restaging on letrozole/ribociclib.',
 [('Lungs/Pleura','No suspicious pulmonary nodule or thoracic adenopathy.'),('Liver','Segment VIII lesion measures 2.1 cm, previously 2.2 cm. Segment IVa lesion 1.1 cm, previously 1.2 cm. No new lesion.'),('Abdomen/Pelvis','No new visceral or nodal metastatic disease.'),('Bones','Unchanged sclerotic metastases at T8 and L2. No pathologic fracture.')],
 ['Stable hepatic and osseous metastatic disease.','No new metastatic disease.']),
('10_followup_ct_4.pdf','RAD26-007944','01/28/2026','CT 10/29/2025.','Metastatic breast cancer; interval restaging.',
 [('Lungs/Pleura','No suspicious pulmonary nodule, pleural effusion, or thoracic adenopathy.'),('Liver','Segment VIII lesion measures 2.1 cm, unchanged. Segment IVa lesion 1.1 cm, unchanged. No new hepatic lesion.'),('Abdomen/Pelvis','No new visceral or nodal metastatic disease.'),('Bones','Stable sclerotic metastases at T8 and L2. No new osseous lesion or pathologic fracture.')],
 ['Stable hepatic metastatic disease.','Stable osseous metastatic disease.','No new metastatic disease.']),
]
for fn,acc,date,comp,ind,findings,impression in rad_specs:
    radiology_doc(OUT/'baseline'/fn,acc,date,comp,ind,findings,impression)

lab_doc(OUT/'baseline/06_selected_labs_scanned.pdf')

oncology_note(OUT/'baseline/07_oncology_followup_dose_adjustment.pdf','ONC-250722-017','07/22/2025','Medical Oncology Follow-up Note',
              'Returns during cycle 6 of letrozole/ribociclib. She feels generally well with mild fatigue and no fever or infectious symptoms. CBC today shows neutropenia. Most recent CT continues to show controlled disease.',
              ['Letrozole 2.5 mg by mouth daily','Ribociclib 600 mg by mouth daily, days 1-21 of each 28-day cycle - HOLD starting today'],
              'Metastatic ER-positive/HER2-negative breast cancer with continued radiographic disease control. Grade 3 neutropenia attributed to ribociclib.',
              ['Hold ribociclib for 7 days and repeat CBC.','Continue letrozole without interruption.','If ANC recovers, resume ribociclib at reduced dose of 400 mg daily, days 1-21 of each 28-day cycle.','Proceed with scheduled restaging CT.'],
              addendum='07/29/2025: Repeat CBC reviewed with ANC recovery. Ribociclib resumed at 400 mg daily on days 1-21 of each 28-day cycle. Letrozole 2.5 mg daily continues without change.')

radiology_doc(OUT/'new_evidence/11_newest_ct_progression.pdf','RAD26-026331','04/29/2026','CT 01/28/2026.','Metastatic breast cancer; restaging after approximately 15 months of endocrine/CDK4/6 therapy.',
 [('Lungs/Pleura','No suspicious pulmonary nodule, pleural effusion, or thoracic adenopathy.'),('Liver','Dominant segment VIII metastasis now measures 3.4 cm, previously 2.1 cm. Segment IVa lesion measures 1.5 cm, previously 1.1 cm. New 1.2 cm hypoenhancing lesion in segment II is suspicious for new metastasis.'),('Abdomen/Pelvis','No new nodal disease or ascites.'),('Bones','Known sclerotic metastases at T8 and L2 are unchanged. No new destructive lesion or pathologic fracture.')],
 ['Interval progression of hepatic metastatic disease, including substantial enlargement of the dominant segment VIII lesion and a new segment II metastasis.','Osseous metastatic disease is unchanged.','No pulmonary metastatic disease.'])

genomic_doc(OUT/'extension/12_post_progression_ctdna_esr1.pdf')

# Update fixture hashes without changing clinical fixture payloads
manifest_path=OUT/'demo_fixtures/extraction_manifest.json'
old_manifest=json.loads(manifest_path.read_text())
newdocs={}
for oldsha,entry in old_manifest['documents'].items():
    fpath=OUT/entry['source_file']
    sha=hashlib.sha256(fpath.read_bytes()).hexdigest()
    fx=OUT/'demo_fixtures'/entry['fixture_file']
    payload=json.loads(fx.read_text())
    payload['document_sha256']=sha
    payload['source_file']=entry['source_file']
    fx.write_text(json.dumps(payload,indent=2))
    newdocs[sha]=entry
old_manifest['documents']=newdocs
manifest_path.write_text(json.dumps(old_manifest,indent=2))

case_path=OUT/'case_manifest.json'
case=json.loads(case_path.read_text())
case['version']='v4_clinical_export_records'
case['institution']='Meridian Vale Health System / Apex Molecular Diagnostics (fictional)'
case_path.write_text(json.dumps(case,indent=2))

(OUT/'README.md').write_text('''# Maya Rowan synthetic oncology record packet - v4\n\nThis packet is fully synthetic. No real patient, clinician, institution, accession number, signature, or medical record is represented.\n\nVersion 4 deliberately looks like a mixed real-world chart export rather than a designed brochure. Different departments use different layouts. Pathology and laboratory documents are scan-like raster copies with mild skew/noise; oncology notes resemble EHR print views; radiology reports use a plain signed-result layout; molecular testing resembles an outside reference-lab report. The synthetic label is intentionally small but visible on every page.\n\nThe clinical facts and dates are unchanged from the canonical OncoTwin demo patient so deterministic extraction fixtures and the real frozen forecast remain stable.\n''')

(OUT/'REALISM_RESEARCH_NOTES.md').write_text('''# Record realism notes - v4\n\nDesign references used for structure, not copied branding or text:\n- Johns Hopkins breast pathology educational material: patient identifiers, accession/date fields, clinical history, specimen/body site, diagnosis, gross description, pathologist identity, and ER/PR/HER2 special studies.\n- Public sample CBC reports: dense patient/specimen identity block, result/flag/unit/reference-range columns, verification metadata, and simple monochrome tabulation.\n- Public hospital/radiology samples: clinical history/indication, technique, comparison, findings, impression, and report-status/version metadata.\n\nAuthenticity choices:\n- no common polished design system across departments;\n- mostly monochrome typography and thin rules;\n- EHR-like field labels and timestamps;\n- accessions/barcodes and signed/final status;\n- scan artifacts on pathology/lab pages;\n- different external-lab appearance for molecular testing;\n- small synthetic label retained on every page.\n''')

print('generated', OUT)
for sha,entry in newdocs.items(): print(sha[:12],entry['source_file'])
