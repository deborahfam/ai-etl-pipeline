"""
Generate 5 realistic-looking invoice images using Pillow.
Each invoice has a unique company, style, currency, and layout.
"""

import os
from PIL import Image, ImageDraw, ImageFont
from datetime import date, timedelta
import textwrap

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "invoices")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Helper: try to load a TrueType font, fall back to default
# ---------------------------------------------------------------------------

def _font(size, bold=False):
    """Return a TrueType font at the requested size. Works on macOS, Linux, Windows."""
    candidates_regular = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/SFNSText.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]
    candidates_bold = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/SFNSText-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
    ]
    paths = candidates_bold if bold else candidates_regular
    for p in paths:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size, index=1 if bold and p.endswith(".ttc") else 0)
            except Exception:
                try:
                    return ImageFont.truetype(p, size)
                except Exception:
                    continue
    return ImageFont.load_default()


def _serif_font(size, bold=False):
    """Return a serif font for variety."""
    candidates = [
        ("/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf" if bold
         else "/System/Library/Fonts/Supplemental/Times New Roman.ttf"),
        "/System/Library/Fonts/NewYork.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf" if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    ]
    for p in candidates:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    return _font(size, bold)


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def draw_text(draw, x, y, text, font, fill="black", anchor=None):
    draw.text((x, y), text, font=font, fill=fill, anchor=anchor)


def draw_line(draw, x1, y1, x2, y2, fill="black", width=1):
    draw.line([(x1, y1), (x2, y2)], fill=fill, width=width)


def draw_rect(draw, x1, y1, x2, y2, outline="black", fill=None, width=1):
    draw.rectangle([(x1, y1), (x2, y2)], outline=outline, fill=fill, width=width)


def text_width(draw, text, font):
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0]


def text_height(draw, text, font):
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[3] - bbox[1]


def right_text(draw, x_right, y, text, font, fill="black"):
    w = text_width(draw, text, font)
    draw.text((x_right - w, y), text, font=font, fill=fill)


# ---------------------------------------------------------------------------
# Logo placeholder
# ---------------------------------------------------------------------------

def draw_logo_placeholder(draw, x, y, w, h, color="#3366AA"):
    draw_rect(draw, x, y, x + w, y + h, outline=color, fill=color, width=0)
    f = _font(int(h * 0.35), bold=True)
    tw = text_width(draw, "LOGO", f)
    th = text_height(draw, "LOGO", f)
    draw_text(draw, x + (w - tw) // 2, y + (h - th) // 2 - 2, "LOGO", f, fill="white")


# ---------------------------------------------------------------------------
# Invoice 1: TechFlow Solutions
# ---------------------------------------------------------------------------

def invoice_techflow():
    W, H = 850, 1100
    bg = "#FFFFFF"
    accent = "#1A5276"
    img = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(img)
    m = 50  # margin

    # Header band
    draw_rect(d, 0, 0, W, 90, fill=accent, outline=accent)
    draw_logo_placeholder(d, m, 15, 60, 60, accent)
    f_title = _font(28, bold=True)
    draw_text(d, 130, 20, "TechFlow Solutions", f_title, fill="white")
    f_sub = _font(12)
    draw_text(d, 130, 55, "742 Innovation Drive, Austin, TX 78701  |  info@techflow.io  |  +1 512-555-0198", f_sub, fill="#D5E8F0")

    # Invoice label
    f_inv = _font(22, bold=True)
    draw_text(d, m, 110, "INVOICE", f_inv, fill=accent)

    f_label = _font(11, bold=True)
    f_val = _font(11)

    # Invoice meta (right side)
    right_text(d, W - m, 110, "Invoice #: INV-2026-0041", f_val, fill="#333333")
    right_text(d, W - m, 128, "Date: March 15, 2026", f_val, fill="#333333")
    right_text(d, W - m, 146, "Due Date: April 14, 2026", f_val, fill="#333333")

    # Bill To
    y = 180
    draw_text(d, m, y, "Bill To:", f_label, fill="#555555")
    y += 18
    for line in ["Acme Corp", "1200 Market Street, Suite 400", "San Francisco, CA 94103", "billing@acmecorp.com"]:
        draw_text(d, m, y, line, f_val, fill="#333333")
        y += 16

    # Table
    cols = [m, m + 340, m + 430, m + 560, W - m]  # desc, qty, unit, total
    headers = ["Description", "Qty", "Unit Price", "Total"]
    ty = 300
    draw_rect(d, m, ty, W - m, ty + 28, fill=accent, outline=accent)
    f_th = _font(11, bold=True)
    for i, h in enumerate(headers):
        if i == 0:
            draw_text(d, cols[i] + 8, ty + 7, h, f_th, fill="white")
        else:
            right_text(d, cols[i + 1] - 8, ty + 7, h, f_th, fill="white")

    items = [
        ("Cloud Infrastructure Setup & Configuration", 1, 4500.00),
        ("Annual Software License (Enterprise)", 10, 1200.00),
        ("Data Migration Services", 1, 3200.00),
        ("24/7 Premium Support (12 months)", 1, 6000.00),
    ]

    ty += 28
    f_row = _font(11)
    stripe = "#F2F7FB"
    for idx, (desc, qty, price) in enumerate(items):
        row_bg = stripe if idx % 2 == 0 else bg
        draw_rect(d, m, ty, W - m, ty + 28, fill=row_bg, outline=row_bg)
        draw_text(d, cols[0] + 8, ty + 7, desc, f_row, fill="#333333")
        right_text(d, cols[2] - 8, ty + 7, str(qty), f_row, fill="#333333")
        right_text(d, cols[3] - 8, ty + 7, f"${price:,.2f}", f_row, fill="#333333")
        total = qty * price
        right_text(d, cols[4] - 8, ty + 7, f"${total:,.2f}", f_row, fill="#333333")
        ty += 28

    draw_line(d, m, ty, W - m, ty, fill="#CCCCCC", width=1)

    # Totals
    ty += 15
    subtotal = sum(q * p for _, q, p in items)
    tax = subtotal * 0.0825
    total = subtotal + tax
    for label, val in [("Subtotal", f"${subtotal:,.2f}"), ("Tax (8.25%)", f"${tax:,.2f}")]:
        right_text(d, cols[3] - 8, ty, label, f_label, fill="#555555")
        right_text(d, cols[4] - 8, ty, val, f_row, fill="#333333")
        ty += 22

    draw_line(d, cols[3] - 100, ty - 4, W - m, ty - 4, fill=accent, width=2)
    ty += 4
    f_total = _font(14, bold=True)
    right_text(d, cols[3] - 8, ty, "Total Due", f_total, fill=accent)
    right_text(d, cols[4] - 8, ty, f"${total:,.2f}", f_total, fill=accent)

    # Footer
    f_foot = _font(9)
    draw_line(d, m, H - 60, W - m, H - 60, fill="#CCCCCC")
    draw_text(d, m, H - 45, "Payment Terms: Net 30  |  Bank: First National, Acct: 29301847  |  Thank you for your business!", f_foot, fill="#888888")

    img.save(os.path.join(OUTPUT_DIR, "invoice_techflow.png"))
    print("  Created invoice_techflow.png")


# ---------------------------------------------------------------------------
# Invoice 2: Garcia & Associates  (mixed EN/ES, EUR)
# ---------------------------------------------------------------------------

def invoice_garcia():
    W, H = 850, 1100
    bg = "#FFFDF5"
    accent = "#8B4513"
    img = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(img)
    m = 50

    # Left color bar
    draw_rect(d, 0, 0, 12, H, fill=accent, outline=accent)

    # Header
    draw_logo_placeholder(d, m, 40, 55, 55, "#C0792A")
    f_title = _font(26, bold=True)
    draw_text(d, 120, 40, "Garcia & Associates", f_title, fill=accent)
    f_sub = _font(11)
    draw_text(d, 120, 72, "Consultoría Legal y Financiera", f_sub, fill="#7A5C3A")
    draw_text(d, 120, 88, "Calle Gran Vía 28, 28013 Madrid, España  |  +34 91 555 4321", f_sub, fill="#999999")

    # FACTURA / INVOICE
    f_inv = _font(20, bold=True)
    draw_text(d, m, 140, "FACTURA / INVOICE", f_inv, fill=accent)
    draw_line(d, m, 168, 280, 168, fill=accent, width=2)

    f_label = _font(11, bold=True)
    f_val = _font(11)

    # Meta right
    rx = W - m
    right_text(d, rx, 140, "Factura No: GA-2026-0117", f_val, fill="#333")
    right_text(d, rx, 158, "Fecha / Date: 22 Feb 2026", f_val, fill="#333")
    right_text(d, rx, 176, "Vencimiento / Due: 22 Mar 2026", f_val, fill="#333")

    # Bill to
    y = 210
    draw_text(d, m, y, "Facturar a / Bill To:", f_label, fill="#555")
    y += 18
    for line in ["Empresa Soluciones Globales S.L.", "Av. de la Constitución 12", "41001 Sevilla, España", "contacto@solucionesglobales.es"]:
        draw_text(d, m, y, line, f_val, fill="#333")
        y += 16

    # Table
    cols = [m, m + 360, m + 440, m + 560, W - m]
    headers = ["Descripción / Description", "Cant.", "Precio Unit.", "Total"]
    ty = 320
    draw_rect(d, m, ty, W - m, ty + 28, fill=accent, outline=accent)
    f_th = _font(11, bold=True)
    for i, h in enumerate(headers):
        if i == 0:
            draw_text(d, cols[i] + 8, ty + 7, h, f_th, fill="white")
        else:
            right_text(d, cols[i + 1] - 8, ty + 7, h, f_th, fill="white")

    items = [
        ("Auditoría financiera anual / Annual financial audit", 1, 8500.00),
        ("Asesoría fiscal trimestral / Quarterly tax advisory", 4, 1750.00),
        ("Traducción documentos legales / Legal doc translation", 12, 320.00),
    ]

    ty += 28
    f_row = _font(11)
    for idx, (desc, qty, price) in enumerate(items):
        row_bg = "#FFF8EC" if idx % 2 == 0 else bg
        draw_rect(d, m, ty, W - m, ty + 28, fill=row_bg, outline=row_bg)
        draw_text(d, cols[0] + 8, ty + 7, desc, f_row, fill="#333")
        right_text(d, cols[2] - 8, ty + 7, str(qty), f_row, fill="#333")
        right_text(d, cols[3] - 8, ty + 7, f"{price:,.2f} EUR", f_row, fill="#333")
        total = qty * price
        right_text(d, cols[4] - 8, ty + 7, f"{total:,.2f} EUR", f_row, fill="#333")
        ty += 28

    draw_line(d, m, ty, W - m, ty, fill="#CCCCCC")

    # Totals
    ty += 15
    subtotal = sum(q * p for _, q, p in items)
    tax = subtotal * 0.21  # Spanish IVA
    total = subtotal + tax
    for label, val in [("Subtotal", f"{subtotal:,.2f} EUR"), ("IVA (21%)", f"{tax:,.2f} EUR")]:
        right_text(d, cols[3] - 8, ty, label, f_label, fill="#555")
        right_text(d, cols[4] - 8, ty, val, f_row, fill="#333")
        ty += 22

    draw_line(d, cols[3] - 100, ty - 4, W - m, ty - 4, fill=accent, width=2)
    ty += 4
    f_total = _font(14, bold=True)
    right_text(d, cols[3] - 8, ty, "Total a Pagar", f_total, fill=accent)
    right_text(d, cols[4] - 8, ty, f"{total:,.2f} EUR", f_total, fill=accent)

    # Footer
    f_foot = _font(9)
    draw_line(d, m, H - 70, W - m, H - 70, fill="#CCCCCC")
    draw_text(d, m, H - 55, "Condiciones de pago: 30 días  |  IBAN: ES91 2100 0418 4502 0005 1332  |  Gracias por su confianza.", f_foot, fill="#888")

    img.save(os.path.join(OUTPUT_DIR, "invoice_garcia.png"))
    print("  Created invoice_garcia.png")


# ---------------------------------------------------------------------------
# Invoice 3: Nordic Supplies AB  (SEK, serif font style)
# ---------------------------------------------------------------------------

def invoice_nordic():
    W, H = 850, 1150
    bg = "#F5F7FA"
    accent = "#2C3E6B"
    img = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(img)
    m = 50

    # Top accent band
    draw_rect(d, 0, 0, W, 8, fill=accent, outline=accent)

    # Header
    draw_logo_placeholder(d, m, 25, 55, 55, accent)
    f_title = _serif_font(26, bold=True)
    draw_text(d, 120, 25, "Nordic Supplies AB", f_title, fill=accent)
    f_sub = _serif_font(11)
    draw_text(d, 120, 56, "Storgatan 14, 111 51 Stockholm, Sweden", f_sub, fill="#666")
    draw_text(d, 120, 72, "Org.nr: 556123-4567  |  info@nordicsupplies.se  |  +46 8 555 1234", f_sub, fill="#666")

    f_inv = _serif_font(20, bold=True)
    draw_text(d, m, 110, "FAKTURA / INVOICE", f_inv, fill=accent)
    draw_line(d, m, 136, 270, 136, fill=accent, width=2)

    f_label = _serif_font(11, bold=True)
    f_val = _serif_font(11)

    rx = W - m
    right_text(d, rx, 110, "Fakturanr: NS-2026-0583", f_val, fill="#333")
    right_text(d, rx, 128, "Fakturadatum: 2026-01-10", f_val, fill="#333")
    right_text(d, rx, 146, "Forfallodatum: 2026-02-09", f_val, fill="#333")
    right_text(d, rx, 164, "Betalningsvillkor: 30 dagar netto", f_val, fill="#333")

    # Bill to
    y = 195
    draw_text(d, m, y, "Kund / Customer:", f_label, fill="#555")
    y += 18
    for line in ["Bergström & Partners HB", "Vasagatan 7, 3tr", "411 24 Göteborg, Sweden", "faktura@bergstrom.se"]:
        draw_text(d, m, y, line, f_val, fill="#333")
        y += 16

    # Table
    cols = [m, m + 310, m + 380, m + 480, m + 590, W - m]
    headers = ["Beskrivning / Description", "Antal", "Enhet", "À-pris", "Belopp"]
    ty = 310
    draw_rect(d, m, ty, W - m, ty + 28, fill=accent, outline=accent)
    f_th = _serif_font(11, bold=True)
    for i, h in enumerate(headers):
        if i == 0:
            draw_text(d, cols[i] + 8, ty + 7, h, f_th, fill="white")
        else:
            right_text(d, cols[i + 1] - 8, ty + 7, h, f_th, fill="white")

    items = [
        ("Kontorsstol Ergonomisk Pro", 10, "st", 4995.00),
        ("Skrivbord Höj/Sänkbart 160cm", 10, "st", 7490.00),
        ("LED Skrivbordslampa", 10, "st", 1295.00),
        ("Kabeldragning & installation", 1, "tjänst", 12500.00),
        ("Fraktkostnad / Shipping", 1, "tjänst", 3500.00),
    ]

    ty += 28
    f_row = _serif_font(11)
    stripe = "#E8ECF2"
    for idx, (desc, qty, unit, price) in enumerate(items):
        row_bg = stripe if idx % 2 == 0 else bg
        draw_rect(d, m, ty, W - m, ty + 28, fill=row_bg, outline=row_bg)
        draw_text(d, cols[0] + 8, ty + 7, desc, f_row, fill="#333")
        right_text(d, cols[2] - 8, ty + 7, str(qty), f_row, fill="#333")
        right_text(d, cols[3] - 8, ty + 7, unit, f_row, fill="#333")
        right_text(d, cols[4] - 8, ty + 7, f"{price:,.0f} kr", f_row, fill="#333")
        total = qty * price
        right_text(d, cols[5] - 8, ty + 7, f"{total:,.0f} kr", f_row, fill="#333")
        ty += 28

    draw_line(d, m, ty, W - m, ty, fill="#AAAAAA")

    # Totals
    ty += 15
    subtotal = sum(q * p for _, q, _, p in items)
    moms = subtotal * 0.25  # Swedish VAT
    total = subtotal + moms
    for label, val in [("Netto", f"{subtotal:,.0f} kr"), ("Moms (25%)", f"{moms:,.0f} kr")]:
        right_text(d, cols[4] - 8, ty, label, f_label, fill="#555")
        right_text(d, cols[5] - 8, ty, val, f_row, fill="#333")
        ty += 22

    draw_line(d, cols[4] - 100, ty - 4, W - m, ty - 4, fill=accent, width=2)
    ty += 4
    f_total = _serif_font(14, bold=True)
    right_text(d, cols[4] - 8, ty, "Att betala", f_total, fill=accent)
    right_text(d, cols[5] - 8, ty, f"{total:,.0f} kr", f_total, fill=accent)

    # Payment details box
    ty += 50
    draw_rect(d, m, ty, W - m, ty + 70, outline="#CCCCCC", fill="#FFFFFF", width=1)
    draw_text(d, m + 15, ty + 10, "Betalningsinformation:", f_label, fill=accent)
    draw_text(d, m + 15, ty + 30, "Bankgiro: 5812-4693   |   Swish: 123 456 78 90   |   Ref: NS-2026-0583", f_val, fill="#333")
    draw_text(d, m + 15, ty + 48, "Dröjsmålsränta: 8% vid försenad betalning.", f_val, fill="#888")

    # Footer accent
    draw_rect(d, 0, H - 8, W, H, fill=accent, outline=accent)

    img.save(os.path.join(OUTPUT_DIR, "invoice_nordic.png"))
    print("  Created invoice_nordic.png")


# ---------------------------------------------------------------------------
# Invoice 4: CloudPeak Services  (GBP, minimal)
# ---------------------------------------------------------------------------

def invoice_cloudpeak():
    W, H = 850, 950
    bg = "#FFFFFF"
    accent = "#0D7C66"
    img = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(img)
    m = 60

    # Minimal header - just text, no band
    f_title = _font(30, bold=True)
    draw_text(d, m, 50, "CloudPeak", f_title, fill=accent)
    f_light = _font(14)
    draw_text(d, m, 86, "Services", f_light, fill="#AAAAAA")

    # Right-aligned company info
    f_sub = _font(10)
    rx = W - m
    for i, line in enumerate(["CloudPeak Services Ltd", "45 Canary Wharf, London E14 5AB", "VAT: GB 123 4567 89", "hello@cloudpeak.co.uk"]):
        right_text(d, rx, 50 + i * 16, line, f_sub, fill="#888")

    # Divider
    draw_line(d, m, 125, W - m, 125, fill="#EEEEEE", width=1)

    # INVOICE title
    f_inv = _font(16, bold=True)
    draw_text(d, m, 145, "INVOICE", f_inv, fill="#333")

    f_label = _font(10, bold=True)
    f_val = _font(10)

    # Two-column meta
    draw_text(d, m, 175, "Invoice No:", f_label, fill="#888")
    draw_text(d, m + 80, 175, "CP-00294", f_val, fill="#333")
    draw_text(d, m, 192, "Date:", f_label, fill="#888")
    draw_text(d, m + 80, 192, "5 March 2026", f_val, fill="#333")
    draw_text(d, m, 209, "Due Date:", f_label, fill="#888")
    draw_text(d, m + 80, 209, "4 April 2026", f_val, fill="#333")

    draw_text(d, 400, 175, "Bill To:", f_label, fill="#888")
    for i, line in enumerate(["Whitfield & Co", "88 Regent Street", "London W1B 5RS"]):
        draw_text(d, 400, 192 + i * 16, line, f_val, fill="#333")

    # Table - minimal style with just lines
    cols = [m, m + 400, m + 490, W - m]
    headers = ["Description", "Qty", "Amount"]
    ty = 280
    draw_line(d, m, ty, W - m, ty, fill=accent, width=2)
    ty += 10
    f_th = _font(10, bold=True)
    draw_text(d, cols[0], ty, headers[0], f_th, fill="#555")
    right_text(d, cols[2] - 8, ty, headers[1], f_th, fill="#555")
    right_text(d, cols[3], ty, headers[2], f_th, fill="#555")
    ty += 22
    draw_line(d, m, ty, W - m, ty, fill="#DDDDDD")

    items = [
        ("Cloud Architecture Consultation (40 hrs)", 1, 6800.00),
        ("Monthly Managed Hosting - Q1 2026", 3, 1450.00),
    ]

    f_row = _font(11)
    ty += 12
    for desc, qty, price in items:
        total = qty * price
        draw_text(d, cols[0], ty, desc, f_row, fill="#333")
        right_text(d, cols[2] - 8, ty, str(qty), f_row, fill="#333")
        right_text(d, cols[3], ty, f"£{total:,.2f}", f_row, fill="#333")
        ty += 30

    draw_line(d, m, ty, W - m, ty, fill="#DDDDDD")

    # Totals
    ty += 20
    subtotal = sum(q * p for _, q, p in items)
    vat = subtotal * 0.20
    total = subtotal + vat
    for label, val in [("Subtotal", f"£{subtotal:,.2f}"), ("VAT (20%)", f"£{vat:,.2f}")]:
        right_text(d, cols[2] + 30, ty, label, f_label, fill="#888")
        right_text(d, cols[3], ty, val, f_row, fill="#333")
        ty += 22

    draw_line(d, cols[2] - 50, ty, W - m, ty, fill=accent, width=2)
    ty += 10
    f_total = _font(13, bold=True)
    right_text(d, cols[2] + 30, ty, "Total", f_total, fill="#333")
    right_text(d, cols[3], ty, f"£{total:,.2f}", f_total, fill=accent)

    # Footer
    f_foot = _font(9)
    draw_line(d, m, H - 60, W - m, H - 60, fill="#EEEEEE")
    draw_text(d, m, H - 42, "Payment: BACS  |  Sort: 20-45-12  |  Account: 43218765  |  Ref: CP-00294", f_foot, fill="#AAAAAA")

    img.save(os.path.join(OUTPUT_DIR, "invoice_cloudpeak.png"))
    print("  Created invoice_cloudpeak.png")


# ---------------------------------------------------------------------------
# Invoice 5: Tanaka Industries  (JPY, dense layout, 6 items)
# ---------------------------------------------------------------------------

def invoice_tanaka():
    W, H = 850, 1200
    bg = "#FAFAFA"
    accent = "#C41E3A"
    dark = "#1A1A2E"
    img = Image.new("RGB", (W, H), bg)
    d = ImageDraw.Draw(img)
    m = 45

    # Header band with two-tone
    draw_rect(d, 0, 0, W, 6, fill=accent, outline=accent)
    draw_rect(d, 0, 6, W, 95, fill=dark, outline=dark)
    draw_logo_placeholder(d, m, 18, 55, 55, accent)
    f_title = _font(24, bold=True)
    draw_text(d, 115, 18, "Tanaka Industries", f_title, fill="white")
    f_jp = _font(13)
    draw_text(d, 115, 48, "田中工業株式会社", f_jp, fill="#CCCCCC")
    f_sub = _font(10)
    draw_text(d, 115, 70, "3-14-1 Shinjuku, Shinjuku-ku, Tokyo 160-0022  |  +81 3-5555-7890", f_sub, fill="#999999")

    # Invoice heading
    f_inv = _font(18, bold=True)
    draw_text(d, m, 110, "請求書 / INVOICE", f_inv, fill=dark)
    draw_line(d, m, 134, 230, 134, fill=accent, width=2)

    f_label = _font(10, bold=True)
    f_val = _font(10)

    # Meta
    rx = W - m
    right_text(d, rx, 110, "Invoice #: TI-2026-1042", f_val, fill="#333")
    right_text(d, rx, 126, "Issue Date: 2026-03-01", f_val, fill="#333")
    right_text(d, rx, 142, "Due Date: 2026-03-31", f_val, fill="#333")
    right_text(d, rx, 158, "Payment: Bank Transfer", f_val, fill="#333")

    # Bill To / Ship To side by side
    y = 185
    draw_text(d, m, y, "Bill To / 請求先:", f_label, fill="#555")
    draw_text(d, 420, y, "Ship To / 納品先:", f_label, fill="#555")
    y += 18
    bill_lines = ["Yamamoto Electronics Co., Ltd.", "山本電子株式会社", "2-8-5 Akihabara, Chiyoda-ku", "Tokyo 101-0021"]
    ship_lines = ["Yamamoto Warehouse", "6-2-1 Shinagawa, Minato-ku", "Tokyo 108-0075", ""]
    for bl, sl in zip(bill_lines, ship_lines):
        draw_text(d, m, y, bl, f_val, fill="#333")
        draw_text(d, 420, y, sl, f_val, fill="#333")
        y += 15

    # Table - dense
    cols = [m, m + 40, m + 360, m + 420, m + 520, m + 620, W - m]
    headers = ["#", "Description / 品目", "Qty", "Unit Price", "Disc.", "Amount"]
    ty = 305
    draw_rect(d, m, ty, W - m, ty + 26, fill=dark, outline=dark)
    f_th = _font(10, bold=True)
    draw_text(d, cols[0] + 6, ty + 6, headers[0], f_th, fill="white")
    draw_text(d, cols[1] + 6, ty + 6, headers[1], f_th, fill="white")
    for i in range(2, 6):
        right_text(d, cols[i + 1] - 6, ty + 6, headers[i], f_th, fill="white")

    items = [
        ("Industrial Sensor Module (ISM-400)", 50, 12800, 0),
        ("Precision Servo Motor (PSM-220)", 20, 45600, 5),
        ("Control Board CB-X1 (assembled)", 30, 8900, 0),
        ("Wiring Harness Kit (WHK-100)", 50, 3200, 10),
        ("Custom Enclosure (aluminum, CNC)", 10, 67000, 0),
        ("Installation & Calibration Service", 1, 350000, 0),
    ]

    ty += 26
    f_row = _font(10)
    stripe = "#F0F0F0"
    for idx, (desc, qty, price, disc) in enumerate(items):
        row_bg = stripe if idx % 2 == 0 else bg
        draw_rect(d, m, ty, W - m, ty + 26, fill=row_bg, outline=row_bg)
        draw_text(d, cols[0] + 10, ty + 6, str(idx + 1), f_row, fill="#666")
        draw_text(d, cols[1] + 6, ty + 6, desc, f_row, fill="#333")
        right_text(d, cols[3] - 6, ty + 6, str(qty), f_row, fill="#333")
        right_text(d, cols[4] - 6, ty + 6, f"¥{price:,}", f_row, fill="#333")
        disc_str = f"{disc}%" if disc else "-"
        right_text(d, cols[5] - 6, ty + 6, disc_str, f_row, fill="#333")
        amount = qty * price * (1 - disc / 100)
        right_text(d, cols[6] - 6, ty + 6, f"¥{amount:,.0f}", f_row, fill="#333")
        ty += 26

    draw_line(d, m, ty, W - m, ty, fill="#AAAAAA")

    # Totals
    ty += 12
    subtotal = sum(q * p * (1 - dc / 100) for _, q, p, dc in items)
    tax = subtotal * 0.10  # Japanese consumption tax
    total = subtotal + tax
    for label, val in [("小計 / Subtotal", f"¥{subtotal:,.0f}"), ("消費税 (10%)", f"¥{tax:,.0f}")]:
        right_text(d, cols[5] - 6, ty, label, f_label, fill="#555")
        right_text(d, cols[6] - 6, ty, val, f_row, fill="#333")
        ty += 20

    draw_line(d, cols[4] - 30, ty - 2, W - m, ty - 2, fill=accent, width=2)
    ty += 6
    f_total = _font(14, bold=True)
    right_text(d, cols[5] - 6, ty, "合計 / Total", f_total, fill=dark)
    right_text(d, cols[6] - 6, ty, f"¥{total:,.0f}", f_total, fill=accent)

    # Bank details box
    ty += 45
    draw_rect(d, m, ty, W - m, ty + 75, outline="#CCCCCC", fill="#FFFFFF", width=1)
    draw_text(d, m + 12, ty + 8, "振込先 / Bank Details:", f_label, fill=dark)
    for i, line in enumerate([
        "Bank: Mizuho Bank, Shinjuku Branch  |  Account: 1234567",
        "Swift: MHCBJPJT  |  Beneficiary: Tanaka Industries Co., Ltd.",
        "Please include invoice number as payment reference."
    ]):
        draw_text(d, m + 12, ty + 26 + i * 16, line, f_val, fill="#555")

    # Notes
    ty += 95
    draw_text(d, m, ty, "備考 / Notes:", f_label, fill="#555")
    draw_text(d, m, ty + 16, "Delivery lead time: 3-4 weeks from order confirmation. Warranty: 12 months.", f_val, fill="#888")

    # Footer
    draw_rect(d, 0, H - 6, W, H, fill=accent, outline=accent)

    img.save(os.path.join(OUTPUT_DIR, "invoice_tanaka.png"))
    print("  Created invoice_tanaka.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Generating invoices in {OUTPUT_DIR} ...")
    invoice_techflow()
    invoice_garcia()
    invoice_nordic()
    invoice_cloudpeak()
    invoice_tanaka()
    print("Done! 5 invoices generated.")
