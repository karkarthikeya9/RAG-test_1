"""
make_test_pdf.py — generate a small synthetic PDF for validating the
multimodal ingestion pipeline (text, headings, table, image, flowchart).

    python scripts/make_test_pdf.py

Output: data/test_multimodal.pdf
"""

import os

from PIL import Image, ImageDraw

import pymupdf

OUT_PATH = os.path.join("data", "test_multimodal.pdf")
IMG_DIR = os.path.join("scripts", "_tmp_imgs")


def make_image(name, draw_fn, size=(480, 320)):
    os.makedirs(IMG_DIR, exist_ok=True)
    img = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(img)
    draw_fn(draw, size)
    path = os.path.join(IMG_DIR, name)
    img.save(path)
    return path


def draw_client_server(draw, size):
    w, h = size
    draw.rectangle([40, 70, 160, 130], outline="black", width=2)
    draw.text((55, 90), "Client", fill="black")
    draw.rectangle([180, 70, 300, 130], outline="black", width=2)
    draw.text((192, 90), "App Server", fill="black")
    draw.rectangle([320, 70, 440, 130], outline="black", width=2)
    draw.text((332, 90), "Database", fill="black")
    draw.line([160, 100, 178, 100], fill="black", width=2)
    draw.polygon([(178, 94), (190, 100), (178, 106)], fill="black")
    draw.text((150, 78), "HTTP request", fill="black")
    draw.line([300, 100, 318, 100], fill="black", width=2)
    draw.polygon([(318, 94), (330, 100), (318, 106)], fill="black")
    draw.text((300, 78), "SQL query", fill="black")
    draw.line([160, 120, 178, 120], fill="black", width=2)
    draw.polygon([(178, 114), (190, 120), (178, 126)], fill="black")
    draw.text((150, 136), "response", fill="black")
    draw.text((40, 20), "Figure 1: Client-Server Architecture", fill="black")


def draw_login_flowchart(draw, size):
    w, h = size
    draw.ellipse([170, 10, 300, 70], outline="black", width=2)
    draw.text((205, 25), "Start", fill="black")
    draw.rectangle([140, 100, 330, 160], outline="black", width=2)
    draw.text((165, 115), "Enter password", fill="black")
    draw.polygon([(235, 160), (250, 180), (265, 160)], fill="black")
    draw.polygon([(235, 160), (250, 140), (265, 160)], outline="black")
    draw.rectangle([235, 90, 265, 180], outline="black", width=2)
    draw.text((243, 197), "Correct?", fill="black")
    draw.line([250, 185, 250, 205], fill="black", width=2)
    draw.rectangle([120, 240, 200, 300], outline="black", width=2)
    draw.text((128, 255), "Display error", fill="black")
    draw.line([130, 222, 130, 238], fill="black", width=2)
    draw.polygon([(130, 238), (118, 248), (130, 258)], fill="black")
    draw.line([130, 258, 130, 390], fill="black", width=2)
    draw.line([130, 390, 250, 390], fill="black", width=2)
    draw.line([250, 390, 250, 185], fill="black", width=2)
    draw.polygon([(240, 185), (250, 170), (260, 185)], fill="black")
    draw.rectangle([320, 240, 400, 300], outline="black", width=2)
    draw.text((335, 255), "Dashboard", fill="black")
    draw.line([265, 210, 360, 210], fill="black", width=2)
    draw.line([360, 210, 360, 238], fill="black", width=2)
    draw.polygon([(350, 238), (360, 250), (370, 238)], fill="black")
    draw.text((150, 1), "Figure 2: User Login Flowchart", fill="black")


def build_pdf():
    client_srv = make_image("client_server.png", draw_client_server)
    login = make_image("login_flowchart.png", draw_login_flowchart)

    doc = pymupdf.open()

    page1 = doc.new_page()
    page1.insert_textbox(
        pymupdf.Rect(50, 50, 545, 100),
        "# Client Server Architecture",
        fontsize=24,
    )
    page1.insert_textbox(
        pymupdf.Rect(50, 120, 545, 220),
        "The client communicates with the server over HTTP. The application "
        "server processes requests, accesses the database when required, and "
        "returns a response to the client.",
        fontsize=11,
    )
    page1.insert_image(pymupdf.Rect(60, 250, 520, 500), filename=client_srv)

    page2 = doc.new_page()
    page2.insert_textbox(
        pymupdf.Rect(50, 50, 545, 100),
        "## HTTP Status Codes",
        fontsize=20,
    )
    page2.insert_textbox(
        pymupdf.Rect(50, 110, 545, 140),
        "The server returns the following status codes.",
        fontsize=11,
    )
    cols = [40, 200]
    rows = [150, 185, 220, 255]
    for x in cols:
        page2.draw_line(pymupdf.Point(x, rows[0]), pymupdf.Point(x, rows[-1]))
    for y in rows:
        page2.draw_line(pymupdf.Point(40, y), pymupdf.Point(360, y))
    page2.draw_rect(pymupdf.Rect(40, rows[0], 360, rows[-1]), color=(0, 0, 0))
    page2.insert_text(pymupdf.Point(58, 178), "200", fontsize=11)
    page2.insert_text(pymupdf.Point(150, 178), "200 OK", fontsize=11)
    page2.insert_text(pymupdf.Point(58, 213), "404", fontsize=11)
    page2.insert_text(pymupdf.Point(150, 213), "404 Not Found", fontsize=11)
    page2.insert_text(pymupdf.Point(58, 248), "500", fontsize=11)
    page2.insert_text(pymupdf.Point(150, 248), "500 Server Error", fontsize=11)
    page2.insert_text(pymupdf.Point(220, 178), "OK", fontsize=11)
    page2.insert_text(pymupdf.Point(220, 213), "Resource not found", fontsize=11)
    page2.insert_text(pymupdf.Point(220, 248), "Internal error", fontsize=11)

    page3 = doc.new_page()
    page3.insert_textbox(
        pymupdf.Rect(50, 40, 545, 90),
        "## User Login Flowchart",
        fontsize=20,
    )
    page3.insert_textbox(
        pymupdf.Rect(50, 95, 545, 120),
        "The login procedure is described below.",
        fontsize=11,
    )
    page3.insert_image(pymupdf.Rect(60, 130, 440, 560), filename=login)

    doc.save(OUT_PATH)
    print(f"[DONE] wrote {OUT_PATH} ({doc.page_count} pages)")
    doc.close()


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    build_pdf()