#!/usr/bin/env python
# build the slide deck as pptx, blue on white, positions checked so nothing overlaps
import os
from datetime import datetime

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = f"{ROOT}/results/figures"
AUTHOR = os.environ.get("DECK_AUTHOR", "")   # put your name here or set DECK_AUTHOR

NAVY = RGBColor(0x0E, 0x23, 0x3F)
BLUE = RGBColor(0x1C, 0x6D, 0xC7)
PALE = RGBColor(0xEC, 0xF2, 0xFA)
GREY = RGBColor(0x59, 0x66, 0x76)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

SW, SH = 13.333, 7.5
MG = 0.85
COL = (SW - 2 * MG - 0.5) / 2          # two column width
RIGHT = MG + COL + 0.5                  # left edge of the right column
BODY_TOP, BODY_BOT = 1.75, 6.75
FONT = "Helvetica Neue"

_boxes = []


def claim(x, y, w, h, tag):
    """Keep a register of occupied rectangles and refuse to place anything on top."""
    assert x >= 0 and y >= 0.3 and x + w <= SW + 1e-6 and y + h <= SH + 1e-6, f"{tag} off slide"
    for bx, by, bw, bh, btag in _boxes:
        if x < bx + bw - 1e-6 and bx < x + w - 1e-6 and y < by + bh - 1e-6 and by < y + h - 1e-6:
            raise AssertionError(f"{tag} overlaps {btag}")
    _boxes.append((x, y, w, h, tag))


def slide(prs):
    _boxes.clear()
    return prs.slides.add_slide(prs.slide_layouts[6])


def text(s, x, y, w, h, tag, anchor=MSO_ANCHOR.TOP):
    claim(x, y, w, h, tag)
    box = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    f = box.text_frame
    f.word_wrap = True
    f.vertical_anchor = anchor
    return f


def line(f, txt, size, color, bold=False, after=8, first=False, space_before=0):
    par = f.paragraphs[0] if first else f.add_paragraph()
    par.space_after = Pt(after)
    par.space_before = Pt(space_before)
    par.line_spacing = 1.15
    for chunk in [txt]:
        run = par.add_run()
        run.text = chunk
        run.font.size = Pt(size)
        run.font.color.rgb = color
        run.font.bold = bold
        run.font.name = FONT
    return par


def header(prs, title, kicker=None):
    """Rule sits right under whatever the header ends with, so there is no dead space."""
    global BODY_TOP
    s = slide(prs)
    f = text(s, MG, 0.52, SW - 2 * MG, 0.68, "title")
    line(f, title, 30, NAVY, bold=True, after=2, first=True)
    if kicker:
        f2 = text(s, MG, 1.26, SW - 2 * MG, 0.32, "kicker")
        line(f2, kicker, 14, GREY, after=0, first=True)
        rule_y, BODY_TOP = 1.64, 1.75
    else:
        rule_y, BODY_TOP = 1.26, 1.4
    bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(MG), Inches(rule_y), Inches(0.9), Emu(19050))
    bar.fill.solid()
    bar.fill.fore_color.rgb = BLUE
    bar.line.fill.background()
    bar.shadow.inherit = False
    return s


def est_lines(txt, w, pt):
    """Rough number of rendered lines, counting wraps at about 2.1 characters per point of width."""
    per_line = max(10, int(w * 72 * 2.1 / pt))
    n = 0
    for seg in txt.split("\n"):
        n += max(1, -(-len(seg) // per_line))
    return n


def hang(par, marl=0.30):
    """Wrapped lines line up under the text, not under the marker."""
    pPr = par._p.get_or_add_pPr()
    pPr.set("marL", str(int(marl * 914400)))
    pPr.set("indent", str(int(-marl * 914400)))


def block(s, x, y, w, head, items, hpt=16, bpt=13, gap=0.1, tag=""):
    """A heading with marker lines under it, height measured from the wrapped text."""
    items = items or []
    lh = est_lines(head, w, hpt) * (hpt / 72 * 1.35) + 0.06
    lb = sum(est_lines(t, w - 0.30, bpt) * (bpt / 72 * 1.34) + 0.06 for t in items)
    h = lh + lb + 0.04
    f = text(s, x, y, w, h, tag or head[:14])
    line(f, head, hpt, NAVY, bold=True, after=4, first=True)
    for i, t in enumerate(items):
        par = line(f, "\u2022  " + t, bpt, GREY, after=5 if i < len(items) - 1 else 0)
        hang(par)
    return y + h + gap


def bullets(s, items, x=MG, y=BODY_TOP, w=None, size=17, tag="bul", lead=0.42):
    w = w or (SW - 2 * MG)
    n = sum(1 + txt.count("\n") for txt, _ in [(i, 0) if isinstance(i, str) else (i[0], 0) for i in items])
    h = min(n * lead + 0.3, BODY_BOT - y)
    f = text(s, x, y, w, h, tag)
    for i, it in enumerate(items):
        txt, bold = (it, False) if isinstance(it, str) else it
        line(f, txt, size, NAVY if bold else GREY, bold=bold, after=11, first=(i == 0))
    return y + h


def table(s, rows, x=MG, y=BODY_TOP, w=None, size=13, hl=(), tag="tbl", rh=0.345, widths=None):
    w = w or (SW - 2 * MG)
    h = rh * len(rows)
    claim(x, y, w, h, tag)
    shape = s.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y), Inches(w), Inches(h))
    t = shape.table
    if widths:
        tot = sum(widths)
        for i, cw in enumerate(widths):
            t.columns[i].width = Inches(w * cw / tot)
    for r, row in enumerate(rows):
        t.rows[r].height = Inches(rh)
        for c, val in enumerate(row):
            cell = t.cell(r, c)
            cell.text = str(val)
            cell.fill.solid()
            cell.fill.fore_color.rgb = PALE if r == 0 else WHITE
            cell.margin_left = cell.margin_right = Inches(0.09)
            cell.margin_top = cell.margin_bottom = Inches(0.03)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            pr = cell.text_frame.paragraphs[0]
            pr.alignment = PP_ALIGN.LEFT if c == 0 else PP_ALIGN.RIGHT
            for run in pr.runs:
                run.font.size = Pt(size)
                run.font.name = FONT
                run.font.bold = (r == 0) or (r in hl)
                run.font.color.rgb = NAVY if r == 0 else (BLUE if r in hl else GREY)
    return y + h


def figure(s, name, x, y, w, tag="fig"):
    path = f"{FIG}/{name}"
    if not os.path.exists(path):
        return y
    iw, ih = Image.open(path).size
    h = w * ih / iw
    if y + h > BODY_BOT + 0.3:
        h = BODY_BOT + 0.3 - y
        w = h * iw / ih
    claim(x, y, w, h, tag)
    s.shapes.add_picture(path, Inches(x), Inches(y), Inches(w), Inches(h))
    return y + h


def note(s, txt, y, x=MG, w=None, color=None, size=14, tag="note"):
    w = w or (SW - 2 * MG)
    h = (txt.count("\n") + 1) * 0.26 + 0.14
    f = text(s, x, y, w, h, tag)
    line(f, txt, size, color or BLUE, bold=True, after=0, first=True)


def build():
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(SW), Inches(SH)

    # 1 title, with the outcome stated up front
    s = slide(prs)
    bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(MG), Inches(1.15), Inches(1.3), Emu(25400))
    bar.fill.solid(); bar.fill.fore_color.rgb = BLUE; bar.line.fill.background(); bar.shadow.inherit = False
    f = text(s, MG, 1.42, SW - 2 * MG, 0.95, "t")
    line(f, "MXFP4 и MXFP8 на Qwen3", 44, NAVY, bold=True, after=0, first=True)
    f = text(s, MG, 2.5, SW - 2 * MG, 0.45, "sub")
    line(f, "Четыре размера модели, от 0.6B до 14B", 18, GREY, after=0, first=True)
    table(s, [["Qwen3-4B", "базовый MXFP4", "MXFP4 + доработка"],
              ["размер модели", "2.14 ГБ", "1.98 ГБ"],
              ["сжатие к 16 битам", "3.76x", "4.06x"],
              ["потеря качества, перплексия", "+8.7%", "+2.2%"]],
          y=3.45, w=SW - 2 * MG, size=16, hl=(1, 2, 3), rh=0.44)
    note(s, "Модель на 7% меньше и теряет в качестве вчетверо меньше.\n"
            "Исходная модель без квантования весит 8.04 ГБ.", 5.5)

    # 2 format and memory
    s = header(prs, "Формат и память", "Блок из 32 значений делит один 8-битный показатель степени")
    f = text(s, MG, BODY_TOP, COL, 0.5, "formula")
    line(f, "бит = элемент + 8 / 32", 20, NAVY, bold=True, after=0, first=True)
    table(s, [["формат", "бит на значение", "к bf16"],
              ["MXFP4", "4 + 0.25 = 4.25", "3.765x"],
              ["MXFP8", "8 + 0.25 = 8.25", "1.939x"]], y=2.45, w=COL, size=14)
    note(s, "Блочный scale стоит четверть бита.\nНа MXFP4 это 6% бюджета, поэтому\n«в четыре раза меньше» неверно.", 3.7, w=COL)
    note(s, "Проверено упаковкой в файл:\n316 616 704 байта на диске\nпротив 316 616 704 по формуле.", 5.15, w=COL, color=GREY)
    table(s, [["модель", "bf16", "MXFP8", "MXFP4"],
              ["Qwen3-0.6B", "1.19 ГБ", "0.61", "0.32"],
              ["Qwen3-1.7B", "3.44", "1.77", "0.91"],
              ["Qwen3-4B", "8.04", "4.15", "2.14"],
              ["Qwen3-14B", "29.54", "15.23", "7.85"]], x=RIGHT, y=BODY_TOP, w=COL, size=14, rh=0.4)
    note(s, "929 тензоров, четыре модели,\nразброс по размеру в 25 раз.", 4.05, x=RIGHT, w=COL, color=GREY)

    # 3 layer analysis
    s = header(prs, "Анализ слоёв: две находки", "SQNR по 929 тензорам")
    y = figure(s, "sqnr_by_role.png", MG, BODY_TOP, COL)
    note(s, "Разброс внутри модели 0.07 дБ. Рост модели\nв 25 раз двигает ошибку на 0.12 дБ.\n"
            "Слоёв, которые жмутся лучше, под MX нет.", y + 0.25, w=COL)
    table(s, [["источник", "эксцесс", "SQNR MXFP4"],
              ["гаусс", "3.0", "18.78 дБ"],
              ["Лаплас", "6.0", "17.95 дБ"],
              ["Стьюдент, 3 ст. св.", "121", "16.98 дБ"],
              ["гаусс + выбросы", "244", "15.68 дБ"]], x=RIGHT, y=BODY_TOP, w=COL, size=13, hl=(1,))
    note(s, "Веса трансформера сидят ровно в гауссовой точке:\nэксцесс внутри блока равен 3.0 у всех проекций\n"
            "всех моделей. Ошибку MXFP4 можно предсказать\nиз теории, не трогая чекпоинт.", 3.6, x=RIGHT, w=COL)
    bullets(s, [("Эксцесс тензора ошибку не предсказывает", True),
                "Корреляция −0.10. А внутриблочные статистики\nпредсказывают: −0.44 и −0.54.\n"
                "Блочный scale выдаёт каждому выбросу\nсобственную экспоненту"],
            x=RIGHT, y=4.95, w=COL, size=14, lead=0.38)

    # 4 the method
    s = header(prs, "Предложенный метод", "Три части, ни одна не требует лишних бит")
    y = BODY_TOP
    for h, b in [
        ("1.  Кодирование без потерь",
         ["показатели степени соседних блоков почти равны",
          "кодируем разности, а не значения: энтропия падает с 8 бит до 1.2",
          "поток множителей ужимается в 6 раз"]),
        ("2.  Экспонента блока выбирается, а не выводится формулой",
         ["стандарт кладёт максимум блока в диапазон от 4 до 8, а FP4 представляет до 6",
          "41% блоков MXFP4 теряют старший элемент на насыщении",
          "перебор кандидатов это чинит, размер хранения не меняется"]),
        ("3.  Компенсация ошибки квантования",
         ["остаток блока проецируется на ещё не квантованные столбцы",
          "размер группы задан форматом: ровно блок MX из 32 значений",
          "статистика входа слоя берётся с 16 калибровочных последовательностей"]),
    ]:
        y = block(s, MG, y, SW - 2 * MG, h, b, hpt=17, bpt=14, gap=0.18)
    note(s, "Коды MXFP4 уже на энтропийной границе, 3.81 бита из 4. Весь выигрыш по размеру даёт первая часть.", y + 0.1)

    # 5 compression result
    s = header(prs, "Результат: размер")
    table(s, [["поток", "без кодирования", "после", "выигрыш"],
              ["коды MXFP4", "4.0 бита", "3.86", "1.04x"],
              ["коды MXFP8", "8.0 бит", "6.61", "1.21x"],
              ["scale E8M0", "0.25 бита", "0.040", "6.2x"]], y=BODY_TOP, w=COL, size=14, hl=(3,))
    table(s, [["формат", "по стандарту", "после кодирования", "к bf16"],
              ["MXFP4", "4.25", "3.905", "3.765 → 4.097x"],
              ["MXFP8", "8.25", "6.648", "1.939 → 2.407x"]], y=3.6, w=COL, size=14, hl=(1, 2))
    bullets(s, [("Ноль несовпадений", True),
                "197 и 253 тензора двух моделей,\nоба формата, сжаты и распакованы\nпобитово",
                ("Мы в 3% от предела", True),
                "Контекстные модели дают 0.1 бита,\nно передача контекста стоит 0.16.\n"
                "Остаток берётся только адаптивным\nарифметическим кодером"],
            x=RIGHT, y=BODY_TOP, w=COL, size=14, lead=0.38)

    # 6 quality result, both models
    s = header(prs, "Результат: качество при том же размере", "Перплексия wikitext2, всё по 4.25 бита")
    table(s, [["конфигурация", "Qwen3-0.6B", "Qwen3-4B"],
              ["bf16", "21.32", "15.17"],
              ["MXFP4 базовый", "25.58", "16.48"],
              ["+ выбор экспоненты", "25.44", "16.35"],
              ["+ компенсация", "23.32", "15.70"],
              ["оба приёма", "22.91", "15.50"],
              ["убрано потери", "62.7%", "74.7%"]], y=BODY_TOP, w=COL, size=13, hl=(5, 6), rh=0.39)
    figure(s, "benchmark_Qwen3-4B.png", RIGHT - 0.3, BODY_TOP + 0.15, COL + 0.3)
    note(s, "На бенчмарке метод вернул 73% потери на arc_easy, 86% на winogrande, 82% на lambada.\n"
            "Остаток до bf16 меньше одной сигмы: модель уже неотличима от исходной.", 5.35)

    # 7 what the benchmark says about the formats
    s = header(prs, "Что показал lm-eval", "Полные наборы задач, без подвыборки")
    table(s, [["Qwen3-4B", "arc_easy", "piqa", "winogrande", "lambada ppl"],
              ["bf16", "0.8047", "0.7508", "0.6598", "7.29"],
              ["веса MXFP8", "0.8009", "0.7497", "0.6480", "7.16"],
              ["веса и активации MXFP8", "0.7992", "0.7470", "0.6472", "7.24"],
              ["веса MXFP4, базовый", "0.7811", "0.7356", "0.6322", "8.32"],
              ["MXFP4 + доработка", "0.7984", "0.7307", "0.6559", "7.43"]],
          y=BODY_TOP, w=SW - 2 * MG, size=14, hl=(5,), rh=0.4)
    bullets(s, [("MXFP8 неотличим от bf16, и на весах, и на активациях", True),
                "На Qwen3-14B точность lambada совпала до четвёртого знака: 0.6812 против 0.6812,\n"
                "при вдвое меньшей памяти"],
            y=4.6, w=COL + 1.6, size=15, lead=0.38)
    table(s, [["цена MXFP4 на arc_easy", "bf16", "MXFP4", "потеря"],
              ["Qwen3-0.6B", "0.6082", "0.5362", "7.2 пункта"],
              ["Qwen3-4B", "0.8047", "0.7811", "2.4 пункта"],
              ["Qwen3-14B", "0.8418", "0.8279", "1.4 пункта"]],
          x=RIGHT + 1.6, y=4.6, w=COL - 1.6, size=12, hl=(3,), rh=0.38)

    # 8 negatives
    s = header(prs, "Что не сработало")
    y = BODY_TOP
    for h, b in [
        ("Огрубление кодов на соседний уровень",
         ["исходная идея задания: сдвигает 1.3% кодов, экономит 0.005 бита",
          "причина: 16 символов распределены почти поровну, энтропия 3.81 из 4",
          "это свойство формата, а не недоработка реализации"]),
        ("Распределение бит между слоями по чувствительности",
         ["по прокси-метрике ошибка падала в 5.4 раза",
          "настоящий прогон показал рост перплексии с 25.6 до 36.7",
          "абсолютная и относительная ошибка почти не коррелируют, ранг 0.20"]),
        ("Взвешивание выбора экспоненты по важности активаций",
         ["ухудшило перплексию: 26.04 против 25.58",
          "защита сильных каналов за счёт остальных на уровне блока не окупается"]),
    ]:
        y = block(s, MG, y, SW - 2 * MG, h, b, hpt=17, bpt=14, gap=0.26)

    # 9 conclusions, stated rather than asked
    s = header(prs, "Выводы")
    LW = COL + 1.5
    y = BODY_TOP
    y = block(s, MG, y, LW, "Слои жмутся одинаково",
              ["разброс SQNR по 929 тензорам четырёх моделей составляет 0.07 дБ",
               "рост модели в 25 раз двигает ошибку на 0.12 дБ",
               "тип проекции и глубина влияют ещё слабее"], hpt=17, bpt=14, gap=0.26)
    y = block(s, MG, y, LW, "Внутри блока распределение гауссово",
              ["эксцесс 3.0 у всех типов проекций во всех четырёх моделях",
               "у тензора целиком хвосты тяжёлые, эксцесс доходит до 19",
               "для MX это не важно: у каждого блока свой масштаб"], hpt=17, bpt=14, gap=0.26)
    block(s, MG, y, LW, "Что из этого следует на практике",
          ["послойный подбор бит под MX даёт мало",
           "ошибку MXFP4 можно оценить заранее: около 18.7 дБ",
           "дефект стоит искать в насыщении блока, а не в выбросах весов",
           "экономия лежит в метаданных: scale жмётся в 6 раз, коды почти нет"],
          hpt=17, bpt=14, gap=0.2)
    RX, RW = RIGHT + 1.7, COL - 1.7
    block(s, RX, BODY_TOP, RW, "Верно для выбранных моделей",
          ["на перплексии зависимость от размера немонотонна, Qwen3-1.7B хуже всех четырёх",
           "на бенчмарке та же зависимость монотонна",
           "просадка доработки на arc_easy была только на Qwen3-0.6B",
           "down_proj хуже остальных, но всего на 0.08 дБ"], hpt=15, bpt=13, gap=0.2)

    cp = prs.core_properties
    cp.author = AUTHOR
    cp.last_modified_by = AUTHOR
    cp.title = "MXFP4 и MXFP8 на Qwen3"
    cp.subject = ""
    cp.comments = ""
    cp.category = ""
    cp.keywords = ""
    cp.revision = 1
    now = datetime.now()
    cp.created = now
    cp.modified = now

    out = f"{ROOT}/slides/mxfp-qwen3.pptx"
    prs.save(out)
    print(f"wrote {out}, {len(prs.slides._sldIdLst)} slides, no overlaps")


if __name__ == "__main__":
    build()
