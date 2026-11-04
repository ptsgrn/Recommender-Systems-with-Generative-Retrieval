"""Build editable slides: install python-pptx separately, then run this module."""
import json
from pathlib import Path


def build():
    from pptx import Presentation
    from pptx.dml.color import RGBColor
    from pptx.util import Inches, Pt

    slides = json.loads(Path('docs/slides.json').read_text())
    assert sum(s['seconds'] for s in slides if not s['backup']) == 600
    deck = Presentation()
    deck.slide_width, deck.slide_height = Inches(13.333), Inches(7.5)
    for number, content in enumerate(slides, 1):
        slide = deck.slides.add_slide(deck.slide_layouts[6])
        slide.background.fill.solid()
        slide.background.fill.fore_color.rgb = RGBColor(247, 249, 248)

        def box(x, y, width, height, text, size, color, bold=False):
            shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(width), Inches(height))
            frame = shape.text_frame
            frame.word_wrap = True
            frame.margin_left = frame.margin_right = 0
            paragraph = frame.paragraphs[0]
            paragraph.text = text
            paragraph.font.name = 'Sarabun'
            paragraph.font.size = Pt(size)
            paragraph.font.bold = bold
            paragraph.font.color.rgb = RGBColor(*color)
            return shape

        box(.65, .35, 12, .35, 'TIGER  /  ORAL EXAM  /  ' + ('BACKUP' if content['backup'] else f'{number:02d}'), 13, (26, 108, 105), True)
        box(.65, .95, 12, 1.0, content['title'], 30, (20, 43, 53), True)
        for index, bullet in enumerate(content['bullets']):
            box(.85, 2.15 + index * .92, 11.65, .86, bullet, 23, (30, 48, 57))
        box(.65, 6.35, 12, .36, content['speaker'] + (f"  ·  {content['seconds']} seconds" if not content['backup'] else ''), 13, (26, 108, 105))
        box(.65, 6.92, 12, .38, content['source'], 10, (83, 98, 107))
        slide.notes_slide.notes_text_frame.text = content['notes'] + '\n\nEvidence: ' + content['source']
    destination = Path('docs/oral_exam.pptx')
    deck.save(destination)
    restored = Presentation(destination)
    assert len(restored.slides) == len(slides)
    assert all(s.notes_slide.notes_text_frame.text for s in restored.slides)
    print(f'Built and reopened {destination}: {len(slides)} slides, with notes.')


if __name__ == '__main__':
    build()
