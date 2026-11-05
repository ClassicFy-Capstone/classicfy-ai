"""Small written scores with hand-computable quarter positions."""

from pathlib import Path
import tempfile
import unittest

from preprocessing.musicxml_score import read_musicxml


def attributes(beats='4', denominator=4, divisions=4, fifths=0, mode=None):
    mode_xml = '' if mode is None else f'<mode>{mode}</mode>'
    return (f'<attributes><divisions>{divisions}</divisions><key><fifths>{fifths}</fifths>{mode_xml}</key>'
            f'<time><beats>{beats}</beats><beat-type>{denominator}</beat-type></time></attributes>')


def note(identifier, step='C', duration=4, *, voice='1', staff='1', extra='', chord=False,
         grace=False, rest=False, alter=0, octave=4):
    identifier = '' if identifier is None else f' id="{identifier}"'
    body = '<rest/>' if rest else f'<pitch><step>{step}</step><alter>{alter}</alter><octave>{octave}</octave></pitch>'
    return (f'<note{identifier}>' + ('<chord/>' if chord else '') + ('<grace/>' if grace else '')
            + body + ('' if grace else f'<duration>{duration}</duration>')
            + ('' if voice is None else f'<voice>{voice}</voice>')
            + ('' if staff is None else f'<staff>{staff}</staff>') + extra + '</note>')


def direction(body, *, staff=None, voice=None, offset=None):
    return ('<direction><direction-type>' + body + '</direction-type>'
            + ('' if offset is None else f'<offset>{offset}</offset>')
            + ('' if voice is None else f'<voice>{voice}</voice>')
            + ('' if staff is None else f'<staff>{staff}</staff>') + '</direction>')


def measure(body, number=1, *, implicit=False):
    return f'<measure number="{number}" implicit="{"yes" if implicit else "no"}">{body}</measure>'


def score_xml(*measures):
    return ('<?xml version="1.0"?><score-partwise version="4.0"><part-list>'
            '<score-part id="P1"><part-name>Piano</part-name></score-part></part-list>'
            '<part id="P1">' + ''.join(measures) + '</part></score-partwise>')


def example_xml():
    first = (attributes() + direction('<words>Allegro</words>')
             + direction('<metronome><beat-unit>quarter</beat-unit><per-minute>120</per-minute></metronome>')
             + direction('<dynamics><p/></dynamics>')
             + note('c1', extra='<tie type="start"/><notations><slur type="start" number="1"/><articulations><staccato/></articulations></notations>')
             + note('e1', 'E', chord=True, extra='<notations><articulations><tenuto/></articulations></notations>')
             + note('c2', duration=8, extra='<tie type="stop"/><notations><slur type="stop" number="1"/></notations>')
             + note('rest1', rest=True))
    second = (attributes(beats='3', fifths=2, mode='minor')
              + direction('<wedge type="crescendo" number="1"/>')
              + note('grace1', 'D', grace=True, extra='<beam number="1">begin</beam>')
              + note('grace2', 'E', grace=True, extra='<beam number="1">end</beam>')
              + note('f1', 'F', alter=1)
              + direction('<dynamics><sfz/></dynamics>') + note('g1', 'G')
              + direction('<wedge type="stop" number="1"/>') + note('a1', 'A'))
    return score_xml(measure(first), measure(second, 2))


class XMLFixture(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.folder = Path(self.temporary.name)
        self.serial = 0

    def read(self, xml=None, *, traversal=None):
        self.serial += 1
        path = self.folder / f'score-{self.serial}.musicxml'
        path.write_text(example_xml() if xml is None else xml, encoding='utf-8')
        return read_musicxml(path, traversal=traversal)
