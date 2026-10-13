"""Render unchanged source scores with highlighted cue anchors for review.

Optional Verovio/CairoSVG only affect review graphics, not cue computation.
Original XML files stay unchanged. Engraved score excerpts retain source text.
"""

from collections import defaultdict
import html
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from validate_score_beat_bridge import write


def render_review(out,scores,cases):
    import verovio
    import cairosvg
    folder=out/'scores';folder.mkdir()
    rendered=[];sections=[]
    for piece,score in scores.items():
        local=[c for c in cases if c['piece']==piece]
        toolkit=verovio.toolkit();toolkit.resetXmlIdSeed(20261013)
        options=dict(inputFrom='musicxml',pageWidth=1800,pageHeight=1350,scale=40,
                     breaks='auto',svgViewBox=True)
        toolkit.setOptions(options)
        if not toolkit.loadFile(str(score.source_path)):raise ValueError('Score rendering failed: '+piece)
        groups=defaultdict(list);colors=defaultdict(dict)
        for case in local:
            ids=json.loads(case['source_xml_ids'])
            pages={toolkit.getPageWithElement(n) for n in ids if n is not None}
            if 0 in pages or not pages:raise ValueError('Review source note ID missing in engraving: '+case['case_id'])
            for page in pages:
                groups[page].append(case)
                for i,identifier in enumerate(ids):
                    if identifier is not None and toolkit.getPageWithElement(identifier)==page:
                        colors[page][identifier]='#2563eb' if i==0 and len(ids)>1 else '#dc2626'
        slug=piece.replace('/midi_score.mid','').replace('/','_')
        for page,page_cases in sorted(groups.items()):
            svg=toolkit.renderToSVG(page)
            # Source IDs, engraved glyphs and original text remain present.
            node=ET.fromstring(svg);style=ET.Element('{http://www.w3.org/2000/svg}style')
            style.text='\n'.join(f'#{identifier}, #{identifier} * {{fill:{color}!important;stroke:{color}!important;}}'
                                  for identifier,color in sorted(colors[page].items()))
            node.insert(0,style);svg=ET.tostring(node,encoding='unicode')
            filename=f'{slug}_page_{page:02d}'
            (folder/(filename+'.svg')).write_text(svg)
            cairosvg.svg2png(bytestring=svg.encode(),write_to=str(folder/(filename+'.png')),
                            output_width=1600,background_color='white')
            for case in page_cases:
                rendered.append(dict(case_id=case['case_id'],piece=piece,page=page,
                    svg_path='scores/'+filename+'.svg',png_path='scores/'+filename+'.png',
                    source_xml_ids=case['source_xml_ids'],verovio_version=toolkit.getVersion()))
            listing=''.join('<li><strong>'+html.escape(c['case_id'])+'</strong> — '+
                html.escape(c['selection_reason'])+'; quarter '+html.escape(c['position_quarters'])+
                '; '+html.escape(c['scope'])+'; '+html.escape(c['cue_kinds'])+'</li>' for c in page_cases)
            sections.append('<section><h2>'+html.escape(piece.replace('/midi_score.mid',''))+
                ' · page '+str(page)+'</h2><ul>'+listing+'</ul><img src="scores/'+filename+
                '.png" alt="악보 검수 페이지"><p><a href="scores/'+filename+'.svg">확대 가능한 원본 SVG</a></p></section>')
    write(out/'rendered_cases.csv',rendered)
    (out/'score_review.html').write_text('''<!doctype html><html lang="ko"><meta charset="utf-8">
<title>프레이즈 후보 악보 검수</title><style>body{font-family:system-ui,sans-serif;max-width:1400px;margin:32px auto;padding:0 24px;color:#1f2937;background:#f8fafc}section{background:white;border:1px solid #cbd5e1;border-radius:12px;padding:20px;margin:24px 0}img{width:100%;height:auto}li{margin:8px 0}a{color:#1d4ed8}</style>
<h1>프레이즈 후보 악보 검수</h1><p>빨강·파랑 음표/쉼표는 후보의 근거 기호입니다. 표시한 음표 뒤의 경계 또는 쉼의 시작을 후보로 저장했습니다. 색칠 자체가 프레이즈 확정을 뜻하지 않습니다.</p>
<p>기준은 쓰인 악보의 quarter 위치이며, 반복 회차는 별도 연결표에 있습니다. 같은 slur가 짧은 연결·동기·프레이즈 중 무엇에 해당하는지 주변 선율과 다른 성부를 함께 확인하세요.</p>'''+''.join(sections)+
        '<p>입력: 고정 nASAP MusicXML. 원본 악보 및 포함된 출처 문구를 보존했으며 연주 feature·음원은 사용하지 않았습니다.</p></html>')
    (out/'renderer.json').write_text(json.dumps(dict(verovio=verovio.toolkit().getVersion(),
        cairosvg=cairosvg.__version__,options=options,rendered_pages=len(sections),
        purpose='Source notation inspection only, not phrase accuracy validation'),indent=2)+'\n')
