// KCC(한국컴퓨터종합학술대회) 논문 양식을 따르는 조판 규칙.
// 근거: 학부 제공 `보고서양식(KCC학술대회).hwp` 의 유의사항 —
//   A4 세로 · 여백 위 30mm / 아래 20mm / 좌우 10mm · 제목(국문·영문)과 요약은 1단, 본문·참고문헌은 2단(단 간격 5mm)
//   장·절 번호는 1., 1.1 · 그림 명칭은 아래, 표 명칭은 위(그림 1, 표 1) · 참고문헌은 인용 순서대로 번호
//   글자 9pt 이상(글꼴 자유) · PDF 제출 권장.
// 표·그림 안의 글자도 9pt 이상으로 둔다.

#let serif = ("Noto Serif CJK KR",)
#let sans = ("Noto Sans CJK KR",)
#let mono = ("D2Coding ligature", "DejaVu Sans Mono")   // 폭이 좁은 고정폭(0.5em): 9pt 로도 표의 식별자가 들어간다

#let ink2 = luma(70)      // 보조 글자색(머리말·그림 안 설명)
#let rule = luma(120)     // 표·그림의 선

#let kcc-paper(
  header-text: none,
  title-ko: none,
  title-en: none,
  authors-ko: none,
  affiliation-ko: none,
  email: none,
  authors-en: none,
  affiliation-en: none,
  abstract: none,
  doc-title: none,
  draft-note: none,     // 초안일 때 제목 위에 두는 알림(none 이면 없음)
  body,
) = {
  set document(title: doc-title)
  set page(
    paper: "a4",
    margin: (top: 30mm, bottom: 20mm, left: 10mm, right: 10mm),
    columns: 2,
    header: if header-text != none {
      block(width: 100%, inset: (bottom: 1.6mm), stroke: (bottom: 0.5pt + rule), {
        set text(font: sans, size: 9pt, fill: ink2)
        align(center, header-text)
      })
    },
    header-ascent: 7mm,
    footer: context align(center, text(font: serif, size: 9pt, counter(page).display("1"))),
    footer-descent: 8mm,
  )
  set columns(gutter: 5mm)
  // overhang: false — 문장 부호가 여백으로 삐져나가지 않게(여백 규정을 엄격히 지킨다)
  set text(font: serif, size: 10pt, lang: "ko", region: "KR", overhang: false)
  set par(justify: true, leading: 0.64em, spacing: 0.64em, first-line-indent: (amount: 1em, all: true))

  // 장·절: 1., 1.1
  set heading(numbering: (..n) => {
    let p = n.pos()
    if p.len() == 1 { numbering("1.", ..p) } else { numbering("1.1", ..p) }
  })
  show heading: it => {
    set text(font: serif, weight: "bold", size: if it.level == 1 { 10.5pt } else { 10pt })
    set par(first-line-indent: 0pt, justify: false)
    v(if it.level == 1 { 1.0em } else { 0.75em }, weak: true)
    block(below: 0.65em, sticky: true, {
      if it.numbering != none {
        counter(heading).display(it.numbering)
        h(0.5em)
      }
      it.body
    })
  }

  // 그림(명칭 아래)·표(명칭 위)
  set figure(gap: 0.7em)
  show figure.where(kind: image): set figure(supplement: [그림])
  show figure.where(kind: table): set figure(supplement: [표])
  set figure.caption(separator: h(0.5em))
  show figure.caption: it => {
    set text(size: 9pt)
    set par(justify: true, first-line-indent: 0pt, leading: 0.5em)
    align(left, it)
  }
  show figure.where(kind: table): set figure.caption(position: top)
  show figure: set block(breakable: false)
  show figure: set par(first-line-indent: 0pt)
  set table(stroke: none, inset: (x: 2.6pt, y: 2.7pt))
  show table: set text(size: 9pt)
  show table: set par(justify: false, first-line-indent: 0pt, leading: 0.45em)

  // raw 의 기본 크기(0.8em)에 곱해진다: 0.8 × 1.25 = 주변 글자와 같은 크기(본문 10pt, 표 9pt)
  show raw: set text(font: mono, size: 1.25em)
  set list(indent: 0.6em, body-indent: 0.4em, marker: [•])
  set enum(indent: 0.6em, body-indent: 0.4em)

  // 참고문헌: 저자, 제목, 학술지명, 권, 호, 쪽수, 발행년도 순(kcc.csl). 서지 용어는 영어로 둔다.
  set bibliography(style: "kcc.csl", title: none)
  // 영문 서지는 라틴 글꼴을 앞에 둔다(기울임꼴과 ć 같은 확장 라틴 문자가 필요하다). URL 이 줄을 벌리지 않게 왼쪽 맞춤.
  show bibliography: set text(font: ("Libertinus Serif",) + serif, size: 9pt, lang: "en", region: "US")
  show bibliography: set par(justify: false, first-line-indent: 0pt, leading: 0.5em, spacing: 0.55em)

  // 제목부(1단)
  place(top + center, float: true, scope: "parent", clearance: 7mm, {
    set par(justify: false, first-line-indent: 0pt, leading: 0.6em, spacing: 0.7em)
    set align(center)
    if draft-note != none {
      block(width: 100% - 18mm, inset: (x: 3mm, y: 1.8mm), stroke: 0.6pt + rule, radius: 2pt, {
        set text(font: sans, size: 9pt)
        set par(justify: false, leading: 0.5em)
        align(left, draft-note)
      })
      v(4mm, weak: true)
    }
    text(font: sans, size: 16pt, weight: "bold", title-ko)
    v(5mm, weak: true)
    text(size: 10pt, authors-ko)
    linebreak()
    text(size: 10pt, affiliation-ko)
    linebreak()
    text(size: 10pt, email)
    v(6mm, weak: true)
    text(font: sans, size: 14pt, weight: "bold", title-en)
    v(4.5mm, weak: true)
    text(size: 10pt, authors-en)
    linebreak()
    text(size: 10pt, affiliation-en)
    v(7mm, weak: true)
    text(size: 10pt, weight: "bold", [요#h(1.2em)약])
    v(3mm, weak: true)
    block(width: 100% - 18mm, {
      set align(left)
      set text(size: 9pt)
      set par(justify: true, first-line-indent: (amount: 1em, all: true), leading: 0.62em)
      abstract
    })
  })

  body
}

// 참고문헌 제목(번호 없는 가운데 정렬)
#let refs-title = {
  v(1.0em, weak: true)
  align(center, text(weight: "bold", size: 10.5pt, [참#h(0.5em)고#h(0.5em)문#h(0.5em)헌]))
  v(0.7em, weak: true)
}
