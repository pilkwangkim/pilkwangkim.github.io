# 글의 분류와 언어 정보

글의 주소를 유지하려면 기존 파일명과 `permalink`를 바꾸지 않습니다. `categories`는 기존 Chirpy 분류에 사용하고, 주제별 안내는 아래 메타데이터로 구성합니다.

```yaml
categories: [AI, Kaggle]
tags: [kaggle, biohub, cell-tracking, microscopy, oof, working-note]
topic: biohub
lang: ko
translation_key: biohub-01
series: biohub
series_order: 1
```

- `topic`: `_data/topics.yml`에 등록된 주제 ID 하나를 사용합니다. 해당 주제의 `tag`도 글의 태그에 포함합니다.
- `lang`: 한국어는 `ko`, 영어는 `en`입니다. 글의 언어를 나타내는 `korean` 태그는 사용하지 않습니다.
- `translation_key`: 같은 글의 한국어판과 영어판에 동일한 값을 사용합니다. 번역이 없는 글도 고유한 키를 가집니다. 키 하나에 같은 언어의 글이 둘 이상 들어가면 안 됩니다.
- `series`, `series_order`: 실제 연속된 시리즈에만 사용합니다. `_data/series.yml`의 ID와 양의 정수 순번을 함께 적습니다. 번역 쌍은 같은 ID와 순번을 사용합니다.
- `hidden: true`: 참고용 글의 기존 목록 제외 설정입니다. 주제별 추천 목록에도 숨김 글을 넣지 않습니다.

## 태그 작성 원칙

태그는 영어 소문자와 숫자를 사용하며 단어는 하이픈으로 연결합니다. `cheat-sheet`, `cross-validation`, `playground-series`처럼 표기를 통일합니다. 주제의 대표 태그와 분야 태그를 앞에 두고, 나머지는 알파벳순으로 정리합니다. 한국어판과 영어판은 의미상 같은 태그 목록을 사용합니다.

새 태그를 만들기 전에 기존 글과 `_data/tag_aliases.yml`을 확인합니다. 기존 표기를 바꾸면 옛 태그 URL의 slug를 별칭 데이터에 추가해 기존 링크를 유지합니다. 단순 표기 변경으로 URL slug가 같다면 별칭이 필요하지 않습니다. 예를 들어 `cheat sheet`를 `cheat-sheet`로 바꾸어도 slug는 `cheat-sheet`입니다. 과거 `/tags/korean/`은 한국어 글 모음으로 유지합니다.

서로 다른 의미의 키워드는 합치지 않습니다. 예를 들어 `gemma`와 `gemma-4`, `leakage`와 `leakage-control`, `tool-use`와 `tool-calling`, `portfolio-design`과 `model-portfolio`는 각 글에서 다루는 대상이나 범위가 다릅니다.

현재 통일한 표기는 다음과 같습니다.

| 이전 표기 | 사용할 표기 |
| --- | --- |
| `playground` | `playground-series` |
| `cv` | `cross-validation` |
| `MMLM` | `march-machine-learning-mania` |
| `benchmarks` | `benchmark` |
| `agents` | `ai-agents` |
| `auc` | `roc-auc` |
| `birdclef` | `birdclef-2026` |
| `rsna` | `rsna-knee` |
| `casmi` | `casmi-2026` |
| `retrospective` | `competition-retrospective` |
| `cheat sheet` | `cheat-sheet` |

`roc-auc`는 ROC-AUC를 다루는 글에 사용합니다. PR-AUC 등 다른 지표는 별도의 태그로 구분합니다. 대회명이 붙은 태그는 해당 연도·대회를 다루는 글에 사용합니다.

## 주제와 시리즈 관리

`_data/topics.yml`은 `id`, `group`, `title`, `description`, `tag`를 갖는 목록입니다. `group`은 `ai`, `physics`, `essays`, `reference` 중 하나입니다. 주제의 소개와 추천 읽기 순서는 이 파일에서 관리하고, 전체 글 목록은 각 글의 `topic`에서 자동으로 모읍니다. `recommended`에는 URL 대신 `translation_key`를 적어 언어별 링크를 선택할 수 있게 합니다.

`_data/series.yml`은 `id`, `title`, `topic`을 갖는 목록입니다. 현재 AI Agent Security 1–12편, BioHub 1–9편, ROGII 1–3편, ARC-AGI-3 본편 1–2편이 등록되어 있습니다. ARC-AGI-3 Research Note R1은 본편 3편으로 취급하지 않습니다. Playground의 서로 다른 회차나 단일 Working Note에 임의의 연속 순번을 붙이지 않습니다.

새 글을 추가한 뒤 메타데이터 검사와 Jekyll 빌드를 실행하고, 주제 목록·번역 링크·시리즈 순서가 실제 화면에서 맞는지 확인합니다.

주제 카드와 사이드바의 주제 순서는 `_data/topics.yml`에 적은 순서를 함께 따릅니다. 추천 목록은 `recommended`에 적은 순서, 전체 글 목록은 최신순, 시리즈 목차는 `series_order` 순입니다. 전체 글의 날짜가 같으면 `translation_key` 순으로 정렬합니다. 옛 태그 모음도 최신순이며 같은 날짜·번역 키의 글은 한국어판을 먼저 표시합니다.

```sh
bundle exec ruby tools/check_blog_metadata.rb
JEKYLL_ENV=production bundle exec jekyll build
bundle exec htmlproofer _site --disable-external
```

이 Mac에서 시스템 Ruby 대신 설치된 Ruby를 쓰려면 명령 앞에 `PATH=/opt/homebrew/opt/ruby/bin:$PATH`를 붙입니다. 전체 정리 때의 본문·주소 보존 기록은 `_draft/_checks/layout-20261007/`에 있습니다.

사이드바의 PC 접기 기능은 모바일 기본 메뉴와 따로 동작하며 선택을 브라우저에 저장합니다. 사이드바 안의 AI·Kaggle, 물리, 에세이 그룹은 각각 접고 펼칠 수 있고, 글이나 주제 안내를 읽을 때는 해당 그룹이 자동으로 열립니다. 주제 링크는 공개 주제 데이터에서 자동 생성합니다. 메뉴가 화면보다 길면 메뉴 영역 안에서 스크롤합니다.

주제 목록의 언어 선택도 저장되고, `?lang=ko` 또는 `?lang=en` 링크로 특정 언어를 바로 열 수 있습니다. 직접 링크로 선택한 언어도 이후 주제 탐색에 유지됩니다. JavaScript를 사용할 수 없으면 전체 목록과 언어별 링크를 그대로 제공하고, 그룹 접기는 기본 HTML 기능으로 동작합니다.

로컬 레이아웃과 스타일은 Chirpy 7.6.0을 기준으로 작성되어 `Gemfile`에 버전을 고정했습니다. 테마를 올릴 때는 `_layouts/post.html`, `_includes/sidebar.html`, `_includes/topbar.html`, `_includes/post-nav.html`, `assets/css/jekyll-theme-chirpy.scss`와 새 테마의 원본을 비교한 뒤 PC·모바일 화면을 확인합니다. `_sass/custom.scss`에는 화면 확대율이 만드는 소수 픽셀 너비에서도 메뉴 버튼이 사라지지 않도록 전환 경계를 보완한 스타일이 있습니다.
