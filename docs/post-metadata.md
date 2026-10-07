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
- `hidden: true`: 참고용 글의 기존 목록 제외 설정입니다. 주제별 글 목록에도 숨김 글을 넣지 않습니다.

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

`_data/topics.yml`은 `id`, `group`, `title`, `description`, `tag`를 갖는 목록입니다. `group`은 `ai`, `physics`, `essays`, `reference` 중 하나입니다. 주제의 소개는 이 파일에서 관리하고, 글 목록은 각 글의 `topic`에서 자동으로 모읍니다. 모든 주제 페이지는 언어 선택 아래에 `Latest articles`와 `All articles` 버튼을 제공합니다. 기본 화면은 선택한 언어의 최신 5편이며, 전체 보기 버튼의 편수도 선택한 언어를 기준으로 계산합니다. 추천 목록은 따로 표시하지 않습니다.

`_data/topic_groups.yml`에서 사이드바와 Topics 화면의 상위 그룹 이름·설명·표시 순서를 함께 관리합니다. `physics` 그룹의 표시 이름은 `Physics, Mathematics & Algorithms`이며 물리·수학·알고리즘의 개념, 유도 과정, 계산 방법을 다룹니다. 현재 하위 주제인 Quantum Transport의 이름과 설명은 실제 글의 범위에 맞게 유지합니다. Essays의 설명은 `Thoughts on everyday life.`입니다.

AI · Kaggle의 첫 화면과 사이드바에는 `_data/topic_categories.yml`에 정의한 세 분류만 표시합니다. AI Agent Harness Engineering은 Gemma 4, AI Agent Security, ARC-AGI-3를 묶습니다. Tabular & Predictive Modeling은 Playground·March Machine Learning Mania 등 표형 데이터 예측과 영상·음향·분자·지층 데이터의 예측 모델링을 함께 모읍니다. Reinforcement Learning & Game Agents는 Orbit Wars, Pokémon TCG, Kaggriculture, Maze Crawler의 강화학습 및 게임 에이전트 연구를 묶습니다. 각 분류의 `topics`에 대회별 주제 ID를 적으며 모든 AI 주제를 정확히 한 분류에 배정합니다.

글의 `topic`과 기존 대회별 안내 URL은 유지합니다. AI 상위 분류와 기존 AI 주제 페이지의 글 목록은 대회·주제별로 묶습니다. 한 편뿐인 주제도 같은 접힌 목록을 사용합니다. 상위 분류의 `All articles`에는 해당 분류에 속한 모든 주제 묶음과 글이 들어가며, 별도의 `Competition guides` 메뉴나 대회별 카드 목록은 표시하지 않습니다. 글과 기존 대회별 안내에서도 상위 분류로 이동할 수 있습니다. 물리·수학·알고리즘과 에세이는 기존 하위 주제 구성을 사용하고, 독립된 글은 접힌 묶음 없이 바로 표시합니다.

최신 보기에는 선택한 언어의 최근 5편에 포함된 글만 최신순으로 표시하며, `View all articles`로 해당 주제의 전체 묶음을 펼칩니다. 모든 AI 주제 묶음은 최신 보기와 전체 보기 모두 기본으로 접혀 있습니다. Latest/All 버튼으로 보기를 바꿀 때도 접힌 상태로 돌아가고, 사용자가 묶음의 제목이나 전체 글 링크를 직접 선택했을 때 펼칩니다. 묶음의 글 수와 전체 글 링크의 편수는 선택한 언어를 기준으로 계산합니다. ARC-AGI-3는 본편 1·2편과 Research Note R1을 한 주제 묶음에 담고, Playground는 서로 다른 회차를 한 주제 묶음에 담습니다.

`_data/series.yml`은 `id`, `title`, `topic`을 갖는 목록입니다. 현재 AI Agent Security 1–12편, BioHub 1–9편, ROGII 1–3편, ARC-AGI-3 본편 1–2편이 등록되어 있습니다. 글 본문의 시리즈 목차와 이전·다음 링크는 이 실제 시리즈의 같은 언어 글을 `series_order` 순서로 연결합니다. 목록에서 같은 주제로 묶어도 글의 `series`와 `series_order`는 바꾸지 않습니다. ARC-AGI-3 Research Note R1은 본편 3편으로 취급하지 않으며, Playground의 서로 다른 회차나 단일 Working Note에 임의의 연속 순번을 붙이지 않습니다. 기존 시리즈의 전체 목록 앵커도 유지합니다.

새 글을 추가한 뒤 메타데이터 검사와 Jekyll 빌드를 실행하고, 주제 목록·번역 링크·시리즈 순서가 실제 화면에서 맞는지 확인합니다.

AI 상위 분류 카드와 사이드바는 `_data/topic_categories.yml` 순서를 따릅니다. 물리·수학·알고리즘과 에세이는 `_data/topics.yml` 순서입니다. 글 목록의 묶음은 가장 최근 글의 발행일 내림차순이며, 같은 날짜면 묶음 키 순입니다. 최신 보기의 묶음 안에서도 발행일 내림차순·같은 날짜의 `translation_key` 순입니다. 전체 보기에서는 실제 시리즈와 독립 글을 각각 읽기 단위로 보고, 단위의 첫 발행일 오름차순·같은 날짜의 단위 키 순으로 배치합니다. 실제 시리즈 안에서는 `series_order` 순서를 유지하므로 ARC-AGI-3는 본편 1편 → 2편 → R1 순이며, Playground의 독립 회차는 오래된 글부터 표시합니다. 최신 5편은 언어 필터를 적용한 뒤 전체 글의 발행일 내림차순·같은 날짜의 `translation_key` 순으로 선택합니다. 옛 태그 모음도 최신순이며 같은 날짜·번역 키의 글은 한국어판을 먼저 표시합니다.

오른쪽 최근 업데이트는 수정 시각의 최신순이며, 같은 수정 시각이면 발행일의 최신순으로 정렬합니다. 한·영 번역 쌍은 한 편으로 묶고, 현재 페이지의 언어에 맞는 버전을 우선 연결합니다. 해당 언어가 없으면 한국어판, 영어판 순으로 선택합니다. 숨김 글은 이 목록에서도 제외합니다.

```sh
bundle exec ruby tools/check_blog_metadata.rb
JEKYLL_ENV=production bundle exec jekyll build
bundle exec htmlproofer _site --disable-external
```

이 Mac에서 시스템 Ruby 대신 설치된 Ruby를 쓰려면 명령 앞에 `PATH=/opt/homebrew/opt/ruby/bin:$PATH`를 붙입니다. 전체 정리 때의 본문·주소 보존 기록은 `_draft/_checks/layout-20261007/`에 있습니다.

사이드바의 PC 접기 기능은 모바일 기본 메뉴와 따로 동작하며 선택을 브라우저에 저장합니다. 사이드바 안의 AI·Kaggle, 물리·수학·알고리즘, 에세이 그룹은 각각 접고 펼칠 수 있고, 글이나 주제 안내를 읽을 때는 해당 그룹이 자동으로 열립니다. 주제 링크는 공개 주제 데이터에서 자동 생성합니다. 메뉴가 화면보다 길면 메뉴 영역 안에서 스크롤합니다.

주제 목록은 영어가 기본입니다. 영어 기본값을 도입하기 전 브라우저에 저장된 한글·전체 선택은 새 설정에 넘기지 않고, 이후 명시적으로 고른 언어는 `pilkwang:language:v2`에 저장합니다. `?lang=ko` 또는 `?lang=en` 링크로 특정 언어를 바로 열 수 있고 `All`에서도 영어 제목을 우선 표시합니다. `?view=all`이나 `#all-articles`로 전체 글을 바로 열 수 있습니다. JavaScript를 사용할 수 없으면 전체 목록과 언어별 링크를 제공하고, 그룹 접기는 기본 HTML 기능으로 동작합니다.

메뉴와 주제 안내 UI는 사이트 설정의 영어를 사용합니다. `_includes/lang.html`은 글의 `page.lang` 대신 `site.lang`으로 UI 언어를 결정합니다. 글의 실제 언어 정보는 HTML의 `lang`과 번역 대상 링크의 `hreflang`에 유지합니다. `Article language`의 `All`, `Korean`, `English` 버튼은 글 목록을 거르는 기능이며 메뉴 언어를 바꾸지 않습니다. 주제 페이지 자체의 소개와 안내는 영어입니다. 최근 업데이트는 UI 언어와 별도로 현재 글의 언어를 우선하므로 캐시 키에도 `article_lang`을 포함합니다.

페이지가 이전 PWA 캐시의 탐색 스크립트·스타일과 섞이지 않도록, 로컬 스타일 소스·탐색 JavaScript·테마 버전을 고정한 Gemfile의 내용으로 만든 공통 버전을 두 파일 URL의 `v` 매개변수에 붙입니다. `_layouts/default.html`은 테마 head를 그대로 사용하면서 CSS URL만 버전으로 구분합니다. 같은 소스는 로컬과 배포 빌드에서 같은 버전을 사용합니다. 빌드 중 환경별로 생성되는 Gemfile.lock은 버전 계산에서 제외합니다.

로컬 레이아웃과 스타일은 Chirpy 7.6.0을 기준으로 작성되어 `Gemfile`에 버전을 고정했습니다. 테마를 올릴 때는 `_layouts/default.html`, `_layouts/post.html`, `_includes/lang.html`, `_includes/sidebar.html`, `_includes/topbar.html`, `_includes/post-nav.html`, `_includes/update-list.html`, `assets/css/jekyll-theme-chirpy.scss`와 새 테마의 원본을 비교한 뒤 PC·모바일 화면을 확인합니다. `_sass/custom.scss`에는 화면 확대율이 만드는 소수 픽셀 너비에서도 메뉴 버튼이 사라지지 않도록 전환 경계를 보완한 스타일이 있습니다.
