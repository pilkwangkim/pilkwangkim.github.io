---
title: "Markdown 문법 & Chirpy 테마 치트시트"
date: 2025-12-12 16:00:00 +0900
categories: [Reference, Private]
tags: [markdown, latex, cheat sheet]
math: true
hidden: true
---

이 글은 `hidden: true` 설정이 되어 있어 블로그 목록에는 뜨지 않습니다.
URL을 통해 들어와서 문법을 참고하는 용도로 사용하세요.

---

## 1. 텍스트 강조 (Text Formatting)

- **굵게(Bold)**: `**텍스트**` (단축키: `Ctrl + B`)
- *기울임(Italic)*: `*텍스트*` (단축키: `Ctrl + I`)
- ~~취소선(Strikethrough)~~: `~~텍스트~~`
- `인라인 코드`: 문장 중간에 `변수명` 강조 (백틱 ` 사용)

---

## 2. 수식 입력 (LaTeX)

> **필수:** 머리말에 `math: true`가 있어야 작동합니다.
{: .prompt-warning }

### 인라인 수식 (본문 중간)
문장 중간에 $E = mc^2$ 처럼 수식을 넣을 때는 달러($) 하나로 감쌉니다.
`$ \vec{F} = m\vec{a} $`

### 블록 수식 (별도 줄)
달러($) 두 개로 감쌉니다.

$$
i\hbar \frac{\partial}{\partial t} \Psi = \hat{H} \Psi
$$

**작성법:**
```latex
$$
\int_{-\infty}^{\infty} e^{-x^2} dx = \sqrt{\pi}
$$
```

---

## 3. 코드 블록 (Code Block)

백틱(`) 3개를 사용하며, 언어 이름(python, bash, c++)을 적으면 색상이 입혀집니다.

```python
import numpy as np

def calculate_bandgap(materials):
    """
    밴드갭 계산 함수
    """
    return 1.12 # eV
```

```bash
$ git add .
$ git commit -m "Update post"
$ git push origin main
```

---

## 4. 이미지 및 크기 조절 (Chirpy 전용)

### 기본 이미지
`![설명](/assets/img/sample.png)`

### 크기 및 정렬 조절
이미지 코드 바로 뒤에 `{ ... }`를 붙입니다. (미리보기에선 텍스트로 보일 수 있음)

- **너비 50%**: `![설명](/assets/img/file.png){: .w-50 }`
- **너비 75%**: `![설명](/assets/img/file.png){: .w-75 }`
- **왼쪽 정렬**: `![설명](/assets/img/file.png){: .left }`
- **오른쪽 정렬**: `![설명](/assets/img/file.png){: .right }`
- **그림자 효과**: `![설명](/assets/img/file.png){: .shadow }`

---

## 5. 인용문 및 프롬프트 (Alerts)

### 일반 인용구
> 이것은 일반적인 인용구입니다.
> `>` 기호를 사용합니다.

### Chirpy 전용 프롬프트 (강력 추천 ⭐)
`>` 뒤에 내용을 쓰고, **바로 다음 줄**에 속성을 적습니다.

> **팁:** 글을 쓸 때 미리보기 단축키는 `Ctrl + K` 누르고 `V` 입니다.
{: .prompt-tip }

> **정보:** 이 기능은 Jekyll 4.0 이상에서 지원됩니다.
{: .prompt-info }

> **주의:** `_config.yml` 파일은 함부로 수정하지 마세요.
{: .prompt-warning }

> **위험:** 파일을 삭제하면 복구할 수 없습니다.
{: .prompt-danger }

---

## 6. 목록 (Lists)

**순서 없는 목록:**
- 아이템 1 (`-` 또는 `*` 사용)
- 아이템 2
  - 들여쓰기 (탭 키)

**순서 있는 목록:**
1. 첫 번째
2. 두 번째
3. 세 번째

---

## 7. 표 (Tables)

| 파라미터  | 값 (Unit) |              설명 |
| :-------- | :-------: | ----------------: |
| `ECUT`    |  520 eV   |     Cutoff Energy |
| `KPOINTS` |   4x4x4   |    Monkhorst-Pack |
| `ISMEAR`  |     0     | Gaussian Smearing |

---

## 8. 링크 (Links)

- [내 블로그 홈으로 가기](https://pilkwangkim.github.io)
- `[표시할 텍스트](URL주소)`