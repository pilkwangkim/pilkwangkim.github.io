# frozen_string_literal: true

require 'tmpdir'
require 'fileutils'
require 'nokogiri'
require_relative 'source'

class LocalEditorSourceTest
  POST = '_posts/2026-4Q/2026-10-24-editor-test.md'

  def setup
    @root = Dir.mktmpdir('blog-editor-source-')
    FileUtils.mkdir_p(File.join(@root, '_posts/2026-4Q'))
  end

  def teardown
    FileUtils.remove_entry(@root)
  end

  def write_post(body, yaml: "title: \"Original title\"\ndate: 2026-10-24 12:00:00 +0900\nimage:\n  alt: \"Cover caption\"\n", newline: "\n")
    @raw = "---\n#{yaml}---\n#{body}".gsub("\n", newline)
    File.binwrite(File.join(@root, POST), @raw)
    source
  end

  def source
    LocalEditor::Source.new(root: @root, path: POST)
  end

  def segment(document, text)
    document[:segments].find { |entry| entry[:text] == text } || flunk("Missing segment: #{text.inspect}")
  end

  def save_text(editor, original, replacement)
    document = editor.document
    item = segment(document, original)
    editor.save(revision: document[:revision], edits: [{ id: item[:id], text: replacement, original: original }])
  end

  def read_post
    File.binread(File.join(@root, POST)).force_encoding('UTF-8')
  end

  def body_of(value)
    value.split(/^---\s*$\n?/, 3).last
  end

  def html(editor)
    document = Kramdown::Document.new(body_of(read_post), input: 'GFM', hard_wrap: false)
    editor.annotate(document.root)
    Nokogiri::HTML.fragment(document.to_html)
  end

  def test_changes_a_korean_sentence_and_preserves_every_other_byte
    body = <<~'MARKDOWN'
      ## Keep heading

      한글 문장을 바로 고칩니다.

      An unchanged equation $$E=mc^2$$ stays intact.

      | Heading | Value |
      |---|---|
      | Unchanged | $x+y$ |

      ```python
      print("Keep all code bytes.")
      ```
    MARKDOWN
    editor = write_post(body)
    original = read_post
    result = save_text(editor, '한글 문장을 바로 고칩니다.', '한글 문장을 더 자연스럽게 고칩니다.')
    assert_equal original.sub('한글 문장을 바로 고칩니다.', '한글 문장을 더 자연스럽게 고칩니다\\.'), read_post
    assert_equal 1, result[:changed]
    assert_equal '한글 문장을 더 자연스럽게 고칩니다.', segment(result[:document], '한글 문장을 더 자연스럽게 고칩니다.')[:text]
    assert_equal original, File.read(Dir[File.join(@root, '.local-editor/backups/*.md')].first)
  end

  def test_tables_code_display_and_inline_math_have_no_plain_text_targets
    editor = write_post(<<~'MARKDOWN')
      Before $x+y$ after.

      $$
      a &= b \\
      c &= d
      $$

      | Protected table | Other column |
      |---|---|
      | Cell value | Other value |

      `protected_code()` and readable prose.

      ```text
      Protected fenced code
      ```
    MARKDOWN
    texts = editor.document[:segments].map { |item| item[:text] }
    assert_includes texts, 'Before '
    assert_includes texts, ' after.'
    refute texts.any? { |text| text.include?('x+y') || text.include?('Protected') || text.include?('Cell value') || text.include?('protected_code') }
    rendered = html(editor)
    assert_empty rendered.css('table [data-local-edit-id], code [data-local-edit-id]')
    assert_equal 'Before $x+y$ after.', rendered.at('p').text
    assert_equal 2, rendered.at('p').css('[data-local-edit-id]').length
    assert rendered.css('[data-local-edit-id]').all? { |item| item['class'].split.include?('tex2jax_ignore') }
  end

  def test_html_captions_keep_markup_attributes_links_and_math
    editor = write_post(<<~'MARKDOWN')
      <figure>
        <img src="figure.svg" alt="A protected attribute">
        <figcaption><strong>Figure 1.</strong> A visible caption with $rA$ and <a href="https://example.com/caption">a link</a>.</figcaption>
      </figure>

      <p class="table-caption"><span class="table-label">Table 1.</span> A table caption.</p>
    MARKDOWN
    document = editor.document
    assert_equal 'caption', segment(document, ' A visible caption with ')[:kind]
    assert_equal 'caption', segment(document, ' A table caption.')[:kind]
    original = read_post
    result = save_text(editor, ' A visible caption with ', ' A clear caption with ')
    assert_equal original.sub(' A visible caption with ', ' A clear caption with '), read_post
    assert_equal 1, result[:changed]
    rendered = html(editor)
    assert_equal 'A protected attribute', rendered.at('img')['alt']
    assert_equal 'https://example.com/caption', rendered.at('a')['href']
    assert_equal 'Figure 1. A clear caption with $rA$ and a link.', rendered.at('figcaption').text
  end

  def test_component_skip_marker_protects_repeated_literals_without_hiding_adjacent_prose
    protected = '<span data-local-editor-skip title="a > b"><span>Repeated.</span><span>$x$</span></span>'
    body = "#{protected}<span>Repeated.</span><span>$x$</span>\n"
    editor = write_post(body)
    original = read_post
    repeated = editor.document[:segments].select { |item| item[:text] == 'Repeated.' }
    assert_equal 1, repeated.length
    assert_equal ['x'], editor.document[:structures][:math].map { |item| item[:latex] }
    rendered = html(editor)
    assert_empty rendered.at('[data-local-editor-skip]').css('[data-local-edit-id], [data-local-math-id], [data-local-table-id], [data-local-cell-id]')
    assert_equal 1, rendered.css('[data-local-edit-id]').length
    assert_equal 1, rendered.css('[data-local-math-id]').length

    equation = editor.document[:structures][:math].first
    saved = editor.save(revision: editor.revision,
                        edits: [{ id: repeated.first[:id], text: 'Changed.' }, { id: equation[:id], text: 'y' }])
    assert_equal original.sub('<span>Repeated.</span><span>$x$</span>' + "\n", '<span>Changed\\.</span><span>$y$</span>' + "\n"), read_post
    assert_includes read_post, protected
    editor.undo(revision: saved[:document][:revision], receipt: saved[:receipt])
    assert_equal original, read_post
  end

  def test_liquid_expanded_component_cannot_claim_matching_article_text_or_math
    literal = "Repeated. $x$\n"
    body = "{{ site.component }}\n\n#{literal}"
    editor = write_post(body)
    input = "<section data-local-editor-skip>\n<p>Repeated. $x$</p>\n</section>\n\n#{literal}"
    parsed = Kramdown::Document.new(input, input: 'GFM', hard_wrap: false)
    editor.annotate(parsed.root, input: input)
    rendered = Nokogiri::HTML.fragment(parsed.to_html)
    assert_empty rendered.at('[data-local-editor-skip]').css('[data-local-edit-id], [data-local-math-id]')
    assert_equal ['Repeated. '], rendered.css('[data-local-edit-id]').map(&:text)
    assert_equal [editor.document[:structures][:math].first[:id]], rendered.css('[data-local-math-id]').map { |node| node['data-local-math-id'] }
  end

  def test_skip_marker_in_a_code_example_does_not_lock_following_prose
    editor = write_post("```html\n<section data-local-editor-skip>\n```\n\nVisible prose.\n")
    assert_equal ['Visible prose.'], editor.document[:segments].reject { |item| item[:field] }.map { |item| item[:text] }
    assert_equal ['Visible prose.'], html(editor).css('[data-local-edit-id]').map(&:text)
  end

  def test_html_caption_numeric_math_inequality_does_not_protect_surrounding_prose
    editor = write_post(<<~'MARKDOWN')
      <figcaption>Figure 4. Positive at $r>1.2$, and negative at $r<0.8$. The reward stays unchanged.</figcaption>
    MARKDOWN
    original = read_post
    texts = editor.document[:segments].map { |item| item[:text] }
    assert_includes texts, 'Figure 4. Positive at '
    assert_includes texts, ', and negative at '
    assert_includes texts, '. The reward stays unchanged.'
    refute texts.any? { |text| text.include?('r>1.2') || text.include?('r<0.8') }
    save_text(editor, '. The reward stays unchanged.', '. The reward itself stays unchanged.')
    assert_equal original.sub('. The reward stays unchanged.', '. The reward itself stays unchanged.'), read_post
    assert_includes html(editor).at('figcaption').text, '$r>1.2$, and negative at $r<0.8$'
  end

  def test_updates_only_yaml_scalar_token_and_preserves_comments_and_body
    editor = write_post("Body text.\n", yaml: <<~YAML)
      # Preserve this comment.
      title: 'Original title' # Keep the trailing comment.
      description: >-
        A folded
        description.
      date: 2026-10-24 12:00:00 +0900
      tags: [example, original]
      image:
        path: /assets/cover.webp
        alt: Cover caption
    YAML
    original = read_post
    result = save_text(editor, 'Original title', 'A "better" title: one')
    expected = original.sub("'Original title'", '"A \\"better\\" title: one"')
    assert_equal expected, read_post
    assert_equal 'A "better" title: one', result[:document][:fields]['title'][:text]
    result = save_text(editor, 'A folded description.', 'A short description.')
    assert_includes read_post, "description: \"A short description.\"\ndate:"
    assert_equal body_of(original), body_of(read_post)
    assert_equal 'A short description.', result[:document][:fields]['description'][:text]
  end

  def test_caption_and_title_both_accept_apostrophes_without_changing_yaml_structure
    editor = write_post("Plain text.\n")
    document = editor.document
    result = editor.save(revision: document[:revision], edits: [
                           { 'id' => 'field:title', 'text' => "Writer's title" },
                           { 'id' => 'field:image.alt', 'text' => "A figure's caption" }
                         ])
    assert_equal "Writer's title", result[:document][:fields]['title'][:text]
    assert_equal "A figure's caption", result[:document][:fields]['image.alt'][:text]
    assert_includes read_post, 'date: 2026-10-24 12:00:00 +0900'
  end

  def test_keeps_crlf_bytes_outside_the_changed_span
    editor = write_post("한글 본문입니다.\n\nKeep this line.\n", newline: "\r\n")
    original = read_post
    save_text(editor, '한글 본문입니다.', '새 문장입니다.')
    assert_equal original.sub('한글 본문입니다.', '새 문장입니다\\.'), read_post
    assert_equal original.count("\r"), read_post.count("\r")
  end

  def test_repeated_label_does_not_patch_link_destinations_or_html_attributes
    editor = write_post(<<~'MARKDOWN')
      [same](https://example.com/same) same after.

      <p title="Visible text">Visible text</p>
    MARKDOWN
    original = read_post
    save_text(editor, ' same after.', ' clear after.')
    assert_includes read_post, '[same](https://example.com/same) clear after\\.'
    save_text(editor, 'Visible text', 'Changed text')
    assert_includes read_post, '<p title="Visible text">Changed text</p>'
    assert_equal original.scan('https://example.com/same'), read_post.scan('https://example.com/same')
  end

  def test_pasted_markdown_and_html_are_literal_and_remain_editable
    editor = write_post("Ordinary sentence.\n")
    changed = 'A **literal** [label](path) <script> & 5 dollars: $5.'
    save_text(editor, 'Ordinary sentence.', changed)
    rendered = html(editor)
    assert_empty rendered.css('script, strong, a')
    assert_equal changed, rendered.text.strip
    # Escaped Markdown can be mapped again; an entity creates a separate safe
    # text leaf, so users can still edit either surrounding part in place.
    assert rendered.css('[data-local-edit-id]').any? { |item| item.text.include?('literal') }
  end

  def test_optimistic_concurrency_rejects_an_external_change
    editor = write_post("Sentence one.\n")
    revision = editor.document[:revision]
    item = segment(editor.document, 'Sentence one.')
    File.open(File.join(@root, POST), 'a') { |file| file.write("\nExternal edit.\n") }
    assert_raises(LocalEditor::Conflict) { editor.save(revision: revision, edits: [{ id: item[:id], text: 'Replacement.' }]) }
    assert_includes read_post, 'External edit.'
    assert_empty Dir[File.join(@root, '.local-editor/backups/*.md')]
  end

  def test_undo_restores_exact_original_bytes_and_refuses_stale_receipt
    editor = write_post("Sentence one.\n")
    original = read_post
    saved = save_text(editor, 'Sentence one.', 'Sentence two.')
    undone = editor.undo(revision: saved[:document][:revision], receipt: saved[:receipt])
    assert_equal original, read_post
    assert_equal Digest::SHA256.hexdigest(original), undone[:document][:revision]
    assert_raises(LocalEditor::Conflict) { editor.undo(revision: undone[:document][:revision], receipt: saved[:receipt]) }
  end

  def test_rejects_unknown_duplicate_protected_and_structural_edits
    editor = write_post("Plain sentence.\n")
    doc = editor.document
    item = segment(doc, 'Plain sentence.')
    [
      [{ id: 'unknown', text: 'No' }],
      [{ id: item[:id], text: 'First' }, { id: item[:id], text: 'Second' }],
      [{ id: 'field:title', text: "New\ntitle" }],
      [{ id: 'field:title', text: '<script>alert(1)</script>' }],
      [{ id: item[:id], text: '$x+y$' }],
      [{ id: item[:id], text: '{% include secret %}' }]
    ].each do |edits|
      assert_raises(LocalEditor::InvalidEdit) { editor.save(revision: doc[:revision], edits: edits) }
    end
    assert_equal @raw, read_post
  end

  def test_post_liquid_ast_wraps_only_known_exact_source_ranges
    editor = write_post("Before {{ site.title }} after.\n\nA stable paragraph.\n")
    expanded = "Before A completely different site title after.\n\nA stable paragraph.\n"
    document = Kramdown::Document.new(expanded, input: 'GFM')
    editor.annotate(document.root)
    rendered = Nokogiri::HTML.fragment(document.to_html)
    assert_empty rendered.css('p').first.css('[data-local-edit-id]')
    assert_equal 'A stable paragraph.', rendered.css('[data-local-edit-id]').last.text
    assert rendered.css('[data-local-edit-id]').all? { |span| editor.document[:segments].any? { |item| item[:id] == span['data-local-edit-id'] } }
  end

  def test_liquid_generated_duplicate_does_not_reuse_a_real_source_target
    editor = write_post("{{ site.snippet }} **word**\n\nA stable paragraph.\n")
    original = read_post
    input = "**word** **word**\n\nA stable paragraph.\n"
    document = Kramdown::Document.new(input, input: 'GFM', hard_wrap: false)
    editor.annotate(document.root, input: input)
    rendered = Nokogiri::HTML.fragment(document.to_html)
    assert_empty rendered.css('p').first.css('[data-local-edit-id]')
    assert_equal ['A stable paragraph.'], rendered.css('[data-local-edit-id]').map(&:text)
    assert_equal original, read_post
  end

  def test_multiline_liquid_expansion_protects_shifted_duplicate_source_locations
    editor = write_post("A trusted prefix.\n\n{{ site.snippet }}\n\n**word**\n\nA stable ending.\n")
    original = read_post
    input = "A trusted prefix.\n\n**word**\n\n**word**\n\n**word**\n\nA stable ending.\n"
    document = Kramdown::Document.new(input, input: 'GFM', hard_wrap: false)
    editor.annotate(document.root, input: input)
    rendered = Nokogiri::HTML.fragment(document.to_html)
    assert_equal ['A trusted prefix.', 'word', 'A stable ending.'], rendered.css('[data-local-edit-id]').map(&:text)
    assert_empty rendered.css('p')[1].css('[data-local-edit-id]')
    assert_empty rendered.css('p')[2].css('[data-local-edit-id]')
    assert_equal original, read_post
  end

  def test_liquid_control_blocks_protect_shifted_text_even_with_equal_total_line_counts
    editor = write_post("A trusted prefix.\n\n{{ site.snippet }}\n**word**\n{% comment %}\none hidden line\n{% endcomment %}\n\nA stable paragraph.\n")
    original = read_post
    # A three-line output followed by a removed comment block compensates for
    # the extra newlines. Input line 4 contains generated text, while the real
    # source occurrence has moved to input line 6.
    input = "A trusted prefix.\n\n**word**\n**word**\n**word**\n**word**\n\n\nA stable paragraph.\n"
    assert_equal body_of(original).lines.length, input.lines.length
    document = Kramdown::Document.new(input, input: 'GFM', hard_wrap: false)
    editor.annotate(document.root, input: input)
    rendered = Nokogiri::HTML.fragment(document.to_html)
    assert_equal ['A trusted prefix.', 'A stable paragraph.'], rendered.css('[data-local-edit-id]').map(&:text)
    assert_equal original, read_post
  end

  def test_simple_liquid_link_expansion_keeps_following_prose_editable
    editor = write_post("[Read the translation]({% post_url 2026-10-24-test %})\n\nA stable paragraph.\n")
    input = "[Read the translation](/posts/test/)\n\nA stable paragraph.\n"
    document = Kramdown::Document.new(input, input: 'GFM', hard_wrap: false)
    editor.annotate(document.root, input: input)
    rendered = Nokogiri::HTML.fragment(document.to_html)
    assert_empty rendered.css('p').first.css('[data-local-edit-id]')
    assert_equal ['A stable paragraph.'], rendered.css('[data-local-edit-id]').map(&:text)
  end

  def test_known_url_expansions_before_a_later_control_block_keep_earlier_prose_editable
    editor = write_post("[Translation]({{ site.baseurl }}/posts/translation/)\n\nAn editable early paragraph.\n\n{% if site.show_extra %}\nConditional text.\n{% endif %}\n\nA protected ending.\n")
    input = "[Translation](/posts/translation/)\n\nAn editable early paragraph.\n\n\nConditional text.\n\n\nA protected ending.\n"
    assert_equal body_of(read_post).lines.length, input.lines.length
    document = Kramdown::Document.new(input, input: 'GFM', hard_wrap: false)
    editor.annotate(document.root, input: input)
    rendered = Nokogiri::HTML.fragment(document.to_html)
    assert_equal ['An editable early paragraph.', 'A protected ending.'], rendered.css('[data-local-edit-id]').map(&:text)
  end

  def test_url_expansions_cross_style_prefix_and_suffix_without_mapping_a_generated_lab
    body = <<~'MARKDOWN'
      <style>
      .example { color: blue; }
      </style>

      [Translation]({{ site.baseurl }}/posts/translation/) · [Home]({{ site.url }}/)

      An editable early paragraph.

      {% include interactive/article-lab.html id="example" %}

      An editable trailing paragraph.

      [Source]({{ site.url }}{{ site.baseurl }}/source/)

      A stable ending.
    MARKDOWN
    editor = write_post(body)
    original = read_post
    component = "<section data-local-editor-skip>\n<p>An editable early paragraph.</p>\n<p>An editable trailing paragraph. $x$</p>\n</section>"
    input = body.sub('{{ site.baseurl }}', '/blog').gsub('{{ site.url }}', 'https://example.com')
                .gsub('{{ site.baseurl }}', '/blog')
                .sub('{% include interactive/article-lab.html id="example" %}', component)
    refute body.lines.length == input.lines.length
    document = Kramdown::Document.new(input, input: 'GFM', hard_wrap: false)
    unannotated = Nokogiri::HTML.fragment(document.to_html)
    editor.annotate(document.root, input: input)
    rendered = Nokogiri::HTML.fragment(document.to_html)
    assert_equal ['An editable early paragraph.', 'An editable trailing paragraph.', 'A stable ending.'],
                 rendered.css('[data-local-edit-id]').map(&:text)
    assert_empty rendered.at('[data-local-editor-skip]').css('[data-local-edit-id], [data-local-math-id], [data-local-table-id], [data-local-cell-id]')
    assert_empty rendered.css('a [data-local-edit-id]')
    rendered.css('[data-local-edit-id]').each { |node| node.replace(node.children) }
    assert_equal unannotated.to_html, rendered.to_html
    assert_equal original, read_post
  end

  def test_url_anchored_prefix_and_suffix_map_only_the_real_repeated_occurrences
    body = "[Before]({{ site.baseurl }}/before/)\n\n**same**\n\n{% include example %}\n\n**same**\n\n[After]({{ site.url }}/after/)\n"
    editor = write_post(body)
    original = read_post
    input = "[Before](/before/)\n\n**same**\n\n**same**\n\n<section data-local-editor-skip>\n<p>same</p>\n</section>\n\n**same**\n\n[After](https://example.com/after/)\n"
    document = Kramdown::Document.new(input, input: 'GFM', hard_wrap: false)
    editor.annotate(document.root, input: input)
    rendered = Nokogiri::HTML.fragment(document.to_html)
    expected = editor.document[:segments].select { |item| item[:text] == 'same' }.map { |item| item[:id] }
    assert_equal 2, expected.length
    assert_equal expected, rendered.css('[data-local-edit-id]').map { |node| node['data-local-edit-id'] }
    assert_empty rendered.css('p')[2].css('[data-local-edit-id]')
    assert_empty rendered.at('[data-local-editor-skip]').css('[data-local-edit-id]')
    assert_equal original, read_post
  end

  def interactive_component(id, key: id, preset: '')
    <<~HTML
      <section id="lab-#{key}" class="article-lab" data-article-lab="#{id}" data-preset="#{preset}" data-local-editor-skip>
      <p>Repeated. $x+1$</p>
      </section>
    HTML
  end

  def test_multiple_interactive_includes_keep_literal_islands_equations_and_tables_editable
    body = <<~'MARKDOWN'

      [Translation]({{ site.baseurl }}/posts/translation/)

      Repeated.

      {% include interactive/article-lab.html id="first" %}

      Repeated.

      한글 문장 $x+1$ 뒤의 설명입니다.
      다음 줄도 같은 문단입니다.

      $$
      y = \frac{a}{b}
      $$

      | Symbol | Value |
      |:---|---:|
      | $x$ | **2** |

      {% include interactive/negf-barrier.html %}

      두 번째 중간 문단입니다.

      {% include interactive/article-lab.html id='second' key="alternate" preset="sample" %}

      Repeated.
    MARKDOWN
    editor = write_post(body)
    original = read_post
    input = body.sub("\n", '').sub('{{ site.baseurl }}', '/blog')
                .sub('{% include interactive/article-lab.html id="first" %}', "\n" + interactive_component('first'))
                .sub('{% include interactive/negf-barrier.html %}', "\n<section class=\"negf-lab\" data-negf-lab data-local-editor-skip>\n<p>두 번째 중간 문단입니다. $x+1$</p>\n</section>\n")
                .sub("{% include interactive/article-lab.html id='second' key=\"alternate\" preset=\"sample\" %}", interactive_component('second', key: 'alternate', preset: 'sample'))
    document = Kramdown::Document.new(input, input: 'GFM', hard_wrap: false)
    unannotated = Nokogiri::HTML.fragment(document.to_html)
    editor.annotate(document.root, input: input)
    rendered = Nokogiri::HTML.fragment(document.to_html)
    ids = rendered.css('[data-local-edit-id]').map { |node| node['data-local-edit-id'] }
    repeated = editor.document[:segments].select { |item| item[:text] == 'Repeated.' }
    assert_equal 3, repeated.length
    assert_equal repeated.map { |item| item[:id] }, rendered.css('[data-local-edit-id]').select { |node| node.text == 'Repeated.' }.map { |node| node['data-local-edit-id'] }
    ['한글 문장 ', ' 뒤의 설명입니다.', '다음 줄도 같은 문단입니다.', '두 번째 중간 문단입니다.'].each do |text|
      assert ids.include?(segment(editor.document, text)[:id]), "Missing middle-island text: #{text}"
    end
    assert_equal editor.document[:structures][:math].map { |item| item[:id] }, rendered.css('[data-local-math-id]').map { |node| node['data-local-math-id'] }
    assert_equal editor.document[:structures][:tables].map { |item| item[:id] }, rendered.css('[data-local-table-id]').map { |node| node['data-local-table-id'] }
    assert_empty rendered.css('[data-local-editor-skip] [data-local-edit-id], [data-local-editor-skip] [data-local-math-id], [data-local-editor-skip] [data-local-table-id], [data-local-editor-skip] [data-local-cell-id]')
    rendered.css('[data-local-edit-id], [data-local-math-id]').each { |node| node.replace(node.children) }
    rendered.css('[data-local-table-id], [data-local-cell-id]').each do |node|
      node.remove_attribute('data-local-table-id')
      node.remove_attribute('data-local-cell-id')
    end
    assert_equal unannotated.to_html.gsub(/\n{2,}/, "\n\n"), rendered.to_html.gsub(/\n{2,}/, "\n\n")
    editor.save(revision: editor.revision, edits: [{ id: repeated[1][:id], text: 'The middle occurrence.', original: 'Repeated.' }])
    expected = original.sub("%}\n\nRepeated.", "%}\n\nThe middle occurrence\\.")
    assert_equal expected, read_post
  end

  def test_interactive_islands_refuse_fake_missing_nested_or_mismatched_components
    body = "Early prose.\n\n{% include interactive/article-lab.html id=\"first\" %}\n\nMiddle prose.\n\n{% include interactive/article-lab.html id=\"second\" %}\n\nLate prose.\n"
    first = interactive_component('first')
    second = interactive_component('second')
    variants = [
      [body, first.sub('data-article-lab="first"', 'data-article-lab="fake"'), second],
      [body, first, ''],
      [body, first.sub('</section>', '<section data-local-editor-skip>Nested</section></section>'), second],
      [body, first + interactive_component('extra'), second],
      [body.sub('id="first"', 'id=page.component'), first, second],
      [body.sub('{% include interactive/article-lab.html id="first" %}', '{% if site.example %}' + "\n" + '{% include interactive/article-lab.html id="first" %}' + "\n" + '{% endif %}'), first, second],
      [body.sub('{% include interactive/article-lab.html id="first" %}', '{% include interactive/article-lab.html id="first" %} inline'), first, second],
      [body, first.sub('id="lab-first"', 'id="lab-wrong"'), second]
    ]
    variants.each do |source_body, rendered_first, rendered_second|
      editor = write_post(source_body)
      input = "Early prose.\n\n#{rendered_first}\n\nMiddle prose.\n\n#{rendered_second}\n\nLate prose.\n"
      document = Kramdown::Document.new(input, input: 'GFM', hard_wrap: false)
      editor.annotate(document.root, input: input)
      rendered = Nokogiri::HTML.fragment(document.to_html)
      refute rendered.css('[data-local-edit-id]').map(&:text).include?('Middle prose.')
      assert_empty rendered.css('[data-local-editor-skip] [data-local-edit-id]')
    end
  end

  def test_interactive_islands_require_the_entire_intervening_literal_span_to_match
    body = "Early prose.\n\n{% include interactive/article-lab.html id=\"first\" %}\n\nRepeated.\n\nMiddle prose.\n\n{% include interactive/article-lab.html id=\"second\" %}\n\nLate prose.\n"
    editor = write_post(body)
    input = body.sub('{% include interactive/article-lab.html id="first" %}', interactive_component('first'))
                .sub('{% include interactive/article-lab.html id="second" %}', interactive_component('second'))
                .sub("\n\nRepeated.\n", "\n\nGenerated extra text.\n\nRepeated.\n")
    document = Kramdown::Document.new(input, input: 'GFM', hard_wrap: false)
    editor.annotate(document.root, input: input)
    rendered = Nokogiri::HTML.fragment(document.to_html)
    refute rendered.css('[data-local-edit-id]').map(&:text).include?('Middle prose.')
  end

  def test_closed_template_intro_and_outro_leave_interactive_literal_islands_editable
    body = <<~'MARKDOWN'
      Early prose.

      {% assign later = site.posts | first %}
      > Update.{% if later %} [Later]({{ later.url }}){% endif %}

      The literal introduction after the update.

      {% include interactive/article-lab.html id="first" %}

      Middle prose. $x+1$

      {% include interactive/article-lab.html id="second" %}

      The literal ending before the footer.

      {% if site.footer %}
      Footer.
      {% endif %}

      Late prose.
    MARKDOWN
    editor = write_post(body)
    input = body.sub('{% assign later = site.posts | first %}', '')
                .sub('> Update.{% if later %} [Later]({{ later.url }}){% endif %}', '> Update. [Later](/later/)')
                .sub('{% include interactive/article-lab.html id="first" %}', interactive_component('first'))
                .sub('{% include interactive/article-lab.html id="second" %}', interactive_component('second'))
                .sub('{% if site.footer %}', '').sub('{% endif %}', '')
    document = Kramdown::Document.new(input, input: 'GFM', hard_wrap: false)
    editor.annotate(document.root, input: input)
    rendered = Nokogiri::HTML.fragment(document.to_html)
    texts = rendered.css('[data-local-edit-id]').map(&:text)
    ['Early prose.', 'The literal introduction after the update.', 'Middle prose. ', 'The literal ending before the footer.', 'Late prose.'].each do |text|
      assert texts.include?(text), "Missing template-adjacent literal: #{text}"
    end
    refute texts.include?('Footer.')
    refute texts.include?('Update.')
    assert_equal editor.document[:structures][:math].map { |item| item[:id] }, rendered.css('[data-local-math-id]').map { |node| node['data-local-math-id'] }
    assert_empty rendered.css('[data-local-editor-skip] [data-local-edit-id]')
  end

  def test_template_control_blocks_crossing_interactive_components_stay_locked
    wrappers = [
      ['{% if site.enabled %}', '{% endif %}'],
      ['{% for item in site.items %}', '{% endfor %}'],
      ['{% capture output %}', '{% endcapture %}']
    ]
    wrappers.each do |opening, closing|
      body = "Early prose.\n\n#{opening}\n\n{% include interactive/article-lab.html id=\"first\" %}\n\nMiddle prose.\n\n{% include interactive/article-lab.html id=\"second\" %}\n\n#{closing}\n\nLate prose.\n"
      editor = write_post(body)
      input = body.sub(opening, '').sub(closing, '')
                  .sub('{% include interactive/article-lab.html id="first" %}', interactive_component('first'))
                  .sub('{% include interactive/article-lab.html id="second" %}', interactive_component('second'))
      document = Kramdown::Document.new(input, input: 'GFM', hard_wrap: false)
      editor.annotate(document.root, input: input)
      rendered = Nokogiri::HTML.fragment(document.to_html)
      refute rendered.css('[data-local-edit-id]').map(&:text).include?('Middle prose.')
      assert_empty rendered.css('[data-local-editor-skip] [data-local-edit-id]')
    end
  end

  def test_single_line_url_tags_preserve_adjacent_prefix_and_suffix_targets
    body = "<style>p { color: blue; }</style>\n\n[Read]({% post_url 2026-10-24-test %})\n[Download]({% link assets/example.csv %})\n\nEarly prose.\n\n{% include example %}\n\nLate prose.\n\n[Again]({% link assets/example.csv %})\n"
    editor = write_post(body)
    original = read_post
    input = body.gsub('{% post_url 2026-10-24-test %}', '/posts/test/')
                .gsub('{% link assets/example.csv %}', '/assets/example.csv')
                .sub('{% include example %}', "<section data-local-editor-skip>\n<p>Early prose.</p>\n</section>")
    document = Kramdown::Document.new(input, input: 'GFM', hard_wrap: false)
    editor.annotate(document.root, input: input)
    rendered = Nokogiri::HTML.fragment(document.to_html)
    assert_equal ['Early prose.', 'Late prose.'], rendered.css('[data-local-edit-id]').map(&:text)
    assert_empty rendered.css('a [data-local-edit-id]')
    assert_equal original, read_post
  end

  def test_url_prefix_matching_rejects_unknown_tokens_trim_controls_and_changed_literals
    examples = [
      ["[Read]({{ site.baseurl }}/read/)", '[Other](/read/)'],
      ["[Read]({{ site.title }}/read/)", '[Read](/read/)'],
      ["[Read]({{ site.baseurl | append: '/extra' }}/read/)", '[Read](/extra/read/)'],
      ["[Read]({{- site.baseurl -}}/read/)", '[Read](/read/)'],
      ["[Read]({{ site.baseurl }}/read/)", "[Read](/blog\n/read/)"]
    ]
    examples.each do |source_url, expanded_url|
      body = "<style>p { color: blue; }</style>\n\n#{source_url}\n\nDo not map this prefix.\n\n{% include example %}\n\nA stable ending.\n"
      editor = write_post(body)
      original = read_post
      input = body.sub(source_url, expanded_url).sub('{% include example %}', "<section data-local-editor-skip>\n<p>Do not map this prefix.</p>\n</section>")
      document = Kramdown::Document.new(input, input: 'GFM', hard_wrap: false)
      editor.annotate(document.root, input: input)
      rendered = Nokogiri::HTML.fragment(document.to_html)
      assert_equal ['A stable ending.'], rendered.css('[data-local-edit-id]').map(&:text)
      assert_equal original, read_post
    end
  end

  def test_initial_liquid_control_block_keeps_the_exact_repeated_terminal_tail_editable
    editor = write_post("{% assign enabled = true %}\n{% if enabled %}\nGenerated note.\n{% endif %}\n\nA stable introduction.\n\n**same**\n\n**same**\n\nA stable ending.\n")
    original = read_post
    input = "\n\nGenerated note.\n\n\nA stable introduction.\n\n**same**\n\n**same**\n\nA stable ending.\n"
    document = Kramdown::Document.new(input, input: 'GFM', hard_wrap: false)
    editor.annotate(document.root, input: input)
    rendered = Nokogiri::HTML.fragment(document.to_html)
    assert_empty rendered.css('p').first.css('[data-local-edit-id]')
    assert_equal ['A stable introduction.', 'same', 'same', 'A stable ending.'], rendered.css('[data-local-edit-id]').map(&:text)
    expected_ids = editor.document[:segments].select { |item| item[:text] == 'same' }.map { |item| item[:id] }
    assert_equal expected_ids, rendered.css('[data-local-edit-id]').select { |item| item.text == 'same' }.map { |item| item['data-local-edit-id'] }
    assert_equal original, read_post
  end

  def test_multiline_output_maps_only_real_repeated_suffix_targets_and_saves_the_selected_occurrence
    editor = write_post("{{ site.snippet }}\n\n**same**\n\n**same**\n\nA stable ending.\n")
    original = read_post
    input = "**same**\n\n**same**\n\n**same**\n\n**same**\n\nA stable ending.\n"
    document = Kramdown::Document.new(input, input: 'GFM', hard_wrap: false)
    editor.annotate(document.root, input: input)
    rendered = Nokogiri::HTML.fragment(document.to_html)
    assert_empty rendered.css('p')[0].css('[data-local-edit-id]')
    assert_empty rendered.css('p')[1].css('[data-local-edit-id]')
    repeated = rendered.css('[data-local-edit-id]').select { |node| node.text == 'same' }
    expected_ids = editor.document[:segments].select { |item| item[:text] == 'same' }.map { |item| item[:id] }
    assert_equal expected_ids, repeated.map { |node| node['data-local-edit-id'] }
    saved = editor.save(revision: editor.revision, edits: [{ id: repeated.last['data-local-edit-id'], text: 'changed' }])
    assert_equal original.sub("**same**\n\nA stable ending.", "**changed**\n\nA stable ending."), read_post
    editor.undo(revision: saved[:document][:revision], receipt: saved[:receipt])
    assert_equal original, read_post
  end

  def test_anchored_metadata_is_protected_without_breaking_aliases
    editor = write_post("A stable paragraph.\n", yaml: <<~YAML)
      title: &title Original title
      description: *title
      date: 2026-10-24
      image:
        alt: &caption Cover caption
      retained_caption: *caption
    YAML
    original = read_post
    assert_empty editor.document[:fields]
    assert_raises(LocalEditor::InvalidEdit) do
      editor.save(revision: editor.revision, edits: [{ id: 'field:title', text: 'New title' }])
    end
    assert_equal original, read_post
    save_text(editor, 'A stable paragraph.', 'A revised paragraph.')
    metadata = Psych.unsafe_load(read_post.split(/^---\s*$\n?/, 3)[1])
    assert_equal 'Original title', metadata['description']
    assert_equal 'Cover caption', metadata['retained_caption']
    assert_equal original.sub('A stable paragraph.', 'A revised paragraph\\.'), read_post
  end

  def test_unicode_line_separators_round_trip_in_yaml_metadata
    editor = write_post("A stable paragraph.\n")
    original = read_post
    changed = "한글\u0085title\u2028line\u2029ending"
    result = save_text(editor, 'Original title', changed)
    assert_equal changed, result[:document][:fields]['title'][:text]
    assert_equal changed, Psych.unsafe_load(read_post.split(/^---\s*$\n?/, 3)[1])['title']
    assert_equal body_of(original), body_of(read_post)
    editor.undo(revision: result[:document][:revision], receipt: result[:receipt])
    assert_equal original, read_post
  end

  def test_rejects_path_traversal_and_symlinks
    write_post("Text.\n")
    File.symlink(File.join(@root, POST), File.join(@root, '_posts/link.md'))
    [POST.sub('_posts/', '../_posts/'), File.join(@root, POST), '_posts/link.md'].each do |path|
      assert_raises(LocalEditor::InvalidPath) { LocalEditor::Source.new(root: @root, path: path) }
    end
  end

  def test_identical_labels_in_protected_constructs_are_never_patched
    fixtures = [
      ['`word` **word**', 'word'],
      ['![imageword](x) **imageword**', 'imageword'],
      ['$value$ **value**', 'value'],
      ['`$value$` **value**', 'value'],
      ['<span title="word"></span> **word**', 'word'],
      ['<table><tr><td>word</td></tr></table> **word**', 'word'],
      ['<p><code>word</code><strong>word</strong></p>', 'word'],
      ['<!-- word --> **word**', 'word'],
      ['[label](word) **word**', 'word'],
      ['<p><script>word</script><strong>word</strong></p>', 'word'],
      ['`한글` **한글**', '한글'],
      ["[^word] **word**\n\n[^word]: A footnote.", 'word']
    ]
    fixtures.each do |body, target|
      editor = write_post("#{body}\n")
      original = read_post
      save_text(editor, target, 'EDITED')
      expected = if body.include?("**#{target}**")
                   original.sub("**#{target}**", '**EDITED**')
                 else
                   original.sub("<strong>#{target}</strong>", '<strong>EDITED</strong>')
                 end
      assert_equal expected, read_post
    end
  end

  def test_wrapped_paragraph_lines_and_entity_captions_render_identically
    editor = write_post(<<~'MARKDOWN')
      The first part of a paragraph,
      followed by a second line with **bold text**.

      <figcaption>One &amp; two &lt; three, followed by $x$ and prose.</figcaption>
    MARKDOWN
    doc = editor.document
    refute doc[:segments].any? { |item| item[:kind] != 'metadata' && item[:text].include?("\n") }
    original = Nokogiri::HTML.fragment(Kramdown::Document.new(body_of(read_post), input: 'GFM', hard_wrap: false).to_html)
    annotated = html(editor)
    annotated.css('[data-local-edit-id], [data-local-math-id]').each { |item| item.replace(item.children) }
    assert_equal original.to_html, annotated.to_html
    save_text(editor, 'The first part of a paragraph,', 'The revised part of a paragraph,')
    assert_includes read_post, "The revised part of a paragraph,\nfollowed by a second line"
  end

  def test_jekyll_leading_blank_line_trim_keeps_body_targets_aligned
    editor = write_post("\n\n## A heading\n\nThe visible paragraph.\n")
    input = body_of(read_post).lstrip
    document = Kramdown::Document.new(input, input: 'GFM', hard_wrap: false)
    editor.annotate(document.root, input: input)
    annotated = Nokogiri::HTML.fragment(document.to_html)
    assert_equal ['A heading', 'The visible paragraph.'], annotated.css('[data-local-edit-id]').map(&:text)
    save_text(editor, 'The visible paragraph.', 'The revised paragraph.')
    assert_includes read_post, "---\n\n\n## A heading\n"
  end

  def test_ambiguous_nested_image_label_fails_closed
    editor = write_post("![nested [label] word](image.png) **word**\n")
    item = editor.document[:segments].find { |entry| entry[:text] == 'word' }
    if item
      original = read_post
      save_text(editor, 'word', 'EDITED')
      assert_equal original.sub('**word**', '**EDITED**'), read_post
    else
      assert_equal @raw, read_post
    end
  end

  def test_metadata_snapshot_has_typed_values_and_separate_writability
    editor = write_post("A stable paragraph.\n", yaml: <<~YAML)
      title: Original title
      tags: [one, two]
      categories: [AI, Kaggle]
      topic: example
      date: 2026-10-24 12:00:00 +0900
      lang: en
    YAML
    snapshot = editor.document[:metadata]
    assert_equal POST, editor.path
    assert_equal ['one', 'two'], editor.front_matter['tags']
    assert_equal ['one', 'two'], editor.metadata_values['tags']
    assert_equal '2026-10-24 12:00:00 +0900', snapshot[:values]['date']
    assert_equal nil, snapshot[:values]['series']
    assert_equal nil, snapshot[:values]['series_order']
    assert_equal LocalEditor::Source::METADATA_FIELDS, snapshot[:writable].keys
    assert snapshot[:writable].values.all?
    refute editor.document[:segments].any? { |item| item[:field] == 'tags' }
  end

  def test_metadata_flow_lists_change_only_their_value_tokens
    editor = write_post("Body with $x+y$ and code.\n", yaml: <<~YAML)
      # Keep this header.
      title: Original title
      tags: [one, "two # words"] # Keep tags comment.
      categories: [AI, Kaggle]
      topic: example
      date: 2026-10-24 12:00:00 +0900
    YAML
    original = read_post
    result = editor.save(revision: editor.revision,
                         metadata: { tags: ['three', 'four'], categories: ['Physics'] })
    expected = original.sub('[one, "two # words"]', '["three","four"]').sub('[AI, Kaggle]', '["Physics"]')
    assert_equal expected, read_post
    assert_equal 2, result[:changed]
    assert_equal ['three', 'four'], result[:document][:metadata][:values]['tags']
    editor.undo(revision: result[:document][:revision], receipt: result[:receipt])
    assert_equal original, read_post
  end

  def test_metadata_block_lists_preserve_item_and_neighbor_comments
    editor = write_post("The original body.\n", yaml: <<~YAML)
      title: Original title
      tags:
        - old # Keep item comment.
        - "two # literal"
      # Keep adjacent comment.
      topic: original # Keep topic comment.
      date: 2026-10-24 12:00:00 +0900
    YAML
    original = read_post
    result = editor.save(revision: editor.revision, metadata: { 'tags' => ['new', 'tag'], 'topic' => 'revised' })
    assert_equal ['new', 'tag'], editor.front_matter['tags']
    assert_equal 'revised', editor.front_matter['topic']
    assert_includes read_post, '  # Keep item comment.'
    assert_includes read_post, '# Keep adjacent comment.'
    assert_includes read_post, 'topic: "revised" # Keep topic comment.'
    assert_equal body_of(original), body_of(read_post)
    editor.undo(revision: result[:document][:revision], receipt: result[:receipt])
    assert_equal original, read_post
  end

  def test_metadata_insertions_preserve_unicode_crlf_body_and_existing_lines
    editor = write_post("한글 본문입니다.\n\n$$x+y$$\n", yaml: "title: '원래 제목' # 제목 주석\ndate: 2026-10-24 12:00:00 +0900\n", newline: "\r\n")
    original = read_post
    editor.save(revision: editor.revision,
                metadata: { tags: ['example', 'new-tag'], categories: ['AI', 'Kaggle'], topic: 'example',
                            series: 'example-series', series_order: 0, lang: 'ko' })
    assert_equal body_of(original), body_of(read_post)
    assert_includes read_post, "title: '원래 제목' # 제목 주석\r\ndate: 2026-10-24 12:00:00 +0900\r\n"
    assert_includes read_post, "series_order: 0\r\nlang: \"ko\"\r\n---\r\n"
    refute read_post.match?(/(?<!\r)\n/)
    assert_equal 'example-series', editor.front_matter['series']
    assert_equal 0, editor.front_matter['series_order']
  end

  def test_metadata_can_remove_series_and_order_without_losing_comments
    editor = write_post("Body.\n", yaml: <<~YAML)
      title: Original title
      series: original # Keep series comment.
      series_order: 2
      date: 2026-10-24 12:00:00 +0900
    YAML
    original = read_post
    saved = editor.save(revision: editor.revision, metadata: { 'series' => nil, 'series_order' => nil })
    assert_equal original.sub('series: original', '').sub('series_order: 2', ''), read_post
    refute editor.front_matter.key?('series')
    refute editor.front_matter.key?('series_order')
    editor.undo(revision: saved[:document][:revision], receipt: saved[:receipt])
    assert_equal original, read_post
  end

  def test_metadata_folded_scalar_changes_without_consuming_the_next_key
    editor = write_post("Body.\n", yaml: <<~YAML)
      title: Original title
      topic: >-
        old-topic
      # A neighboring comment.
      date: 2026-10-24 12:00:00 +0900
    YAML
    original = read_post
    saved = editor.save(revision: editor.revision, metadata: { 'topic' => 'new-topic' })
    assert_includes read_post, "topic: \"new-topic\"\n# A neighboring comment.\ndate:"
    assert_equal 'new-topic', editor.front_matter['topic']
    assert_equal body_of(original), body_of(read_post)
    editor.undo(revision: saved[:document][:revision], receipt: saved[:receipt])
    assert_equal original, read_post
  end

  def test_metadata_folded_scalar_header_comment_is_retained
    editor = write_post("Body.\n", yaml: <<~YAML)
      title: Original title
      topic: >- # Keep header comment.
        original-topic
      date: 2026-10-24 12:00:00 +0900
    YAML
    original = read_post
    saved = editor.save(revision: editor.revision, metadata: { topic: 'new-topic' })
    assert_equal 'new-topic', editor.front_matter['topic']
    assert_includes read_post, '# Keep header comment.'
    assert_includes read_post, 'date: 2026-10-24 12:00:00 +0900'
    assert_equal body_of(original), body_of(read_post)
    editor.undo(revision: saved[:document][:revision], receipt: saved[:receipt])
    assert_equal original, read_post
  end

  def test_metadata_null_scalar_can_be_replaced_after_an_unspaced_colon
    editor = write_post("Body.\n", yaml: "title: Original title\ntags:\nlang:\ndate: 2026-10-24 12:00:00 +0900\n")
    editor.save(revision: editor.revision, metadata: { tags: ['new-tag'], lang: 'en' })
    assert_equal ['new-tag'], editor.front_matter['tags']
    assert_equal 'en', editor.front_matter['lang']
    assert_includes read_post, "tags: [\"new-tag\"]\nlang: \"en\"\n"
  end

  def test_metadata_last_empty_scalar_and_new_fields_can_change_together
    editor = write_post("Body.\n", yaml: "title: Original title\ndate: 2026-10-24 12:00:00 +0900\nlang:\n")
    editor.save(revision: editor.revision, metadata: { lang: 'en', tags: ['new-tag'], topic: 'example' })
    assert_equal 'en', editor.front_matter['lang']
    assert_equal ['new-tag'], editor.front_matter['tags']
    assert_equal 'example', editor.front_matter['topic']
    assert_includes read_post, "lang: \"en\"\ntags: [\"new-tag\"]\ntopic: \"example\"\n---\n"
  end

  def test_metadata_indentless_block_sequence_gets_a_valid_flow_indentation
    editor = write_post("Body.\n", yaml: <<~YAML)
      title: Original title
      categories:
      - AI
      - Kaggle
      tags:
      - example # Keep comment.
      topic: example
      date: 2026-10-24 12:00:00 +0900
    YAML
    original = read_post
    saved = editor.save(revision: editor.revision, metadata: { categories: ['Research'], tags: ['new-tag'] })
    assert_equal ['Research'], editor.front_matter['categories']
    assert_equal ['new-tag'], editor.front_matter['tags']
    assert_includes read_post, "categories:\n  [\"Research\"]\ntags:\n  [\"new-tag\"]\n# Keep comment.\n"
    assert_equal body_of(original), body_of(read_post)
    editor.undo(revision: saved[:document][:revision], receipt: saved[:receipt])
    assert_equal original, read_post
  end

  def test_metadata_and_inline_edits_share_one_exact_backup_and_undo
    editor = write_post("A stable paragraph.\n", yaml: "title: Original title\ntopic: example\ndate: 2026-10-24 12:00:00 +0900\n")
    original = read_post
    body_item = segment(editor.document, 'A stable paragraph.')
    saved = editor.save(revision: editor.revision,
                        edits: [{ id: body_item[:id], text: 'A revised paragraph.' }, { id: 'field:title', text: 'A revised title' }],
                        metadata: { topic: 'revised', tags: ['revised'], lang: 'en' })
    assert_equal 5, saved[:changed]
    assert_equal 'A revised title', editor.front_matter['title']
    assert_equal 'revised', editor.front_matter['topic']
    assert_includes read_post, 'A revised paragraph\\.'
    assert_equal 1, Dir[File.join(@root, '.local-editor/backups/*.md')].length
    editor.undo(revision: saved[:document][:revision], receipt: saved[:receipt])
    assert_equal original, read_post
  end

  def test_metadata_rejects_anchors_aliases_duplicates_and_merges
    fixtures = [
      ["tags: &keywords [one, two]\nretained: *keywords\n", 'tags'],
      ["retained: &keywords [one, two]\ntags: *keywords\n", 'tags'],
      ["tags: [one, &word two]\nretained: *word\n", 'tags'],
      ["retained: &word one\ntags: [*word, two]\n", 'tags'],
      ["topic: first\ntopic: duplicate\n", 'topic'],
      ["defaults: &defaults {topic: example}\n<<: *defaults\n", 'topic'],
      ["tags: {<<: {one: two}}\n", 'tags']
    ]
    fixtures.each do |yaml, field|
      editor = write_post("A stable paragraph.\n", yaml: "title: Original title\n#{yaml}date: 2026-10-24 12:00:00 +0900\n")
      original = read_post
      refute editor.document[:metadata][:writable][field]
      value = field == 'tags' ? ['changed'] : 'changed'
      assert_raises(LocalEditor::InvalidEdit) { editor.save(revision: editor.revision, metadata: { field => value }) }
      assert_equal original, read_post
    end
  end

  def test_metadata_rejects_wrong_types_unknown_keys_controls_and_invalid_dates
    editor = write_post("Body.\n")
    original = read_post
    invalid = [
      { tags: 'tag' }, { tags: [1] }, { tags: ["bad\ntag"] }, { tags: ["nul\x00tag"] },
      { categories: [false] }, { topic: '' }, { topic: "bad\rvalue" }, { series: [] },
      { series_order: '1' }, { series_order: -1 }, { series_order: 1.5 },
      { lang: 'kr' }, { lang: nil }, { date: '2026-10-24' },
      { date: '2026-02-30 12:00:00 +0900' }, { date: '2026-10-24 24:00:00 +0900' },
      { date: '2026-10-24 12:00:00 +2460' }, { permalink: '/changed/' },
      { tags: [], 'tags' => [] }
    ]
    invalid.each do |metadata|
      assert_raises(LocalEditor::InvalidEdit) { editor.save(revision: editor.revision, metadata: metadata) }
      assert_equal original, read_post
    end
    assert_empty Dir[File.join(@root, '.local-editor/backups/*.md')]
  end

  def test_metadata_date_and_language_round_trip_with_an_explicit_offset
    editor = write_post("Body.\n")
    original = read_post
    saved = editor.save(revision: editor.revision, metadata: { date: '2030-02-28 23:59:59 -0530', lang: 'ko' })
    assert_equal '2030-02-28 23:59:59 -0530', editor.metadata_values['date']
    assert_equal 'ko', editor.metadata_values['lang']
    assert_equal body_of(original), body_of(read_post)
    editor.undo(revision: saved[:document][:revision], receipt: saved[:receipt])
    assert_equal original, read_post
  end

  def test_prepared_metadata_write_is_read_only_and_refuses_changed_source
    editor = write_post("Body.\n")
    original = read_post
    prepared = editor.prepare_save(revision: editor.revision, metadata: { tags: ['new'], lang: 'en' })
    assert_equal original, read_post
    assert_empty Dir[File.join(@root, '.local-editor/backups/*.md')]
    assert_equal 2, prepared[:changed]
    assert_equal ['metadata:tags', 'metadata:lang'], prepared[:edited_ids]
    assert_equal original, prepared[:before]
    File.open(File.join(@root, POST), 'a') { |file| file.write("External change.\n") }
    assert_raises(LocalEditor::Conflict) { editor.commit_prepared!(prepared) }
    assert_includes read_post, 'External change.'
  end

  def test_prepared_save_and_undo_can_commit_inside_one_shared_transaction
    editor = write_post("Body.\n")
    original = read_post
    saved = nil
    LocalEditor::Source.with_transaction(root: @root) do
      prepared = editor.prepare_save(revision: editor.revision, metadata: { tags: ['new'], lang: 'en' })
      saved = editor.commit_prepared!(prepared, lock: false)
    end
    assert_equal ['new'], editor.front_matter['tags']
    prepared_undo = editor.prepare_undo(revision: editor.revision, receipt: saved[:receipt])
    refute original == read_post
    assert_equal original, prepared_undo[:after]
    LocalEditor::Source.with_transaction(root: @root) { editor.commit_prepared!(prepared_undo, lock: false) }
    assert_equal original, read_post
  end

  def test_identical_metadata_does_not_create_a_backup_or_normalize_raw_yaml
    editor = write_post("Body.\n", yaml: "title: Original title\ntags: [one, 'two'] # Keep this style.\nlang: en\ndate: 2026-10-24 12:00:00 +0900\n")
    original = read_post
    result = editor.save(revision: editor.revision,
                         metadata: { tags: ['one', 'two'], lang: 'en', date: '2026-10-24 12:00:00 +0900', series: nil })
    assert_equal 0, result[:changed]
    assert_equal nil, result[:receipt]
    assert_equal original, read_post
    assert_empty Dir[File.join(@root, '.local-editor/backups/*.md')]
  end

  def assert(condition, message = 'Assertion failed')
    raise message unless condition
  end

  def refute(condition, message = 'Refutation failed')
    assert(!condition, message)
  end

  def assert_equal(expected, actual)
    assert(expected == actual, "Expected #{expected.inspect}; received #{actual.inspect}")
  end

  def assert_includes(collection, value)
    assert(collection.include?(value), "Expected #{collection.inspect} to include #{value.inspect}")
  end

  def assert_empty(collection)
    assert(collection.empty?, "Expected #{collection.inspect} to be empty")
  end

  def assert_raises(expected)
    begin
      yield
    rescue expected => error
      return error
    end
    raise "Expected #{expected}"
  end

  def flunk(message)
    raise message
  end
end

failures = []
tests = LocalEditorSourceTest.instance_methods.grep(/^test_/).sort
tests.each do |name|
  test = LocalEditorSourceTest.new
  begin
    test.setup
    test.public_send(name)
    puts "PASS #{name}"
  rescue StandardError => error
    failures << name
    warn "FAIL #{name}: #{error.message}\n#{error.backtrace.first(4).join("\n")}"
  ensure
    test.teardown
  end
end
puts "#{tests.length - failures.length}/#{tests.length} source preservation tests passed"
exit(failures.empty? ? 0 : 1)
