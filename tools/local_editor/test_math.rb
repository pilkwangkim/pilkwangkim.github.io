# frozen_string_literal: true

require 'nokogiri'
require 'tmpdir'
require 'fileutils'
require_relative 'source'
require_relative 'math'

class LocalEditorMathTest
  def engine(body, offset: 31)
    LocalEditor::MathSource.new(body, offset)
  end

  def annotate(body, input: body, map: nil)
    editor = engine(body)
    document = Kramdown::Document.new(input, input: 'GFM', hard_wrap: false)
    before = document.to_html
    editor.annotate(document.root, input: input, line_map: map)
    [editor, Nokogiri::HTML.fragment(document.to_html), before]
  end

  def apply(body, editor, changes, offset: 31)
    result = (' ' * offset + body).b
    editor.patches(changes).sort_by { |patch| -patch[:start] }.each do |patch|
      assert_equal patch[:source].b, result.byteslice(patch[:start]...patch[:finish])
      result[patch[:start]...patch[:finish]] = patch[:replacement].b
    end
    result.byteslice(offset..).force_encoding('UTF-8')
  end

  def test_single_inline_dollars_and_duplicate_unique_ids
    body = '한국어 $x+1$ and $x+1$.'
    editor, dom, = annotate(body)
    assert_equal ['x+1', 'x+1'], editor.document.map { |item| item[:latex] }
    assert_equal 2, editor.document.map { |item| item[:id] }.uniq.length
    assert_equal editor.document.map { |item| item[:id] }, dom.css('[data-local-math-id]').map { |node| node['data-local-math-id'] }
    first, second = editor.document
    result = apply(body, editor, [{ id: second[:id], text: 'y+2', original: 'x+1' }])
    assert_equal '한국어 $x+1$ and $y+2$.', result
    assert_equal false, first[:display]
  end

  def test_skipped_components_lock_nested_equations_and_leave_adjacent_math_editable
    protected = '<section data-local-editor-skip title="a > b"><section><p>$x$ $$y$$</p></section><p>$z$</p></section>'
    body = protected + '<p>$x$ $$y$$</p>'
    editor, dom, before = annotate(body)
    assert_equal ['x', 'y'], editor.document.map { |item| item[:latex] }
    assert_equal 0, dom.at('[data-local-editor-skip]').css('[data-local-math-id]').length
    assert_equal 2, dom.css('[data-local-math-id]').length
    changed = apply(body, editor, [{ id: editor.document.first[:id], text: 'q', original: 'x' }])
    assert_equal protected + '<p>$q$ $$y$$</p>', changed
    dom.css('[data-local-math-id]').each { |node| node.replace(node.children) }
    assert_equal before, dom.to_html
  end

  def test_skip_marker_text_inside_another_attribute_is_not_a_marker
    body = '<p title="a data-local-editor-skip example">$x$</p>'
    editor, dom, = annotate(body)
    assert_equal ['x'], editor.document.map { |item| item[:latex] }
    assert_equal 1, dom.css('[data-local-math-id]').length
  end

  def test_skip_marker_in_a_code_example_does_not_lock_following_math
    body = "```html\n<section data-local-editor-skip>\n```\n\nVisible $x$.\n"
    editor, dom, = annotate(body)
    assert_equal ['x'], editor.document.map { |item| item[:latex] }
    assert_equal 1, dom.css('[data-local-math-id]').length
  end

  def test_multiline_display_preserves_outer_whitespace_and_exact_bytes
    body = "Before.\r\n\r\n$$\r\n  \\begin{aligned}\r\na &= b \\\\\r\nc &= d\r\n\\end{aligned}  \r\n$$\r\n\r\nAfter.\r\n"
    editor = engine(body)
    item = editor.document.fetch(0)
    assert item[:display]
    result = apply(body, editor, [{ id: item[:id], text: 'E=mc^2', original: item[:latex] }])
    assert_equal "Before.\r\n\r\n$$\r\n  E=mc^2  \r\n$$\r\n\r\nAfter.\r\n", result
  end

  def test_native_display_and_inline_double_dollars_keep_conversion
    body = "Inline $$x+1$$ remains.\n\n$$\na=b\n$$\n"
    editor, dom, before = annotate(body)
    assert_equal [false, true], editor.document.map { |item| item[:display] }
    assert_equal 2, dom.css('[data-local-math-id]').length
    dom.css('[data-local-math-id]').each { |node| node.replace(node.children) }
    assert_equal before.gsub(/\s+/, ' ').strip, dom.to_html.gsub(/\s+/, ' ').strip
  end

  def test_raw_html_caption_and_math_near_less_than_or_entities
    body = '<figure><figcaption>The ratio $r&lt;1$ and $A>0$ applies.</figcaption></figure>'
    editor, dom, before = annotate(body)
    assert_equal ['r<1', 'A>0'], editor.document.map { |item| item[:latex] }
    assert_equal 2, dom.css('figcaption [data-local-math-id]').length
    dom.css('[data-local-math-id]').each { |node| node.replace(node.children) }
    assert_equal before, dom.to_html
  end

  def test_code_tables_attributes_scripts_comments_and_urls_are_protected
    body = <<~'MD'
      Prose $editable$.
      `code $no$` and `` $not$ ``.

      ```tex
      $fenced$
      ```

      ~~~
      $$tilde$$
      ~~~

          $indented$

      <!-- $comment$ -->
      <script>const text = '$script$';</script>
      <style>.x { content: '$style$' }</style>
      <a title="$attribute$" href="/$path$/">A</a>
      [Link](https://example.com/$url$)
      <table><tr><td>$html_table$</td></tr></table>

      | Value | Other |
      |---|---|
      | $markdown_table$ | Text |

      Final $yes$.
    MD
    assert_equal ['editable', 'yes'], engine(body).document.map { |item| item[:latex] }
  end

  def test_escaped_dollars_and_currency_are_not_equations
    body = 'Cost \$20 and $30 per ticket, then $40. Valid $x$ and $5$.'
    assert_equal ['x', '5'], engine(body).document.map { |item| item[:latex] }
  end

  def test_escaped_dollar_inside_equation_is_preserved
    body = 'Price $\\text{cost}=\\$5$.'
    editor = engine(body)
    assert_equal ['\\text{cost}=\\$5'], editor.document.map { |item| item[:latex] }
    item = editor.document.first
    result = apply(body, editor, [{ id: item[:id], text: '\\text{cost}=\\$8' }])
    assert_equal 'Price $\\text{cost}=\\$8$.', result
  end

  def test_latex_only_limits_injection_and_inline_newlines
    editor = engine('Text $x$ and $$y$$.')
    id = editor.document.first[:id]
    [nil, '', "x\ny", '$z$', '\\(x\\)', '{{ site.title }}', '{% include a %}', '<script>x</script>', '\\href{javascript:x}{y}', "x\x00y"].each do |text|
      assert_raises(LocalEditor::InvalidEdit) { editor.patches([{ id: id, text: text }]) }
    end
    assert_raises(LocalEditor::Conflict) { editor.patches([{ id: id, text: 'y', original: 'wrong' }]) }
    assert_raises(LocalEditor::InvalidEdit) { editor.patches([{ id: id, text: 'y' }, { id: id, text: 'z' }]) }
    assert_raises(LocalEditor::InvalidEdit) { editor.patches([{ id: 'other', text: 'y' }]) }
    assert_equal [], editor.patches([{ id: id, text: 'x' }])
    assert_equal 1, editor.patches([{ id: id, text: 'x < y' }]).length
    assert_equal 1, editor.patches([{ id: id, text: 'x<y' }]).length
    assert_equal 1, editor.patches([{ id: id, text: 'a \\\\[2pt] b' }]).length
  end

  def test_state_defining_tex_commands_cannot_leak_preview_state
    editor = engine('Before $x+1$ after.')
    id = editor.document.first[:id]
    %w[newcommand renewcommand providecommand DeclareRobustCommand newenvironment renewenvironment provideenvironment def gdef edef xdef let futurelet global DeclareMathOperator Newextarrow newextarrow definecolor colorlet begingroup endgroup begingroupReset begingroupSandbox csname endcsname expandafter catcode mathcode setOptions].each do |command|
      assert_raises(LocalEditor::InvalidEdit) { editor.patches([{ id: id, text: "\\#{command}{a}{b}" }]) }
    end
    assert_equal 1, editor.patches([{ id: id, text: '\\begin{aligned}a&=b\\end{aligned}' }]).length
    legacy = engine('$\\newcommand{\\foo}{x}\\foo$')
    existing = legacy.document.first
    assert_equal [], legacy.patches([{ id: existing[:id], text: existing[:latex] }])
  end

  def test_liquid_generated_formula_never_steals_literal_formula_id
    body = "{{ site.formula }}\n\nLiteral $x$.\n"
    input = "$x$\n\nLiteral $x$.\n"
    editor, dom, = annotate(body, input: input, map: { 2 => 2, 3 => 3 })
    nodes = dom.css('[data-local-math-id]')
    assert_equal 1, nodes.length
    assert_equal editor.document.first[:id], nodes.first['data-local-math-id']
    assert_equal 'Literal $x$.', nodes.first.parent.text
  end

  def test_multiline_liquid_expansion_uses_only_mapped_literal_tail
    body = "{% include formula %}\n\nLiteral $x$.\n"
    input = "Generated $x$.\nAnother $x$.\n\nLiteral $x$.\n"
    editor, dom, = annotate(body, input: input, map: { 3 => 2, 4 => 3 })
    assert_equal [editor.document.first[:id]], dom.css('[data-local-math-id]').map { |node| node['data-local-math-id'] }
  end

  def test_template_tokens_and_comment_blocks_are_not_math
    body = "{{ '$output$' }} {% assign formula='$assigned$' %}\n{% comment %}\n$comment$\n{% endcomment %}\nGood $x$.\n"
    assert_equal ['x'], engine(body).document.map { |item| item[:latex] }
  end

  def test_slash_delimiters_are_collected_without_changing_native_markdown_output
    body = "Use \\(x+y\\) and \\[z\\].\n\n<figcaption>Raw \\(q\\).</figcaption>\n"
    editor, dom, before = annotate(body)
    assert_equal ['\\(', '\\[', '\\('], editor.document.map { |item| item[:delimiter] }
    assert_equal [editor.document.last[:id]], dom.css('[data-local-math-id]').map { |node| node['data-local-math-id'] }
    dom.css('[data-local-math-id]').each { |node| node.replace(node.children) }
    assert_equal before, dom.to_html
  end

  def test_plain_parentheses_from_consumed_slash_delimiters_have_no_formula_targets
    body = 'The assignment \\(U_2\\) and bracket \\[x+y\\] stay as native prose. Valid $z$ remains editable.'
    editor, dom, before = annotate(body)
    assert_equal 3, editor.document.length
    assert_equal [editor.document.last[:id]], dom.css('[data-local-math-id]').map { |node| node['data-local-math-id'] }
    assert_equal '(U_2)', dom.text[/\(U_2\)/]
    dom.css('[data-local-math-id]').each { |node| node.replace(node.children) }
    assert_equal before, dom.to_html
  end

  def test_markdown_decoded_braces_keep_original_native_rendering_and_source_ownership
    body = 'Set $a\\in\\{1,2\\}$ and $x\\neq y$. 한국어 끝.'
    editor, dom, before = annotate(body)
    assert_equal 2, dom.css('[data-local-math-id]').length
    assert_equal ['a\\in{1,2}', 'x\\neq y'], editor.document.map { |item| item[:latex] }
    dom.css('[data-local-math-id]').each { |node| node.replace(node.children) }
    assert_equal before, dom.to_html
    item = editor.document.first
    result = apply(body, editor, [{ id: item[:id], text: 'a\\in\\{3,4\\}' }])
    assert_equal 'Set $a\\in\\\\{3,4\\\\}$ and $x\\neq y$. 한국어 끝.', result
    assert_equal 'Set $a\\in\\{3,4\\}$ and $x\\neq y$. 한국어 끝.', Nokogiri::HTML.fragment(Kramdown::Document.new(result, input: 'GFM').to_html).text.strip
    assert_equal [], editor.patches([{ id: item[:id], text: item[:latex], original: item[:latex] }])
  end

  def test_changed_inline_math_round_trips_rendered_tex_and_remains_editable
    body = 'Before $x$ after.'
    ['a\\in\\{1,2\\}', '\\left\\|x\\right\\|', '\\text{cost}=\\$5', 'x_{a_b}', 'a*b*c', 'a \\\\[2pt] b'].each do |latex|
      editor = engine(body)
      item = editor.document.first
      changed = apply(body, editor, [{ id: item[:id], original: item[:latex], text: latex }])
      current, dom, native = annotate(changed)
      assert_equal "Before $#{latex}$ after.", dom.text.strip
      assert_equal latex, current.document.first[:latex]
      assert_equal 1, dom.css('[data-local-math-id]').length
      dom.css('[data-local-math-id]').each { |node| node.replace(node.children) }
      assert_equal native, dom.to_html
      assert_equal [], current.patches([{ id: current.document.first[:id], text: latex, original: latex }])
    end
  end

  def test_adjacent_inline_equations_cannot_pair_markdown_emphasis_after_an_edit
    body = '목표는 $q_i^{(\\ell)}$, 영상 모델의 예측은 $p_i=f_\\theta(X_i)$이다.'
    editor = engine(body)
    item = editor.document.first
    changed = apply(body, editor, [{ id: item[:id], text: '\\hat{q}_i', original: item[:latex] }])
    current, dom, = annotate(changed)
    assert_equal '목표는 $\\hat{q}_i$, 영상 모델의 예측은 $p_i=f_\\theta(X_i)$이다.', dom.text.strip
    assert_equal ['\\hat{q}_i', 'p_i=f_\\theta(X_i)'], current.document.map { |equation| equation[:latex] }
    assert_equal 2, dom.css('[data-local-math-id]').length
    assert_equal 0, dom.css('em').length
    assert_equal body[body.index(',')..], changed[changed.index(',')..]
  end

  def test_encoded_inline_math_save_and_undo_restore_exact_source_bytes
    Dir.mktmpdir('local-editor-encoded-math-') do |root|
      FileUtils.mkdir_p(File.join(root, '_posts'))
      path = '_posts/2026-10-24-Math.md'
      absolute = File.join(root, path)
      original = "---\r\ntitle: Exact source\r\n---\r\nBefore $x$ and $p_i=f_\\theta(X_i)$ after.\r\n"
      File.binwrite(absolute, original)
      source = LocalEditor::Source.new(root: root, path: path)
      item = source.document[:structures][:math].first
      latex = '\\hat{q}_i+\\text{cost}=\\$5+\\{1,2\\}'
      saved = source.save(revision: source.revision, edits: [{ id: item[:id], original: item[:latex], text: latex }])
      assert_equal latex, saved[:document][:structures][:math].first[:latex]
      changed = File.binread(absolute).force_encoding('UTF-8')
      body = changed.split("---\r\n", 3).last
      assert_equal "Before $#{latex}$ and $p_i=f_\\theta(X_i)$ after.", Nokogiri::HTML.fragment(Kramdown::Document.new(body, input: 'GFM').to_html).text.strip
      assert_equal original[0...original.index('$x$')], changed[0...changed.index('$')]
      assert_equal original.split('$x$', 2).last, changed[changed.index(' and ')..]
      source.undo(revision: saved[:document][:revision], receipt: saved[:receipt])
      assert_equal original.b, File.binread(absolute)
    end
  end

  def test_raw_caption_entities_are_editable_as_rendered_tex_without_rewriting_noops
    body = '<figcaption>Ratio $r&lt;1$ and $a\\in\\{1,2\\}$ remain.</figcaption>'
    editor, dom, before = annotate(body)
    item = editor.document.first
    assert_equal 'r<1', item[:latex]
    assert_equal [], editor.patches([{ id: item[:id], original: 'r<1', text: 'r<1' }])
    changed = apply(body, editor, [{ id: item[:id], original: 'r<1', text: 'r<2' }])
    assert_equal body.sub('r&lt;1', 'r<2'), changed
    assert_equal 'Ratio $r<2$ and $a\\in\\{1,2\\}$ remain.', Nokogiri::HTML.fragment(Kramdown::Document.new(changed, input: 'GFM').to_html).text.strip
    dom.css('[data-local-math-id]').each { |node| node.replace(node.children) }
    assert_equal before, dom.to_html
  end

  def test_html_raw_captions_keep_backslash_escapes_in_rendered_children
    body = '<figcaption>Set $a\\in\\{1,2\\}$ works.</figcaption>'
    editor, dom, before = annotate(body)
    assert_equal 1, dom.css('[data-local-math-id]').length
    dom.css('[data-local-math-id]').each { |node| node.replace(node.children) }
    assert_equal before, dom.to_html
  end

  def test_an_unclosed_fence_protects_everything_to_eof
    body = "Visible $x$.\n\n~~~text\n$hidden$\n"
    assert_equal ['x'], engine(body).document.map { |item| item[:latex] }
  end

  def test_identical_native_math_nodes_have_unique_correct_ids
    body = '$$x$$ and $$x$$ and $$y$$.'
    editor, dom, = annotate(body)
    assert_equal editor.document.map { |item| item[:id] }, dom.css('[data-local-math-id]').map { |node| node['data-local-math-id'] }
  end

  def assert(value, message = 'Assertion failed')
    raise message unless value
  end
  def assert_equal(expected, actual)
    assert(expected == actual, "Expected #{expected.inspect}, got #{actual.inspect}")
  end
  def assert_raises(type)
    yield
    raise "Expected #{type}"
  rescue type
    true
  end
end

failures = []
tests = LocalEditorMathTest.instance_methods.grep(/^test_/).sort
tests.each do |name|
  begin
    LocalEditorMathTest.new.public_send(name)
    puts "PASS #{name}"
  rescue StandardError => error
    failures << name
    warn "FAIL #{name}: #{error.message}\n#{error.backtrace.first(3).join("\n")}"
  end
end
puts "#{tests.length - failures.length}/#{tests.length} math source tests passed"
exit(failures.empty? ? 0 : 1)
