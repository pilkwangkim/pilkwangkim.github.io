# frozen_string_literal: true

require 'nokogiri'
require_relative 'source'
require_relative 'tables'

class LocalEditorTablesTest
  def setup; end
  def teardown; end

  def engine(body, offset = 0)
    LocalEditor::TableSource.new(body, offset)
  end

  def cell(editor, text, occurrence = 0)
    editor.document.flat_map { |table| table[:rows].flatten }.select { |item| item[:source] == text }[occurrence] || raise("Missing cell #{text.inspect}")
  end

  def edit(editor, original, replacement, occurrence = 0)
    item = cell(editor, original, occurrence)
    { id: item[:id], original: item[:source], text: replacement }
  end

  def apply(raw, patches)
    result = raw.b.dup
    patches.sort_by { |patch| [-patch[:start], -patch[:finish]] }.each do |patch|
      assert_equal patch[:source].b, result.byteslice(patch[:start]...patch[:finish])
      result[patch[:start]...patch[:finish]] = patch[:replacement].b
    end
    result.force_encoding(Encoding::UTF_8)
  end

  def table_html(editor, body, input: body, line_map: nil)
    parsed = Kramdown::Document.new(input, input: 'GFM', hard_wrap: false)
    editor.annotate(parsed.root, input: input, line_map: line_map)
    Nokogiri::HTML.fragment(parsed.to_html)
  end

  def test_cell_patch_preserves_unicode_crlf_alignment_spaces_and_other_source_bytes
    body = "한글 앞문장.\r\n\r\n| 이름  |    수치 |\r\n| :---- | ------: |\r\n|  예전 값  | $x=1$ |\r\n\r\nUnchanged after.\r\n"
    prefix = "---\r\ntitle: Preserve\r\n---\r\n"
    editor = engine(body, prefix.bytesize)
    changes = editor.patches([edit(editor, '예전 값', '새로운 값')])
    assert_equal prefix + body.sub('예전 값', '새로운 값'), apply(prefix + body, changes)
    assert_equal [:left, :right], editor.document.first[:alignment]
    assert_equal 1, changes.length
    assert changes.first[:start] > prefix.bytesize
  end

  def test_skipped_component_tables_are_not_registered_or_annotated
    table = "| Name | Value |\n| --- | --- |\n| Same | $x$ |\n"
    protected = "<section data-local-editor-skip markdown=\"1\">\n\n#{table}\n</section>\n"
    body = "#{protected}\n#{table}"
    editor = engine(body)
    assert_equal 1, editor.document.length
    rendered = table_html(editor, body)
    assert_equal 2, rendered.css('table').length
    assert_empty rendered.at('[data-local-editor-skip]').css('[data-local-table-id], [data-local-cell-id]')
    assert_equal 1, rendered.css('[data-local-table-id]').length
    changed = apply(body, editor.patches([edit(editor, 'Same', 'Changed')]))
    assert_equal "#{protected}\n#{table.sub('Same', 'Changed')}", changed
  end

  def test_cell_source_preserves_markdown_math_code_and_reference_links
    body = <<~'MD'
      | Label | Value |
      | --- | --- |
      | **Strong** $\alpha$ | `one|two` [docs][primary] |

      [primary]: https://example.com/docs "Documentation"
    MD
    editor = engine(body)
    assert_equal 1, editor.document.length
    assert_equal '`one|two` [docs][primary]', cell(editor, '`one|two` [docs][primary]')[:source]
    updated = apply(body, editor.patches([edit(editor, '**Strong** $\alpha$', '**Clear** $\beta$')]))
    assert_equal body.sub('**Strong** $\alpha$', '**Clear** $\beta$'), updated
    preview = Nokogiri::HTML.fragment(editor.preview(editor.document.first[:id]))
    assert_equal 'one|two', preview.at('code').text
    assert_includes preview.text, 'docs'
    assert_empty preview.css('a, img, script')
  end

  def test_pipes_are_escaped_outside_code_without_changing_column_count
    body = "| A | B |\n| --- | --- |\n| Original | Keep |\n"
    editor = engine(body)
    updated = apply(body, editor.patches([edit(editor, 'Original', 'a|b and `c|d` and $|x|$')]))
    assert_includes updated, 'a\\|b and `c|d` and $\\|x\\|$'
    html = Nokogiri::HTML.fragment(Kramdown::Document.new(updated, input: 'GFM').to_html)
    assert_equal 2, html.css('tbody td').length
    assert_equal 'a|b and c|d and $|x|$', html.at('tbody td').text
  end

  def test_crlf_reference_definitions_remain_unchanged_when_editing_their_table
    body = "| A | B |\r\n| --- | --- |\r\n| Old | [Docs][ref] |\r\n\r\n[ref]: https://example.com \"Reference\"\r\n"
    editor = engine(body)
    assert_equal 1, editor.document.length
    assert_equal body.sub('Old', 'New'), apply(body, editor.patches([edit(editor, 'Old', 'New')]))
    html = table_html(editor, body)
    assert_equal 'https://example.com', html.at('a')['href']
    assert_equal 'Reference', html.at('a')['title']
  end

  def test_existing_backslash_pipe_and_double_backtick_code_are_preserved
    body = "| A | B |\n| --- | --- |\n| Original | Keep |\n"
    editor = engine(body)
    value = 'a\\|b and ``x`|y``'
    updated = apply(body, editor.patches([edit(editor, 'Original', value)]))
    assert_includes updated, "| #{value} | Keep |"
    html = Nokogiri::HTML.fragment(Kramdown::Document.new(updated, input: 'GFM').to_html)
    assert_equal 2, html.css('tbody td').length
    assert_equal 'x`|y', html.at('code').text
  end

  def test_empty_cells_have_unique_zero_length_insert_ranges
    body = "|  | B |\n| --- | --- |\n|   |  |\n"
    editor = engine(body)
    empties = editor.document.first[:rows].flatten.select { |item| item[:source].empty? }
    assert_equal 3, empties.length
    assert_equal 3, empties.map { |item| item[:id] }.uniq.length
    changes = editor.patches(empties.map.with_index { |item, index| { id: item[:id], original: '', text: "New #{index}" } })
    assert changes.all? { |patch| patch[:start] == patch[:finish] }
    updated = apply(body, changes)
    assert_equal 3, updated.scan(/New \d/).length
    assert_includes updated, "|   New 1|  New 2|"
    assert_includes updated, "| --- | --- |\n"
  end

  def test_table_without_outer_pipes_is_supported_and_preserves_delimiters
    body = "Header A | Header B\n--- | ---\nOld | Keep\n"
    editor = engine(body)
    assert_equal 1, editor.document.length
    assert_equal body.sub('Old', 'New'), apply(body, editor.patches([edit(editor, 'Old', 'New')]))
  end

  def test_fenced_table_examples_and_html_tables_have_no_edit_targets
    body = <<~'MD'
      ```markdown
      | Example | Keep |
      | --- | --- |
      | Code | Protected |
      ```

      <table class="original"><tr><td>HTML stays protected.</td></tr></table>

      | Real | Column |
      | --- | --- |
      | Editable | Cell |
    MD
    editor = engine(body)
    assert_equal 1, editor.document.length
    assert_equal 'Real', editor.document.first[:rows].first.first[:source]
    html = table_html(editor, body)
    assert_empty html.css('pre [data-local-cell-id], table.original [data-local-cell-id]')
    assert_equal 4, html.css('table[data-local-table-id] [data-local-cell-id]').length
  end

  def test_duplicate_tables_and_repeated_cell_values_have_distinct_positions
    table = "| A | B |\n| --- | --- |\n| Same | Same |\n"
    body = "#{table}\n#{table}"
    editor = engine(body)
    assert_equal 2, editor.document.length
    ids = editor.document.flat_map { |item| item[:rows].flatten.map { |entry| entry[:id] } }
    assert_equal 8, ids.uniq.length
    assert_equal body.sub(/Same(?= \|\n\z)/, 'Last'), apply(body, editor.patches([edit(editor, 'Same', 'Last', 3)]))
    assert_equal 2, table_html(editor, body).css('table[data-local-table-id]').length
  end

  def test_annotations_require_all_exact_source_lines_not_just_similar_cell_text
    body = "\n\n| A | B |\n| --- | --- |\n| Old | Keep |\n"
    editor = engine(body)
    input = body.sub('Old', 'Generated')
    assert_empty table_html(editor, body, input: input, line_map: { 3 => 3, 4 => 4, 5 => 5 }).css('[data-local-table-id]')
    map = { 1 => 3, 2 => 4, 3 => 5 }
    assert_equal 1, table_html(editor, body, input: body.delete_prefix("\n\n"), line_map: map).css('[data-local-table-id]').length
    assert_empty table_html(editor, body, input: body.delete_prefix("\n\n"), line_map: { 1 => 3, 2 => nil, 3 => 5 }).css('[data-local-table-id]')
  end

  def test_liquid_generated_duplicate_table_never_steals_a_real_tables_ids
    table = "| A | B |\n| --- | --- |\n| Keep | Same |\n"
    body = "{{ site.table }}\n\n#{table}"
    input = "#{table}\n#{table}"
    editor = engine(body)
    line_map = { 5 => 3, 6 => 4, 7 => 5 }
    html = table_html(editor, body, input: input, line_map: line_map)
    assert_equal 2, html.css('table').length
    assert_empty html.css('table').first.css('[data-local-cell-id]')
    assert_equal editor.document.first[:id], html.css('table').last['data-local-table-id']
    assert_equal 4, html.css('table').last.css('[data-local-cell-id]').length
  end

  def test_ial_attributes_are_preserved_and_never_included_in_cell_patches
    body = "| Name | Value |\n| --- | ---: |\n| Old | 1 |\n{: #table-1 .biohub-table style=\"--c1: 40%; --c2: 60%\"}\n\nAfter.\n"
    editor = engine(body)
    assert_equal 1, editor.document.length
    updated = apply(body, editor.patches([edit(editor, 'Old', 'New')]))
    assert_equal body.sub('Old', 'New'), updated
    html = table_html(editor, body)
    table = html.at('table[data-local-table-id]')
    assert_equal 'table-1', table['id']
    assert_equal '--c1: 40%; --c2: 60%', table['style']
    assert_equal 'biohub-table', table['class']
    preview = Nokogiri::HTML.fragment(editor.preview(editor.document.first[:id]))
    assert_nil preview.at('table')['style']
    assert_nil preview.at('table')['id']
  end

  def test_newlines_control_characters_and_active_markup_cannot_escape_a_cell
    body = "| A | B |\n| --- | --- |\n| Old | Keep |\n"
    editor = engine(body)
    ["Two\nrows", "Two\rrows", "Null\u0000byte", '<script>alert(1)</script>',
     '<img src="https://example.com/pixel">', '<span onclick="alert(1)">x</span>',
     '[x](javascript:alert(1))', '[x](data:text/html,test)', '{{ site.secret }}', '{% include x %}',
     'Text {: onclick="bad"}'].each do |value|
      assert_raises(LocalEditor::InvalidEdit) { editor.patches([edit(editor, 'Old', value)]) }
    end
    assert_equal body, apply(body, [])
  end

  def test_harmless_inline_formatting_math_inequalities_and_links_are_allowed
    body = "| A | B |\n| --- | --- |\n| Old | Keep |\n"
    editor = engine(body)
    ['**Bold** _emphasis_ `code`', '$x<y$ and $r>1$', '[Docs](https://example.com)',
     '[Local](/posts/one/)', '<sub>i</sub><sup>2</sup><br>Next'].each do |value|
      assert_equal 1, editor.patches([edit(editor, 'Old', value)]).length
    end
  end

  def test_equations_in_cells_reject_the_same_html_link_and_loader_commands_as_equation_edits
    body = "| A | B |\n| --- | --- |\n| Old | Keep |\n"
    editor = engine(body)
    %w[href url style class cssId htmlClass htmlId htmlData require].each do |command|
      ["$\\#{command}{https://example.com}{x}$", "$ \\#{command}{x} $", "$$\\#{command}{x}$$", "\\(\\#{command}{x}\\)",
       "\\[\\#{command}{x}\\]", "$|x| + \\#{command}{x}$"].each do |value|
        assert_raises(LocalEditor::InvalidEdit) { editor.patches([edit(editor, 'Old', value)]) }
        assert_raises(LocalEditor::InvalidEdit) { editor.preview(editor.document.first[:id], [edit(editor, 'Old', value)]) }
      end
    end
  end

  def test_equation_definitions_cannot_mutate_mathjax_state_from_a_cell_preview
    body = "| A | B |\n| --- | --- |\n| Old | Keep |\n"
    editor = engine(body)
    %w[newcommand renewcommand providecommand DeclareRobustCommand newenvironment renewenvironment
       provideenvironment def gdef edef xdef let futurelet global DeclareMathOperator Newextarrow
       newextarrow definecolor colorlet begingroup endgroup begingroupReset begingroupSandbox
       csname endcsname expandafter catcode mathcode setOptions].each do |command|
      assert_raises(LocalEditor::InvalidEdit) { editor.patches([edit(editor, 'Old', "$\\#{command}{x}$")]) }
    end
    assert_raises(LocalEditor::InvalidEdit) { editor.patches([edit(editor, 'Old', '&#36;&#92;newcommand{&#92;x}{1}&#36;')]) }
    assert_equal 1, editor.patches([edit(editor, 'Old', '`$\\newcommand{\\x}{1}$`')]).length
  end

  def test_entity_decoded_math_cannot_create_active_links_after_html_sanitization
    body = "| A | B |\n| --- | --- |\n| Old | Keep |\n"
    editor = engine(body)
    ['$&#92;href{https://example.com}{x}$', '&#36;&#92;url{https://example.com}&#36;',
     '$&#x5c;require{html}&#x5c;htmlClass{active}{x}$'].each do |value|
      assert_raises(LocalEditor::InvalidEdit) { editor.patches([edit(editor, 'Old', value)]) }
    end
  end

  def test_code_examples_and_ordinary_markdown_links_remain_allowed
    body = "| A | B |\n| --- | --- |\n| Old | Keep |\n"
    editor = engine(body)
    ['`$\\href{https://example.com}{x}$`', '<code>$\\url{https://example.com}$</code>',
     '[Docs](https://example.com/path)', '[Local](/posts/one/)', '$\\frac{x}{y} + \\text{normal}$'].each do |value|
      assert_equal 1, editor.patches([edit(editor, 'Old', value)]).length
    end
  end

  def test_shape_change_is_rejected_even_for_valid_single_line_cell_inputs
    body = "| A | B |\n| --- | --- |\n| Old | Keep |\n"
    editor = engine(body)
    assert_raises(LocalEditor::InvalidEdit) do
      editor.patches([edit(editor, 'A', '---'), edit(editor, 'B', '---')])
    end
  end

  def test_unknown_duplicate_stale_and_invalid_cell_inputs_fail_closed
    body = "| A | B |\n| --- | --- |\n| Old | Keep |\n"
    editor = engine(body)
    assert_raises(LocalEditor::InvalidEdit) { editor.patches([{ id: 'cell-unknown', text: 'No' }]) }
    one = edit(editor, 'Old', 'New')
    assert_raises(LocalEditor::InvalidEdit) { editor.patches([one, one]) }
    assert_raises(LocalEditor::Conflict) { editor.patches([one.merge(original: 'Stale')]) }
    assert_raises(LocalEditor::InvalidEdit) { editor.patches([one.merge(text: nil)]) }
    assert_raises(LocalEditor::InvalidEdit) { editor.patches([one.merge(text: 'x' * 100_001)]) }
    assert_empty editor.patches([edit(editor, 'Old', 'Old')])
  end

  def test_preview_is_native_table_html_without_navigation_network_or_executable_attributes
    body = "| A | B |\n| --- | --- |\n| [Link](https://example.com) | ![Image](https://example.com/image.png) |\n"
    editor = engine(body)
    html = Nokogiri::HTML.fragment(editor.preview(editor.document.first[:id]))
    assert_equal 1, html.css('table').length
    assert_equal 2, html.css('tbody td').length
    assert_empty html.css('a, img, script, [href], [src]')
    assert_equal 'Link', html.css('tbody td').first.text
    assert_raises(LocalEditor::InvalidEdit) { editor.preview('table-unknown') }
  end

  def test_preview_accepts_only_its_own_cells
    table = "| A | B |\n| --- | --- |\n| Old | Keep |\n"
    body = "#{table}\n#{table}"
    editor = engine(body)
    assert_raises(LocalEditor::InvalidEdit) { editor.preview(editor.document.first[:id], [edit(editor, 'Old', 'New', 1)]) }
    html = Nokogiri::HTML.fragment(editor.preview(editor.document.first[:id], [edit(editor, 'Old', 'New')]))
    assert_equal 'New', html.at('tbody td').text
    assert_equal body, body.dup
  end

  def assert(condition, message = 'Assertion failed')
    raise message unless condition
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

  def assert_nil(value)
    assert(value.nil?, "Expected nil; received #{value.inspect}")
  end

  def assert_raises(expected)
    begin
      yield
    rescue expected => error
      return error
    end
    raise "Expected #{expected}"
  end
end

failures = []
tests = LocalEditorTablesTest.instance_methods.grep(/^test_/).sort
tests.each do |name|
  test = LocalEditorTablesTest.new
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
puts "#{tests.length - failures.length}/#{tests.length} table source tests passed"
exit(failures.empty? ? 0 : 1)
