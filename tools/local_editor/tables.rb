# frozen_string_literal: true

require 'cgi'
require 'digest'
require 'kramdown'
require 'kramdown-parser-gfm'
require 'nokogiri'
require_relative 'math'

module LocalEditor
  # Native tables retain their original rows, separators and whitespace. Only
  # the content range of a named cell can be replaced, never the entire table.
  class TableSource
    MAX_CELL_BYTES = 100_000
    SAFE_TAGS = %w[br strong em code sub sup span].freeze
    UNSAFE_MATH_COMMAND = /\\(?:href|url|style|class|cssId|htmlClass|htmlId|htmlData|require)\b/
    STATE_DEFINING_MATH_COMMAND = /\\(?:newcommand|renewcommand|providecommand|DeclareRobustCommand|newenvironment|renewenvironment|provideenvironment|def|gdef|edef|xdef|let|futurelet|global|DeclareMathOperator|Newextarrow|newextarrow|definecolor|colorlet|begingroup|endgroup|begingroupReset|begingroupSandbox|csname|endcsname|expandafter|catcode|mathcode|setOptions)\b/

    def initialize(body, body_offset = 0, ast: nil)
      @body = body
      @offset = body_offset
      @lines = body.lines
      @starts = [0]
      @lines.each { |line| @starts << @starts.last + line.bytesize }
      # Native reference links can be defined far below a table. Retain their
      # parsing context for table-only validation and previews; source ranges
      # still address the original table exclusively.
      @definitions = []
      body.gsub("\r\n", "\n").to_enum(:scan, Kramdown::Parser::Kramdown::LINK_DEFINITION_START).each do
        definition = Regexp.last_match[0]
        @definitions << definition unless definition.match?(/\A[ \t]*\[\^/)
      end
      @tables = []
      walk(ast || Kramdown::Document.new(body, input: 'GFM', hard_wrap: false).root) do |node|
        next unless node.type == :table

        table = extract(node)
        @tables << table if table
      end
      @cells = @tables.flat_map { |table| table[:cells] }.to_h { |cell| [cell[:id], cell] }
    end

    def document
      @tables.map do |table|
        {
          id: table[:id], format: 'markdown', rows: table[:rows].map do |row|
            row.map { |cell| cell.slice(:id, :source, :header, :row, :column) }
          end,
          row_count: table[:rows].length, column_count: table[:rows].first.length,
          header_rows: table[:rows].count { |row| row.first[:header] }, alignment: table[:alignment],
          label: "Table #{@tables.index(table) + 1}"
        }
      end
    end

    def patches(edits)
      raise InvalidEdit, 'Supply a list of table cell edits.' unless edits.is_a?(Array) && edits.length <= 2000

      seen = {}
      selected = edits.filter_map do |edit|
        raise InvalidEdit, 'Each table edit must name a cell.' unless edit.is_a?(Hash)

        id = edit[:id] || edit['id']
        cell = @cells[id]
        raise InvalidEdit, 'This table cell is protected or no longer matches the source.' unless cell
        raise InvalidEdit, 'A table cell was supplied more than once.' if seen[id]

        seen[id] = true
        text = edit.key?(:text) ? edit[:text] : edit['text']
        original = edit.key?(:original) ? edit[:original] : edit['original']
        raise Conflict, 'The original table cell no longer matches. Reload before saving.' if !original.nil? && original != cell[:source]
        next if text == cell[:source]

        validate_cell!(text)
        replacement = escape_pipes(text)
        cell.slice(:id, :start, :finish, :source).merge(replacement: replacement, table_id: cell[:table_id])
      end
      selected.group_by { |patch| patch[:table_id] }.each do |table_id, replacements|
        table = @tables.find { |candidate| candidate[:id] == table_id }
        updated = patched_table(table, replacements)
        parsed = native(updated).root.children.reject { |node| node.type == :blank }
        unless parsed.length == 1 && parsed.first.type == :table && shape(parsed.first) == table[:shape]
          raise InvalidEdit, 'Keep the existing table rows, columns and header. Edit one cell at a time.'
        end
      end
      selected.map { |patch| patch.reject { |key, _| key == :table_id } }
    end

    # The preview has no navigable links, image loads or executable attributes.
    # MathJax can render its ordinary math text after insertion in the editor.
    def preview(table_id, edits = [])
      table = @tables.find { |candidate| candidate[:id] == table_id }
      raise InvalidEdit, 'Choose a known table.' unless table

      selected = patches(edits)
      unless selected.all? { |patch| patch[:start] >= table[:start] && patch[:finish] <= table[:finish] }
        raise InvalidEdit, 'A table preview can only include cells from that table.'
      end
      fragment = Nokogiri::HTML.fragment(native(patched_table(table, selected)).to_html)
      fragment.css('script, style, img, iframe, object, embed, svg, math, link, meta, input, button, form').remove
      fragment.css('a').each { |node| node.replace(node.children) }
      fragment.css('*').each do |node|
        node.attribute_nodes.each do |attribute|
          next if attribute.name == 'style' && attribute.value.match?(/\Atext-align:\s*(?:left|right|center);?\z/)

          node.remove_attribute(attribute.name)
        end
      end
      fragment.to_html
    end

    def annotate(root, input: nil, line_map: nil)
      rendered_lines = input&.lines
      walk(root) do |node|
        next unless node.type == :table && node.options[:location]

        rendered_line = node.options[:location]
        source_line = line_map ? line_map[rendered_line] : rendered_line
        table = @tables.find { |candidate| candidate[:line] == source_line }
        next unless table && signature(node) == table[:signature]
        next unless table[:line_count].times.all? do |index|
          expected = table[:line] + index
          mapped = line_map ? line_map[rendered_line + index] : rendered_line + index
          mapped == expected && (!rendered_lines || rendered_lines[rendered_line + index - 1] == @lines[expected - 1])
        end

        node.attr['data-local-table-id'] = table[:id]
        native_rows(node).zip(table[:rows]).each do |(row, _header), cells|
          row.children.zip(cells).each { |cell_node, cell| cell_node.attr['data-local-cell-id'] = cell[:id] }
        end
      end
      root
    end

    private

    def native(markdown)
      context = @definitions&.any? ? "#{markdown}\n\n#{@definitions.join}" : markdown
      Kramdown::Document.new(context, input: 'GFM', hard_wrap: false)
    end

    def walk(node, &block)
      return if SkippedContent.node?(node)

      yield node
      # A table owns all of its cells. Nested snippets cannot become tables.
      return if node.type == :table

      node.children.each { |child| walk(child, &block) }
    end

    def native_rows(table)
      table.children.flat_map do |section|
        section.children.map { |row| [row, section.type == :thead] }
      end
    end

    def shape(table)
      [table.options[:alignment], table.children.map { |section| [section.type, section.children.map { |row| row.children.length }] }]
    end

    def signature(node)
      [node.type, node.value, node.attr.reject { |name, _| name.start_with?('data-local-') },
       node.options.reject { |key, _| %i[location].include?(key) }, node.children.map { |child| signature(child) }]
    end

    def extract(node)
      first = node.options[:location]
      return unless first.is_a?(Integer) && first.positive?

      lines = []
      @lines.drop(first - 1).each do |line|
        break if line.strip.empty? || !line.include?('|') || line.match?(/\A(?: {4}|\t|\s*>)/)

        lines << line
      end
      return if lines.empty?

      # Existing table IAL attributes (responsive classes, column widths and
      # captions) belong to the table but are never writable cell content.
      following = @lines[first - 1 + lines.length]
      lines << following if following&.match?(/\A[ \t]*\{:[^\r\n]*\}[ \t]*\r?\n?\z/)

      # Reject any uncertain source extent, including an adjacent paragraph
      # that merely happens to contain a pipe.
      parsed = native(lines.join).root.children.reject { |child| child.type == :blank }
      return unless parsed.length == 1 && parsed.first.type == :table && signature(parsed.first) == signature(node)

      source = lines.join
      start = @offset + @starts[first - 1]
      table_id = "table-#{Digest::SHA256.hexdigest("#{start}:#{source}")[0, 24]}"
      row_nodes = native_rows(node)
      rows = []
      leading_pipe = lines.first.match?(/\A[ \t]*\|/)
      lines.each_with_index do |line, line_index|
        next if separator?(line) || line.match?(/\A[ \t]*\{:/)

        content = line.sub(/\r?\n\z/, '')
        cells = split_cells(content, leading_pipe)
        native_row, header = row_nodes[rows.length]
        return unless native_row && cells.length == native_row.children.length

        row = rows.length
        rows << cells.each_with_index.map do |range, column|
          raw = content.b.byteslice(range[0]...range[1]).force_encoding(Encoding::UTF_8)
          left = raw[/\A[ \t]*/].bytesize
          right = raw[/[ \t]*\z/].bytesize
          cell_start = @offset + @starts[first - 1 + line_index] + range[0] + left
          text = raw.strip
          finish = text.empty? ? cell_start : @offset + @starts[first - 1 + line_index] + range[1] - right
          {
            id: "cell-#{Digest::SHA256.hexdigest("#{table_id}:#{row}:#{column}:#{cell_start}")[0, 24]}",
            table_id: table_id, start: cell_start, finish: finish, source: text,
            header: header, row: row, column: column
          }
        end
      end
      return unless rows.length == row_nodes.length && rows.any?

      {
        id: table_id, line: first, line_count: lines.length, start: start, finish: start + source.bytesize,
        source: source, rows: rows, cells: rows.flatten, alignment: node.options[:alignment],
        shape: shape(node), signature: signature(node)
      }
    end

    def separator?(line)
      line.match?(/\A[+|: \t-]*-[+|: \t-]*\r?\n?\z/) || line.match?(/\A[+|: \t=]*=[+|: \t=]*\r?\n?\z/)
    end

    def protected_inline_ranges(line)
      binary = line.b
      ranges = []
      cursor = 0
      while cursor < binary.bytesize
        if binary.byteslice(cursor, 5)&.downcase == '<code' && (match = /\A<code\b[^>]*>.*?<\/code>/im.match(binary.byteslice(cursor..)))
          ranges << (cursor...(cursor + match[0].bytesize))
          cursor += match[0].bytesize
        elsif binary.getbyte(cursor) == 96 && (cursor.zero? || binary.getbyte(cursor - 1) != 92)
          length = binary.byteslice(cursor..)[/\A`+/].bytesize
          ending = /(?<!`)`{#{length}}(?!`)/.match(binary, cursor + length)
          if ending
            ranges << (cursor...(ending.end(0)))
            cursor = ending.end(0)
          else
            cursor += length
          end
        else
          cursor += 1
        end
      end
      ranges
    end

    def split_cells(line, leading_pipe)
      binary = line.b
      protected = protected_inline_ranges(binary)
      pipes = binary.bytes.each_index.select do |index|
        binary.getbyte(index) == 124 && (index.zero? || binary.getbyte(index - 1) != 92) &&
          protected.none? { |range| range.cover?(index) }
      end
      boundaries = [-1] + pipes + [binary.bytesize]
      cells = boundaries.each_cons(2).map { |left, right| [left + 1, right] }
      cells.shift if leading_pipe && binary.byteslice(cells.first[0]...cells.first[1]).strip.empty?
      cells.pop if cells.any? && binary.byteslice(cells.last[0]...cells.last[1]).strip.empty?
      cells
    end

    def escape_pipes(text)
      binary = text.b
      protected = protected_inline_ranges(binary)
      result = +''.b
      binary.bytes.each_with_index do |byte, index|
        if byte == 124 && (index.zero? || binary.getbyte(index - 1) != 92) && protected.none? { |range| range.cover?(index) }
          result << '\\'
        end
        result << byte
      end
      result.force_encoding(Encoding::UTF_8)
    end

    def validate_cell!(text)
      unless text.is_a?(String) && text.valid_encoding? && text.bytesize <= MAX_CELL_BYTES &&
             !text.match?(/[\r\n\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F]/)
        raise InvalidEdit, 'Table cells must be a single line of Markdown, no longer than 100 KB.'
      end
      if text.include?('{{') || text.include?('{%') || text.include?('{:')
        raise InvalidEdit, 'Liquid templates and HTML attributes cannot be inserted into table cells.'
      end
      text.scan(/<\/?[a-zA-Z][a-zA-Z0-9:_-]*(?=[\s\/>])[^>]*>|<![^>]*>/).each do |tag|
        next if tag.match?(/\A<\/?(?:#{SAFE_TAGS.join('|')})\s*\/?>\z/i)

        raise InvalidEdit, 'Use Markdown formatting in table cells; active HTML and attributes are protected.'
      end
      document = native("| Header | Other |\n| --- | --- |\n| #{escape_pipes(text)} | Keep |\n")
      validate_ast!(document.root)
      # MathJax runs after the Markdown preview is sanitized. Reject the same
      # HTML/link TeX commands as MathSource before they can create new links or
      # attributes during that later pass. Also inspect rendered text: entities
      # such as &#92; or &#36; must not bypass the source-level check.
      rendered = Nokogiri::HTML.fragment(document.to_html).at('tbody td')
      rendered&.css('code, pre')&.remove
      [text, rendered&.text].compact.uniq.each do |candidate|
        # Cells can introduce new delimiters, including spaced $ ... $ forms
        # that the article mapper conservatively skips as possible currency.
        # Normalize only this validation copy so commands in those forms cannot
        # bypass the same checks. The saved source remains untouched.
        candidates = [candidate, candidate.gsub(/\$[ \t]+/, '$').gsub(/[ \t]+\$/, '$')].uniq
        candidates.each do |normalized|
          math = MathSource.new(normalized, 0, ast: Kramdown::Element.new(:root))
          equations = math.document
          if equations.any? { |equation| equation[:latex].match?(UNSAFE_MATH_COMMAND) }
            raise InvalidEdit, 'Equations in table cells cannot contain HTML/link commands.'
          end
          if equations.any? { |equation| equation[:latex].match?(STATE_DEFINING_MATH_COMMAND) }
            raise InvalidEdit, 'Keep equations self-contained. Macro, environment and color definitions cannot be edited here.'
          end
        end
      end
    end

    def validate_ast!(node)
      if node.type == :img || node.attr.keys.any? { |key| key.match?(/\Aon/i) || %w[src srcdoc style].include?(key) }
        raise InvalidEdit, 'Images and active HTML cannot be inserted into table cells.'
      end
      if node.type == :html_element && !SAFE_TAGS.include?(node.value.to_s.downcase)
        raise InvalidEdit, 'Use Markdown formatting in table cells; active HTML is protected.'
      end
      if node.type == :a
        href = CGI.unescapeHTML(node.attr['href'].to_s).gsub(/[\s\u0000-\u001F\u007F]/, '')
        scheme = href[/\A([a-z][a-z0-9+.-]*):/i, 1]
        raise InvalidEdit, 'Use an ordinary web, mail or relative link in the table.' if scheme && !%w[http https mailto].include?(scheme.downcase)
      end
      node.children.each { |child| validate_ast!(child) }
    end

    def patched_table(table, replacements)
      result = table[:source].b.dup
      replacements.sort_by { |patch| [-patch[:start], -patch[:finish]] }.each do |patch|
        left = patch[:start] - table[:start]
        right = patch[:finish] - table[:start]
        raise Conflict, 'A table source range changed. Reload before saving.' unless result.byteslice(left...right) == patch[:source].b

        result[left...right] = patch[:replacement].b
      end
      result.force_encoding(Encoding::UTF_8)
    end
  end
end
