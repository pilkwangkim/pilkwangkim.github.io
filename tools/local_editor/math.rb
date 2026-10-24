# frozen_string_literal: true

require 'digest'
require 'cgi'
require 'kramdown'
require 'kramdown-parser-gfm'

module LocalEditor
  # Interactive components own their rendered descendants. Keep complete HTML
  # ranges locked as well as skipping AST nodes: locations identify lines, so a
  # repeated literal next to a component must never match its protected text.
  module SkippedContent
    VOID_TAGS = %w[area base br col embed hr img input link meta param source track wbr].freeze
    HTML_TOKEN = /<!--[\s\S]*?(?:-->|\z)|<\/?([A-Za-z][A-Za-z0-9:-]*)(?=[\s\/>])(?:[^>"']|"[^"]*"|'[^']*')*>/n
    ATTRIBUTE = /\s+([A-Za-z_:][A-Za-z0-9_.:-]*)(?:\s*=\s*(?:"[^"]*"|'[^']*'|[^\s"'=<>`]+))?/n

    def self.node?(node)
      node.type == :html_element && node.attr.keys.any? { |name| name.casecmp?('data-local-editor-skip') }
    end

    def self.ranges(body, ignored: [])
      stack = []
      ranges = []
      body.b.to_enum(:scan, HTML_TOKEN).each do
        match = Regexp.last_match
        next unless match[1]
        next if ignored.any? { |first, finish| match.begin(0) < finish && match.end(0) > first }

        tag = match[1].downcase
        token = match[0]
        if token.start_with?('</')
          index = stack.rindex { |entry| entry[:tag] == tag }
          next unless index

          stack.slice!(index..).each do |entry|
            ranges << [entry[:start], match.end(0)] if entry[:skip]
          end
        elsif !VOID_TAGS.include?(tag) && !token.match?(/\/\s*>\z/n)
          skip = token.scan(ATTRIBUTE).any? { |attribute| attribute.first.casecmp?('data-local-editor-skip') }
          stack << { tag: tag, start: match.begin(0), skip: skip }
        end
      end
      stack.each { |entry| ranges << [entry[:start], body.bytesize] if entry[:skip] }
      ranges
    end
  end

  # Keeps exact source positions rather than serializing a rendered equation.
  # Dollar math in tables belongs to TableSource, which owns the complete cell.
  class MathSource
    MAX_LATEX_BYTES = 100_000
    STATE_DEFINING_COMMANDS = /\\(?:newcommand|renewcommand|providecommand|DeclareRobustCommand|newenvironment|renewenvironment|provideenvironment|def|gdef|edef|xdef|let|futurelet|global|DeclareMathOperator|Newextarrow|newextarrow|definecolor|colorlet|begingroup|endgroup|begingroupReset|begingroupSandbox|csname|endcsname|expandafter|catcode|mathcode|setOptions)\b/
    BLOCKED_TAGS = %w[table thead tbody tfoot tr th td code pre script style svg math iframe textarea].freeze
    BLOCKED_TYPES = %i[table codeblock codespan raw comment img footnote].freeze

    def initialize(body, body_offset, ast: nil, project_text: true)
      @body = body
      @binary = body.b
      @body_offset = body_offset
      @line_starts = [0]
      @binary.each_line { |line| @line_starts << @line_starts.last + line.bytesize }
      @protected = []
      root = ast || Kramdown::Document.new(body, input: 'GFM', hard_wrap: false).root
      protect_source(root)
      @segments = scan
      project_text_segments(root) if project_text
    end

    def document
      @segments.map { |segment| segment.slice(:id, :latex, :display, :delimiter) }
    end

    def patches(edits)
      unless edits.is_a?(Array) && edits.length <= 2000
        raise InvalidEdit, 'Supply no more than 2,000 equation edits.'
      end

      seen = {}
      edits.filter_map do |edit|
        raise InvalidEdit, 'Each equation edit must identify an equation.' unless edit.is_a?(Hash)

        id = edit[:id] || edit['id']
        raise InvalidEdit, 'An equation was supplied more than once.' if seen[id]

        seen[id] = true
        segment = @segments.find { |item| item[:id] == id }
        raise InvalidEdit, 'This equation is protected or its source changed. Reload the page.' unless segment

        text = edit.key?(:text) ? edit[:text] : edit['text']
        original = edit.key?(:original) ? edit[:original] : edit['original']
        if !original.nil? && original != segment[:latex]
          raise Conflict, 'The original equation no longer matches. Reload before saving.'
        end
        next if text == segment[:latex]

        validate!(text, segment)
        payload = segment[:mode] == :markdown ? encode_markdown_math(text) : text
        replacement = segment[:opening] + segment[:leading] + payload + segment[:trailing] + segment[:closing]
        segment.slice(:id, :start, :finish, :source).merge(replacement: replacement)
      end
    end

    # line_map maps rendered-input lines to exact, unchanged source lines. A nil
    # map is appropriate only for the original Markdown, before Liquid runs.
    def annotate(root, input: nil, line_map: nil)
      @annotation_map = line_map || exact_line_map(input || @body)
      @annotation_cursor = 0
      @annotated = {}
      visit(root)
      root
    end

    private

    # Single-dollar math is ordinary text to Kramdown. It therefore consumes
    # Markdown escapes before MathJax sees the equation. Project those decoded
    # text leaves back to their exact original byte boundaries so the editor
    # shows the same TeX that the page renders, including escaped dollars.
    def project_text_segments(root)
      @annotation_map = exact_line_map(@body)
      @annotation_cursor = 0
      project_text_node(root)
      @segments.sort_by! { |segment| segment[:body_start] }
    end

    def project_text_node(node, raw_html: false)
      return if BLOCKED_TYPES.include?(node.type)
      return if SkippedContent.node?(node)
      return if node.type == :html_element && BLOCKED_TAGS.include?(node.value.to_s.downcase)

      raw_html = node.options[:content_model] == :raw if node.type == :html_element
      if node.type == :text && node.value.is_a?(String) && !node.value.empty? && (bounds = bounds_for(node))
        matches = text_ranges(node.value, [bounds.first, @annotation_cursor].max, bounds.last)
        if matches.length == 1
          first, finish, inverse = matches.first
          @annotation_cursor = finish
          # The inverse is one-to-one at complete UTF-8 character boundaries.
          original_positions = inverse.invert
          probe = self.class.new(node.value, 0, ast: Kramdown::Element.new(:root), project_text: false)
          projected = probe.instance_variable_get(:@segments).filter_map do |segment|
            left = original_positions[segment[:body_start]]
            right = original_positions[segment[:body_finish]]
            next unless left && right

            source = utf8_slice(left, right)
            interior = source.byteslice(segment[:opening].bytesize...-segment[:closing].bytesize)
            id = "math-#{Digest::SHA256.hexdigest([@body_offset + left, @body_offset + right, source].join(':'))[0, 24]}"
            segment.merge(id: id, start: @body_offset + left, finish: @body_offset + right,
                          body_start: left, body_finish: right, source: source,
                          leading: interior[/\A\s*/], trailing: interior[/\s*\z/],
                          latex: raw_html ? CGI.unescapeHTML(segment[:latex]) : segment[:latex],
                          mode: raw_html ? :html : :markdown)
          end
          @segments.reject! do |segment|
            segment[:body_start] >= first && segment[:body_finish] <= finish &&
              (segment[:delimiter] == '$' || projected.any? { |item| item[:body_start] == segment[:body_start] && item[:body_finish] == segment[:body_finish] })
          end
          @segments.concat(projected)
        end
      end
      node.children.each { |child| project_text_node(child, raw_html: raw_html) }
    end

    def encode_markdown_math(text)
      # Keep commands readable, doubling only slashes that Markdown would
      # consume. Protect Markdown formatting/link punctuation in the payload.
      text.gsub(/\\(?=[\\`*_{}\[\]()<>#+.!|~=$-])/) { '\\\\' }
          .gsub(/(?<!\\)(?:\\\\)*\K[*_`\[\]]/) { |character| "\\#{character}" }
    end

    def escaped?(position)
      count = 0
      cursor = position - 1
      while cursor >= 0 && @binary.getbyte(cursor) == 92
        count += 1
        cursor -= 1
      end
      count.odd?
    end

    def overlap?(start, finish)
      @protected.any? { |left, right| start < right && finish > left }
    end

    def protect_source(root)
      # HTML attributes, comments and raw executable/structural containers are
      # never interpreted as equations. A quoted > does not end an HTML tag.
      @binary.to_enum(:scan, /<!--[\s\S]*?(?:-->|\z)|<\/?[A-Za-z][A-Za-z0-9:-]*(?=[\s\/>])(?:[^>"']|"[^"]*"|'[^']*')*>|<![^>]*>|<\?[^>]*>/n).each do
        match = Regexp.last_match
        @protected << [match.begin(0), match.end(0)]
      end
      ignored = []
      BLOCKED_TAGS.each do |tag|
        @binary.to_enum(:scan, /<#{tag}\b(?:[^>"']|"[^"]*"|'[^']*')*>[\s\S]*?(?:<\/#{tag}\s*>|\z)/in).each do
          match = Regexp.last_match
          @protected << [match.begin(0), match.end(0)]
          ignored << [match.begin(0), match.end(0)]
        end
      end
      @binary.to_enum(:scan, /\{%\s*(comment|raw)\b[\s\S]*?\{%\s*end\1\s*%\}|\{\{[\s\S]*?\}\}|\{%[\s\S]*?%\}/n).each do
        match = Regexp.last_match
        @protected << [match.begin(0), match.end(0)]
      end
      @binary.to_enum(:scan, /!\[(?:\\.|[^\]\r\n])*\](?:\((?:\\.|[^)])*\)|\[(?:\\.|[^\]\r\n])*\])?|\]\((?:\\.|[^)])*\)|^ {0,3}\[[^\]\r\n]+\]:[^\r\n]*/n).each do
        match = Regexp.last_match
        @protected << [match.begin(0), match.end(0)]
      end
      code_start = @protected.length
      protect_fences
      protect_inline_code
      protect_ast_blocks(root)
      ignored.concat(@protected.drop(code_start))
      @protected.concat(SkippedContent.ranges(@body, ignored: ignored))
    end

    def protect_fences
      position = 0
      lines = @binary.lines
      index = 0
      while index < lines.length
        line = lines[index]
        opening = /\A {0,3}(`{3,}|~{3,})[^\r\n]*(?:\r?\n|\z)/n.match(line)
        unless opening
          position += line.bytesize
          index += 1
          next
        end
        first = position
        character = opening[1][0]
        minimum = opening[1].length
        position += line.bytesize
        index += 1
        while index < lines.length
          candidate = lines[index]
          position += candidate.bytesize
          index += 1
          break if candidate.match?(/\A {0,3}#{Regexp.escape(character)}{#{minimum},}[ \t]*(?:\r?\n|\z)/n)
        end
        @protected << [first, position]
      end
    end

    def protect_inline_code
      position = 0
      while (opening = /`+/n.match(@binary, position))
        if escaped?(opening.begin(0)) || overlap?(opening.begin(0), opening.end(0))
          position = opening.end(0)
          next
        end
        closing = /(?<!`)#{Regexp.escape(opening[0])}(?!`)/n.match(@binary, opening.end(0))
        if closing
          @protected << [opening.begin(0), closing.end(0)]
          position = closing.end(0)
        else
          position = opening.end(0)
        end
      end
    end

    def protect_ast_blocks(node)
      if %i[table codeblock].include?(node.type) && node.options[:location].is_a?(Integer)
        line = node.options[:location] - 1
        first = @line_starts[line]
        if first
          ending = line + 1
          ending += 1 while ending < @line_starts.length - 1 && !@binary.byteslice(@line_starts[ending]...@line_starts[ending + 1]).strip.empty?
          @protected << [first, @line_starts[ending] || @binary.bytesize]
        end
        return
      end
      node.children.each { |child| protect_ast_blocks(child) }
    end

    def scan
      result = []
      position = 0
      while position < @binary.bytesize
        opening = if @binary.byteslice(position, 2) == '$$' && !escaped?(position)
                    '$$'
                  elsif @binary.getbyte(position) == 36 && !escaped?(position) && @binary.getbyte(position + 1) && !@binary.byteslice(position + 1, 1).match?(/\s/n)
                    '$'
                  elsif ['\\(', '\\['].include?(@binary.byteslice(position, 2)) && !escaped?(position)
                    @binary.byteslice(position, 2)
                  end
        unless opening && !overlap?(position, position + opening.bytesize)
          position += 1
          next
        end
        closing = { '$' => '$', '$$' => '$$', '\\(' => '\\)', '\\[' => '\\]' }.fetch(opening)
        finish_opening = position + opening.bytesize
        closing_position = find_closing(closing, finish_opening, inline: opening == '$' || opening == '\\(')
        if !closing_position || overlap?(position, closing_position + closing.bytesize)
          position += opening.bytesize
          next
        end
        finish = closing_position + closing.bytesize
        source = @binary.byteslice(position...finish).dup.force_encoding(Encoding::UTF_8)
        interior = @binary.byteslice(finish_opening...closing_position).dup.force_encoding(Encoding::UTF_8)
        leading = interior[/\A\s*/]
        trailing = interior[/\s*\z/]
        latex = interior.strip
        if !latex.empty? && latex.valid_encoding?
          display = opening == '\\[' || (opening == '$$' && block_position?(position, finish))
          id = "math-#{Digest::SHA256.hexdigest([@body_offset + position, @body_offset + finish, source].join(':'))[0, 24]}"
          result << { id: id, latex: latex, display: display, delimiter: opening, opening: opening,
                      closing: closing, leading: leading, trailing: trailing, source: source,
                      start: @body_offset + position, finish: @body_offset + finish,
                      body_start: position, body_finish: finish }
        end
        position = finish
      end
      result
    end

    def find_closing(delimiter, position, inline:)
      cursor = position
      while (candidate = @binary.index(delimiter.b, cursor))
        return nil if inline && @binary.byteslice(position...candidate).include?("\n")
        unless escaped?(candidate)
          return candidate if delimiter != '$'
          # Do not jump across an intervening currency dollar to a later math
          # closer. Every unescaped dollar terminates this candidate.
          return candidate if candidate > position && !@binary.byteslice(candidate - 1, 1).match?(/\s/n) && !@binary.byteslice(candidate + 1, 1).to_s.match?(/[0-9$]/n)
          return nil
        end
        cursor = candidate + delimiter.bytesize
      end
      nil
    end

    def block_position?(first, finish)
      line_start = @binary.rindex("\n", first - 1) if first.positive?
      line_start = line_start ? line_start + 1 : 0
      line_end = @binary.index("\n", finish) || @binary.bytesize
      @binary.byteslice(line_start...first).strip.empty? && @binary.byteslice(finish...line_end).strip.empty?
    end

    def validate!(text, segment)
      unless text.is_a?(String) && text.encoding != Encoding::ASCII_8BIT && text.valid_encoding? && text.bytesize <= MAX_LATEX_BYTES && !text.strip.empty?
        raise InvalidEdit, 'An equation must be nonempty valid UTF-8 and shorter than 100 KB.'
      end
      if text.match?(/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/) || text.include?("\r")
        raise InvalidEdit, 'An equation cannot contain control characters.'
      end
      if !segment[:display] && text.include?("\n")
        raise InvalidEdit, 'Inline equations must stay on one line.'
      end
      if text.include?('{{') || text.include?('{%') || text.match?(/<\/?[A-Za-z][A-Za-z0-9:-]*(?:\s[^>]*|\/?)>|<!(?:--|DOCTYPE)|<\/?(?:script|style|iframe|img|svg)\b/i) || text.match?(/\\(?:href|url|style|class|cssId|htmlClass|htmlId|htmlData|require)\b/)
        raise InvalidEdit, 'Equations cannot contain HTML, template syntax or HTML/link commands.'
      end
      if text.match?(STATE_DEFINING_COMMANDS)
        raise InvalidEdit, 'Keep equations self-contained. Macro, environment and color definitions cannot be edited here.'
      end
      if text.match?(/(?<!\\)(?:\\\\)*\$/) || slash_delimiter?(text)
        raise InvalidEdit, 'Enter only LaTeX without surrounding math delimiters.'
      end
    end

    def slash_delimiter?(text)
      # A TeX row break followed by spacing, \\\\[2pt], has two
      # backslashes and is not a MathJax opening delimiter.
      binary = text.b
      binary.to_enum(:scan, /\\[()\[\]]/n).any? do
        match = Regexp.last_match
        count = 1
        cursor = match.begin(0) - 1
        while cursor >= 0 && binary.getbyte(cursor) == 92
          count += 1
          cursor -= 1
        end
        count.odd?
      end
    end

    def exact_line_map(input)
      original = @body.lines
      rendered = input.lines
      return original.each_index.to_h { |index| [index + 1, index + 1] } if input == @body

      # The caller normally supplies Source's stronger Liquid-aware map. Without
      # it, only an unchanged prefix before any template token is reliable.
      result = {}
      original.each_index do |index|
        break if original[index] != rendered[index] || original[index].include?('{{') || original[index].include?('{%')

        result[index + 1] = index + 1
      end
      result
    end

    def bounds_for(node)
      line = node.options[:location]
      return unless line.is_a?(Integer) && (source_line = @annotation_map[line])

      count = node.value.to_s.count("\n")
      return unless (0..count).all? { |delta| @annotation_map[line + delta] == source_line + delta }

      [@line_starts[source_line - 1], @line_starts[source_line + count] || @binary.bytesize]
    end

    def visit(node)
      return if BLOCKED_TYPES.include?(node.type)
      return if SkippedContent.node?(node)
      return if node.type == :html_element && (BLOCKED_TAGS.include?(node.value.to_s.downcase) || node.attr['data-local-table-id'])

      node.children = node.children.flat_map do |child|
        if child.type == :math
          annotate_native_math(child)
        elsif child.type == :text
          annotate_text(child)
        else
          visit(child)
          child
        end
      end
    end

    def candidates_for(node)
      line = node.options[:location]
      return [] unless line.is_a?(Integer) && (source_line = @annotation_map[line])

      first = @line_starts[source_line - 1]
      finish = @line_starts[source_line] || @binary.bytesize
      @segments.select do |segment|
        # Kramdown trims a display node's value and excludes its delimiter
        # lines. Locate its opening line, then verify every source line covered
        # by the complete original equation rather than the trimmed payload.
        covered = segment[:source].count("\n")
        segment[:body_start] >= [first, @annotation_cursor].max && segment[:body_start] < finish &&
          (0..covered).all? { |delta| @annotation_map[line + delta] == source_line + delta } &&
          !@annotated[segment[:id]]
      end
    end

    def annotate_native_math(node)
      matches = candidates_for(node).select { |segment| segment[:delimiter] == '$$' && segment[:latex] == node.value.to_s.strip }
      return node if matches.empty?

      segment = matches.first
      @annotation_cursor = segment[:body_finish]
      @annotated[segment[:id]] = true
      wrapper = wrapper_for(segment, block: node.options[:category] == :block)
      wrapper.children << node
      wrapper
    end

    def annotate_text(node)
      bounds = bounds_for(node)
      return node unless bounds && node.value.is_a?(String) && !node.value.empty?

      first, finish = bounds
      lower = [first, @annotation_cursor].max
      matches = text_ranges(node.value, lower, finish)
      # Repeated leaves across Markdown markup have only line-level locations;
      # source ownership is not certain enough to make those leaves editable.
      return node unless matches.length == 1

      location, ending, inverse = matches.first
      contained = @segments.select do |segment|
        segment[:body_start] >= location && segment[:body_finish] <= ending && !@annotated[segment[:id]]
      end
      @annotation_cursor = ending
      return node if contained.empty?

      output = []
      cursor = location
      contained.each do |segment|
        # Keep the converter's already-decoded text, including its treatment of
        # Markdown escapes. Only the source patch uses the original LaTeX.
        next unless inverse.key?(segment[:body_start]) && inverse.key?(segment[:body_finish])
        rendered = node.value.b.byteslice(inverse.fetch(segment[:body_start])...inverse.fetch(segment[:body_finish])).force_encoding(Encoding::UTF_8)
        # Ordinary Markdown consumes \( and \[ as punctuation escapes.
        # Those visible parentheses/brackets are not MathJax input and must not
        # advertise a formula editor that would preview a different rendering.
        # Raw HTML captions retain the delimiters and remain eligible.
        next unless rendered.start_with?(segment[:opening]) && rendered.end_with?(segment[:closing])

        output << Kramdown::Element.new(:text, node.value.b.byteslice(inverse.fetch(cursor)...inverse.fetch(segment[:body_start])).force_encoding(Encoding::UTF_8), {}, node.options.dup) if cursor < segment[:body_start]
        wrapper = wrapper_for(segment)
        wrapper.children << Kramdown::Element.new(:text, rendered, {}, node.options.dup)
        output << wrapper
        @annotated[segment[:id]] = true
        cursor = segment[:body_finish]
      end
      output << Kramdown::Element.new(:text, node.value.b.byteslice(inverse.fetch(cursor)...inverse.fetch(ending)).force_encoding(Encoding::UTF_8), {}, node.options.dup) if cursor < ending
      output
    end

    def text_ranges(value, lower, finish)
      punctuation = /[\\`*_{}\[\]()<>#+.!|~=-]/
      pattern = value.each_char.map do |character|
        escaped = Regexp.escape(character.b)
        if character == "\n"
          '\r?\n'
        elsif character.match?(punctuation)
          "(?:\\\\#{escaped}|#{escaped})"
        else
          escaped
        end
      end.join.b
      expression = Regexp.new(pattern, Regexp::NOENCODING)
      matches = []
      position = lower
      while (match = expression.match(@binary, position)) && match.end(0) <= finish
        first = match.begin(0)
        last = match.end(0)
        unless overlap?(first, last)
          decoded = +''
          inverse = { first => 0 }
          raw = utf8_slice(first, last)
          raw_cursor = first
          characters = raw.each_char.to_a
          index = 0
          while index < characters.length
            character = characters[index]
            consumed = character.bytesize
            if raw != value && character == '\\' && characters[index + 1]&.match?(punctuation)
              index += 1
              character = characters[index]
              consumed += character.bytesize
            elsif raw != value && character == "\r" && characters[index + 1] == "\n"
              index += 1
              character = "\n"
              consumed += 1
            end
            decoded << character
            raw_cursor += consumed
            inverse[raw_cursor] = decoded.bytesize
            index += 1
          end
          matches << [first, last, inverse] if decoded == value
        end
        position = match.end(0)
        position += 1 if match.end(0) == match.begin(0)
      end
      matches.uniq { |first, last, _inverse| [first, last] }
    end

    def utf8_slice(first, finish)
      @binary.byteslice(first...finish).dup.force_encoding(Encoding::UTF_8)
    end

    def wrapper_for(segment, block: false)
      Kramdown::Element.new(:html_element, block ? 'div' : 'span',
                           { 'data-local-math-id' => segment[:id], 'class' => 'local-edit-math tex2jax_process' },
                           category: block ? :block : :span, content_model: block ? :block : :span, is_closed: false)
    end
  end
end
