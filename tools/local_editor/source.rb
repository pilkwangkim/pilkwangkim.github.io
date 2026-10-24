# frozen_string_literal: true

require 'cgi'
require 'digest'
require 'date'
require 'fileutils'
require 'json'
require 'kramdown'
require 'kramdown-parser-gfm'
require 'pathname'
require 'psych'
require 'securerandom'
require 'tempfile'
require 'time'
require_relative 'math'
require_relative 'tables'

module LocalEditor
  class Error < StandardError; end
  class Conflict < Error; end
  class InvalidEdit < Error; end
  class InvalidPath < Error; end

  # The editor patches only mapped source ranges. It never serializes a Markdown
  # document or dumps its YAML front matter back to disk.
  class Source
    MAX_TEXT_BYTES = 100_000
    EDITABLE_FIELDS = [%w[title], %w[description], %w[image alt], %w[image caption]].freeze
    METADATA_FIELDS = %w[tags categories topic series series_order date lang].freeze

    attr_reader :path, :front_matter

    def self.with_transaction(root:)
      directory = File.join(File.realpath(root), '.local-editor')
      FileUtils.mkdir_p(directory, mode: 0o700)
      raise InvalidPath, 'The editor data directory cannot be a symlink.' unless File.realpath(directory) == directory

      lock_path = File.join(directory, 'source.lock')
      raise InvalidPath, 'The editor lock cannot be a symlink.' if File.symlink?(lock_path)

      File.open(lock_path, File::RDWR | File::CREAT, 0o600) do |lock|
        lock.flock(File::LOCK_EX)
        yield
      ensure
        lock.flock(File::LOCK_UN)
      end
    end

    def initialize(root:, path:)
      @root = File.realpath(root)
      supplied = Pathname.new(path.to_s)
      raise InvalidPath, 'Choose an existing Markdown post.' if supplied.absolute? || supplied.each_filename.include?('..')

      @path = supplied.cleanpath.to_s
      unless @path.start_with?('_posts/') && @path.match?(/\.markdown\z|\.md\z/i)
        raise InvalidPath, 'Only Markdown files under _posts can be edited.'
      end

      @absolute_path = File.join(@root, @path)
      assert_path!
      reload!
    rescue Errno::ENOENT, Errno::ENOTDIR => e
      raise InvalidPath, e.message
    end

    def document
      {
        path: @path,
        revision: revision,
        segments: @segments.map { |segment| public_segment(segment) },
        fields: @segments.select { |segment| segment[:field] }.to_h do |segment|
          [segment[:field], public_segment(segment)]
        end,
        metadata: { values: metadata_values, writable: @metadata_writable.dup },
        structures: { math: @math_source.document, tables: @table_source.document }
      }
    end

    def metadata_values
      METADATA_FIELDS.to_h do |field|
        value = @front_matter[field]
        normalized = case value
                     when Time, DateTime then value.strftime('%Y-%m-%d %H:%M:%S %z')
                     when Date then value.iso8601
                     else value
                     end
        [field, normalized]
      end
    end

    def revision
      Digest::SHA256.hexdigest(@raw)
    end

    # Called on the normal Jekyll/Kramdown AST immediately before its native
    # converter. Liquid-expanded or otherwise uncertain text is left untouched.
    def annotate(root, input: nil)
      # Jekyll removes the blank lines immediately after the closing front
      # matter delimiter. Align line-only AST locations without touching the
      # converter input, its AST, or any source bytes.
      adjustment = if input.is_a?(String)
                     original_prefix = @body[/\A(?:[ \t]*\r?\n)*/].to_s
                     input_prefix = input[/\A(?:[ \t]*\r?\n)*/].to_s
                     [original_prefix.count("\n") - input_prefix.count("\n"), 0].max
                   else
                     0
                   end
      line_map = if input.is_a?(String)
                     original_lines = @body.lines.drop(adjustment)
                     rendered_lines = input.lines
                     unsafe_line, unsafe_end_line = uncertain_template_bounds
                     indices = if original_lines.length == rendered_lines.length
                                 original_lines.each_index.select do |index|
                                   source_line = index + adjustment + 1
                                   original_lines[index] == rendered_lines[index] && (!unsafe_line || source_line < unsafe_line)
                                 end
                               else
                                 # A changed line count invalidates following
                                 # AST locations even before a later control
                                 # block. Single-line URL substitutions can be
                                 # crossed, but only exact lines become targets.
                                 original_lines.each_index.take_while do |index|
                                   source_line = index + adjustment + 1
                                   matching_url_expanded_line?(original_lines[index], rendered_lines[index]) &&
                                     (!unsafe_line || source_line < unsafe_line)
                                 end.select do |index|
                                   original_lines[index] == rendered_lines[index]
                                 end
                               end
                     mapping = indices.to_h { |index| [index + 1, index + adjustment + 1] }
                     original_index = original_lines.length - 1
                     rendered_index = rendered_lines.length - 1
                     # The literal terminal tail follows the last template
                     # control/output token, so it cannot be generated by that
                     # token. Match it from EOF instead of using the uncertain
                     # middle's line numbers or repeated text occurrences.
                     while original_index >= 0 && rendered_index >= 0 &&
                           original_index + adjustment + 1 > (unsafe_end_line || 0) &&
                           matching_url_expanded_line?(original_lines[original_index], rendered_lines[rendered_index])
                       source_line = original_index + adjustment + 1
                       rendered_line = rendered_index + 1
                       if original_lines[original_index] == rendered_lines[rendered_index]
                         mapping[rendered_line] = if mapping.key?(rendered_line) && mapping[rendered_line] != source_line
                                                    nil
                                                  else
                                                    source_line
                                                  end
                       end
                       original_index -= 1
                       rendered_index -= 1
                     end
                     mapping
                   end
      # Known interactive includes replace complete standalone lines with
      # component-owned sections. Their ordered boundaries can establish the
      # literal islands between several includes without searching for similar
      # prose or trusting generic Liquid output.
      if input.is_a?(String) && (islands = interactive_include_line_map(root, input, adjustment))
        line_map = line_map.merge(islands) { |_line, existing, mapped| existing == mapped ? mapped : nil }
      end
      @table_source.annotate(root, input: input || @body, line_map: line_map)
      @math_source.annotate(root, input: input || @body, line_map: line_map)
      Mapper.new(@body, @body_offset, allowed_ids: @segments.to_h { |segment| [segment[:id], true] },
                 line_adjust: adjustment, line_map: line_map).annotate(root)
      root
    end

    def prepare_save(revision:, edits: [], metadata: {})
      unless edits.is_a?(Array) && edits.length.between?(0, 2000)
        raise InvalidEdit, 'Supply no more than 2,000 text edits.'
      end

      current = read_current!
      assert_revision!(current, revision)

      patches = []
      seen = {}
      math_edits = []
      cell_edits = []
      edits.each do |edit|
        raise InvalidEdit, 'Each edit must name an editable text segment.' unless edit.is_a?(Hash)

        id = edit[:id] || edit['id']
        text = edit.key?(:text) ? edit[:text] : edit['text']
        raise InvalidEdit, 'An edit was supplied more than once.' if seen[id]

        seen[id] = true
        if id.is_a?(String) && id.start_with?('math-')
          math_edits << edit
          next
        elsif id.is_a?(String) && id.start_with?('cell-')
          cell_edits << edit
          next
        end
        segment = @segments.find { |candidate| candidate[:id] == id }
        raise InvalidEdit, 'This text is protected or no longer matches the source. Reload the page.' unless segment

        expected = edit.key?(:original) ? edit[:original] : edit['original']
        if !expected.nil? && expected != segment[:text]
          raise Conflict, 'The original text no longer matches. Reload before saving.'
        end
        next if text == segment[:text]

        validate_text!(text, segment)

        replacement = if segment[:field]
                        encode_scalar(text, segment[:source])
                      elsif segment[:mode] == :html
                        CGI.escapeHTML(text).gsub('$', '&#36;')
                      else
                        encode_markdown(text)
                      end
        patches << segment.merge(replacement: replacement)
      end

      patches.concat(@math_source.patches(math_edits))
      patches.concat(@table_source.patches(cell_edits))

      metadata_patches, metadata_ids = prepare_metadata(metadata)
      edited_ids = patches.map { |patch| patch[:id] } + metadata_ids
      patches.concat(metadata_patches)
      patches.sort_by { |patch| [patch[:start], patch[:finish]] }.each_cons(2) do |left, right|
        if left[:finish] > right[:start]
          raise InvalidEdit, 'Overlapping edits cannot be saved together.'
        end
      end
      result = current.b.dup
      patches.sort_by { |patch| [-patch[:start], -patch[:finish], patch[:append] ? -1 : 0] }.each do |patch|
        actual = result.byteslice(patch[:start], patch[:finish] - patch[:start])
        unless actual == patch[:source].b
          raise Conflict, 'A source range changed. Reload before saving.'
        end
        result[patch[:start]...patch[:finish]] = patch[:replacement].b
      end
      result.force_encoding(Encoding::UTF_8)
      validate_prepared_yaml!(result) unless metadata_ids.empty?
      { before: current, after: result, changed: edited_ids.length, edited_ids: edited_ids, action: 'save' }
    end

    def save(revision:, edits: [], metadata: {})
      with_lock do
        commit_prepared!(prepare_save(revision: revision, edits: edits, metadata: metadata), lock: false)
      end
    end

    def commit_prepared!(prepared, lock: true)
      operation = lambda do
        unless prepared.is_a?(Hash) && prepared[:before].is_a?(String) && prepared[:after].is_a?(String) &&
               prepared[:changed].is_a?(Integer) && prepared[:changed] >= 0 && prepared[:edited_ids].is_a?(Array) &&
               %w[save undo].include?(prepared[:action].to_s)
          raise InvalidEdit, 'The prepared source change is invalid.'
        end
        current = read_current!
        assert_revision!(current, Digest::SHA256.hexdigest(prepared[:before]))
        next({ document: document, receipt: nil, changed: 0 }) if current == prepared[:after]

        receipt = write_with_backup!(current, prepared[:after], prepared[:edited_ids], prepared[:action].to_s)
        reload!
        { document: document, receipt: receipt, changed: prepared[:changed] }
      end
      lock ? with_lock(&operation) : operation.call
    end

    def prepare_undo(revision:, receipt:)
      unless receipt.is_a?(String) && receipt.match?(/\A[a-zA-Z0-9-]{12,100}\z/)
        raise InvalidEdit, 'Invalid undo receipt.'
      end

      current = read_current!
      assert_revision!(current, revision)
      directory = backup_directory
      record = JSON.parse(File.read(File.join(directory, "#{receipt}.json")))
      unless record['path'] == @path && record['after'] == Digest::SHA256.hexdigest(current)
        raise Conflict, 'Only the latest matching save can be undone. Reload the page.'
      end
      backup = File.binread(File.join(directory, "#{receipt}.md"))
      unless Digest::SHA256.hexdigest(backup) == record['before']
        raise Conflict, 'The saved backup does not match its receipt.'
      end
      { before: current, after: backup.force_encoding(Encoding::UTF_8), changed: 1, edited_ids: [], action: 'undo' }
    rescue Errno::ENOENT, JSON::ParserError
      raise InvalidEdit, 'The undo receipt or its backup could not be found.'
    end

    def undo(revision:, receipt:)
      with_lock { commit_prepared!(prepare_undo(revision: revision, receipt: receipt), lock: false) }
    end

    def preview_table(table_id:, edits: [])
      @table_source.preview(table_id, edits)
    end

    private

    def interactive_include_line_map(root, input, adjustment)
      original_lines = @body.lines
      includes = original_lines.each_with_index.filter_map do |line, index|
        descriptor = interactive_include_descriptor(line)
        descriptor&.merge(line: index, token: line.strip)
      end
      return if includes.empty?

      # A loop or conditional must not cross any component boundary. Closed
      # template regions before/after the complete component span can retain
      # their existing fail-closed mapping without locking its literal middle.
      prefix = []
      suffix = []
      known = includes.map { |item| item[:token] }
      @body.to_enum(:scan, /\{\{.*?\}\}|\{%.*?%\}/m).each do
        match = Regexp.last_match
        token = match[0]
        next if single_line_url_token?(token)

        first_line = @body[0...match.begin(0)].count("\n")
        last_line = @body[0...match.end(0)].count("\n")
        if (descriptor = includes.find { |item| item[:line] == first_line && item[:token] == token })
          known.delete_at(known.index(descriptor[:token]))
        elsif last_line < includes.first[:line]
          prefix << { token: token, first: first_line, last: last_line }
        elsif first_line > includes.last[:line]
          suffix << { token: token, first: first_line, last: last_line }
        else
          return
        end
      end
      return unless known.empty? && closed_template_region?(prefix) && closed_template_region?(suffix)

      rendered_lines = input.lines
      ranges = SkippedContent.ranges(input)
      components = root.children.select { |node| SkippedContent.node?(node) }
      return unless ranges.length == includes.length && components.length == includes.length

      rendered_bounds = ranges.sort_by(&:first).zip(includes, components).map do |(first, finish), descriptor, node|
        return unless node.value.to_s.downcase == 'section'
        return unless interactive_component_matches?(node.attr, descriptor)

        first_line = input.b.byteslice(0...first).count("\n")
        last_line = input.b.byteslice(0...finish).count("\n")
        return unless node.options[:location] == first_line + 1

        line_start = rendered_lines.take(first_line).sum(&:bytesize)
        line_finish = rendered_lines.take(last_line + 1).sum(&:bytesize)
        return unless input.b.byteslice(line_start...first).match?(/\A[ \t]*\z/n)
        return unless input.b.byteslice(finish...line_finish).match?(/\A[ \t]*(?:\r?\n)?\z/n)
        return unless input.b.byteslice(first...finish).match?(%r{</section\s*>\z}in)

        [first_line, last_line + 1]
      end

      mapping = {}
      source_cursor = adjustment
      rendered_cursor = 0
      # Each entire island must agree, allowing only the same narrow URL
      # substitutions as the existing prefix/suffix mapper. Leading/trailing
      # blank lines belong to include formatting and never identify text.
      (includes + [nil]).each_with_index do |descriptor, index|
        source_finish = descriptor ? descriptor[:line] : original_lines.length
        rendered_finish = descriptor ? rendered_bounds[index].first : rendered_lines.length
        source_cursor = [source_cursor, prefix.last[:last] + 1].max if index.zero? && prefix.any?
        source_finish = [source_finish, suffix.first[:first]].min if !descriptor && suffix.any?
        source_rows = (source_cursor...source_finish).to_a
        rendered_rows = (rendered_cursor...rendered_finish).to_a
        source_rows.shift while source_rows.any? && original_lines[source_rows.first].strip.empty?
        source_rows.pop while source_rows.any? && original_lines[source_rows.last].strip.empty?
        rendered_rows.shift while rendered_rows.any? && rendered_lines[rendered_rows.first].strip.empty?
        rendered_rows.pop while rendered_rows.any? && rendered_lines[rendered_rows.last].strip.empty?
        rendered_rows = source_rows.empty? ? [] : rendered_rows.last(source_rows.length) if index.zero? && prefix.any?
        rendered_rows = rendered_rows.first(source_rows.length) if !descriptor && suffix.any?
        return unless source_rows.length == rendered_rows.length
        return unless source_rows.zip(rendered_rows).all? do |source_row, rendered_row|
          matching_url_expanded_line?(original_lines[source_row], rendered_lines[rendered_row])
        end

        source_rows.zip(rendered_rows).each do |source_row, rendered_row|
          mapping[rendered_row + 1] = source_row + 1 if original_lines[source_row] == rendered_lines[rendered_row]
        end
        next unless descriptor

        source_cursor = descriptor[:line] + 1
        rendered_cursor = rendered_bounds[index].last
      end
      mapping
    end

    def closed_template_region?(tokens)
      stack = []
      blocks = %w[if unless case for tablerow capture comment raw ifchanged]
      singles = %w[assign echo include render cycle increment decrement break continue]
      tokens.all? do |entry|
        token = entry[:token]
        next true if token.start_with?('{{')

        name = token[/\A\{%-?\s*([a-z_]+)/, 1]
        if blocks.include?(name)
          stack << name
          true
        elsif name&.start_with?('end')
          stack.pop == name.delete_prefix('end')
        elsif %w[else elsif when].include?(name)
          !stack.empty?
        else
          singles.include?(name)
        end
      end && stack.empty?
    end

    def interactive_include_descriptor(line)
      match = line.match(/\A[ \t]*\{%\s*include\s+interactive\/(article-lab|negf-barrier)\.html([^\r\n{}]*?)%\}[ \t]*(?:\r?\n|\z)\z/)
      return unless match

      kind, remaining = match.captures
      values = {}
      until remaining.strip.empty?
        parameter = remaining.match(/\A\s+(id|key|preset)=(["'])([A-Za-z0-9_.-]*)\2/)
        return unless parameter && !values.key?(parameter[1])

        values[parameter[1]] = parameter[3]
        remaining = remaining[parameter.end(0)..]
      end
      return if kind == 'negf-barrier' && !values.empty?
      return if kind == 'article-lab' && (values['id'].to_s.empty? || (values.key?('key') && values['key'].empty?))

      { kind: kind, values: values }
    end

    def interactive_component_matches?(attributes, descriptor)
      values = descriptor[:values]
      if descriptor[:kind] == 'article-lab'
        attributes['class'].to_s.split.include?('article-lab') &&
          attributes['data-article-lab'] == values['id'] &&
          attributes['id'] == "lab-#{values['key'] || values['id']}" &&
          attributes['data-preset'].to_s == values.fetch('preset', '')
      else
        attributes['class'].to_s.split.include?('negf-lab') && attributes.key?('data-negf-lab')
      end
    end

    # URL outputs preserve the line count, so their surrounding literal bytes
    # can anchor adjacent unchanged lines. Never map the expanded line itself:
    # its changed columns and generated text still have no editable source range.
    def matching_url_expanded_line?(original, rendered)
      return true if original == rendered
      return false unless original.is_a?(String) && rendered.is_a?(String)

      position = 0
      pattern = +'\A'
      tokens = 0
      original.to_enum(:scan, /\{\{.*?\}\}|\{%.*?%\}/m).each do
        match = Regexp.last_match
        return false unless single_line_url_token?(match[0])

        pattern << Regexp.escape(original[position...match.begin(0)])
        # Empty baseurl is valid. Spaces, markup, another Liquid token, and
        # newlines are not URL substitutions accepted by this narrow mapper.
        pattern << %q([^\s<>"'{}]*)
        position = match.end(0)
        tokens += 1
      end
      return false if tokens.zero?

      pattern << Regexp.escape(original[position..]) << '\z'
      Regexp.new(pattern).match?(rendered)
    end

    def single_line_url_token?(token)
      return false if token.include?("\n") || token.include?("\r") || token.match?(/\{[{%]-|-[%}]\}/)

      token.match?(/\A\{\{\s*site\.(?:baseurl|url)\s*\}\}\z/) ||
        token.match?(/\A\{%\s*(?:post_url|link)\s+[^{}\r\n]+%\}\z/)
    end

    def uncertain_template_bounds
      first_line = nil
      last_line = nil
      @body.to_enum(:scan, /\{\{.*?\}\}|\{%.*?%\}/m).each do
        match = Regexp.last_match
        token = match[0]
        whitespace_control = token.match?(/\{[{%]-|-[%}]\}/)
        safe_output = token.match?(/\A\{\{\s*site\.(?:baseurl|url)\s*\}\}\z/)
        safe_tag = token.match?(/\A\{%\s*(?:post_url|link)\b/)
        next if (safe_output || safe_tag) && !token.include?("\n") && !whitespace_control

        # Unknown output may itself contain newlines. A later control block can
        # compensate for those lines, so equal total counts do not prove that
        # the remaining source positions still correspond. URL substitutions
        # before this boundary retain their own unchanged prose lines.
        line = @body[0...match.begin(0)].count("\n") + 1
        beginning = whitespace_control ? [line - 1, 1].max : line
        first_line = [first_line || beginning, beginning].min
        last_line = @body[0...match.end(0)].count("\n") + 1
      end
      [first_line, last_line]
    end

    def assert_path!
      posts = File.realpath(File.join(@root, '_posts'))
      absolute = File.realpath(@absolute_path)
      unless absolute.start_with?("#{posts}/") && absolute == @absolute_path && File.file?(absolute)
        raise InvalidPath, 'Symlinks and paths outside _posts cannot be edited.'
      end
    end

    def read_current!
      assert_path!
      value = File.binread(@absolute_path).force_encoding(Encoding::UTF_8)
      raise InvalidEdit, 'The post is not valid UTF-8.' unless value.valid_encoding?

      value
    end

    def reload!
      @raw = read_current!
      match = @raw.match(/\A---[ \t]*\r?\n(.*?)\r?\n---[ \t]*(?:\r?\n|\z)/m)
      raise InvalidEdit, 'The post must begin with YAML front matter.' unless match

      @body_offset = match[0].bytesize
      @body = @raw.byteslice(@body_offset..)
      yaml_offset = @raw[0...match.begin(1)].bytesize
      @yaml = match[1]
      @yaml_offset = yaml_offset
      @yaml_finish = yaml_offset + @yaml.bytesize
      @newline = @raw[/\A---[ \t]*(\r?\n)/, 1]
      @yaml_document = Psych.parse(@yaml)
      @front_matter = Psych.safe_load(@yaml, permitted_classes: [Date, Time], aliases: true)
      unless @front_matter.is_a?(Hash) && @yaml_document&.root.is_a?(Psych::Nodes::Mapping)
        raise InvalidEdit, 'The post front matter must be a YAML mapping.'
      end
      @yaml_lines = @yaml.lines
      @yaml_starts = [0]
      @yaml_lines.each { |line| @yaml_starts << @yaml_starts.last + line.bytesize }
      @metadata_nodes = {}
      root = @yaml_document.root
      pairs = root.children.each_slice(2).to_a
      names = pairs.map { |key, _| key.is_a?(Psych::Nodes::Scalar) ? key.value : nil }
      uncertain_root = root.style == Psych::Nodes::Mapping::FLOW || !root.anchor.nil? ||
                       names.include?(nil) || names.include?('<<') || names.uniq.length != names.length
      @metadata_writable = METADATA_FIELDS.to_h do |field|
        pair = pairs.find { |key, _| key.is_a?(Psych::Nodes::Scalar) && key.value == field }
        @metadata_nodes[field] = pair if pair
        [field, !uncertain_root && (!pair || pair.none? { |node| yaml_reference?(node) })]
      end
      @segments = scalar_segments(match[1], yaml_offset)
      root = Kramdown::Document.new(@body, input: 'GFM', hard_wrap: false).root
      @table_source = TableSource.new(@body, @body_offset, ast: root)
      @math_source = MathSource.new(@body, @body_offset, ast: root)
      mapper = Mapper.new(@body, @body_offset)
      mapper.annotate(root)
      @segments.concat(mapper.segments)
      @segments.sort_by! { |segment| segment[:start] }
      @segments.each_cons(2) do |left, right|
        raise InvalidEdit, 'Overlapping editable ranges were detected.' if left[:finish] > right[:start]
      end
    rescue Psych::Exception, Kramdown::Error => e
      raise InvalidEdit, "The post could not be parsed: #{e.message}"
    end

    def yaml_reference?(node)
      uncertain_mapping = if node.is_a?(Psych::Nodes::Mapping)
                            keys = node.children.each_slice(2).map do |key, _|
                              key.is_a?(Psych::Nodes::Scalar) ? key.value : nil
                            end
                            keys.include?(nil) || keys.include?('<<') || keys.uniq.length != keys.length
                          end
      uncertain_mapping || node.is_a?(Psych::Nodes::Alias) || (node.respond_to?(:anchor) && !node.anchor.nil?) ||
        (node.respond_to?(:children) && node.children&.any? { |child| yaml_reference?(child) })
    end

    def yaml_range(node)
      from = @yaml_starts.fetch(node.start_line) + @yaml_lines.fetch(node.start_line, '')[0...node.start_column].bytesize
      to = @yaml_starts.fetch(node.end_line) + @yaml_lines.fetch(node.end_line, '')[0...node.end_column].bytesize
      [from, to]
    end

    def metadata_string!(value, field)
      unless value.is_a?(String) && value.valid_encoding? && value.encoding != Encoding::ASCII_8BIT &&
             value.bytesize.between?(1, 1000) && !value.strip.empty? &&
             !value.match?(/[\x00-\x1F\x7F\u0085\u2028\u2029]/)
        raise InvalidEdit, "#{field} must contain plain text on one line."
      end
      value
    end

    def validate_metadata_value!(field, value)
      case field
      when 'tags', 'categories'
        unless value.is_a?(Array) && value.length <= 500
          raise InvalidEdit, "#{field} must be a list of text values."
        end
        value.each { |item| metadata_string!(item, field) }
      when 'series', 'series_order'
        return if value.nil?

        if field == 'series_order'
          unless value.is_a?(Integer) && value >= 0
            raise InvalidEdit, 'Series order must be an integer of zero or greater.'
          end
        else
          metadata_string!(value, field)
        end
      when 'date'
        metadata_string!(value, field)
        unless value.match?(/\A\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} [+-]\d{4}\z/) &&
               value[-4, 2].to_i <= 23 && value[-2, 2].to_i <= 59
          raise InvalidEdit, 'Date must include a valid date, time, and explicit UTC offset.'
        end
        begin
          parsed = DateTime.strptime(value, '%Y-%m-%d %H:%M:%S %z')
          raise ArgumentError unless parsed.strftime('%Y-%m-%d %H:%M:%S %z') == value
        rescue ArgumentError
          raise InvalidEdit, 'Date must include a valid date, time, and explicit UTC offset.'
        end
      when 'lang'
        raise InvalidEdit, 'Article language must be en or ko.' unless %w[en ko].include?(value)
      else
        metadata_string!(value, field)
      end
    end

    def prepare_metadata(metadata)
      raise InvalidEdit, 'Metadata must be an object of editable fields.' unless metadata.is_a?(Hash)

      patches = []
      insertions = []
      ids = []
      seen = {}
      metadata.each do |key, value|
        field = key.to_s
        unless METADATA_FIELDS.include?(field) && !seen[field]
          raise InvalidEdit, 'Only tags, categories, topic, series, series order, date, and language can be edited.'
        end
        seen[field] = true
        raise InvalidEdit, "#{field} uses protected YAML syntax and must be edited in the source." unless @metadata_writable[field]

        validate_metadata_value!(field, value)
        next if metadata_values[field] == value

        pair = @metadata_nodes[field]
        next if pair.nil? && value.nil?

        ids << "metadata:#{field}"
        if pair
          key_node, node = pair
          from, to = yaml_range(node)
          from = yaml_range(key_node).first if value.nil?
          source = @yaml.byteslice(from...to)
          replacement = value.nil? ? '' : encode_metadata(value)
          if from == to && !replacement.empty? && @yaml.byteslice(0...from).end_with?(':')
            replacement = " #{replacement}"
          end
          if !replacement.empty? && node.start_line > key_node.start_line && node.start_column.zero?
            # YAML permits an unindented block sequence after a mapping key,
            # but its replacement flow sequence must be indented.
            replacement = "  #{replacement}"
          end
          # Block sequences include their item comments in the AST token
          # range. Keep those comments as standalone YAML comments when their
          # values are replaced. Inline trailing comments lie outside the
          # token range and stay in their exact original position.
          comments = metadata_comments(node, from, to)
          ending = source.end_with?("\r\n") ? "\r\n" : (source.end_with?("\n") ? "\n" : '')
          replacement += ending unless replacement.empty?
          unless comments.empty?
            replacement += @newline unless replacement.empty? || replacement.end_with?("\n")
            replacement += comments.join
          end
          patches << { start: @yaml_offset + from, finish: @yaml_offset + to,
                       source: source, replacement: replacement }
        else
          insertions << "#{field}: #{encode_metadata(value)}"
        end
      end
      unless insertions.empty?
        patches << { start: @yaml_finish, finish: @yaml_finish, source: '',
                     replacement: @newline + insertions.join(@newline), append: true }
      end
      [patches, ids]
    end

    def encode_metadata(value)
      return value.to_s if value.is_a?(Integer)

      JSON.generate(value).gsub(/[\u0085\u2028\u2029]/) { |character| format('\\u%04x', character.ord) }
    end

    def metadata_comments(node, from, to)
      if node.is_a?(Psych::Nodes::Scalar) &&
         [Psych::Nodes::Scalar::LITERAL, Psych::Nodes::Scalar::FOLDED].include?(node.style)
        left, right = yaml_range(node)
        header = @yaml.byteslice(left...right).lines.first
        comment = header&.match(/(?<!\S)#/)
        return comment ? [(' ' * node.start_column) + header[comment.begin(0)..]] : []
      end
      return [] unless node.is_a?(Psych::Nodes::Sequence) || node.is_a?(Psych::Nodes::Mapping)

      scalar_ranges = []
      visit = lambda do |child|
        if child.is_a?(Psych::Nodes::Scalar)
          scalar_ranges << yaml_range(child)
        else
          child.children&.each { |nested| visit.call(nested) }
        end
      end
      visit.call(node)
      comments = []
      cursor = from
      @yaml.byteslice(from...to).each_line do |line|
        line.to_enum(:scan, /(?<!\S)#/).each do
          match = Regexp.last_match
          offset = cursor + line[0...match.begin(0)].bytesize
          next if scalar_ranges.any? { |left, right| offset >= left && offset < right }

          prefix = line[0...match.begin(0)]
          indentation = prefix.match?(/\A[ \t]*\z/) ? prefix : (' ' * node.start_column)
          comments << indentation + line[match.begin(0)..]
          break
        end
        cursor += line.bytesize
      end
      comments
    end

    def validate_prepared_yaml!(raw)
      match = raw.match(/\A---[ \t]*\r?\n(.*?)\r?\n---[ \t]*(?:\r?\n|\z)/m)
      raise InvalidEdit, 'The edited front matter is invalid.' unless match

      values = Psych.safe_load(match[1], permitted_classes: [Date, Time], aliases: true)
      raise InvalidEdit, 'The edited front matter must remain a mapping.' unless values.is_a?(Hash)
    rescue Psych::Exception => error
      raise InvalidEdit, "The edited front matter could not be parsed: #{error.message}"
    end

    def scalar_segments(yaml, offset)
      document = Psych.parse(yaml)
      return [] unless document&.root.is_a?(Psych::Nodes::Mapping)

      lines = yaml.lines
      starts = [0]
      lines.each { |line| starts << starts.last + line.bytesize }
      EDITABLE_FIELDS.filter_map do |field_path|
        node = document.root
        field_path.each do |part|
          break node = nil unless node.is_a?(Psych::Nodes::Mapping)

          pairs = node.children.each_slice(2).select { |key, _| key.is_a?(Psych::Nodes::Scalar) && key.value == part }
          break node = nil unless pairs.length == 1

          node = pairs.first.last
        end
        # Replacing an anchored token would remove its anchor and can leave
        # aliases elsewhere in the front matter dangling. Such metadata stays
        # protected; changing it requires an explicit YAML edit.
        next unless node.is_a?(Psych::Nodes::Scalar) && node.anchor.nil?

        from = starts.fetch(node.start_line) + lines.fetch(node.start_line, '')[0...node.start_column].bytesize
        to = starts.fetch(node.end_line) + lines.fetch(node.end_line, '')[0...node.end_column].bytesize
        source = yaml.byteslice(from...to)
        field = field_path.join('.')
        { id: "field:#{field}", field: field, kind: 'metadata', text: node.value,
          start: offset + from, finish: offset + to, source: source, mode: :yaml }
      end
    end

    def public_segment(segment)
      segment.slice(:id, :text, :kind, :field)
    end

    def validate_text!(text, segment)
      unless text.is_a?(String) && text.encoding != Encoding::ASCII_8BIT && text.valid_encoding? && text.bytesize <= MAX_TEXT_BYTES
        raise InvalidEdit, 'Edited text must be valid UTF-8 and shorter than 100 KB.'
      end
      raise InvalidEdit, 'Text cannot contain control characters.' if text.match?(/[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]/)
      if text.include?("\n") || text.include?("\r")
        raise InvalidEdit, 'Edit one paragraph or caption at a time without adding line breaks.'
      end
      if segment[:field] == 'title' && text.strip.empty?
        raise InvalidEdit, 'The article title cannot be empty.'
      end
      if segment[:field] && text.match?(/[<>]/)
        raise InvalidEdit, 'Titles and cover metadata must be plain text without HTML angle brackets.'
      end
      if text.include?('{{') || text.include?('{%')
        raise InvalidEdit, 'Liquid template syntax cannot be inserted into article text.'
      end
      if text.match?(/\\[\(\[]/) || text.scan(/(?<!\\)\$/).length > 1
        raise InvalidEdit, 'Math is protected. Edit the surrounding prose without adding math delimiters.'
      end
    end

    def encode_scalar(text, source)
      ending = source.end_with?("\r\n") ? "\r\n" : (source.end_with?("\n") ? "\n" : '')
      # YAML folds a literal NEL into a space even inside a JSON-style quoted
      # scalar. Explicit escapes preserve all Unicode line separators exactly.
      JSON.generate(text).gsub(/[\u0085\u2028\u2029]/) { |character| format('\\u%04x', character.ord) } + ending
    end

    def encode_markdown(text)
      CGI.escapeHTML(text).gsub(/[\\`*_{}#+.!|~=-]/) { |character| "\\#{character}" }
         .gsub('[', '&#91;').gsub(']', '&#93;').gsub('$', '&#36;')
    end

    def assert_revision!(current, expected)
      unless expected.is_a?(String) && expected == Digest::SHA256.hexdigest(current) && expected == revision
        raise Conflict, 'The file changed after this page was loaded. Reload before saving.'
      end
    end

    def local_directory
      directory = File.join(@root, '.local-editor')
      FileUtils.mkdir_p(directory, mode: 0o700)
      raise InvalidPath, 'The editor data directory cannot be a symlink.' unless File.realpath(directory) == directory

      directory
    end

    def backup_directory
      directory = File.join(local_directory, 'backups')
      FileUtils.mkdir_p(directory, mode: 0o700)
      raise InvalidPath, 'The backup directory cannot be a symlink.' unless File.realpath(directory) == directory

      directory
    end

    def with_lock
      self.class.with_transaction(root: @root) { yield }
    end

    def write_with_backup!(before, after, edited_ids, action)
      assert_path!
      receipt = "#{Time.now.utc.strftime('%Y%m%dT%H%M%S')}-#{SecureRandom.hex(12)}"
      directory = backup_directory
      File.binwrite(File.join(directory, "#{receipt}.md"), before)
      File.chmod(0o600, File.join(directory, "#{receipt}.md"))
      record = { path: @path, before: Digest::SHA256.hexdigest(before), after: Digest::SHA256.hexdigest(after),
                 edited_ids: edited_ids, action: action, created_at: Time.now.utc.strftime('%Y-%m-%dT%H:%M:%SZ') }
      File.write(File.join(directory, "#{receipt}.json"), JSON.pretty_generate(record), mode: 'w', perm: 0o600)
      mode = File.stat(@absolute_path).mode & 0o777
      Tempfile.create(['.local-edit-', '.tmp'], File.dirname(@absolute_path)) do |temporary|
        temporary.binmode
        temporary.write(after)
        temporary.flush
        temporary.fsync
        File.chmod(mode, temporary.path)
        assert_path!
        unless Digest::SHA256.file(@absolute_path).hexdigest == record[:before]
          raise Conflict, 'The file changed while saving. Reload before saving again.'
        end
        File.rename(temporary.path, @absolute_path)
      end
      receipt
    end
  end

  class Mapper
    attr_reader :segments

    BLOCKED_TYPES = %i[table codeblock codespan math raw comment img footnote].freeze
    BLOCKED_TAGS = %w[table thead tbody tfoot tr th td code pre script style svg math iframe textarea].freeze
    CAPTION_TAGS = %w[figcaption].freeze
    MATH = /(?<!\\)\$\$.*?(?<!\\)\$\$|(?<!\\)\$(?!\s).*?(?<!\\)\$|\\\(.*?\\\)|\\\[.*?\\\]/m

    def initialize(body, body_offset, allowed_ids: nil, line_adjust: 0, line_map: nil)
      @body = body
      @binary = body.b
      @body_offset = body_offset
      @allowed_ids = allowed_ids
      @line_adjust = line_adjust
      @line_map = line_map
      @segments = []
      @cursor = 0
      @lines = [0]
      body.each_line { |line| @lines << @lines.last + line.bytesize }
      @forbidden = []
      body.to_enum(:scan, /<[A-Za-z!\/?][^>]*>|\{\{.*?\}\}|\{%.*?%\}|\]\((?:\\.|[^)])*\)|^ {0,3}\[[^\]]+\]:[^\r\n]*/m).each do
        match = Regexp.last_match
        @forbidden << [body[0...match.begin(0)].bytesize, body[0...match.end(0)].bytesize]
      end
      # AST locations identify lines rather than columns. Protected inline
      # constructs must therefore be excluded before searching a repeated label
      # on that line (for example `word` **word**).
      @locked = []
      body.to_enum(:scan, /!\[(?:\\.|[^\]\r\n])*\](?:\((?:\\.|[^)])*\)|\[(?:\\.|[^\]\r\n])*\])?|\[\^[^\]\r\n]*\]/m).each do
        match = Regexp.last_match
        @locked << [body[0...match.begin(0)].bytesize, body[0...match.end(0)].bytesize]
      end
      lock_code_ranges
      body.to_enum(:scan, /<(table|thead|tbody|tfoot|tr|th|td|code|pre|script|style|svg|math|iframe|textarea)\b[^>]*>.*?<\/\1\s*>/im).each do
        match = Regexp.last_match
        @locked << [body[0...match.begin(0)].bytesize, body[0...match.end(0)].bytesize]
      end
      @locked.concat(SkippedContent.ranges(body, ignored: @locked))
      @math_ranges = []
      body.to_enum(:scan, MATH).each do
        match = Regexp.last_match
        @math_ranges << [body[0...match.begin(0)].bytesize, body[0...match.end(0)].bytesize]
      end
    end

    def annotate(root)
      visit(root, { kind: 'body', html: false, permitted: true })
      root
    end

    private

    def visit(node, context)
      return if BLOCKED_TYPES.include?(node.type)
      return if SkippedContent.node?(node)
      return if node.type == :html_element && node.attr['data-local-math-id']

      next_context = context.dup
      if node.type == :header
        next_context[:kind] = 'heading'
      elsif node.type == :html_element
        tag = node.value.to_s.downcase
        return if BLOCKED_TAGS.include?(tag)

        next_context[:html] = node.options[:content_model] == :raw || context[:html]
        if CAPTION_TAGS.include?(tag) || node.attr['class'].to_s.split.any? { |name| name.include?('caption') }
          next_context[:kind] = 'caption'
        end
      elsif node.type == :p && node.children.length == 1 && node.children.first.type == :em
        next_context[:kind] = 'caption'
      end

      node.children = node.children.flat_map do |child|
        if child.type == :text
          map_text(child, next_context)
        else
          visit(child, next_context)
          child
        end
      end
    end

    def map_text(node, context)
      value = node.value
      line = node.options[:location]
      return node unless value.is_a?(String) && value.match?(/[\p{L}\p{N}]/) && line.is_a?(Integer) && line.positive?
      return node if value.include?('{{') || value.include?('{%')

      if @line_map
        source_line = @line_map[line]
        return node unless source_line
        return node unless (0..value.count("\n")).all? { |delta| @line_map[line + delta] == source_line + delta }

        line = source_line
      else
        line += @line_adjust
      end

      from_line = @lines[line - 1]
      return node unless from_line

      end_line = line + value.count("\n")
      upper = @lines[end_line] || @binary.bytesize
      lower = [from_line, @cursor].max
      range = find_range(value, lower, upper, html: context[:html])
      return node unless range

      from, finish = range
      @cursor = finish
      raw = @body.byteslice(from...finish)
      # Escaped Markdown and entities are admitted only when their exact decoded
      # text agrees. A mixed math/text leaf uses literal offsets exclusively.
      if raw != value && (value.match?(MATH) || raw.match?(MATH) || value.include?("\n"))
        return node
      end
      chunks = []
      position = 0
      value.to_enum(:scan, /#{MATH}|\r?\n/m).each do
        match = Regexp.last_match
        chunks << [position, match.begin(0), false] if match.begin(0) > position
        chunks << [match.begin(0), match.end(0), true]
        position = match.end(0)
      end
      chunks << [position, value.length, false] if position < value.length
      return node if chunks.empty?

      chunks.map do |start_character, end_character, protected_math|
        text = value[start_character...end_character]
        child = Kramdown::Element.new(:text, text, {}, node.options.dup)
        next child if protected_math || !text.match?(/[\p{L}\p{N}]/)

        start_offset = from + value[0...start_character].bytesize
        end_offset = from + value[0...end_character].bytesize
        if chunks.length == 1
          start_offset = from
          end_offset = finish
        end
        source = @body.byteslice(start_offset...end_offset)
        next child if overlaps?(@locked + @math_ranges, start_offset, end_offset)
        id = "text-#{Digest::SHA256.hexdigest([@body_offset + start_offset, @body_offset + end_offset, source].join(':'))[0, 24]}"
        visible = context[:html] ? CGI.unescapeHTML(text) : text
        segment = { id: id, text: visible, kind: context[:kind], start: @body_offset + start_offset,
                    finish: @body_offset + end_offset, source: source, mode: context[:html] ? :html : :markdown }
        next child if @allowed_ids && !@allowed_ids[id]

        @segments << segment
        wrapper = Kramdown::Element.new(:html_element, 'span',
                                       { 'data-local-edit-id' => id, 'class' => 'local-edit-text tex2jax_ignore' },
                                       category: :span, content_model: :span, is_closed: false)
        wrapper.children << child
        wrapper
      end
    end

    def find_range(value, lower, upper, html:)
      return if lower >= upper

      candidates = []
      offset = @binary.index(value.b, lower)
      while offset && offset + value.bytesize <= upper
        before = offset.positive? ? @binary.getbyte(offset - 1) : nil
        candidates << [offset, offset + value.bytesize] unless before == 92
        offset = @binary.index(value.b, offset + 1)
      end
      unless html
        pattern = value.each_char.map do |character|
          escaped = Regexp.escape(character.b)
          if character == "\n"
            '\r?\n'
          elsif character.match?(/[\\`*_{}\[\]()<>#+.!|~=-]/)
            "(?:\\\\#{escaped}|#{escaped})"
          else
            escaped
          end
        end.join.b
        expression = Regexp.new(pattern, Regexp::NOENCODING)
        location = lower
        while (match = expression.match(@binary, location)) && match.end(0) <= upper
          candidates << [match.begin(0), match.end(0)]
          location = match.end(0)
        end
      end
      protected_ranges = @forbidden + @locked + (value.match?(MATH) ? [] : @math_ranges)
      valid = candidates.uniq.sort_by { |start_offset, end_offset| [start_offset, -(end_offset - start_offset)] }.reject do |start_offset, end_offset|
        overlaps?(protected_ranges, start_offset, end_offset)
      end
      # A line-level AST location cannot establish which of several identical
      # source occurrences produced this node. Fail closed instead of guessing.
      return if valid.map(&:first).uniq.length > 1

      valid.first
    end

    def overlaps?(ranges, start_offset, end_offset)
      ranges.any? { |left, right| start_offset < right && end_offset > left }
    end

    def lock_code_ranges
      position = 0
      while (match = /`+/.match(@binary, position))
        opening = match[0]
        closing = /(?<!`)#{Regexp.escape(opening)}(?!`)/.match(@binary, match.end(0))
        if closing
          @locked << [match.begin(0), closing.end(0)]
          position = closing.end(0)
        else
          position = match.end(0)
        end
      end
      # Tilde-fenced blocks contain no backticks to find in the pass above.
      @body.to_enum(:scan, /^ {0,3}(~{3,})[^\r\n]*\r?\n.*?^ {0,3}\1~*[ \t]*(?:\r?\n|\z)/m).each do
        match = Regexp.last_match
        @locked << [@body[0...match.begin(0)].bytesize, @body[0...match.end(0)].bytesize]
      end
    end
  end
end
