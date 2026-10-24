#!/usr/bin/env ruby
# frozen_string_literal: true

# Compare a local editor build with the normal production build of the same posts.
# Neither this check nor its Source instances write source files.
require 'json'
require 'nokogiri'
require 'optparse'
require 'uri'
require_relative 'source'

options = {}
parser = OptionParser.new do |opts|
  opts.banner = 'Usage: bundle exec ruby tools/local_editor/check_output.rb --production SITE --preview SITE [--map MAP]'
  opts.on('--production SITE', 'Normal production output directory') { |value| options[:production] = value }
  opts.on('--preview SITE', 'Local editor output directory') { |value| options[:preview] = value }
  opts.on('--map MAP', 'Editor map (default: maps.json beside preview site)') { |value| options[:map] = value }
  opts.on('-h', '--help', 'Show this help') { puts opts; exit }
end

begin
  parser.parse!
  raise OptionParser::MissingArgument, '--production and --preview are required' unless options[:production] && options[:preview]
  raise OptionParser::InvalidArgument, 'unexpected arguments' unless ARGV.empty?

  production = File.realpath(options[:production])
  preview = File.realpath(options[:preview])
  map_path = options[:map] || File.join(File.dirname(preview), 'maps.json')
  manifest = JSON.parse(File.read(map_path))
  documents = manifest.fetch('documents', manifest)
  raise 'The map must contain a nonempty documents object.' unless documents.is_a?(Hash) && !documents.empty?

  root = File.realpath(File.join(__dir__, '../..'))
  failures = []
  stats = { articles: 0, mapped_body_targets: 0, mapped_metadata_targets: 0, mapped_equations: 0, mapped_tables: 0, mapped_cells: 0, protected_elements: 0, production_files_scanned: 0 }
  leakage = %r{/__local_editor/|data-local-(?:edit|math|table|cell)-id|local-editor-context|BLOG_LOCAL_EDITOR|\.local-editor/(?:backups|builds)/}
  Dir.glob(File.join(production, '**', '*'), File::FNM_DOTMATCH).each do |path|
    next unless File.file?(path) && %w[.html .css .js .json].include?(File.extname(path).downcase)

    stats[:production_files_scanned] += 1
    failures << "Editor data leaked into production: #{path.delete_prefix(production + '/')}" if File.binread(path).match?(leakage)
  end
  %w[tools .local-editor].each do |name|
    failures << "Private editor directory in production: #{name}" if File.exist?(File.join(production, name))
  end

  resolve_page = lambda do |directory, url|
    uri = URI.parse(url)
    raise "Invalid article URL: #{url}" if uri.host || uri.scheme || !uri.path.start_with?('/')

    relative = URI::DEFAULT_PARSER.unescape(uri.path).delete_prefix('/')
    path = File.expand_path(relative, directory)
    path = File.join(path, 'index.html') if File.directory?(path)
    real = File.realpath(path)
    raise "Article path outside output: #{url}" unless real.start_with?(directory + '/')

    real
  end
  normalized_text = ->(text) { text.gsub(/\s+/, ' ').strip }
  block_selector = 'p,h1,h2,h3,h4,h5,h6,figcaption,caption,dt,dd,summary'
  protected_selector = 'table,pre,code,script,style,svg,math,mjx-container'
  protected_names = protected_selector.split(',')
  canonical_protected = lambda do |content|
    # Production compress_html removes formatting whitespace between tags.
    # Keep text, attributes, code and TeX byte-for-byte otherwise.
    content.css(protected_selector).map do |node|
      html = node.to_html.gsub(/>\s+</, '><').strip
      node.name == 'style' ? JSON.generate([node.name, node.attribute_nodes.map { |attribute| [attribute.name, attribute.value] }.sort, node.text.gsub(/\s+/, ' ').strip]) : html
    end
  end

  documents.each do |url, descriptor|
    begin
      raise 'Invalid map descriptor.' unless descriptor.is_a?(Hash) && descriptor['path'].is_a?(String)

      source = LocalEditor::Source.new(root: root, path: descriptor['path'])
      document = source.document
      raise 'Preview map does not match the current source revision.' unless descriptor['revision'] == document[:revision]

      local_html = Nokogiri::HTML(File.read(resolve_page.call(preview, url)))
      public_html = Nokogiri::HTML(File.read(resolve_page.call(production, url)))
      local_article = local_html.at_css('main article')
      public_article = public_html.at_css('main article')
      raise 'An article is missing from preview or production.' unless local_article && public_article

      local_content = local_article.at_css('.content')
      public_content = public_article.at_css('.content')
      raise 'An article content region is missing.' unless local_content && public_content
      raise 'Editor client/context is missing from the preview.' unless local_html.at_css('#local-editor-context') && local_html.at_css('script[src="/__local_editor/client.js"]')

      by_id = document[:segments].to_h { |segment| [segment[:id], segment] }
      targets = local_article.css('[data-local-edit-id]')
      seen = {}
      targets.each do |node|
        id = node['data-local-edit-id']
        segment = by_id[id]
        raise "Unknown mapped target: #{id}" unless segment
        raise "Duplicate mapped target: #{id}" if seen[id]
        raise "Mapped text differs from source: #{id}" unless node.text == segment[:text]
        raise "Protected content was mapped: #{id}" if node.ancestors.any? { |parent| protected_names.include?(parent.name) || parent.key?('data-local-editor-skip') } || node.key?('data-local-editor-skip') || node.xpath('.//*').any? { |child| protected_names.include?(child.name) }

        seen[id] = true
        stats[segment[:field] ? :mapped_metadata_targets : :mapped_body_targets] += 1
      end
      if document[:segments].any? { |segment| !segment[:field] } && local_content.css('[data-local-edit-id]').empty?
        raise 'Body source segments exist but the rendered body has zero editable targets.'
      end

      structures = document.fetch(:structures)
      raise 'An interactive component contains source editing targets.' unless local_content.css('[data-local-editor-skip] [data-local-edit-id], [data-local-editor-skip] [data-local-math-id], [data-local-editor-skip] [data-local-table-id], [data-local-editor-skip] [data-local-cell-id]').empty?
      equations = structures[:math].to_h { |item| [item[:id], item] }
      tables = structures[:tables].to_h { |item| [item[:id], item] }
      local_content.css('[data-local-math-id]').each do |node|
        id = node['data-local-math-id']
        raise "Unknown or repeated equation target: #{id}" unless equations.key?(id) && !seen[id]
        raise "Equation mapped inside protected content: #{id}" if node.ancestors.any? { |parent| %w[table pre code script style].include?(parent.name) }

        seen[id] = true
        stats[:mapped_equations] += 1
      end
      local_content.css('table[data-local-table-id]').each do |node|
        id = node['data-local-table-id']
        table = tables[id]
        raise "Unknown or repeated table target: #{id}" unless table && !seen[id]

        seen[id] = true
        expected_cells = table[:rows].flatten.to_h { |cell| [cell[:id], cell] }
        mapped_cells = node.css('[data-local-cell-id]')
        raise "Table cell count changed: #{id}" unless mapped_cells.length == expected_cells.length

        mapped_cells.each do |cell|
          cell_id = cell['data-local-cell-id']
          raise "Unknown or repeated table cell: #{cell_id}" unless expected_cells.key?(cell_id) && !seen[cell_id]

          seen[cell_id] = true
          stats[:mapped_cells] += 1
        end
        stats[:mapped_tables] += 1
      end
      # These attributes identify source cells without changing native table
      # HTML. Remove only our annotations before comparing protected output.
      local_content.css('[data-local-table-id], [data-local-cell-id]').each do |node|
        node.remove_attribute('data-local-table-id')
        node.remove_attribute('data-local-cell-id')
      end

      # Compare individual reading blocks to preserve meaningful spaces within
      # sentences, and all remaining text without inter-block formatting spaces.
      local_blocks = local_content.css(block_selector).map { |node| normalized_text.call(node.text) }
      public_blocks = public_content.css(block_selector).map { |node| normalized_text.call(node.text) }
      raise 'Rendered article paragraphs, headings or captions changed.' unless local_blocks == public_blocks
      raise 'Rendered article text changed.' unless local_content.text.gsub(/\s+/, '') == public_content.text.gsub(/\s+/, '')
      raise 'Math, table, code or other protected HTML changed.' unless canonical_protected.call(local_content) == canonical_protected.call(public_content)

      stats[:protected_elements] += local_content.css(protected_selector).length
      %w[header\ >\ h1 header\ >\ .post-desc header\ figcaption].each do |selector|
        left = local_article.at_css(selector)
        right = public_article.at_css(selector)
        raise "Rendered metadata changed: #{selector}" unless left&.text == right&.text
      end
      stats[:articles] += 1
    rescue StandardError => error
      failures << "#{url}: #{error.message}"
    end
  end
  raise 'No body editing targets were found in the preview.' if stats[:mapped_body_targets].zero?

  if failures.any?
    warn failures.map { |failure| "FAIL #{failure}" }.join("\n")
    warn JSON.generate(stats)
    exit 1
  end
  puts "#{stats[:articles]} articles passed: #{stats[:mapped_body_targets]} body targets, #{stats[:mapped_metadata_targets]} metadata targets; #{stats[:protected_elements]} protected elements unchanged."
  puts "#{stats[:mapped_equations]} equations, #{stats[:mapped_tables]} tables and #{stats[:mapped_cells]} cells match known source targets."
  puts "#{stats[:production_files_scanned]} production files contain no local editor data."
rescue OptionParser::ParseError => error
  warn error.message
  warn parser
  exit 2
rescue StandardError => error
  warn "Local editor output check failed: #{error.message}"
  exit 1
end
