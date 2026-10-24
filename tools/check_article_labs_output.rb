#!/usr/bin/env ruby
# frozen_string_literal: true

require 'json'
require 'yaml'
require 'date'
require 'nokogiri'
require 'optparse'
require 'uri'

options = {}
OptionParser.new do |parser|
  parser.on('--site PATH') { |value| options[:site] = value }
  parser.on('--metadata PATH') { |value| options[:metadata] = value }
  parser.on('--output PATH') { |value| options[:output] = value }
end.parse!
abort 'Required: --site and --metadata' unless options[:site] && options[:metadata]
root = File.expand_path('..', __dir__)
site = File.expand_path(options[:site])
metadata = JSON.parse(File.read(options[:metadata]))
abort 'Metadata must pass' unless metadata['status'] == 'passed'
registry = Dir.glob(File.join(root, '_data/article_labs_*.yml')).each_with_object({}) do |path, result|
  YAML.safe_load_file(path).each do |id, definition|
    abort "Repeated lab definition #{id}" if result.key?(id)
    result[id] = definition
  end
end
errors = []
check = ->(condition, message) { errors << message unless condition }
seen = []
pages = []
metadata.fetch('post_urls').each do |source, url|
  raw = File.read(File.join(root, source))
  front = YAML.safe_load(raw.split(/^---\s*$\n?/, 3)[1], permitted_classes: [Date, Time], aliases: true)
  next if front['published'] == false

  html = Nokogiri::HTML(File.read(File.join(site, url.delete_prefix('/'), 'index.html')))
  declared = Array(front['interactive_labs'])
  includes = raw.scan(/\{%\s*include\s+interactive\/article-lab\.html\s+id="([^"]+)"[^%]*%\}/).flatten
  labs = html.css('[data-article-lab]')
  check.call(labs.map { |lab| lab['data-article-lab'] } == includes, "#{url}: includes differ from rendered labs")
  check.call(labs.map { |lab| lab['id'] }.uniq.length == labs.length, "#{url}: repeated lab section ID")
  scripts = html.css('script[src]').map { |node| URI.parse(node['src']).path }
  styles = html.css('link[rel="stylesheet"][href]').map { |node| URI.parse(node['href']).path }
  expected = declared.empty? ? [] : ['/assets/js/article-labs.js', *declared.map { |bundle| "/assets/js/article-labs-#{bundle}.js" }]
  check.call(scripts.grep(%r{/assets/js/article-labs(?:-[a-z-]+)?\.js\z}) == expected, "#{url}: incorrect isolated script loading/order")
  check.call(styles.count('/assets/css/article-labs.css') == (declared.empty? ? 0 : 1), "#{url}: incorrect isolated styles")
  (expected + (declared.empty? ? [] : ['/assets/css/article-labs.css'])).each do |asset|
    relative = asset.delete_prefix('/')
    check.call(File.binread(File.join(root, relative)) == File.binread(File.join(site, relative)), "#{url}: stale generated asset #{asset}")
  end
  html.css('script[src*="article-labs"],link[href*="article-labs.css"]').each do |node|
    address = node['src'] || node['href']
    check.call(URI.parse(address).query.to_s.match?(/\Av=[0-9a-f]{12}\z/), "#{url}: missing asset fingerprint")
  end
  labs.each do |lab|
    id = lab['data-article-lab']; definition = registry.fetch(id)
    language = front['lang'] == 'ko' ? 'ko' : 'en'
    check.call(lab['data-language'] == language, "#{url}/#{id}: language mismatch")
    check.call(lab.at_css('h3').text == definition.fetch('title').fetch(language), "#{url}/#{id}: title mismatch")
    check.call(lab['class'].split.include?('tex2jax_ignore') && lab.key?('data-local-editor-skip'), "#{url}/#{id}: missing math/editor boundary")
    check.call(lab.css('[data-local-edit-id],[data-local-math-id],[data-local-table-id],[data-local-cell-id]').empty?, "#{url}/#{id}: source editor mapped widget UI")
    check.call(lab.at_css('fieldset[disabled]') && lab.at_css('button[data-lab-reset][disabled]'), "#{url}/#{id}: controls active before initialization")
    check.call(lab.at_css('[role="status"][aria-live="polite"]') && lab.at_css('noscript'), "#{url}/#{id}: missing status/fallback")
    check.call(lab['data-asset-version'].to_s.match?(/\A[0-9a-f]{12}\z/), "#{url}/#{id}: missing data fingerprint")
    module_script = html.css('script[src]').find { |node| URI.parse(node['src']).path == "/assets/js/article-labs-#{definition['bundle']}.js" }
    module_version = module_script && URI.decode_www_form(URI.parse(module_script['src']).query.to_s).to_h['v']
    check.call(lab['data-asset-version'] == module_version, "#{url}/#{id}: data/module versions differ")
    Array(definition['data']).each do |path|
      check.call(File.file?(File.join(site, path.delete_prefix('/'))), "#{url}/#{id}: missing public data #{path}")
      check.call(File.binread(File.join(root, path.delete_prefix('/'))) == File.binread(File.join(site, path.delete_prefix('/'))), "#{url}/#{id}: stale public data #{path}")
    end
    seen << id
  end
  pages << { source: source, url: url, language: front['lang'], bundles: declared, labs: includes } unless labs.empty?
end
check.call(seen.uniq.sort == registry.keys.sort, 'Some registered labs have no published embed')
report = { status: errors.empty? ? 'passed' : 'failed', pages: pages.length, embeds: seen.length, lab_types: seen.uniq.length,
           bundles: registry.values.map { |definition| definition['bundle'] }.uniq.length, coverage: pages, errors: errors }
File.write(options[:output], JSON.pretty_generate(report) + "\n") if options[:output]
puts JSON.pretty_generate(report.reject { |key, _| key == :coverage })
exit(errors.empty? ? 0 : 1)
