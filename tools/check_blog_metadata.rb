#!/usr/bin/env ruby
# frozen_string_literal: true

# Verify metadata contracts and, when supplied, preservation against a pre-edit
# snapshot. Run with the repository's Bundler environment:
#   bundle exec ruby tools/check_blog_metadata.rb --baseline path/to/posts.json

require 'date'
require 'digest'
require 'json'
require 'optparse'
require 'yaml'
require 'jekyll'

options = { source: File.expand_path('..', __dir__) }
OptionParser.new do |parser|
  parser.on('--source PATH') { |value| options[:source] = File.expand_path(value) }
  parser.on('--baseline PATH') { |value| options[:baseline] = File.expand_path(value) }
  parser.on('--output PATH') { |value| options[:output] = File.expand_path(value) }
end.parse!

errors = []
check = ->(condition, message) { errors << message unless condition }
source = options[:source]
normalize = lambda do |value|
  case value
  when Time, Date then value.to_s
  when Array then value.map { |item| normalize.call(item) }
  when Hash then value.transform_values { |item| normalize.call(item) }
  else value
  end
end
read_yaml = lambda do |path|
  YAML.safe_load(File.read(path), permitted_classes: [Date, Time], aliases: true)
end

posts = Dir.glob(File.join(source, '_posts', '**', '*.{md,markdown,html}')).sort.map do |path|
  raw = File.binread(path)
  parts = raw.split(/^---\s*$\n?/, 3)
  if parts.length != 3 || !parts.first.empty?
    errors << "Missing front matter: #{path}"
    next
  end
  {
    'path' => path.delete_prefix("#{source}/"),
    'front_matter' => YAML.safe_load(parts[1], permitted_classes: [Date, Time], aliases: true),
    'body_sha256' => Digest::SHA256.hexdigest(parts[2])
  }
end.compact

topics_path = File.join(source, '_data', 'topics.yml')
series_path = File.join(source, '_data', 'series.yml')
aliases_path = File.join(source, '_data', 'tag_aliases.yml')
topics = File.file?(topics_path) ? read_yaml.call(topics_path) : []
series = File.file?(series_path) ? read_yaml.call(series_path) : []
aliases = File.file?(aliases_path) ? read_yaml.call(aliases_path) : {}
check.call(topics.is_a?(Array) && !topics.empty?, 'Topics must be a nonempty list')
check.call(series.is_a?(Array), 'Series must be a list')
check.call(aliases.is_a?(Hash), 'Tag aliases must be a mapping')
exit 1 unless topics.is_a?(Array) && series.is_a?(Array) && aliases.is_a?(Hash)

topic_by_id = topics.to_h { |item| [item['id'], item] }
series_by_id = series.to_h { |item| [item['id'], item] }
check.call(topic_by_id.length == topics.length, 'Topic IDs must be unique')
check.call(series_by_id.length == series.length, 'Series IDs must be unique')
check.call(topic_by_id.keys.all? { |id| id.is_a?(String) && id.match?(/\A[a-z0-9]+(?:-[a-z0-9]+)*\z/) }, 'Topic IDs must be canonical slugs')

posts.each do |post|
  data = post['front_matter']
  path = post['path']
  check.call(%w[ko en].include?(data['lang']), "#{path}: lang must be ko or en")
  tags = data['tags']
  check.call(tags.is_a?(Array) && !tags.empty?, "#{path}: tags must be a nonempty list")
  next unless tags.is_a?(Array)

  check.call(tags.uniq == tags, "#{path}: duplicate tags")
  tags.each do |tag|
    check.call(tag.is_a?(String) && tag.match?(/\A[a-z0-9]+(?:-[a-z0-9]+)*\z/), "#{path}: noncanonical tag #{tag.inspect}")
  end
  check.call((tags & %w[korean english ko kr en]).empty?, "#{path}: language belongs in lang, not tags")
  check.call(data['translation_key'].is_a?(String) && !data['translation_key'].empty?, "#{path}: missing translation_key")
  membership = data['topic']
  check.call(topic_by_id.key?(membership), "#{path}: unknown topic #{membership.inspect}")
  topic = topic_by_id[membership]
  check.call(tags.include?(topic['tag']), "#{path}: topic tag #{topic['tag'].inspect} missing") if topic && topic['tag']
  next unless data['series']

  check.call(series_by_id.key?(data['series']), "#{path}: unknown series #{data['series'].inspect}")
  check.call(data['series_order'].is_a?(Integer) && data['series_order'].positive?, "#{path}: series_order must be a positive integer")
  definition = series_by_id[data['series']]
  check.call(membership == definition['topic'], "#{path}: series topic differs from topic") if definition && definition['topic']
end

groups = posts.group_by { |post| post['front_matter']['translation_key'] }
groups.each do |key, members|
  next unless key

  languages = members.map { |post| post['front_matter']['lang'] }
  check.call(languages.uniq == languages, "Translation #{key}: duplicate language")
  next unless members.length > 1

  %w[tags topic series series_order].each do |field|
    values = members.map { |post| post['front_matter'][field] }
    values = values.map { |value| value.is_a?(Array) ? value.sort : value }
    check.call(values.uniq.length == 1, "Translation #{key}: inconsistent #{field}")
  end
end

posts.select { |post| post['front_matter']['series'] }.group_by do |post|
  data = post['front_matter']
  [data['series'], data['lang']]
end.each do |(id, language), members|
  orders = members.map { |post| post['front_matter']['series_order'] }
  check.call(orders.uniq == orders, "Series #{id}/#{language}: duplicate order")
  check.call(orders.all? { |order| order.is_a?(Integer) && order.positive? }, "Series #{id}/#{language}: invalid order")
end

current_tags = posts.flat_map { |post| post['front_matter']['tags'] || [] }.uniq
aliases.each do |legacy, target|
  check.call(legacy.is_a?(String) && !legacy.empty?, "Invalid alias slug #{legacy.inspect}")
  check.call(current_tags.include?(target), "Alias #{legacy}: target #{target.inspect} has no posts")
  check.call(legacy != Jekyll::Utils.slugify(target), "Alias #{legacy}: redundant self redirect")
  check.call(!current_tags.any? { |tag| Jekyll::Utils.slugify(tag) == legacy }, "Alias #{legacy}: conflicts with current tag route")
end

original_urls = {}
if options[:baseline]
  baseline = JSON.parse(File.read(options[:baseline]))
  baseline_by_path = baseline.to_h { |post| [post['path'], post] }
  check.call(posts.map { |post| post['path'] }.sort == baseline_by_path.keys.sort, 'Post source paths changed')
  posts.each do |post|
    original = baseline_by_path[post['path']]
    next unless original

    check.call(post['body_sha256'] == original['body_sha256'], "#{post['path']}: article body changed")
    original['front_matter'].each do |key, value|
      next if %w[tags lang].include?(key)

      check.call(normalize.call(post['front_matter'][key]) == value, "#{post['path']}: original #{key} changed")
    end
  end

  original_tags = baseline.flat_map { |post| post['front_matter']['tags'] || [] }.uniq
  current_slugs = current_tags.map { |tag| Jekyll::Utils.slugify(tag) }
  original_tags.each do |tag|
    slug = Jekyll::Utils.slugify(tag)
    # Language used to be a tag; a dedicated language archive preserves its URL.
    check.call(current_slugs.include?(slug) || aliases.key?(slug) || slug == 'korean', "Original tag route /tags/#{slug}/ has no current tag or alias")
  end

  # Compute both URLs through Jekyll itself; filename punctuation and case matter.
  config = Jekyll.configuration('source' => source, 'quiet' => true)
  site = Jekyll::Site.new(config)
  site.reset
  site.read
  current_docs = site.posts.docs.to_h { |post| [post.relative_path, post] }
  baseline.each do |original|
    path = original['path']
    old = Jekyll::Document.new(File.join(source, path), site: site, collection: site.posts)
    old.merge_data!(original['front_matter'])
    old.send(:populate_title)
    original_urls[path] = old.url
    current = current_docs[path]
    if original['front_matter']['published'] != false
      check.call(!current.nil?, "#{path}: original published article is missing from Jekyll")
      check.call(current.url == old.url, "#{path}: post URL changed (#{old.url} -> #{current.url})") if current
    end
  end
end

report = {
  'status' => errors.empty? ? 'passed' : 'failed',
  'posts' => posts.length,
  'canonical_tags' => current_tags.length,
  'translation_groups' => groups.length,
  'bilingual_pairs' => groups.count { |_, members| members.length == 2 },
  'topics' => topics.length,
  'series' => series.length,
  'tag_aliases' => aliases.length,
  'body_and_metadata_preservation_checked' => !!options[:baseline],
  'original_urls' => original_urls,
  'errors' => errors
}
File.write(options[:output], JSON.pretty_generate(report) + "\n") if options[:output]
puts JSON.pretty_generate(report.reject { |key, _| key == 'original_urls' })
exit(errors.empty? ? 0 : 1)
