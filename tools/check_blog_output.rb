#!/usr/bin/env ruby
# frozen_string_literal: true

# Check the generated navigation, preserved routes, and bilingual/series links.
# Run after Jekyll builds; this script does not rebuild or change the site.
#   bundle exec ruby tools/check_blog_output.rb --site-dir _site \
#     --metadata-report path/to/metadata-check.json --baseline path/to/posts.json

require 'date'
require 'json'
require 'nokogiri'
require 'open3'
require 'optparse'
require 'time'
require 'yaml'
require 'jekyll'

options = { source: File.expand_path('..', __dir__) }
OptionParser.new do |parser|
  parser.on('--source PATH') { |value| options[:source] = File.expand_path(value) }
  parser.on('--site-dir PATH') { |value| options[:site_dir] = File.expand_path(value) }
  parser.on('--metadata-report PATH') { |value| options[:metadata_report] = File.expand_path(value) }
  parser.on('--baseline PATH') { |value| options[:baseline] = File.expand_path(value) }
  parser.on('--output PATH') { |value| options[:output] = File.expand_path(value) }
end.parse!
abort 'Required: --site-dir, --metadata-report, --baseline' unless %i[site_dir metadata_report baseline].all? { |key| options[key] }

errors = []
check = ->(condition, message) { errors << message unless condition }
source = options[:source]
site_dir = options[:site_dir]
metadata = JSON.parse(File.read(options[:metadata_report]))
abort 'Metadata report must pass first' unless metadata['status'] == 'passed'
baseline = JSON.parse(File.read(options[:baseline]))
config = YAML.safe_load(File.read(File.join(source, '_config.yml')), permitted_classes: [Date, Time], aliases: true)
baseurl = config['baseurl'].to_s.delete_suffix('/')
urls = metadata.fetch('original_urls')
read_yaml = ->(path) { YAML.safe_load(File.read(path), permitted_classes: [Date, Time], aliases: true) }
topics = read_yaml.call(File.join(source, '_data', 'topics.yml'))
aliases = read_yaml.call(File.join(source, '_data', 'tag_aliases.yml'))

posts = baseline.map do |item|
  parts = File.binread(File.join(source, item['path'])).split(/^---\s*$\n?/, 3)
  read_yaml_data = YAML.safe_load(parts[1], permitted_classes: [Date, Time], aliases: true)
  read_yaml_data.merge('path' => item['path'], 'url' => urls.fetch(item['path']))
end
visible_posts = posts.reject { |post| post['hidden'] || post['published'] == false }
date_epoch = lambda do |post|
  date = post.fetch('date')
  date.respond_to?(:to_time) ? date.to_time.to_f : Time.parse(date.to_s).to_f
end
archive_order = lambda do |members|
  members.sort_by do |post|
    [-date_epoch.call(post), post['translation_key'].to_s, post['lang'] == 'ko' ? 0 : 1, post['url']]
  end.map { |post| post['url'] }
end

html_cache = {}
html_for = lambda do |url|
  path = File.join(site_dir, url.delete_prefix('/'), 'index.html')
  check.call(File.file?(path), "Missing generated route #{url}")
  next nil unless File.file?(path)

  html_cache[url] ||= Nokogiri::HTML(File.read(path))
end
hrefs = ->(nodes) { nodes.map { |node| node['href'].delete_prefix(baseurl) } }

urls.each do |path, url|
  next if baseline.find { |item| item['path'] == path }.dig('front_matter', 'published') == false

  html = html_for.call(url)
  next unless html

  check.call(html.at_css('#desktop-sidebar-toggle[aria-controls="sidebar"][aria-expanded]'), "#{url}: missing accessible desktop sidebar control")
  check.call(html.at_css('script[src$="/assets/js/blog-navigation.js"]'), "#{url}: missing navigation JavaScript")
  post = posts.find { |item| item['path'] == path }
  check.call(html.at_css('html')['lang'] == post['lang'], "#{url}: document language differs from article language")
end

hub = html_for.call('/topics/')
active_topics = topics.reject { |topic| topic['group'] == 'reference' || visible_posts.none? { |post| post['topic'] == topic['id'] } }
if hub
  check.call(hub.at_css('html')['lang'] == 'ko', 'Topics hub must identify its Korean prose language')
  check.call(hrefs.call(hub.css('.topic-card')) == active_topics.map { |topic| "/topics/#{topic['id']}/" }, 'Topics hub cards differ from published topic groups or YAML order')
  check.call(hub.css('[data-language-controls] button[data-language-filter]').map { |button| button['data-language-filter'] }.sort == %w[all en ko], 'Topics hub language controls missing')
end

active_topics.each do |topic|
  url = "/topics/#{topic['id']}/"
  html = html_for.call(url)
  next unless html

  check.call(html.at_css('html')['lang'] == 'ko', "#{url}: topic introduction language must be Korean")
  check.call(html.css('[data-language-controls]').length == 1, "#{url}: exactly one language selector is required")
  expected = visible_posts.select { |post| post['topic'] == topic['id'] }
  groups = expected.group_by { |post| post['translation_key'] }
  rows = html.css('.topic-all-posts .topic-post-row')
  check.call(rows.length == groups.length, "#{url}: bilingual articles are not grouped correctly")
  check.call(hrefs.call(rows.css('a[data-post-language]')).sort == expected.map { |post| post['url'] }.sort, "#{url}: full article links differ from published topic posts")
  rows.each do |row|
    row_urls = hrefs.call(row.css('a[data-post-language]'))
    members = expected.select { |post| row_urls.include?(post['url']) }
    check.call(members.map { |post| post['translation_key'] }.uniq.length == 1, "#{url}: row merges different articles")
    check.call(row['data-post-languages'].split.sort == members.map { |post| post['lang'] }.sort, "#{url}: incorrect row language metadata")
  end
  article_order = groups.sort_by do |key, members|
    [-members.map { |post| date_epoch.call(post) }.max, key.to_s]
  end.map(&:first)
  emitted_order = rows.map do |row|
    first_url = hrefs.call(row.css('a[data-post-language]')).first
    expected.find { |post| post['url'] == first_url }&.fetch('translation_key')
  end
  check.call(emitted_order == article_order, "#{url}: full articles must be newest first, with translation_key ordering for equal dates")
  recommended = html.css('.reading-path-list .topic-post-row').map do |row|
    first_url = hrefs.call(row.css('a[data-post-language]')).first
    expected.find { |post| post['url'] == first_url }&.fetch('translation_key')
  end
  check.call(recommended == Array(topic['recommended']), "#{url}: recommended reading order changed")
end

visible_posts.each do |post|
  html = html_for.call(post['url'])
  next unless html

  counterparts = visible_posts.select { |other| other['translation_key'] == post['translation_key'] && other['url'] != post['url'] }
  check.call(hrefs.call(html.css('.post-guide-translation')).sort == counterparts.map { |other| other['url'] }.sort, "#{post['url']}: translation link differs from counterpart")
  next unless post['series']

  parts = visible_posts.select { |other| other['series'] == post['series'] && other['lang'] == post['lang'] }.sort_by { |other| other['series_order'] }
  index = parts.index { |part| part['url'] == post['url'] }
  actual_contents = hrefs.call(html.css('.series-toc ol a'))
  check.call(actual_contents == parts.map { |part| part['url'] }, "#{post['url']}: series contents differ from ordered same-language parts")
  current = html.css('.series-toc [aria-current="page"]')
  check.call(current.length == 1 && hrefs.call(current).first == post['url'], "#{post['url']}: series contents do not identify current part")
  previous = index.positive? ? parts[index - 1]['url'] : nil
  following = parts[index + 1]&.fetch('url')
  check.call(hrefs.call(html.css('.series-nav a[rel="prev"]')).first == previous, "#{post['url']}: wrong previous series part")
  check.call(hrefs.call(html.css('.series-nav a[rel="next"]')).first == following, "#{post['url']}: wrong next series part")
end

original_tag_slugs = baseline.flat_map { |item| item.dig('front_matter', 'tags') || [] }.map { |tag| Jekyll::Utils.slugify(tag) }.uniq
original_tag_slugs.each { |slug| html_for.call("/tags/#{slug}/") }
aliases.each do |slug, canonical|
  html = html_for.call("/tags/#{slug}/")
  next unless html

  expected = visible_posts.select { |post| post['tags'].include?(canonical) }
  check.call(hrefs.call(html.css('#page-tag li a')) == archive_order.call(expected), "/tags/#{slug}/: alias archive members or date/language ordering differ")
end
korean = html_for.call('/tags/korean/')
if korean
  check.call(hrefs.call(korean.css('#page-tag li a')) == archive_order.call(visible_posts.select { |post| post['lang'] == 'ko' }), '/tags/korean/: language archive members or date ordering differ')
end

# Every generated sidebar exposes the same public topic guides. The current
# article or guide opens its own group; general navigation starts collapsed.
html_for.call('/')
topic_by_id = active_topics.to_h { |topic| [topic['id'], topic] }
topic_by_url = active_topics.to_h { |topic| ["/topics/#{topic['id']}/", topic] }
post_by_url = posts.to_h { |post| [post['url'], post] }
expected_topic_links = active_topics.map { |topic| "/topics/#{topic['id']}/" }
sidebar_pages_checked = 0
html_cache.each do |url, html|
  section = html.at_css('#sidebar .sidebar-topics')
  check.call(!section.nil?, "#{url}: sidebar topic navigation missing")
  next unless section

  sidebar_pages_checked += 1
  links = section.css('a[data-topic-id]')
  check.call(hrefs.call(links) == expected_topic_links, "#{url}: sidebar topics differ from public topic guides or YAML order")
  check.call(links.map { |link| link['data-topic-id'] }.sort == topic_by_id.keys.sort, "#{url}: sidebar topic IDs missing, duplicated, or private")
  groups = section.css('details[data-topic-group]')
  check.call(groups.map { |group| group['data-topic-group'] } == active_topics.map { |topic| topic['group'] }.uniq, "#{url}: sidebar topic groups differ from public groups or YAML order")
  groups.each do |group|
    check.call(group.at_css('summary'), "#{url}: topic group has no native keyboard-operable summary")
    group.css('a[data-topic-id]').each do |link|
      topic = topic_by_id[link['data-topic-id']]
      check.call(topic && topic['group'] == group['data-topic-group'], "#{url}: sidebar topic #{link['data-topic-id']} appears in wrong group")
    end
  end
  article = post_by_url[url]
  current_topic = topic_by_url[url] || (article && !article['hidden'] && topic_by_id[article['topic']])
  open_groups = groups.select { |group| group.key?('open') }.map { |group| group['data-topic-group'] }
  current_links = links.select { |link| link.key?('aria-current') }
  if current_topic
    check.call(open_groups == [current_topic['group']], "#{url}: current topic group must be the only open group")
    check.call(current_links.length == 1 && current_links.first['data-topic-id'] == current_topic['id'], "#{url}: sidebar does not identify current topic")
    expected_current = topic_by_url.key?(url) ? 'page' : 'location'
    check.call(current_links.first&.[]('aria-current') == expected_current, "#{url}: sidebar aria-current must identify #{expected_current}")
  else
    check.call(open_groups.empty?, "#{url}: general navigation must start with topic groups closed")
    check.call(current_links.empty?, "#{url}: general navigation must not mark a current topic")
  end
end

# Chirpy treats div[class^='language-'] as highlighted code. Discovery controls
# must keep their own styling and remain outside that selector.
language_controls_checked = 0
html_cache.each do |url, html|
  html.css('[data-language-controls]').each do |control|
    language_controls_checked += 1
    classes = control['class'].to_s.split
    check.call(classes.include?('discovery-language-controls'), "#{url}: language controls lack their dedicated styling class")
    check.call(!(control.name == 'div' && control['class'].to_s.start_with?('language-')), "#{url}: language controls collide with Chirpy's code-block selector")
  end
end
check.call(language_controls_checked == active_topics.length + 1, 'Language selectors must cover the Topics hub and every topic guide exactly once')

# Validate the emitted inline hook, where production HTML compression can alter
# JavaScript comments, as well as the standalone navigation script.
inline_hooks = html_cache.values.flat_map do |html|
  html.css('script:not([src])').map(&:text).select { |script| script.include?('pilkwang:sidebar') }
end.uniq
check.call(!inline_hooks.empty?, 'Generated pages have no sidebar preference restore hook')
syntax_sources = inline_hooks.each_with_index.map { |script, index| ["generated sidebar hook #{index + 1}", script] }
navigation_script = File.join(site_dir, 'assets', 'js', 'blog-navigation.js')
check.call(File.file?(navigation_script), 'Generated navigation JavaScript file is missing')
syntax_sources << ['generated blog-navigation.js', File.read(navigation_script)] if File.file?(navigation_script)
syntax_sources.each do |label, javascript|
  _, stderr, status = Open3.capture3('node', '--check', stdin_data: javascript)
  check.call(status.success?, "#{label}: invalid JavaScript syntax: #{stderr.strip}")
end

report = {
  'status' => errors.empty? ? 'passed' : 'failed',
  'original_post_routes' => urls.length,
  'original_tag_routes' => original_tag_slugs.length,
  'topic_pages' => active_topics.length,
  'series_posts' => visible_posts.count { |post| post['series'] },
  'translation_links' => visible_posts.count { |post| visible_posts.count { |other| other['translation_key'] == post['translation_key'] } == 2 },
  'generated_pages_checked' => html_cache.length,
  'sidebar_topic_pages_checked' => sidebar_pages_checked,
  'sidebar_public_topics' => active_topics.length,
  'language_control_blocks_checked' => language_controls_checked,
  'topic_article_order_pages_checked' => active_topics.length,
  'generated_javascript_syntax_checks' => syntax_sources.length,
  'errors' => errors
}
File.write(options[:output], JSON.pretty_generate(report) + "\n") if options[:output]
puts JSON.pretty_generate(report)
exit(errors.empty? ? 0 : 1)
