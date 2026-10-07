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
topic_categories = read_yaml.call(File.join(source, '_data', 'topic_categories.yml'))
topic_groups = read_yaml.call(File.join(source, '_data', 'topic_groups.yml'))
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
normalized_text = ->(node) { node&.text.to_s.gsub(/\s+/, ' ').strip }
language_name = { 'ko' => 'Korean', 'en' => 'English' }
inherited_language = lambda do |node|
  (node ? [node, *node.ancestors] : []).find { |ancestor| ancestor.element? && ancestor.key?('lang') }&.[]('lang')
end
english_ui = lambda do |node, url, label|
  check.call(!node.nil?, "#{url}: #{label} is missing")
  next unless node

  check.call(inherited_language.call(node) == 'en', "#{url}: #{label} does not identify its English UI language")
  values = [normalized_text.call(node), node['aria-label'].to_s, node['title'].to_s, node['placeholder'].to_s]
  check.call(values.none? { |value| value.match?(/\p{Hangul}/) }, "#{url}: #{label} contains Korean UI text")
end

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
active_categories = topic_categories.select do |category|
  visible_posts.any? { |post| category['topics'].include?(post['topic']) }
end
navigation_entries = active_categories + active_topics.reject { |topic| topic['group'] == 'ai' }
guide_entries = active_topics + active_categories
legacy_topic_by_id = active_topics.to_h { |topic| [topic['id'], topic] }
category_by_topic = active_categories.each_with_object({}) do |category, parents|
  category['topics'].each { |id| parents[id] = category }
end
guide_posts = lambda do |guide|
  ids = guide['topics'] || [guide['id']]
  visible_posts.select { |post| ids.include?(post['topic']) }
end
if hub
  check.call(hub.at_css('html')['lang'] == 'en', 'Topics hub must identify its English prose language')
  check.call(hrefs.call(hub.css('.topic-card')) == navigation_entries.map { |entry| "/topics/#{entry['id']}/" }, 'Topics hub cards differ from navigation categories or YAML order')
  check.call(hub.css('.topic-card').map { |card| card['data-topic-entry'] } == navigation_entries.map { |entry| entry['id'] }, 'Topics hub entry IDs differ from navigation categories')
  check.call(hub.css('.topic-category-grid .topic-card').map { |card| card['data-topic-entry'] } == active_categories.map { |category| category['id'] }, 'AI navigation must show only the three category cards')
  hub.css('.topic-card').zip(navigation_entries).each do |card, entry|
    next unless entry

    members = guide_posts.call(entry)
    check.call(card['data-post-languages'].to_s.split.sort == members.map { |post| post['lang'] }.uniq.sort, "Hub #{entry['id']}: card languages differ from public members")
    count = members.map { |post| post['translation_key'] }.uniq.length
    count_label = "#{count} #{count == 1 ? 'article' : 'articles'}"
    check.call(normalized_text.call(card.at_css('.topic-card-meta')).match?(/\A#{Regexp.escape(count_label)}(?:\s|$)/), "Hub #{entry['id']}: English article count is not grouped by translation key")
  end
  check.call(hub.css('[data-language-controls] button[data-language-filter]').map { |button| button['data-language-filter'] }.sort == %w[all en ko], 'Topics hub language controls missing')
end

guide_entries.each do |topic|
  url = "/topics/#{topic['id']}/"
  html = html_for.call(url)
  next unless html

  check.call(html.at_css('html')['lang'] == 'en', "#{url}: topic introduction language must be English")
  check.call(html.css('[data-language-controls]').length == 1, "#{url}: exactly one language selector is required")
  expected = guide_posts.call(topic)
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
  # Article titles keep their content language. The version-picker labels are
  # English UI, while hreflang still describes each destination article.
  html.css('.topic-post-row').each do |row|
    links = row.css('a[data-post-language]')
    members = expected.select { |post| hrefs.call(links).include?(post['url']) }
    titles = row.css('[data-post-title-language]')
    check.call(titles.map { |title| title['data-post-title-language'] }.sort == members.map { |post| post['lang'] }.sort, "#{url}: article title languages differ from row versions")
    titles.each do |title|
      version = members.find { |post| post['lang'] == title['data-post-title-language'] }
      check.call(version && title['lang'] == version['lang'], "#{url}: article title language differs from its content")
      check.call(version && normalized_text.call(title) == version['title'].to_s.gsub(/\s+/, ' ').strip, "#{url}: article title differs from source title")
    end
    links.each do |link|
      target = hrefs.call([link]).first
      version = members.find { |post| post['url'] == target }
      next unless version

      check.call(link['lang'] == 'en' && link['hreflang'] == version['lang'] && link['data-post-language'] == version['lang'], "#{url}: version picker must separate English label language from destination language")
      check.call(normalized_text.call(link) == language_name[version['lang']], "#{url}: version picker label must use its English language name")
      english_ui.call(link, url, 'article language picker')
    end
  end
  article_order = groups.sort_by do |key, members|
    [-members.map { |post| date_epoch.call(post) }.max, key.to_s]
  end.map(&:first)
  emitted_order = rows.map do |row|
    first_url = hrefs.call(row.css('a[data-post-language]')).first
    expected.find { |post| post['url'] == first_url }&.fetch('translation_key')
  end
  check.call(emitted_order == article_order, "#{url}: full articles must be newest first, with translation_key ordering for equal dates")
  check.call(html.css('.reading-path, .reading-path-list').empty?, "#{url}: recommendation lists must not appear in topic details")
  latest_lists = html.css('.topic-all-posts')
  check.call(latest_lists.length == 1, "#{url}: topic detail must have exactly one latest article list")
  check.call(normalized_text.call(latest_lists.first&.at_css('h2')) == 'Latest articles', "#{url}: chronological article list must be headed Latest articles")
  if topic['topics']
    member_guides = html.css('details.topic-member-guides')
    check.call(member_guides.length == 1 && !member_guides.first.key?('open'), "#{url}: member topic guides must start collapsed")
    expected_members = topic['topics'].select { |id| legacy_topic_by_id.key?(id) }
    cards = html.css('.topic-member-guides .topic-card')
    check.call(cards.map { |card| card['data-topic-entry'] } == expected_members, "#{url}: member guide cards differ from category topics or their order")
    check.call(hrefs.call(cards) == expected_members.map { |id| "/topics/#{id}/" }, "#{url}: member guide URLs differ from preserved topic routes")
    check.call(html.css('.topic-all-posts, details.topic-member-guides').first == latest_lists.first, "#{url}: latest articles must appear before the competition guide disclosure")
  end
  parent = category_by_topic[topic['id']]
  expected_back = parent ? "/topics/#{parent['id']}/" : '/topics/'
  check.call(hrefs.call(html.css('.topic-back-link')) == [expected_back], "#{url}: back link does not return to its parent category or hub")
end

visible_posts.each do |post|
  html = html_for.call(post['url'])
  next unless html

  counterparts = visible_posts.select { |other| other['translation_key'] == post['translation_key'] && other['url'] != post['url'] }
  check.call(hrefs.call(html.css('.post-guide-translation')).sort == counterparts.map { |other| other['url'] }.sort, "#{post['url']}: translation link differs from counterpart")
  html.css('.post-guide-translation').each do |link|
    counterpart = counterparts.find { |other| other['url'] == hrefs.call([link]).first }
    next unless counterpart

    check.call(link['lang'] == 'en' && link['hreflang'] == counterpart['lang'], "#{post['url']}: translation label language must be English and hreflang must identify its target")
    check.call(normalized_text.call(link) == "Read in #{language_name[counterpart['lang']]}", "#{post['url']}: translation link must use an English label")
    english_ui.call(link, post['url'], 'translation picker')
  end
  parent = category_by_topic[post['topic']]
  expected_category = parent ? ["/topics/#{parent['id']}/"] : []
  check.call(hrefs.call(html.css('.post-guide-category')) == expected_category, "#{post['url']}: post guide category differs from topic membership")
  check.call(hrefs.call(html.css('.post-guide-topic')) == ["/topics/#{post['topic']}/"], "#{post['url']}: post guide must preserve its legacy topic link")
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
navigation_by_id = navigation_entries.to_h { |entry| [entry['id'], entry] }
guide_by_url = guide_entries.to_h { |entry| ["/topics/#{entry['id']}/", entry] }
post_by_url = posts.to_h { |post| [post['url'], post] }
expected_topic_links = navigation_entries.map { |entry| "/topics/#{entry['id']}/" }
sidebar_pages_checked = 0
html_cache.each do |url, html|
  section = html.at_css('#sidebar .sidebar-topics')
  check.call(!section.nil?, "#{url}: sidebar topic navigation missing")
  next unless section

  sidebar_pages_checked += 1
  links = section.css('a[data-topic-id]')
  check.call(hrefs.call(links) == expected_topic_links, "#{url}: sidebar topics differ from public topic guides or YAML order")
  check.call(links.map { |link| link['data-topic-id'] }.sort == navigation_by_id.keys.sort, "#{url}: sidebar navigation IDs missing, duplicated, private, or ungrouped legacy AI topics")
  groups = section.css('details[data-topic-group]')
  check.call(groups.map { |group| group['data-topic-group'] } == navigation_entries.map { |entry| entry['group'] }.uniq, "#{url}: sidebar topic groups differ from public groups or YAML order")
  groups.each do |group|
    check.call(group.at_css('summary'), "#{url}: topic group has no native keyboard-operable summary")
    group.css('a[data-topic-id]').each do |link|
      topic = navigation_by_id[link['data-topic-id']]
      check.call(topic && topic['group'] == group['data-topic-group'], "#{url}: sidebar topic #{link['data-topic-id']} appears in wrong group")
    end
  end
  article = post_by_url[url]
  page_guide = guide_by_url[url]
  current_topic_id = page_guide && page_guide['id']
  current_topic_id ||= article['topic'] if article && !article['hidden']
  current_topic = category_by_topic[current_topic_id] || navigation_by_id[current_topic_id]
  open_groups = groups.select { |group| group.key?('open') }.map { |group| group['data-topic-group'] }
  current_links = links.select { |link| link.key?('aria-current') }
  if current_topic
    check.call(open_groups == [current_topic['group']], "#{url}: current topic group must be the only open group")
    check.call(current_links.length == 1 && current_links.first['data-topic-id'] == current_topic['id'], "#{url}: sidebar does not identify current topic")
    expected_current = url == "/topics/#{current_topic['id']}/" ? 'page' : 'location'
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
    check.call(control['lang'] == 'en' && control['aria-label'] == 'Filter articles by language', "#{url}: article language filter must identify its English UI purpose")
    check.call(normalized_text.call(control.at_css('.language-label')) == 'Article language', "#{url}: article language filter heading must be English")
    buttons = control.css('button[data-language-filter]')
    check.call(buttons.length == 3 && buttons.to_h { |button| [button['data-language-filter'], normalized_text.call(button)] } == { 'all' => 'All', 'ko' => 'Korean', 'en' => 'English' }, "#{url}: article language filter labels must be All, Korean, and English")
    english_ui.call(control, url, 'article language filter')
  end
end
check.call(language_controls_checked == guide_entries.length + 1, 'Language selectors must cover the Topics hub and every legacy/category guide exactly once')

# UI language is independent of article language. Select only navigation labels
# and helper prose here: Korean article titles, tag/category names, and contents
# are valid content and must not be treated as interface translations.
english_ui_pages_checked = 0
sidebar_tab_labels = %w[home topics categories tags archives about]
sidebar_group_labels = topic_groups.to_h { |group| [group['id'], group['title']] }
guide_by_id = guide_entries.to_h { |entry| [entry['id'], entry] }
html_cache.each do |url, html|
  english_ui_pages_checked += 1
  check.call(html.at_css('html')['data-ui-language'] == 'en', "#{url}: document UI language must remain English")
  check.call(html.at_css('body')&.[]('lang') == 'en', "#{url}: body must identify its English interface language")
  %w[#sidebar #topbar-wrapper #panel-wrapper #tail-wrapper].each do |selector|
    wrapper = html.at_css(selector)
    check.call(wrapper && inherited_language.call(wrapper) == 'en', "#{url}: #{selector} must inherit the English UI language")
  end
  main = html.at_css('main')
  check.call(main && inherited_language.call(main) == html.at_css('html')['lang'], "#{url}: main content language must match the document content language")

  tabs = html.css('#sidebar .sidebar-navigation > ul.nav > li.nav-item > a.nav-link > span')
  check.call(tabs.map { |tab| normalized_text.call(tab).downcase } == sidebar_tab_labels, "#{url}: sidebar tabs must use the six English navigation labels")
  tabs.each { |tab| english_ui.call(tab, url, 'sidebar tab') }
  heading = html.at_css('#sidebar-topics-label')
  check.call(normalized_text.call(heading) == 'Explore topics', "#{url}: sidebar topic heading must be English")
  english_ui.call(heading, url, 'sidebar topic heading')
  html.css('.sidebar-topic-group').each do |group|
    label = group.at_css('summary > span')
    check.call(normalized_text.call(label) == sidebar_group_labels[group['data-topic-group']], "#{url}: sidebar topic group label must be English")
    english_ui.call(label, url, 'sidebar topic group label')
  end
  html.css('.sidebar-topics a[data-topic-id]').each do |link|
    entry = navigation_by_id[link['data-topic-id']]
    next unless entry

    count = guide_posts.call(entry).map { |post| post['translation_key'] }.uniq.length
    count_label = "#{count} #{count == 1 ? 'article' : 'articles'}"
    check.call(link.at_css('.sidebar-topic-count')&.[]('aria-label') == count_label, "#{url}: sidebar article count must use an English label and group translations")
    english_ui.call(link, url, 'sidebar topic link')
  end
  breadcrumb_home = html.at_css('#breadcrumb > span:first-child')
  check.call(normalized_text.call(breadcrumb_home) == 'Home', "#{url}: first breadcrumb must remain Home")
  english_ui.call(breadcrumb_home, url, 'home breadcrumb')
  search = html.at_css('#search-input')
  check.call(search&.[]('placeholder') == 'Search...', "#{url}: search placeholder must remain English")
  english_ui.call(search, url, 'search field')
  toggle = html.at_css('#desktop-sidebar-toggle')
  check.call(toggle&.[]('aria-label') == 'Collapse sidebar' && toggle&.[]('title') == 'Collapse sidebar', "#{url}: initial desktop sidebar toggle label must remain English")
  english_ui.call(toggle, url, 'desktop sidebar toggle')
  recent_heading = html.at_css('#access-lastmod > h2.panel-heading')
  check.call(normalized_text.call(recent_heading) == 'Recently Updated', "#{url}: recent updates heading must remain English")
  english_ui.call(recent_heading, url, 'recent updates heading')

  selectors = [
    '#search-cancel', '#sidebar .sidebar-bottom button', '#panel-wrapper h2.panel-heading',
    '.discovery-header', '.discovery-eyebrow', '.discovery-note', '.topic-back-link',
    '.topic-member-guides > summary',
    '.topic-all-posts > h2', '[data-language-empty]', '.post-guide-topic',
    '.post-guide-category', '.series-toc > summary', '.series-nav-label', '.series-nav-boundary'
  ]
  html.css(selectors.join(', ')).each { |node| english_ui.call(node, url, 'navigation or guide label') }
  html.css('.topic-card').each do |card|
    english_ui.call(card, url, 'topic card')
    entry = guide_by_id[card['data-topic-entry']]
    next unless entry

    count = guide_posts.call(entry).map { |post| post['translation_key'] }.uniq.length
    count_label = "#{count} #{count == 1 ? 'article' : 'articles'}"
    check.call(normalized_text.call(card.at_css('.topic-card-meta')).match?(/\A#{Regexp.escape(count_label)}(?:\s|$)/), "#{url}: topic card count must use English article units and group translations")
  end
end

# Recent updates are grouped globally, but each page picks its own language.
# Do not infer modification order from source dates: the Git hook supplies live
# last_modified_at values, which are absent from the preservation snapshot.
visible_translation_groups = visible_posts.group_by { |post| post['translation_key'] }
expected_recent_count = [visible_translation_groups.length, 5].min
recent_updates_pages_checked = 0
recent_article_keys = nil
html_cache.each do |url, html|
  panel = html.at_css('#access-lastmod')
  check.call(!panel.nil?, "#{url}: recent updates panel is missing")
  next unless panel

  recent_updates_pages_checked += 1
  links = panel.css('li a')
  check.call(links.length == expected_recent_count, "#{url}: recent updates must contain #{expected_recent_count} distinct articles")
  page_language = html.at_css('html')['lang'].to_s.split('-').first
  keys = []
  links.each do |link|
    target = hrefs.call([link]).first
    post = post_by_url[target]
    check.call(post && !post['hidden'] && post['published'] != false, "#{url}: recent updates contain a hidden, unpublished, or unknown post #{target}")
    next unless post && !post['hidden'] && post['published'] != false

    key = post['translation_key']
    keys << key
    check.call(link.ancestors('li').first&.[]('data-translation-key') == key, "#{url}: recent update row key differs from its article")
    members = visible_translation_groups.fetch(key)
    primary = members.min_by { |member| [member['lang'] == 'ko' ? 0 : 1, member['url']] }
    selected = members.find { |member| member['lang'] == page_language } || primary
    check.call(target == selected['url'], "#{url}: recent update #{key} does not use page language #{page_language} or its primary fallback")
    check.call(link['lang'] == selected['lang'] && link['hreflang'] == selected['lang'], "#{url}: recent update #{key} has incorrect language attributes")
  end
  check.call(keys.uniq == keys, "#{url}: recent updates repeat a bilingual article")
  recent_article_keys ||= keys
  check.call(keys == recent_article_keys, "#{url}: recent article groups or their global order differ across pages")
end

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
  'topic_category_pages' => active_categories.length,
  'series_posts' => visible_posts.count { |post| post['series'] },
  'translation_links' => visible_posts.count { |post| visible_posts.count { |other| other['translation_key'] == post['translation_key'] } == 2 },
  'generated_pages_checked' => html_cache.length,
  'sidebar_topic_pages_checked' => sidebar_pages_checked,
  'sidebar_navigation_entries' => navigation_entries.length,
  'language_control_blocks_checked' => language_controls_checked,
  'english_ui_pages_checked' => english_ui_pages_checked,
  'topic_article_order_pages_checked' => guide_entries.length,
  'recent_updates_pages_checked' => recent_updates_pages_checked,
  'recent_article_keys' => recent_article_keys,
  'generated_javascript_syntax_checks' => syntax_sources.length,
  'errors' => errors
}
File.write(options[:output], JSON.pretty_generate(report) + "\n") if options[:output]
puts JSON.pretty_generate(report)
exit(errors.empty? ? 0 : 1)
