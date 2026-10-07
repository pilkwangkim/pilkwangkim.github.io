#!/usr/bin/env ruby
# frozen_string_literal: true

# Check the generated navigation, preserved routes, and bilingual/series links.
# Run after Jekyll builds; this script does not rebuild or change the site.
#   bundle exec ruby tools/check_blog_output.rb --site-dir _site \
#     --metadata-report path/to/metadata-check.json --baseline path/to/posts.json

require 'date'
require 'digest'
require 'json'
require 'nokogiri'
require 'open3'
require 'optparse'
require 'time'
require 'uri'
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
asset_paths = %w[assets/js/blog-navigation.js assets/css/jekyll-theme-chirpy.scss Gemfile] + Dir.glob('_sass/**/*.scss', base: source)
asset_fingerprint = Digest::SHA256.hexdigest(asset_paths.sort.map do |path|
  path + "\0" + File.binread(File.join(source, path)) + "\0"
end.join)[0, 12]
urls = metadata['post_urls'] || metadata.fetch('original_urls')
read_yaml = ->(path) { YAML.safe_load(File.read(path), permitted_classes: [Date, Time], aliases: true) }
topics = read_yaml.call(File.join(source, '_data', 'topics.yml'))
topics_by_id = topics.to_h { |topic| [topic['id'], topic] }
topic_categories = read_yaml.call(File.join(source, '_data', 'topic_categories.yml'))
topic_groups = read_yaml.call(File.join(source, '_data', 'topic_groups.yml'))
registered_series = read_yaml.call(File.join(source, '_data', 'series.yml')).to_h { |series| [series['id'], series] }
aliases = read_yaml.call(File.join(source, '_data', 'tag_aliases.yml'))

posts = urls.map do |path, url|
  parts = File.binread(File.join(source, path)).split(/^---\s*$\n?/, 3)
  read_yaml_data = YAML.safe_load(parts[1], permitted_classes: [Date, Time], aliases: true)
  read_yaml_data.merge('path' => path, 'url' => url)
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
hrefs = ->(nodes) { nodes.map { |node| node['href'].to_s.delete_prefix(baseurl) } }
asset_uri = lambda do |value|
  URI.parse(value.to_s)
rescue URI::InvalidURIError
  nil
end
asset_nodes = lambda do |html, selector, attribute, path|
  html.css(selector).filter_map do |node|
    parsed = asset_uri.call(node[attribute])
    [node, parsed] if parsed && parsed.path.to_s.delete_prefix(baseurl) == path
  end
end
normalized_text = ->(node) { node&.text.to_s.gsub(/\s+/, ' ').strip }
language_name = { 'ko' => '한국어', 'en' => 'English' }
legacy_language_name = { 'ko' => 'Korean', 'en' => 'English' }
preferred_version = lambda do |members, language|
  candidates = members.select { |post| post['lang'] == language }
  candidates = members if candidates.empty?
  candidates.min_by { |post| [post['lang'] == 'en' ? 0 : 1, post['article_version'] == 'original' ? 0 : 1, post['url']] }
end
inherited_language = lambda do |node|
  (node ? [node, *node.ancestors] : []).find { |ancestor| ancestor.element? && ancestor.key?('lang') }&.[]('lang')
end
english_ui = lambda do |node, url, label|
  check.call(!node.nil?, "#{url}: #{label} is missing")
  next unless node
  next if node.matches?('a[hreflang="ko"]')

  check.call(inherited_language.call(node) == 'en', "#{url}: #{label} does not identify its English UI language")
  english_content = node.dup
  # Korean language choices identify themselves as 한국어; the surrounding
  # discovery/navigation interface stays in English.
  english_content.css('button[data-language-filter="ko"], a[hreflang="ko"]').remove
  values = [normalized_text.call(english_content), node['aria-label'].to_s, node['title'].to_s, node['placeholder'].to_s]
  check.call(values.none? { |value| value.match?(/\p{Hangul}/) }, "#{url}: #{label} contains Korean UI text")
end
iso_epoch = lambda do |value, label|
  Time.iso8601(value.to_s).to_f
rescue ArgumentError
  check.call(false, "#{label}: missing or invalid ISO 8601 date")
  nil
end

urls.each do |path, url|
  next if posts.find { |item| item['path'] == path }['published'] == false

  html = html_for.call(url)
  next unless html

  check.call(html.at_css('#desktop-sidebar-toggle[aria-controls="sidebar"][aria-expanded]'), "#{url}: missing accessible desktop sidebar control")
  check.call(asset_nodes.call(html, 'script[src]', 'src', '/assets/js/blog-navigation.js').length == 1, "#{url}: missing or duplicated navigation JavaScript base path")
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
  check.call(hub.css('[data-article-view-controls]').empty?, 'Topics hub must not display article-list view controls')
end

topic_article_panels_checked = 0
topic_article_bundles_checked = 0
guide_entries.each do |topic|
  url = "/topics/#{topic['id']}/"
  html = html_for.call(url)
  next unless html

  check.call(html.at_css('html')['lang'] == 'en', "#{url}: topic introduction language must be English")
  check.call(html.css('[data-language-controls]').length == 1, "#{url}: exactly one language selector is required")
  expected = guide_posts.call(topic)
  articles = expected.group_by { |post| post['translation_key'] }
  view_controls = html.css('[data-article-view-controls]')
  check.call(view_controls.length == 1, "#{url}: exactly one Latest/All view selector is required")
  if (control = view_controls.first)
    check.call(control['class'].to_s.split.include?('article-view-controls') && control['role'] == 'group' && control['aria-label'] == 'Article view' && control['lang'] == 'en', "#{url}: article view controls must identify their English UI purpose")
    view_buttons = control.css('button[data-article-view]')
    check.call(view_buttons.map { |button| button['data-article-view'] } == %w[latest all], "#{url}: view controls must contain Latest and All exactly once")
    view_buttons.each do |button|
      view = button['data-article-view']
      check.call(button['aria-controls'] == "#{view}-articles", "#{url}: #{view} view button must control its corresponding article panel")
      check.call(button['aria-pressed'] == (view == 'latest' ? 'true' : 'false'), "#{url}: Latest must be the initially selected article view")
      label = view == 'latest' ? 'Latest articles' : "All articles (#{articles.length})"
      check.call(normalized_text.call(button) == label, "#{url}: view button must be labelled #{label}")
    end
    all_count = control.css('[data-article-view="all"] [data-article-count]')
    check.call(all_count.length == 1 && normalized_text.call(all_count.first) == articles.length.to_s, "#{url}: All articles count must use logical articles rather than versions or series bundles")
    english_ui.call(control, url, 'article view selector')
  end
  catalog = articles.each_with_object({}) do |(key, members), collection|
    series_id = members.first['series']
    group_id = if topic['group'] == 'ai'
                 "topic:#{members.first['topic']}"
               else
                 registered_series.key?(series_id) ? "series:#{series_id}" : "article:#{key}"
               end
    (collection[group_id] ||= {})[key] = members
  end
  bundles = catalog.each_with_object({}) do |(group_id, group_articles), collection|
    members = group_articles.values.flatten
    topic_id = members.first['topic']
    series = registered_series.values.find { |definition| members.any? { |post| post['series'] == definition['id'] } }
    if group_id.start_with?('topic:')
      topic_entry = topics_by_id.fetch(topic_id)
      collection[group_id] = { 'id' => series&.fetch('id') || topic_id, 'topic' => topic_id, 'title' => topic_entry['title'] }
    elsif group_id.start_with?('series:')
      collection[group_id] = { 'id' => series.fetch('id'), 'topic' => topic_id, 'title' => series.fetch('title') }
    end
  end
  catalog_order = catalog.sort_by do |group_id, group_articles|
    [-group_articles.values.flatten.map { |post| date_epoch.call(post) }.max, group_id]
  end.map(&:first)
  check.call(html.css('[data-article-panel]').length == 2, "#{url}: exactly two article view panels are required")
  { 'latest' => ['#latest-articles', '.topic-latest-posts', 'Latest articles'],
    'all' => ['#all-articles', '.topic-all-posts', 'All articles'] }.each do |view, (selector, css_class, heading)|
    panels = html.css(selector)
    check.call(panels.length == 1, "#{url}: exactly one #{view} article panel is required")
    panel = panels.first
    next unless panel

    topic_article_panels_checked += 1
    label = "#{url} #{view}"
    check.call(panel['data-article-panel'] == view && panel.matches?(css_class), "#{label}: article panel metadata or class differs from its view")
    check.call(!panel.key?('hidden') && panel['aria-hidden'] != 'true' && !panel['class'].to_s.split.include?('d-none'), "#{label}: the full server-rendered panel must remain accessible without JavaScript")
    check.call(normalized_text.call(panel.at_css('h2')) == heading, "#{label}: article panel must be headed #{heading}")
    check.call(panel['data-article-limit'] == '5', "#{label}: latest view must declare its five-article limit") if view == 'latest'
    rows = panel.css('.topic-post-row')
    check.call(rows.length == articles.length, "#{label}: server-rendered rows must contain every logical article exactly once")
    emitted_keys = rows.map { |row| row['data-article-key'] }
    check.call(emitted_keys.sort_by(&:to_s) == articles.keys.sort_by(&:to_s), "#{label}: article keys are missing, duplicated, or unknown")
    check.call(hrefs.call(rows.css('a[data-post-language]')).sort == expected.map { |post| post['url'] }.sort, "#{label}: article version links differ from published topic posts")
    rows.each do |row|
      key = row['data-article-key']
      members = articles[key]
      check.call(!members.nil?, "#{label}: unknown logical article #{key.inspect}")
      next unless members

      links = row.css('a[data-post-language]')
      check.call(hrefs.call(links).sort == members.map { |post| post['url'] }.sort, "#{label}: row #{key} merges or loses article versions")
      check.call(row['data-post-languages'].to_s.split.sort == members.map { |post| post['lang'] }.uniq.sort, "#{label}: row #{key} has incorrect language metadata")
      rendered_date = iso_epoch.call(row['data-article-date'], "#{label} article #{key}")
      expected_date = members.map { |post| date_epoch.call(post) }.max
      check.call(rendered_date && (rendered_date - expected_date).abs < 0.001, "#{label}: article #{key} date metadata differs from publication date")
      if row.key?('data-series-order')
        check.call(row['data-series-order'].match?(/\A[1-9]\d*\z/) && members.all? { |post| post['series_order'].to_s == row['data-series-order'] }, "#{label}: article #{key} has incorrect series-order metadata")
      end
      check.call(row['data-series-order'] == members.first['series_order']&.to_s, "#{label}: article #{key} must preserve its actual series order without assigning a new part number")

      # Each language has one preferred title, linking to Original when present.
      # Language-choice labels identify themselves; English remains primary.
      titles = row.css('[data-post-title-language]')
      check.call(titles.map { |title| title['data-post-title-language'] }.sort == members.map { |post| post['lang'] }.uniq.sort, "#{label}: article #{key} must have one title per language")
      primary_language = members.any? { |post| post['lang'] == 'en' } ? 'en' : members.first['lang']
      primary_titles = titles.reject { |title| title.key?('hidden') || title['aria-hidden'] == 'true' }
      check.call(primary_titles.length == 1 && primary_titles.first['data-post-title-language'] == primary_language, "#{label}: article #{key} must show its English primary title or available-language fallback")
      titles.each do |title|
        version = preferred_version.call(members, title['data-post-title-language'])
        check.call(version && title['lang'] == version['lang'], "#{label}: article #{key} title language differs from its content")
        check.call(version && normalized_text.call(title) == version['title'].to_s.gsub(/\s+/, ' ').strip, "#{label}: article #{key} title differs from source title")
        title_link = title.at_css('a')
        check.call(version && title_link && hrefs.call([title_link]) == [version['url']], "#{label}: article #{key} title must link to its own language version")
      end
      links.each do |link|
        version = members.find { |post| post['url'] == hrefs.call([link]).first }
        next unless version

        check.call(link['lang'] == version['lang'] && link['hreflang'] == version['lang'] && link['data-post-language'] == version['lang'], "#{label}: language picker must identify its label and destination language")
        check.call(normalized_text.call(link) == language_name[version['lang']], "#{label}: version picker label must use English or 한국어")
        english_ui.call(link, url, 'article language picker')
      end
      if members.any? { |member| member['article_version'] }
        editions = row.css('[data-article-version]')
        check.call(editions.map { |edition| edition['data-article-version'] } == %w[original compact], "#{label}: #{key} must expose Original before Compact")
        editions.each do |edition|
          version = edition['data-article-version']
          expected_edition = members.select { |member| member['article_version'] == version }
          check.call(hrefs.call(edition.css('a[data-post-language]')).sort == expected_edition.map { |member| member['url'] }.sort, "#{label}: #{key}/#{version} links mix versions or languages")
        end
      end
    end

    blocks = panel.css('[data-article-group]')
    check.call(blocks.map { |block| block['data-article-group'] } == catalog_order, "#{label}: article groups must be newest first, with group-key ordering for equal dates")
    blocks.each do |block|
      group_id = block['data-article-group']
      group_articles = catalog[group_id]
      next unless group_articles

      group_date = iso_epoch.call(block['data-group-date'], "#{label} group #{group_id}")
      expected_group_date = group_articles.values.flatten.map { |post| date_epoch.call(post) }.max
      check.call(group_date && (group_date - expected_group_date).abs < 0.001, "#{label}: group #{group_id} date must identify its newest article")
      expected_part_keys = if view == 'latest'
                             group_articles.sort_by do |key, members|
                               [-members.map { |post| date_epoch.call(post) }.max, key.to_s]
                             end.map(&:first)
                           else
                             sequences = group_articles.group_by do |key, members|
                               members.first['series'] ? "series:#{members.first['series']}" : "article:#{key}"
                             end
                             sequences.sort_by do |key, entries|
                               [entries.map { |_, members| members.map { |post| date_epoch.call(post) }.max }.min, key]
                             end.flat_map do |_, entries|
                               entries.sort_by do |key, members|
                                 [members.first['series_order'] || 0, members.map { |post| date_epoch.call(post) }.max, key.to_s]
                               end.map(&:first)
                             end
                           end
      actual_part_keys = block.css('.topic-post-row').map { |row| row['data-article-key'] }
      part_order = view == 'latest' ? 'latest date order' : 'reading order of actual series and independent articles'
      check.call(actual_part_keys == expected_part_keys, "#{label}: group #{group_id} must contain exactly its own articles in #{part_order}")
      bundle = bundles[group_id]
      disclosures = block.css('details[data-bundle-id]')
      unless bundle
        check.call(disclosures.empty?, "#{label}: non-AI standalone articles must remain visible without an invented bundle")
        next
      end

      check.call(disclosures.length == 1, "#{label}: group #{group_id} must use exactly one native topic or series disclosure")
      details = disclosures.first
      next unless details

      topic_article_bundles_checked += 1
      bundle_id = bundle['id']
      prefix = view == 'latest' ? 'latest-series-' : 'series-'
      check.call(details['data-bundle-id'] == bundle_id && details['id'] == "#{prefix}#{bundle_id}", "#{label}: bundle #{group_id} must preserve its registered series anchor or topic ID")
      check.call(details['data-topic-id'] == bundle['topic'], "#{label}: bundle #{group_id} must identify its actual topic")
      check.call(!details.key?('open'), "#{label}: every topic and series disclosure must start collapsed, including single-article topics")
      bundle_heading = details.at_css('.topic-series-heading')
      check.call(normalized_text.call(bundle_heading) == bundle['title'].to_s.gsub(/\s+/, ' ').strip, "#{label}: bundle #{group_id} must always use its original English title")
      check.call(bundle_heading && bundle_heading['lang'] == 'en', "#{label}: bundle #{group_id} heading must identify its English language")
      check.call(bundle_heading && !bundle_heading.key?('data-discovery-title') && !bundle_heading.key?('data-discovery-title-language') && bundle_heading.css('[data-discovery-title], [data-discovery-title-language]').empty?, "#{label}: bundle #{group_id} heading must not contain bilingual title variants")
      check.call(details['data-post-languages'].to_s.split.sort == group_articles.values.flatten.map { |post| post['lang'] }.uniq.sort, "#{label}: bundle #{group_id} languages differ from its public articles")
      article_count = group_articles.length
      check.call(normalized_text.call(details.at_css('[data-bundle-count]')) == article_count.to_s, "#{label}: bundle #{group_id} must count logical articles rather than language versions")
      check.call(normalized_text.call(details.at_css('[data-bundle-count-label]')) == (article_count == 1 ? 'article' : 'articles'), "#{label}: bundle #{group_id} must use the correct English article count unit")
      if view == 'latest'
        full_links = details.css('a[data-open-bundle]')
        check.call(full_links.length == 1 && full_links.first['data-open-bundle'] == bundle_id && full_links.first['href'] == "#series-#{bundle_id}", "#{label}: full bundle link must target its All-view disclosure")
        check.call(normalized_text.call(full_links.first) == "View all articles (#{article_count})", "#{label}: full bundle link must use the English label and complete logical article count")
        check.call(full_links.first&.css('[data-bundle-total-count]')&.map { |count| normalized_text.call(count) } == [article_count.to_s], "#{label}: full bundle count must be available for language filtering")
      else
        check.call(details.css('a[data-open-bundle]').empty?, "#{label}: All-view bundle must not repeat its own full catalog link")
      end
    end
    check.call(panel.css('details[data-bundle-id]').map { |details| details['data-bundle-id'] }.sort == bundles.values.map { |bundle| bundle['id'] }.sort, "#{label}: topic or series disclosures are missing, duplicated, or unknown")
    check.call(panel.css('details[data-series-id]').empty?, "#{label}: topic catalog bundles must not invent article-level series metadata")
  end
  check.call(html.css('.reading-path, .reading-path-list').empty?, "#{url}: recommendation lists must not appear in topic details")
  check.call(html.css('.topic-latest-posts').length == 1 && html.css('.topic-all-posts').length == 1, "#{url}: Latest and All must each have one dedicated article list")
  check.call(html.css('.topic-member-guides').empty?, "#{url}: separate competition guide blocks must not appear")
  check.call(html.css('.topic-detail .topic-card').empty?, "#{url}: topic details must use the unified article catalog instead of separate competition cards")
  check.call(html.css('.topic-detail summary').none? { |summary| normalized_text.call(summary).start_with?('Competition guides') }, "#{url}: Competition guides navigation must not appear")
  parent = category_by_topic[topic['id']]
  expected_back = parent ? "/topics/#{parent['id']}/" : '/topics/'
  check.call(hrefs.call(html.css('.topic-back-link')) == [expected_back], "#{url}: back link does not return to its parent category or hub")
end

visible_posts.each do |post|
  html = html_for.call(post['url'])
  next unless html

  counterparts = visible_posts.select { |other| other['translation_key'] == post['translation_key'] && (post['article_version'] || other['url'] != post['url']) }
  check.call(hrefs.call(html.css('.post-guide-translation')).sort == counterparts.map { |other| other['url'] }.sort, "#{post['url']}: translation link differs from counterpart")
  html.css('.post-guide-translation').each do |link|
    counterpart = counterparts.find { |other| other['url'] == hrefs.call([link]).first }
    next unless counterpart

    expected_label = post['article_version'] ? language_name[counterpart['lang']] : "Read in #{legacy_language_name[counterpart['lang']]}"
    expected_language = post['article_version'] ? counterpart['lang'] : 'en'
    check.call(link['lang'] == expected_language && link['hreflang'] == counterpart['lang'], "#{post['url']}: translation link must identify its label and destination language")
    check.call(normalized_text.call(link) == expected_label, "#{post['url']}: translation link has the wrong language label")
    english_ui.call(link, post['url'], 'translation picker')
  end
  if post['article_version']
    selected = html.css('.post-guide-translation[aria-current="page"]')
    check.call(hrefs.call(selected) == [post['url']], "#{post['url']}: version selector must mark only the current page")
    editions = html.css('.post-guide [data-article-version]')
    check.call(editions.map { |edition| edition['data-article-version'] } == %w[original compact], "#{post['url']}: post guide must show Original then Compact")
    editions.each do |edition|
      version = edition['data-article-version']
      expected_edition = counterparts.select { |member| member['article_version'] == version }
      check.call(hrefs.call(edition.css('.post-guide-translation')).sort == expected_edition.map { |member| member['url'] }.sort, "#{post['url']}: #{version} selector must preserve version/language pairing")
    end
  end
  alternates = html.css('link[rel="alternate"][hreflang]')
  expected_alternates = visible_posts.select { |other| other['translation_key'] == post['translation_key'] && other['article_version'] == post['article_version'] }
  actual_alternates = alternates.map { |link| [link['hreflang'], URI.parse(link['href']).path.delete_prefix(baseurl)] }
  check.call(actual_alternates.sort == expected_alternates.map { |other| [other['lang'], other['url']] }.sort, "#{post['url']}: hreflang alternates must stay within the same version")
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
    check.call(buttons.length == 3 && buttons.to_h { |button| [button['data-language-filter'], normalized_text.call(button)] } == { 'all' => 'All', 'ko' => '한국어', 'en' => 'English' }, "#{url}: article language filter labels must be All, 한국어, and English")
    check.call(buttons.map { |button| button['data-language-filter'] } == %w[en ko all], "#{url}: article language controls must present English first")
    check.call(buttons.all? { |button| button['aria-pressed'] == (button['data-language-filter'] == 'en' ? 'true' : 'false') }, "#{url}: English must be the initially selected article language")
    buttons.each do |button|
      expected_language = button['data-language-filter'] == 'ko' ? 'ko' : 'en'
      check.call(inherited_language.call(button) == expected_language, "#{url}: article language filter button must identify its label language")
    end
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
    '[data-bundle-id] > summary',
    '.topic-latest-posts > h2', '.topic-all-posts > h2', '[data-article-view-controls]', '[data-language-empty]', '.post-guide-topic',
    '.post-guide-category', '.series-toc > summary', '.series-nav-label', '.series-nav-boundary'
  ]
  html.css(selectors.join(', ')).each { |node| english_ui.call(node, url, 'navigation or guide label') }
  html.css('.topic-card').each do |card|
    english_ui.call(card, url, 'topic card')
    entry = guide_by_id[card['data-topic-entry']]
    next unless entry

    card_heading = card.at_css('h3')
    check.call(normalized_text.call(card_heading) == entry['title'].to_s.gsub(/\s+/, ' ').strip, "#{url}: topic card #{entry['id']} must always use its original English title")
    check.call(inherited_language.call(card_heading) == 'en', "#{url}: topic card #{entry['id']} heading must identify its English language")
    check.call(card.css('[data-discovery-title], [data-discovery-title-language]').empty?, "#{url}: topic card #{entry['id']} must not contain bilingual title variants")
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
    selected = preferred_version.call(members, page_language)
    check.call(target == selected['url'], "#{url}: recent update #{key} does not use page language #{page_language} or its primary fallback")
    check.call(link['lang'] == selected['lang'] && link['hreflang'] == selected['lang'], "#{url}: recent update #{key} has incorrect language attributes")
  end
  check.call(keys.uniq == keys, "#{url}: recent updates repeat a bilingual article")
  recent_article_keys ||= keys
  check.call(keys == recent_article_keys, "#{url}: recent article groups or their global order differ across pages")
end

# The theme CSS and navigation script share one source fingerprint so a cached
# PWA document cannot pair new markup with assets from the prior layout.
versioned_asset_pages_checked = 0
html_cache.each do |url, html|
  versioned_asset_pages_checked += 1
  [['script[src]', 'src', '/assets/js/blog-navigation.js'],
   ['link[rel="stylesheet"][href]', 'href', '/assets/css/jekyll-theme-chirpy.css']].each do |selector, attribute, path|
    matches = asset_nodes.call(html, selector, attribute, path)
    check.call(matches.length == 1, "#{url}: missing or duplicated versioned asset #{path}")
    next unless matches.length == 1

    parsed = matches.first.last
    versions = URI.decode_www_form(parsed.query.to_s).select { |key, _| key == 'v' }.map(&:last)
    check.call(versions == [asset_fingerprint], "#{url}: #{path} must use shared source fingerprint v=#{asset_fingerprint}")
  rescue ArgumentError
    check.call(false, "#{url}: #{path} has an invalid asset-version query")
  end
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
  'original_post_routes' => metadata.fetch('original_urls').length,
  'all_post_routes' => urls.length,
  'versioned_articles' => visible_posts.select { |post| post['article_version'] }.map { |post| post['translation_key'] }.uniq.length,
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
  'topic_article_panels_checked' => topic_article_panels_checked,
  'topic_article_bundles_checked' => topic_article_bundles_checked,
  'recent_updates_pages_checked' => recent_updates_pages_checked,
  'recent_article_keys' => recent_article_keys,
  'blog_asset_version' => asset_fingerprint,
  'versioned_asset_pages_checked' => versioned_asset_pages_checked,
  'generated_javascript_syntax_checks' => syntax_sources.length,
  'errors' => errors
}
File.write(options[:output], JSON.pretty_generate(report) + "\n") if options[:output]
puts JSON.pretty_generate(report)
exit(errors.empty? ? 0 : 1)
