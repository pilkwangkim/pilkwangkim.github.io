#!/usr/bin/env ruby
require 'date'
require 'digest'
require 'json'
require 'nokogiri'
require 'uri'
require 'yaml'

root = File.expand_path('../../..', __dir__)
qa = __dir__
site = '/tmp/pilkwang-technical-covers-20261007-site'
old_site = '/tmp/pilkwang-corporate-single-20261007-site'
manifest = JSON.parse(File.read(File.join(qa, 'cover-prompts.json')))
baseline = JSON.parse(File.read(File.join(qa, 'baseline-posts.json')))
metadata = JSON.parse(File.read(File.join(qa, 'metadata-check.json')))
errors = []
checks = 0
check = lambda { |condition, message| checks += 1; errors << message unless condition }
normalize = lambda do |v|
  case v
  when Time, Date then v.to_s
  when Array then v.map { |x| normalize.call(x) }
  when Hash then v.transform_values { |x| normalize.call(x) }
  else v
  end
end
selected = manifest.fetch('covers').flat_map { |r| r.fetch('source_posts') }
current = baseline.map do |row|
  raw = File.binread(File.join(root, row['path']))
  parts = raw.split(/^---\s*$\n?/, 3)
  data = normalize.call(YAML.safe_load(parts[1], permitted_classes: [Date, Time], aliases: true))
  check.call(Digest::SHA256.hexdigest(parts[2]) == row['body_sha256'], "#{row['path']}: body changed")
  check.call(data.reject { |k, _| k == 'image' } == row['front_matter'].reject { |k, _| k == 'image' }, "#{row['path']}: other metadata changed")
  if selected.include?(row['path'])
    check.call(!row['front_matter'].key?('image') && data.key?('image'), "#{row['path']}: not a previously coverless article")
  else
    check.call(data == row['front_matter'], "#{row['path']}: unrelated metadata changed")
  end
  [row['path'], data]
end.to_h
check.call(selected.uniq.length == 26 && manifest['covers'].length == 14, 'Expected 14 articles / 26 pages')
check.call(current.values.none? { |p| !%w[essays reference].include?(p['topic']) && !p['hidden'] && p['published'] != false && !p['image'] }, 'A visible technical article still lacks a cover')
pages = []
manifest['covers'].each do |cover|
  asset = File.join(root, cover['path'].delete_prefix('/'))
  check.call(Digest::SHA256.file(asset).hexdigest == cover['sha256'], "#{cover['translation_key']}: asset hash mismatch")
  cover['source_posts'].each do |path|
    data = current.fetch(path)
    url = metadata.fetch('post_urls').fetch(path)
    image = data.fetch('image')
    alt = cover.fetch(data['lang'] == 'en' ? 'alt_en' : 'alt_ko')
    check.call(image['path'] == cover['path'] && image['alt'] == alt && image['hide_caption'] == true, "#{path}: cover metadata mismatch")
    html = Nokogiri::HTML(File.read(File.join(site, url.delete_prefix('/'), 'index.html')))
    preview = html.at_css('article header .preview-img img, article header img.preview-img')
    check.call(preview && preview['src'] == cover['path'] && preview['alt'] == alt, "#{url}: rendered cover mismatch")
    check.call(html.css('article header figcaption').empty?, "#{url}: unexpected visible alt caption")
    og = html.at_css('meta[property="og:image"]')
    check.call(og && URI(og['content']).path == cover['path'], "#{url}: OG image mismatch")
    old_file = File.join(old_site, url.delete_prefix('/'), 'index.html')
    check.call(File.file?(old_file), "#{url}: missing pre-edit built page")
    if File.file?(old_file)
      old_html = Nokogiri::HTML(File.read(old_file))
      check.call(html.at_css('article .content').to_html == old_html.at_css('article .content').to_html, "#{url}: rendered body changed")
    end
    pages << { path: path, url: url, lang: data['lang'], cover: cover['path'], alt: alt }
  end
end
report = {
  status: errors.empty? ? 'passed' : 'failed',
  assertions: checks,
  posts_checked: baseline.length,
  source_body_and_metadata_preservation: true,
  article_pages_checked: pages.length,
  logical_articles: manifest['covers'].length,
  reused_body_figures: manifest['reused_body_figures'],
  generated_covers: manifest['generated_covers'],
  pages: pages,
  errors: errors
}
File.write(File.join(qa, 'cover-contract-check.json'), JSON.pretty_generate(report) + "\n")
puts JSON.pretty_generate(report.reject { |k, _| k == :pages })
exit(errors.empty? ? 0 : 1)
