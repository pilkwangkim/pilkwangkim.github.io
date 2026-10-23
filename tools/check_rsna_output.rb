# frozen_string_literal: true
# Inspect generated local HTML without a browser or network access.
require 'json'
require 'nokogiri'
require 'pathname'
require 'digest'
require 'time'

root = File.expand_path('..', __dir__)
checks = File.join(root, '_draft/_checks/rsna-series-20261022')
site = ARGV.fetch(0)
source = JSON.parse(File.read(File.join(checks, 'source-check.json')))
math = JSON.parse(File.read(File.join(checks, 'math-inputs.json')))
errors = []
rows = []
normalize = ->(text) { text.gsub(/\s+/, '') }
source['posts'].each do |post|
  file = File.join(site, post['url'].delete_prefix('/'), 'index.html')
  unless File.file?(file)
    errors << "Missing route #{post['url']}"
    next
  end
  doc = Nokogiri::HTML(File.read(file))
  article = doc.at_css('article.rsna-series')
  errors << "#{post['url']}: missing scoped layout" unless article
  next unless article
  errors << "#{post['url']}: expected one H1" unless article.css('h1').size == 1
  time = article.at_css('.post-meta time[data-ts]')
  errors << "#{post['url']}: rendered publication date differs" unless time && time['data-ts'].to_i == Time.parse(post['date']).to_i
  toc = article.at_css('.series-toc')
  series_count = source['posts'].count { |p| p['lang'] == post['lang'] }
  errors << "#{post['url']}: incorrect zero-based series contents" unless toc && toc.at_css('ol')['start'] == '0' && toc.css('li').size == series_count
  errors << "#{post['url']}: missing translation" unless article.css('.post-guide-translation').any? { |a| a['hreflang'] != post['lang'] }
  content = article.at_css('.content')
  errors << "#{post['url']}: missing content" unless content
  next unless content
  explanations = content.css('details')
  errors << "#{post['url']}: explanation count differs" unless explanations.size == post['explanations']
  explanations.each do |detail|
    errors << "#{post['url']}: malformed explanation" unless detail.css('summary').size == 1 && !detail.at_css('summary').text.strip.empty?
    errors << "#{post['url']}: explanation is not initially collapsed" if detail.key?('open')
    errors << "#{post['url']}: explanation Markdown was not parsed" if detail.css('p, ul, ol').empty?
  end
  errors << "#{post['url']}: footnote count differs" unless content.css('.footnotes li[id]').size == post.fetch('footnotes', 0)
  tables = content.css('table')
  tables.each_with_index do |table, i|
    errors << "#{post['url']}: table#{i} lacks responsive wrapper" unless table.ancestors.any? { |p| p['class'].to_s.split.include?('table-wrapper') }
    n = table.css('thead th').size
    errors << "#{post['url']}: empty table header" if table.css('thead th').any? { |th| th.text.strip.empty? }
    table.css('tbody tr').each do |tr|
      errors << "#{post['url']}: inconsistent table cells" unless tr.css('td').size == n
    end
  end
  raw_text = normalize.call(content.text)
  formulae = math.select { |x| x['post'] == File.basename(post['path']) }
  formulae.each do |formula|
    errors << "#{post['url']}: TeX changed by Markdown: #{formula['tex']}" unless raw_text.include?(normalize.call(formula['tex']))
  end
  post['images'].each do |url|
    present = article.css('img').any? { |img| img['src'] == url || img['data-src'] == url }
    present ||= article.css('picture source').any? { |src| src['srcset'].to_s.split(',').any? { |candidate| candidate.strip.split.first == url } }
    errors << "#{post['url']}: image absent in output #{url}" unless present
  end
  rows << { 'url'=>post['url'], 'lang'=>post['lang'], 'date'=>post['date'], 'tables'=>tables.size, 'explanations'=>explanations.size,
            'formulas'=>formulae.size, 'html_sha256'=>Digest::SHA256.file(file).hexdigest }
end
report = { 'status'=>errors.empty? ? 'passed' : 'failed', 'pages'=>rows, 'errors'=>errors,
           'limits'=>'Generated HTML structure and TeX preservation only. Browser desktop/mobile layout was blocked by the origin approval gate.' }
File.write(File.join(checks, 'output-check.json'), JSON.pretty_generate(report)+"\n")
puts JSON.pretty_generate({status:report['status'],pages:rows.size,tables:rows.sum { |r| r['tables'] },explanations:rows.sum { |r| r['explanations'] }, errors:errors})
exit(errors.empty? ? 0 : 1)
