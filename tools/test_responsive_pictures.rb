#!/usr/bin/env ruby
# frozen_string_literal: true

require_relative '../_plugins/responsive-pictures'

filter = Object.new.extend(ResponsivePictures)
checks = 0
assert = lambda do |condition, message|
  raise message unless condition

  checks += 1
end

# Representative Chirpy 7.6 image refactoring, including popup and lazy-load attrs.
picture = <<~HTML.chomp
  <picture class="rsna-results-figure contrast" id="sampling" data-note="a > b">
    <source media="(max-width: 620px)" srcset="/mobile.png" width="760" height="1350">
    <source type="image/webp" srcset="/desktop.webp">
    <a href="/desktop.png" class="popup img-link shimmer" data-fancybox="article" aria-label="Enlarge figure"><img src="/desktop.png" alt="A &amp; B" width="1800" height="1050" loading="lazy" data-local-edit-skip="true"></a>
  </picture>
HTML
prefix = '<div data-x="exact &quot;bytes&quot;">한국어 $x+y$</div>'
caption = '<p><em>Figure 1. A caption with <strong>markup</strong>.</em></p>'
suffix = '<script type="application/json">{"same": "value"}</script>'
original = prefix + picture + caption + suffix
result = filter.responsive_pictures(original)
doc = Nokogiri::HTML5.fragment(result)
outer = doc.at_css('a.rsna-results-figure')
assert.call(!outer.nil?, 'Figure class must move to the outer popup link')
assert.call(outer.element_children.map(&:name) == ['picture'], 'Popup must contain the picture')
assert.call(outer['href'] == '/desktop.png' && outer['class'].split == %w[popup img-link rsna-results-figure], 'Popup target and classes must survive without a permanent loading overlay')
assert.call(outer['data-fancybox'] == 'article' && outer['aria-label'] == 'Enlarge figure', 'Popup attributes must survive')
inner = outer.at_css('picture')
assert.call(inner['class'].split == %w[contrast shimmer] && inner['id'] == 'sampling' && inner['data-note'] == 'a > b', 'Picture must retain its attributes and own the image loading effect')
assert.call(inner.element_children.map(&:name) == %w[source source img], 'Sources and fallback image must be direct picture children')
before = Nokogiri::HTML5.fragment(picture)
assert.call(inner.css('source').map { |node| node.attributes.transform_values(&:value) } == before.css('source').map { |node| node.attributes.transform_values(&:value) }, 'Source order, dimensions and URLs must survive')
assert.call(inner.at_css('img').attributes.transform_values(&:value) == before.at_css('img').attributes.transform_values(&:value), 'Every image attribute, including dimensions and lazy loading, must survive')
assert.call(outer.next_element&.to_html == Nokogiri::HTML5.fragment(caption).to_html, 'Caption must remain adjacent to the styled figure wrapper')
assert.call(result.start_with?(prefix) && result.end_with?(caption + suffix), 'Surrounding content must remain byte-identical')
assert.call(filter.responsive_pictures(result) == result, 'Repair must be idempotent')

# Chirpy handles both load events and already-complete lazy images by clearing
# shimmer/blur from the direct image parent. Neither effect may remain outside it.
%w[shimmer blur].each do |loading_class|
  fixture = picture.sub('img-link shimmer', "img-link #{loading_class}")
  if loading_class == 'blur'
    fixture = fixture.sub('loading="lazy"', 'data-lqip="true" data-src="/full.png"')
  end
  loading_doc = Nokogiri::HTML5.fragment(filter.responsive_pictures(fixture))
  image = loading_doc.at_css('img')
  assert.call(image.parent.name == 'picture' && image.parent['class'].split.include?(loading_class), "#{loading_class} must be on the direct image parent")
  assert.call(!loading_doc.at_css('a')['class'].split.include?(loading_class), "#{loading_class} must not remain on the outer link")
  image.parent.remove_class(loading_class)
  assert.call(loading_doc.css(".#{loading_class}").empty?, "Chirpy's direct-parent cleanup must remove the complete #{loading_class} effect")
  assert.call(image['data-lqip'] == 'true' && image['data-src'] == '/full.png', 'LQIP loading attributes must survive') if loading_class == 'blur'
end

untouched = [
  '<p><a href="/ordinary.png" class="popup img-link shimmer"><img src="/ordinary.png" loading="lazy"></a></p>',
  '<picture class="other-figure"><source srcset="/mobile.png"><a class="popup"><img src="/desktop.png"></a></picture>',
  '<picture class="rsna-results-figure"><source srcset="/mobile.png"><img src="/desktop.png"></picture>',
  '<a class="popup rsna-results-figure"><picture><source srcset="/mobile.png"><img src="/desktop.png"></picture></a>',
  '<picture class="rsna-results-figure"><source srcset="/mobile.png"><a class="popup"><img src="/desktop.png">A linked caption</a></picture>',
  '<picture class="rsna-results-figure"><source srcset="/mobile.png"><a class="popup"><img src="/one.png"></a><a class="popup"><img src="/two.png"></a></picture>',
  '<picture class="rsna-results-figure"><source srcset="/mobile.png"><a class="popup"><span><img src="/desktop.png"></span></a></picture>',
  '<picture class="rsna-results-figure"><source srcset="/mobile.png"><a href="/custom"><img src="/desktop.png"></a></picture>',
  '<pre>&lt;picture class="rsna-results-figure"&gt;example&lt;/picture&gt;</pre>',
  '<!-- rsna-results-figure --> <p>Exact unchanged text.</p>'
]
untouched.each do |html|
  assert.call(filter.responsive_pictures(html) == html, "Unrelated or already valid markup changed: #{html}")
end

multiple = filter.responsive_pictures(picture + caption + untouched.first + picture)
assert.call(Nokogiri::HTML5.fragment(multiple).css('a.rsna-results-figure > picture > img').length == 2, 'Every matching picture must be repaired')
assert.call(multiple.include?(caption + untouched.first), 'Content between repaired fragments must remain exact')
assert.call(Liquid::Template.parse('{{ html | responsive_pictures }}').render!('html' => original) == result, 'Liquid must expose the repair filter')

puts "PASS #{checks} responsive-picture checks: valid direct image ancestry, source selection markup, popup/lazy-load attributes, caption adjacency, idempotence and untouched surrounding content"
