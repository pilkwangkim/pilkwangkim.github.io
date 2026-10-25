# frozen_string_literal: true

require 'liquid'
require 'nokogiri'

module ResponsivePictures
  FIGURE_CLASS = 'rsna-results-figure'
  PICTURE = /<picture\b(?:[^>"']|"[^"]*"|'[^']*')*>.*?<\/picture\s*>/im

  # Chirpy wraps each image in a popup link, including images inside <picture>.
  # Keep the fallback image a direct child so the browser can select <source>.
  # Only repaired picture fragments are serialized; all surrounding HTML stays exact.
  def responsive_pictures(input)
    source = input.to_s
    return source unless source.include?(FIGURE_CLASS)

    changed = false
    repaired = source.gsub(PICTURE) do |fragment|
      replacement = ResponsivePictures.repair(fragment)
      changed ||= !replacement.nil?
      replacement || fragment
    end
    changed ? repaired : source
  end

  def self.content_children(node)
    node.children.reject { |child| child.text? && child.text.strip.empty? }
  end

  def self.repair(fragment)
    document = Nokogiri::HTML5.fragment(fragment)
    picture = document.element_children.first
    return unless content_children(document) == [picture] && picture&.name == 'picture'

    classes = picture['class'].to_s.split
    return unless classes.include?(FIGURE_CLASS)

    children = content_children(picture)
    return unless children.all? { |child| child.element? && %w[source a].include?(child.name) }
    return unless children.any? { |child| child.name == 'source' }

    links = children.select { |child| child.name == 'a' }
    return unless links.length == 1

    link = links.first
    return unless link['class'].to_s.split.include?('popup')

    images = content_children(link)
    return unless images.length == 1 && images.first.element? && images.first.name == 'img'

    image = images.first
    link.unlink
    image.unlink
    link.children.remove
    picture.add_child(image)
    # Chirpy clears loading effects on img.parentElement, which is now picture.
    link_classes = link['class'].to_s.split
    loading_classes = link_classes & %w[shimmer blur]
    remaining_classes = (classes.reject { |name| name == FIGURE_CLASS } + loading_classes).uniq
    if remaining_classes.empty?
      picture.remove_attribute('class')
    else
      picture['class'] = remaining_classes.join(' ')
    end
    link['class'] = (link_classes.reject { |name| loading_classes.include?(name) } + [FIGURE_CLASS]).uniq.join(' ')
    picture.replace(link)
    link.add_child(picture)
    document.to_html
  end
end

Liquid::Template.register_filter(ResponsivePictures)
