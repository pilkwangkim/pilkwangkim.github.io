#!/usr/bin/env ruby
# frozen_string_literal: true

require 'jekyll'
require 'yaml'
require_relative '../_plugins/article-labs'

fixture = { 'z' => ['x', 1], 'a' => { 'ko' => '위상', 'en' => 'Phase →' } }
expected = '{"a":{"en":"Phase →","ko":"위상"},"z":["x",1]}'.b
raise 'Registry serialization must use stable UTF-8 JSON bytes' unless ArticleLabAssets.registry_bytes(fixture) == expected

reverse = lambda do |value|
  case value
  when Hash then value.to_a.reverse.to_h.transform_values { |item| reverse.call(item) }
  when Array then value.map { |item| reverse.call(item) }
  else value
  end
end

registries = Dir.glob(File.expand_path('../_data/article_labs_*.yml', __dir__))
registries.each do |path|
  definitions = YAML.safe_load_file(path)
  original = ArticleLabAssets.registry_bytes(definitions)
  raise "Registry order changed fingerprint bytes: #{path}" unless original == ArticleLabAssets.registry_bytes(reverse.call(definitions))

  changed = Marshal.load(Marshal.dump(definitions))
  changed.values.first.fetch('title')['en'] += ' changed'
  raise "Title change did not affect fingerprint bytes: #{path}" if original == ArticleLabAssets.registry_bytes(changed)
end
puts "PASS stable registry JSON fixture, key-order independence and content sensitivity for #{registries.length} bundles"
