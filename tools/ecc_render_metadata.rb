# frozen_string_literal: true

# Add current Jekyll routes to the source metadata check for rendered-output QA.
# check_blog_metadata.rb populates routes only in its historical baseline mode;
# a new article needs current routing without pretending to preserve an old body.
require 'json'
require 'jekyll'

input, output = ARGV
abort 'Usage: bundle exec ruby tools/ecc_render_metadata.rb INPUT.json OUTPUT.json' unless input && output
report = JSON.parse(File.read(input))
abort 'Source metadata must pass first' unless report['status'] == 'passed'
site = Jekyll::Site.new(Jekyll.configuration('source' => File.expand_path('..', __dir__), 'quiet' => true))
site.reset
site.read
report['post_urls'] = site.posts.docs.to_h { |post| [post.relative_path, post.url] }
abort 'No current Jekyll routes found' if report['post_urls'].empty?
report['routing_source'] = 'Current Jekyll site.read; not historical source-preservation evidence'
File.write(output, JSON.pretty_generate(report) + "\n")
puts "PASS current routing for #{report['post_urls'].length} posts"
