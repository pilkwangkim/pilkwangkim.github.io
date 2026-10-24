# frozen_string_literal: true

# This integration is activated only by tools/edit.sh. Normal development and
# production builds do not load the editor, source maps, or filesystem API.
if ENV['BLOG_LOCAL_EDITOR'] == '1' && Jekyll.env == 'development'
  require 'json'
  require 'fileutils'
  require 'nokogiri'
  require_relative '../tools/local_editor/source'
  require 'jekyll/converters/markdown/kramdown_parser'

  class Jekyll::Converters::Markdown::LocalEditor < Jekyll::Converters::Markdown::KramdownParser
    def convert(content)
      source = Thread.current[:blog_local_editor_source]
      Thread.current[:blog_local_editor_source] = nil
      return super unless source

      document = Kramdown::JekyllDocument.new(content, @config)
      source.annotate(document.root, input: content)
      document.to_html
    end
  end

  # Excerpts can populate Jekyll's Markdown cache before post rendering. The
  # editable body must always be converted with this document's source map.
  module LocalEditor::MappedConversion
    def convert(content)
      return super unless Thread.current[:blog_local_editor_source]

      setup
      @parser.convert(content)
    end
  end
  Jekyll::Converters::Markdown.prepend(LocalEditor::MappedConversion)

  Jekyll::Hooks.register :site, :after_init do |site|
    site.config['markdown'] = 'LocalEditor'
    site.config['pwa'] = { 'enabled' => false, 'cache' => { 'enabled' => false } }
  end

  Jekyll::Hooks.register :posts, :pre_render do |post|
    # The local equation/table preview also needs the site's existing MathJax
    # runtime when the saved article has not previously used mathematics.
    post.data['math'] = true
    source = LocalEditor::Source.new(root: post.site.source, path: post.relative_path.delete_prefix('/'))
    post.instance_variable_set(:@blog_local_editor_source, source)
    Thread.current[:blog_local_editor_source] = source
  end

  Jekyll::Hooks.register :posts, :post_render do |post|
    Thread.current[:blog_local_editor_source] = nil
    source = post.instance_variable_get(:@blog_local_editor_source)
    next unless source

    html = Nokogiri::HTML(post.output)
    article = html.at_css('main article')
    next unless article

    fields = source.document.fetch(:fields)
    {
      'title' => 'header > h1',
      'description' => 'header > .post-desc',
      'image.alt' => 'header figcaption'
    }.each do |field, selector|
      node = article.at_css(selector)
      entry = fields[field]
      next unless node && entry && node.element_children.empty? && node.text == entry[:text]

      node['data-local-edit-id'] = entry[:id]
      node['data-local-edit-kind'] = field == 'image.alt' ? 'caption' : 'metadata'
      node['class'] = [node['class'], 'tex2jax_ignore'].compact.join(' ')
    end

    context = Nokogiri::XML::Node.new('script', html)
    context['id'] = 'local-editor-context'
    context['type'] = 'application/json'
    context.content = JSON.generate(url: post.url).gsub('<', '\\u003c')
    html.at_css('body').add_child(context)
    stylesheet = Nokogiri::XML::Node.new('link', html)
    stylesheet['rel'] = 'stylesheet'
    stylesheet['href'] = '/__local_editor/style.css'
    html.at_css('head').add_child(stylesheet)
    script = Nokogiri::XML::Node.new('script', html)
    script['src'] = '/__local_editor/client.js'
    script['defer'] = 'defer'
    html.at_css('body').add_child(script)
    post.output = html.to_html
  end

  Jekyll::Hooks.register :site, :post_write do |site|
    path = ENV['BLOG_LOCAL_EDITOR_MAP_PATH']
    next if path.to_s.empty?

    documents = site.posts.docs.to_h do |post|
      source = post.instance_variable_get(:@blog_local_editor_source)
      [post.url, { path: post.relative_path.delete_prefix('/'), revision: source&.document&.fetch(:revision) }]
    end
    FileUtils.mkdir_p(File.dirname(path))
    File.write(path, JSON.generate(documents: documents))
  end
end
