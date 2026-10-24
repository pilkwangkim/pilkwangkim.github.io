# frozen_string_literal: true

require 'digest'

Jekyll::Hooks.register :site, :post_read do |site|
  registry = {}
  bundles = {}
  site.data.each do |key, definitions|
    next unless key.start_with?('article_labs_') && definitions.is_a?(Hash)

    definitions.each do |id, definition|
      raise Jekyll::Errors::FatalException, "Duplicate article lab #{id}" if registry.key?(id)
      unless definition.is_a?(Hash) && definition['bundle'].to_s.match?(/\A[a-z][a-z0-9-]*\z/) &&
             %w[en ko].all? { |language| definition.dig('title', language).is_a?(String) }
        raise Jekyll::Errors::FatalException, "Invalid article lab #{id}"
      end

      registry[id] = definition
      bundles[definition['bundle']] = true
    end
  end
  site.data['article_labs'] = registry
  common = %w[assets/css/article-labs.css assets/js/article-labs.js _includes/interactive/article-lab.html]
  content = common.map { |path| path + "\0" + File.binread(File.join(site.source, path)) + "\0" }.join
  site.data['article_labs_version'] = Digest::SHA256.hexdigest(content)[0, 12]
  site.data['article_lab_bundle_versions'] = bundles.keys.to_h do |bundle|
    path = "assets/js/article-labs-#{bundle}.js"
    raise Jekyll::Errors::FatalException, "Missing article lab bundle #{path}" unless File.file?(File.join(site.source, path))

    definitions = registry.select { |_, definition| definition['bundle'] == bundle }
    dependencies = definitions.values.flat_map { |definition| Array(definition['data']) }.uniq.sort.map do |asset|
      unless asset.is_a?(String) && asset.start_with?('/assets/') && !asset.split('/').include?('..')
        raise Jekyll::Errors::FatalException, "Invalid article lab data path #{asset.inspect}"
      end
      asset + "\0" + File.binread(File.join(site.source, asset.delete_prefix('/'))) + "\0"
    end.join
    [bundle, Digest::SHA256.hexdigest(content + File.binread(File.join(site.source, path)) + definitions.to_s.b + dependencies)[0, 12]]
  end
  site.posts.docs.each do |post|
    declared = Array(post.data['interactive_labs'])
    declared.each do |bundle|
      raise Jekyll::Errors::FatalException, "Unknown article lab bundle #{bundle} in #{post.relative_path}" unless bundles[bundle]
    end
    post.content.scan(/\{%\s*include\s+interactive\/article-lab\.html\s+id=["']([^"']+)["']/).flatten.each do |id|
      definition = registry[id]
      raise Jekyll::Errors::FatalException, "Unknown article lab #{id} in #{post.relative_path}" unless definition
      unless declared.include?(definition['bundle'])
        raise Jekyll::Errors::FatalException, "Undeclared bundle for #{id} in #{post.relative_path}"
      end
    end
  end
end
