# frozen_string_literal: true

require 'time'
require 'digest'

module Jekyll
  # Keep the hand-written topic introduction separate from automatically grouped
  # posts. Only the Topics document belongs to the tabs collection.
  class TopicPages < Generator
    safe true
    priority :lowest

    def generate(site)
      asset_sources = ['assets/js/blog-navigation.js', 'assets/css/jekyll-theme-chirpy.scss', 'Gemfile',
                       *Dir.glob('_sass/**/*.scss', base: site.source)].sort
      asset_content = asset_sources.map do |path|
        path + "\0" + File.binread(File.join(site.source, path)) + "\0"
      end.join
      site.data['blog_asset_version'] = Digest::SHA256.hexdigest(asset_content)[0, 12]
      add_topic_labels(site)
      posts = site.posts.docs.reject { |post| post.data['hidden'] == true }
      topics = Array(site.data['topics'])
      categories = Array(site.data['topic_categories'])
      @topics_by_id = topics.to_h { |topic| [topic['id'], topic] }
      @series_by_id = Array(site.data['series']).to_h { |series| [series['id'], series] }
      parent_by_id = category_parents(categories)
      site.data['topic_category_by_id'] = parent_by_id
      index = topics.filter_map do |topic|
        next if topic['group'] == 'reference'

        topic_posts = posts.select { |post| post.data['topic'] == topic['id'] }
        next if topic_posts.empty?

        articles = group_articles(topic_posts)
        url = "/topics/#{topic['id']}/"
        page = PageWithoutAFile.new(site, site.source, "topics/#{topic['id']}", 'index.html')
        page.data.merge!(
          'layout' => 'topic', 'title' => topic['title'],
          'description' => topic['description'], 'permalink' => url,
          'lang' => 'en', 'topic_entry' => topic, 'articles' => articles,
          'article_groups' => article_groups(articles, group_topics: topic['group'] == 'ai'),
          'post_count' => topic_posts.size,
          'parent_category' => parent_by_id[topic['id']]
        )
        site.pages << page
        topic.merge('url' => url, 'article_count' => articles.size,
                    'post_count' => topic_posts.size,
                    'languages' => topic_posts.map { |post| post.data['lang'] }.uniq.sort)
      end
      site.data['topic_index'] = index
      category_index = generate_categories(site, categories, index, posts)
      site.data['topic_navigation'] = category_index + index.reject { |topic| topic['group'] == 'ai' }
      site.data['recently_updated'] = group_articles(posts).sort_by do |article|
        [-article['modified_at'].to_f, -article['date'].to_f, article['translation_key'].to_s]
      end.first(5)
      generate_tag_aliases(site, posts)
    end

    private

    def category_parents(categories)
      categories.each_with_object({}) do |category, parents|
        parent = category.slice('id', 'title', 'group').merge('url' => "/topics/#{category['id']}/")
        Array(category['topics']).each do |id|
          if parents.key?(id)
            raise Errors::FatalException, "Topic #{id}: assigned to multiple navigation categories"
          end

          parents[id] = parent
        end
      end
    end

    def generate_categories(site, categories, topic_index, posts)
      topics_by_id = topic_index.to_h { |topic| [topic['id'], topic] }
      categories.map do |category|
        member_ids = Array(category['topics'])
        member_ids.each do |id|
          topics_by_id.fetch(id) do
            raise Errors::FatalException, "Category #{category['id']}: unknown or empty topic #{id}"
          end
        end
        category_posts = posts.select { |post| member_ids.include?(post.data['topic']) }
        articles = group_articles(category_posts)
        url = "/topics/#{category['id']}/"
        page = PageWithoutAFile.new(site, site.source, "topics/#{category['id']}", 'index.html')
        page.data.merge!(
          'layout' => 'topic', 'title' => category['title'],
          'description' => category['description'], 'permalink' => url,
          'lang' => 'en', 'category_entry' => category,
          'articles' => articles,
          'article_groups' => article_groups(articles, group_topics: true),
          'post_count' => category_posts.size
        )
        site.pages << page
        category.merge('url' => url, 'article_count' => articles.size,
                       'post_count' => category_posts.size,
                       'languages' => category_posts.map { |post| post.data['lang'] }.uniq.sort)
      end
    end

    def add_topic_labels(site)
      locales = site.data['locales'] || {}
      # Chirpy ships Korean as ko-KR; posts use the standard short language code.
      locales['ko'] ||= locales['ko-KR']
      { 'en' => 'Topics', 'ko' => 'Topics', 'ko-KR' => 'Topics' }.each do |locale, label|
        next unless locales[locale].is_a?(Hash)

        (locales[locale]['tabs'] ||= {})['topics'] = label
      end
    end

    def group_articles(posts)
      posts.group_by { |post| post.data['translation_key'] || post.url }.map do |key, versions|
        versions.sort_by! do |post|
          [post.data['lang'] == 'en' ? 0 : 1, edition_order(post.data['article_version']), post.url]
        end
        article_versions = versions.map do |post|
          { 'title' => post.data['title'], 'url' => post.url, 'lang' => post.data['lang'],
            'article_version' => post.data['article_version'] }
        end
        preferred_versions = article_versions.group_by { |version| version['lang'] }.values.map(&:first)
        editions = if versions.any? { |post| post.data['article_version'] }
                     article_versions.group_by { |version| version['article_version'] }
                                     .sort_by { |edition, _| edition_order(edition) }
                                     .map do |edition, members|
                       { 'id' => edition, 'label' => edition == 'original' ? 'Original' : 'Compact',
                         'versions' => members, 'languages' => members.map { |version| version['lang'] }.uniq }
                     end
                   end
        {
          'translation_key' => key,
          'versions' => article_versions,
          'preferred_versions' => preferred_versions,
          'editions' => editions,
          'languages' => versions.map { |post| post.data['lang'] }.uniq,
          'primary_language' => versions.first.data['lang'],
          'topic' => versions.first.data['topic'],
          'series' => versions.first.data['series'],
          'series_order' => versions.first.data['series_order'],
          'date' => versions.map(&:date).max,
          'modified_at' => versions.map { |post| modified_at(post) }.max
        }
      end.sort_by { |article| [-article['date'].to_f, article['translation_key'].to_s] }
    end

    def edition_order(edition)
      edition == 'compact' ? 1 : 0
    end

    def article_groups(articles, group_topics: false)
      articles.group_by do |article|
        if group_topics
          "topic:#{article['topic']}"
        else
          article['series'] ? "series:#{article['series']}" : "article:#{article['translation_key']}"
        end
      end.map do |key, members|
        topic_id = members.first['topic']
        series = @series_by_id.values.find do |definition|
          members.any? { |article| article['series'] == definition['id'] }
        end
        bundle_id = group_topics ? (series && series['id']) || topic_id : series && series['id']
        title = group_topics ? @topics_by_id.fetch(topic_id)['title'] : series && series['title']
        {
          'key' => key, 'bundle_id' => bundle_id, 'topic_id' => topic_id, 'title' => title,
          'articles' => reading_order(members),
          'latest_articles' => members.sort_by { |article| [-article['date'].to_f, article['translation_key'].to_s] },
          'languages' => members.flat_map { |article| article['languages'] }.uniq,
          'date' => members.map { |article| article['date'] }.max
        }
      end.sort_by { |group| [-group['date'].to_f, group['key']] }
    end

    # Topic bundles can contain a numbered series alongside independent notes.
    # Keep each real sequence intact without assigning new part numbers.
    def reading_order(articles)
      sequences = articles.group_by do |article|
        article['series'] ? "series:#{article['series']}" : "article:#{article['translation_key']}"
      end
      sequences.sort_by do |key, members|
        [members.map { |article| article['date'].to_f }.min, key]
      end.flat_map do |_, members|
        members.sort_by do |article|
          [article['series_order'] || 0, article['date'].to_f, article['translation_key'].to_s]
        end
      end
    end

    def modified_at(post)
      Time.parse(post.data['last_modified_at'].to_s)
    rescue ArgumentError
      post.date
    end

    # Preserve incoming links to old tag slugs while the current Tags page only
    # exposes the canonical vocabulary. These aliases are ordinary archive pages.
    def generate_tag_aliases(site, posts)
      occupied = site.pages.map(&:url)
      (site.data['tag_aliases'] || {}).each do |legacy_slug, canonical_tag|
        alias_posts = posts.select { |post| Array(post.data['tags']).include?(canonical_tag) }
        add_tag_alias(site, occupied, legacy_slug, canonical_tag, alias_posts)
      end
      korean_posts = posts.select { |post| post.data['lang'] == 'ko' }
      add_tag_alias(site, occupied, 'korean', 'Korean articles', korean_posts, 'en')
    end

    def add_tag_alias(site, occupied, slug, title, posts, lang = nil)
      url = "/tags/#{slug}/"
      return if occupied.include?(url) || posts.empty?

      page = PageWithoutAFile.new(site, site.source, "tags/#{slug}", 'index.html')
      page.data.merge!(
        'layout' => 'tag', 'title' => title, 'permalink' => url,
        'posts' => posts.sort_by { |post| [-post.date.to_f, post.data['translation_key'].to_s, post.data['lang'] == 'ko' ? 0 : 1, post.url] },
        'sitemap' => false
      )
      page.data['lang'] = lang if lang
      site.pages << page
      occupied << url
    end
  end
end
