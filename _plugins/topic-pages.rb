# frozen_string_literal: true

require 'time'

module Jekyll
  # Keep the hand-written topic introduction separate from automatically grouped
  # posts. Only the Topics document belongs to the tabs collection.
  class TopicPages < Generator
    safe true
    priority :lowest

    def generate(site)
      add_topic_labels(site)
      posts = site.posts.docs.reject { |post| post.data['hidden'] == true }
      topics = Array(site.data['topics'])
      categories = Array(site.data['topic_categories'])
      parent_by_id = category_parents(categories)
      site.data['topic_category_by_id'] = parent_by_id
      index = topics.filter_map do |topic|
        next if topic['group'] == 'reference'

        topic_posts = posts.select { |post| post.data['topic'] == topic['id'] }
        next if topic_posts.empty?

        articles = group_articles(topic_posts)
        recommended = recommended_articles(topic, articles)
        url = "/topics/#{topic['id']}/"
        page = PageWithoutAFile.new(site, site.source, "topics/#{topic['id']}", 'index.html')
        page.data.merge!(
          'layout' => 'topic', 'title' => topic['title'],
          'description' => topic['description'], 'permalink' => url,
          'lang' => 'en', 'topic_entry' => topic, 'articles' => articles,
          'recommended_articles' => recommended, 'post_count' => topic_posts.size,
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
        members = member_ids.map do |id|
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
          'lang' => 'en', 'category_entry' => category, 'member_topics' => members,
          'articles' => articles, 'recommended_articles' => recommended_articles(category, articles),
          'post_count' => category_posts.size
        )
        site.pages << page
        category.merge('url' => url, 'article_count' => articles.size,
                       'post_count' => category_posts.size,
                       'languages' => category_posts.map { |post| post.data['lang'] }.uniq.sort)
      end
    end

    def recommended_articles(entry, articles)
      by_key = articles.to_h { |article| [article['translation_key'], article] }
      Array(entry['recommended']).map do |key|
        by_key.fetch(key) do
          raise Errors::FatalException, "Topic #{entry['id']}: unknown recommended translation_key #{key}"
        end
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
        versions.sort_by! { |post| [post.data['lang'] == 'ko' ? 0 : 1, post.url] }
        {
          'translation_key' => key,
          'versions' => versions.map do |post|
            { 'title' => post.data['title'], 'url' => post.url, 'lang' => post.data['lang'] }
          end,
          'languages' => versions.map { |post| post.data['lang'] },
          'primary_language' => versions.first.data['lang'],
          'date' => versions.map(&:date).max,
          'modified_at' => versions.map { |post| modified_at(post) }.max
        }
      end.sort_by { |article| [-article['date'].to_f, article['translation_key'].to_s] }
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
