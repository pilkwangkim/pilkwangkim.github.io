# frozen_string_literal: true

require 'date'
require 'digest'
require 'json'
require 'securerandom'
require 'time'
require 'yaml'
require_relative 'source'

module LocalEditor
  # Repository-aware rules live outside the byte-range mapper. A metadata edit
  # is preflighted as a complete translation/edition group before any write.
  class Metadata
    FIELDS = %w[tags categories topic series series_order date lang].freeze
    SHARED = %w[tags categories topic series series_order].freeze
    LANGUAGE_TAGS = %w[korean english ko kr en].freeze
    SLUG = /\A[a-z0-9]+(?:-[a-z0-9]+)*\z/

    def initialize(root:)
      @root = File.realpath(root)
    end

    def enrich(source)
      records = inventory
      current = records.find { |entry| entry[:path] == source.path }
      raise InvalidEdit, 'The article is no longer in this blog.' unless current

      metadata = source.document.fetch(:metadata).dup
      metadata[:options] = options(records)
      metadata[:related] = related(records, current).map do |entry|
        { path: entry[:path], revision: entry[:revision], lang: entry[:data]['lang'],
          article_version: entry[:data]['article_version'] }
      end
      metadata
    end

    def save(source:, revision:, edits:, metadata:, related_revisions: {})
      raise InvalidEdit, 'Metadata changes must be an object.' unless metadata.is_a?(Hash)
      raise InvalidEdit, 'Related article revisions must be an object.' unless related_revisions.is_a?(Hash)

      unknown = metadata.keys - FIELDS
      raise InvalidEdit, "Unsupported metadata: #{unknown.join(', ')}." unless unknown.empty?

      Source.with_transaction(root: @root) do
        records = inventory
        current = records.find { |entry| entry[:path] == source.path }
        raise Conflict, 'The article changed after editing began. Refresh preview.' unless current && current[:revision] == revision

        changes = normalize(metadata, current[:data])
        # The UI sends only dirty fields. Preserve source ordering and original
        # date spelling when no semantic value actually changed.
        changes.delete_if { |key, value| equivalent?(key, current[:data][key], value) }
        shared_changes = changes.select { |key, _| SHARED.include?(key) }
        peers = shared_changes.empty? ? [] : related(records, current)
        peers.each do |entry|
          unless related_revisions[entry[:path]] == entry[:revision]
            raise Conflict, 'A translation or edition changed after editing began. Refresh preview before saving metadata.'
          end
        end

        affected = [current, *peers]
        prepared = affected.map do |entry|
          item = entry[:path] == source.path ? source : Source.new(root: @root, path: entry[:path])
          updates = entry[:path] == source.path ? changes : shared_changes
          [item, item.prepare_save(revision: entry[:revision],
                                   edits: entry[:path] == source.path ? edits : [], metadata: updates)]
        end
        proposed = records.map do |entry|
          change = prepared.find { |item, _| item.path == entry[:path] }
          change ? entry.merge(data: parse_front_matter(change[1].fetch(:after))) : entry
        end
        validate!(proposed, changed_paths: affected.map { |entry| entry[:path] }) unless changes.empty?
        commit_group(prepared, anchor: source.path)
      end
    end

    def batch_receipt?(receipt)
      receipt.is_a?(String) && receipt.match?(/\Abatch-[a-f0-9-]{36}\z/)
    end

    def undo(source:, revision:, receipt:)
      raise InvalidEdit, 'Invalid metadata undo receipt.' unless batch_receipt?(receipt)

      Source.with_transaction(root: @root) do
        record = JSON.parse(File.read(File.join(backup_directory, "#{receipt}.json")))
        unless record['type'] == 'metadata-batch' && record['anchor'] == source.path && record['components'].is_a?(Array)
          raise InvalidEdit, 'The metadata undo receipt does not match this article.'
        end
        raise Conflict, 'The article changed after this save. Refresh preview.' unless source.revision == revision

        prepared = record['components'].map do |component|
          item = Source.new(root: @root, path: component.fetch('path'))
          [item, item.prepare_undo(revision: component.fetch('after'), receipt: component.fetch('receipt'))]
        end
        records = inventory
        proposed = records.map do |entry|
          change = prepared.find { |item, _| item.path == entry[:path] }
          change ? entry.merge(data: parse_front_matter(change[1].fetch(:after))) : entry
        end
        validate!(proposed, changed_paths: prepared.map { |item, _| item.path })
        commit_group(prepared, anchor: source.path)
      end
    rescue Errno::ENOENT, JSON::ParserError, KeyError
      raise InvalidEdit, 'The metadata undo receipt or its backup could not be found.'
    end

    private

    def inventory
      Dir.glob(File.join(@root, '_posts', '**', '*.{md,markdown}')).sort.map do |absolute|
        relative = absolute.delete_prefix("#{@root}/")
        real = File.realpath(absolute)
        posts = File.realpath(File.join(@root, '_posts'))
        raise InvalidPath, 'Posts must remain inside the blog.' unless real.start_with?("#{posts}/") && !File.symlink?(absolute)

        raw = File.binread(absolute)
        { path: relative, revision: Digest::SHA256.hexdigest(raw), data: parse_front_matter(raw) }
      end
    end

    def parse_front_matter(raw)
      text = raw.dup.force_encoding(Encoding::UTF_8)
      match = text.match(/\A---[ \t]*\r?\n(.*?)^---[ \t]*\r?\n/m)
      raise InvalidEdit, 'An article has invalid YAML front matter.' unless match

      data = YAML.safe_load(match[1], permitted_classes: [Date, Time], aliases: true)
      raise InvalidEdit, 'Article front matter must be a mapping.' unless data.is_a?(Hash)

      data
    rescue Psych::Exception => e
      raise InvalidEdit, "The article metadata is invalid: #{e.message}"
    end

    def configuration(name, fallback)
      path = File.join(@root, '_data', "#{name}.yml")
      File.file?(path) ? YAML.safe_load(File.read(path), permitted_classes: [Date, Time], aliases: true) : fallback
    end

    def definitions
      topics = configuration('topics', [])
      series = configuration('series', [])
      aliases = configuration('tag_aliases', {})
      unless topics.is_a?(Array) && series.is_a?(Array) && aliases.is_a?(Hash)
        raise InvalidEdit, 'The blog topic, series or tag-alias configuration is invalid.'
      end
      [topics, series, aliases]
    end

    def options(records)
      topics, series, = definitions
      {
        tags: records.flat_map { |entry| Array(entry[:data]['tags']) }.uniq.sort,
        categories: records.flat_map { |entry| Array(entry[:data]['categories']) }.uniq.sort,
        topics: topics.map { |item| item.slice('id', 'title', 'tag') },
        series: series.map { |item| item.slice('id', 'title', 'topic') },
        languages: [{ id: 'en', title: 'English' }, { id: 'ko', title: '한국어' }]
      }
    end

    def related(records, current)
      key = current[:data]['translation_key']
      return [] unless key.is_a?(String) && !key.empty?

      records.select { |entry| entry[:path] != current[:path] && entry[:data]['translation_key'] == key }
    end

    def normalize(changes, current)
      result = changes.dup
      topics, series, aliases = definitions
      if result.key?('topic')
        raise InvalidEdit, 'Choose a registered topic.' unless topics.any? { |item| item['id'] == result['topic'] }
      end
      if result.key?('tags') || result.key?('topic')
        values = result.fetch('tags', current['tags'])
        raise InvalidEdit, 'Tags must be a list of at most 64 names.' unless values.is_a?(Array) && values.length <= 64

        tags = values.map do |tag|
          raise InvalidEdit, 'Each tag must be a name.' unless tag.is_a?(String) && tag.bytesize <= 100

          slug = tag.strip.downcase.gsub(/\s+/, '-')
          aliases.fetch(slug, slug)
        end
        raise InvalidEdit, 'Use lowercase words joined with hyphens for tags.' unless tags.all? { |tag| tag.is_a?(String) && tag.match?(SLUG) }
        raise InvalidEdit, 'Set article language with Language, not a language tag.' unless (tags & LANGUAGE_TAGS).empty?

        topic = topics.find { |item| item['id'] == result.fetch('topic', current['topic']) }
        raise InvalidEdit, 'Choose a registered topic before editing tags.' unless topic

        primary = topic['tag']
        result['tags'] = [primary, *(tags - [primary]).uniq.sort].compact
      end
      if result.key?('categories')
        values = result['categories']
        unless values.is_a?(Array) && values.length.between?(1, 8) && values.all? { |value| safe_name?(value) }
          raise InvalidEdit, 'Categories need 1–8 plain-text names of up to 80 characters.'
        end
        result['categories'] = values.map(&:strip).uniq
      end
      if result.key?('series')
        result['series'] = nil if result['series'] == ''
        unless result['series'].nil? || series.any? { |item| item['id'] == result['series'] }
          raise InvalidEdit, 'Choose a registered series or None.'
        end
        result['series_order'] = nil if result['series'].nil?
      end
      if result.key?('series_order')
        result['series_order'] = nil if result['series_order'] == ''
        value = result['series_order']
        unless value.nil? || (value.is_a?(Integer) && value.between?(0, 1_000_000))
          raise InvalidEdit, 'Series order must be a whole number from 0 to 1,000,000.'
        end
      end
      result['date'] = normalize_date(result['date']) if result.key?('date')
      if result.key?('lang') && !%w[en ko].include?(result['lang'])
        raise InvalidEdit, 'Article language must be English or 한국어.'
      end
      result
    end

    def safe_name?(value)
      value.is_a?(String) && value.strip.length.between?(1, 80) && !value.match?(/[\r\n\x00-\x1f<>`{}]/)
    end

    def normalize_date(value)
      unless value.is_a?(String) && (match = value.strip.match(/\A(\d{4})-(\d{2})-(\d{2})[T ](\d{2}):(\d{2})(?::(\d{2}))?\s*(Z|[+-]\d{2}:?\d{2})\z/))
        raise InvalidEdit, 'Use a date, time and explicit timezone, such as 2026-10-24 10:00:00 +0900.'
      end
      year, month, day, hour, minute, second = match.captures.first(6).map { |part| part.to_i }
      zone = match[7]
      zone_hours, zone_minutes = zone == 'Z' ? [0, 0] : [zone[1, 2].to_i, zone.delete(':')[-2, 2].to_i]
      unless Date.valid_date?(year, month, day) && hour < 24 && minute < 60 && second < 60 &&
             zone_hours <= 14 && zone_minutes < 60 && (zone_hours < 14 || zone_minutes.zero?)
        raise InvalidEdit, 'The publication date or timezone is invalid.'
      end
      offset = (zone_hours * 60 + zone_minutes) * 60 * (zone.start_with?('-') ? -1 : 1)
      DateTime.new(year, month, day, hour, minute, second, Rational(offset, 86_400)).strftime('%Y-%m-%d %H:%M:%S %z')
    end

    def equivalent?(key, old, value)
      return old.nil? && value.nil? if old.nil? || value.nil?
      return date_value(old) == date_value(value) if key == 'date'

      old == value
    end

    def date_value(value)
      value.respond_to?(:strftime) ? value.strftime('%Y-%m-%d %H:%M:%S %z') : normalize_date(value.to_s)
    rescue ArgumentError
      value
    end

    def validate!(records, changed_paths:)
      topics, series, aliases = definitions
      by_topic = topics.to_h { |item| [item['id'], item] }
      by_series = series.to_h { |item| [item['id'], item] }
      records.select { |entry| changed_paths.include?(entry[:path]) }.each do |entry|
        data = entry[:data]
        tags = data['tags']
        unless tags.is_a?(Array) && !tags.empty? && tags.uniq == tags && tags.all? { |tag| tag.is_a?(String) && tag.match?(SLUG) } && (tags & LANGUAGE_TAGS).empty?
          raise InvalidEdit, 'Tags must be unique canonical names, without language tags.'
        end
        topic = by_topic[data['topic']]
        raise InvalidEdit, 'Choose a registered topic and include its topic tag.' unless topic && tags.include?(topic['tag'])
        unless data['categories'].is_a?(Array) && data['categories'].length.between?(1, 8) && data['categories'].all? { |name| safe_name?(name) }
          raise InvalidEdit, 'Categories must be a nonempty list of plain-text names.'
        end
        raise InvalidEdit, 'Article language must be English or 한국어.' unless %w[en ko].include?(data['lang'])
        if data['series']
          definition = by_series[data['series']]
          raise InvalidEdit, 'The selected series must belong to the selected topic.' unless definition && definition['topic'] == data['topic']
          order = data['series_order']
          raise InvalidEdit, 'A series needs a nonnegative whole-number order.' unless order.is_a?(Integer) && order >= 0
        elsif !data['series_order'].nil?
          raise InvalidEdit, 'Choose a series before setting series order.'
        end
      end
      records.group_by { |entry| entry[:data]['translation_key'] }.each do |key, entries|
        next unless key && entries.any? { |entry| changed_paths.include?(entry[:path]) }

        slots = entries.map { |entry| entry[:data].values_at('lang', 'article_version') }
        raise InvalidEdit, 'This translation group already has that language and article version.' unless slots.uniq == slots
        versions = entries.group_by { |entry| entry[:data]['article_version'] }
        if versions.keys.any? { |version| !version.nil? }
          unless !versions.key?(nil) && versions.keys.sort == %w[compact original] && versions.values.all? { |edition| edition.any? { |entry| entry[:data]['lang'] == 'en' } }
            raise InvalidEdit, 'Original and Compact editions must each retain an English article.'
          end
        end
        %w[tags topic series series_order].each do |field|
          values = entries.map { |entry| entry[:data][field] }.map { |value| value.is_a?(Array) ? value.sort : value }
          raise InvalidEdit, "Translations and editions must share #{field}." unless values.uniq.length <= 1
        end
      end
      records.select { |entry| entry[:data]['series'] }.group_by do |entry|
        entry[:data].values_at('series', 'lang', 'article_version')
      end.each do |_, entries|
        next unless entries.any? { |entry| changed_paths.include?(entry[:path]) }

        orders = entries.map { |entry| entry[:data]['series_order'] }
        raise InvalidEdit, 'That series order is already used by another article in this language and edition.' unless orders.uniq == orders
      end
      tags = records.flat_map { |entry| Array(entry[:data]['tags']) }.uniq
      aliases.each do |legacy, target|
        raise InvalidEdit, "Keep at least one article tagged #{target}; /tags/#{legacy}/ redirects there." unless tags.include?(target)
      end
    end

    def backup_directory
      work = File.join(@root, '.local-editor')
      FileUtils.mkdir_p(work, mode: 0o700)
      raise InvalidPath, 'The editor data directory cannot be a symlink.' unless File.realpath(work) == work

      directory = File.join(work, 'backups')
      FileUtils.mkdir_p(directory, mode: 0o700)
      raise InvalidPath, 'The backup directory cannot be a symlink.' unless File.realpath(directory) == directory

      directory
    end

    def commit_group(prepared, anchor:)
      components = []
      prepared.each do |item, plan|
        next if plan.fetch(:changed).zero?

        result = item.commit_prepared!(plan, lock: false)
        components << { path: item.path, receipt: result.fetch(:receipt),
                        before: Digest::SHA256.hexdigest(plan.fetch(:before)),
                        after: result.fetch(:document).fetch(:revision) }
      end
      current = Source.new(root: @root, path: anchor)
      return { document: current.document, receipt: nil, changed: 0, affected_paths: [] } if components.empty?

      receipt = "batch-#{SecureRandom.uuid}"
      record = { type: 'metadata-batch', anchor: anchor, components: components, created_at: Time.now.utc.iso8601 }
      File.write(File.join(backup_directory, "#{receipt}.json"), JSON.generate(record), mode: 'w', perm: 0o600)
      { document: current.document, receipt: receipt,
        changed: prepared.sum { |_, plan| plan.fetch(:changed) }, affected_paths: components.map { |part| part[:path] } }
    rescue StandardError => error
      rollback_errors = []
      components.reverse_each do |component|
        begin
          item = Source.new(root: @root, path: component[:path])
          plan = item.prepare_undo(revision: component[:after], receipt: component[:receipt])
          item.commit_prepared!(plan, lock: false)
        rescue StandardError => rollback_error
          rollback_errors << rollback_error.message
        end
      end
      unless rollback_errors.empty?
        raise Error, "The metadata save stopped; an externally changed file could not be rolled back: #{rollback_errors.join('; ')}"
      end
      raise error
    end
  end
end
