#!/usr/bin/env ruby
# frozen_string_literal: true

# All source writes here are confined to isolated temporary blog fixtures.
require 'tmpdir'
require 'fileutils'
require_relative 'metadata'

module MetadataCommitFailure
  def commit_prepared!(prepared, lock: true)
    if prepared[:action] == (Thread.current[:metadata_test_fail_action] || 'save') && Thread.current[:metadata_test_fail_path] == path
      Thread.current[:metadata_test_fail_path] = nil
      raise LocalEditor::Error, 'Simulated peer write failure'
    end
    super
  end
end
LocalEditor::Source.prepend(MetadataCommitFailure)

class MetadataTest
  attr_reader :assertions

  def assert(value, message = 'Assertion failed')
    @assertions = (@assertions || 0) + 1
    raise message unless value
  end

  def assert_equal(expected, actual, message = nil)
    assert(expected == actual, message || "Expected #{expected.inspect}, got #{actual.inspect}")
  end

  def assert_nil(actual)
    assert(actual.nil?, "Expected nil, got #{actual.inspect}")
  end

  def assert_match(pattern, actual)
    assert(pattern.match?(actual.to_s), "Expected #{actual.inspect} to match #{pattern.inspect}")
  end

  def assert_includes(actual, expected)
    assert(actual.include?(expected), "Expected #{actual.inspect} to include #{expected.inspect}")
  end

  def assert_raises(error, message = nil)
    raised = nil
    begin
      yield
    rescue error => exception
      raised = exception
    end
    assert(raised, message || "Expected #{error}")
    raised
  end
  def setup
    @root = Dir.mktmpdir('local-editor-metadata-')
    FileUtils.mkdir_p(File.join(@root, '_posts'))
    FileUtils.mkdir_p(File.join(@root, '_data'))
    File.write(File.join(@root, '_data/topics.yml'), YAML.dump([
      { 'id' => 'research', 'title' => 'Research', 'tag' => 'research' },
      { 'id' => 'science', 'title' => 'Science', 'tag' => 'science' }
    ]))
    File.write(File.join(@root, '_data/series.yml'), YAML.dump([
      { 'id' => 'research-notes', 'title' => 'Research Notes', 'topic' => 'research' },
      { 'id' => 'science-notes', 'title' => 'Science Notes', 'topic' => 'science' }
    ]))
    File.write(File.join(@root, '_data/tag_aliases.yml'), YAML.dump('cv' => 'cross-validation'))
    @english = add_post('Test', lang: 'en', key: 'pair', order: 0)
    @korean = add_post('Test-KR', lang: 'ko', key: 'pair', order: 0)
    @other = add_post('Other', lang: 'en', key: 'other', order: 1)
    @service = LocalEditor::Metadata.new(root: @root)
  end

  def teardown
    Thread.current[:metadata_test_fail_path] = nil
    Thread.current[:metadata_test_fail_action] = nil
    FileUtils.remove_entry(@root)
  end

  def add_post(name, lang:, key:, order: nil, version: nil)
    data = { 'title' => name, 'date' => Time.new(2026, 10, 20, 10, 0, 0, '+09:00'),
             'categories' => %w[AI Kaggle], 'tags' => %w[research cross-validation],
             'topic' => 'research', 'lang' => lang, 'translation_key' => key }
    data.merge!('series' => 'research-notes', 'series_order' => order) unless order.nil?
    data['article_version'] = version if version
    path = "_posts/2026-10-20-#{name}.md"
    File.write(File.join(@root, path), YAML.dump(data) + "---\n\nA clear sentence.\n\n$$x^2 + y^2$$\n\n| x | y |\n|---|---|\n| 1 | 2 |\n")
    path
  end

  def source(path = @english)
    LocalEditor::Source.new(root: @root, path: path)
  end

  def bytes(path)
    File.binread(File.join(@root, path))
  end

  def values(path)
    source(path).metadata_values
  end

  def save(changes = {}, path: @english, edits: [], related: nil, **keyword_changes)
    changes = changes.merge(keyword_changes)
    item = source(path)
    peers = related || @service.enrich(item).fetch(:related).to_h { |peer| [peer[:path], peer[:revision]] }
    @service.save(source: item, revision: item.revision, edits: edits, metadata: changes, related_revisions: peers)
  end

  def test_options_and_related_snapshot
    metadata = @service.enrich(source)
    assert_equal %w[AI Kaggle], metadata.dig(:options, :categories)
    assert_equal %w[en ko], metadata.dig(:options, :languages).map { |item| item[:id] }
    assert_equal [@korean], metadata[:related].map { |item| item[:path] }
    assert_equal source(@korean).revision, metadata[:related].first[:revision]
    assert_equal 'pair', source.front_matter['translation_key']
  end

  def test_shared_tags_aliases_and_exact_group_undo
    before = [bytes(@english), bytes(@korean), bytes(@other)]
    saved = save('tags' => ['CV', 'Research', 'new-tag', 'new-tag'])
    assert_match(/\Abatch-/, saved[:receipt])
    assert_equal %w[research cross-validation new-tag], values(@english)['tags']
    assert_equal values(@english)['tags'], values(@korean)['tags']
    assert_equal before[2], bytes(@other)
    assert_equal [@english, @korean], saved[:affected_paths]
    @service.undo(source: source, revision: source.revision, receipt: saved[:receipt])
    assert_equal before, [bytes(@english), bytes(@korean), bytes(@other)]
  end

  def test_categories_synchronize_all_translation_editions
    original_en = add_post('Test-Original', lang: 'en', key: 'pair', order: 0, version: 'original')
    original_ko = add_post('Test-Original-KR', lang: 'ko', key: 'pair', order: 0, version: 'original')
    [@english, @korean].each do |path|
      File.write(File.join(@root, path), bytes(path).sub('translation_key: pair', "translation_key: pair\narticle_version: compact"))
    end
    saved = save('categories' => ['Science', 'Methods'])
    [@english, @korean, original_en, original_ko].each do |path|
      assert_equal %w[Science Methods], values(path)['categories']
    end
    assert_equal 4, saved[:affected_paths].length
  end

  def test_date_changes_only_selected_article_and_future_is_supported
    korean_before = bytes(@korean)
    tags_before = source.metadata_values['tags']
    saved = save('date' => '2099-10-24T15:31:00+09:00')
    assert_equal '2099-10-24 15:31:00 +0900', values(@english)['date']
    assert_equal korean_before, bytes(@korean)
    assert_equal tags_before, values(@english)['tags']
    assert_equal [@english], saved[:affected_paths]
  end

  def test_language_changes_only_selected_unpaired_article
    before = bytes(@english)
    save({ 'lang' => 'ko' }, path: @other)
    assert_equal 'ko', values(@other)['lang']
    assert_equal before, bytes(@english)
  end

  def test_timezone_change_with_same_instant_is_saved_and_undone
    before = bytes(@english)
    saved = save('date' => '2026-10-20 01:00:00 +0000')
    assert_equal '2026-10-20 01:00:00 +0000', values(@english)['date']
    assert_equal [@english], saved[:affected_paths]
    @service.undo(source: source, revision: source.revision, receipt: saved[:receipt])
    assert_equal before, bytes(@english)
  end

  def test_duplicate_language_and_versioned_english_protection
    assert_raises(LocalEditor::InvalidEdit) { save('lang' => 'ko') }
    add_post('Solo-Original', lang: 'en', key: 'versioned', version: 'original')
    compact = add_post('Solo-Compact', lang: 'en', key: 'versioned', version: 'compact')
    assert_raises(LocalEditor::InvalidEdit) { save({ 'lang' => 'ko' }, path: compact) }
  end

  def test_series_validation_and_removal
    original = bytes(@english)
    assert_raises(LocalEditor::InvalidEdit) { save('series_order' => 1) }
    assert_raises(LocalEditor::InvalidEdit) { save('series' => 'science-notes') }
    assert_raises(LocalEditor::InvalidEdit) { save('series_order' => 0.5) }
    assert_equal original, bytes(@english)
    save('series' => nil)
    assert_nil values(@english)['series']
    assert_nil values(@korean)['series']
    assert_nil values(@english)['series_order']
    assert_raises(LocalEditor::InvalidEdit) { save('series_order' => 2) }
  end

  def test_topic_adds_its_tag_and_requires_matching_series
    assert_raises(LocalEditor::InvalidEdit) { save('topic' => 'science') }
    save('topic' => 'science', 'series' => 'science-notes', 'series_order' => 0)
    assert_equal 'science', values(@english)['topic']
    assert_equal 'science', values(@english)['tags'].first
    assert_equal values(@english)['tags'], values(@korean)['tags']
  end

  def test_stale_peer_prevents_all_writes
    snapshot = @service.enrich(source)[:related].to_h { |peer| [peer[:path], peer[:revision]] }
    File.open(File.join(@root, @korean), 'a') { |file| file.write("\nExternal edit.\n") }
    before = [bytes(@english), bytes(@korean)]
    assert_raises(LocalEditor::Conflict) { save({ 'tags' => %w[research new-tag] }, related: snapshot) }
    assert_equal before, [bytes(@english), bytes(@korean)]
  end

  def test_missing_peer_snapshot_prevents_shared_edit
    assert_raises(LocalEditor::Conflict) { save({ 'categories' => ['Research'] }, related: {}) }
  end

  def test_mixed_prose_metadata_and_undo_preserve_math_tables
    before = [bytes(@english), bytes(@korean)]
    segment = source.document[:segments].find { |entry| entry[:text] == 'A clear sentence.' }
    saved = save({ 'categories' => ['Research'] }, edits: [{ id: segment[:id], original: segment[:text], text: 'A precise sentence.' }])
    assert saved[:document][:segments].any? { |entry| entry[:text] == 'A precise sentence.' }
    assert_includes bytes(@korean), 'A clear sentence.'
    assert_includes bytes(@english), '$$x^2 + y^2$$'
    assert_includes bytes(@english), "| x | y |\n|---|---|\n| 1 | 2 |"
    @service.undo(source: source, revision: source.revision, receipt: saved[:receipt])
    assert_equal before, [bytes(@english), bytes(@korean)]
  end

  def test_failed_second_write_rolls_back_first
    before = [bytes(@english), bytes(@korean)]
    Thread.current[:metadata_test_fail_path] = @korean
    error = assert_raises(LocalEditor::Error) { save('categories' => ['Changed']) }
    assert_match(/Simulated/, error.message)
    assert_equal before, [bytes(@english), bytes(@korean)]
  end

  def test_group_undo_preflights_every_peer
    saved = save('categories' => ['Changed'])
    File.open(File.join(@root, @korean), 'a') { |file| file.write("\nExternal change.\n") }
    before = [bytes(@english), bytes(@korean)]
    assert_raises(LocalEditor::Conflict) { @service.undo(source: source, revision: source.revision, receipt: saved[:receipt]) }
    assert_equal before, [bytes(@english), bytes(@korean)]
  end

  def test_failed_second_undo_write_rolls_back_and_receipt_can_be_retried
    original = [bytes(@english), bytes(@korean)]
    saved = save('categories' => ['Changed'])
    changed = [bytes(@english), bytes(@korean)]
    Thread.current[:metadata_test_fail_path] = @korean
    Thread.current[:metadata_test_fail_action] = 'undo'
    assert_raises(LocalEditor::Error) do
      @service.undo(source: source, revision: source.revision, receipt: saved[:receipt])
    end
    assert_equal changed, [bytes(@english), bytes(@korean)]
    @service.undo(source: source, revision: source.revision, receipt: saved[:receipt])
    assert_equal original, [bytes(@english), bytes(@korean)]
  end

  def test_noop_does_not_write_or_create_a_receipt
    before = bytes(@english)
    saved = save('date' => '2026-10-20T10:00:00+09:00', 'lang' => 'en')
    assert_equal 0, saved[:changed]
    assert_nil saved[:receipt]
    assert_equal before, bytes(@english)
  end

  def test_invalid_metadata_and_dates_leave_sources_intact
    before = [bytes(@english), bytes(@korean)]
    [
      { 'tags' => %w[research korean] }, { 'tags' => ['<unsafe>'] },
      { 'topic' => 'unregistered' }, { 'categories' => [] }, { 'categories' => ['<script>'] },
      { 'series' => 'invented' }, { 'series_order' => -1 }, { 'lang' => 'fr' },
      { 'date' => '2026-02-30 10:00:00 +0900' }, { 'date' => '2026-10-24 25:00:00 +0900' },
      { 'date' => '2026-10-24 10:00:00' }, { 'date' => '2026-10-24T10:00:00+14:30' },
      { 'translation_key' => 'new-key' }, { 'permalink' => '/changed/' }
    ].each { |change| assert_raises(LocalEditor::InvalidEdit, change.inspect) { save(change) } }
    assert_equal before, [bytes(@english), bytes(@korean)]
  end

  def test_legacy_alias_target_cannot_lose_last_article
    File.delete(File.join(@root, @other))
    assert_raises(LocalEditor::InvalidEdit) { save('tags' => ['research']) }
  end
end

tests = MetadataTest.instance_methods.grep(/\Atest_/).sort
assertions = 0
failures = []
tests.each do |name|
  test = MetadataTest.new
  begin
    test.setup
    test.public_send(name)
    assertions += test.assertions.to_i
  rescue StandardError => error
    failures << "#{name}: #{error.class}: #{error.message}\n#{error.backtrace.first(5).join("\n")}"
  ensure
    test.teardown
  end
end
abort(failures.join("\n\n")) unless failures.empty?
puts "#{tests.length} metadata tests / #{assertions} assertions passed in isolated temporary blogs."
