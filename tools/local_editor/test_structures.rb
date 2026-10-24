# frozen_string_literal: true

# End-to-end source/API contracts; all writes use an isolated temporary blog.
require 'tmpdir'
require 'fileutils'
require 'json'
require 'nokogiri'
require_relative 'server'

checks = 0
check = lambda do |condition, message|
  raise message unless condition
  checks += 1
end
reject = lambda do |error, &block|
  begin
    block.call
  rescue error
    checks += 1
    next
  end
  raise "Expected #{error}"
end
Dir.mktmpdir('local-editor-structures-') do |root|
  FileUtils.mkdir_p(File.join(root, '_posts'))
  File.write(File.join(root, '_config.yml'), "exclude: [tools]\n")
  relative = '_posts/2026-10-24-Structures.md'
  path = File.join(root, relative)
  original = <<~'POST'
    ---
    title: Structure test
    date: 2026-10-24 10:00:00 +0900
    tags: [test]
    lang: en
    ---
    Before $x+1$ after **text**.

    $$
    y=\frac{a}{b}
    $$

    | Symbol | Estimate |
    |:---|---:|
    | $x$ | **2** |
  POST
  File.write(path, original)
  source = LocalEditor::Source.new(root: root, path: relative)
  document = source.document
  math = document[:structures][:math]
  table = document[:structures][:tables].first
  cell = table[:rows].flatten.find { |item| item[:source] == '**2**' }
  check.call(math.length == 2 && table[:rows].flatten.length == 4, 'Math/table ownership overlaps')
  edits = [
    {id: math[0][:id], original: math[0][:latex], text: 'x+2'},
    {id: math[1][:id], original: math[1][:latex], text: 'y=\frac{a}{b+1}'},
    {id: cell[:id], original: cell[:source], text: '**3** $z$'}
  ]
  preview = Nokogiri::HTML.fragment(source.preview_table(table_id: table[:id], edits: [edits.last]))
  check.call(preview.at('tbody td:last-child').text.strip == '3 $z$', 'Cell preview lost Markdown or LaTeX')
  check.call(File.binread(path) == original, 'Preview wrote the source')
  saved = source.save(revision: document[:revision], edits: edits, metadata: {tags: %w[test revised]})
  expected = original.sub('$x+1$', '$x+2$').sub('y=\frac{a}{b}', 'y=\frac{a}{b+1}').sub('**2**', '**3** $z$').sub('tags: [test]', 'tags: ["test","revised"]')
  check.call(File.binread(path) == expected, 'Combined math/table/metadata save changed unrelated bytes')
  source.undo(revision: saved[:document][:revision], receipt: saved[:receipt])
  check.call(File.binread(path) == original, 'Combined Undo did not restore exact bytes')
  reject.call(LocalEditor::InvalidEdit) do
    source.save(revision: source.revision, edits: [edits.last, {id: math[0][:id], text: "x\ny"}])
  end
  check.call(File.binread(path) == original, 'Invalid equation partially wrote valid table cell')
  reject.call(LocalEditor::InvalidEdit) do
    source.save(revision: source.revision, edits: [edits[0], {id: cell[:id], text: "new\nrow"}])
  end
  check.call(File.binread(path) == original, 'Invalid cell partially wrote valid equation')
  reject.call(LocalEditor::Conflict) { source.save(revision: 'stale', edits: edits) }

  server = LocalEditor::Server.new(root: root, port: 4298, open: false)
  url = '/posts/Structures/'
  server.instance_variable_set(:@documents, {url => {'path' => relative, 'revision' => source.revision}})
  server.instance_variable_set(:@state, {status: 'ready', generation: 1, build_id: 'structures', error: nil})
  request_class = Struct.new(:path, :request_method, :query, :body, :headers) do
    def [](key); headers[key]; end
  end
  invoke = lambda do |endpoint, payload, expected_status, headers = {}|
    method = payload ? 'POST' : 'GET'
    headers = {'host' => '127.0.0.1:4298', 'origin' => 'http://127.0.0.1:4298', 'content-type' => 'application/json'}.merge(headers)
    request = request_class.new("/__local_editor/#{endpoint}", method, {'url' => url}, payload && JSON.generate(payload), headers)
    response = WEBrick::HTTPResponse.new(WEBrick::Config::HTTP)
    server.send(:handle, request, response)
    check.call(response.status == expected_status, "#{endpoint}: #{response.status}, expected #{expected_status}: #{response.body}")
    JSON.parse(response.body)
  end
  api_document = invoke.call('document', nil, 200)
  payload = {url: url, token: api_document['token'], revision: source.revision, table_id: table[:id], edits: [edits.last]}
  preview = invoke.call('preview', payload, 200)
  check.call(Nokogiri::HTML.fragment(preview['html']).at('strong').text == '3', 'API preview changed cell value')
  invoke.call('preview', payload.merge(token: 'wrong'), 403)
  invoke.call('preview', payload, 403, {'origin' => 'https://example.com'})
  invoke.call('preview', payload.merge(revision: 'stale'), 409)
  invoke.call('preview', payload.merge(table_id: 'table-unknown'), 422)
  invoke.call('preview', payload.merge(edits: [{id: cell[:id], text: '<img src="https://example.com">'}]), 422)
  check.call(File.binread(path) == original && server.instance_variable_get(:@state)[:generation] == 1, 'API preview mutated source or rebuild state')
ensure
  server&.stop
  server&.instance_variable_get(:@http)&.listeners&.each(&:close)
end
puts "#{checks} structure source/API integration checks passed."
