#!/usr/bin/env ruby
# frozen_string_literal: true

# Isolated API checks: uses temporary posts and bypasses only the Jekyll subprocess.
# Run: bundle exec ruby tools/local_editor/test_server.rb
require 'tmpdir'
require 'fileutils'
require 'json'
require 'digest'
require 'net/http'
require_relative 'server'
class Request
  attr_reader :path, :request_method, :query, :body
  def initialize(path, method: 'GET', headers: {}, query: {}, body: nil)
    @path, @request_method, @query, @body = path, method, query, body
    @headers = { 'host' => '127.0.0.1:4297' }.merge(headers)
  end
  def [](key); @headers[key]; end
end
checks = 0
Dir.mktmpdir('local-editor-api-') do |root|
  FileUtils.mkdir_p(File.join(root, '_posts'))
  File.write(File.join(root, '_config.yml'), "exclude: [tools]\npwa:\n  enabled: true\n")
  relative = '_posts/2026-10-24-Test.md'
  path = File.join(root, relative)
  original = "---\ntitle: Test\n---\n\nA clear sentence with **emphasis**.\n\n$$x^2 + y^2$$\n\n| x | y |\n|---|---|\n| 1 | 2 |\n"
  File.write(path, original)
  server = LocalEditor::Server.new(root: root, port: 4297, open: false)
  destination = File.join(root, '.local-editor', 'fake-site')
  FileUtils.mkdir_p(destination)
  File.write(File.join(destination, 'index.html'), '<h1>Preview</h1>')
  File.write(File.join(root, '.local-editor', 'outside.txt'), 'private')
  server.instance_variable_set(:@destination, File.realpath(destination))
  url = '/posts/Test/'
  source = LocalEditor::Source.new(root: root, path: relative)
  server.instance_variable_set(:@documents, {url => {'path' => relative, 'revision' => source.revision}})
  server.instance_variable_set(:@state, {status: 'ready', generation: 1, build_id: 'test', error: nil})
  def server.rebuild_in_background(id); @queued_build = id; end
  invoke = lambda do |request, expected|
    response = WEBrick::HTTPResponse.new(WEBrick::Config::HTTP)
    server.send(:handle, request, response)
    raise "#{request.request_method} #{request.path}: #{response.status}, expected #{expected}: #{response.body}" unless response.status == expected
    checks += 1
    response
  end
  response = invoke.call(Request.new('/__local_editor/document', query: {'url' => url}), 200)
  document = JSON.parse(response.body)
  token = document.fetch('token')
  raise 'No segments' if document.fetch('segments').empty?
  invoke.call(Request.new('/__local_editor/status', headers: {'host'=>'malicious.example:4297'}), 403)
  invoke.call(Request.new('/__local_editor/status', headers: {'origin'=>'https://attacker.example'}), 403)
  invoke.call(Request.new('/'), 200)
  invoke.call(Request.new('/../outside.txt'), 403)
  invoke.call(Request.new('/__local_editor/document', query: {'url'=>'/../../etc/passwd'}), 404)
  headers = {'origin'=>'http://127.0.0.1:4297','content-type'=>'application/json'}
  invoke.call(Request.new('/__local_editor/save', method:'POST', headers:{'content-type'=>'application/json'}, body:'{}'), 403)
  invoke.call(Request.new('/__local_editor/save', method:'POST', headers:headers, body:'{}'), 403)
  invoke.call(Request.new('/__local_editor/save', method:'POST', headers:headers, body:'['), 400)
  invoke.call(Request.new('/__local_editor/save', method:'POST', headers:headers.merge('content-length'=>'900000'), body:'{}'), 413)
  segment = document['segments'].find { |item| item['text'].include?('clear sentence') }
  raise 'No paragraph segment' unless segment
  payload = {url:url,token:token,revision:document['revision'],edits:[{id:segment['id'],text:'A polished sentence with ',original:segment['text']}]}
  response = invoke.call(Request.new('/__local_editor/save', method:'POST', headers:headers, body:JSON.generate(payload)), 202)
  saved = JSON.parse(response.body)
  raise 'Missing receipt/build_id' unless saved['receipt'] && saved['build_id']
  raise 'Math changed' unless File.read(path).include?('$$x^2 + y^2$$')
  raise 'Table changed' unless File.read(path).include?("| x | y |\n|---|---|\n| 1 | 2 |")
  invoke.call(Request.new('/__local_editor/document', query:{'url'=>url}), 503)
  revision = saved.fetch('document').fetch('revision')
  server.instance_variable_get(:@documents)[url]['revision'] = revision
  server.instance_variable_get(:@state)[:status] = 'ready'
  unchanged_segment = saved.fetch('document').fetch('segments').find { |item| item[:text] == 'A polished sentence with ' || item['text'] == 'A polished sentence with ' }
  raise 'No saved paragraph segment' unless unchanged_segment
  unchanged_id = unchanged_segment[:id] || unchanged_segment['id']
  unchanged_text = unchanged_segment[:text] || unchanged_segment['text']
  state_before_noop = server.instance_variable_get(:@state).dup
  build_before_noop = server.instance_variable_get(:@queued_build)
  noop_payload = {url:url, token:token, revision:revision,
                  edits:[{id:unchanged_id, text:unchanged_text, original:unchanged_text}]}
  noop = JSON.parse(invoke.call(Request.new('/__local_editor/save', method:'POST', headers:headers,
                                           body:JSON.generate(noop_payload)), 200).body)
  raise 'Unchanged save lost the previous undo receipt' unless noop['receipt'] == saved['receipt']
  raise 'Unchanged save rebuilt the preview' unless server.instance_variable_get(:@state) == state_before_noop &&
                                                       server.instance_variable_get(:@queued_build) == build_before_noop
  raise 'Unchanged save did not report ready/zero changes' unless noop['status'] == 'ready' && noop['changed'] == 0
  invoke.call(Request.new('/__local_editor/save', method:'POST', headers:headers, body:JSON.generate(payload)), 409)
  undo = {url:url,token:token,revision:revision,receipt:saved['receipt']}
  invoke.call(Request.new('/__local_editor/undo', method:'POST', headers:headers, body:JSON.generate(undo)), 202)
  raise 'Undo did not restore exact bytes' unless File.binread(path) == original
  source = LocalEditor::Source.new(root:root,path:relative)
  server.instance_variable_get(:@documents)[url]['revision'] = source.revision
  server.instance_variable_get(:@state)[:status] = 'ready'
  File.open(path, 'a') { |f| f.write("\nExternal update.\n") }
  invoke.call(Request.new('/__local_editor/document', query:{'url'=>url}), 409)
  invoke.call(Request.new('/__local_editor/rebuild', method:'POST', headers:headers, body:JSON.generate(token:token)), 202)
  invoke.call(Request.new('/__local_editor/rebuild', method:'POST', headers:headers, body:JSON.generate(token:token)), 503)
  server.send(:prepare_config)
  config = YAML.safe_load(File.read(File.join(root,'.local-editor','config.yml')))
  raise 'PWA not disabled' if config.dig('pwa','enabled') || config.dig('pwa','cache','enabled')
  raise 'Work directory not excluded' unless config['exclude'].include?('.local-editor')
  raise 'Scheduled posts disappear from the local editor' unless config['future'] == true
  server.stop
  server.instance_variable_get(:@http).listeners.each(&:close)
end
Dir.mktmpdir('local-editor-metadata-api-') do |root|
  FileUtils.mkdir_p(File.join(root, '_posts'))
  FileUtils.mkdir_p(File.join(root, '_data'))
  File.write(File.join(root, '_config.yml'), 'exclude: [tools]')
  File.write(File.join(root, '_data/topics.yml'), YAML.dump([{ 'id'=>'research', 'title'=>'Research', 'tag'=>'research' }]))
  File.write(File.join(root, '_data/series.yml'), YAML.dump([]))
  english = '_posts/2026-10-24-Test.md'
  korean = '_posts/2026-10-24-Test-KR.md'
  [english, korean].each do |path|
    data = { 'title'=>'Test', 'date'=>Time.new(2026,10,20,10,0,0,'+09:00'),
             'tags'=>%w[research example], 'categories'=>['AI'], 'topic'=>'research',
             'lang'=>path == english ? 'en' : 'ko', 'translation_key'=>'test' }
    File.write(File.join(root,path), YAML.dump(data) + "---\n\nA clear sentence.\n")
  end
  original = [english,korean].to_h { |path| [path,File.binread(File.join(root,path))] }
  server = LocalEditor::Server.new(root:root,port:4297,open:false)
  english_url, korean_url = '/posts/Test/', '/posts/Test-KR/'
  paths = {english_url=>english,korean_url=>korean}
  make_ready = lambda do
    descriptors = paths.to_h do |url,path|
      [url,{'path'=>path,'revision'=>LocalEditor::Source.new(root:root,path:path).revision}]
    end
    server.instance_variable_set(:@documents,descriptors)
    server.instance_variable_set(:@state,{status:'ready',generation:1,build_id:'metadata-test',error:nil})
  end
  make_ready.call
  def server.rebuild_in_background(id); @queued_build = id; end
  invoke = lambda do |request,expected|
    response = WEBrick::HTTPResponse.new(WEBrick::Config::HTTP)
    server.send(:handle,request,response)
    raise "Metadata API #{request.path}: expected #{expected}, got #{response.status}: #{response.body}" unless response.status == expected
    checks += 1
    JSON.parse(response.body)
  end
  document = invoke.call(Request.new('/__local_editor/document',query:{'url'=>english_url}),200)
  token = document.fetch('token')
  headers = {'origin'=>'http://127.0.0.1:4297','content-type'=>'application/json'}
  raise 'Metadata choices missing' unless document.dig('metadata','options','topics').first['id'] == 'research'
  raise 'Related article snapshot missing' unless document.dig('metadata','related').first['path'] == korean
  post = lambda do |endpoint,payload,expected|
    invoke.call(Request.new("/__local_editor/#{endpoint}",method:'POST',headers:headers,
                            body:JSON.generate(payload.merge(url:english_url,token:token))),expected)
  end
  dated = post.call('save',{revision:document['revision'],metadata:{date:'2099-10-24T10:00:00+09:00'}},202)
  raise 'Date save altered translation' unless File.binread(File.join(root,korean)) == original[korean]
  raise 'Save response missing enriched metadata' unless dated.dig('document','metadata','related').first['revision']
  raise 'Future date not preserved' unless dated.dig('document','metadata','values','date') == '2099-10-24 10:00:00 +0900'
  make_ready.call
  noop = post.call('save',{revision:dated.dig('document','revision'),metadata:{date:'2099-10-24 10:00:00 +0900'}},200)
  raise 'Metadata no-op lost undo' unless noop['receipt'] == dated['receipt'] && noop['changed'] == 0
  server.instance_variable_get(:@state)[:status] = 'error'
  undone = post.call('undo',{revision:dated.dig('document','revision'),receipt:dated['receipt']},202)
  raise 'Error-state metadata undo failed' unless File.binread(File.join(root,english)) == original[english]
  raise 'Undo response metadata missing options' unless undone.dig('document','metadata','options','languages').length == 2
  make_ready.call
  document = invoke.call(Request.new('/__local_editor/document',query:{'url'=>english_url}),200)
  post.call('save',{revision:document['revision'],metadata:{categories:['Research']}},409)
  peers = document.dig('metadata','related').to_h { |entry| [entry['path'],entry['revision']] }
  server.instance_variable_get(:@receipts)[korean_url] = 'stale-peer-receipt'
  shared = post.call('save',{revision:document['revision'],metadata:{categories:['Research']},related_revisions:peers},202)
  raise 'Peer undo receipt retained after group save' if server.instance_variable_get(:@receipts).key?(korean_url)
  [english,korean].each do |path|
    actual = LocalEditor::Source.new(root:root,path:path).metadata_values['categories']
    raise 'Shared metadata did not synchronize' unless actual == ['Research']
  end
  make_ready.call
  File.open(File.join(root,korean),'a') { |file| file.write("\nExternal change.\n") }
  current_english = File.binread(File.join(root,english))
  post.call('undo',{revision:shared.dig('document','revision'),receipt:shared['receipt']},409)
  raise 'Failed group undo altered its first article' unless File.binread(File.join(root,english)) == current_english
  File.write(File.join(root,korean),File.read(File.join(root,korean)).delete_suffix("\nExternal change.\n"))
  post.call('undo',{revision:shared.dig('document','revision'),receipt:shared['receipt']},202)
  raise 'Group undo not exact' unless original.all? { |path,raw| File.binread(File.join(root,path)) == raw }
  make_ready.call
  document = invoke.call(Request.new('/__local_editor/document',query:{'url'=>english_url}),200)
  post.call('save',{revision:document['revision'],metadata:{lang:'ko'}},422)
  post.call('save',{revision:document['revision'],metadata:['invalid']},400)
  post.call('save',{revision:document['revision'],edits:'invalid'},400)
  post.call('save',{revision:document['revision'],metadata:{translation_key:'changed'}},422)
  noop = post.call('save',{revision:document['revision'],metadata:{},edits:[]},200)
  raise 'Empty change request unexpectedly wrote files' unless noop['changed'] == 0 && original.all? { |path,raw| File.binread(File.join(root,path)) == raw }
  # A catalogue failure after the actual source commit must still return the
  # save receipt, reserve its rebuild and permit exact Undo.
  metadata_service = server.instance_variable_get(:@metadata)
  original_enrich = metadata_service.method(:enrich)
  calls = 0
  metadata_service.define_singleton_method(:enrich) do |item|
    calls += 1
    raise LocalEditor::InvalidEdit,'Simulated concurrent catalogue error' if calls == 2

    original_enrich.call(item)
  end
  segment = document['segments'].find { |entry| entry['text'] == 'A clear sentence.' }
  saved = post.call('save',{revision:document['revision'],edits:[{id:segment['id'],original:segment['text'],text:'A better sentence.'}]},202)
  raise 'Post-commit catalogue failure lost save receipt' unless saved['receipt'] && server.instance_variable_get(:@receipts)[english_url] == saved['receipt']
  raise 'Post-commit catalogue failure skipped rebuild' unless server.instance_variable_get(:@state)[:status] == 'building'
  raise 'Post-commit catalogue fallback lost options/values' unless saved.dig('document','metadata','options','topics').first['id'] == 'research' && saved.dig('document','metadata','values','lang') == 'en'
  metadata_service.define_singleton_method(:enrich) { |item| original_enrich.call(item) }
  make_ready.call
  post.call('undo',{revision:saved.dig('document','revision'),receipt:saved['receipt']},202)
  raise 'Post-commit catalogue fallback undo was not exact' unless original.all? { |path,raw| File.binread(File.join(root,path)) == raw }
  server.stop
  server.instance_variable_get(:@http).listeners.each(&:close)
end
Dir.mktmpdir('local-editor-http-api-') do |root|
  FileUtils.mkdir_p(File.join(root, '_posts'))
  FileUtils.mkdir_p(File.join(root, '_data'))
  File.write(File.join(root, '_config.yml'), 'exclude: [tools]')
  File.write(File.join(root, '_data/topics.yml'), YAML.dump([{ 'id'=>'research', 'title'=>'Research', 'tag'=>'research' }]))
  File.write(File.join(root, '_data/series.yml'), YAML.dump([]))
  paths = {'/posts/HTTP/'=>'_posts/2026-10-24-HTTP.md', '/posts/HTTP-KR/'=>'_posts/2026-10-24-HTTP-KR.md'}
  paths.each_with_index do |(_, path), index|
    data = {'title'=>'HTTP test', 'date'=>Time.new(2026,10,24,10,0,0,'+09:00'), 'tags'=>['research'],
            'categories'=>['AI'], 'topic'=>'research', 'lang'=>index.zero? ? 'en' : 'ko', 'translation_key'=>'http'}
    File.write(File.join(root, path), YAML.dump(data) + "---\n\nA sentence with $x+1$.\n\n$$\ny=\\frac{a}{b}\n$$\n\n| Symbol | Value |\n|:---|---:|\n| $x$ | **2** |\n")
  end
  original = paths.values.to_h { |path| [path,File.binread(File.join(root,path))] }
  server = LocalEditor::Server.new(root:root,port:4297,open:false)
  make_ready = lambda do
    server.instance_variable_set(:@documents, paths.to_h do |url,path|
      [url,{'path'=>path,'revision'=>LocalEditor::Source.new(root:root,path:path).revision}]
    end)
    server.instance_variable_set(:@state,{status:'ready',generation:1,build_id:'http-test',error:nil})
  end
  make_ready.call
  def server.rebuild_in_background(id); @queued_build = id; end
  http_server = server.instance_variable_get(:@http)
  # Starting WEBrick itself verifies request decoding and wire response handling;
  # only the expensive Jekyll rebuild remains replaced by a deterministic queue.
  http_thread = Thread.new { http_server.start }
  http = Net::HTTP.new('127.0.0.1',4297,nil)
  http.open_timeout = http.read_timeout = 3
  http.start
  invoke = lambda do |method, endpoint, payload, expected, headers = {}|
    request = method == :get ? Net::HTTP::Get.new(endpoint) : Net::HTTP::Post.new(endpoint)
    request['Origin'] = 'http://127.0.0.1:4297'
    request['Content-Type'] = 'application/json'
    headers.each { |key,value| request[key] = value }
    request.body = JSON.generate(payload) if payload
    response = http.request(request)
    raise "Wire HTTP #{endpoint}: #{response.code}, expected #{expected}: #{response.body}" unless response.code.to_i == expected
    raise 'Wire API response was cacheable' unless response['cache-control'].include?('no-store')
    raise 'Wire API response was not JSON' unless response['content-type'].start_with?('application/json')
    checks += 1
    JSON.parse(response.body)
  end
  english_url, english = paths.first
  document = invoke.call(:get,"/__local_editor/document?#{URI.encode_www_form(url:english_url)}",nil,200)
  token = document.fetch('token')
  math = document.dig('structures','math')
  table = document.dig('structures','tables').first
  cell = table['rows'].flatten.find { |entry| entry['source'] == '**2**' }
  cell_edit = {id:cell['id'],original:cell['source'],text:'**3** $z^2$'}
  payload = {url:english_url,token:token,revision:document['revision'],table_id:table['id'],edits:[cell_edit]}
  preview = invoke.call(:post,'/__local_editor/preview',payload,200)
  raise 'Wire preview lost table cell Markdown/LaTeX' unless preview['html'].include?('<strong>3</strong> $z^2$')
  invoke.call(:post,'/__local_editor/preview',payload.merge(token:'wrong'),403)
  invoke.call(:post,'/__local_editor/preview',payload,403,{'Origin'=>'https://example.com'})
  invoke.call(:post,'/__local_editor/preview',payload.merge(revision:'stale'),409)
  invoke.call(:post,'/__local_editor/preview',payload.merge(edits:{}),400)
  raise 'Wire preview or rejected request modified a post' unless original.all? { |path,raw| File.binread(File.join(root,path)) == raw }
  mixed_edits = [cell_edit, *math.map do |entry|
    {id:entry['id'],original:entry['latex'],text:entry['display'] ? 'y=\\frac{a}{b+1}' : 'x+2'}
  end]
  peers = document.dig('metadata','related').to_h { |entry| [entry['path'],entry['revision']] }
  saved = invoke.call(:post,'/__local_editor/save',payload.merge(edits:mixed_edits,metadata:{categories:['Science']},related_revisions:peers),202)
  english_after = File.binread(File.join(root,english))
  raise 'Wire mixed save missed inline or display equation' unless english_after.include?('$x+2$') && english_after.include?('y=\\frac{a}{b+1}')
  raise 'Wire mixed save missed table cell' unless english_after.include?('**3** $z^2$')
  expected_body = original[english].split("---\n",3).last.sub('$x+1$','$x+2$').sub('y=\\frac{a}{b}','y=\\frac{a}{b+1}').sub('**2**','**3** $z^2$')
  raise 'Wire mixed save changed unrelated article content' unless english_after.split("---\n",3).last == expected_body
  paths.values.each do |path|
    raise 'Wire shared metadata save missed a translation' unless LocalEditor::Source.new(root:root,path:path).metadata_values['categories'] == ['Science']
  end
  korean = paths.values.last
  korean_after = File.binread(File.join(root,korean))
  raise 'Wire shared metadata save changed translated body' unless korean_after.split("---\n",3).last == original[korean].split("---\n",3).last
  invoke.call(:post,'/__local_editor/save',payload,503)
  make_ready.call
  invoke.call(:post,'/__local_editor/save',payload,409)
  undo = {url:english_url,token:token,revision:saved.dig('document','revision'),receipt:saved['receipt']}
  invoke.call(:post,'/__local_editor/undo',undo,202)
  raise 'Wire group Undo did not restore exact source bytes' unless original.all? { |path,raw| File.binread(File.join(root,path)) == raw }
ensure
  http&.finish if http&.started?
  server&.stop
  http_thread&.join(3)
  server&.instance_variable_get(:@http)&.listeners&.each { |socket| socket.close unless socket.closed? }
end
puts "#{checks} local API checks passed; math/table byte preservation and exact undo passed."
