# frozen_string_literal: true

require 'digest'
require 'fileutils'
require 'json'
require 'rbconfig'
require 'securerandom'
require 'thread'
require 'uri'
require 'webrick'
require 'yaml'
require 'date'
require_relative 'source'
require_relative 'metadata'

module LocalEditor
  # Builds into a fresh directory, then switches the complete HTML/map pair together.
  # Only this loopback process can write sources; no editor endpoint is in the site.
  class Server
    MAX_REQUEST_BYTES = 512 * 1024
    HOST = '127.0.0.1'

    class RequestError < StandardError
      attr_reader :status

      def initialize(status, message)
        @status = status
        super(message)
      end
    end

    def initialize(root:, port: 4210, open: true)
      @root = File.realpath(root)
      @posts_root = File.realpath(File.join(@root, '_posts'))
      @port = port
      @open = open
      @origin = "http://#{HOST}:#{port}"
      @token = SecureRandom.hex(32)
      @work = File.join(@root, '.local-editor')
      @mutex = Mutex.new
      @state = { status: 'starting', build_id: nil, generation: 0, error: nil }
      @documents = {}
      @receipts = {}
      @metadata = LocalEditor::Metadata.new(root: @root)
      @process = nil
      @stopping = false
      @http = WEBrick::HTTPServer.new(
        BindAddress: HOST, Port: @port,
        Logger: WEBrick::Log.new($stderr, WEBrick::Log::WARN),
        AccessLog: [], DoNotReverseLookup: true,
        RequestTimeout: 20, MaxClients: 8,
        ServerSoftware: 'Local blog editor'
      )
      @http.mount_proc('/') { |request, response| handle(request, response) }
    rescue Errno::EADDRINUSE
      raise "Port #{port} is already in use. Choose another port with --port PORT."
    end

    def start
      FileUtils.mkdir_p(@work)
      prepare_config
      %w[INT TERM].each { |signal| trap(signal) { stop } }
      puts 'Building the local blog preview…'
      build!
      puts "\nLocal blog editor: #{@origin}/"
      puts 'Open an article and click Edit page, Metadata, or Tables & equations.'
      puts 'Changes stay in local Markdown files. Press Ctrl-C to stop.'
      if @open && RUBY_PLATFORM.include?('darwin')
        Process.detach(Process.spawn('open', @origin, out: File::NULL, err: File::NULL))
      end
      @http.start unless @stopping
    ensure
      @http&.shutdown
      stop_build_process
    end

    def stop
      @stopping = true
      @http&.shutdown
      stop_build_process
    end

    private

    def prepare_config
      configured = YAML.safe_load(File.read(File.join(@root, '_config.yml')),
                                  permitted_classes: [Date, Time], aliases: true) || {}
      overrides = {
        'url' => @origin, 'baseurl' => '',
        'future' => true,
        'pwa' => { 'enabled' => false, 'cache' => { 'enabled' => false } },
        'exclude' => (Array(configured['exclude']) + ['.local-editor']).uniq,
        'local_editor' => { 'enabled' => true }
      }
      @config_path = File.join(@work, 'config.yml')
      File.write(@config_path, YAML.dump(overrides))
    end

    def new_build_id
      "#{Time.now.utc.strftime('%Y%m%dT%H%M%S')}-#{SecureRandom.hex(4)}"
    end

    def build!(id: new_build_id)
      directory = File.join(@work, 'builds', id)
      destination = File.join(directory, 'site')
      manifest_path = File.join(directory, 'maps.json')
      FileUtils.mkdir_p(directory)
      @mutex.synchronize { @state = @state.merge(status: 'building', build_id: id, error: nil) }
      log_path = File.join(directory, 'build.log')
      env = {
        'BLOG_LOCAL_EDITOR' => '1', 'JEKYLL_ENV' => 'development',
        'BLOG_LOCAL_EDITOR_MAP_PATH' => manifest_path
      }
      command = [RbConfig.ruby, '-S', 'bundle', 'exec', 'jekyll', 'build',
                 '--source', @root, '--destination', destination,
                 '--config', "#{File.join(@root, '_config.yml')},#{@config_path}"]
      success = File.open(log_path, 'w') do |log|
        process = Process.spawn(env, *command, chdir: @root, out: log, err: log, pgroup: true)
        @mutex.synchronize { @process = process }
        _pid, result = Process.wait2(process)
        @mutex.synchronize { @process = nil }
        result.success?
      end
      raise Interrupt if @stopping
      unless success
        tail = File.readlines(log_path).last(18).join.strip
        raise "Jekyll build failed. #{tail}\nBuild log: #{log_path}"
      end
      raise "The local editor build did not create its document map: #{manifest_path}" unless File.file?(manifest_path)

      manifest = JSON.parse(File.read(manifest_path))
      documents = manifest.fetch('documents', manifest)
      raise 'The editor document map is invalid.' unless documents.is_a?(Hash)

      @mutex.synchronize do
        @destination = File.realpath(destination)
        @documents = documents
        @state = @state.merge(status: 'ready', generation: @state[:generation] + 1, error: nil)
      end
      remove_old_builds(directory)
      puts "Preview ready (#{documents.length} editable articles)."
      id
    rescue StandardError => e
      @mutex.synchronize { @state = @state.merge(status: 'error', error: e.message) }
      raise
    end

    def remove_old_builds(current)
      # Keep the current and previous output for diagnosis; source backups are elsewhere.
      directories = Dir.glob(File.join(@work, 'builds', '*')).select { |entry| File.directory?(entry) }.sort.reverse
      directories.drop(2).each { |entry| FileUtils.rm_rf(entry) unless entry == current }
    end

    def stop_build_process
      process = @process
      return unless process

      Process.kill('TERM', -process)
    rescue Errno::ESRCH
      nil
    end

    def handle(request, response)
      response['Cache-Control'] = 'no-store, max-age=0'
      response['X-Content-Type-Options'] = 'nosniff'
      response['X-Frame-Options'] = 'DENY'
      response['Cross-Origin-Resource-Policy'] = 'same-origin'
      validate_host!(request)
      if request.path.start_with?('/__local_editor/')
        handle_api(request, response)
      else
        raise RequestError.new(405, 'Use GET or HEAD to read the preview.') unless %w[GET HEAD].include?(request.request_method)

        serve_file(request, response)
      end
    rescue RequestError => e
      json(response, e.status, error: e.message)
    rescue LocalEditor::Conflict => e
      json(response, 409, error: e.message)
    rescue LocalEditor::Error => e
      json(response, 422, error: e.message)
    rescue JSON::ParserError
      json(response, 400, error: 'The request must contain valid JSON.')
    rescue StandardError => e
      warn "Local editor request failed: #{e.class}: #{e.message}"
      json(response, 500, error: 'The local editor could not complete this request. See its terminal output.')
    end

    def validate_host!(request)
      raise RequestError.new(403, 'This editor accepts loopback requests only.') unless request['host'] == "#{HOST}:#{@port}"

      origin = request['origin']
      raise RequestError.new(403, 'A different origin cannot access this editor.') if origin && origin != @origin
    end

    def handle_api(request, response)
      case request.path
      when '/__local_editor/client.js', '/__local_editor/style.css'
        require_method!(request, 'GET')
        filename = request.path.end_with?('client.js') ? 'client.js' : 'style.css'
        path = File.join(@root, 'tools', 'local_editor', filename)
        raise RequestError.new(404, 'The editor client is not installed.') unless File.file?(path)

        response['Content-Type'] = filename.end_with?('.js') ? 'text/javascript; charset=utf-8' : 'text/css; charset=utf-8'
        response.body = File.binread(path)
        response.status = 200
      when '/__local_editor/status'
        require_method!(request, 'GET')
        json(response, 200, @mutex.synchronize { @state.merge(token: @token) })
      when '/__local_editor/document'
        require_method!(request, 'GET')
        result = @mutex.synchronize do
          ensure_ready!
          url = normalize_url(request.query['url'])
          source = source_for(url)
          document = source.document
          ensure_built_revision!(url, document)
          document[:metadata] = @metadata.enrich(source)
          document.merge(url: url, token: @token, build_id: @state[:build_id],
                         generation: @state[:generation], receipt: @receipts[url])
        end
        json(response, 200, result)
      when '/__local_editor/rebuild'
        require_method!(request, 'POST')
        authenticated_payload(request)
        result = @mutex.synchronize do
          ensure_ready! unless %w[ready error].include?(@state[:status])
          build_id = new_build_id
          @state = @state.merge(status: 'building', build_id: build_id, error: nil)
          { status: 'building', build_id: build_id }
        end
        rebuild_in_background(result[:build_id])
        json(response, 202, result)
      when '/__local_editor/preview'
        require_method!(request, 'POST')
        payload = authenticated_payload(request)
        result = @mutex.synchronize do
          ensure_ready!
          url = normalize_url(payload['url'])
          source = source_for(url)
          document = source.document
          ensure_built_revision!(url, document)
          unless payload['revision'].is_a?(String) && payload['revision'] == document[:revision]
            raise Conflict, 'This article changed after editing began. Refresh preview before previewing a table.'
          end
          edits = payload.fetch('edits', [])
          raise RequestError.new(400, 'Table preview edits must be an array.') unless edits.is_a?(Array) && edits.length <= 2000

          { html: source.preview_table(table_id: payload['table_id'], edits: edits) }
        end
        json(response, 200, result)
      when '/__local_editor/save', '/__local_editor/undo'
        require_method!(request, 'POST')
        payload = authenticated_payload(request)

        result = @mutex.synchronize do
          undo = request.path.end_with?('/undo')
          ensure_ready! unless undo && @state[:status] == 'error'
          url = normalize_url(payload['url'])
          source = source_for(url)
          document = source.document
          ensure_built_revision!(url, document) unless undo && @state[:status] == 'error'
          revision = payload['revision']
          raise RequestError.new(400, 'The source revision is missing.') unless revision.is_a?(String)
          unless revision == (document[:revision] || document['revision'])
            raise RequestError.new(409, 'This article changed after editing began. Refresh preview before saving.')
          end
          # Validate and retain a usable catalogue before touching any source.
          # A later, unrelated external edit must not strand a successful write
          # without its receipt or rebuild reservation.
          metadata_before_save = @metadata.enrich(source)

          saved = if undo
                    receipt = payload['receipt']
                    raise RequestError.new(400, 'There is no saved edit to undo.') unless receipt.is_a?(String) && receipt == @receipts[url]

                    if @metadata.batch_receipt?(receipt)
                      @metadata.undo(source: source, revision: revision, receipt: receipt)
                    else
                      source.undo(revision: revision, receipt: receipt)
                    end
                  else
                    edits = payload.fetch('edits', [])
                    metadata = payload.fetch('metadata', {})
                    raise RequestError.new(400, 'The edited blocks must be an array.') unless edits.is_a?(Array)
                    raise RequestError.new(400, 'Metadata changes must be an object.') unless metadata.is_a?(Hash)
                    if metadata.empty? && !edits.empty?
                      source.save(revision: revision, edits: edits)
                    else
                      @metadata.save(source: source, revision: revision, edits: edits, metadata: metadata,
                                     related_revisions: payload.fetch('related_revisions', {}))
                    end
                  end
          saved_document = saved[:document] || saved['document']
          if saved_document
            saved_document[:metadata] = metadata_after_save(source.path, saved_document, metadata_before_save)
          end
          if !undo && (saved[:changed] || saved['changed']) == 0
            # A browser may submit an unchanged edit after reverting its text.
            # Keep the preceding undo receipt and the current preview intact.
            next saved.merge(url: url, status: 'ready', receipt: @receipts[url],
                             build_id: @state[:build_id], generation: @state[:generation])
          end

          affected = saved[:affected_paths] || []
          @documents.each do |peer_url, descriptor|
            peer_path = descriptor.is_a?(Hash) ? descriptor['path'] : descriptor
            @receipts.delete(peer_url) if affected.include?(peer_path)
          end
          @receipts[url] = saved[:receipt] || saved['receipt']
          # Reserve the build before releasing the mutex: another tab cannot save
          # against a source that the currently served HTML has not rendered yet.
          build_id = new_build_id
          @state = @state.merge(status: 'building', build_id: build_id, error: nil)
          saved.merge(url: url, status: 'building', build_id: build_id)
        end
        if result[:status] == 'ready'
          json(response, 200, result)
        else
          rebuild_in_background(result[:build_id])
          json(response, 202, result)
        end
      else
        raise RequestError.new(404, 'No such local editor endpoint.')
      end
    end

    def require_method!(request, method)
      raise RequestError.new(405, "Use #{method} for this endpoint.") unless request.request_method == method
    end

    def metadata_after_save(path, document, previous)
      @metadata.enrich(LocalEditor::Source.new(root: @root, path: path))
    rescue StandardError => error
      warn "Metadata catalogue changed after saving; using the saved snapshot: #{error.class}: #{error.message}"
      related = Array(previous[:related]).map do |entry|
        begin
          entry.merge(revision: LocalEditor::Source.new(root: @root, path: entry[:path]).revision)
        rescue StandardError
          entry
        end
      end
      (document[:metadata] || {}).merge(options: previous[:options], related: related)
    end

    def rebuild_in_background(build_id)
      Thread.new do
        build!(id: build_id)
      rescue Interrupt
        nil
      rescue StandardError => e
        warn e.message
      end
    end

    def authenticated_payload(request)
      validate_write!(request)
      payload = parse_payload(request)
      unless same_token?(payload['token'])
        raise RequestError.new(403, 'The editor token is missing or expired. Reload this page.')
      end

      payload
    end

    def validate_write!(request)
      raise RequestError.new(403, 'Save from a page opened in this local editor.') unless request['origin'] == @origin
      raise RequestError.new(415, 'Send application/json when saving.') unless request['content-type'].to_s.split(';').first.to_s.strip == 'application/json'
      raise RequestError.new(413, 'This edit is too large.') if request['content-length'].to_i > MAX_REQUEST_BYTES
    end

    def parse_payload(request)
      body = request.body.to_s
      raise RequestError.new(413, 'This edit is too large.') if body.bytesize > MAX_REQUEST_BYTES

      payload = JSON.parse(body)
      raise RequestError.new(400, 'The request must be a JSON object.') unless payload.is_a?(Hash)

      payload
    end

    def same_token?(value)
      return false unless value.is_a?(String) && value.bytesize == @token.bytesize

      value.bytes.zip(@token.bytes).reduce(0) { |difference, (left, right)| difference | (left ^ right) }.zero?
    end

    def ensure_ready!
      return if @state[:status] == 'ready'

      message = @state[:status] == 'error' ? @state[:error] : 'The preview is rebuilding. Wait for it to finish, then reload.'
      raise RequestError.new(503, message)
    end

    def normalize_url(value)
      raise RequestError.new(400, 'Choose an article URL.') unless value.is_a?(String) && value.start_with?('/') && !value.start_with?('//')

      uri = URI.parse(value)
      raise RequestError.new(400, 'Choose a local article URL.') if uri.host || uri.scheme || uri.path.include?("\0")

      uri.path
    rescue URI::InvalidURIError
      raise RequestError.new(400, 'The article URL is invalid.')
    end

    def source_for(url)
      descriptor = @documents[url]
      path = descriptor.is_a?(Hash) ? descriptor['path'] : descriptor
      raise RequestError.new(404, 'This page is not an editable article.') unless path.is_a?(String)

      absolute = File.realpath(File.join(@root, path))
      unless absolute.start_with?("#{@posts_root}/") && File.file?(absolute) && %w[.md .markdown].include?(File.extname(absolute).downcase)
        raise RequestError.new(403, 'Only Markdown posts inside this blog can be edited.')
      end

      LocalEditor::Source.new(root: @root, path: path)
    rescue Errno::ENOENT
      raise RequestError.new(404, 'The article source no longer exists.')
    end

    def ensure_built_revision!(url, document)
      descriptor = @documents[url]
      expected = descriptor.is_a?(Hash) && descriptor['revision']
      unless expected.is_a?(String)
        raise RequestError.new(503, 'The preview has no source revision. Restart the local editor to rebuild it.')
      end
      return if expected == (document[:revision] || document['revision'])

      raise RequestError.new(409, 'Source changed outside this preview. Refresh preview before editing.')
    end

    def serve_file(request, response)
      destination = @mutex.synchronize { @destination }
      raise RequestError.new(503, 'The initial blog preview is still building.') unless destination

      decoded = URI::DEFAULT_PARSER.unescape(request.path)
      raise RequestError.new(400, 'The requested path is invalid.') if decoded.include?("\0") || decoded.include?('\\')

      candidate = File.expand_path(decoded.sub(%r{\A/+}, ''), destination)
      candidate = File.join(candidate, 'index.html') if File.directory?(candidate)
      raise RequestError.new(404, 'This page does not exist in the local preview.') unless File.file?(candidate)

      real = File.realpath(candidate)
      raise RequestError.new(403, 'This path is outside the local preview.') unless real.start_with?("#{destination}/")

      response['Content-Type'] = WEBrick::HTTPUtils.mime_type(real, WEBrick::HTTPUtils::DefaultMimeTypes)
      response['Content-Length'] = File.size(real).to_s
      response.body = request.request_method == 'HEAD' ? '' : File.binread(real)
      response.status = 200
    end

    def json(response, status, value)
      response.status = status
      response['Content-Type'] = 'application/json; charset=utf-8'
      response.body = JSON.generate(value)
    end
  end
end
