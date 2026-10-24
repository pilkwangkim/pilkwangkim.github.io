#!/usr/bin/env ruby
# frozen_string_literal: true

require 'bundler/setup'
require 'optparse'

options = { port: 4210, open: true }
parser = OptionParser.new do |opts|
  opts.banner = 'Usage: tools/edit.sh [--port PORT] [--no-open]'
  opts.on('--port PORT', Integer, 'Loopback port (default: 4210)') { |value| options[:port] = value }
  opts.on('--no-open', 'Do not open the browser automatically') { options[:open] = false }
  opts.on('-h', '--help', 'Show this help') { puts opts; exit }
end

begin
  parser.parse!
  raise OptionParser::InvalidArgument, 'unexpected arguments' unless ARGV.empty?
  raise OptionParser::InvalidArgument, 'port must be between 1024 and 65535' unless (1024..65_535).cover?(options[:port])

  require_relative 'local_editor/server'
  root = File.realpath(File.join(__dir__, '..'))
  LocalEditor::Server.new(root: root, **options).start
rescue OptionParser::ParseError => e
  warn e.message
  warn parser
  exit 2
rescue Interrupt
  warn "\nLocal editor stopped."
rescue StandardError => e
  warn "Local editor: #{e.message}"
  exit 1
end
