# frozen_string_literal: true

require 'digest'

# The explorer and its embedded bibliography must advance as one cache version.
Jekyll::Hooks.register :site, :post_read do |site|
  paths = %w[
    _data/quantum_access_literature.json
    _includes/interactive/quantum-access-literature.html
    assets/css/quantum-access-literature.css
    assets/js/quantum-access-literature.js
    assets/data/quantum-access-literature.json
  ]
  bytes = paths.map do |path|
    absolute = File.join(site.source, path)
    raise Jekyll::Errors::FatalException, "Missing literature explorer asset #{path}" unless File.file?(absolute)

    path + "\0" + File.binread(absolute) + "\0"
  end.join
  site.data['quantum_access_literature_version'] = Digest::SHA256.hexdigest(bytes)[0, 12]
  # Liquid's year-only sort can reorder equal-year entries across Ruby versions.
  # Keep the source snapshot intact and give the bibliography a total order.
  site.data['quantum_access_literature_by_year'] = site.data.fetch('quantum_access_literature').fetch('nodes').sort_by do |paper|
    [paper.fetch('year'), paper.fetch('label'), paper.fetch('id')]
  end
end
