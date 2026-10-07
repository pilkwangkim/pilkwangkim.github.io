/* Site-specific navigation. Keep Chirpy's mobile drawer and article tools intact. */
(() => {
  'use strict';

  const root = document.documentElement;
  const desktop = window.matchMedia('(min-width: 850px)');
  const sidebar = document.getElementById('sidebar');
  const toggle = document.getElementById('desktop-sidebar-toggle');
  const readPreference = (key) => {
    try { return localStorage.getItem(key); } catch (_) { return null; }
  };
  const writePreference = (key, value) => {
    try { localStorage.setItem(key, value); } catch (_) { /* Optional preference. */ }
  };

  function syncSidebar() {
    const collapsed = root.dataset.sidebarCollapsed === 'true';
    if (sidebar) sidebar.inert = desktop.matches && collapsed;
    if (!toggle) return;
    const label = collapsed ? 'Expand sidebar' : 'Collapse sidebar';
    toggle.setAttribute('aria-expanded', String(!collapsed));
    toggle.setAttribute('aria-label', label);
    toggle.title = label;
    toggle.querySelector('i')?.classList.toggle('fa-indent', collapsed);
    toggle.querySelector('i')?.classList.toggle('fa-outdent', !collapsed);
  }

  toggle?.addEventListener('click', () => {
    const collapsed = root.dataset.sidebarCollapsed !== 'true';
    root.dataset.sidebarCollapsed = String(collapsed);
    writePreference('pilkwang:sidebar', collapsed ? 'collapsed' : 'expanded');
    syncSidebar();
  });
  desktop.addEventListener('change', syncSidebar);
  syncSidebar();

  const controls = [...document.querySelectorAll('[data-language-controls]')];
  if (!controls.length) return;
  root.dataset.navigationReady = 'true';
  const validLanguages = ['en', 'ko', 'all'];
  const languagePreferenceKey = 'pilkwang:language:v2';
  const articleScopes = [...document.querySelectorAll('[data-article-scope]')];
  let selectedLanguage = preferredLanguage();
  let selectedView = preferredView();

  function preferredLanguage() {
    const query = new URL(window.location.href).searchParams.get('lang');
    if (validLanguages.includes(query)) {
      writePreference(languagePreferenceKey, query);
      return query;
    }
    const selected = readPreference(languagePreferenceKey);
    if (validLanguages.includes(selected)) return selected;
    // This version starts in English without reusing the old default choice.
    writePreference(languagePreferenceKey, 'en');
    return 'en';
  }

  function preferredView() {
    const url = new URL(window.location.href);
    return url.searchParams.get('view') === 'all' || url.hash === '#all-articles' || url.hash.startsWith('#series-') ? 'all' : 'latest';
  }

  function matchesLanguage(row) {
    return selectedLanguage === 'all' || row.dataset.postLanguages?.split(/\s+/).includes(selectedLanguage);
  }

  function articleRows(container) {
    const rows = [...container.querySelectorAll('.topic-post-row')];
    if (container.matches('.topic-post-row')) rows.unshift(container);
    return rows;
  }

  function articleDate(row) {
    const value = row.dataset.articleDate || '';
    const epoch = Number(value);
    if (value && Number.isFinite(epoch)) return Math.abs(epoch) < 1e12 ? epoch * 1000 : epoch;
    return Date.parse(value) || 0;
  }

  function syncEmptyMessages() {
    document.querySelectorAll('[data-language-empty]').forEach((message) => {
      const articleScope = message.closest('[data-article-scope]');
      const scope = articleScope?.querySelector('[data-article-panel="' + selectedView + '"]')
        || message.closest('[data-language-scope]') || document;
      const rows = articleScope ? scope.querySelectorAll('.topic-post-row') : scope.querySelectorAll('[data-post-languages]');
      message.hidden = [...rows].some((row) => !row.hidden && !row.closest('[hidden]'));
    });
  }

  function applyView(selected, updateUrl = false) {
    selectedView = selected === 'all' ? 'all' : 'latest';
    articleScopes.forEach((scope) => {
      scope.querySelectorAll('[data-article-panel]').forEach((panel) => {
        const rows = articleRows(panel);
        rows.forEach((row) => { row.hidden = !matchesLanguage(row); });
        if (panel.dataset.articlePanel === 'latest') {
          const limit = Number.parseInt(panel.dataset.articleLimit, 10) || 5;
          const eligible = rows.filter((row) => !row.hidden).sort((left, right) => {
            const dateDifference = articleDate(right) - articleDate(left);
            if (dateDifference) return dateDifference;
            const leftKey = left.dataset.articleKey || '';
            const rightKey = right.dataset.articleKey || '';
            return leftKey < rightKey ? -1 : leftKey > rightKey ? 1 : 0;
          });
          eligible.forEach((row, index) => { row.hidden = index >= limit; });
        }
        panel.querySelectorAll('details[data-bundle-id]').forEach((bundle) => {
          const count = articleRows(bundle).filter((row) => !row.hidden).length;
          bundle.hidden = count === 0;
          bundle.querySelectorAll('[data-bundle-count]').forEach((label) => { label.textContent = String(count); });
          bundle.querySelectorAll('[data-bundle-count-label]').forEach((label) => { label.textContent = count === 1 ? 'article' : 'articles'; });
          const total = articleRows(bundle).filter(matchesLanguage).length;
          bundle.querySelectorAll('[data-bundle-total-count]').forEach((label) => { label.textContent = String(total); });
        });
        panel.querySelectorAll('[data-article-group]').forEach((group) => {
          group.hidden = !articleRows(group).some((row) => !row.hidden);
        });
        panel.hidden = panel.dataset.articlePanel !== selectedView;
      });
      scope.querySelectorAll('[data-article-view]').forEach((button) => {
        button.setAttribute('aria-pressed', String(button.dataset.articleView === selectedView));
      });
      const count = [...scope.querySelectorAll('.topic-all-posts .topic-post-row')].filter((row) => !row.hidden).length;
      scope.querySelectorAll('[data-article-count]').forEach((label) => { label.textContent = String(count); });
    });
    if (updateUrl) {
      const url = new URL(window.location.href);
      url.searchParams.set('view', selectedView);
      if (selectedView === 'latest' && (url.hash === '#all-articles' || url.hash.startsWith('#series-'))) url.hash = '';
      history.replaceState(null, '', url);
    }
    syncEmptyMessages();
  }

  function applyLanguage(selected, updateUrl = false) {
    selectedLanguage = selected;
    controls.forEach((control) => {
      control.querySelectorAll('[data-language-filter]').forEach((button) => {
        button.setAttribute('aria-pressed', String(button.dataset.languageFilter === selected));
      });
    });
    const rows = [...document.querySelectorAll('[data-post-languages]')];
    rows.forEach((row) => {
      row.hidden = !matchesLanguage(row);
    });
    document.querySelectorAll('.topic-post-row').forEach((row) => {
      const available = row.dataset.postLanguages?.split(/\s+/) || [];
      row.querySelectorAll('[data-post-language]').forEach((link) => {
        link.hidden = selected !== 'all' && link.dataset.postLanguage !== selected;
      });
      const titles = [...row.querySelectorAll('[data-post-title-language]')];
      const preferred = selected === 'all' ? (available.includes('en') ? 'en' : 'ko') : selected;
      titles.forEach((title) => { title.hidden = title.dataset.postTitleLanguage !== preferred; });
    });
    document.querySelectorAll('[data-discovery-title]').forEach((heading) => {
      const titles = [...heading.querySelectorAll('[data-discovery-title-language]')];
      const preferred = selected === 'ko' && titles.some((title) => title.dataset.discoveryTitleLanguage === 'ko') ? 'ko' : 'en';
      titles.forEach((title) => { title.hidden = title.dataset.discoveryTitleLanguage !== preferred; });
    });
    document.querySelectorAll('[data-language-section]').forEach((section) => {
      section.hidden = ![...section.querySelectorAll('[data-post-languages]')].some((row) => !row.hidden);
    });
    if (updateUrl) {
      const url = new URL(window.location.href);
      url.searchParams.set('lang', selected);
      history.replaceState(null, '', url);
    }
    applyView(selectedView);
  }

  controls.forEach((control) => {
    control.addEventListener('click', (event) => {
      const button = event.target.closest('[data-language-filter]');
      if (!button || !validLanguages.includes(button.dataset.languageFilter)) return;
      const selected = button.dataset.languageFilter;
      writePreference(languagePreferenceKey, selected);
      applyLanguage(selected, true);
    });
  });

  articleScopes.forEach((scope) => {
    scope.addEventListener('click', (event) => {
      const button = event.target.closest('[data-article-view]');
      if (button) {
        scope.querySelectorAll('details[data-bundle-id]').forEach((bundle) => { bundle.open = false; });
        applyView(button.dataset.articleView, true);
        return;
      }
      const link = event.target.closest('[data-open-bundle]');
      if (!link) return;
      const details = [...scope.querySelectorAll('[data-article-panel="all"] details[data-bundle-id]')]
        .find((bundle) => bundle.dataset.bundleId === link.dataset.openBundle);
      if (!details) return;
      event.preventDefault();
      applyView('all');
      details.open = true;
      const url = new URL(window.location.href);
      url.searchParams.set('view', 'all');
      url.hash = details.id;
      history.replaceState(null, '', url);
      details.scrollIntoView({ block: 'start' });
    });
  });

  function openHashSeries() {
    const hash = new URL(window.location.href).hash;
    if (!hash.startsWith('#series-')) return;
    const details = document.getElementById(hash.slice(1));
    if (!details?.matches('details[data-bundle-id]') || !details.closest('[data-article-panel="all"]')) return;
    details.open = true;
    requestAnimationFrame(() => { details.scrollIntoView({ block: 'start' }); });
  }

  function applyLocation() {
    selectedView = preferredView();
    applyLanguage(preferredLanguage());
    openHashSeries();
  }
  window.addEventListener('popstate', applyLocation);
  window.addEventListener('hashchange', applyLocation);
  applyLanguage(selectedLanguage);
  openHashSeries();
})();
