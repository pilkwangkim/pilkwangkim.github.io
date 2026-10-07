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
    const korean = root.lang.startsWith('ko');
    const label = korean
      ? (collapsed ? '사이드바 펼치기' : '사이드바 접기')
      : (collapsed ? 'Expand sidebar' : 'Collapse sidebar');
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
  const validLanguages = ['all', 'ko', 'en'];
  function preferredLanguage() {
    const query = new URL(window.location.href).searchParams.get('lang');
    if (validLanguages.includes(query)) {
      writePreference('pilkwang:language', query);
      return query;
    }
    const selected = readPreference('pilkwang:language');
    return validLanguages.includes(selected) ? selected : 'all';
  }

  function applyLanguage(selected, updateUrl = false) {
    controls.forEach((control) => {
      control.querySelectorAll('[data-language-filter]').forEach((button) => {
        button.setAttribute('aria-pressed', String(button.dataset.languageFilter === selected));
      });
    });
    const rows = [...document.querySelectorAll('[data-post-languages]')];
    rows.forEach((row) => {
      const available = row.dataset.postLanguages.split(/\s+/);
      row.hidden = selected !== 'all' && !available.includes(selected);
      row.querySelectorAll('[data-post-language]').forEach((link) => {
        link.hidden = selected !== 'all' && link.dataset.postLanguage !== selected;
      });
      const titles = [...row.querySelectorAll('[data-post-title-language]')];
      const preferred = selected === 'all' ? (available.includes('ko') ? 'ko' : 'en') : selected;
      titles.forEach((title) => { title.hidden = title.dataset.postTitleLanguage !== preferred; });
    });
    document.querySelectorAll('[data-language-section]').forEach((section) => {
      section.hidden = ![...section.querySelectorAll('[data-post-languages]')].some((row) => !row.hidden);
    });
    document.querySelectorAll('[data-language-empty]').forEach((message) => {
      const scope = message.closest('[data-language-scope]') || document;
      message.hidden = [...scope.querySelectorAll('[data-post-languages]')].some((row) => !row.hidden);
    });
    if (updateUrl) {
      const url = new URL(window.location.href);
      if (selected === 'all') url.searchParams.delete('lang');
      else url.searchParams.set('lang', selected);
      history.replaceState(null, '', url);
    }
  }

  controls.forEach((control) => {
    control.addEventListener('click', (event) => {
      const button = event.target.closest('[data-language-filter]');
      if (!button || !validLanguages.includes(button.dataset.languageFilter)) return;
      const selected = button.dataset.languageFilter;
      writePreference('pilkwang:language', selected);
      applyLanguage(selected, true);
    });
  });
  window.addEventListener('popstate', () => {
    applyLanguage(preferredLanguage());
  });
  applyLanguage(preferredLanguage());
})();
