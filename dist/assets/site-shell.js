(function () {
  const STORAGE_KEY = "hp_theme";
  const COOKIE_NAME = "hp_theme";
  const COOKIE_MAX_AGE = 60 * 60 * 24 * 365;

  function readCookieTheme() {
    const match = document.cookie.match(/(?:^|;\s*)hp_theme=(dark|light)(?:;|$)/i);
    return match ? match[1].toLowerCase() : null;
  }

  function readTheme() {
    const cookieTheme = readCookieTheme();
    if (cookieTheme === "dark" || cookieTheme === "light") {
      return cookieTheme;
    }
    try {
      const stored = window.localStorage.getItem(STORAGE_KEY);
      if (stored === "dark" || stored === "light") {
        return stored;
      }
    } catch {}
    return null;
  }

  function persistTheme(theme) {
    try {
      window.localStorage.setItem(STORAGE_KEY, theme);
    } catch {}
    document.cookie =
      COOKIE_NAME +
      "=" +
      theme +
      "; path=/; max-age=" +
      COOKIE_MAX_AGE +
      "; SameSite=Lax";
  }

  function applyTheme(theme) {
    const isDark = theme === "dark";
    document.querySelectorAll(".theme-toggle").forEach((input) => {
      if (!(input instanceof HTMLInputElement)) return;
      input.checked = isDark;
      input.setAttribute(
        "aria-label",
        isDark ? "Switch to light mode" : "Switch to dark mode"
      );
    });
    document.documentElement.style.colorScheme = isDark ? "dark" : "light";
  }

  function syncFromStoredTheme() {
    const theme = readTheme();
    if (theme === "dark" || theme === "light") {
      persistTheme(theme);
      applyTheme(theme);
    }
  }

  document.addEventListener("change", (event) => {
    const target = event.target;
    if (!(target instanceof HTMLInputElement)) return;
    if (!target.classList.contains("theme-toggle")) return;
    const theme = target.checked ? "dark" : "light";
    persistTheme(theme);
    applyTheme(theme);
  });

  window.addEventListener("pageshow", syncFromStoredTheme);
  syncFromStoredTheme();
})();
