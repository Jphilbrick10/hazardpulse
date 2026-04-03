(function () {
  const STORAGE_KEY = "hp_theme";
  const COOKIE_NAME = "hp_theme";
  const COOKIE_MAX_AGE = 60 * 60 * 24 * 365;
  const THEME_ATTRIBUTE = "data-theme";
  const DARK_META_COLOR = "#0A0F1A";
  const LIGHT_META_COLOR = "#f6f9ff";

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
    const secure = window.location.protocol === "https:" ? "; Secure" : "";
    document.cookie =
      COOKIE_NAME +
      "=" +
      theme +
      "; path=/; max-age=" +
      COOKIE_MAX_AGE +
      "; SameSite=Lax" +
      secure;
  }

  function applyDocumentTheme(theme) {
    const isDark = theme === "dark";
    document.documentElement.setAttribute(THEME_ATTRIBUTE, isDark ? "dark" : "light");
    if (document.body) {
      document.body.setAttribute(THEME_ATTRIBUTE, isDark ? "dark" : "light");
    }
    const themeColor = document.querySelector('meta[name="theme-color"]');
    if (themeColor) {
      themeColor.setAttribute("content", isDark ? DARK_META_COLOR : LIGHT_META_COLOR);
    }
    document.documentElement.style.colorScheme = isDark ? "dark" : "light";
  }

  function syncThemeToggles(theme) {
    const isDark = theme === "dark";
    document.querySelectorAll(".theme-toggle").forEach((input) => {
      if (!(input instanceof HTMLInputElement)) return;
      input.checked = isDark;
      input.setAttribute(
        "aria-label",
        isDark ? "Switch to light mode" : "Switch to dark mode"
      );
    });
  }

  function applyTheme(theme) {
    applyDocumentTheme(theme);
    syncThemeToggles(theme);
  }

  function syncFromStoredTheme() {
    const theme = readTheme();
    if (theme === "dark" || theme === "light") {
      persistTheme(theme);
      applyTheme(theme);
    }
  }

  const initialTheme = readTheme();
  if (initialTheme === "dark" || initialTheme === "light") {
    applyDocumentTheme(initialTheme);
  }

  document.addEventListener("change", (event) => {
    const target = event.target;
    if (!(target instanceof HTMLInputElement)) return;
    if (!target.classList.contains("theme-toggle")) return;
    const theme = target.checked ? "dark" : "light";
    persistTheme(theme);
    applyTheme(theme);
  });

  document.addEventListener("DOMContentLoaded", syncFromStoredTheme, { once: true });
  window.addEventListener("pageshow", syncFromStoredTheme);
  syncFromStoredTheme();
})();
