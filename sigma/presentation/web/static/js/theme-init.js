// Runs before first paint (classic script) so there is no light/dark flash.
(function () {
  var stored = null;
  try { stored = localStorage.getItem("sigma-theme"); } catch (e) { /* storage unavailable */ }
  var dark = stored ? stored === "dark" : window.matchMedia("(prefers-color-scheme: dark)").matches;
  document.documentElement.classList.toggle("dark", dark);
})();
