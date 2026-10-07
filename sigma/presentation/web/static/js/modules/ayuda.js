// Help page: renders the LaTeX in the theoretical reference with KaTeX (auto-render).
// Loaded as a module after katex.min.js and katex-auto-render.min.js, so `renderMathInElement` is ready.

if (window.renderMathInElement) {
  renderMathInElement(document.body, {
    delimiters: [
      { left: "$$", right: "$$", display: true },
      { left: "$", right: "$", display: false },
    ],
    throwOnError: false,
  });
}
