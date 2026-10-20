import MarkdownIt from "markdown-it";
import { HtmlBasePlugin, IdAttributePlugin } from "@11ty/eleventy";
import { tex } from "@mdit/plugin-tex";
import katex from "katex";
import highlight from "highlight.js/lib/core";
import python from "highlight.js/lib/languages/python";
import css from "highlight.js/lib/languages/css";
import rss from "@11ty/eleventy-plugin-rss";

highlight.registerLanguage("python", python);
highlight.registerLanguage("css", css);

export default function (eleventy) {
  eleventy.addPlugin(rss);
  eleventy.addPlugin(HtmlBasePlugin);
  eleventy.addPlugin(IdAttributePlugin);
  const markdown = new MarkdownIt({
    html: true,
    highlight(code, language) {
      return highlight.getLanguage(language)
        ? highlight.highlight(code, { language }).value
        : "";
    },
  }).use(tex, {
    render(content, displayMode) {
      return katex.renderToString(content, {
        displayMode, throwOnError: true, strict: "error", trust: false,
      });
    },
  });
  const renderFence = markdown.renderer.rules.fence;
  markdown.renderer.rules.fence = (...args) => renderFence(...args).replace("<pre>", '<pre tabindex="0">');
  eleventy.setLibrary("md", markdown);

  eleventy.addPassthroughCopy({ public: "/" });
  eleventy.addPassthroughCopy("src/assets");
  eleventy.addPassthroughCopy({ "node_modules/katex/dist/katex.min.css": "assets/katex/katex.min.css" });
  eleventy.addPassthroughCopy({ "node_modules/katex/dist/fonts/*.woff2": "assets/katex/fonts" });
  eleventy.addPassthroughCopy({ "node_modules/katex/LICENSE": "assets/katex/LICENSE" });

  eleventy.addFilter("dateLabel", (value) => new Intl.DateTimeFormat("en", {
    day: "numeric", month: "long", year: "numeric", timeZone: "UTC",
  }).format(new Date(value)));
  eleventy.addFilter("isoDate", (value) => new Date(value).toISOString().slice(0, 10));
  eleventy.addFilter("absoluteUrl", (path, base) => new URL(path, base).href);

  return {
    dir: { input: "src", output: "dist" },
    templateFormats: ["md", "njk"],
    markdownTemplateEngine: false,
    htmlTemplateEngine: "njk",
  };
}
