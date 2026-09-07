import { McpServer, createMcpHandler } from "@modelcontextprotocol/server";
import { z } from "zod";

let workerEnv;

function toGithubApiUrl(rawUrl) {
  const m = rawUrl.match(/^https:\/\/raw\.githubusercontent\.com\/([^/]+)\/([^/]+)\/([^/]+)\/(.+)$/);
  if (!m) return null;
  const [, owner, repo, branch, path] = m;
  return `https://api.github.com/repos/${owner}/${repo}/contents/${path}?ref=${branch}`;
}

function createServer() {
  const server = new McpServer({
    name: "hello-server",
    version: "1.0.0"
  });

  server.registerTool(
    "hello",
    {
      description: "Returns a greeting",
      inputSchema: z.object({
        name: z.string().optional()
      })
    },
    async ({ name }) => ({
      content: [
        {
          type: "text",
          text: `Hello, ${name ?? "World"}!`
        }
      ]
    })
  );

  server.registerTool(
    "ask_ai",
    {
      description: "Fetches a file from a public or private (via token) URL and asks Gemini to analyze it, returning only Gemini's answer.",
      inputSchema: z.object({
        source_url: z.string().describe("Public URL of the file (e.g. raw.githubusercontent.com link)"),
        task: z.string().describe("What to do with the file, e.g. 'find bugs'")
      })
    },
    async ({ source_url, task }) => {
      const env = workerEnv;
      if (!env) {
        return { content: [{ type: "text", text: "env hâlâ yakalanamadı" }] };
      }

      const apiUrl = toGithubApiUrl(source_url);
      const fetchUrl = apiUrl || source_url;

      const fileRes = await fetch(fetchUrl, {
        headers: {
          Authorization: `Bearer ${env.GITHUB_TOKEN}`,
          Accept: "application/vnd.github.raw+json",
          "User-Agent": "ai-router-worker"
        }
      });
      if (!fileRes.ok) {
        return { content: [{ type: "text", text: `Dosya çekilemedi: ${fileRes.status}` }] };
      }
      const fileContent = await fileRes.text();

      const prompt = `${task}\n\nKısa ve öz cevap ver, sadece bulguları listele, dosyayı tekrar yazma.\n\n---DOSYA---\n${fileContent}`;

      const geminiRes = await fetch(
        `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=${env.GEMINI_API_KEY}`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ contents: [{ parts: [{ text: prompt }] }] })
        }
      );

      if (!geminiRes.ok) {
        const errText = await geminiRes.text();
        return { content: [{ type: "text", text: `Gemini hata: ${geminiRes.status} — ${errText}` }] };
      }

      const data = await geminiRes.json();
      const answer = data?.candidates?.[0]?.content?.parts?.[0]?.text ?? "Gemini boş cevap döndü.";
      return { content: [{ type: "text", text: answer }] };
    }
  );

  return server;
}

function captureEnv(fn, boundTo) {
  return function (...args) {
    if (args.length >= 2) workerEnv = args[1];
    return fn.apply(boundTo, args);
  };
}

const rawHandler = createMcpHandler(createServer);

export default new Proxy(rawHandler, {
  apply(target, thisArg, args) {
    if (args.length >= 2) workerEnv = args[1];
    return Reflect.apply(target, thisArg, args);
  },
  get(target, prop, receiver) {
    const value = Reflect.get(target, prop, receiver);
    return typeof value === "function" ? captureEnv(value, target) : value;
  }
});
