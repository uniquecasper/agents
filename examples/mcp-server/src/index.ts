import { McpServer, createMcpHandler } from "@modelcontextprotocol/server";
import { z } from "zod";

// ─────────────────────────────────────────────────────────────
// AI Router v2.1 — free-tier fallback zincirleri
// Bir modelin kotası (429) bitince / model yoksa (404) / sunucu
// meşgulse (500/503) sıradaki modele otomatik geçer.
// Limitler: AI Studio free tier (Eki 2026) — günlük istek (RPD)
// ─────────────────────────────────────────────────────────────

let workerEnv: any;

const GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta";

const TEXT_ALIASES: Record<string, string[]> = {
  // Varsayılan: kaliteli modeller (günde 20'şer) → bitince hafif modeller (500/gün)
  flash: [
    "gemini-3.8-flash", // 20/gün
    "gemini-3.7-flash", // 20/gün
    "gemini-3.6-flash", // 20/gün
    "gemini-3.5-flash", // 20/gün
    "gemini-3-flash-preview", // 20/gün
    "gemini-2.5-flash", // 20/gün
    "gemini-3.1-flash-lite", // 500/gün — kaliteli zincir bitince hafif zincire düş
    "gemini-3.5-flash-lite", // 500/gün
    "gemini-2.5-flash-lite", // 20/gün (son çare)
  ],
  // Hafif/toplu işler
  lite: ["gemini-3.1-flash-lite", "gemini-3.5-flash-lite", "gemini-2.5-flash-lite"],
  // Günde 14.4K ama context 16K — büyük dosya SIĞMAZ
  gemma: ["gemma-4-31b-it", "gemma-4-26b-a4b-it"],
};

const DEFAULT_TEXT = "flash";
const EMBED_MODELS = ["gemini-embedding-2", "gemini-embedding-001"];

// Bu durumlarda sıradaki modele geç (diğer hatalarda zinciri kır — prompt hatası vs.)
const RETRY_STATUSES = new Set([404, 429, 500, 503]);

const text = (t: string) => ({ content: [{ type: "text" as const, text: t }] });

function getKey(): string | null {
  return workerEnv?.GEMINI_API_KEY ?? null;
}

function resolveChain(modelInput?: string): string[] {
  const m = (modelInput ?? DEFAULT_TEXT).trim();
  return TEXT_ALIASES[m] ?? [m];
}

// thinkingLevel sadece Gemini 3+ ailesinde var; 2.5 ve Gemma'da 400 verir
function supportsThinkingLevel(model: string): boolean {
  return /^gemini-3/.test(model);
}

function toGithubApiUrl(rawUrl: string): string | null {
  const m = rawUrl.match(
    /^https:\/\/raw\.githubusercontent\.com\/([^/]+)\/([^/]+)\/([^/]+)\/(.+)$/
  );
  if (!m) return null;
  const [, owner, repo, branch, path] = m;
  return `https://api.github.com/repos/${owner}/${repo}/contents/${path}?ref=${branch}`;
}

type GenResult =
  | { ok: true; answer: string; usedModel: string; skipped: string[] }
  | { ok: false; error: string };

async function generate(chain: string[], prompt: string, thinking: boolean): Promise<GenResult> {
  const key = getKey();
  if (!key) return { ok: false, error: "GEMINI_API_KEY secret olarak eklenmemiş." };

  const skipped: string[] = [];
  const errors: string[] = [];

  for (const model of chain) {
    const body: any = { contents: [{ parts: [{ text: prompt }] }] };
    if (thinking && supportsThinkingLevel(model)) {
      body.generationConfig = { thinkingConfig: { thinkingLevel: "high" } };
    }

    let res: Response;
    try {
      res = await fetch(`${GEMINI_BASE}/models/${model}:generateContent`, {
        method: "POST",
        headers: { "Content-Type": "application/json", "x-goog-api-key": key },
        body: JSON.stringify(body),
      });
    } catch (err: any) {
      // Ağ hatası — sıradaki modeli dene
      skipped.push(model);
      errors.push(`${model} → network: ${err?.message ?? String(err)}`);
      continue;
    }

    if (res.ok) {
      const data: any = await res.json();
      const answer =
        data?.candidates?.[0]?.content?.parts?.map((p: any) => p.text ?? "").join("") ?? "";
      return {
        ok: true,
        answer: answer || "Gemini boş cevap döndü.",
        usedModel: model,
        skipped,
      };
    }

    const errText = await res.text();
    skipped.push(model);
    errors.push(`${model} → ${res.status}: ${errText.slice(0, 200)}`);
    if (!RETRY_STATUSES.has(res.status)) break;
  }

  return { ok: false, error: errors.join("\n") };
}

function formatResult(r: GenResult): string {
  if (!r.ok) return `Gemini hata (tüm zincir denendi):\n${r.error}`;
  const note = r.skipped.length ? `\n(atlanan: ${r.skipped.join(", ")})` : "";
  return `${r.answer}\n\n— model: ${r.usedModel}${note}`;
}

function createServer() {
  const server = new McpServer({ name: "ai-router", version: "2.1.0" });

  server.registerTool(
    "hello",
    { description: "Returns a greeting", inputSchema: z.object({ name: z.string().optional() }) },
    async ({ name }: { name?: string }) => text(`Hello, ${name ?? "World"}!`)
  );

  server.registerTool(
    "list_models",
    {
      description:
        "Lists models this API key can see (exact IDs). Use to verify/fix model IDs. Optional filter substring, e.g. 'flash', 'embedding'.",
      inputSchema: z.object({ filter: z.string().optional() }),
    },
    async ({ filter }: { filter?: string }) => {
      const key = getKey();
      if (!key) return text("GEMINI_API_KEY secret olarak eklenmemiş.");
      const res = await fetch(`${GEMINI_BASE}/models?pageSize=200`, {
        headers: { "x-goog-api-key": key },
      });
      if (!res.ok) return text(`list_models hata: ${res.status} — ${(await res.text()).slice(0, 300)}`);
      const data: any = await res.json();
      const rows = (data.models ?? [])
        .map(
          (m: any) =>
            `${String(m.name).replace("models/", "")} [${(m.supportedGenerationMethods ?? []).join(",")}]`
        )
        .filter((r: string) => !filter || r.toLowerCase().includes(filter.toLowerCase()));
      return text(rows.length ? rows.join("\n") : "Eşleşen model yok.");
    }
  );

  server.registerTool(
    "ask_text",
    {
      description:
        "Ask a Gemini/Gemma text model a question. model: 'flash' (default; falls through 3.8→3.7→3.6→3.5→3→2.5 then lite models when quota runs out), 'lite' (500/day), 'gemma' (16K context only), or a raw model ID.",
      inputSchema: z.object({
        prompt: z.string().describe("The question/task"),
        model: z.string().optional(),
        extended_thinking: z.boolean().optional().describe("Deeper reasoning (slower). Default false."),
      }),
    },
    async ({
      prompt,
      model,
      extended_thinking,
    }: {
      prompt: string;
      model?: string;
      extended_thinking?: boolean;
    }) => {
      try {
        const r = await generate(resolveChain(model), prompt, !!extended_thinking);
        return text(formatResult(r));
      } catch (err: any) {
        return text(`Beklenmedik hata: ${err?.message ?? String(err)}`);
      }
    }
  );

  server.registerTool(
    "ask_ai",
    {
      description:
        "Fetches a file from a public or private (via token) GitHub URL and asks Gemini to analyze it, returning only the answer. model: 'flash' (default), 'lite', or raw ID. (gemma önerilmez: 16K context.)",
      inputSchema: z.object({
        source_url: z
          .string()
          .describe("URL of the file (raw.githubusercontent.com link works for private repos too)"),
        task: z.string().describe("What to do with the file, e.g. 'find bugs'"),
        model: z.string().optional(),
        extended_thinking: z.boolean().optional(),
      }),
    },
    async ({
      source_url,
      task,
      model,
      extended_thinking,
    }: {
      source_url: string;
      task: string;
      model?: string;
      extended_thinking?: boolean;
    }) => {
      try {
        if (!workerEnv?.GITHUB_TOKEN) return text("GITHUB_TOKEN secret olarak eklenmemiş.");
        // GÜVENLİK: GITHUB_TOKEN sadece GitHub'a gider. Başka host'a asla gönderme.
        const fetchUrl = toGithubApiUrl(source_url) ?? source_url;
        let host = "";
        try {
          host = new URL(fetchUrl).hostname;
        } catch {
          return text("Geçersiz URL.");
        }
        if (host !== "api.github.com") {
          return text("Sadece raw.githubusercontent.com veya api.github.com linkleri desteklenir.");
        }
        const fileRes = await fetch(fetchUrl, {
          headers: {
            Authorization: `Bearer ${workerEnv.GITHUB_TOKEN}`,
            Accept: "application/vnd.github.raw+json",
            "User-Agent": "ai-router-worker",
          },
        });
        if (!fileRes.ok) return text(`Dosya çekilemedi: ${fileRes.status}`);
        const fileContent = await fileRes.text();
        const prompt = `${task}\n\nKısa ve öz cevap ver, sadece bulguları listele, dosyayı tekrar yazma.\n\n---DOSYA---\n${fileContent}`;
        const r = await generate(resolveChain(model), prompt, !!extended_thinking);
        return text(formatResult(r));
      } catch (err: any) {
        return text(`Beklenmedik hata: ${err?.message ?? String(err)}`);
      }
    }
  );

  server.registerTool(
    "embed",
    {
      description:
        "Embeds text with Gemini Embedding. Returns dimension + first 8 values by default; set full=true for the whole vector.",
      inputSchema: z.object({
        text: z.string(),
        full: z.boolean().optional(),
      }),
    },
    async ({ text: input, full }: { text: string; full?: boolean }) => {
      try {
        const key = getKey();
        if (!key) return text("GEMINI_API_KEY secret olarak eklenmemiş.");

        let lastError = "";
        for (const model of EMBED_MODELS) {
          const res = await fetch(`${GEMINI_BASE}/models/${model}:embedContent`, {
            method: "POST",
            headers: { "Content-Type": "application/json", "x-goog-api-key": key },
            body: JSON.stringify({ content: { parts: [{ text: input }] } }),
          });
          if (res.ok) {
            const data: any = await res.json();
            const v: number[] = data?.embedding?.values ?? [];
            return text(
              full
                ? JSON.stringify(v)
                : `model=${model} dim=${v.length}\nilk 8 değer: ${JSON.stringify(v.slice(0, 8))}\n(tamamı için full=true)`
            );
          }
          lastError = `${model} → ${res.status}: ${(await res.text()).slice(0, 200)}`;
          if (!RETRY_STATUSES.has(res.status)) break;
        }
        return text(`Embed hata: ${lastError}`);
      } catch (err: any) {
        return text(`Beklenmedik hata: ${err?.message ?? String(err)}`);
      }
    }
  );

  return server;
}

function captureEnv(fn: Function, boundTo: any) {
  return function (...args: any[]) {
    if (args.length >= 2) workerEnv = args[1];
    return fn.apply(boundTo, args);
  };
}

const rawHandler: any = createMcpHandler(createServer);

export default new Proxy(rawHandler, {
  apply(target, thisArg, args) {
    if (args.length >= 2) workerEnv = args[1];
    return Reflect.apply(target, thisArg, args);
  },
  get(target, prop, receiver) {
    const value = Reflect.get(target, prop, receiver);
    return typeof value === "function" ? captureEnv(value, target) : value;
  },
});
