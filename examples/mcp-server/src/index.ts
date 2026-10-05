import { McpServer, createMcpHandler } from "@modelcontextprotocol/server";
import { z } from "zod";

// ─────────────────────────────────────────────────────────────
// AI Router v2.2 — profiller + detaylı ayarlar + free-tier fallback
//
// profile:  fast | deep | code | web | json   (hepsi opsiyonel)
// override: system_instruction, temperature, top_p, max_output_tokens,
//           thinking_level, stop_sequences, google_search,
//           code_execution, url_context, json_mode
// Öncelik:  açık verilen parametre > profil > API varsayılanı
// ─────────────────────────────────────────────────────────────

let workerEnv: any;

const GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta";

// Her alias bir zincir: kota (429) bitince / model yoksa (404) / sunucu meşgulse (500, 503) sıradakine geç.
// gemini-2.5-* yeni hesaplara kapalı (404) — çıkarıldı.
const TEXT_ALIASES: Record<string, string[]> = {
  flash: [
    "gemini-3.8-flash", // 20/gün
    "gemini-3.7-flash", // 20/gün
    "gemini-3.6-flash", // 20/gün
    "gemini-3.5-flash", // 20/gün
    "gemini-3-flash-preview", // 20/gün
    "gemini-3.1-flash-lite", // 500/gün — kaliteli zincir bitince hafif zincire düş
    "gemini-3.5-flash-lite", // 500/gün
  ],
  lite: ["gemini-3.1-flash-lite", "gemini-3.5-flash-lite"],
  // Günde 14.4K ama context 16K; düşünme notlarını cevaba döküyor — sadece basit işler
  gemma: ["gemma-4-31b-it", "gemma-4-26b-a4b-it"],
};

const DEFAULT_TEXT = "flash";
const EMBED_MODELS = ["gemini-embedding-2", "gemini-embedding-001"];
const RETRY_STATUSES = new Set([404, 429, 500, 503]);

// ── Ayar tipleri ──────────────────────────────────────────────
type ThinkingLevel = "minimal" | "low" | "medium" | "high";

type GenOpts = {
  system?: string;
  temperature?: number;
  topP?: number;
  maxOutputTokens?: number;
  thinkingLevel?: ThinkingLevel;
  stopSequences?: string[];
  googleSearch?: boolean;
  codeExecution?: boolean;
  urlContext?: boolean;
  json?: boolean;
};

// ── Profiller ─────────────────────────────────────────────────
const PROFILES: Record<string, { model?: string; opts: GenOpts }> = {
  // Hızlı/kısa cevap: az düşünme, kısa çıktı
  fast: { opts: { thinkingLevel: "low", maxOutputTokens: 2048 } },
  // Derin analiz/muhakeme: tam düşünme
  deep: { opts: { thinkingLevel: "high" } },
  // Kod/hesap: orta düşünme + Python çalıştırma
  code: { opts: { thinkingLevel: "medium", codeExecution: true } },
  // Web: Google Search + verilen linkleri okuma (Gemini 3'te free tier'da Search kapalı olabilir)
  web: { opts: { thinkingLevel: "medium", googleSearch: true, urlContext: true } },
  // Yapılandırılmış çıktı: sadece JSON döner
  json: { opts: { thinkingLevel: "low", json: true } },
};

const optionShape = {
  model: z.string().optional().describe("flash (default) | lite | gemma | raw model ID"),
  profile: z
    .enum(["fast", "deep", "code", "web", "json"])
    .optional()
    .describe(
      "fast: quick/short. deep: full reasoning. code: Python execution. web: Google Search + URL reading. json: JSON-only output."
    ),
  system_instruction: z.string().optional().describe("Tone/role instructions for the model"),
  temperature: z.number().min(0).max(2).optional(),
  top_p: z.number().min(0).max(1).optional(),
  max_output_tokens: z.number().int().positive().optional(),
  thinking_level: z.enum(["minimal", "low", "medium", "high"]).optional(),
  stop_sequences: z.array(z.string()).optional(),
  google_search: z.boolean().optional().describe("Google Search grounding (may be unavailable on free tier for Gemini 3)"),
  code_execution: z.boolean().optional(),
  url_context: z.boolean().optional().describe("Let the model read URLs mentioned in the prompt"),
  json_mode: z.boolean().optional().describe("Force JSON output"),
  extended_thinking: z.boolean().optional().describe("Shortcut for thinking_level=high"),
};

// ── Yardımcılar ───────────────────────────────────────────────
const text = (t: string) => ({ content: [{ type: "text" as const, text: t }] });

function getKey(): string | null {
  return workerEnv?.GEMINI_API_KEY ?? null;
}

function resolveChain(modelInput?: string): string[] {
  const m = (modelInput ?? DEFAULT_TEXT).trim();
  return TEXT_ALIASES[m] ?? [m];
}

const isGemma = (model: string) => model.startsWith("gemma");
// thinkingLevel sadece Gemini 3+ ailesinde var
const supportsThinkingLevel = (model: string) => /^gemini-3/.test(model);

// Profil + açık parametreler → tek bir istek yapılandırması
function resolveRequest(a: any): { chain: string[]; opts: GenOpts; profile?: string } {
  const p = a.profile ? PROFILES[a.profile] : undefined;
  const base: GenOpts = p?.opts ?? {};
  const opts: GenOpts = {
    system: a.system_instruction ?? base.system,
    temperature: a.temperature ?? base.temperature,
    topP: a.top_p ?? base.topP,
    maxOutputTokens: a.max_output_tokens ?? base.maxOutputTokens,
    thinkingLevel: a.thinking_level ?? (a.extended_thinking ? "high" : base.thinkingLevel),
    stopSequences: a.stop_sequences ?? base.stopSequences,
    googleSearch: a.google_search ?? base.googleSearch,
    codeExecution: a.code_execution ?? base.codeExecution,
    urlContext: a.url_context ?? base.urlContext,
    json: a.json_mode ?? base.json,
  };
  return { chain: resolveChain(a.model ?? p?.model), opts, profile: a.profile };
}

function toGithubApiUrl(rawUrl: string): string | null {
  const m = rawUrl.match(
    /^https:\/\/raw\.githubusercontent\.com\/([^/]+)\/([^/]+)\/([^/]+)\/(.+)$/
  );
  if (!m) return null;
  const [, owner, repo, branch, path] = m;
  return `https://api.github.com/repos/${owner}/${repo}/contents/${path}?ref=${branch}`;
}

// strip=true → düşünme ve search ayarlarını çıkar (model reddederse son çare)
function buildBody(model: string, prompt: string, o: GenOpts, strip: boolean): any {
  const gemma = isGemma(model);
  // Gemma system instruction desteklemiyor → prompt'un başına ekle
  const promptText = gemma && o.system ? `${o.system}\n\n${prompt}` : prompt;
  const body: any = { contents: [{ parts: [{ text: promptText }] }] };
  if (!gemma && o.system) body.systemInstruction = { parts: [{ text: o.system }] };

  const gc: any = {};
  if (o.temperature !== undefined) gc.temperature = o.temperature;
  if (o.topP !== undefined) gc.topP = o.topP;
  if (o.maxOutputTokens !== undefined) gc.maxOutputTokens = o.maxOutputTokens;
  if (o.stopSequences?.length) gc.stopSequences = o.stopSequences;
  if (o.json) gc.responseMimeType = "application/json";
  if (!strip && o.thinkingLevel && supportsThinkingLevel(model)) {
    gc.thinkingConfig = { thinkingLevel: o.thinkingLevel };
  }
  if (Object.keys(gc).length) body.generationConfig = gc;

  // JSON modunda tool'lar kapalı (API ikisini birlikte kabul etmiyor); Gemma tool desteklemiyor
  if (!gemma && !o.json) {
    const tools: any[] = [];
    if (!strip && o.googleSearch) tools.push({ google_search: {} });
    if (o.codeExecution) tools.push({ code_execution: {} });
    if (o.urlContext) tools.push({ url_context: {} });
    if (tools.length) body.tools = tools;
  }
  return body;
}

type GenResult =
  | { ok: true; answer: string; usedModel: string; skipped: string[]; notes: string[] }
  | { ok: false; error: string };

async function post(model: string, key: string, body: any): Promise<Response> {
  return fetch(`${GEMINI_BASE}/models/${model}:generateContent`, {
    method: "POST",
    headers: { "Content-Type": "application/json", "x-goog-api-key": key },
    body: JSON.stringify(body),
  });
}

async function generate(chain: string[], prompt: string, o: GenOpts): Promise<GenResult> {
  const key = getKey();
  if (!key) return { ok: false, error: "GEMINI_API_KEY secret olarak eklenmemiş." };

  const skipped: string[] = [];
  const errors: string[] = [];
  const notes: string[] = [];

  for (const model of chain) {
    let res: Response;
    try {
      res = await post(model, key, buildBody(model, prompt, o, false));

      // Model thinking/search ayarını reddettiyse (400/403): ayarsız bir kez daha dene
      if (!res.ok && (res.status === 400 || res.status === 403) && (o.thinkingLevel || o.googleSearch)) {
        const retry = await post(model, key, buildBody(model, prompt, o, true));
        if (retry.ok) {
          res = retry;
          notes.push(`${model}: thinking/search ayarı reddedildi, çıkarılıp çalıştırıldı`);
        }
      }
    } catch (err: any) {
      skipped.push(model);
      errors.push(`${model} → network: ${err?.message ?? String(err)}`);
      continue;
    }

    if (res.ok) {
      const data: any = await res.json();
      const parts: any[] = data?.candidates?.[0]?.content?.parts ?? [];
      let answer = parts.map((p: any) => p.text ?? "").join("");
      // code execution çıktısı ayrı part olarak gelir — görünür kıl
      for (const p of parts) {
        if (p.executableCode?.code) answer += `\n\n[kod]\n${p.executableCode.code}`;
        if (p.codeExecutionResult?.output) answer += `\n[çıktı]\n${p.codeExecutionResult.output}`;
      }
      return {
        ok: true,
        answer: answer.trim() || "Gemini boş cevap döndü.",
        usedModel: model,
        skipped,
        notes,
      };
    }

    const errText = await res.text();
    skipped.push(model);
    errors.push(`${model} → ${res.status}: ${errText.slice(0, 200)}`);
    if (!RETRY_STATUSES.has(res.status)) break;
  }

  return { ok: false, error: errors.join("\n") };
}

function formatResult(r: GenResult, profile?: string): string {
  if (!r.ok) return `Gemini hata (tüm zincir denendi):\n${r.error}`;
  const tag = profile ? ` | profil: ${profile}` : "";
  const skip = r.skipped.length ? `\n(atlanan: ${r.skipped.join(", ")})` : "";
  const notes = r.notes.length ? `\n(${r.notes.join("; ")})` : "";
  return `${r.answer}\n\n— model: ${r.usedModel}${tag}${skip}${notes}`;
}

// ── MCP server ────────────────────────────────────────────────
function createServer() {
  const server = new McpServer({ name: "ai-router", version: "2.2.0" });

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
        "Ask a Gemini/Gemma text model. model: 'flash' (default; auto-falls through 3.8→3.7→3.6→3.5→3 then lite models when quota runs out), 'lite' (500/day), 'gemma' (16K context, noisy), or a raw model ID. profile: fast | deep | code | web | json. All other settings (temperature, thinking_level, system_instruction, tools...) are optional overrides.",
      inputSchema: z.object({ prompt: z.string().describe("The question/task"), ...optionShape }),
    },
    async (args: any) => {
      try {
        const { chain, opts, profile } = resolveRequest(args);
        const r = await generate(chain, args.prompt, opts);
        return text(formatResult(r, profile));
      } catch (err: any) {
        return text(`Beklenmedik hata: ${err?.message ?? String(err)}`);
      }
    }
  );

  server.registerTool(
    "ask_ai",
    {
      description:
        "Fetches a file from a public or private (via token) GitHub URL and asks Gemini to analyze it, returning only the answer. Same model/profile/settings options as ask_text. (gemma not recommended: 16K context; lite is weak at counting.)",
      inputSchema: z.object({
        source_url: z
          .string()
          .describe("URL of the file (raw.githubusercontent.com link works for private repos too)"),
        task: z.string().describe("What to do with the file, e.g. 'find bugs'"),
        ...optionShape,
      }),
    },
    async (args: any) => {
      try {
        if (!workerEnv?.GITHUB_TOKEN) return text("GITHUB_TOKEN secret olarak eklenmemiş.");

        // GÜVENLİK: GITHUB_TOKEN sadece GitHub'a gider. Başka host'a asla gönderme.
        const fetchUrl = toGithubApiUrl(args.source_url) ?? args.source_url;
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

        const prompt = `${args.task}\n\nKısa ve öz cevap ver, sadece bulguları listele, dosyayı tekrar yazma.\n\n---DOSYA---\n${fileContent}`;
        const { chain, opts, profile } = resolveRequest(args);
        const r = await generate(chain, prompt, opts);
        return text(formatResult(r, profile));
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
      inputSchema: z.object({ text: z.string(), full: z.boolean().optional() }),
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

// ── Env yakalama (Proxy) ──────────────────────────────────────
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
