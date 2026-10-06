import { McpServer, createMcpHandler } from "@modelcontextprotocol/server";
import { z } from "zod";

// ─────────────────────────────────────────────────────────────
// AI Router v2.3 — profiller + detaylı ayarlar + free-tier fallback + medya/ajan/embedding araçları
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
  // Web: verilen linkleri okuma. Google Search free tier'da Gemini 3'te kapalı (429) — istersen google_search=true ile aç
  web: { opts: { thinkingLevel: "medium", urlContext: true } },
  // Yapılandırılmış çıktı: sadece JSON döner
  json: { opts: { thinkingLevel: "low", json: true } },
};

const optionShape = {
  model: z.string().optional().describe("flash (default) | lite | gemma | raw model ID"),
  profile: z
    .enum(["fast", "deep", "code", "web", "json"])
    .optional()
    .describe(
      "fast: quick/short. deep: full reasoning. code: Python execution. web: reads URLs in the prompt (Google Search is opt-in via google_search=true). json: JSON-only output."
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
type Media = { mime: string; data: string };

function buildBody(model: string, prompt: string, o: GenOpts, strip: boolean, media?: Media): any {
  const gemma = isGemma(model);
  // Gemma system instruction desteklemiyor → prompt'un başına ekle
  const promptText = gemma && o.system ? `${o.system}\n\n${prompt}` : prompt;
  const parts: any[] = [];
  if (media) parts.push({ inlineData: { mimeType: media.mime, data: media.data } });
  parts.push({ text: promptText });
  const body: any = { contents: [{ parts }] };
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

async function generate(chain: string[], prompt: string, o: GenOpts, media?: Media): Promise<GenResult> {
  const key = getKey();
  if (!key) return { ok: false, error: "GEMINI_API_KEY secret olarak eklenmemiş." };

  const skipped: string[] = [];
  const errors: string[] = [];
  const notes: string[] = [];

  for (const model of chain) {
    let res: Response;
    try {
      res = await post(model, key, buildBody(model, prompt, o, false, media));

      // Model thinking/search ayarını reddettiyse (400/403) ya da search kotası free tier'da 0 ise (429):
      // ayarsız bir kez daha dene. Retry'ın cevabı esas alınır (ör. retry 429 ise sıradaki modele geçilir).
      const stripCandidate =
        !res.ok &&
        ((o.thinkingLevel && (res.status === 400 || res.status === 403)) ||
          (o.googleSearch && (res.status === 400 || res.status === 403 || res.status === 429)));
      if (stripCandidate) {
        const retry = await post(model, key, buildBody(model, prompt, o, true, media));
        if (retry.ok) notes.push(`${model}: thinking/search ayarı çıkarılıp çalıştırıldı`);
        res = retry;
      }
    } catch (err: any) {
      skipped.push(model);
      errors.push(`${model} → network: ${err?.message ?? String(err)}`);
      continue;
    }

    if (res.ok) {
      const data: any = await res.json();
      const outParts: any[] = data?.candidates?.[0]?.content?.parts ?? [];
      let answer = outParts.map((p: any) => p.text ?? "").join("");
      // code execution çıktısı ayrı part olarak gelir — görünür kıl
      for (const p of outParts) {
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

// ── Yardımcılar: GitHub dosyası, medya, cosine ────────────────
async function fetchGithubFile(sourceUrl: string): Promise<{ ok: true; content: string } | { ok: false; error: string }> {
  if (!workerEnv?.GITHUB_TOKEN) return { ok: false, error: "GITHUB_TOKEN secret olarak eklenmemiş." };
  // GÜVENLİK: GITHUB_TOKEN sadece GitHub'a gider. Başka host'a asla gönderme.
  const fetchUrl = toGithubApiUrl(sourceUrl) ?? sourceUrl;
  let host = "";
  try {
    host = new URL(fetchUrl).hostname;
  } catch {
    return { ok: false, error: "Geçersiz URL." };
  }
  if (host !== "api.github.com") {
    return { ok: false, error: "Sadece raw.githubusercontent.com veya api.github.com linkleri desteklenir." };
  }
  const res = await fetch(fetchUrl, {
    headers: {
      Authorization: `Bearer ${workerEnv.GITHUB_TOKEN}`,
      Accept: "application/vnd.github.raw+json",
      "User-Agent": "ai-router-worker",
    },
  });
  if (!res.ok) return { ok: false, error: `Dosya çekilemedi: ${res.status}` };
  return { ok: true, content: await res.text() };
}

const MAX_MEDIA_BYTES = 3 * 1024 * 1024; // Worker CPU limiti yüzünden küçük tut

// Herkese açık https linkinden medya çek (token EKLENMEZ). Private/IP hedeflerini reddet.
async function fetchMedia(
  url: string,
  mimeOverride?: string
): Promise<{ ok: true; media: Media } | { ok: false; error: string }> {
  let u: URL;
  try {
    u = new URL(url);
  } catch {
    return { ok: false, error: "Geçersiz URL." };
  }
  const h = u.hostname.toLowerCase();
  const isIpLiteral = /^[0-9.]+$/.test(h) || h.includes(":") || h === "localhost" || !h.includes(".");
  if (u.protocol !== "https:" || isIpLiteral) {
    return { ok: false, error: "Sadece herkese açık https alan adı linkleri desteklenir." };
  }
  const res = await fetch(u.toString(), { headers: { "User-Agent": "ai-router-worker" } });
  if (!res.ok) return { ok: false, error: `Medya çekilemedi: ${res.status}` };
  const declared = Number(res.headers.get("content-length") ?? 0);
  if (declared > MAX_MEDIA_BYTES) {
    return { ok: false, error: `Dosya çok büyük (${Math.round(declared / 1024)}KB, limit ${MAX_MEDIA_BYTES / 1024 / 1024}MB).` };
  }
  const buf = new Uint8Array(await res.arrayBuffer());
  if (buf.byteLength > MAX_MEDIA_BYTES) {
    return { ok: false, error: `Dosya çok büyük (limit ${MAX_MEDIA_BYTES / 1024 / 1024}MB).` };
  }
  let bin = "";
  const chunk = 0x8000;
  for (let i = 0; i < buf.length; i += chunk) {
    bin += String.fromCharCode.apply(null, Array.from(buf.subarray(i, i + chunk)));
  }
  let mime = (mimeOverride ?? res.headers.get("content-type") ?? "application/octet-stream").split(";")[0].trim().toLowerCase();
  // Bazı sunucular ses dosyalarını genel/eski MIME ile gönderiyor; Gemini bunları reddediyor
  const MIME_FIX: Record<string, string> = {
    "application/ogg": "audio/ogg",
    "application/x-ogg": "audio/ogg",
    "audio/x-wav": "audio/wav",
    "audio/wave": "audio/wav",
    "audio/x-mpeg": "audio/mpeg",
    "audio/mp3": "audio/mpeg",
    "audio/x-m4a": "audio/mp4",
    "audio/m4a": "audio/mp4",
  };
  mime = MIME_FIX[mime] ?? mime;
  return { ok: true, media: { mime, data: btoa(bin) } };
}

function cosine(a: number[], b: number[]): number {
  let dot = 0, na = 0, nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return dot / (Math.sqrt(na) * Math.sqrt(nb) || 1);
}

async function embedOne(key: string, input: string): Promise<{ model: string; values: number[] } | { error: string }> {
  let lastError = "";
  for (const model of EMBED_MODELS) {
    const res = await fetch(`${GEMINI_BASE}/models/${model}:embedContent`, {
      method: "POST",
      headers: { "Content-Type": "application/json", "x-goog-api-key": key },
      body: JSON.stringify({ content: { parts: [{ text: input }] } }),
    });
    if (res.ok) {
      const data: any = await res.json();
      return { model, values: data?.embedding?.values ?? [] };
    }
    lastError = `${model} → ${res.status}: ${(await res.text()).slice(0, 200)}`;
    if (!RETRY_STATUSES.has(res.status)) break;
  }
  return { error: lastError };
}

// Interactions API cevabından düz metni topla (şema bilinmediği için savunmacı)
function collectText(node: any, out: string[] = []): string[] {
  if (!node || typeof node !== "object") return out;
  if (Array.isArray(node)) {
    node.forEach((n) => collectText(n, out));
    return out;
  }
  for (const [k, v] of Object.entries(node)) {
    if (k === "text" && typeof v === "string") out.push(v);
    else if (typeof v === "object") collectText(v, out);
  }
  return out;
}

const TRANSCRIBE_CHAIN = [
  "gemini-3.5-transcribe", // 25/gün
  "gemini-3.8-flash",
  "gemini-3.7-flash",
  "gemini-3.5-flash",
  "gemini-3.1-flash-lite",
];

const AGENT_MODEL = "antigravity-preview-09-2026"; // free: 100/gün

// ── MCP server ────────────────────────────────────────────────
function createServer() {
  const server = new McpServer({ name: "ai-router", version: "2.3.1" });

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
        const file = await fetchGithubFile(args.source_url);
        if (!file.ok) return text(file.error);
        const fileContent = file.content;

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

  server.registerTool(
    "ask_media",
    {
      description:
        "Analyze an image, audio, PDF or short video from a PUBLIC https URL (max 3MB) with Gemini (flash chain). Good for screenshots (HWiNFO, error dialogs), photos, scanned docs. Same model/profile/settings options as ask_text.",
      inputSchema: z.object({
        media_url: z.string().describe("Public https URL of the file"),
        task: z.string().describe("What to do with it, e.g. 'bu ekran görüntüsünde hata ne?'"),
        mime_type: z.string().optional().describe("Override if the server sends a wrong content-type"),
        ...optionShape,
      }),
    },
    async (args: any) => {
      try {
        const m = await fetchMedia(args.media_url, args.mime_type);
        if (!m.ok) return text(m.error);
        const { chain, opts, profile } = resolveRequest(args);
        const r = await generate(chain, args.task, opts, m.media);
        return text(formatResult(r, profile));
      } catch (err: any) {
        return text(`Beklenmedik hata: ${err?.message ?? String(err)}`);
      }
    }
  );

  server.registerTool(
    "transcribe",
    {
      description:
        "Speech-to-text from a PUBLIC https audio URL (max 3MB, e.g. mp3/wav/m4a/ogg). Uses gemini-3.5-transcribe (25/day), falls back to flash models.",
      inputSchema: z.object({
        audio_url: z.string().describe("Public https URL of the audio file"),
        language: z.string().optional().describe("Hint, e.g. 'Turkish'"),
        mime_type: z.string().optional(),
      }),
    },
    async (args: any) => {
      try {
        const m = await fetchMedia(args.audio_url, args.mime_type);
        if (!m.ok) return text(m.error);
        const hint = args.language ? ` The language is ${args.language}.` : "";
        const prompt = `Transcribe this audio verbatim.${hint} Output only the transcript, no commentary.`;
        const r = await generate(TRANSCRIBE_CHAIN, prompt, {}, m.media);
        return text(formatResult(r));
      } catch (err: any) {
        return text(`Beklenmedik hata: ${err?.message ?? String(err)}`);
      }
    }
  );

  server.registerTool(
    "count_tokens",
    {
      description:
        "Count tokens for a text or a GitHub file BEFORE sending it. Use to check if it fits Gemma (16K), the 250K tokens/min free limit, etc.",
      inputSchema: z.object({
        text: z.string().optional(),
        source_url: z.string().optional().describe("raw.githubusercontent.com / api.github.com file"),
        model: z.string().optional().describe("alias or model ID (default: flash chain's first)"),
      }),
    },
    async (args: any) => {
      try {
        const key = getKey();
        if (!key) return text("GEMINI_API_KEY secret olarak eklenmemiş.");
        let content: string = args.text ?? "";
        if (!content && args.source_url) {
          const f = await fetchGithubFile(args.source_url);
          if (!f.ok) return text(f.error);
          content = f.content;
        }
        if (!content) return text("text veya source_url ver.");
        const model = resolveChain(args.model)[0];
        const res = await fetch(`${GEMINI_BASE}/models/${model}:countTokens`, {
          method: "POST",
          headers: { "Content-Type": "application/json", "x-goog-api-key": key },
          body: JSON.stringify({ contents: [{ parts: [{ text: content }] }] }),
        });
        if (!res.ok) return text(`countTokens hata: ${res.status} — ${(await res.text()).slice(0, 300)}`);
        const data: any = await res.json();
        const n = data?.totalTokens ?? 0;
        const fits = `Gemma 16K: ${n <= 16000 ? "sığar" : "SIĞMAZ"} | flash 250K/dk: ${n <= 250000 ? "sığar" : "SIĞMAZ"}`;
        return text(`${n} token (${content.length} karakter) — model: ${model}\n${fits}`);
      } catch (err: any) {
        return text(`Beklenmedik hata: ${err?.message ?? String(err)}`);
      }
    }
  );

  server.registerTool(
    "embed_rank",
    {
      description:
        "Semantic search: rank up to 20 candidate texts by similarity to a query using Gemini Embedding (cosine). Free: 100/min, 1K/day — each candidate costs 1 request.",
      inputSchema: z.object({
        query: z.string(),
        candidates: z.array(z.string()).min(1).max(20),
        top_k: z.number().int().positive().optional(),
      }),
    },
    async (args: any) => {
      try {
        const key = getKey();
        if (!key) return text("GEMINI_API_KEY secret olarak eklenmemiş.");
        const all = await Promise.all([args.query, ...args.candidates].map((t: string) => embedOne(key, t)));
        const q = all[0];
        if ("error" in q) return text(`Embed hata: ${q.error}`);
        const scored: { i: number; score: number }[] = [];
        for (let i = 1; i < all.length; i++) {
          const e = all[i];
          if ("error" in e) return text(`Embed hata (aday ${i}): ${e.error}`);
          scored.push({ i: i - 1, score: cosine(q.values, e.values) });
        }
        scored.sort((a, b) => b.score - a.score);
        const top = scored.slice(0, args.top_k ?? scored.length);
        const lines = top.map(
          (s, rank) => `${rank + 1}. [${s.score.toFixed(3)}] (#${s.i}) ${args.candidates[s.i].slice(0, 120)}`
        );
        return text(`${lines.join("\n")}\n\n— model: ${q.model}`);
      } catch (err: any) {
        return text(`Beklenmedik hata: ${err?.message ?? String(err)}`);
      }
    }
  );

  server.registerTool(
    "run_agent",
    {
      description:
        "EXPERIMENTAL. Run Google's Antigravity managed agent (Linux sandbox: writes/runs code, files, web). Free: 100/day, so use for real multi-step tasks only. Returns the agent's final answer + interaction_id/environment_id to continue the same session. Can take minutes.",
      inputSchema: z.object({
        task: z.string(),
        previous_interaction_id: z.string().optional().describe("Continue a conversation"),
        environment_id: z.string().optional().describe("Reuse the same sandbox (files persist)"),
      }),
    },
    async (args: any) => {
      try {
        const key = getKey();
        if (!key) return text("GEMINI_API_KEY secret olarak eklenmemiş.");
        const body: any = {
          agent: AGENT_MODEL,
          input: [{ type: "text", text: args.task }],
          environment: args.environment_id ?? { type: "remote" },
        };
        if (args.previous_interaction_id) body.previous_interaction_id = args.previous_interaction_id;
        const res = await fetch(`${GEMINI_BASE}/interactions`, {
          method: "POST",
          headers: { "Content-Type": "application/json", "x-goog-api-key": key },
          body: JSON.stringify(body),
        });
        const raw = await res.text();
        if (!res.ok) return text(`Agent hata: ${res.status} — ${raw.slice(0, 400)}`);
        let data: any;
        try {
          data = JSON.parse(raw);
        } catch {
          return text(`Agent beklenmedik cevap: ${raw.slice(0, 400)}`);
        }
        let out: string = data.output_text ?? "";
        if (!out && Array.isArray(data.steps)) {
          for (let i = data.steps.length - 1; i >= 0 && !out; i--) out = collectText(data.steps[i]).join("\n");
        }
        if (!out) out = `(düz metin bulunamadı) ${raw.slice(0, 2000)}`;
        if (out.length > 20000) out = out.slice(0, 20000) + "\n…(kesildi)";
        return text(`${out}\n\n— agent: ${AGENT_MODEL} | interaction_id: ${data.id ?? "?"} | environment_id: ${data.environment_id ?? "?"}`);
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
