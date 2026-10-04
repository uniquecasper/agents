// ============================================================
// AI Router — index.ts (genişletilmiş sürüm)
// ⚠️ DOSYANIN EN ÜSTÜNDEKİ import satırlarını (createMcpHandler, McpServer, z)
//    eski dosyadan AYNEN KORU. Bu dosya sadece "import'lardan sonrası" içindir.
//    Eski kodda zod'un adı `z` değilse aşağıdaki `z.` kullanımlarını ona göre düzelt.
// ============================================================

type Env = {
  GEMINI_API_KEY?: string;
  GITHUB_TOKEN?: string;
};

let workerEnv: Env | undefined;

const GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta";

// ---------- Model takma adları ----------
// Her alias bir ZİNCİR: ilki denenir, kota (429) dolarsa sıradakine düşer.
// ⚠️ Model ID'leri tablodaki isimlerden tahmin — emin değilsek `list_models` ile doğrula.
// Yanlış ID gelirse 404 döner, hata mesajında söylenir; buradan düzeltmen yeter.
const TEXT_ALIASES: Record<string, string[]> = {
  flash: ["gemini-3.8-flash", "gemini-3.5-flash-lite"], // varsayılan: kalite → kota bitince lite
  lite: ["gemini-3.5-flash-lite"], // günde 500, hafif/toplu işler
  gemma: ["gemma-4-31b-it"], // günde 14.4K ama context 16K — büyük dosya SIĞMAZ
};
const DEFAULT_TEXT = "flash";
const EMBED_MODEL = "gemini-embedding-2";

// ---------- Yardımcılar ----------
const text = (t: string) => ({ content: [{ type: "text" as const, text: t }] });

function getKey(): string | null {
  return workerEnv?.GEMINI_API_KEY ?? null;
}

function resolveChain(modelInput?: string): string[] {
  const m = (modelInput ?? DEFAULT_TEXT).trim();
  // Alias ise zinciri, değilse ham model ID'sini tek elemanlı zincir yap
  return TEXT_ALIASES[m] ?? [m];
}

function toGithubApiUrl(rawUrl: string): string | null {
  const m = rawUrl.match(
    /^https:\/\/raw\.githubusercontent\.com\/([^/]+)\/([^/]+)\/([^/]+)\/(.+)$/,
  );
  if (!m) return null;
  const [, owner, repo, branch, path] = m;
  return `https://api.github.com/repos/${owner}/${repo}/contents/${path}?ref=${branch}`;
}

type GenResult = { ok: true; answer: string; usedModel: string } | { ok: false; error: string };

// Zincirdeki modelleri sırayla dener; sadece 429 (kota) ve 503 (yoğunluk) durumunda bir sonrakine geçer.
async function generate(chain: string[], prompt: string, thinking: boolean): Promise<GenResult> {
  const key = getKey();
  if (!key) return { ok: false, error: "GEMINI_API_KEY secret olarak eklenmemiş." };

  const body: Record<string, unknown> = { contents: [{ parts: [{ text: prompt }] }] };
  if (thinking) body.generationConfig = { thinkingConfig: { thinkingLevel: "high" } };

  let lastError = "";
  for (const model of chain) {
    const res = await fetch(`${GEMINI_BASE}/models/${model}:generateContent`, {
      method: "POST",
      headers: { "Content-Type": "application/json", "x-goog-api-key": key },
      body: JSON.stringify(body),
    });

    if (res.ok) {
      const data: any = await res.json();
      const answer = data?.candidates?.[0]?.content?.parts?.map((p: any) => p.text ?? "").join("") ?? "";
      return { ok: true, answer: answer || "Gemini boş cevap döndü.", usedModel: model };
    }

    const errText = await res.text();
    lastError = `${model} → ${res.status}: ${errText.slice(0, 300)}`;
    if (res.status !== 429 && res.status !== 503) break; // kota/yoğunluk dışı hata: fallback anlamsız
  }
  return { ok: false, error: lastError };
}

// ---------- MCP server ----------
function createServer() {
  const server = new McpServer({ name: "ai-router", version: "2.0.0" });

  // 1) hello — bağlantı testi
  server.registerTool(
    "hello",
    { description: "Returns a greeting", inputSchema: z.object({ name: z.string().optional() }) },
    async ({ name }: { name?: string }) => text(`Hello, ${name ?? "World"}!`),
  );

  // 2) list_models — key'in gerçekten erişebildiği modeller (ID doğrulama için)
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
      const res = await fetch(`${GEMINI_BASE}/models?pageSize=200`, { headers: { "x-goog-api-key": key } });
      if (!res.ok) return text(`list_models hata: ${res.status} — ${(await res.text()).slice(0, 300)}`);
      const data: any = await res.json();
      const rows = (data.models ?? [])
        .map((m: any) => `${String(m.name).replace("models/", "")} [${(m.supportedGenerationMethods ?? []).join(",")}]`)
        .filter((r: string) => !filter || r.toLowerCase().includes(filter.toLowerCase()));
      return text(rows.length ? rows.join("\n") : "Eşleşen model yok.");
    },
  );

  // 3) ask_text — düz metin sorusu, model seçilebilir
  server.registerTool(
    "ask_text",
    {
      description:
        "Ask a Gemini/Gemma text model a question. model: 'flash' (default, falls back to lite when quota is out), 'lite' (500/day), 'gemma' (16K context only), or a raw model ID.",
      inputSchema: z.object({
        prompt: z.string().describe("The question/task"),
        model: z.string().optional(),
        extended_thinking: z.boolean().optional().describe("Deeper reasoning (slower). Default false."),
      }),
    },
    async ({ prompt, model, extended_thinking }: { prompt: string; model?: string; extended_thinking?: boolean }) => {
      try {
        const r = await generate(resolveChain(model), prompt, !!extended_thinking);
        return text(r.ok ? `${r.answer}\n\n— model: ${r.usedModel}` : `Gemini hata: ${r.error}`);
      } catch (err: any) {
        return text(`Beklenmedik hata: ${err?.message ?? String(err)}`);
      }
    },
  );

  // 4) ask_ai — eski tool, ADI KORUNDU (proje notları/promptlar buna referans veriyor)
  server.registerTool(
    "ask_ai",
    {
      description:
        "Fetches a file from a public or private (via token) GitHub URL and asks Gemini to analyze it, returning only the answer. model: 'flash' (default), 'lite', or raw ID. (gemma önerilmez: 16K context.)",
      inputSchema: z.object({
        source_url: z.string().describe("URL of the file (raw.githubusercontent.com link works for private repos too)"),
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
        const fetchUrl = toGithubApiUrl(source_url) ?? source_url;
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
        return text(r.ok ? `${r.answer}\n\n— model: ${r.usedModel}` : `Gemini hata: ${r.error}`);
      } catch (err: any) {
        return text(`Beklenmedik hata: ${err?.message ?? String(err)}`);
      }
    },
  );

  // 5) embed — metin → vektör. Tam vektör Claude'un context'ini şişirir, o yüzden varsayılan: özet.
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
        const res = await fetch(`${GEMINI_BASE}/models/${EMBED_MODEL}:embedContent`, {
          method: "POST",
          headers: { "Content-Type": "application/json", "x-goog-api-key": key },
          body: JSON.stringify({ content: { parts: [{ text: input }] } }),
        });
        if (!res.ok) return text(`Embed hata: ${res.status} — ${(await res.text()).slice(0, 300)}`);
        const data: any = await res.json();
        const v: number[] = data?.embedding?.values ?? [];
        return text(
          full
            ? JSON.stringify(v)
            : `dim=${v.length}\nilk 8 değer: ${JSON.stringify(v.slice(0, 8))}\n(tamamı için full=true)`,
        );
      } catch (err: any) {
        return text(`Beklenmedik hata: ${err?.message ?? String(err)}`);
      }
    },
  );

  return server;
}

// ---------- Env yakalama (eski koddaki Proxy mantığı AYNEN) ----------
function captureEnv(fn: Function, boundTo: unknown) {
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
    
