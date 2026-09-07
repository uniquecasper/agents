server.registerTool(
    "ask_ai",
    {
      description: "Fetches a file from a public or private (via token) URL and asks Gemini to analyze it, returning only Gemini's answer.",
      inputSchema: z.object({
        source_url: z.string().describe("Public URL of the file (e.g. raw.githubusercontent.com link)"),
        task: z.string().describe("What to do with the file, e.g. 'find bugs'")
      })
    },
    async ({ source_url, task }, { env }) => {
      const fileRes = await fetch(source_url, {
        headers: {
          Authorization: `Bearer ${env.GITHUB_TOKEN}`
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
