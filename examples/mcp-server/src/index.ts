import { McpServer, createMcpHandler } from "@modelcontextprotocol/server";
import { z } from "zod";

function createServer(req, env) {
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
    async (input, ...rest) => {
      const ctx = rest[0] || {};
      return {
        content: [{
          type: "text",
          text: `http keys: ${ctx.http ? JSON.stringify(Object.keys(ctx.http)) : "yok"} | mcpReq keys: ${ctx.mcpReq ? JSON.stringify(Object.keys(ctx.mcpReq)) : "yok"}`
        }]
      };
    }
  );

  return server;
}

export default createMcpHandler(createServer);
