import { NodeFileSystem } from "@effect/platform-node"
import { HttpRecorder } from "@opencode-ai/http-recorder"
import { describe, expect, test } from "bun:test"
import { tool } from "ai"
import { Effect, Layer, Stream } from "effect"
import { FetchHttpClient } from "effect/unstable/http"
import path from "node:path"
import z from "zod"
import { Auth } from "@/auth"
import { Config } from "@/config/config"
import { AppRuntime } from "@/effect/app-runtime"
import { attach } from "@/effect/run-service"
import { Plugin } from "@/plugin"
import { Provider } from "@/provider/provider"
import { ModelID, ProviderID } from "@/provider/schema"
import { Filesystem } from "@/util/filesystem"
import { LLMClient, RequestExecutor } from "@opencode-ai/llm/route"
import type { Agent } from "../../src/agent/agent"
import { WithInstance } from "../../src/project/with-instance"
import { LLM } from "../../src/session/llm"
import { MessageV2 } from "../../src/session/message-v2"
import { MessageID, SessionID } from "../../src/session/schema"
import type { ModelsDev } from "../../src/provider/models"
import { tmpdir } from "../fixture/fixture"

const CASSETTE = "session/native-openai-tool-call"
const FIXTURES_DIR = path.join(import.meta.dir, "../fixtures/recordings")
const OPENAI_API_KEY = process.env.OPENCODE_RECORD_OPENAI_API_KEY ?? process.env.OPENAI_API_KEY

const shouldRecord = process.env.RECORD === "true"
const canRun = shouldRecord
  ? Boolean(OPENAI_API_KEY)
  : HttpRecorder.hasCassetteSync(CASSETTE, { directory: FIXTURES_DIR })
const recordedTest = canRun ? test : test.skip

async function loadFixture(providerID: string, modelID: string) {
  const data = await Filesystem.readJson<Record<string, ModelsDev.Provider>>(
    path.join(import.meta.dir, "../tool/fixtures/models-api.json"),
  )
  const provider = data[providerID]
  if (!provider) throw new Error(`Missing provider in fixture: ${providerID}`)
  const model = provider.models[modelID]
  if (!model) throw new Error(`Missing model in fixture: ${modelID}`)
  return model
}

async function getModel(providerID: ProviderID, modelID: ModelID) {
  return AppRuntime.runPromise(
    Effect.gen(function* () {
      const provider = yield* Provider.Service
      return yield* provider.getModel(providerID, modelID)
    }),
  )
}

const openAIConfig = (model: ModelsDev.Provider["models"][string]): Partial<Config.Info> => ({
  enabled_providers: ["openai"],
  provider: {
    openai: {
      name: "OpenAI",
      env: ["OPENAI_API_KEY"],
      npm: "@ai-sdk/openai",
      api: "https://api.openai.com/v1",
      models: {
        [model.id]: JSON.parse(JSON.stringify(model)) as NonNullable<
          NonNullable<Config.Info["provider"]>[string]["models"]
        >[string],
      },
      options: {
        apiKey: OPENAI_API_KEY ?? "fixture-openai-key",
        baseURL: "https://api.openai.com/v1",
      },
    },
  },
})

function recordedLLMLayer() {
  const cassetteService = HttpRecorder.Cassette.fileSystem({ directory: FIXTURES_DIR }).pipe(
    Layer.provide(NodeFileSystem.layer),
  )
  const recorder = HttpRecorder.recordingLayer(CASSETTE, {
    mode: shouldRecord ? "record" : "replay",
    metadata: {
      provider: "openai",
      protocol: "openai-responses",
      route: "openai-responses",
      tags: ["opencode", "native", "tool-call"],
    },
  }).pipe(Layer.provide(FetchHttpClient.layer))
  const executor = RequestExecutor.layer.pipe(Layer.provide(recorder))
  const client = LLMClient.layer.pipe(Layer.provide(executor))

  return LLM.layer.pipe(
    Layer.provide(Auth.defaultLayer),
    Layer.provide(Config.defaultLayer),
    Layer.provide(Provider.defaultLayer),
    Layer.provide(Plugin.defaultLayer),
    Layer.provide(client),
    Layer.provide(cassetteService),
  )
}

async function collect(layer: Layer.Layer<LLM.Service>, input: LLM.StreamInput) {
  return Array.from(
    await Effect.runPromise(
      attach(LLM.Service.use((svc) => svc.stream(input).pipe(Stream.runCollect))).pipe(Effect.provide(layer)),
    ),
  )
}

describe("session.llm native recorded", () => {
  recordedTest("uses real RequestExecutor with HTTP recorder for native OpenAI tools", async () => {
    const model = await loadFixture("openai", "gpt-4.1-mini")
    let executed: unknown

    await using tmp = await tmpdir({ config: openAIConfig(model) })

    await WithInstance.provide({
      directory: tmp.path,
      fn: async () => {
        const previous = process.env.OPENCODE_LLM_RUNTIME
        process.env.OPENCODE_LLM_RUNTIME = "native"
        try {
          const sessionID = SessionID.make("session-recorded-native-tool")
          const agent = {
            name: "test",
            mode: "primary",
            prompt: "Call tools exactly as instructed.",
            options: {},
            permission: [{ permission: "*", pattern: "*", action: "allow" }],
            temperature: 0,
          } satisfies Agent.Info

          const resolved = await getModel(ProviderID.openai, ModelID.make(model.id))
          const events = await collect(recordedLLMLayer(), {
            user: {
              id: MessageID.make("msg_user-recorded-native-tool"),
              sessionID,
              role: "user",
              time: { created: 0 },
              agent: agent.name,
              model: { providerID: ProviderID.make("openai"), modelID: ModelID.make(model.id) },
            } satisfies MessageV2.User,
            sessionID,
            model: resolved,
            agent,
            system: ["You must call the lookup tool exactly once with query weather. Do not answer in text."],
            messages: [{ role: "user", content: "Use lookup." }],
            toolChoice: "required",
            tools: {
              lookup: tool({
                description: "Lookup data.",
                inputSchema: z.object({ query: z.string() }),
                execute: async (args, options) => {
                  executed = { args, toolCallId: options.toolCallId }
                  return { output: "looked up" }
                },
              }),
            },
          })

          expect(events.filter((event) => event.type === "step-finish")).toHaveLength(1)
          expect(events.filter((event) => event.type === "finish")).toHaveLength(1)
          expect(events.some((event) => event.type === "tool-result")).toBe(true)
          expect(executed).toMatchObject({ args: { query: "weather" }, toolCallId: expect.any(String) })
        } finally {
          if (previous === undefined) delete process.env.OPENCODE_LLM_RUNTIME
          else process.env.OPENCODE_LLM_RUNTIME = previous
        }
      },
    })
  })
})
