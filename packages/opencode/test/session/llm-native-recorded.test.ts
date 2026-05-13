import { NodeFileSystem } from "@effect/platform-node"
import { HttpRecorder } from "@opencode-ai/http-recorder"
import { describe, expect } from "bun:test"
import { tool } from "ai"
import { Effect, Layer, Stream } from "effect"
import { FetchHttpClient } from "effect/unstable/http"
import path from "node:path"
import z from "zod"
import { Auth } from "@/auth"
import { Config } from "@/config/config"
import { Plugin } from "@/plugin"
import { Provider } from "@/provider/provider"
import { ModelID, ProviderID } from "@/provider/schema"
import { Filesystem } from "@/util/filesystem"
import { LLMClient, RequestExecutor } from "@opencode-ai/llm/route"
import type { Agent } from "../../src/agent/agent"
import { LLM } from "../../src/session/llm"
import { MessageV2 } from "../../src/session/message-v2"
import { MessageID, SessionID } from "../../src/session/schema"
import type { ModelsDev } from "../../src/provider/models"
import { TestInstance } from "../fixture/fixture"
import { testEffect } from "../lib/effect"

const CASSETTE = "session/native-openai-tool-call"
const FIXTURES_DIR = path.join(import.meta.dir, "../fixtures/recordings")
const OPENAI_API_KEY = process.env.OPENCODE_RECORD_OPENAI_API_KEY ?? process.env.OPENAI_API_KEY

const shouldRecord = process.env.RECORD === "true"
const canRun = shouldRecord
  ? Boolean(OPENAI_API_KEY)
  : HttpRecorder.hasCassetteSync(CASSETTE, { directory: FIXTURES_DIR })

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

function recordedNativeLLMLayer() {
  const cassetteService = HttpRecorder.Cassette.fileSystem({ directory: FIXTURES_DIR }).pipe(
    Layer.provide(NodeFileSystem.layer),
  )
  // Only the HTTP client is recorded; RequestExecutor and the opencode LLM stack remain real.
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

  const providerLayer = Provider.defaultLayer.pipe(
    Layer.provide(Auth.defaultLayer),
    Layer.provide(Config.defaultLayer),
    Layer.provide(Plugin.defaultLayer),
  )
  const llmLayer = LLM.layer.pipe(
    Layer.provide(Auth.defaultLayer),
    Layer.provide(Config.defaultLayer),
    Layer.provide(Provider.defaultLayer),
    Layer.provide(Plugin.defaultLayer),
    Layer.provide(client),
    Layer.provide(cassetteService),
  )

  return Layer.mergeAll(providerLayer, llmLayer)
}

const it = testEffect(recordedNativeLLMLayer())
const recordedInstance = canRun ? it.instance : it.instance.skip

const writeConfig = (directory: string, model: ModelsDev.Provider["models"][string]) =>
  Effect.promise(() =>
    Bun.write(
      path.join(directory, "opencode.json"),
      JSON.stringify({ $schema: "https://opencode.ai/config.json", ...openAIConfig(model) }),
    ),
  )

const getModel = (providerID: ProviderID, modelID: ModelID) =>
  Effect.gen(function* () {
    const provider = yield* Provider.Service
    return yield* provider.getModel(providerID, modelID)
  })

const collect = (input: LLM.StreamInput) =>
  Effect.gen(function* () {
    const llm = yield* LLM.Service
    return Array.from(yield* llm.stream(input).pipe(Stream.runCollect))
  })

const nativeRuntime = <A, E, R>(effect: Effect.Effect<A, E, R>) => {
  return Effect.acquireUseRelease(
    Effect.sync(() => {
      const previous = process.env.OPENCODE_LLM_RUNTIME
      process.env.OPENCODE_LLM_RUNTIME = "native"
      return previous
    }),
    () => effect,
    (previous) =>
      Effect.sync(() => {
        if (previous === undefined) delete process.env.OPENCODE_LLM_RUNTIME
        else process.env.OPENCODE_LLM_RUNTIME = previous
      }),
  )
}

describe("session.llm native recorded", () => {
  recordedInstance("uses real RequestExecutor with HTTP recorder for native OpenAI tools", () =>
    Effect.gen(function* () {
      const test = yield* TestInstance
      const model = yield* Effect.promise(() => loadFixture("openai", "gpt-4.1-mini"))
      yield* writeConfig(test.directory, model)

      const sessionID = SessionID.make("session-recorded-native-tool")
      const agent = {
        name: "test",
        mode: "primary",
        prompt: "Call tools exactly as instructed.",
        options: {},
        permission: [{ permission: "*", pattern: "*", action: "allow" }],
        temperature: 0,
      } satisfies Agent.Info
      const resolved = yield* getModel(ProviderID.openai, ModelID.make(model.id))
      let executed: unknown

      const events = yield* nativeRuntime(
        collect({
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
        }),
      )

      expect(events.filter((event) => event.type === "step-finish")).toHaveLength(1)
      expect(events.filter((event) => event.type === "finish")).toHaveLength(1)
      expect(events.some((event) => event.type === "tool-result")).toBe(true)
      expect(executed).toMatchObject({ args: { query: "weather" }, toolCallId: expect.any(String) })
    }),
  )
})
