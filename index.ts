import { ollamaKeepAlive } from "../../services/ai/ollama-runtime";

/**
 * Universal AI client for Bun.
 *
 * NO AI SDKs.
 * NO dotenv.
 * NO openai package.
 * NO ollama package.
 * NO @google/genai.
 * NO @anthropic-ai/sdk.
 * NO cohere-ai.
 * NO @huggingface/inference.
 *
 * Everything goes through native fetch().
 *
 * Bun automatically loads .env files.
 *
 * ------------------------------------------------------------
 * EXAMPLE
 * ------------------------------------------------------------
 *
 * const ai = new AIClient();
 * const response = await ai.chat({
 *   system_prompt: "You are a useful assistant.",
 *   user_prompt: "Hello",
 *   provider: "OpenAI",
 *   model: "gpt-5.6-sol",
 *
 *   options: {
 *     temperature: 0.2,
 *
 *     image: true,
 *     image_provider: "OpenAI",
 *     image_model: "gpt-image-2",
 *
 *     audio: true,
 *     audio_provider: "OpenAI",
 *     audio_model: "gpt-4o-mini-tts",
 *     audio_voice: "alloy",
 *
 *     video: false,
 *   },
 * });
 *
 * console.log(response.choices[0].message.content);
 * console.log(response.choices[0].message.image);
 * console.log(response.choices[0].message.audio);
 *
 * const price = await ai.getPrice(
 *   "OpenAI",
 *   "gpt-5.6-sol",
 *   "text",
 *   { prompt: "Hello", expectedOutputTokens: 500 },
 * );
 * console.log(price.providerCostUSD, price.sellPriceUSD);
 *
 * ------------------------------------------------------------
 */

export type AIProvider =
  | "OpenAI"
  | "Anthropic"
  | "Google"
  | "xAI"
  | "MistralAI"
  | "DeepSeek"
  | "OpenRouter"
  | "Fireworks"
  | "Together"
  | "Groq"
  | "Perplexity"
  | "Cohere"
  | "HuggingFace"
  | "Stability"
  | "Hermes"
  | "Ollama";

export type AIModality =
  | "text"
  | "vision"
  | "image"
  | "video"
  | "audio"
  | "tts"
  | "stt"
  | "embedding"
  | "rerank";

export type PricingModality = "text" | "image" | "video" | "tts" | "stt" | "audio";

export type ImagePriceTable = Record<string, number>;
export type VideoPriceTable = Record<string, number>;

export type ModelPrice = {
  currency: "USD";

  /** Standard text/chat rates, USD per 1M tokens. */
  inputPer1MTokens?: number;
  cachedInputPer1MTokens?: number;
  cacheWritePer1MTokens?: number;
  outputPer1MTokens?: number;

  /** Multimodal token rates, USD per 1M tokens. */
  textInputPer1MTokens?: number;
  imageInputPer1MTokens?: number;
  imageOutputPer1MTokens?: number;
  audioInputPer1MTokens?: number;
  audioOutputPer1MTokens?: number;

  /** Flat/rate-based media pricing. */
  perImageUSD?: number;
  perImageByVariantUSD?: ImagePriceTable;
  perSecondUSD?: number;
  perSecondByResolutionUSD?: VideoPriceTable;
  perMinuteUSD?: number;
  perHourUSD?: number;
  per1MCharactersUSD?: number;
  per1000CharactersUSD?: number;
  perRequestUSD?: number;
  per1000PagesUSD?: number;

  /** Optional alternate rates, e.g. DeepSeek off-peak. */
  offPeak?: Partial<ModelPrice>;

  /** Long-context surcharge starts after this many input tokens. */
  longContextAfterTokens?: number;
  longContextInputMultiplier?: number;
  longContextOutputMultiplier?: number;

  /** True only for genuinely zero provider inference cost, e.g. local Ollama. */
  free?: boolean;

  /** Price exists but depends on upstream provider/runtime and must be fetched dynamically. */
  dynamic?: boolean;

  source?: string;
  updatedAt?: string;
  notes?: string[];
};

export type PriceRequest = {
  /** Prompt can be passed here, or pass a string as the fourth getPrice() argument. */
  prompt?: string;
  systemPrompt?: string;

  /** Exact counts win over estimates. */
  inputTokens?: number;
  cachedInputTokens?: number;
  cacheWriteTokens?: number;
  outputTokens?: number;

  /** Used when outputTokens is unknown before text generation. */
  expectedOutputTokens?: number;

  /** Media quantities. */
  quantity?: number;
  seconds?: number;
  minutes?: number;
  hours?: number;
  characters?: number;
  pages?: number;

  /** Image/video variant selectors. */
  quality?: string;
  size?: string;
  resolution?: string;

  /** Optional token counts for token-billed image/audio endpoints. */
  imageInputTokens?: number;
  imageOutputTokens?: number;
  audioInputTokens?: number;
  audioOutputTokens?: number;

  /** Selling policy overrides for this quote. */
  targetMarginPercent?: number;
  safetyBufferPercent?: number;
  minimumProfitUSD?: number;
  minimumSellPriceUSD?: number;
};

export type PriceQuote = {
  provider: AIProvider;
  model: string;
  modality: PricingModality;
  currency: "USD";

  providerCostUSD: number;
  bufferedCostUSD: number;
  sellPriceUSD: number;
  profitUSD: number;
  grossMarginPercent: number;

  estimated: boolean;

  usage: {
    inputTokens?: number;
    cachedInputTokens?: number;
    cacheWriteTokens?: number;
    outputTokens?: number;
    quantity?: number;
    seconds?: number;
    minutes?: number;
    hours?: number;
    characters?: number;
    pages?: number;
    quality?: string;
    size?: string;
    resolution?: string;
  };

  rate: ModelPrice;
  assumptions: string[];
};

export type AIClientOptions = {
  pricing?: {
    /** Gross margin, not markup. 40% margin means sell = cost / 0.60. */
    targetMarginPercent?: number;
    /** Protects estimates against token/media variance. */
    safetyBufferPercent?: number;
    minimumProfitUSD?: number;
    minimumSellPriceUSD?: number;
    defaultExpectedOutputTokens?: number;
    roundUSDTo?: number;
  };
};

export type ModelInfo = {
  id: string;
  categories: AIModality[];
  description?: string;
  deprecated?: boolean;
  /** Provider cost. Never confuse this with your retail/sell price. */
  price?: ModelPrice;
};

export type ProviderInfo = {
  id: AIProvider;
  name: string;

  api: {
    baseURL: string;
    models?: string;
    chat?: string;
    image?: string;
    video?: string;
    tts?: string;
    stt?: string;
  };

  env: string[];

  categories: AIModality[];

  openAICompatible?: boolean;

  featuredModels: ModelInfo[];
};

/** Legacy-friendly aliases. */
export type TypeProvider = AIProvider;
export type TypeModels = string;

const PRICE_UPDATED_AT = "2026-10-07";

/**
 * Verified static list prices for the models we explicitly know.
 * Live catalogs can contain many more models; OpenRouter pricing is enriched dynamically.
 * Unknown pricing intentionally stays unknown: getPrice() fails instead of pretending it is free.
 */
export const MODEL_PRICING: Record<string, ModelPrice> = {
  // ---------------- OpenAI ----------------
  "OpenAI:gpt-6-astra": {
    currency: "USD",
    inputPer1MTokens: 10,
    cachedInputPer1MTokens: 1,
    cacheWritePer1MTokens: 12.5,
    outputPer1MTokens: 50,
    longContextAfterTokens: 272000,
    longContextInputMultiplier: 2,
    longContextOutputMultiplier: 1.5,
    updatedAt: PRICE_UPDATED_AT,
    source: "https://developers.openai.com/api/docs/models/gpt-6-astra",
  },
  "OpenAI:gpt-6.1-sol": {
    currency: "USD",
    inputPer1MTokens: 2,
    cachedInputPer1MTokens: 0.1,
    cacheWritePer1MTokens: 2.5,
    outputPer1MTokens: 10,
    longContextAfterTokens: 272000,
    longContextInputMultiplier: 2,
    longContextOutputMultiplier: 1.5,
    updatedAt: PRICE_UPDATED_AT,
    source: "https://developers.openai.com/api/docs/pricing",
  },
  "OpenAI:gpt-6-luna": {
    currency: "USD",
    inputPer1MTokens: 0.1,
    cachedInputPer1MTokens: 0.01,
    cacheWritePer1MTokens: 0.125,
    outputPer1MTokens: 0.5,
    longContextAfterTokens: 272000,
    longContextInputMultiplier: 2,
    longContextOutputMultiplier: 1.5,
    updatedAt: PRICE_UPDATED_AT,
    source: "https://developers.openai.com/api/docs/pricing",
  },
  "OpenAI:gpt-5.6-sol": {
    currency: "USD",
    inputPer1MTokens: 4,
    cachedInputPer1MTokens: 0.4,
    cacheWritePer1MTokens: 5,
    outputPer1MTokens: 20,
    longContextAfterTokens: 272000,
    longContextInputMultiplier: 2,
    longContextOutputMultiplier: 1.5,
    updatedAt: PRICE_UPDATED_AT,
    source: "https://developers.openai.com/api/docs/models/gpt-5.6-sol",
    notes: ["Promotional pricing is stated by OpenAI as available at least through 2026-11-21."],
  },
  "OpenAI:gpt-5.6-terra": {
    currency: "USD",
    inputPer1MTokens: 2,
    cachedInputPer1MTokens: 0.2,
    cacheWritePer1MTokens: 2.5,
    outputPer1MTokens: 12,
    longContextAfterTokens: 272000,
    longContextInputMultiplier: 2,
    longContextOutputMultiplier: 1.5,
    updatedAt: PRICE_UPDATED_AT,
    source: "https://developers.openai.com/api/docs/models/gpt-5.6-terra",
  },
  "OpenAI:gpt-5.6-luna": {
    currency: "USD",
    inputPer1MTokens: 0.2,
    cachedInputPer1MTokens: 0.02,
    cacheWritePer1MTokens: 0.25,
    outputPer1MTokens: 1.2,
    longContextAfterTokens: 272000,
    longContextInputMultiplier: 2,
    longContextOutputMultiplier: 1.5,
    updatedAt: PRICE_UPDATED_AT,
    source: "https://developers.openai.com/api/docs/models/gpt-5.6-luna",
  },
  "OpenAI:gpt-image-2": {
    currency: "USD",
    textInputPer1MTokens: 2.5,
    imageInputPer1MTokens: 4,
    imageOutputPer1MTokens: 15,
    updatedAt: PRICE_UPDATED_AT,
    perImageByVariantUSD: {
      "low:1024x1024": 0.006,
      "low:1024x1536": 0.005,
      "low:1536x1024": 0.005,
      "medium:1024x1024": 0.053,
      "medium:1024x1536": 0.041,
      "medium:1536x1024": 0.041,
      "high:1024x1024": 0.211,
      "high:1024x1536": 0.165,
      "high:1536x1024": 0.165,
    },
    source: "https://developers.openai.com/api/docs/guides/image-generation",
    notes: [
      "Per-image values are image output cost; prompt/image input tokens are added separately.",
    ],
  },
  "OpenAI:gpt-image-2.5-sunburst": {
    currency: "USD",
    textInputPer1MTokens: 5,
    imageInputPer1MTokens: 8,
    imageOutputPer1MTokens: 30,
    updatedAt: PRICE_UPDATED_AT,
    source: "https://developers.openai.com/api/docs/pricing",
    notes: ["Exact per-image output cost depends on generated output tokens, quality and size."],
  },
  "OpenAI:gpt-image-2.5-flare": {
    currency: "USD",
    textInputPer1MTokens: 5,
    imageInputPer1MTokens: 8,
    imageOutputPer1MTokens: 30,
    updatedAt: PRICE_UPDATED_AT,
    source: "https://developers.openai.com/api/docs/pricing",
    notes: ["Exact per-image output cost depends on generated output tokens, quality and size."],
  },
  "OpenAI:gpt-4o-mini-tts": {
    currency: "USD",
    inputPer1MTokens: 0.6,
    audioOutputPer1MTokens: 12,
    updatedAt: PRICE_UPDATED_AT,
    source: "https://developers.openai.com/api/docs/models/gpt-4o-mini-tts",
  },
  "OpenAI:gpt-transcribe": {
    currency: "USD",
    perMinuteUSD: 0.0045,
    updatedAt: PRICE_UPDATED_AT,
    source: "https://developers.openai.com/api/docs/pricing",
  },
  "OpenAI:gpt-4o-transcribe": {
    currency: "USD",
    perMinuteUSD: 0.006,
    inputPer1MTokens: 2.5,
    outputPer1MTokens: 10,
    updatedAt: PRICE_UPDATED_AT,
    source: "https://developers.openai.com/api/docs/pricing",
  },
  "OpenAI:gpt-4o-mini-transcribe": {
    currency: "USD",
    perMinuteUSD: 0.003,
    inputPer1MTokens: 1.25,
    outputPer1MTokens: 5,
    updatedAt: PRICE_UPDATED_AT,
    source: "https://developers.openai.com/api/docs/pricing",
  },

  // ---------------- Anthropic ----------------
  "Anthropic:claude-fable-5": {
    currency: "USD",
    inputPer1MTokens: 10,
    outputPer1MTokens: 50,
    updatedAt: PRICE_UPDATED_AT,
    source: "https://docs.anthropic.com/",
  },
  "Anthropic:claude-opus-5-5": {
    currency: "USD",
    inputPer1MTokens: 4,
    cachedInputPer1MTokens: 0.2,
    cacheWritePer1MTokens: 5,
    outputPer1MTokens: 20,
    updatedAt: PRICE_UPDATED_AT,
    source: "https://www.anthropic.com/claude-opus-5-5",
  },
  "Anthropic:claude-opus-5": {
    currency: "USD",
    inputPer1MTokens: 5,
    cachedInputPer1MTokens: 0.5,
    cacheWritePer1MTokens: 6.25,
    outputPer1MTokens: 25,
    updatedAt: PRICE_UPDATED_AT,
    source: "https://www.anthropic.com/news/claude-opus-5",
  },
  "Anthropic:claude-opus-4-8": {
    currency: "USD",
    inputPer1MTokens: 5,
    outputPer1MTokens: 25,
    updatedAt: PRICE_UPDATED_AT,
    source: "https://www.anthropic.com/news/claude-opus-5",
  },
  "Anthropic:claude-sonnet-5": {
    currency: "USD",
    inputPer1MTokens: 2,
    outputPer1MTokens: 10,
    updatedAt: PRICE_UPDATED_AT,
    source: "https://docs.anthropic.com/",
  },
  "Anthropic:claude-haiku-4-5-20251001": {
    currency: "USD",
    inputPer1MTokens: 1,
    outputPer1MTokens: 5,
    updatedAt: PRICE_UPDATED_AT,
    source: "https://docs.anthropic.com/",
  },

  // ---------------- Google ----------------
  "Google:gemini-3.1-flash-lite": {
    currency: "USD",
    inputPer1MTokens: 0.25,
    audioInputPer1MTokens: 0.5,
    cachedInputPer1MTokens: 0.025,
    outputPer1MTokens: 1.5,
    updatedAt: PRICE_UPDATED_AT,
    source: "https://ai.google.dev/gemini-api/docs/pricing",
  },
  "Google:gemini-3.1-pro-preview": {
    currency: "USD",
    inputPer1MTokens: 2,
    outputPer1MTokens: 12,
    longContextAfterTokens: 200000,
    longContextInputMultiplier: 2,
    longContextOutputMultiplier: 1.5,
    updatedAt: PRICE_UPDATED_AT,
    source: "https://ai.google.dev/gemini-api/docs/gemini-3",
  },
  "Google:gemini-3-flash-preview": {
    currency: "USD",
    inputPer1MTokens: 0.5,
    audioInputPer1MTokens: 1,
    cachedInputPer1MTokens: 0.05,
    outputPer1MTokens: 3,
    updatedAt: PRICE_UPDATED_AT,
    source: "https://ai.google.dev/gemini-api/docs/pricing",
  },
  "Google:gemini-3.1-flash-image": {
    currency: "USD",
    textInputPer1MTokens: 0.5,
    perImageByVariantUSD: {
      "0.5K": 0.045,
      "1K": 0.067,
      "2K": 0.101,
      "4K": 0.151,
    },
    updatedAt: PRICE_UPDATED_AT,
    source: "https://ai.google.dev/gemini-api/docs/pricing",
  },
  "Google:gemini-3.1-flash-image-preview": {
    currency: "USD",
    textInputPer1MTokens: 0.25,
    perImageUSD: 0.067,
    updatedAt: PRICE_UPDATED_AT,
    source: "https://ai.google.dev/gemini-api/docs/gemini-3",
  },
  "Google:gemini-3-pro-image": {
    currency: "USD",
    textInputPer1MTokens: 2,
    perImageUSD: 0.134,
    updatedAt: PRICE_UPDATED_AT,
    source: "https://ai.google.dev/gemini-api/docs/gemini-3",
  },
  "Google:veo-3.1-generate-preview": {
    currency: "USD",
    perSecondByResolutionUSD: { "720p": 0.4, "1080p": 0.4, "4k": 0.6 },
    updatedAt: PRICE_UPDATED_AT,
    source: "https://ai.google.dev/gemini-api/docs/pricing",
  },
  "Google:veo-3.1-fast-generate-preview": {
    currency: "USD",
    perSecondByResolutionUSD: { "720p": 0.1, "1080p": 0.12, "4k": 0.3 },
    updatedAt: PRICE_UPDATED_AT,
    source: "https://ai.google.dev/gemini-api/docs/pricing",
  },
  "Google:veo-3.1-lite-generate-preview": {
    currency: "USD",
    perSecondByResolutionUSD: { "720p": 0.05, "1080p": 0.08 },
    updatedAt: PRICE_UPDATED_AT,
    source: "https://ai.google.dev/gemini-api/docs/pricing",
  },
  "Google:lyria-3.5": {
    currency: "USD",
    perRequestUSD: 0.08,
    updatedAt: PRICE_UPDATED_AT,
    source: "https://ai.google.dev/gemini-api/docs/pricing",
    notes: ["Price is per generated full song."],
  },

  // ---------------- xAI ----------------
  "xAI:grok-4.7": {
    currency: "USD",
    inputPer1MTokens: 2,
    outputPer1MTokens: 6,
    updatedAt: PRICE_UPDATED_AT,
    source: "https://docs.x.ai/developers/models",
  },
  "xAI:grok-imagine-image-2.0": {
    currency: "USD",
    perImageByVariantUSD: {
      "low:1K": 0.04,
      "low:1.5K": 0.05,
      "low:2K": 0.06,
      "medium:1K": 0.06,
      "medium:1.5K": 0.07,
      "medium:2K": 0.08,
    },
    updatedAt: PRICE_UPDATED_AT,
    source: "https://docs.x.ai/developers/pricing",
  },
  "xAI:grok-imagine-image": {
    currency: "USD",
    perImageByVariantUSD: { "1K": 0.02, "2K": 0.02 },
    updatedAt: PRICE_UPDATED_AT,
    source: "https://docs.x.ai/developers/pricing",
  },
  "xAI:grok-imagine-image-quality": {
    currency: "USD",
    perImageByVariantUSD: { "1K": 0.05, "1.5K": 0.06, "2K": 0.07 },
    updatedAt: PRICE_UPDATED_AT,
    source: "https://docs.x.ai/developers/pricing",
  },
  "xAI:grok-imagine-video-1.5": {
    currency: "USD",
    perSecondByResolutionUSD: { "480p": 0.08, "720p": 0.14, "1080p": 0.25 },
    updatedAt: PRICE_UPDATED_AT,
    source: "https://docs.x.ai/developers/models/grok-imagine-video-1.5",
  },

  // ---------------- Mistral ----------------
  "MistralAI:mistral-large-latest": {
    currency: "USD",
    inputPer1MTokens: 0.5,
    outputPer1MTokens: 1.5,
    updatedAt: PRICE_UPDATED_AT,
    source: "https://mistral.ai/pricing/api/",
  },
  "MistralAI:mistral-medium-latest": {
    currency: "USD",
    inputPer1MTokens: 1.5,
    outputPer1MTokens: 7.5,
    updatedAt: PRICE_UPDATED_AT,
    source: "https://mistral.ai/pricing/api/",
  },
  "MistralAI:mistral-small-latest": {
    currency: "USD",
    inputPer1MTokens: 0.15,
    outputPer1MTokens: 0.6,
    updatedAt: PRICE_UPDATED_AT,
    source: "https://mistral.ai/pricing/api/",
  },
  "MistralAI:voxtral": {
    currency: "USD",
    perMinuteUSD: 0.004,
    inputPer1MTokens: 0.1,
    outputPer1MTokens: 0.4,
    updatedAt: PRICE_UPDATED_AT,
    source: "https://mistral.ai/pricing/api/",
  },
  "MistralAI:voxtral-tts": {
    currency: "USD",
    per1MCharactersUSD: 16,
    updatedAt: PRICE_UPDATED_AT,
    source: "https://docs.mistral.ai/models/voxtral-tts-26-03",
  },

  // ---------------- DeepSeek ----------------
  "DeepSeek:deepseek-flash": {
    currency: "USD",
    inputPer1MTokens: 0.3,
    cachedInputPer1MTokens: 0.006,
    outputPer1MTokens: 1.2,
    offPeak: {
      inputPer1MTokens: 0.15,
      cachedInputPer1MTokens: 0.003,
      outputPer1MTokens: 0.6,
    },
    updatedAt: PRICE_UPDATED_AT,
    source: "https://api-docs.deepseek.com/quick_start/pricing/",
  },
  "DeepSeek:deepseek-v4-flash": {
    currency: "USD",
    inputPer1MTokens: 0.3,
    cachedInputPer1MTokens: 0.006,
    outputPer1MTokens: 1.2,
    offPeak: {
      inputPer1MTokens: 0.15,
      cachedInputPer1MTokens: 0.003,
      outputPer1MTokens: 0.6,
    },
    updatedAt: PRICE_UPDATED_AT,
    source: "https://api-docs.deepseek.com/quick_start/pricing/",
    notes: ["Legacy alias currently billed at DeepSeek Flash pricing."],
  },
  "DeepSeek:deepseek-v4-pro": {
    currency: "USD",
    inputPer1MTokens: 1.32,
    cachedInputPer1MTokens: 0.044,
    outputPer1MTokens: 3.96,
    offPeak: {
      inputPer1MTokens: 0.66,
      cachedInputPer1MTokens: 0.022,
      outputPer1MTokens: 1.98,
    },
    updatedAt: PRICE_UPDATED_AT,
    source: "https://api-docs.deepseek.com/quick_start/pricing/",
  },

  // ---------------- Groq ----------------
  "Groq:openai/gpt-oss-120b": {
    currency: "USD",
    inputPer1MTokens: 0.15,
    outputPer1MTokens: 0.6,
    updatedAt: PRICE_UPDATED_AT,
    source: "https://console.groq.com/docs/models",
  },
  "Groq:openai/gpt-oss-20b": {
    currency: "USD",
    inputPer1MTokens: 0.075,
    outputPer1MTokens: 0.3,
    updatedAt: PRICE_UPDATED_AT,
    source: "https://console.groq.com/docs/models",
  },
  "Groq:qwen/qwen3.8-27b": {
    currency: "USD",
    inputPer1MTokens: 0.8,
    outputPer1MTokens: 4,
    updatedAt: PRICE_UPDATED_AT,
    source: "https://console.groq.com/docs/models",
  },
  "Groq:whisper-large-v3": {
    currency: "USD",
    perHourUSD: 0.111,
    updatedAt: PRICE_UPDATED_AT,
    source: "https://console.groq.com/docs/speech-to-text",
  },
  "Groq:whisper-large-v3-turbo": {
    currency: "USD",
    perHourUSD: 0.04,
    updatedAt: PRICE_UPDATED_AT,
    source: "https://console.groq.com/docs/speech-to-text",
  },
  "Groq:canopylabs/orpheus-v1-english": {
    currency: "USD",
    per1MCharactersUSD: 22,
    updatedAt: PRICE_UPDATED_AT,
    source: "https://console.groq.com/docs/models",
  },
  "Groq:canopylabs/orpheus-arabic-saudi": {
    currency: "USD",
    per1MCharactersUSD: 40,
    updatedAt: PRICE_UPDATED_AT,
    source: "https://console.groq.com/docs/models",
  },

  // ---------------- Cohere ----------------
  "Cohere:command-a": {
    currency: "USD",
    inputPer1MTokens: 2.5,
    outputPer1MTokens: 10,
    updatedAt: PRICE_UPDATED_AT,
    source: "https://docs.cohere.com/docs/command-a",
  },
  "Cohere:command-a-03-2025": {
    currency: "USD",
    inputPer1MTokens: 2.5,
    outputPer1MTokens: 10,
    updatedAt: PRICE_UPDATED_AT,
    source: "https://docs.cohere.com/docs/command-a",
  },
  "Cohere:command-r-08-2024": {
    currency: "USD",
    inputPer1MTokens: 0.15,
    outputPer1MTokens: 0.6,
    updatedAt: PRICE_UPDATED_AT,
    source: "https://docs.cohere.com/docs/command-r",
  },

  // ---------------- Stability ----------------
  "Stability:stable-image-core": {
    currency: "USD",
    perImageUSD: 0.03,
    updatedAt: PRICE_UPDATED_AT,
    source: "https://platform.stability.ai/pricing",
    notes: ["3 Stability credits per successful generation; 1 credit = $0.01."],
  },
};

/**
 * This is NOT intended to contain every existing model.
 *
 * featuredModels = useful/current well-known models.
 *
 * Use:
 *
 *   await listModels("OpenAI")
 *
 * to retrieve the provider's live model catalog where supported.
 */
export const AI_PROVIDERS: ProviderInfo[] = [
  {
    id: "OpenAI",
    name: "OpenAI",

    api: {
      baseURL: "https://api.openai.com/v1",
      models: "/models",
      chat: "/chat/completions",
      image: "/images/generations",
      tts: "/audio/speech",
      stt: "/audio/transcriptions",
    },

    env: ["OPENAI_API_KEY"],

    categories: ["text", "vision", "image", "audio", "tts", "stt", "embedding"],

    openAICompatible: true,

    featuredModels: [
      { id: "gpt-6-astra", categories: ["text", "vision"] },
      { id: "gpt-6.1-sol", categories: ["text", "vision"] },
      { id: "gpt-6-luna", categories: ["text", "vision"] },
      {
        id: "gpt-5.6-sol",
        categories: ["text", "vision"],
      },
      {
        id: "gpt-5.6-terra",
        categories: ["text", "vision"],
      },
      {
        id: "gpt-5.6-luna",
        categories: ["text", "vision"],
      },

      {
        id: "gpt-image-2",
        categories: ["image"],
      },

      {
        id: "gpt-4o-mini-tts",
        categories: ["tts", "audio"],
      },

      {
        id: "gpt-transcribe",
        categories: ["stt", "audio"],
      },

      {
        id: "gpt-4o-transcribe",
        categories: ["stt", "audio"],
      },

      {
        id: "gpt-4o-mini-transcribe",
        categories: ["stt", "audio"],
      },
    ],
  },

  {
    id: "Anthropic",
    name: "Anthropic / Claude",

    api: {
      baseURL: "https://api.anthropic.com",
      models: "/v1/models",
      chat: "/v1/messages",
    },

    env: ["ANTHROPIC_API_KEY"],

    categories: ["text", "vision"],

    featuredModels: [
      { id: "claude-opus-5-5", categories: ["text", "vision"] },
      {
        id: "claude-opus-5",
        categories: ["text", "vision"],
      },
      {
        id: "claude-sonnet-5",
        categories: ["text", "vision"],
      },
      {
        id: "claude-fable-5",
        categories: ["text", "vision"],
      },
      {
        id: "claude-opus-4-8",
        categories: ["text", "vision"],
      },
      {
        id: "claude-sonnet-4-6",
        categories: ["text", "vision"],
      },
      {
        id: "claude-haiku-4-5-20251001",
        categories: ["text", "vision"],
      },
    ],
  },

  {
    id: "Google",
    name: "Google Gemini",

    api: {
      baseURL: "https://generativelanguage.googleapis.com",
      models: "/v1beta/models",
      chat: "/v1beta/openai/chat/completions",
    },

    env: ["GEMINI_API_KEY", "GOOGLE_API_KEY"],

    categories: ["text", "vision", "image", "video", "audio"],

    openAICompatible: true,

    featuredModels: [
      { id: "gemini-3.1-pro-preview", categories: ["text", "vision", "audio"] },
      { id: "gemini-3.1-flash-lite", categories: ["text", "vision", "audio"] },
      { id: "gemini-3-flash-preview", categories: ["text", "vision", "audio"] },
      {
        id: "gemini-3.1-flash-image",
        categories: ["image", "vision", "text"],
      },
      {
        id: "gemini-3.1-flash-lite-image",
        categories: ["image", "vision", "text"],
      },
      {
        id: "gemini-3-pro-image",
        categories: ["image", "vision", "text"],
      },
      {
        id: "gemini-nano-banana-2.1",
        categories: ["image", "vision"],
      },

      {
        id: "veo-3.1-generate-preview",
        categories: ["video"],
      },
      {
        id: "veo-3.1-lite-generate-preview",
        categories: ["video"],
      },

      {
        id: "lyria-3.5",
        categories: ["audio"],
      },
      {
        id: "lyria-3-clip-preview",
        categories: ["audio"],
      },
    ],
  },

  {
    id: "xAI",
    name: "xAI / Grok",

    api: {
      baseURL: "https://api.x.ai/v1",
      models: "/models",
      chat: "/chat/completions",
      image: "/images/generations",
      video: "/videos/generations",
    },

    env: ["XAI_API_KEY"],

    categories: ["text", "vision", "image", "video", "audio"],

    openAICompatible: true,

    featuredModels: [
      {
        id: "grok-4.7",
        categories: ["text", "vision"],
      },
      {
        id: "grok-imagine-image-2.0",
        categories: ["image"],
      },
      {
        id: "grok-imagine-image-quality",
        categories: ["image"],
      },
      {
        id: "grok-imagine-video-1.5",
        categories: ["video"],
      },
    ],
  },

  {
    id: "MistralAI",
    name: "Mistral AI",

    api: {
      baseURL: "https://api.mistral.ai/v1",
      models: "/models",
      chat: "/chat/completions",
    },

    env: ["MISTRAL_API_KEY", "MistralAI_API_KEY"],

    categories: ["text", "vision", "audio", "tts", "embedding"],

    openAICompatible: true,

    featuredModels: [
      {
        id: "mistral-large-latest",
        categories: ["text", "vision"],
      },
      {
        id: "mistral-medium-latest",
        categories: ["text", "vision"],
      },
      {
        id: "voxtral",
        categories: ["audio"],
      },
      {
        id: "voxtral-tts",
        categories: ["audio", "tts"],
      },
    ],
  },

  {
    id: "DeepSeek",
    name: "DeepSeek",

    api: {
      baseURL: "https://api.deepseek.com",
      models: "/models",
      chat: "/chat/completions",
    },

    env: ["DEEPSEEK_API_KEY"],

    categories: ["text"],

    openAICompatible: true,

    featuredModels: [
      {
        id: "deepseek-v4-pro",
        categories: ["text"],
      },
      {
        id: "deepseek-v4-flash",
        categories: ["text"],
      },
    ],
  },

  {
    id: "OpenRouter",
    name: "OpenRouter",

    api: {
      baseURL: "https://openrouter.ai/api/v1",
      models: "/models",
      chat: "/chat/completions",
      video: "/videos",
    },

    env: ["OPENROUTER_API_KEY"],

    categories: ["text", "vision", "image", "video", "audio", "embedding", "rerank"],

    openAICompatible: true,

    featuredModels: [
      {
        id: "openrouter/auto",
        categories: ["text", "vision"],
      },
    ],
  },

  {
    id: "Fireworks",
    name: "Fireworks AI",

    api: {
      baseURL: "https://api.fireworks.ai/inference/v1",
      models: "/models",
      chat: "/chat/completions",
    },

    env: ["FIREWORKS_API_KEY"],

    categories: ["text", "vision", "image", "audio", "embedding"],

    openAICompatible: true,

    featuredModels: [],
  },

  {
    id: "Together",
    name: "Together AI",

    api: {
      baseURL: "https://api.together.xyz/v1",
      models: "/models",
      chat: "/chat/completions",
      image: "/images/generations",
    },

    env: ["TOGETHER_API_KEY"],

    categories: ["text", "vision", "image", "video", "audio", "embedding"],

    openAICompatible: true,

    featuredModels: [],
  },

  {
    id: "Groq",
    name: "Groq",

    api: {
      baseURL: "https://api.groq.com/openai/v1",
      models: "/models",
      chat: "/chat/completions",
      tts: "/audio/speech",
      stt: "/audio/transcriptions",
    },

    env: ["GROQ_API_KEY"],

    categories: ["text", "vision", "tts", "stt", "audio"],

    openAICompatible: true,

    featuredModels: [
      {
        id: "openai/gpt-oss-120b",
        categories: ["text"],
      },
      {
        id: "openai/gpt-oss-20b",
        categories: ["text"],
      },
      {
        id: "qwen/qwen3.8-27b",
        categories: ["text", "vision"],
      },
      {
        id: "whisper-large-v3",
        categories: ["stt"],
      },
      {
        id: "whisper-large-v3-turbo",
        categories: ["stt"],
      },
      {
        id: "canopylabs/orpheus-v1-english",
        categories: ["tts"],
      },
      {
        id: "canopylabs/orpheus-arabic-saudi",
        categories: ["tts"],
      },
    ],
  },

  {
    id: "Perplexity",
    name: "Perplexity",

    api: {
      baseURL: "https://api.perplexity.ai",
      chat: "/chat/completions",
    },

    env: ["PERPLEXITY_API_KEY"],

    categories: ["text"],

    openAICompatible: true,

    featuredModels: [
      {
        id: "sonar",
        categories: ["text"],
      },
    ],
  },

  {
    id: "Cohere",
    name: "Cohere",

    api: {
      baseURL: "https://api.cohere.com",
      models: "/v1/models",
      chat: "/v2/chat",
    },

    env: ["COHERE_API_KEY"],

    categories: ["text", "vision", "stt", "embedding", "rerank"],

    featuredModels: [
      {
        id: "command-a",
        categories: ["text"],
      },
      {
        id: "command-a-reasoning",
        categories: ["text"],
      },
      {
        id: "command-a-vision",
        categories: ["text", "vision"],
      },
    ],
  },

  {
    id: "HuggingFace",
    name: "Hugging Face",

    api: {
      baseURL: "https://router.huggingface.co",
      chat: "/v1/chat/completions",
    },

    env: ["HUGGINGFACE_API_KEY", "HF_TOKEN"],

    categories: ["text", "vision", "image", "video", "audio", "stt", "embedding"],

    openAICompatible: true,

    featuredModels: [
      {
        id: "black-forest-labs/FLUX.1-dev",
        categories: ["image"],
      },
    ],
  },

  {
    id: "Stability",
    name: "Stability AI",

    api: {
      baseURL: "https://api.stability.ai",
      image: "/v2beta/stable-image/generate/core",
    },

    env: ["STABILITY_API_KEY"],

    categories: ["image"],

    featuredModels: [
      {
        id: "stable-image-core",
        categories: ["image"],
      },
    ],
  },

  {
    id: "Hermes",
    name: "Hermes Agent Proxy",

    api: {
      // Hermes proxy is local by default and exposes the OpenAI wire format.
      baseURL: process.env.HERMES_PROXY_BASE_URL || "http://127.0.0.1:8645/v1",
      models: "/models",
      chat: "/chat/completions",
    },

    // The proxy accepts any bearer token, but keeping it configurable avoids
    // putting credentials in source and supports a hardened reverse proxy.
    env: ["HERMES_API_KEY"],

    categories: ["text", "vision"],

    openAICompatible: true,

    featuredModels: [
      { id: "auto", categories: ["text", "vision"] },
    ],
  },

  {
    id: "Ollama",
    name: "Ollama",

    api: {
      baseURL: process.env.OLLAMA_BASE_URL || "http://127.0.0.1:11434",
      models: "/api/tags",
      chat: "/api/chat",
    },

    env: [],

    categories: ["text", "vision", "embedding"],

    featuredModels: [],
  },
];

for (const provider of AI_PROVIDERS) {
  for (const model of provider.featuredModels) {
    model.price =
      MODEL_PRICING[`${provider.id}:${model.id}`] ||
      (provider.id === "Ollama"
        ? {
            currency: "USD",
            free: true,
            updatedAt: PRICE_UPDATED_AT,
            notes: ["Provider API cost is $0. Local electricity/hardware cost is not included."],
          }
        : {
            currency: "USD",
            dynamic: true,
            updatedAt: PRICE_UPDATED_AT,
            notes: ["Static price is not verified. Resolve live/provider pricing before selling."],
          });
  }
}

/* ============================================================
 * TYPES
 * ============================================================ */

export type GeneratedImage = {
  url?: string;
  b64json?: string;
  mime?: string;
};

export type GeneratedVideo = {
  url?: string;
  bytes_b64?: string;
  mime?: string;
  request_id?: string;
};

export type GeneratedAudio = {
  url?: string;
  bytes_b64?: string;
  mime?: string;
};

export type ChatMessage = {
  role: "assistant";
  content: string;

  image?: GeneratedImage;
  video?: GeneratedVideo;
  audio?: GeneratedAudio;
};

export type ChatCompletion = {
  id: string;
  object: "chat.completion";
  created: number;
  model: string;

  choices: Array<{
    index: number;
    message: ChatMessage;
    finish_reason: string | null;
  }>;

  usage?: {
    prompt_tokens?: number;
    completion_tokens?: number;
    total_tokens?: number;

    [key: string]: any;
  };

  provider?: AIProvider;

  /** Optional quote/cost metadata when caller attaches it. */
  price?: PriceQuote;

  raw?: any;
};

export type UsingOptions = {
  /**
   * Override chat model.
   */
  model?: string;

  /**
   * Key ONLY for the main/chat provider.
   *
   * For cross-provider media prefer:
   * image_apiKey
   * video_apiKey
   * audio_apiKey
   */
  apiKey?: string;

  temperature?: number;
  max_tokens?: number;
  top_p?: number;

  timeout?: number;

  headers?: Record<string, string>;

  /* ---------------- IMAGE ---------------- */

  image?: boolean;

  image_provider?: "OpenAI" | "Google" | "xAI" | "Stability" | "HuggingFace" | "Together";

  image_apiKey?: string;

  image_prompt?: string;

  image_model?: string;

  image_size?: string;

  image_aspectRatio?: string;

  image_resolution?: "512" | "1K" | "2K" | "4K" | string;

  /* ---------------- VIDEO ---------------- */

  video?: boolean;

  video_provider?: "Google" | "xAI" | "OpenRouter";

  video_apiKey?: string;

  video_prompt?: string;

  video_model?: string;

  video_aspectRatio?: string;

  video_resolution?: string;

  video_duration?: number;

  video_waitSeconds?: number;

  /* ---------------- AUDIO / TTS ---------------- */

  audio?: boolean;

  audio_provider?: "OpenAI" | "Groq";

  audio_apiKey?: string;

  audio_prompt?: string;

  audio_model?: string;

  audio_voice?: string;

  audio_format?: "mp3" | "wav" | "opus" | "aac" | "flac" | "ogg";

  audio_speed?: number;
};

export type UsingOpenAIProps = {
  user_prompt: string;
  system_prompt?: string;

  provider: AIProvider;

  model: string;

  stream?: boolean;

  time?: number;

  options?: UsingOptions;
};

/* ============================================================
 * AI CLIENT — SINGLE PUBLIC CLASS
 * ============================================================ */

export class AIClient {
  /** Full provider registry. */
  public readonly providers = AI_PROVIDERS;

  /** Debug logging, controlled by USING_OPENAI_DEBUG=1. */
  private readonly debug = process.env.USING_OPENAI_DEBUG === "1";

  private readonly pricingConfig: Required<NonNullable<AIClientOptions["pricing"]>>;

  public constructor(options: AIClientOptions = {}) {
    this.pricingConfig = {
      targetMarginPercent: options.pricing?.targetMarginPercent ?? 40,

      safetyBufferPercent: options.pricing?.safetyBufferPercent ?? 7.5,

      minimumProfitUSD: options.pricing?.minimumProfitUSD ?? 0.001,

      minimumSellPriceUSD: options.pricing?.minimumSellPriceUSD ?? 0.002,

      defaultExpectedOutputTokens: options.pricing?.defaultExpectedOutputTokens ?? 512,

      roundUSDTo: options.pricing?.roundUSDTo ?? 0.0001,
    };

    /*
     * ВАЖНО:
     * Автоматически привязываем ВСЕ методы AIClient к текущему экземпляру.
     *
     * После этого безопасно работают:
     *
     * const fn = ai.chat;
     * await fn(...)
     *
     * Proxy
     *
     * destructuring:
     * const { chat } = ai;
     *
     * callback:
     * someFunction(ai.chat)
     *
     * И this.debug / this.pricingConfig / private методы
     * больше не превращаются в undefined.
     */
    const proto = Object.getPrototypeOf(this);

    for (const key of Object.getOwnPropertyNames(proto)) {
      if (key === "constructor") {
        continue;
      }

      const descriptor = Object.getOwnPropertyDescriptor(proto, key);

      if (descriptor && typeof descriptor.value === "function") {
        (this as any)[key] = descriptor.value.bind(this);
      }
    }
  }
  /** Main modern alias. */
  public async chat(props: UsingOpenAIProps): Promise<ChatCompletion> {
    return this.usingOpenAI(props);
  }

  /** Additional semantic alias. */
  public async complete(props: UsingOpenAIProps): Promise<ChatCompletion> {
    return this.usingOpenAI(props);
  }

  /** Check provider capability from the registry. */
  public supports(provider: AIProvider, modality: AIModality): boolean {
    return this.getProvider(provider).categories.includes(modality);
  }

  public getProvider(provider: AIProvider): ProviderInfo {
    const found = AI_PROVIDERS.find((x) => x.id === provider);

    if (!found) {
      throw new Error(`Unknown AI provider: ${provider}`);
    }

    return found;
  }

  private getEnv(...names: string[]): string | undefined {
    for (const name of names) {
      const value = process.env[name];

      if (value) {
        return value;
      }
    }

    return undefined;
  }

  private resolveApiKey(provider: AIProvider, explicit?: string): string | undefined {
    if (explicit) return explicit;

    const cfg = this.getProvider(provider);

    return this.getEnv(...cfg.env);
  }

  private requireApiKey(provider: AIProvider, explicit?: string): string {
    if (provider === "Ollama") {
      return "ollama";
    }

    // The local Hermes proxy deliberately accepts any bearer token. Keep a
    // non-secret local default, while still requiring HERMES_API_KEY when the
    // proxy is moved off loopback or protected by a real reverse proxy.
    if (provider === "Hermes" && !explicit && !this.getEnv("HERMES_API_KEY")) {
      const baseURL = this.getProvider(provider).api.baseURL;
      if (/^https?:\/\/(127\.0\.0\.1|localhost)(:\d+)?(?:\/|$)/i.test(baseURL)) {
        return "hermes-local";
      }
    }

    const key = this.resolveApiKey(provider, explicit);

    if (!key) {
      const env = this.getProvider(provider).env.join(" or ");

      throw new Error(`[${provider}] Missing API key. Expected ${env || "API key"}`);
    }

    return key;
  }

  private stripThink(value: string): string {
    return String(value || "")
      .replace(/<think>[\s\S]*?<\/think>/gi, "")
      .trim();
  }

  private sleep(ms: number) {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  private toBase64(data: ArrayBuffer | Uint8Array): string {
    return Buffer.from(data instanceof Uint8Array ? data : new Uint8Array(data)).toString("base64");
  }

  private makeCompletion(
    provider: AIProvider,
    model: string,
    content: string,
    raw?: any,
  ): ChatCompletion {
    return {
      id: raw?.id || `${provider.toLowerCase()}-${Date.now()}`,
      object: "chat.completion",
      created: Math.floor(Date.now() / 1000),
      model,

      provider,

      choices: [
        {
          index: 0,

          message: {
            role: "assistant",
            content: this.stripThink(content),
          },

          finish_reason: raw?.stop_reason || raw?.finish_reason || "stop",
        },
      ],

      usage: raw?.usage,

      raw,
    };
  }

  private async readResponseError(response: Response): Promise<string> {
    let body = "";

    try {
      body = await response.text();
    } catch {}

    try {
      const json = JSON.parse(body);

      return json?.error?.message || json?.message || json?.error || body;
    } catch {
      return body;
    }
  }

  private async ensureOK(response: Response) {
    if (response.ok) return;

    const message = await this.readResponseError(response);

    if (response.status === 401) {
      throw new Error(`Invalid API key: ${message}`);
    }

    if (response.status === 403) {
      throw new Error(`API access forbidden: ${message}`);
    }

    if (response.status === 404) {
      throw new Error(`API endpoint/model not found: ${message}`);
    }

    if (response.status === 429) {
      throw new Error(`Rate limit exceeded: ${message}`);
    }

    if (response.status >= 500) {
      throw new Error(`Provider server error ${response.status}: ${message}`);
    }

    throw new Error(`HTTP ${response.status}: ${message}`);
  }

  private async request(url: string, init: RequestInit = {}, timeout = 90_000): Promise<Response> {
    const controller = new AbortController();

    const timer = setTimeout(() => controller.abort(), timeout);

    try {
      return await fetch(url, {
        ...init,

        signal: controller.signal,
      });
    } catch (error: any) {
      if (error?.name === "AbortError") {
        throw new Error(`Request timed out after ${timeout}ms`);
      }

      throw error;
    } finally {
      clearTimeout(timer);
    }
  }

  private bearerHeaders(token: string, extra: Record<string, string> = {}) {
    return {
      Authorization: `Bearer ${token}`,
      "Content-Type": "application/json",
      ...extra,
    };
  }

  /* ============================================================
   * PRICING
   * ============================================================ */

  /**
   * One pricing method for all modalities.
   *
   * Examples:
   *   await ai.getPrice("OpenAI", "gpt-5.6-sol", "text", "Hello")
   *   await ai.getPrice("Google", "veo-3.1-lite-generate-preview", "video", {
   *     prompt: "...", seconds: 8, resolution: "720p"
   *   })
   */

  public async getPrice(
    provider: AIProvider,
    model: string,
    modality: PricingModality,
    request: string | PriceRequest = {},
  ): Promise<PriceQuote> {
    // Normalize the provider BEFORE resolving its pricing.
    const rawProvider: unknown = provider;

    const providerId =
      typeof rawProvider === "string"
        ? rawProvider
        : rawProvider !== null && typeof rawProvider === "object" && !Array.isArray(rawProvider)
          ? ((
              rawProvider as {
                provider?: unknown;
                id?: unknown;
                name?: unknown;
              }
            ).provider ??
            (rawProvider as { id?: unknown }).id ??
            (rawProvider as { name?: unknown }).name)
          : undefined;

    if (typeof providerId !== "string" || !AI_PROVIDERS.some((item) => item.id === providerId)) {
      throw new Error("INVALID_AI_PROVIDER");
    }

    provider = providerId as AIProvider;

    const input: PriceRequest = typeof request === "string" ? { prompt: request } : request;

    const assumptions: string[] = [];

    // Provider is now guaranteed to be a registered string ID.
    const rate = await this.resolveModelPrice(provider, model);

    if (rate.dynamic && !this.hasUsableRate(rate, modality)) {
      throw new Error(
        `[${provider}:${model}] Pricing is dynamic/unknown. Refusing to quote $0. ` +
          `Add a verified MODEL_PRICING entry or use a provider that exposes live pricing.`,
      );
    }

    let estimated = false;
    let providerCostUSD = 0;

    const prompt = `${input.systemPrompt || ""}\n${input.prompt || ""}`.trim();
    let inputTokens = input.inputTokens;

    if (inputTokens === undefined && prompt) {
      inputTokens = this.estimateTokens(prompt);
      estimated = true;
      assumptions.push(`Input tokens estimated from prompt text: ${inputTokens}.`);
    }

    if (provider === "Ollama" && rate.free) {
      assumptions.push("Ollama provider API cost is zero; local compute/electricity is excluded.");
    } else if (modality === "text") {
      let outputTokens = input.outputTokens;

      if (outputTokens === undefined) {
        outputTokens = input.expectedOutputTokens ?? this.pricingConfig.defaultExpectedOutputTokens;
        estimated = true;
        assumptions.push(`Output tokens are unknown before generation; assumed ${outputTokens}.`);
      }

      providerCostUSD = this.calculateTextCost(provider, rate, {
        inputTokens: inputTokens ?? 0,
        cachedInputTokens: input.cachedInputTokens ?? 0,
        cacheWriteTokens: input.cacheWriteTokens ?? 0,
        outputTokens,
      });
    } else if (modality === "image") {
      const quantity = Math.max(1, input.quantity ?? 1);
      const quality = input.quality || (model === "gpt-image-2" ? "medium" : undefined);
      const size = input.size || (model === "gpt-image-2" ? "1024x1024" : undefined);
      const resolution = input.resolution || this.sizeToResolution(size) || "1K";

      let flat: number | undefined;
      const variants = rate.perImageByVariantUSD;
      if (variants) {
        const keys = [
          quality && size ? `${quality}:${size}` : undefined,
          quality && resolution ? `${quality}:${resolution}` : undefined,
          size,
          resolution,
        ].filter(Boolean) as string[];
        for (const key of keys) {
          if (variants[key] !== undefined) {
            flat = variants[key];
            break;
          }
        }
      }
      flat ??= rate.perImageUSD;

      if (flat !== undefined) {
        providerCostUSD += flat * quantity;
      } else if (
        rate.imageOutputPer1MTokens !== undefined &&
        input.imageOutputTokens !== undefined
      ) {
        providerCostUSD +=
          (input.imageOutputTokens / 1_000_000) * rate.imageOutputPer1MTokens * quantity;
      } else {
        throw new Error(
          `[${provider}:${model}] Image price depends on output tokens/variant. ` +
            `Pass imageOutputTokens, or add perImageUSD/perImageByVariantUSD for the selected quality/size.`,
        );
      }

      if (inputTokens && rate.textInputPer1MTokens !== undefined) {
        providerCostUSD += (inputTokens / 1_000_000) * rate.textInputPer1MTokens;
      }
      if (input.imageInputTokens && rate.imageInputPer1MTokens !== undefined) {
        providerCostUSD += (input.imageInputTokens / 1_000_000) * rate.imageInputPer1MTokens;
      }

      if (!input.quality && quality) assumptions.push(`Image quality defaulted to ${quality}.`);
      if (!input.size && size) assumptions.push(`Image size defaulted to ${size}.`);
    } else if (modality === "video") {
      const seconds = input.seconds ?? 1;
      if (input.seconds === undefined) {
        estimated = true;
        assumptions.push("Video duration was not supplied; quote is for 1 second.");
      }

      const resolution = (input.resolution || "720p").toLowerCase();
      let perSecond = rate.perSecondUSD;
      if (rate.perSecondByResolutionUSD) {
        perSecond =
          rate.perSecondByResolutionUSD[resolution] ??
          rate.perSecondByResolutionUSD[input.resolution || ""];
      }
      if (perSecond === undefined) {
        throw new Error(
          `[${provider}:${model}] No verified video per-second rate for ${resolution}.`,
        );
      }
      providerCostUSD = perSecond * seconds * Math.max(1, input.quantity ?? 1);
    } else if (modality === "tts" || modality === "audio") {
      const characters = input.characters ?? (input.prompt ? input.prompt.length : undefined);

      if (rate.per1MCharactersUSD !== undefined && characters !== undefined) {
        providerCostUSD = (characters / 1_000_000) * rate.per1MCharactersUSD;
      } else if (rate.per1000CharactersUSD !== undefined && characters !== undefined) {
        providerCostUSD = (characters / 1_000) * rate.per1000CharactersUSD;
      } else if (rate.perMinuteUSD !== undefined && input.minutes !== undefined) {
        providerCostUSD = rate.perMinuteUSD * input.minutes;
      } else if (
        rate.audioOutputPer1MTokens !== undefined &&
        input.audioOutputTokens !== undefined
      ) {
        providerCostUSD = (input.audioOutputTokens / 1_000_000) * rate.audioOutputPer1MTokens;
        if (inputTokens && rate.inputPer1MTokens !== undefined) {
          providerCostUSD += (inputTokens / 1_000_000) * rate.inputPer1MTokens;
        }
      } else if (rate.perRequestUSD !== undefined) {
        providerCostUSD = rate.perRequestUSD * Math.max(1, input.quantity ?? 1);
      } else {
        throw new Error(
          `[${provider}:${model}] Not enough information to price ${modality}. ` +
            `Pass characters/minutes/audioOutputTokens as required by this model.`,
        );
      }
    } else if (modality === "stt") {
      if (rate.perMinuteUSD !== undefined) {
        if (input.minutes === undefined) {
          throw new Error(`[${provider}:${model}] STT pricing requires minutes.`);
        }
        providerCostUSD = rate.perMinuteUSD * input.minutes;
      } else if (rate.perHourUSD !== undefined) {
        const hours = input.hours ?? (input.minutes !== undefined ? input.minutes / 60 : undefined);
        if (hours === undefined) {
          throw new Error(`[${provider}:${model}] STT pricing requires minutes or hours.`);
        }
        providerCostUSD = rate.perHourUSD * hours;
      } else {
        throw new Error(`[${provider}:${model}] No verified STT time rate.`);
      }
    }

    if (rate.perRequestUSD !== undefined && modality === "text") {
      providerCostUSD += rate.perRequestUSD;
    }

    const targetMarginPercent = this.clampPercent(
      input.targetMarginPercent ?? this.pricingConfig.targetMarginPercent,
      0,
      95,
    );
    const safetyBufferPercent = this.clampPercent(
      input.safetyBufferPercent ?? this.pricingConfig.safetyBufferPercent,
      0,
      500,
    );
    const minimumProfitUSD = Math.max(
      0,
      input.minimumProfitUSD ?? this.pricingConfig.minimumProfitUSD,
    );
    const minimumSellPriceUSD = Math.max(
      0,
      input.minimumSellPriceUSD ?? this.pricingConfig.minimumSellPriceUSD,
    );

    const bufferedCostUSD = providerCostUSD * (1 + safetyBufferPercent / 100);
    const marginDivisor = 1 - targetMarginPercent / 100;
    const marginSell = marginDivisor > 0 ? bufferedCostUSD / marginDivisor : bufferedCostUSD;
    const rawSell = Math.max(marginSell, bufferedCostUSD + minimumProfitUSD, minimumSellPriceUSD);
    const sellPriceUSD = this.roundUp(rawSell, this.pricingConfig.roundUSDTo);
    const profitUSD = sellPriceUSD - providerCostUSD;
    const grossMarginPercent = sellPriceUSD > 0 ? (profitUSD / sellPriceUSD) * 100 : 0;

    return {
      provider,
      model,
      modality,
      currency: "USD",
      providerCostUSD: this.round(providerCostUSD, 8),
      bufferedCostUSD: this.round(bufferedCostUSD, 8),
      sellPriceUSD,
      profitUSD: this.round(profitUSD, 8),
      grossMarginPercent: this.round(grossMarginPercent, 4),
      estimated,
      usage: {
        inputTokens,
        cachedInputTokens: input.cachedInputTokens,
        cacheWriteTokens: input.cacheWriteTokens,
        outputTokens:
          input.outputTokens ??
          (modality === "text"
            ? (input.expectedOutputTokens ?? this.pricingConfig.defaultExpectedOutputTokens)
            : undefined),
        quantity: input.quantity,
        seconds: input.seconds,
        minutes: input.minutes,
        hours: input.hours,
        characters:
          input.characters ??
          (input.prompt && (modality === "tts" || modality === "audio")
            ? input.prompt.length
            : undefined),
        pages: input.pages,
        quality: input.quality,
        size: input.size,
        resolution: input.resolution,
      },
      rate,
      assumptions,
    };
  }

  private async resolveModelPrice(provider: AIProvider, model: string): Promise<ModelPrice> {
    if (provider === "Ollama" || provider === "Hermes") {
      return {
        currency: "USD",
        free: true,
        updatedAt: PRICE_UPDATED_AT,
        notes: [
          provider === "Ollama"
            ? "Local provider API cost only; hardware/electricity is excluded."
            : "Hermes proxy usage is billed by its upstream account, not by this backend.",
        ],
      };
    }

    const staticRate = MODEL_PRICING[`${provider}:${model}`];
    if (staticRate) return this.applyTimeSensitivePricing(provider, staticRate);

    // OpenRouter exposes live prompt/completion pricing in its model catalog.
    if (provider === "OpenRouter") {
      try {
        const models = await this.listModels(provider);
        const found = models.find((x: any) => (x?.id || x?.name) === model);
        const live = found?.price || this.extractLivePrice(provider, found);
        if (live && this.hasAnyPrice(live)) return live;
      } catch {
        // Fall through to explicit unknown-pricing error below.
      }
    }

    return {
      currency: "USD",
      dynamic: true,
      updatedAt: PRICE_UPDATED_AT,
      notes: ["No verified static price is registered for this model."],
    };
  }

  private applyTimeSensitivePricing(provider: AIProvider, rate: ModelPrice): ModelPrice {
    if (provider !== "DeepSeek" || !rate.offPeak) return rate;
    if (this.isDeepSeekPeakUTC(new Date())) return rate;
    return { ...rate, ...rate.offPeak, offPeak: rate.offPeak };
  }

  private isDeepSeekPeakUTC(now: Date): boolean {
    const day = now.getUTCDay(); // 1..5 Monday-Friday
    if (day < 1 || day > 5) return false;
    const hour = now.getUTCHours();
    return (hour >= 1 && hour < 4) || (hour >= 6 && hour < 10);
  }

  private calculateTextCost(
    provider: AIProvider,
    rate: ModelPrice,
    usage: {
      inputTokens: number;
      cachedInputTokens: number;
      cacheWriteTokens: number;
      outputTokens: number;
    },
  ): number {
    let inputRate = rate.inputPer1MTokens;
    let cachedRate = rate.cachedInputPer1MTokens ?? inputRate;
    let cacheWriteRate = rate.cacheWritePer1MTokens ?? inputRate;
    let outputRate = rate.outputPer1MTokens;

    if (inputRate === undefined || outputRate === undefined) {
      if (rate.free) return 0;
      throw new Error(`[${provider}] This model has no verified text token price.`);
    }

    if (
      rate.longContextAfterTokens !== undefined &&
      usage.inputTokens > rate.longContextAfterTokens
    ) {
      inputRate *= rate.longContextInputMultiplier ?? 1;
      cachedRate = (cachedRate ?? inputRate) * (rate.longContextInputMultiplier ?? 1);
      cacheWriteRate = (cacheWriteRate ?? inputRate) * (rate.longContextInputMultiplier ?? 1);
      outputRate *= rate.longContextOutputMultiplier ?? 1;
    }

    const cached = Math.min(usage.inputTokens, Math.max(0, usage.cachedInputTokens));
    const cacheWrite = Math.min(
      Math.max(0, usage.inputTokens - cached),
      Math.max(0, usage.cacheWriteTokens),
    );
    const regular = Math.max(0, usage.inputTokens - cached - cacheWrite);

    return (
      (regular / 1_000_000) * inputRate +
      (cached / 1_000_000) * (cachedRate ?? inputRate) +
      (cacheWrite / 1_000_000) * (cacheWriteRate ?? inputRate) +
      (usage.outputTokens / 1_000_000) * outputRate
    );
  }

  private estimateTokens(text: string): number {
    if (!text) return 0;
    const cyrillic = (text.match(/[\u0400-\u04FF]/g) || []).length;
    const codeChars = (text.match(/[{}()[\];:=<>/\\|`~]/g) || []).length;
    const cyrRatio = cyrillic / Math.max(1, text.length);
    const codeRatio = codeChars / Math.max(1, text.length);
    const charsPerToken = cyrRatio > 0.25 ? 2.2 : codeRatio > 0.08 ? 3.0 : 3.8;
    return Math.max(1, Math.ceil(text.length / charsPerToken));
  }

  private extractLivePrice(provider: AIProvider, raw: any): ModelPrice | undefined {
    if (!raw) return undefined;

    if (provider === "OpenRouter" && raw.pricing) {
      const p = raw.pricing;
      const perTokenToMillion = (v: any): number | undefined => {
        const n = Number(v);
        return Number.isFinite(n) ? n * 1_000_000 : undefined;
      };
      const direct = (v: any): number | undefined => {
        const n = Number(v);
        return Number.isFinite(n) ? n : undefined;
      };

      const price: ModelPrice = {
        currency: "USD",
        inputPer1MTokens: perTokenToMillion(p.prompt),
        outputPer1MTokens: perTokenToMillion(p.completion),
        perRequestUSD: direct(p.request),
        perImageUSD: direct(p.image),
        updatedAt: PRICE_UPDATED_AT,
        source: "https://openrouter.ai/api/v1/models",
        dynamic: true,
      };
      return price;
    }

    return undefined;
  }

  private enrichLiveModels(provider: AIProvider, models: any[]): any[] {
    return models.map((raw: any) => {
      const id = raw?.id || raw?.name || raw?.model;
      const price =
        (id ? MODEL_PRICING[`${provider}:${id}`] : undefined) ||
        this.extractLivePrice(provider, raw) ||
        (provider === "Ollama"
          ? { currency: "USD", free: true, updatedAt: PRICE_UPDATED_AT }
          : {
              currency: "USD",
              dynamic: true,
              updatedAt: PRICE_UPDATED_AT,
              notes: [
                "Price not exposed by this models endpoint and not present in static registry.",
              ],
            });
      return { ...raw, price };
    });
  }

  private hasAnyPrice(rate: ModelPrice): boolean {
    return Boolean(
      rate.free ||
      rate.inputPer1MTokens !== undefined ||
      rate.outputPer1MTokens !== undefined ||
      rate.perImageUSD !== undefined ||
      rate.perImageByVariantUSD ||
      rate.perSecondUSD !== undefined ||
      rate.perSecondByResolutionUSD ||
      rate.perMinuteUSD !== undefined ||
      rate.perHourUSD !== undefined ||
      rate.per1MCharactersUSD !== undefined ||
      rate.per1000CharactersUSD !== undefined ||
      rate.perRequestUSD !== undefined ||
      rate.imageOutputPer1MTokens !== undefined ||
      rate.audioOutputPer1MTokens !== undefined,
    );
  }

  private hasUsableRate(rate: ModelPrice, modality: PricingModality): boolean {
    if (rate.free) return true;
    if (modality === "text")
      return rate.inputPer1MTokens !== undefined && rate.outputPer1MTokens !== undefined;
    if (modality === "image")
      return Boolean(
        rate.perImageUSD !== undefined ||
        rate.perImageByVariantUSD ||
        rate.imageOutputPer1MTokens !== undefined,
      );
    if (modality === "video")
      return Boolean(rate.perSecondUSD !== undefined || rate.perSecondByResolutionUSD);
    if (modality === "stt") return rate.perMinuteUSD !== undefined || rate.perHourUSD !== undefined;
    return Boolean(
      rate.per1MCharactersUSD !== undefined ||
      rate.per1000CharactersUSD !== undefined ||
      rate.perMinuteUSD !== undefined ||
      rate.audioOutputPer1MTokens !== undefined ||
      rate.perRequestUSD !== undefined,
    );
  }

  private sizeToResolution(size?: string): string | undefined {
    if (!size) return undefined;
    const match = /^(\d+)x(\d+)$/i.exec(size);
    if (!match) return undefined;
    const longest = Math.max(Number(match[1]), Number(match[2]));
    if (longest >= 3000) return "4K";
    if (longest >= 1800) return "2K";
    if (longest >= 1200) return "1.5K";
    if (longest >= 768) return "1K";
    return "0.5K";
  }

  private clampPercent(value: number, min: number, max: number): number {
    return Math.min(max, Math.max(min, value));
  }

  private round(value: number, decimals: number): number {
    const m = 10 ** decimals;
    return Math.round((value + Number.EPSILON) * m) / m;
  }

  private roundUp(value: number, step: number): number {
    if (step <= 0) return value;
    return this.round(Math.ceil((value - Number.EPSILON) / step) * step, 8);
  }

  /* ============================================================
   * LIVE MODEL CATALOG
   * ============================================================ */

  /**
   * Fetch actual provider models.
   *
   * For providers that expose a models endpoint this is preferable
   * to relying on static TypeScript unions.
   */
  public async listModels(
    provider: AIProvider,
    options?: {
      apiKey?: string;
    },
  ): Promise<any[]> {
    const cfg = this.getProvider(provider);

    if (provider === "Ollama") {
      const response = await this.request(`${cfg.api.baseURL}${cfg.api.models}`);

      await this.ensureOK(response);

      const json: any = await response.json();

      return this.enrichLiveModels(provider, json?.models || []);
    }

    if (provider === "Google") {
      const key = this.requireApiKey(provider, options?.apiKey);

      const response = await this.request(
        `${cfg.api.baseURL}${cfg.api.models}?key=${encodeURIComponent(key)}`,
      );

      await this.ensureOK(response);

      const json: any = await response.json();

      return this.enrichLiveModels(provider, json?.models || []);
    }

    if (provider === "Anthropic") {
      const key = this.requireApiKey(provider, options?.apiKey);

      const response = await this.request(`${cfg.api.baseURL}${cfg.api.models}`, {
        headers: {
          "x-api-key": key,
          "anthropic-version": "2023-06-01",
        },
      });

      await this.ensureOK(response);

      const json: any = await response.json();

      return this.enrichLiveModels(provider, json?.data || []);
    }

    if (!cfg.api.models) {
      return cfg.featuredModels;
    }

    const key = this.requireApiKey(provider, options?.apiKey);

    const response = await this.request(`${cfg.api.baseURL}${cfg.api.models}`, {
      headers: this.bearerHeaders(key),
    });

    await this.ensureOK(response);

    const json: any = await response.json();

    return this.enrichLiveModels(provider, json?.data || json?.models || json?.results || []);
  }

  public getFeaturedModels(provider?: AIProvider): ModelInfo[] {
    if (provider) {
      return this.getProvider(provider).featuredModels;
    }

    return AI_PROVIDERS.flatMap((x) =>
      x.featuredModels.map((m) => ({
        ...m,
        provider: x.id,
      })),
    ) as any;
  }

  public getProvidersByCategory(category: AIModality) {
    return AI_PROVIDERS.filter((provider) => provider.categories.includes(category));
  }

  /* ============================================================
   * MAIN FUNCTION
   * ============================================================ */

  public async usingOpenAI(props: UsingOpenAIProps): Promise<ChatCompletion> {
    const model = props.options?.model || props.model;

    let response: ChatCompletion;

    if (this.debug) {
      console.log("[AI request]", {
        provider: props.provider,
        model,
        promptLength: props.user_prompt.length,
      });
    }

    switch (props.provider) {
      case "Ollama":
        response = await this.chatOllama(props);
        break;

      case "Anthropic":
        response = await this.chatAnthropic(props);
        break;

      case "Cohere":
        response = await this.chatCohere(props);
        break;

      case "Stability":
        response = this.makeCompletion(props.provider, model, "Image provider.");
        break;

      default:
        response = await this.chatOpenAICompatible(props);
        break;
    }

    const message = response.choices[0]?.message;

    if (!message) {
      return response;
    }

    /*
     * Media is generated independently.
     *
     * This means:
     *
     * Chat = Anthropic
     * Image = OpenAI
     * Video = xAI
     * Audio = Groq
     *
     * can all be used in one call.
     */

    if (props.options?.image) {
      message.image = await this.generateImage(props);
    }

    if (props.options?.video) {
      message.video = await this.generateVideo(props);
    }

    if (props.options?.audio) {
      message.audio = await this.generateSpeech(props);
    }

    return response;
  }

  /* ============================================================
   * OPENAI-COMPATIBLE CHAT
   * ============================================================ */

  private async chatOpenAICompatible(props: UsingOpenAIProps): Promise<ChatCompletion> {
    const cfg = this.getProvider(props.provider);

    const key = this.requireApiKey(props.provider, props.options?.apiKey);

    const model = props.options?.model || props.model;

    if (!cfg.api.chat) {
      throw new Error(`[${props.provider}] Chat is not supported`);
    }

    let url = `${cfg.api.baseURL}${cfg.api.chat}`;

    let headers: Record<string, string>;

    /*
     * Google's OpenAI compatibility endpoint.
     */
    if (props.provider === "Google") {
      headers = this.bearerHeaders(key);
    } else {
      headers = this.bearerHeaders(key);
    }

    headers = {
      ...headers,
      ...(props.options?.headers || {}),
    };

    const body: any = {
      model,

      messages: [
        ...(props.system_prompt
          ? [
              {
                role: "system",
                content: props.system_prompt,
              },
            ]
          : []),

        {
          role: "user",
          content: props.user_prompt,
        },
      ],

      stream: Boolean(props.stream),

      max_tokens: props.options?.max_tokens ?? 4000,
    };

    /*
     * Newer reasoning models from some providers may reject
     * temperature/top_p.
     *
     * Only send these if caller explicitly gave them.
     */
    if (props.options?.temperature !== undefined) {
      body.temperature = props.options.temperature;
    }

    if (props.options?.top_p !== undefined) {
      body.top_p = props.options.top_p;
    }

    const response = await this.request(
      url,
      {
        method: "POST",

        headers,

        body: JSON.stringify(body),
      },

      props.options?.timeout ?? 90_000,
    );

    await this.ensureOK(response);

    if (props.stream) {
      return await this.consumeOpenAIStream(response, props.provider, model);
    }

    const json: any = await response.json();

    const content = json?.choices?.[0]?.message?.content ?? "";

    const result = this.makeCompletion(props.provider, model, content, json);

    result.id = json.id || result.id;

    result.created = json.created || result.created;

    result.usage = json.usage;

    result.choices[0].finish_reason = json?.choices?.[0]?.finish_reason ?? "stop";

    return result;
  }

  private async consumeOpenAIStream(
    response: Response,
    provider: AIProvider,
    model: string,
  ): Promise<ChatCompletion> {
    if (!response.body) {
      throw new Error("Provider returned an empty stream");
    }

    const reader = response.body.getReader();

    const decoder = new TextDecoder();

    let pending = "";
    let content = "";

    let id: string | undefined;
    let usage: any;

    while (true) {
      const { done, value } = await reader.read();

      if (done) break;

      pending += decoder.decode(value, {
        stream: true,
      });

      const lines = pending.split("\n");

      pending = lines.pop() || "";

      for (let line of lines) {
        line = line.trim();

        if (!line.startsWith("data:")) {
          continue;
        }

        const raw = line.slice(5).trim();

        if (!raw || raw === "[DONE]") {
          continue;
        }

        try {
          const event = JSON.parse(raw);

          id ||= event?.id;

          usage = event?.usage || usage;

          const delta = event?.choices?.[0]?.delta?.content;

          if (typeof delta === "string") {
            content += delta;
          }
        } catch {
          // ignore malformed SSE line
        }
      }
    }

    const out = this.makeCompletion(provider, model, content);

    if (id) out.id = id;

    if (usage) {
      out.usage = usage;
    }

    return out;
  }

  /* ============================================================
   * ANTHROPIC
   * ============================================================ */

  private async chatAnthropic(props: UsingOpenAIProps): Promise<ChatCompletion> {
    const cfg = this.getProvider("Anthropic");

    const key = this.requireApiKey("Anthropic", props.options?.apiKey);

    const model = props.options?.model || props.model;

    const body: any = {
      model,

      max_tokens: props.options?.max_tokens ?? 4000,

      messages: [
        {
          role: "user",
          content: props.user_prompt,
        },
      ],
    };

    if (props.system_prompt) {
      body.system = props.system_prompt;
    }

    /*
     * Don't send temperature by default.
     * Some new Claude model families reject old sampling parameters.
     */
    if (props.options?.temperature !== undefined) {
      body.temperature = props.options.temperature;
    }

    const response = await this.request(
      `${cfg.api.baseURL}${cfg.api.chat}`,
      {
        method: "POST",

        headers: {
          "x-api-key": key,
          "anthropic-version": "2023-06-01",
          "Content-Type": "application/json",
        },

        body: JSON.stringify(body),
      },

      props.options?.timeout ?? 90_000,
    );

    await this.ensureOK(response);

    const json: any = await response.json();

    const content = (json?.content || [])
      .filter((part: any) => part?.type === "text")
      .map((part: any) => part.text || "")
      .join("");

    const out = this.makeCompletion("Anthropic", model, content, json);

    out.usage = {
      prompt_tokens: json?.usage?.input_tokens ?? 0,

      completion_tokens: json?.usage?.output_tokens ?? 0,

      total_tokens: (json?.usage?.input_tokens ?? 0) + (json?.usage?.output_tokens ?? 0),
    };

    return out;
  }

  /* ============================================================
   * COHERE
   * ============================================================ */

  private async chatCohere(props: UsingOpenAIProps): Promise<ChatCompletion> {
    const cfg = this.getProvider("Cohere");

    const key = this.requireApiKey("Cohere", props.options?.apiKey);

    const model = props.options?.model || props.model;

    const body: any = {
      model,

      messages: [
        ...(props.system_prompt
          ? [
              {
                role: "system",
                content: props.system_prompt,
              },
            ]
          : []),

        {
          role: "user",
          content: props.user_prompt,
        },
      ],
    };

    if (props.options?.temperature !== undefined) {
      body.temperature = props.options.temperature;
    }

    if (props.options?.max_tokens !== undefined) {
      body.max_tokens = props.options.max_tokens;
    }

    const response = await this.request(`${cfg.api.baseURL}${cfg.api.chat}`, {
      method: "POST",

      headers: this.bearerHeaders(key),

      body: JSON.stringify(body),
    });

    await this.ensureOK(response);

    const json: any = await response.json();

    const content =
      json?.message?.content
        ?.filter((x: any) => x.type === "text")
        ?.map((x: any) => x.text || "")
        ?.join("") ||
      json?.text ||
      "";

    return this.makeCompletion("Cohere", model, content, json);
  }

  /* ============================================================
   * OLLAMA
   * ============================================================ */

  private async chatOllama(props: UsingOpenAIProps): Promise<ChatCompletion> {
    const cfg = this.getProvider("Ollama");

    const model = props.options?.model || props.model;

    const response = await this.request(`${cfg.api.baseURL}/api/chat`, {
      method: "POST",

      headers: {
        "Content-Type": "application/json",
      },

      body: JSON.stringify({
        model,

        stream: Boolean(props.stream),
        keep_alive: ollamaKeepAlive(),

        messages: [
          ...(props.system_prompt
            ? [
                {
                  role: "system",
                  content: props.system_prompt,
                },
              ]
            : []),

          {
            role: "user",
            content: props.user_prompt,
          },
        ],

        options: {
          ...(props.options?.temperature !== undefined
            ? {
                temperature: props.options.temperature,
              }
            : {}),

          ...(props.options?.top_p !== undefined
            ? {
                top_p: props.options.top_p,
              }
            : {}),
        },
      }),
    });

    await this.ensureOK(response);

    if (!props.stream) {
      const json: any = await response.json();

      return this.makeCompletion("Ollama", model, json?.message?.content || "", json);
    }

    if (!response.body) {
      throw new Error("Ollama returned empty stream");
    }

    const reader = response.body.getReader();

    const decoder = new TextDecoder();

    let pending = "";
    let content = "";

    while (true) {
      const { done, value } = await reader.read();

      if (done) break;

      pending += decoder.decode(value, {
        stream: true,
      });

      const lines = pending.split("\n");

      pending = lines.pop() || "";

      for (const line of lines) {
        if (!line.trim()) continue;

        try {
          const event = JSON.parse(line);

          content += event?.message?.content || "";
        } catch {}
      }
    }

    return this.makeCompletion("Ollama", model, content);
  }

  /* ============================================================
   * IMAGE GENERATION
   * ============================================================ */

  public async generateImage(props: UsingOpenAIProps): Promise<GeneratedImage | undefined> {
    const provider = props.options?.image_provider;

    if (!provider) {
      throw new Error("options.image=true but image_provider is missing");
    }

    const prompt = props.options?.image_prompt || props.user_prompt;

    switch (provider) {
      case "OpenAI":
        return this.generateOpenAIImage(props, prompt);

      case "xAI":
        return this.generateXAIImage(props, prompt);

      case "Together":
        return this.generateTogetherImage(props, prompt);

      case "Google":
        return this.generateGoogleImage(props, prompt);

      case "Stability":
        return this.generateStabilityImage(props, prompt);

      case "HuggingFace":
        return this.generateHuggingFaceImage(props, prompt);
    }
  }

  private async generateOpenAIImage(
    props: UsingOpenAIProps,
    prompt: string,
  ): Promise<GeneratedImage> {
    const key = this.requireApiKey("OpenAI", props.options?.image_apiKey);

    const model = props.options?.image_model || "gpt-image-2";

    const response = await this.request(
      "https://api.openai.com/v1/images/generations",
      {
        method: "POST",

        headers: this.bearerHeaders(key),

        body: JSON.stringify({
          model,
          prompt,

          ...(props.options?.image_size
            ? {
                size: props.options.image_size,
              }
            : {}),

          response_format: "b64_json",
        }),
      },
      180_000,
    );

    await this.ensureOK(response);

    const json: any = await response.json();

    const image = json?.data?.[0];

    if (!image) {
      throw new Error("OpenAI returned no image");
    }

    return {
      url: image.url,

      b64json: image.b64_json,

      mime: "image/png",
    };
  }

  private async generateXAIImage(props: UsingOpenAIProps, prompt: string): Promise<GeneratedImage> {
    const key = this.requireApiKey("xAI", props.options?.image_apiKey);

    const model = props.options?.image_model || "grok-imagine-image-2.0";

    const response = await this.request(
      "https://api.x.ai/v1/images/generations",
      {
        method: "POST",

        headers: this.bearerHeaders(key),

        body: JSON.stringify({
          model,
          prompt,

          response_format: "url",
        }),
      },
      180_000,
    );

    await this.ensureOK(response);

    const json: any = await response.json();

    const image = json?.data?.[0];

    if (!image) {
      throw new Error("xAI returned no image");
    }

    return {
      url: image.url,

      b64json: image.b64_json,

      mime: "image/png",
    };
  }

  private async generateTogetherImage(
    props: UsingOpenAIProps,
    prompt: string,
  ): Promise<GeneratedImage> {
    const key = this.requireApiKey("Together", props.options?.image_apiKey);

    const model = props.options?.image_model;

    if (!model) {
      throw new Error("Together image generation requires options.image_model");
    }

    const response = await this.request(
      "https://api.together.xyz/v1/images/generations",
      {
        method: "POST",

        headers: this.bearerHeaders(key),

        body: JSON.stringify({
          model,
          prompt,

          ...(props.options?.image_size
            ? {
                size: props.options.image_size,
              }
            : {}),
        }),
      },
      180_000,
    );

    await this.ensureOK(response);

    const json: any = await response.json();

    const image = json?.data?.[0];

    if (!image) {
      throw new Error("Together returned no image");
    }

    return {
      url: image.url,

      b64json: image.b64_json,

      mime: "image/png",
    };
  }

  private async generateGoogleImage(
    props: UsingOpenAIProps,
    prompt: string,
  ): Promise<GeneratedImage> {
    const key = this.requireApiKey("Google", props.options?.image_apiKey);

    const model = props.options?.image_model || "gemini-3.1-flash-image";

    const aspectRatio = props.options?.image_aspectRatio || "1:1";

    const imageSize = props.options?.image_resolution || "1K";

    const response = await this.request(
      `https://generativelanguage.googleapis.com/v1/models/${encodeURIComponent(model)}:generateContent`,
      {
        method: "POST",

        headers: {
          "x-goog-api-key": key,
          "Content-Type": "application/json",
        },

        body: JSON.stringify({
          contents: [
            {
              role: "user",

              parts: [
                {
                  text: prompt,
                },
              ],
            },
          ],

          generationConfig: {
            responseModalities: ["IMAGE"],

            responseFormat: {
              image: {
                aspectRatio,
                imageSize,
              },
            },
          },
        }),
      },
      180_000,
    );

    await this.ensureOK(response);

    const json: any = await response.json();

    const parts = json?.candidates?.[0]?.content?.parts || [];

    for (const part of parts) {
      const inline = part?.inlineData || part?.inline_data;

      if (inline?.data) {
        return {
          b64json: inline.data,

          mime: inline.mimeType || inline.mime_type || "image/png",
        };
      }
    }

    throw new Error("Google returned no image data");
  }

  private async generateStabilityImage(
    props: UsingOpenAIProps,
    prompt: string,
  ): Promise<GeneratedImage> {
    const key = this.requireApiKey("Stability", props.options?.image_apiKey);

    const form = new FormData();

    form.set("prompt", prompt);

    form.set("output_format", "png");

    const response = await this.request(
      "https://api.stability.ai/v2beta/stable-image/generate/core",
      {
        method: "POST",

        headers: {
          Authorization: `Bearer ${key}`,

          Accept: "image/png",
        },

        body: form,
      },

      180_000,
    );

    await this.ensureOK(response);

    const bytes = await response.arrayBuffer();

    return {
      b64json: this.toBase64(bytes),

      mime: response.headers.get("content-type") || "image/png",
    };
  }

  private async generateHuggingFaceImage(
    props: UsingOpenAIProps,
    prompt: string,
  ): Promise<GeneratedImage> {
    const key = this.requireApiKey("HuggingFace", props.options?.image_apiKey);

    const model = props.options?.image_model;

    if (!model) {
      throw new Error("HuggingFace image generation requires options.image_model");
    }

    const response = await this.request(
      `https://router.huggingface.co/hf-inference/models/${encodeURIComponent(model)}`,
      {
        method: "POST",

        headers: {
          Authorization: `Bearer ${key}`,

          "Content-Type": "application/json",
        },

        body: JSON.stringify({
          inputs: prompt,
        }),
      },

      180_000,
    );

    await this.ensureOK(response);

    const mime = response.headers.get("content-type") || "image/png";

    /*
     * Some HF endpoints return JSON,
     * image models usually return binary.
     */
    if (mime.includes("application/json")) {
      const json: any = await response.json();

      if (json?.url) {
        return {
          url: json.url,

          mime: "image/png",
        };
      }

      throw new Error(`HuggingFace returned JSON instead of image: ${JSON.stringify(json)}`);
    }

    const bytes = await response.arrayBuffer();

    return {
      b64json: this.toBase64(bytes),

      mime,
    };
  }

  /* ============================================================
   * VIDEO GENERATION
   * ============================================================ */

  public async generateVideo(props: UsingOpenAIProps): Promise<GeneratedVideo | undefined> {
    const provider = props.options?.video_provider || "Google";

    switch (provider) {
      case "Google":
        return this.generateGoogleVideo(props);

      case "xAI":
        return this.generateXAIVideo(props);

      case "OpenRouter":
        return this.generateOpenRouterVideo(props);
    }
  }

  /* ---------------- GOOGLE VEO ---------------- */

  private async generateGoogleVideo(props: UsingOpenAIProps): Promise<GeneratedVideo> {
    const key = this.requireApiKey("Google", props.options?.video_apiKey);

    const model = props.options?.video_model || "veo-3.1-generate-preview";

    const prompt = props.options?.video_prompt || props.user_prompt;

    const aspectRatio = props.options?.video_aspectRatio || "16:9";

    const startResponse = await this.request(
      `https://generativelanguage.googleapis.com/v1beta/models/${encodeURIComponent(model)}:predictLongRunning`,
      {
        method: "POST",

        headers: {
          "x-goog-api-key": key,

          "Content-Type": "application/json",
        },

        body: JSON.stringify({
          instances: [
            {
              prompt,
            },
          ],

          parameters: {
            aspectRatio,
          },
        }),
      },

      180_000,
    );

    await this.ensureOK(startResponse);

    const start: any = await startResponse.json();

    const operationName = start?.name;

    if (!operationName) {
      throw new Error(`Google returned no operation name: ${JSON.stringify(start)}`);
    }

    const waitSeconds = props.options?.video_waitSeconds ?? 300;

    const started = Date.now();

    while (true) {
      if (Date.now() - started > waitSeconds * 1000) {
        return {
          url: operationName,

          mime: "application/json",

          request_id: operationName,
        };
      }

      await this.sleep(5000);

      const statusResponse = await this.request(
        `https://generativelanguage.googleapis.com/v1beta/${operationName}`,
        {
          headers: {
            "x-goog-api-key": key,
          },
        },
      );

      await this.ensureOK(statusResponse);

      const status: any = await statusResponse.json();

      if (!status?.done) {
        continue;
      }

      if (status?.error) {
        throw new Error(`Google Veo error: ${JSON.stringify(status.error)}`);
      }

      const video =
        status?.response?.generateVideoResponse?.generatedSamples?.[0]?.video ||
        status?.response?.generatedVideos?.[0]?.video;

      const uri = video?.uri;

      if (!uri) {
        throw new Error(`Google Veo completed without video URI: ${JSON.stringify(status)}`);
      }

      return {
        url: uri,
        mime: "video/mp4",
        request_id: operationName,
      };
    }
  }

  /* ---------------- XAI VIDEO ---------------- */

  private async generateXAIVideo(props: UsingOpenAIProps): Promise<GeneratedVideo> {
    const key = this.requireApiKey("xAI", props.options?.video_apiKey);

    const prompt = props.options?.video_prompt || props.user_prompt;

    const model = props.options?.video_model || "grok-imagine-video-1.5";

    const body: any = {
      model,
      prompt,
    };

    if (props.options?.video_duration !== undefined) {
      body.duration = props.options.video_duration;
    }

    if (props.options?.video_aspectRatio) {
      body.aspect_ratio = props.options.video_aspectRatio;
    }

    if (props.options?.video_resolution) {
      body.resolution = props.options.video_resolution;
    }

    const response = await this.request(
      "https://api.x.ai/v1/videos/generations",
      {
        method: "POST",

        headers: this.bearerHeaders(key),

        body: JSON.stringify(body),
      },

      180_000,
    );

    await this.ensureOK(response);

    const startedJob: any = await response.json();

    const requestId = startedJob?.request_id;

    if (!requestId) {
      throw new Error(`xAI returned no request_id: ${JSON.stringify(startedJob)}`);
    }

    const waitSeconds = props.options?.video_waitSeconds ?? 300;

    const started = Date.now();

    while (true) {
      if (Date.now() - started > waitSeconds * 1000) {
        return {
          request_id: requestId,

          mime: "application/json",
        };
      }

      await this.sleep(5000);

      const check = await this.request(
        `https://api.x.ai/v1/videos/${encodeURIComponent(requestId)}`,
        {
          headers: {
            Authorization: `Bearer ${key}`,
          },
        },
      );

      await this.ensureOK(check);

      const result: any = await check.json();

      if (result.status === "done") {
        return {
          url: result?.video?.url,

          mime: "video/mp4",

          request_id: requestId,
        };
      }

      if (result.status === "failed" || result.status === "expired") {
        throw new Error(`xAI video ${result.status}: ${JSON.stringify(result)}`);
      }
    }
  }

  /* ---------------- OPENROUTER VIDEO ---------------- */

  private async generateOpenRouterVideo(props: UsingOpenAIProps): Promise<GeneratedVideo> {
    const key = this.requireApiKey("OpenRouter", props.options?.video_apiKey);

    const model = props.options?.video_model;

    if (!model) {
      throw new Error("OpenRouter video generation requires options.video_model");
    }

    const prompt = props.options?.video_prompt || props.user_prompt;

    const body: any = {
      model,
      prompt,
    };

    if (props.options?.video_aspectRatio) {
      body.aspect_ratio = props.options.video_aspectRatio;
    }

    if (props.options?.video_duration) {
      body.duration = props.options.video_duration;
    }

    if (props.options?.video_resolution) {
      body.resolution = props.options.video_resolution;
    }

    const response = await this.request(
      "https://openrouter.ai/api/v1/videos",
      {
        method: "POST",

        headers: this.bearerHeaders(key),

        body: JSON.stringify(body),
      },

      180_000,
    );

    await this.ensureOK(response);

    const json: any = await response.json();

    /*
     * API is asynchronous.
     * Return job/request ID.
     *
     * Polling schemas can differ between model/provider
     * integrations, therefore don't pretend there is one
     * universal output shape here.
     */
    return {
      url: json?.url,

      request_id: json?.id || json?.request_id,

      mime: json?.url ? "video/mp4" : "application/json",
    };
  }

  /* ============================================================
   * TEXT TO SPEECH
   * ============================================================ */

  public async generateSpeech(props: UsingOpenAIProps): Promise<GeneratedAudio> {
    const provider = props.options?.audio_provider || "OpenAI";

    switch (provider) {
      case "OpenAI":
        return this.generateOpenAISpeech(props);

      case "Groq":
        return this.generateGroqSpeech(props);
    }
  }

  private async generateOpenAISpeech(props: UsingOpenAIProps): Promise<GeneratedAudio> {
    const key = this.requireApiKey("OpenAI", props.options?.audio_apiKey);

    const model = props.options?.audio_model || "gpt-4o-mini-tts";

    const text = props.options?.audio_prompt || props.user_prompt;

    const format = props.options?.audio_format || "mp3";

    const response = await this.request(
      "https://api.openai.com/v1/audio/speech",
      {
        method: "POST",

        headers: this.bearerHeaders(key),

        body: JSON.stringify({
          model,

          input: text,

          voice: props.options?.audio_voice || "alloy",

          response_format: format,

          ...(props.options?.audio_speed !== undefined
            ? {
                speed: props.options.audio_speed,
              }
            : {}),
        }),
      },

      180_000,
    );

    await this.ensureOK(response);

    const bytes = await response.arrayBuffer();

    return {
      bytes_b64: this.toBase64(bytes),

      mime: this.mimeForAudioFormat(format),
    };
  }

  private async generateGroqSpeech(props: UsingOpenAIProps): Promise<GeneratedAudio> {
    const key = this.requireApiKey("Groq", props.options?.audio_apiKey);

    const model = props.options?.audio_model || "canopylabs/orpheus-v1-english";

    const format = props.options?.audio_format || "wav";

    const response = await this.request(
      "https://api.groq.com/openai/v1/audio/speech",
      {
        method: "POST",

        headers: this.bearerHeaders(key),

        body: JSON.stringify({
          model,

          input: props.options?.audio_prompt || props.user_prompt,

          voice: props.options?.audio_voice || "autumn",

          response_format: format,

          ...(props.options?.audio_speed !== undefined
            ? {
                speed: props.options.audio_speed,
              }
            : {}),
        }),
      },

      180_000,
    );

    await this.ensureOK(response);

    const bytes = await response.arrayBuffer();

    return {
      bytes_b64: this.toBase64(bytes),

      mime: this.mimeForAudioFormat(format),
    };
  }

  private mimeForAudioFormat(format: string) {
    switch (format) {
      case "wav":
        return "audio/wav";

      case "ogg":
        return "audio/ogg";

      case "opus":
        return "audio/opus";

      case "aac":
        return "audio/aac";

      case "flac":
        return "audio/flac";

      default:
        return "audio/mpeg";
    }
  }

  /* ============================================================
   * SPEECH TO TEXT
   * ============================================================ */

  public async transcribeAudio(options: {
    provider: "OpenAI" | "Groq";

    model?: string;

    apiKey?: string;

    file: Blob | Uint8Array | ArrayBuffer;

    filename?: string;

    mime?: string;

    language?: string;

    prompt?: string;
  }): Promise<{
    text: string;
    raw: any;
  }> {
    const provider = options.provider;

    const key = this.requireApiKey(provider, options.apiKey);

    const model =
      options.model || (provider === "Groq" ? "whisper-large-v3-turbo" : "gpt-transcribe");

    const form = new FormData();

    let blob: Blob;

    if (options.file instanceof Blob) {
      blob = options.file;
    } else {
      blob = new Blob(
        [options.file instanceof Uint8Array ? options.file : new Uint8Array(options.file)],

        {
          type: options.mime || "audio/wav",
        },
      );
    }

    form.set("file", blob, options.filename || "audio.wav");

    form.set("model", model);

    if (options.language) {
      form.set("language", options.language);
    }

    if (options.prompt) {
      form.set("prompt", options.prompt);
    }

    const url =
      provider === "Groq"
        ? "https://api.groq.com/openai/v1/audio/transcriptions"
        : "https://api.openai.com/v1/audio/transcriptions";

    const response = await this.request(
      url,
      {
        method: "POST",

        headers: {
          Authorization: `Bearer ${key}`,
        },

        body: form,
      },

      180_000,
    );

    await this.ensureOK(response);

    const json: any = await response.json();

    return {
      text: json?.text || "",

      raw: json,
    };
  }
}

/* ============================================================
 * DEFAULT INSTANCE + LEGACY FUNCTION EXPORTS
 * ============================================================ */

/**
 * New code should normally use `new AIClient()` or this singleton.
 * Old function imports continue to work through the wrappers below.
 */
export const aiClient = new AIClient();

/** @deprecated Prefer `aiClient.chat()` or `new AIClient().chat()`. */
export const usingOpenAI = (props: UsingOpenAIProps): Promise<ChatCompletion> =>
  aiClient.usingOpenAI(props);

/** @deprecated Prefer `aiClient.listModels()`. */
export const listModels = (provider: AIProvider, options?: { apiKey?: string }): Promise<any[]> =>
  aiClient.listModels(provider, options);

/** @deprecated Prefer `aiClient.getFeaturedModels()`. */
export const getFeaturedModels = (provider?: AIProvider): ModelInfo[] =>
  aiClient.getFeaturedModels(provider);

/** @deprecated Prefer `aiClient.getProvidersByCategory()`. */
export const getProvidersByCategory = (category: AIModality): ProviderInfo[] =>
  aiClient.getProvidersByCategory(category);

/** Legacy convenience wrapper for the unified pricing method. */
export const getPrice = (
  provider: AIProvider,
  model: string,
  modality: PricingModality,
  request: string | PriceRequest = {},
): Promise<PriceQuote> => aiClient.getPrice(provider, model, modality, request);

/** @deprecated Prefer `aiClient.generateImage()`. */
export const generateImage = (props: UsingOpenAIProps): Promise<GeneratedImage | undefined> =>
  aiClient.generateImage(props);

/** @deprecated Prefer `aiClient.generateVideo()`. */
export const generateVideo = (props: UsingOpenAIProps): Promise<GeneratedVideo | undefined> =>
  aiClient.generateVideo(props);

/** @deprecated Prefer `aiClient.generateSpeech()`. */
export const generateSpeech = (props: UsingOpenAIProps): Promise<GeneratedAudio> =>
  aiClient.generateSpeech(props);

/** @deprecated Prefer `aiClient.transcribeAudio()`. */
export const transcribeAudio = (
  options: Parameters<AIClient["transcribeAudio"]>[0],
): ReturnType<AIClient["transcribeAudio"]> => aiClient.transcribeAudio(options);
