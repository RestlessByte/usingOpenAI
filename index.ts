/** @param [string] For work you need download package manager BUN  - https://bun.sh */
/* ---------------------------EXAMPLE---------------------------
  const response = await usingOpenAI({
    system_prompt: 'Simple  write please word "Hi! How I can you help"',
    user_prompt: 'This test function',
    provider: 'Ollama',
    model: 'deepseek-r1:latest',
    stream: false,
    options: {
      // media is optional
      image: false,
      // image_provider: 'OpenAI' | 'Google' | 'Stability' | 'HuggingFace' | 'Together'
      // image_model: 'gpt-image-1' | 'imagen-3.0-generate-002' | 'stable-image-core' | ...
      // image_size: '1024x1024',

      video: false,
      // video_model: 'veo-3.0-generate-preview',
      // video_aspectRatio: '16:9'
    }
  })

  console.log(response?.choices[0].message.content)
  // If image/video requested and supported:
  // response?.choices[0].message.image  -> { b64json?: string; mime?: string; url?: string }
  // response?.choices[0].message.video  -> { url?: string; mime?: string; bytes_b64?: string }
*/// ---------------------------EXAMPLE---------------------------

import { InferenceClient } from '@huggingface/inference';
import { config } from 'dotenv';
import ollama from 'ollama';
import OpenAI from 'openai';

import type { ChatCompletionCreateParamsBase } from 'openai/resources/chat/completions.mjs';
import type { FireworksModelsALL } from './types/models/fireworks.ai/type';
import type { MistralAIModelsALL } from './types/models/mistral.ai/type';

config();

/** Extra SDKs for new providers */
import Anthropic from '@anthropic-ai/sdk';
import { GoogleGenAI } from '@google/genai'; // Gemini / Imagen / Veo 3
import { CohereClient } from 'cohere-ai'; // Cohere v2 chat

/** Helpers */
const HF = new InferenceClient();
const DEBUG = process.env.USING_OPENAI_DEBUG === '1';

const toBase64 = (data: ArrayBuffer | Uint8Array) =>
  Buffer.from(data instanceof Uint8Array ? data : new Uint8Array(data)).toString('base64');

const stripThink = (s: string) =>
  s.replace(/<think>[\s\S]*?<\/think>/g, '').trim().replace(/\n{2,}/g, '\n');

/** @param [string] Get types all AI models */
export type TypeModels = OpenAI.AllModels | MistralAIModelsALL | FireworksModelsALL | string;

type TypeProvider =
  | 'MistralAI'
  | 'OpenAI'
  | 'OpenRouter'
  | 'Fireworks'
  | 'Ollama'
  | 'HuggingFace'
  | 'DeepSeek'
  | 'Anthropic'
  | 'Google'      // Gemini (text), Imagen (images), Veo 3 (video)
  | 'xAI'         // Grok
  | 'Cohere'
  | 'Perplexity'
  | 'Together'
  | 'Groq'
  | 'Stability';

type MediaOut = {
  image?: { b64json?: string; mime?: string; url?: string };
  video?: { url?: string; mime?: string; bytes_b64?: string };
};

type UsingOptions = {
  model?: TypeModels | any;
  apiKey?: string;
  temperature?: number;
  max_tokens?: number;
  top_p?: number;

  /** Optional media toggles */
  image?: boolean;
  image_prompt?: string;
  image_model?: string;
  image_size?: '256x256' | '512x512' | '1024x1024' | string;
  image_provider?: 'OpenAI' | 'Google' | 'Stability' | 'HuggingFace' | 'Together';

  video?: boolean;
  video_prompt?: string;
  video_model?: string; // e.g., 'veo-3.0-generate-preview'
  video_aspectRatio?: '16:9' | '9:16' | '1:1' | string;
  video_personGeneration?: 'allow_all' | 'allow_adult' | 'dont_allow';
  video_waitSeconds?: number; // polling cap for Veo
};

export const usingOpenAI = async (
  props: {
    user_prompt: string,
    system_prompt: string,
    provider: TypeProvider,
    model: TypeModels,
    stream?: boolean,
    options?: UsingOptions
  }
): Promise<OpenAI.Chat.Completions.ChatCompletion | undefined> => {
  const cfg = { provider: process.env.PROVIDER, model: process.env.MODEL_NAME }
  let token: string | undefined;
  let baseURL: string | undefined = undefined;
  // --- Resolve API keys + baseURLs per provider ---
  if (props.provider === 'MistralAI') {
    token = props.options?.apiKey || process.env.MistralAI_API_KEY;
    baseURL = 'https://api.mistral.ai/v1';
  } else if (props.provider === 'OpenRouter') {
    token = props.options?.apiKey || process.env.OpenRouter_API_KEY;
    baseURL = 'https://openrouter.ai/api/v1';
  } else if (props.provider === 'Fireworks') {
    token = props.options?.apiKey || process.env.Fireworks_API_KEY;
    baseURL = 'https://api.fireworks.ai/inference/v1';
  } else if (props.provider === 'OpenAI') {
    token = props.options?.apiKey || process.env.OpenAI_API_KEY;
  } else if (props.provider === 'HuggingFace') {
    token = props.options?.apiKey || process.env.HuggingFace_API_KEY;
  } else if (props.provider === 'DeepSeek') {
    token = props.options?.apiKey || process.env.DeepSeek_API_KEY;
    baseURL = 'https://api.deepseek.com';
  } else if (props.provider === 'Anthropic') {
    token = props.options?.apiKey || process.env.ANTHROPIC_API_KEY;
  } else if (props.provider === 'Google') {
    // Gemini via OpenAI-compat for CHAT; Imagen/Veo via @google/genai below
    token = props.options?.apiKey || process.env.GEMINI_API_KEY || process.env.GOOGLE_API_KEY;
    baseURL = 'https://generativelanguage.googleapis.com/v1beta/openai/';
  } else if (props.provider === 'xAI') {
    token = props.options?.apiKey || process.env.XAI_API_KEY; // Grok
    baseURL = 'https://api.x.ai/v1';
  } else if (props.provider === 'Cohere') {
    token = props.options?.apiKey || process.env.COHERE_API_KEY;
  } else if (props.provider === 'Perplexity') {
    token = props.options?.apiKey || process.env.PERPLEXITY_API_KEY;
    baseURL = 'https://api.perplexity.ai';
  } else if (props.provider === 'Together') {
    token = props.options?.apiKey || process.env.TOGETHER_API_KEY;
    baseURL = 'https://api.together.xyz/v1';
  } else if (props.provider === 'Groq') {
    token = props.options?.apiKey || process.env.GROQ_API_KEY;
    baseURL = 'https://api.groq.com/openai/v1';
  } else if (props.provider === 'Stability') {
    token = props.options?.apiKey || process.env.STABILITY_API_KEY; // images only
  } else if (props.provider === 'Ollama') {
    token = 'Ollama';
    const response = await handleOllama(props);
    return await maybeAttachMedia(props, token, response);
  }

  // --- Token check ---
  if (!token) {
    throw new Error('Missing API token. Add to .env or provide in options.apiKey');
  }

  try {
    if (DEBUG) {
      // соответствие твоему старому стилю, но по умолчанию отключено
      // eslint-disable-next-line no-console
      console.log('Sending request to API', {
        model: props.options?.model || props.model,
        provider: props.provider,
        promptLength: props.user_prompt.length
      });
    }

    // --- Provider-specific (non OpenAI-compat) ---
    if (props.provider === 'HuggingFace') {
      const response = await handleHuggingFaceChat(props, token);
      return await maybeAttachMedia(props, token, response);
    }
    if (props.provider === 'Anthropic') {
      const response = await handleAnthropic(props, token);
      return await maybeAttachMedia(props, token, response);
    }
    if (props.provider === 'Cohere') {
      const response = await handleCohere(props, token);
      return await maybeAttachMedia(props, token, response);
    }
    if (props.provider === 'Stability') {
      // Image-first provider: не делаем чат, сразу минимальный ответ + (возможно) картинка
      const response = buildMinimalOAIResponse(props, '(Stability) See image.');
      return await maybeAttachMedia(props, token, response);
    }

    // --- OpenAI-compatible chat providers ---
    const openai = new OpenAI({
      apiKey: token || cfg.model,
      baseURL: baseURL || cfg.provider,
      timeout: 5000,
    });

    let response = await openai.chat.completions.create({
      model: (props.options?.model || props.model) as string,
      stream: Boolean(props.stream),
      messages: [
        { role: "system", content: `${props.system_prompt}\nProvider: ${props.provider}\nModel: ${props.model}` },
        { role: "user", content: props.user_prompt }
      ],
      top_p: props.options?.top_p,
      temperature: props.options?.temperature ?? 0.2,
      max_tokens: props.options?.max_tokens ?? 800,
    } as ChatCompletionCreateParamsBase);

    // Санация "reasoning"/"think"
    try {
      const m = response?.choices?.[0]?.message as any;
      if (m) {
        if (typeof m.content === 'string' && /<think>/.test(m.content)) {
          m.content = stripThink(m.content);
        }
        delete m.reasoning;
        delete m.thinking;
      }
    } catch { }

    if (DEBUG) {
      // eslint-disable-next-line no-console
      console.log('Received response', {
        id: (response as any).id,
        tokens: (response as any).usage
      });
    }

    return await maybeAttachMedia(props, token, response as any);

  } catch (error: any) {
    // Лаконичные ошибки без лишних логов (как в твоём стиле)
    if (error.status === 401) throw new Error('Invalid API key');
    if (error.status === 429) throw new Error('Rate limit exceeded');
    if (error.code === 'ETIMEDOUT') throw new Error('Request timed out');
    if (error.status >= 500) throw new Error(`Server error: ${error.status}`);
    throw new Error(`[${props.provider}] ${error.message || 'Unknown error'}`);
  }
};

/* --------------------------- Internal helpers --------------------------- */

async function handleOllama(props: any) {
  const response = await ollama.chat({
    model: props.options?.model || (props.model as string),
    messages: [
      { role: 'system', content: props.system_prompt },
      { role: 'user', content: props.user_prompt }
    ],
    stream: Boolean(props.stream)
  });

  let full = '';
  if (props.stream) {
    for await (const part of response) full += part.message.content ?? '';
  } else {
    full = response.message.content ?? '';
  }
  full = stripThink(full);

  const oaiLike = {
    id: 'ollama-' + Date.now(),
    object: 'chat.completion',
    created: Math.floor(Date.now() / 1000),
    model: props.options?.model || (props.model as string),
    choices: [{
      index: 0,
      message: { role: 'assistant', content: full } as any,
      finish_reason: 'stop'
    }],
    usage: {
      prompt_tokens: (props.system_prompt?.length || 0) + (props.user_prompt?.length || 0),
      completion_tokens: full.length,
      total_tokens: (props.system_prompt?.length || 0) + (props.user_prompt?.length || 0) + full.length
    }
  } as OpenAI.Chat.Completions.ChatCompletion as any;

  return oaiLike;
}

async function handleHuggingFaceChat(props: any, token: string) {
  const response = await HF.chatCompletion({
    model: props.options?.model || (props.model as string),
    messages: [
      { role: 'system', content: props.system_prompt },
      { role: 'user', content: props.user_prompt }
    ],
    temperature: props.options?.temperature,
    max_tokens: props.options?.max_tokens,
    top_p: props.options?.top_p
  });

  const content = response.choices?.[0]?.message?.content || '';
  const oaiLike = {
    id: 'hf-' + Date.now(),
    object: 'chat.completion',
    created: Math.floor(Date.now() / 1000),
    model: props.options?.model || (props.model as string),
    choices: [{
      index: 0,
      message: { role: 'assistant', content } as any,
      finish_reason: 'stop'
    }],
    usage: {
      prompt_tokens: response.usage?.prompt_tokens || 0,
      completion_tokens: response.usage?.completion_tokens || 0,
      total_tokens: response.usage?.total_tokens || 0
    }
  } as OpenAI.Chat.Completions.ChatCompletion as any;

  return oaiLike;
}

async function handleAnthropic(props: any, token: string) {
  const client = new Anthropic({ apiKey: token });
  const a = await client.messages.create({
    model: (props.options?.model || props.model) as string,
    max_tokens: props.options?.max_tokens ?? 800,
    temperature: props.options?.temperature ?? 0.2,
    system: props.system_prompt,
    messages: [{ role: 'user', content: props.user_prompt }]
  });
  const text = (a.content || [])
    .map((p: any) => (p?.type === 'text' ? p.text : ''))
    .join('');

  const oaiLike = {
    id: a.id || 'anthropic-' + Date.now(),
    object: 'chat.completion',
    created: Math.floor(Date.now() / 1000),
    model: (props.options?.model || props.model) as string,
    choices: [{
      index: 0,
      message: { role: 'assistant', content: text } as any,
      finish_reason: a.stop_reason || 'stop'
    }],
    usage: {
      prompt_tokens: (a.usage as any)?.input_tokens ?? 0,
      completion_tokens: (a.usage as any)?.output_tokens ?? 0,
      total_tokens: ((a.usage as any)?.input_tokens ?? 0) + ((a.usage as any)?.output_tokens ?? 0)
    }
  } as OpenAI.Chat.Completions.ChatCompletion as any;

  return oaiLike;
}

async function handleCohere(props: any, token: string) {
  const co = new CohereClient({ token });
  const chat = await co.chat({
    model: (props.options?.model || props.model) as string,
    temperature: props.options?.temperature ?? 0.2,
    messages: [
      { role: 'system', content: props.system_prompt },
      { role: 'user', content: props.user_prompt }
    ].map(m => ({ role: m.role, content: [{ type: 'text', text: m.content }] }))
  });

  const text = chat?.text ?? chat?.message?.content?.map((c: any) => c?.text).join('') ?? '';
  const oaiLike = {
    id: chat?.id || 'cohere-' + Date.now(),
    object: 'chat.completion',
    created: Math.floor(Date.now() / 1000),
    model: (props.options?.model || props.model) as string,
    choices: [{
      index: 0,
      message: { role: 'assistant', content: text } as any,
      finish_reason: 'stop'
    }],
    usage: {
      prompt_tokens: (chat?.meta as any)?.tokens?.input_tokens ?? 0,
      completion_tokens: (chat?.meta as any)?.tokens?.output_tokens ?? 0,
      total_tokens: ((chat?.meta as any)?.tokens?.input_tokens ?? 0) + ((chat?.meta as any)?.tokens?.output_tokens ?? 0)
    }
  } as OpenAI.Chat.Completions.ChatCompletion as any;

  return oaiLike;
}

function buildMinimalOAIResponse(props: any, content: string) {
  return {
    id: 'generic-' + Date.now(),
    object: 'chat.completion',
    created: Math.floor(Date.now() / 1000),
    model: (props.options?.model || props.model) as string,
    choices: [{
      index: 0,
      message: { role: 'assistant', content } as any,
      finish_reason: 'stop'
    }],
    usage: { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 }
  } as OpenAI.Chat.Completions.ChatCompletion as any;
}

/** Attach image/video if explicitly requested AND supported AND key exists.
 *  Если что-то не готово — тихо пропускаем без логов.
 */
async function maybeAttachMedia(
  props: any,
  _token: string | undefined,
  response: OpenAI.Chat.Completions.ChatCompletion & { choices: { message: any }[] }
) {
  const msg = response.choices?.[0]?.message as any;
  if (!msg) return response;

  // IMAGE
  if (props?.options?.image === true) {
    const image = await safeGenerateImage(props);
    if (image) msg.image = image;
  }

  // VIDEO
  if (props?.options?.video === true) {
    const video = await safeGenerateVideo(props);
    if (video) msg.video = video;
  }

  return response;
}

/* --------------------------- Safe media wrappers --------------------------- */

async function safeGenerateImage(props: any) {
  // Если провайдер не указан явно — НЕ пытаемся угадывать (чтобы не было случайного OpenAI без ключа)
  const provider: string | undefined = props.options?.image_provider;
  if (!provider) return undefined;

  // Проверяем ключи заранее
  const key =
    (provider === 'OpenAI' && (props.options?.apiKey || process.env.OpenAI_API_KEY)) ||
    (provider === 'Google' && (props.options?.apiKey || process.env.GEMINI_API_KEY || process.env.GOOGLE_API_KEY)) ||
    (provider === 'Stability' && (props.options?.apiKey || process.env.STABILITY_API_KEY)) ||
    (provider === 'HuggingFace' && (props.options?.apiKey || process.env.HuggingFace_API_KEY)) ||
    (provider === 'Together' && (props.options?.apiKey || process.env.TOGETHER_API_KEY)) ||
    '';

  if (!key) return undefined; // тихо выходим

  try {
    return await generateImage(props);
  } catch {
    return undefined; // тихо молчим
  }
}

async function safeGenerateVideo(props: any) {
  // На сегодня — только Veo 3 через Google
  const apiKey = props.options?.apiKey || process.env.GEMINI_API_KEY || process.env.GOOGLE_API_KEY;
  if (!apiKey) return undefined; // нет ключа — выходим
  try {
    return await generateVeoVideo(props);
  } catch {
    return undefined;
  }
}

/* --------------------------- Image generation --------------------------- */
async function generateImage(
  props: any
): Promise<MediaOut['image'] | undefined> {
  const imageProvider = props.options?.image_provider as UsingOptions['image_provider'];
  if (!imageProvider) return undefined;

  const prompt =
    props.options?.image_prompt
    || `Create an illustrative image matching this request: ${props.user_prompt}`;

  const imageModel = props.options?.image_model || (
    imageProvider === 'OpenAI' ? 'gpt-image-1'
      : imageProvider === 'Google' ? 'imagen-3.0-generate-002'
        : imageProvider === 'Stability' ? 'stable-image-core'
          : props.options?.model || props.model
  );

  const size = props.options?.image_size || '1024x1024';

  switch (imageProvider) {
    case 'OpenAI': {
      const key = props.options?.apiKey || process.env.OpenAI_API_KEY;
      if (!key) return undefined;
      const oai = new OpenAI({ apiKey: key });
      const r = await oai.images.generate({ model: imageModel, prompt, size } as any);
      const b64 = r?.data?.[0]?.b64_json;
      if (!b64) return undefined;
      return { b64json: b64, mime: 'image/png' };
    }

    case 'Together': {
      const key = props.options?.apiKey || process.env.TOGETHER_API_KEY;
      if (!key) return undefined;
      const together = new OpenAI({ apiKey: key, baseURL: 'https://api.together.xyz/v1' });
      const r = await together.images.generate({ model: imageModel, prompt, size } as any);
      const b64 = (r as any)?.data?.[0]?.b64_json;
      if (!b64) return undefined;
      return { b64json: b64, mime: 'image/png' };
    }

    case 'Google': {
      const key = props.options?.apiKey || process.env.GEMINI_API_KEY || process.env.GOOGLE_API_KEY;
      if (!key) return undefined;
      const g = new GoogleGenAI({ apiKey: key });
      const res = await g.models.generateImages({ model: imageModel, prompt } as any);
      const bytes = res?.generatedImages?.[0]?.image?.imageBytes as Uint8Array | undefined;
      if (!bytes) return undefined;
      return { b64json: toBase64(bytes), mime: 'image/png' };
    }

    case 'Stability': {
      const key = props.options?.apiKey || process.env.STABILITY_API_KEY;
      if (!key) return undefined;

      const resp = await fetch('https://api.stability.ai/v2beta/stable-image/generate/core', {
        method: 'POST',
        headers: { Authorization: `Bearer ${key}`, Accept: 'image/png' } as any,
        body: (() => {
          const fd = new FormData();
          fd.set('prompt', prompt);
          fd.set('output_format', 'png');
          return fd;
        })()
      });
      if (!resp.ok) return undefined;
      const buf = await resp.arrayBuffer();
      return { b64json: toBase64(buf), mime: 'image/png' };
    }

    case 'HuggingFace': {
      const key = props.options?.apiKey || process.env.HuggingFace_API_KEY;
      if (!key) return undefined;
      const blob = await HF.textToImage({
        model: imageModel, // e.g. "stabilityai/stable-diffusion-xl-base-1.0"
        inputs: prompt
      } as any);
      const ab = await blob.arrayBuffer();
      return { b64json: toBase64(ab), mime: blob.type || 'image/png' };
    }

    default:
      return undefined;
  }
}

/* --------------------------- Video (Veo 3) --------------------------- */
async function generateVeoVideo(props: any): Promise<MediaOut['video'] | undefined> {
  const apiKey = props.options?.apiKey || process.env.GEMINI_API_KEY || process.env.GOOGLE_API_KEY;
  if (!apiKey) return undefined;

  const prompt = props.options?.video_prompt || `Generate an 8s cinematic ${props.options?.video_aspectRatio || '16:9'} video: ${props.user_prompt}`;
  const model = props.options?.video_model || 'veo-3.0-generate-preview';
  const aspectRatio = props.options?.video_aspectRatio || '16:9';

  const g = new GoogleGenAI({ apiKey });
  let op = await g.models.generateVideos({ model, prompt, config: { aspectRatio } } as any);

  const maxWait = Math.max(10, Number(props.options?.video_waitSeconds || 120));
  const started = Date.now();
  while (!(op as any)?.done) {
    if ((Date.now() - started) / 1000 > maxWait) {
      return { url: (op as any)?.name || '', mime: 'application/json' };
    }
    await new Promise(r => setTimeout(r, 10_000));
    op = await g.operations.getVideosOperation({ operation: op as any } as any);
  }

  const file = (op as any)?.response?.generatedVideos?.[0]?.video;
  if (!file) return undefined;

  const dl = await g.files.download({ file, downloadPath: undefined } as any);
  const bytes = (dl as any)?.video?.videoBytes as Uint8Array | undefined;
  if (bytes && bytes.length) return { bytes_b64: toBase64(bytes), mime: 'video/mp4' };
  return { url: (file as any)?.uri || '', mime: 'video/mp4' };
}
