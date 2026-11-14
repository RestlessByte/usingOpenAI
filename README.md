# usingOpenAI — Universal AI Client for Bun

**English** | [Русский](#русский) · [English](#english)

A TypeScript AI client for **Bun** that unifies cloud AI providers, local Ollama, and the Hermes proxy through native `fetch()` — without vendor AI SDKs.

> **Project status:** source-based documentation. Available models and pricing are defined in the repository and may differ from the providers' live APIs. Test the integrations you intend to use.

---

<a id="русский"></a>
## 🇷🇺 Русский

### Возможности

- **16 провайдеров:** OpenAI, Anthropic, Google, xAI, MistralAI, DeepSeek, OpenRouter, Fireworks, Together, Groq, Perplexity, Cohere, HuggingFace, Stability, Hermes и Ollama.
- **Без SDK провайдеров:** HTTP-запросы через нативный `fetch()`; не требуются `openai`, `@anthropic-ai/sdk`, `@google/genai` или пакет `ollama`.
- **Единый интерфейс:** `AIClient.chat()` / `complete()` и унифицированный формат ответа для чата.
- **Мультимодальность:** отдельные методы генерации изображений, видео, синтеза и распознавания речи; возможность запрашивать медиа вместе с чатом.
- **Модели:** реестр избранных моделей и запрос доступного каталога через `listModels()` там, где это поддерживается.
- **Стоимость:** `getPrice()` рассчитывает себестоимость, буфер, цену продажи и маржу на основе доступных тарифов.
- **Локальные модели:** Ollama и прокси Hermes.

> Поддержка конкретной модальности зависит от провайдера и его API. Наличие модели в статическом реестре **не гарантирует** доступ к ней в аккаунте.

### Требования и подключение

- Bun и TypeScript.
- API-ключ нужного облачного провайдера либо запущенный локальный сервер Ollama / Hermes.
- Исходный файл `index.ts` и необходимые внутренние зависимости проекта.

**Важно:** текущий `index.ts` импортирует `ollamaKeepAlive` из `../../services/ai/ollama-runtime`. Если переносишь клиент в другой репозиторий, сначала адаптируй или реализуй этот внутренний модуль. Это не опубликованный автономный npm-пакет; команды `npm install usingOpenAI` здесь нет.

Клонирование исходников:

```bash
git clone https://github.com/RestlessByte/usingOpenAI.git
cd usingOpenAI
```

Bun читает `.env` автоматически. Например:

```dotenv
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
GEMINI_API_KEY=your_google_api_key
OLLAMA_BASE_URL=http://127.0.0.1:11434
HERMES_PROXY_BASE_URL=http://127.0.0.1:8645/v1
USING_OPENAI_DEBUG=0
```

Не коммить реальные ключи: добавь `.env` в `.gitignore`. Для других провайдеров используются соответствующие переменные, например `XAI_API_KEY`, `MISTRAL_API_KEY`, `DEEPSEEK_API_KEY`, `OPENROUTER_API_KEY`, `FIREWORKS_API_KEY`, `TOGETHER_API_KEY`, `GROQ_API_KEY`, `PERPLEXITY_API_KEY`, `COHERE_API_KEY`, `HUGGINGFACE_API_KEY`, `STABILITY_API_KEY`.

### Быстрый старт

Путь импорта подстрой под расположение файла в своём проекте.

```ts
import { AIClient } from "./index";

const ai = new AIClient();

const response = await ai.chat({
  provider: "OpenAI",
  model: "gpt-5.6-sol", // пример ID из реестра; проверь доступность
  system_prompt: "Ты полезный ассистент.",
  user_prompt: "Привет! Объясни, что такое Bun.",
  options: { temperature: 0.2 },
});

console.log(response.choices[0]?.message.content);
```

**Локальная Ollama:**

```ts
const response = await ai.chat({
  provider: "Ollama",
  model: "qwen2.5-coder:7b", // модель должна быть установлена локально
  user_prompt: "Напиши функцию сортировки на TypeScript.",
});

console.log(response.choices[0]?.message.content);
```

**Каталог моделей и возможности:**

```ts
const models = await ai.listModels("OpenAI");
const featured = ai.getFeaturedModels("Google");
const imageProviders = ai.getProvidersByCategory("image");
const hasVision = ai.supports("OpenAI", "vision");

console.log({ models, featured, imageProviders, hasVision });
```

**Чат + изображение от другого провайдера:**

```ts
const response = await ai.chat({
  provider: "Anthropic",
  model: "claude-sonnet-5",
  user_prompt: "Опиши идею футуристической серверной.",
  options: {
    image: true,
    image_provider: "OpenAI",
    image_model: "gpt-image-2",
    image_prompt: "Futuristic server room, clean architectural render",
  },
});

console.log(response.choices[0]?.message.content);
console.log(response.choices[0]?.message.image);
```

> Для такого запроса нужны действующие ключи **обоих** облачных провайдеров. Генерация медиа — дополнительный запрос, поэтому возможны отдельная стоимость и ошибки.

**Расчёт цены:**

```ts
const quote = await ai.getPrice("OpenAI", "gpt-5.6-sol", "text", {
  prompt: "Hello!",
  expectedOutputTokens: 500,
});

console.log(quote.providerCostUSD); // себестоимость API
console.log(quote.sellPriceUSD);     // рассчитанная цена продажи
console.log(quote.estimated);        // использованы ли оценки
```

Если тариф неизвестен, расчёт не должен выдавать его за бесплатный. Для локальной Ollama стоимость API равна нулю, но электричество и оборудование **не включены**.

### Основной API

| Метод | Назначение |
| --- | --- |
| `chat(props)` / `complete(props)` | Отправка запроса в чат |
| `usingOpenAI(props)` | Совместимый исторический метод |
| `listModels(provider, options?)` | Получение каталога моделей |
| `getFeaturedModels(provider?)` | Избранные модели из реестра |
| `getProvidersByCategory(category)` | Провайдеры указанной модальности |
| `supports(provider, modality)` | Проверка категории в реестре |
| `getProvider(provider)` | Конфигурация провайдера |
| `generateImage(props)` | Генерация изображения |
| `generateVideo(props)` | Генерация видео |
| `generateSpeech(props)` | Синтез речи |
| `transcribeAudio(options)` | Распознавание речи (OpenAI / Groq) |
| `getPrice(provider, model, modality, request)` | Расчёт тарифа и цены продажи |

Доступны также экспортированный экземпляр `aiClient`, типы TypeScript и обёртки для старых импортов. Указанные в исходниках модальности: `text`, `vision`, `image`, `video`, `audio`, `tts`, `stt`, `embedding`, `rerank`; это **не** означает наличие отдельного метода генерации для каждой из них.

### Безопасность и ограничения

- Ключи API храни в окружении, не в исходниках и не во frontend-бандле.
- Перед выставлением счетов проверяй живые тарифы: статические цены могут устареть.
- Учитывай ограничения моделей, квоты, задержки и тарификацию отдельных медиа-вызовов.
- В локальной сети защищай Hermes/Ollama от несанкционированного доступа; локальные значения по умолчанию не предназначены для публикации наружу.
- Проверь работоспособность нужных API самостоятельно: этот README описывает интерфейс исходников, а не результаты интеграционных тестов.

---

<a id="english"></a>
## 🇬🇧 English

### Features

- **16 providers:** OpenAI, Anthropic, Google, xAI, MistralAI, DeepSeek, OpenRouter, Fireworks, Together, Groq, Perplexity, Cohere, HuggingFace, Stability, Hermes, and Ollama.
- **No vendor AI SDKs:** native `fetch()` instead of `openai`, `@anthropic-ai/sdk`, `@google/genai`, or the Ollama package.
- **Unified interface:** `AIClient.chat()` / `complete()` with a common chat response shape.
- **Multimodal utilities:** image and video generation, text-to-speech, and speech-to-text; optional media requests alongside chat.
- **Model discovery:** featured-model registry and live `listModels()` where provider APIs support it.
- **Pricing:** `getPrice()` estimates provider costs, buffers, selling prices, and gross margins from available rates.
- **Local inference:** Ollama and the Hermes agent proxy.

> Actual capabilities depend on the selected provider and endpoint. Listing a model in the static registry **does not guarantee** account access or model availability.

### Requirements and setup

- Bun and TypeScript.
- An API key for the selected cloud provider, or a running local Ollama / Hermes server.
- `index.ts` plus the project's required internal dependencies.

**Important:** `index.ts` imports `ollamaKeepAlive` from `../../services/ai/ollama-runtime`. To use it outside its original project layout, adapt or provide that internal module. This README does not assume a published standalone npm package.

```bash
git clone https://github.com/RestlessByte/usingOpenAI.git
cd usingOpenAI
```

Bun loads `.env` files automatically:

```dotenv
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
GEMINI_API_KEY=your_google_api_key
OLLAMA_BASE_URL=http://127.0.0.1:11434
HERMES_PROXY_BASE_URL=http://127.0.0.1:8645/v1
USING_OPENAI_DEBUG=0
```

Add `.env` to `.gitignore` and never commit credentials. Other supported names include `XAI_API_KEY`, `MISTRAL_API_KEY`, `DEEPSEEK_API_KEY`, `OPENROUTER_API_KEY`, `FIREWORKS_API_KEY`, `TOGETHER_API_KEY`, `GROQ_API_KEY`, `PERPLEXITY_API_KEY`, `COHERE_API_KEY`, `HUGGINGFACE_API_KEY`, and `STABILITY_API_KEY`.

### Quick start

Adjust the import path to your project structure.

```ts
import { AIClient } from "./index";

const ai = new AIClient();

const response = await ai.chat({
  provider: "OpenAI",
  model: "gpt-5.6-sol", // example registry ID; verify availability
  system_prompt: "You are a helpful assistant.",
  user_prompt: "Hello! What is Bun?",
  options: { temperature: 0.2 },
});

console.log(response.choices[0]?.message.content);
```

**Local Ollama:**

```ts
const response = await ai.chat({
  provider: "Ollama",
  model: "qwen2.5-coder:7b", // must be installed locally
  user_prompt: "Write a TypeScript sorting function.",
});

console.log(response.choices[0]?.message.content);
```

**Discover models and capabilities:**

```ts
const models = await ai.listModels("OpenAI");
const featured = ai.getFeaturedModels("Google");
const imageProviders = ai.getProvidersByCategory("image");
const hasVision = ai.supports("OpenAI", "vision");

console.log({ models, featured, imageProviders, hasVision });
```

**Chat and generate an image using different providers:**

```ts
const response = await ai.chat({
  provider: "Anthropic",
  model: "claude-sonnet-5",
  user_prompt: "Describe a futuristic server room.",
  options: {
    image: true,
    image_provider: "OpenAI",
    image_model: "gpt-image-2",
    image_prompt: "Futuristic server room, clean architectural render",
  },
});

console.log(response.choices[0]?.message.content);
console.log(response.choices[0]?.message.image);
```

> This example requires valid credentials for **both** providers. Media generation makes an additional request and may incur a separate charge or failure.

**Pricing estimate:**

```ts
const quote = await ai.getPrice("OpenAI", "gpt-5.6-sol", "text", {
  prompt: "Hello!",
  expectedOutputTokens: 500,
});

console.log(quote.providerCostUSD); // API provider cost
console.log(quote.sellPriceUSD);     // calculated selling price
console.log(quote.estimated);        // whether estimates were used
```

Unknown pricing is not silently treated as free. Local Ollama has zero provider API cost, **excluding** hardware and electricity.

### Public API

| Method | Purpose |
| --- | --- |
| `chat(props)` / `complete(props)` | Chat completion |
| `usingOpenAI(props)` | Legacy-compatible chat method |
| `listModels(provider, options?)` | Fetch provider model catalog |
| `getFeaturedModels(provider?)` | Get registry featured models |
| `getProvidersByCategory(category)` | Filter providers by modality |
| `supports(provider, modality)` | Check registry capability |
| `getProvider(provider)` | Read provider configuration |
| `generateImage(props)` | Generate an image |
| `generateVideo(props)` | Generate a video |
| `generateSpeech(props)` | Synthesize speech |
| `transcribeAudio(options)` | Transcribe audio (OpenAI / Groq) |
| `getPrice(provider, model, modality, request)` | Estimate cost and selling price |

The module also exports the `aiClient` singleton, TypeScript types, and legacy function wrappers. Declared modalities are `text`, `vision`, `image`, `video`, `audio`, `tts`, `stt`, `embedding`, and `rerank`; **not all** have a dedicated generation method.

### Security and limitations

- Keep API secrets in environment variables, never in source control or client-side bundles.
- Recheck provider prices before billing customers; static price tables can become outdated.
- Account for model availability, quotas, latency, and separate multimodal API charges.
- Restrict network access to local Hermes/Ollama instances; loopback defaults are not a production security boundary.
- Validate your target integrations: this README documents the inspected source interface, **not** verified end-to-end test results.

---

**Repository:** [RestlessByte/usingOpenAI](https://github.com/RestlessByte/usingOpenAI)
**Author:** [@restlessbyte](https://github.com/RestlessByte)

*License: see the repository's LICENSE file if one is provided. No license is implied by this README.*
