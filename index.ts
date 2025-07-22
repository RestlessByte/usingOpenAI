/** @param [string] For work you need download package manager BUN  - https://bun.sh */
/* ---------------------------EXAMPLE---------------------------
 import {usingOpenAI} from './usingOpenAI/index'
 const response = await usingOpenAI({
    system_prompt: 'Simple  write please word 'Hi! How I can you help'',
    user_prompt: 'This test function',
    provider: 'MistralAI',
    model: 'mistral-large-latest',
    stream: false
  })
  return response.choices[0].message.content -> Response: Hi! How I can you help? 
*/// ---------------------------EXAMPLE---------------------------
import { config } from 'dotenv';
import OpenAI from 'openai';
import { InferenceClient } from '@huggingface/inference'; // Раскомментировано
// Load environment variables
/** @param [string] 
 * List API Reference for using AI: 
 * - FireworksAI: https://fireworks.ai/docs/getting-started/introduction
 * MistralAI : https://docs.mistral.ai/
 * OpenAI: https://platform.openai.com/docs/api-reference/introduction
 */
/** @param [string] Get types all AI models */
export type TypeModels = OpenAI.AllModels | MistralAIModelsALL | FireworksModelsALL
type TypeProvider = 'MistralAI' | 'OpenAI' | 'OpenRouter' | 'Fireworks' | 'Ollama' | 'HuggingFace' | 'DeepSeek'
import ollama from 'ollama'
import type { ChatCompletionCreateParamsBase } from 'openai/resources/chat/completions.mjs';
import type { ChatCompletionUserMessageParam } from 'openai/resources.js';
import type { MistralAIModelsALL } from './types/models/mistral.ai/type';
import type { FireworksModelsALL } from './types/models/fireworks.ai/type';

config();
const HuggingFace = new InferenceClient(); // Инициализация клиента

/**
 * @params [example] -   await usingOpenAI({
    system_prompt: 'Напиши просто тест',
    user_prompt: 'rewqrdwdssdffsd',
    provider: 'MistralAI',
    model: 'mistral-large-latest',
    stream: false
  }).then(e => e?.choices[0].message.content)
 */
export const usingOpenAI = async (
  props: {
    user_prompt: string,
    system_prompt: string,
    provider: TypeProvider,
    model: TypeModels,
    stream?: boolean,
    options?: {
      model?: string
      apiKey?: string
      temperature?: number
      max_tokens?: number
      top_p?: number
    }
  }
): Promise<OpenAI.Chat.Completions.ChatCompletion | undefined> => {
  let token: string | undefined, message, client;
  let baseURL: string | undefined = undefined;

  // Получаем токены из .env или options
  if (props.provider === 'MistralAI') {
    token = props.options?.apiKey || process.env.MistralAI_API_KEY;
    baseURL = 'https://api.mistral.ai/v1';
  }
  else if (props.provider === 'OpenRouter') {
    token = props.options?.apiKey || process.env.OpenRouter_API_KEY;
    baseURL = 'https://openrouter.ai/api/v1';
  }
  else if (props.provider === 'Fireworks') {
    token = props.options?.apiKey || process.env.Fireworks_API_KEY;
    baseURL = 'https://api.fireworks.ai/inference/v1';
  }
  else if (props.provider === 'OpenAI') {

    token = props.options?.apiKey || process.env.OpenAI_API_KEY;
  }
  else if (props.provider === 'HuggingFace') {
    token = props.options?.apiKey || process.env.HuggingFace_API_KEY;
  }
  if (props.provider == 'DeepSeek') {
    token = process.env.DeepSeek_API_KEY
    baseURL = 'https://api.deepseek.com'
  }
  else if (props.provider === 'Ollama') {
    token = 'Ollama';
    const response = await ollama.chat({
      model: props.options?.model || props.model as string,
      messages: [
        { role: 'system', content: props.system_prompt },
        { role: 'user', content: props.user_prompt }
      ],
      stream: props.stream as boolean
    });

    let fullResponse = '';

    // Handle streaming response
    if (props.stream) {
      for await (const part of response) {
        fullResponse += part.message.content;  // Append each content chunk
      }
    }
    // Handle non-streaming response
    else {
      fullResponse = response.message.content;  // Directly get content
    }

    // Format response to OpenAI-compatible structure
    return {
      id: 'ollama-' + Date.now(),
      object: 'chat.completion',
      created: Math.floor(Date.now() / 1000),
      model: props.options?.model || props.model as string,
      choices: [{
        index: 0,
        message: { role: 'assistant', content: fullResponse },
        finish_reason: 'stop'
      }],
      usage: {
        prompt_tokens: props.system_prompt.length + props.user_prompt.length,
        completion_tokens: fullResponse.length,
        total_tokens: props.system_prompt.length + props.user_prompt.length + fullResponse.length
      }
    } as any;
  }

  // Универсальная проверка токена для всех провайдеров
  if (!token) {
    const errorMessage = 'Missing API token. Add to .env or provide in options.apiKey';
    console.error(errorMessage);
    throw new Error(errorMessage);
  }

  try {
    console.log('Sending request to API', {
      model: props.options?.model || props.model,
      provider: props.provider,
      promptLength: props.user_prompt.length
    });

    // Обработка HuggingFace
    if (props.provider === 'HuggingFace') {
      const response = await HuggingFace.chatCompletion({
        model: props.options?.model || props.model as string,
        messages: [
          { role: 'system', content: props.system_prompt },
          { role: 'user', content: props.user_prompt }
        ],
        temperature: props.options?.temperature,
        max_tokens: props.options?.max_tokens,
        top_p: props.options?.top_p
      });

      // Форматирование ответа под OpenAI-совместимый формат
      return {
        id: 'hf-' + Date.now(),
        object: 'chat.completion',
        created: Math.floor(Date.now() / 1000),
        model: props.options?.model || props.model as string,
        choices: [{
          index: 0,
          message: {
            role: 'assistant',
            content: response.choices?.[0]?.message?.content || ''
          },
          finish_reason: 'stop'
        }],
        usage: {
          prompt_tokens: response.usage?.prompt_tokens || 0,
          completion_tokens: response.usage?.completion_tokens || 0,
          total_tokens: response.usage?.total_tokens || 0
        }
      } as any;
    }

    // Обработка остальных провайдеров (OpenAI-совместимых)
    const openai = new OpenAI({
      apiKey: token,
      baseURL: baseURL,
      timeout: 90000,
    });

    const response = await openai.chat.completions.create({
      model: props.options?.model || props.model,
      stream: props.stream as boolean,
      messages: [
        {
          role: "system",
          content: `${props.system_prompt}\nProvider: ${props.provider}\nModel: ${props.model}`
        },
        { role: "user", content: props.user_prompt }
      ],
      top_p: props.options?.top_p,
      temperature: props.options?.temperature ?? 0.2,
      max_tokens: props.options?.max_tokens ?? 800,
    });

    console.log('Received response', {
      id: response.id,
      tokens: response.usage
    });

    return response;

  } catch (error: any) {
    console.error('API Error:', {
      provider: props.provider,
      message: error.message,
      code: error.code,
      status: error.status
    });

    // Улучшенная обработка ошибок
    if (error.status === 401) throw new Error('Invalid API key');
    if (error.status === 429) throw new Error('Rate limit exceeded');
    if (error.code === 'ETIMEDOUT') throw new Error('Request timed out');
    if (error.status >= 500) throw new Error(`Server error: ${error.status}`);

    throw new Error(`[${props.provider}] ${error.message || 'Unknown error'}`);
  }
};

/* ---------------------------EXAMPLE---------------------------
 import {usingOpenAI} from './usingOpenAI/index'
 const response = await usingOpenAI({
    system_prompt: 'Simple  write please word 'Hi! How I can you help'',
    user_prompt: 'This test function',
    provider: 'MistralAI',
    model: 'mistral-large-latest',
    stream: false
  })
  return response.choices[0].message.content -> Response: Hi! How I can you help? 
*/// ---------------------------EXAMPLE---------------------------