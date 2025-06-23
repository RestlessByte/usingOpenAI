import { Ollama } from './../../../ollama-js/src/browser';
// src/ai/mistral.ts
import { config } from 'dotenv';
import OpenAI from 'openai';
// import { InferenceClient } from '@huggingface/inference';
// Load environment variables
/**
 * @params [example] -   await usingOpenAI({
    system_prompt: 'Напиши просто тест',
    user_prompt: 'rewqrdwdssdffsd',
    provider: 'MistralAI',
    model: 'mistral-large-latest',
    stream: false
  }).then(e => e?.choices[0].message.content)
 */
export type TypeModels = OpenAI.AllModels | `mistral-large-latest` | false
type TypeProvider = 'MistralAI' | 'OpenAI' | 'OpenRouter' | 'Ollama' | 'HuggingFace'
import ollama from 'ollama'


config();
// const HugginFace = new InferenceClient()
export const usingOpenAI = async (
  props: {
    user_prompt: string,
    system_prompt: string,
    provider: TypeProvider,
    model: TypeModels,
    stream: boolean,
    options?: {
      temperature?: number
      max_tokens?: number
    }
  }
): Promise<OpenAI.Chat.Completions.ChatCompletion | undefined> => {
  /**
 * @param {string} using Open AI API for other AI powered
 * @param {string} for start work add TOKEN [API KEY] to .env in root directories
 */
  // Fixed environment variable name (MISTRALAI_TOKEN instead of MISTRALLAI_TOKEN)
  let token, message, json, client;
  let baseURL: string | undefined = undefined;
  // Set baseURL only for Mistral models
  if (props.provider == 'MistralAI') {
    token = process.env.MistralAI_TOKEN
    baseURL = 'https://api.mistral.ai/v1';
  }
  if (props.provider == 'OpenRouter') {
    token = process.env.OpenRouter_TOKEN
    baseURL = 'https://openrouter.ai/api/v1'
  }
  if (props.provider == 'Ollama' && props.model == false) {
    token = 'Ollama'
    const response = await ollama.chat({
      model: 'deepseek-r1:latest', messages: [{
        role: 'user',
        content: props.user_prompt
      },
      {
        role: 'system_prompt',
        content: props.system_prompt
      }], stream: props.stream as boolean
    })
    for await (const part of response) {
      message = (part.message.content)
    }

  }
  if (!token) {
    const errorMessage = 'Missing API token in environment variables. Please add either token to .env';
    console.error(errorMessage);
    throw new Error(errorMessage);
  }

  try {
    console.log('Sending request to API', {
      model: props.model,
      promptLength: props.user_prompt.length,
      systemPromptLength: props.system_prompt.length
    });

    if (props.model) {
      const response = await new OpenAI({
        apiKey: token,
        baseURL: baseURL,
        timeout: 90000,

      }).chat.completions.create({
        model: props.model as string,
        stream: props.stream as boolean,
        messages: [
          {
            role: "system", content: `${props.system_prompt}
            Твой провайдер: ${props.provider}
            Твоя модель: ${props.model}
            Ты говоришь на том языке, что и пользователь`
          },
          { role: "user", content: `Запрос от пользователя - ${props.user_prompt}` }
        ],
        temperature: props?.options?.temperature || 0.2,
        max_tokens: props?.options?.max_tokens || 800
      });

      console.log('Received response from API', {
        responseId: response.id,
        usage: response.usage,
      });
      return response
    };
  } catch (error: unknown) {
    // Properly type-checked error handling
    const err = error as {
      message?: string;
      code?: string;
      status?: number;
      statusText?: string;
    };
    console.error('Error calling API:', {
      message: err.message,
      code: err.code,
      status: err.status,
    });

    if (err.status === 401) {
      throw new Error('Authentication error: Invalid API key');
    }

    if (err.status === 429) {
      throw new Error('API rate limit exceeded');
    }

    if (err.code === 'ETIMEDOUT' || err.code === 'ECONNABORTED') {
      throw new Error('API request timed out');
    }

    if (err.status && err.status >= 500) {
      throw new Error(`API server error: ${err.status} ${err.statusText || ''}`);
    }

    throw new Error(`Content generation failed: ${err.message || 'Unknown error'}
      
      INFORMATION:
      Or this provider no can this model - ${props.model}`);
  }
};
console.log(
  await usingOpenAI({
    system_prompt: 'Напиши просто тест',
    user_prompt: 'rewqrdwdssdffsd',
    provider: 'MistralAI',
    model: 'mistral-large-latest',
    stream: false
  }).then(e => e?.choices[0].message.content)
)