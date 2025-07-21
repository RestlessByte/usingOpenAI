# 🚀 Project for Using Neural Networks via Open AI API 🚀
## ## Link to our chat bot in Telegram with using AI Powered [***https://t.me/AIHub0_bot***]

This project is a tool for using neural networks through the Open AI API. To get started, you can deploy Ollama, HuggingFace, OpenAI Chat GPT, OpenRouter, MistralAI, and others. This logic and code were created by the developer: [https://github.com/RestlessByte](https://github.com/RestlessByte) 🌟

**Note:** The code can only be used with mention of this developer. 📝
## How using?
# Download Package Manager **BUN** - ***https://bun.sh/***

```bash
  git@github.com:RestlessByte/usingOpenAI.git && cd usingOpenAI
  mv $pwd.env.example .env && code .env && bun install
```
# Example code
```ts
 import {usingOpenAI} from './usingOpenAI/index'
 
 const response = await usingOpenAI({
    system_prompt: 'Simple  write please word 'Hi! How I can you help'',
    user_prompt: 'This test function',
    provider: 'MistralAI',
    model: 'mistral-large-latest',
    stream: false
  })
  return response.choices[0].message.content // Response: Hi! How I can you help? 
```
---
Thank you for using our project! 😊
