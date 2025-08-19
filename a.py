from rich import print
from rich.panel import Panel
print(Panel("Hello, [red]World!", title="Welcome", subtitle="Thank you"))

json = {
  "object": "chat.completion",
  "choices": [
    {
      "finish_reason": "stop",
      "index": 0,
      "message": {
        "content": " I'm LLaMA, an AI assistant developed by Meta AI that can understand and respond to human input in a conversational manner.",
        "role": "assistant",
        "logprobs": None
      }
    }
  ],
  "id": "chatcmpl-ea7b8242-791f-4656-ba12-e098edeb960e",
  "created": 1695324686.6696231,
  "response_ms": 4072.3050000000003,
  "model": "ollama/llama2",
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 27,
    "total_tokens": 37
  }
}

print(json["choices"][0]["message"]["content"])

import pyrailroad.elements
print(help(pyrailroad.elements))