"""backends.py

Unified chat backends for OpenAI (GPT), Google Gemini (Vertex), and Anthropic (Claude).

- Exposes consistent sync/async classes with `chat(messages, temperature) -> str`.
- Normalizes message formats (system/user/assistant) per provider.
- Supports local Ollama when model == "gpt-oss:20b".
- Includes helpers for Gemini (system_instruction/contents) and Claude (text extraction).
- Requires proper credentials/project/location; Anthropic needs `max_output_tokens`.
"""

import asyncio
from openai import OpenAI, AsyncOpenAI
from typing import List, Dict, Optional
from google import genai
from google.genai import types as gt
from anthropic import Anthropic, AsyncAnthropic


# ============================================================================================= #
#                                        GPT Backend                                            #
# ============================================================================================= #
class GPTSyncBackend:
    """ Synchronous GPT Backend """
    def __init__(
        self,
        api_key: str,
        model: str
    ):
        if model == "gpt-oss:20b":
            self.client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        else:
            self.client = OpenAI(api_key=api_key)
        self.model = model

    def chat(self, messages: List[Dict[str, str]], temperature: float = 1.0) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
        )
        return response.choices[0].message.content or ""


class GPTAsyncBackend:
    """ Asynchronous GPT Backend """
    def __init__(
        self,
        api_key: str,
        model: str,
    ):
        if model == "gpt-oss:20b":
            self.client = AsyncOpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
        else:
            self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
    
    async def chat(self, messages, temperature: float = 1.0 ) -> str:
        response = await self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=temperature
        )
        return response.choices[0].message.content or ""

# ============================================================================================= #
#                                        Gemini Backend                                         #
# ============================================================================================= #
def _to_gemini_messages(messages) -> tuple[Optional[str], list[gt.ContentDict]]:
    system_instruction = None
    contents: list[gt.ContentDict] = []

    for message in messages:
        role = (message.get("role") or "user").lower()
        text = message.get("content", "")

        if role == "system":
            system_instruction = text if system_instruction is None else f"{system_instruction}\n\n{text}"
            continue
        if role == "assistant":
            role = "model"

        contents.append(
            gt.ContentDict(role=role, parts=[gt.PartDict(text=text)])
        )

    return system_instruction, contents


class GeminiSyncBackend:
    """Synchronous Gemini Backend (google-genai, Vertex)"""
    def __init__(self, model: str, project: str, location: str):
        self.client = genai.Client(vertexai=True, project=project, location=location)
        self.model = model

    def chat(self, messages, temperature: float = 1.0) -> str:
        system_instruction, contents = _to_gemini_messages(messages=messages)
        if not contents:
            raise ValueError("GeminiSyncBackend.chat: no user/model messages to send.")

        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config=gt.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=temperature,
            ),
        )
        return getattr(response, "text")


class GeminiAsyncBackend:
    """Asynchronous Gemini Backend (google-genai, wrap sync in a thread)"""
    def __init__(self, model: str, project: str, location: str):
        self.client = genai.Client(vertexai=True, project=project, location=location)
        self.model = model

    async def chat(self, messages, temperature: float = 1.0) -> str:
        def _call() -> str:
            system_instruction, contents = _to_gemini_messages(messages=messages)
            if not contents:
                raise ValueError("GeminiAsyncBackend.chat: no user/model messages to send.")

            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=gt.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=temperature           
                )
            )
            return getattr(response, "text")

        return await asyncio.to_thread(_call)
    

# ============================================================================================= #
#                                        Claude Backend                                         #
# ============================================================================================= #
def _to_claude_messages(messages: List[Dict[str, str]]) -> tuple[Optional[str], List[Dict[str, str]]]:
    system_txt: Optional[str] = None
    output: List[Dict[str, str]] = []

    for message in messages:
        role = (message.get("role") or "user").lower()
        text = message.get("content", "")

        if role == "system":
            system_txt = text if system_txt is None else f"{system_txt}\n\n{text}"
            continue
            
        elif role == "assistant":
            role = "assistant"
        
        output.append({"role": role, "content": text})
    
    return system_txt, output

def _extract_text(response) -> str:
    """Extract text from anthropic Message."""
    if not getattr(response, "content", None):
        return ""
    parts = []
    for block in response.content:
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
    return "".join(parts)


class ClaudeSyncBackend:
    """Synchronous Claude Backend (Anthropic)"""
    def __init__(
            self,
            api_key: str,
            model: str,
            max_output_tokens: int = 1024   # max_tokens parameter must be set
    ):
        self.model = model
        self.client = Anthropic(api_key=api_key)
        self.max_output_tokens = max_output_tokens
    
    def chat(self, messages: List[Dict[str, str]], temperature: float = 1.0) -> str:
        system_instruction, contents = _to_claude_messages(messages=messages)
        if not contents:
            raise ValueError("ClaudeSyncBackend.chat: no user/assistant messages to send.")

        response = self.client.messages.create(
            model=self.model,
            messages=contents,
            system=system_instruction,
            temperature=temperature,
            max_tokens=self.max_output_tokens
        )
        return _extract_text(response=response)
    

class ClaudeAsyncBackend:
    """Asynchronous Claude Backend (Anthropic)"""
    def __init__(
            self,
            api_key: str,
            model: str,
            max_output_tokens: int = 1024
    ):
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.client = AsyncAnthropic(api_key=api_key)
    
    async def chat(self, messages: List[Dict[str, str]], temperature: float = 1.0) -> str:
        system_instruction, contents = _to_claude_messages(messages=messages)
        if not contents:
            raise ValueError("ClaudeAsyncBackend.chat: no user/assistant messages to send.")
        
        response = await self.client.messages.create(
            model=self.model,
            messages=contents,
            system=system_instruction,
            temperature=temperature,
            max_tokens=self.max_output_tokens
        )
        return _extract_text(response=response)
    