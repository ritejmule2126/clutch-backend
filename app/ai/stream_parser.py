from __future__ import annotations
from typing import List, Tuple

class ThinkingStreamParser:
    """
    Production-grade streaming parser that cleanly separates <thinking>...</thinking>
    deliberations from the final user-facing response tokens.
    
    Emits events:
      - ("thinking", delta)
      - ("thinking_done", "")
      - ("token", delta)
    """
    def __init__(self):
        self.in_thinking = False
        self.buffer = ""
        self.clean_answer_chunks: List[str] = []
        self.thought_chunks: List[str] = []

    def feed(self, chunk: str) -> List[Tuple[str, str]]:
        events: List[Tuple[str, str]] = []
        self.buffer += chunk

        while self.buffer:
            if not self.in_thinking:
                if "<thinking>" in self.buffer:
                    pre, post = self.buffer.split("<thinking>", 1)
                    if pre:
                        self.clean_answer_chunks.append(pre)
                        events.append(("token", pre))
                    self.in_thinking = True
                    self.buffer = post
                elif any("<thinking"[:i] == self.buffer[-i:] for i in range(1, 10)):
                    # Partial start tag at the very end of chunk, wait for next token
                    break
                else:
                    self.clean_answer_chunks.append(self.buffer)
                    events.append(("token", self.buffer))
                    self.buffer = ""
            else:
                if "</thinking>" in self.buffer:
                    thought, post = self.buffer.split("</thinking>", 1)
                    if thought:
                        self.thought_chunks.append(thought)
                        events.append(("thinking", thought))
                    events.append(("thinking_done", ""))
                    self.in_thinking = False
                    self.buffer = post.lstrip("\r\n")
                elif any("</thinking>"[:i] == self.buffer[-i:] for i in range(1, 11)):
                    # Partial end tag at the end of chunk, wait
                    break
                else:
                    self.thought_chunks.append(self.buffer)
                    events.append(("thinking", self.buffer))
                    self.buffer = ""

        return events

    def flush(self) -> List[Tuple[str, str]]:
        events: List[Tuple[str, str]] = []
        if self.buffer:
            if self.in_thinking:
                self.thought_chunks.append(self.buffer)
                events.append(("thinking", self.buffer))
                events.append(("thinking_done", ""))
            else:
                self.clean_answer_chunks.append(self.buffer)
                events.append(("token", self.buffer))
            self.buffer = ""
        return events

    @property
    def clean_text(self) -> str:
        return "".join(self.clean_answer_chunks).strip()

    @property
    def thought_text(self) -> str:
        return "".join(self.thought_chunks).strip()
