"""Claude SDK wrapper.

ClaudeSession encapsulates a long-lived ClaudeSDKClient running on a background
asyncio loop. Exposes start/reset/query/cancel_inflight as plain blocking
methods callable from any thread.

Concurrency model:
- A single _query_lock serializes query() and reset() — one logical operation
  on the SDK at a time.
- cancel_inflight() can be called from a third thread (e.g. audio_loop barge-in)
  to abort the current query. The query path detects cancellation/timeout and
  forcibly rebuilds the SDK client before returning, so the next query starts
  on a fresh, known-good client.

Note: actual orchestration (logging, State updates, TTS triggering) lives in
main.py — this module only owns the SDK client lifecycle and message parsing.
"""

import asyncio
import concurrent.futures
import threading
from typing import List, Optional, Tuple

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ClaudeSDKClient,
    TextBlock,
)


SYSTEM_PROMPT = """你是 ShuaiYan 的语音对话助手。

【输出约束 — 必须严格遵守】
- 回复极简，通常 1 句话，最多 2 句
- 这是语音对话，不是文字。不要任何 markdown、列表、代码块、URL、emoji
- 数字读成中文：写"一百二十三"不写"123"
- 自然口语，允许"嗯""那个""所以"开头
- 不要说"作为 AI 助手"这种废话，直接回答

【能力边界】
你能调用以下工具：
- **联网搜索**：WebSearch（搜索引擎）、WebFetch（抓取指定 URL 内容）
- **TTS 控制**：mcp__tts__* 系列（见下面）

你**没有**：读/写本地文件、执行命令、规划任务等能力。不要尝试调 Read/Write/Edit/Bash/Grep/Glob 等。

【联网时的注意】
- 用户问天气/新闻/股价/查某事时，主动用 WebSearch 或 WebFetch
- 搜出来的结果是给你看的，不要原样念。**总结成 1-2 句口语化的话**回复
- 例如查天气：搜"上海明天天气" → 看到结果 → 回复"明天多云转晴，最高十八度"

【TTS 控制工具】
- get_tts_settings: 查询当前音色/语速/音量，以及所有可用音色标签
- set_voice(label): 切换音色 (label 是英文标签，先用 get_tts_settings 看有哪些)
- adjust_rate(delta): 相对调语速，正数加快，负数减慢，单位百分比
- set_rate(value): 直接设语速百分比 (-50 到 +100)
- adjust_volume(delta): 相对调音量
- set_volume(value): 直接设音量百分比
- reset_tts: 全部恢复默认
- speak(text): 触发 TTS 朗读单独的一段文本 ⭐重要

【显示一版 + 朗读一版（speak 工具的用法）】
默认行为：你最终的 TextBlock 会被自动朗读。
但当你的回答**含详细信息**（搜索结果、数字、来源、解释），用户**只需要听简短结论**时，使用 speak 工具：
1. 把详细内容写在普通 TextBlock（终端会显示，但**不会**朗读）
2. 同时调用 speak("简短的1-2句口语版")

调了 speak 之后，你的 TextBlock 不会再被自动朗读，避免重复。

例：用户问"今天上海天气"
- WebSearch 查到完整预报
- 你的 TextBlock 写："今日上海预报：晴转多云，气温 14-22 度，湿度 65%，PM2.5 中等。来源 weather.com.cn"
- 同时 speak("上海今天晴转多云，最高二十二度")
- 用户只听到"上海今天晴转多云，最高二十二度"，但能在屏幕上看到完整数据

简短闲聊（不含详细数据）则不用 speak，直接 TextBlock，会被自动朗读。

当用户说任何关于声音/音色/语速/音量/快/慢/大声/小声/换声音/男声/女声/播音员/少女音/查可用声音 等内容时，
**必须主动调工具**，不要只回答"好的"。

例：
- "你说快点" → 调 adjust_rate(delta=20)
- "用男声说" → 先 get_tts_settings 看 male 的真实标签 → set_voice(label="male")
- "声音太大了" → adjust_volume(delta=-20)
- "有几种声音可选" → 调 get_tts_settings 然后简短报数量和名字

工具调用结果是给你看的内部信息，不要原文念给用户。
工具完成后用 1 句口语化的话确认，比如"好""换好了""调慢了点"。

【再次强调】
你的回复会被 TTS 直接念出来。Markdown 字符会被当成"星号""井号"读出来。永远不要用。

正例："明天上海多云转晴，最高十八度"
反例："**天气预报**：\n- 多云转晴\n- 18°C"
"""


class QueryCancelled(Exception):
    """Raised by query() when an in-flight call was cancelled (barge-in / reset)."""


class ClaudeSession:
    """Long-lived Claude session running on a background asyncio loop."""

    def __init__(self, *, system_prompt: str, mcp_servers: dict,
                 allowed_tools: list, disallowed_tools: list = None,
                 permission_mode: str = 'default', model: str | None = None):
        self._system_prompt = system_prompt
        self._mcp_servers = mcp_servers
        self._allowed_tools = list(allowed_tools)
        self._disallowed_tools = list(disallowed_tools or [])
        self._permission_mode = permission_mode
        self._model = model

        self._loop = asyncio.new_event_loop()
        threading.Thread(target=self._loop.run_forever, daemon=True).start()

        self._client: ClaudeSDKClient | None = None
        self._lock = threading.Lock()             # protects _client, _inflight_task
        self._query_lock = threading.Lock()       # serializes query() and reset()
        # Hold the actual asyncio.Task (not the concurrent.futures.Future);
        # only Task.cancel() actually interrupts a running coroutine.
        # concurrent.futures.Future.cancel() is a no-op once the task starts running.
        self._inflight_task: Optional[asyncio.Task] = None

    def _options(self) -> ClaudeAgentOptions:
        return ClaudeAgentOptions(
            system_prompt=self._system_prompt,
            mcp_servers=self._mcp_servers,
            allowed_tools=self._allowed_tools,
            disallowed_tools=self._disallowed_tools,
            permission_mode=self._permission_mode,
            model=self._model,
        )

    async def _open(self) -> ClaudeSDKClient:
        c = ClaudeSDKClient(options=self._options())
        await c.__aenter__()
        return c

    async def _close(self, c: ClaudeSDKClient):
        try:
            await c.__aexit__(None, None, None)
        except Exception:
            pass

    def _run(self, coro, timeout=30):
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout=timeout)

    def _reopen_client(self):
        """Internal: close current client and open a fresh one. Caller is
        responsible for whatever locking is appropriate (we touch _client only
        under _lock here, which is sufficient since callers hold _query_lock)."""
        with self._lock:
            old = self._client
            self._client = None
        if old is not None:
            try:
                self._run(self._close(old), timeout=10)
            except Exception as e:
                print(f'[claude close warn]: {e}')
        try:
            new_client = self._run(self._open(), timeout=60)
        except Exception as e:
            print(f'[claude open error]: {e}')
            return
        with self._lock:
            self._client = new_client

    def start(self):
        """Open the persistent client. Call once at startup."""
        with self._query_lock:
            self._reopen_client()

    def reset(self):
        """Tear down current session and start fresh — clears history.
        Cancels any in-flight query first so we don't block on its timeout.
        """
        self.cancel_inflight()
        with self._query_lock:
            self._reopen_client()

    def cancel_inflight(self) -> bool:
        """Cancel the current query (if any). Safe from any thread.
        Returns True if there was an in-flight call to cancel.

        Uses loop.call_soon_threadsafe(task.cancel) because Task.cancel() must
        only be called from the loop's thread, and concurrent.futures.Future
        .cancel() is a no-op once the coroutine is running.
        """
        with self._lock:
            task = self._inflight_task
        if task is None or task.done():
            return False
        self._loop.call_soon_threadsafe(task.cancel)
        return True

    async def _start_query(self, client: ClaudeSDKClient, user_text: str):
        """Run on the loop thread. Synchronously creates the inner Task and
        registers it BEFORE awaiting, closing the race where cancel_inflight()
        between submit() and the coroutine's first scheduled tick would see
        _inflight_task=None and silently drop the cancel.
        """
        task = asyncio.create_task(self._do(client, user_text))
        with self._lock:
            self._inflight_task = task
        try:
            return await task
        finally:
            with self._lock:
                self._inflight_task = None

    async def _do(self, client: ClaudeSDKClient, user_text: str):
        final_parts: list = []
        intermediate: list = []
        await client.query(user_text)
        async for message in client.receive_response():
            if not isinstance(message, AssistantMessage):
                continue
            text_parts = []
            has_tool = False
            for block in message.content:
                if isinstance(block, TextBlock):
                    text_parts.append(block.text)
                else:
                    tname = getattr(block, 'name', type(block).__name__)
                    tinput = getattr(block, 'input', None)
                    print(f'  [tool] {tname}({tinput})')
                    intermediate.append({'kind': 'tool', 'name': tname, 'input': tinput})
                    has_tool = True
            joined = ''.join(text_parts).strip()
            if not joined:
                continue
            if has_tool:
                print(f'  [aside] {joined}')
                intermediate.append({'kind': 'aside', 'text': joined})
            else:
                print(f'  [final] {joined}')
                final_parts.append(joined)
        return '\n'.join(final_parts), intermediate

    def query(self, user_text: str, timeout: float = 60.0) -> Tuple[str, List[dict]]:
        """Send `user_text`, return (final_text, intermediate_log).

        Raises QueryCancelled if cancel_inflight() was called from another thread
        (e.g. barge-in) or if the call timed out. In both cases the SDK client is
        forcibly rebuilt before this method returns so the next query is clean.
        """
        with self._query_lock:
            with self._lock:
                client = self._client
            if client is None:
                raise RuntimeError('ClaudeSession not started; call .start() first')

            # _start_query sets self._inflight_task on the loop thread BEFORE
            # any await — eliminates the submit-vs-cancel race window.
            fut = asyncio.run_coroutine_threadsafe(
                self._start_query(client, user_text), self._loop
            )

            cancelled = False
            try:
                return fut.result(timeout=timeout)
            except (concurrent.futures.CancelledError, asyncio.CancelledError):
                cancelled = True
                raise QueryCancelled('query cancelled')
            except concurrent.futures.TimeoutError:
                with self._lock:
                    task = self._inflight_task
                if task is not None and not task.done():
                    self._loop.call_soon_threadsafe(task.cancel)
                cancelled = True
                raise
            finally:
                if cancelled:
                    # Wait briefly for the cancelled task to finish its cleanup
                    # (e.g. SDK client.__aexit__) before we reopen, so _close
                    # doesn't race against an in-progress cleanup.
                    try:
                        fut.result(timeout=2)
                    except Exception:
                        pass
                    # SDK client may still be in a partial state; rebuild before next query.
                    self._reopen_client()
