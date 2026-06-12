"""A small desktop GUI for testing Aileen on your PC.

It's a thin front-end over the same :class:`~aileen.conversation.Conversation`
engine the terminal uses, so what you see here is the real bot. You can either
type a message or hold a conversation by voice (record → transcribe → reply),
with replies optionally spoken aloud via ElevenLabs.

Network calls (LLM, speech-to-text, text-to-speech) run on background threads
so the window never freezes; UI updates are marshalled back to Tk's main thread
with ``root.after``.
"""

from __future__ import annotations

import os
import threading
import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk

from .audio.files import write_temp_wav
from .audio.player import play_pcm16
from .audio.recorder import MicRecorder
from .config import Config
from .conversation import Conversation
from .factory import ConfigError, build_knowledge, build_llm, build_stt, build_tts

_USER = "You"


class AileenGUI:
    def __init__(self, root: tk.Tk, config: Config):
        self.root = root
        self.config = config

        # Providers built lazily: the brain up front; STT on first recording;
        # TTS on the first reply we're asked to speak.
        self.llm = None
        self.knowledge = None
        self.convo: Conversation | None = None
        self.stt = None
        self.tts = None

        self.recorder = MicRecorder(sample_rate=config.mic_sample_rate)
        self.recording = False
        self.busy = False

        self._build_widgets()

        try:
            self.llm = build_llm(config)
            self.knowledge = build_knowledge(config)
            self.convo = Conversation(self.llm, self.knowledge, config)
        except ConfigError as exc:
            self._set_status("Not configured — see message.")
            messagebox.showerror(
                "Configuration error",
                f"{exc}\n\nCopy .env.example to .env, add your keys, then restart.",
            )
            return

        self._say_greeting()

    # ----- UI construction -------------------------------------------------

    def _build_widgets(self) -> None:
        self.root.minsize(480, 560)
        outer = ttk.Frame(self.root, padding=10)
        outer.pack(fill=tk.BOTH, expand=True)

        self.transcript = scrolledtext.ScrolledText(
            outer, wrap=tk.WORD, state=tk.DISABLED, font=("Segoe UI", 11), height=18
        )
        self.transcript.pack(fill=tk.BOTH, expand=True)
        self.transcript.tag_config("user_name", foreground="#1a73e8", font=("Segoe UI", 11, "bold"))
        self.transcript.tag_config("bot_name", foreground="#188038", font=("Segoe UI", 11, "bold"))

        self.status = tk.StringVar(value="Starting…")
        ttk.Label(outer, textvariable=self.status, foreground="#666").pack(
            fill=tk.X, pady=(6, 6)
        )

        entry_row = ttk.Frame(outer)
        entry_row.pack(fill=tk.X)
        self.entry = ttk.Entry(entry_row, font=("Segoe UI", 11))
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.entry.bind("<Return>", lambda _e: self._on_send())
        self.send_btn = ttk.Button(entry_row, text="Send", command=self._on_send)
        self.send_btn.pack(side=tk.LEFT, padx=(6, 0))

        controls = ttk.Frame(outer)
        controls.pack(fill=tk.X, pady=(8, 0))
        self.record_btn = ttk.Button(
            controls, text="🎤  Hold to Talk", command=self._on_record
        )
        self.record_btn.pack(side=tk.LEFT)

        self.speak_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(controls, text="Speak replies", variable=self.speak_var).pack(
            side=tk.LEFT, padx=(10, 0)
        )

        ttk.Button(controls, text="New conversation", command=self._on_reset).pack(
            side=tk.RIGHT
        )

    # ----- helpers ---------------------------------------------------------

    def _ui(self, fn, *args) -> None:
        """Run a UI update on Tk's main thread from a worker thread."""
        self.root.after(0, lambda: fn(*args))

    def _set_status(self, text: str) -> None:
        self.status.set(text)

    def _add_message(self, speaker: str, text: str) -> None:
        tag = "user" if speaker == _USER else "bot"
        self.transcript.config(state=tk.NORMAL)
        self.transcript.insert(tk.END, f"{speaker}: ", (f"{tag}_name",))
        self.transcript.insert(tk.END, f"{text}\n\n")
        self.transcript.see(tk.END)
        self.transcript.config(state=tk.DISABLED)

    def _set_busy(self, busy: bool, status: str | None = None) -> None:
        self.busy = busy
        state = tk.DISABLED if busy else tk.NORMAL
        self.send_btn.config(state=state)
        self.entry.config(state=state)
        self.record_btn.config(state=state)
        if status is not None:
            self._set_status(status)

    def _on_error(self, message: str) -> None:
        self._set_busy(False, "Ready")
        messagebox.showerror("Error", message)

    def _ensure_stt(self) -> bool:
        if self.stt is not None:
            return True
        try:
            self.stt = build_stt(self.config)
            return True
        except ConfigError as exc:
            messagebox.showwarning("Microphone input unavailable", str(exc))
            return False

    def _ensure_tts(self) -> bool:
        if self.tts is not None:
            return True
        try:
            self.tts = build_tts(self.config)
            return True
        except ConfigError as exc:
            messagebox.showwarning(
                "Voice unavailable",
                f"{exc}\n\nReplies will be shown as text only.",
            )
            self.speak_var.set(False)
            return False

    # ----- actions ---------------------------------------------------------

    def _say_greeting(self) -> None:
        greeting = self.convo.greeting()
        self._add_message(self.config.bot_name, greeting)
        self._set_status("Ready")
        if self.speak_var.get() and self._ensure_tts():
            self._speak_async(greeting)

    def _speak_async(self, text: str) -> None:
        """Speak a line aloud on a background thread (used for the greeting)."""
        self._set_busy(True, f"{self.config.bot_name} is speaking…")

        def work() -> None:
            try:
                pcm, sample_rate = self.tts.synthesize(text)
                play_pcm16(pcm, sample_rate)
            except Exception as exc:  # noqa: BLE001 - surfaced to the user
                self._ui(self._on_error, f"Voice error: {exc}")
                return
            self._ui(self._set_busy, False, "Ready")

        threading.Thread(target=work, daemon=True).start()

    def _on_send(self) -> None:
        if self.busy or self.convo is None:
            return
        text = self.entry.get().strip()
        if not text:
            return
        self.entry.delete(0, tk.END)
        self._add_message(_USER, text)
        self._process_turn(text)

    def _on_record(self) -> None:
        if self.convo is None:
            return
        if not self.recording:
            if self.busy or not self._ensure_stt():
                return
            self.recorder.start()
            self.recording = True
            self.record_btn.config(text="⏹  Stop")
            self.send_btn.config(state=tk.DISABLED)
            self.entry.config(state=tk.DISABLED)
            self._set_status("🔴 Recording… click Stop when you're done.")
            return

        # Stop recording and transcribe.
        samples = self.recorder.stop()
        self.recording = False
        self.record_btn.config(text="🎤  Hold to Talk")
        if samples.shape[0] == 0:
            self._set_busy(False, "Didn't catch any audio — try again.")
            return

        self._set_busy(True, "Transcribing…")

        def work() -> None:
            wav_path = write_temp_wav(samples, self.config.mic_sample_rate)
            try:
                text = self.stt.transcribe(wav_path)
            except Exception as exc:  # noqa: BLE001 - surfaced to the user
                self._ui(self._on_error, f"Transcription error: {exc}")
                return
            finally:
                os.remove(wav_path)
            if not text:
                self._ui(self._set_busy, False, "Couldn't make that out — try again.")
                return
            self._ui(self._add_message, _USER, text)
            self._ui(self._process_turn, text)

        threading.Thread(target=work, daemon=True).start()

    def _process_turn(self, user_text: str) -> None:
        """Send one user turn to the brain, then optionally speak the reply."""
        speak = self.speak_var.get()
        if speak and not self._ensure_tts():
            speak = False

        self._set_busy(True, f"{self.config.bot_name} is thinking…")

        def work() -> None:
            try:
                reply = self.convo.handle(user_text)
            except Exception as exc:  # noqa: BLE001 - surfaced to the user
                self._ui(self._on_error, f"Brain error: {exc}")
                return
            self._ui(self._add_message, self.config.bot_name, reply)
            if speak and self.tts is not None:
                self._ui(self._set_status, f"{self.config.bot_name} is speaking…")
                try:
                    pcm, sample_rate = self.tts.synthesize(reply)
                    play_pcm16(pcm, sample_rate)
                except Exception as exc:  # noqa: BLE001 - surfaced to the user
                    self._ui(self._on_error, f"Voice error: {exc}")
                    return
            self._ui(self._set_busy, False, "Ready")

        threading.Thread(target=work, daemon=True).start()

    def _on_reset(self) -> None:
        if self.busy or self.llm is None:
            return
        self.convo = Conversation(self.llm, self.knowledge, self.config)
        self.transcript.config(state=tk.NORMAL)
        self.transcript.delete("1.0", tk.END)
        self.transcript.config(state=tk.DISABLED)
        self._say_greeting()


def main() -> int:
    config = Config.from_env()
    root = tk.Tk()
    root.title(f"{config.bot_name} — test console")
    root.geometry("560x640")
    AileenGUI(root, config)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
