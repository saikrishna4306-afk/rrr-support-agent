import { useEffect, useRef, useState } from "react";

// One id per browser tab. The backend uses it to group a customer's messages.
const SESSION_ID = crypto.randomUUID();

const QUICK_QUESTIONS = [
  "My upper thread keeps breaking",
  "ఎంబ్రాయిడరీ చేసేటప్పుడు సూది విరిగిపోతోంది",
  "मशीन में E21 एरर आ रहा है",
];

const WELCOME = {
  role: "assistant",
  text: "Tell me what is going wrong with your machine. You can write in English, తెలుగు or हिंदी.",
};

export default function App() {
  const [messages, setMessages] = useState([WELCOME]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [recording, setRecording] = useState(false);
  const [evidence, setEvidence] = useState(null);
  const endRef = useRef(null);
  const recorderRef = useRef(null);
  const timerRef = useRef(null);

  // Keep the newest message in view.
  useEffect(() => {
    endRef.current?.scrollIntoView({ block: "end" });
  }, [messages, busy]);

  async function send(text) {
    const question = (text ?? input).trim();
    if (!question || busy) return;
    setInput("");
    setMessages((m) => [...m, { role: "user", text: question }]);
    setBusy(true);
    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: SESSION_ID, message: question }),
      });
      if (!res.ok) throw new Error(`Server returned ${res.status}`);
      const data = await res.json();
      setMessages((m) => [
        ...m,
        {
          role: "assistant",
          text: data.answer,
          sources: data.sources,
          handoff: data.handoff,
          ticketId: data.ticket_id,
        },
      ]);
    } catch {
      setMessages((m) => [
        ...m,
        {
          role: "assistant",
          error: true,
          text: "I could not reach the support server. Check that the backend is running, then send your message again.",
        },
      ]);
    } finally {
      setBusy(false);
    }
  }

  async function sendDiagnosis() {
    const question = input.trim();
    if (!question || !evidence || busy) return;
    setInput(""); setBusy(true);
    setMessages((m) => [...m, { role: "user", text: question }]);
    try {
      const form = new FormData(); form.append("session_id", SESSION_ID); form.append("message", question); form.append("evidence", evidence);
      const res = await fetch("/api/diagnose", { method: "POST", body: form });
      if (!res.ok) throw new Error(); const data = await res.json();
      setMessages((m) => [...m, { role: "assistant", text: data.answer, sources: data.sources, handoff: data.handoff, ticketId: data.ticket_id }]);
      setEvidence(null);
    } catch { setMessages((m) => [...m, { role: "assistant", error: true, text: "I could not upload the media. Please try again or send a text question." }]); }
    finally { setBusy(false); }
  }

  async function wavFromBlob(blob) {
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    const context = new AudioContextClass();
    if (context.state === "suspended") {
      await context.resume();
    }
    try {
      const decoded = await context.decodeAudioData(await blob.arrayBuffer());
      const frames = Math.min(decoded.length, decoded.sampleRate * 30);
      const output = new Float32Array(frames);
      for (let channel = 0; channel < decoded.numberOfChannels; channel += 1) {
        const source = decoded.getChannelData(channel);
        for (let i = 0; i < frames; i += 1) output[i] += source[i] / decoded.numberOfChannels;
      }
      const length = Math.ceil(frames * 16000 / decoded.sampleRate);
      const samples = new Int16Array(length);
      for (let i = 0; i < length; i += 1) samples[i] = Math.max(-1, Math.min(1, output[Math.floor(i * decoded.sampleRate / 16000)])) * 0x7fff;
      const buffer = new ArrayBuffer(44 + samples.byteLength), view = new DataView(buffer);
      const write = (offset, value) => [...value].forEach((c, i) => view.setUint8(offset + i, c.charCodeAt(0)));
      write(0, "RIFF"); view.setUint32(4, 36 + samples.byteLength, true); write(8, "WAVEfmt ");
      view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 1, true);
      view.setUint32(24, 16000, true); view.setUint32(28, 32000, true); view.setUint16(32, 2, true);
      view.setUint16(34, 16, true); write(36, "data"); view.setUint32(40, samples.byteLength, true);
      samples.forEach((sample, i) => view.setInt16(44 + i * 2, sample, true));
      await context.close();
      return new Blob([buffer], { type: "audio/wav" });
    } catch (err) {
      console.warn("AudioContext decode error, returning raw blob:", err);
      await context.close();
      return blob;
    }
  }

  async function stopRecording() {
    clearTimeout(timerRef.current);
    recorderRef.current?.stop();
  }

  async function toggleRecording() {
    if (recording) return stopRecording();
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const parts = [];
      const mimeType = MediaRecorder.isTypeSupported("audio/webm") ? "audio/webm" : (MediaRecorder.isTypeSupported("audio/ogg") ? "audio/ogg" : "");
      const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
      recorderRef.current = recorder;
      recorder.ondataavailable = (event) => event.data.size && parts.push(event.data);
      recorder.onstop = async () => {
        stream.getTracks().forEach((track) => track.stop()); setRecording(false); setBusy(true);
        try {
          const rawBlob = new Blob(parts, { type: recorder.mimeType || "audio/webm" });
          const wavBlob = await wavFromBlob(rawBlob);
          const form = new FormData(); form.append("audio", wavBlob, "recording.wav");
          form.append("session_id", SESSION_ID);
          const res = await fetch("/api/voice", { method: "POST", body: form });
          if (!res.ok) {
            const errText = await res.text();
            throw new Error(`Server error ${res.status}: ${errText}`);
          }
          const data = await res.json();
          setMessages((m) => [...m, { role: "user", text: data.transcript }, { role: "assistant", text: data.answer_text, sources: data.sources, handoff: data.handoff, ticketId: data.ticket_id }]);
          if (data.audio) {
            new Audio(`data:${data.audio_mime_type};base64,${data.audio}`).play().catch(() => {});
          }
        } catch (err) {
          console.error("Voice processing error:", err);
          setMessages((m) => [...m, { role: "assistant", error: true, text: "I could not process that recording. You can still type your question below." }]);
        }
        finally { setBusy(false); }
      };
      recorder.start(); setRecording(true); timerRef.current = setTimeout(stopRecording, 30000);
    } catch (err) {
      console.error("Microphone access error:", err);
      setMessages((m) => [...m, { role: "assistant", error: true, text: "Microphone access was unavailable. Please type your question." }]);
    }
  }

  return (
    <div className="shell">
      <header className="top">
        <h1>RRR Machine Help</h1>
        <p>Embroidery and sewing machine problems, solved step by step</p>
      </header>

      <main className="thread" aria-live="polite">
        {messages.map((m, i) => (
          <article key={i} className={`bubble ${m.role}${m.error ? " error" : ""}`}>
            <p>{m.text}</p>
            {m.sources?.length > 0 && (
              <p className="source">Source: {m.sources[0].title}</p>
            )}
            {m.handoff && (
              <p className="handoff">
                Technician requested{m.ticketId ? `, ticket #${m.ticketId}` : ""}
              </p>
            )}
          </article>
        ))}
        {busy && <article className="bubble assistant typing">Checking the guides…</article>}
        <div ref={endRef} />
      </main>

      {evidence && (
        <div className="evidence-preview">
          <span>📎 Attached: {evidence.name}</span>
          <button onClick={() => setEvidence(null)} title="Remove attachment">✕</button>
        </div>
      )}
      {messages.length === 1 && (
        <div className="quick">
          {QUICK_QUESTIONS.map((q) => (
            <button key={q} onClick={() => send(q)}>
              {q}
            </button>
          ))}
        </div>
      )}

      <div className="composer">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && send()}
          placeholder="Describe the problem"
          aria-label="Describe the problem"
          maxLength={1000}
        />
        <label className="media-button">Photo/video<input type="file" accept="image/*,video/*" onChange={(e) => setEvidence(e.target.files?.[0] || null)} /></label>
        <button onClick={sendDiagnosis} disabled={busy || !input.trim() || !evidence}>Send media</button>
        <button className={`voice-button${recording ? " recording" : ""}`} onClick={toggleRecording} disabled={busy} aria-label={recording ? "Stop recording" : "Record a voice question"}>
          {recording ? "Stop" : "Voice"}
        </button>
        <button onClick={() => send()} disabled={busy || !input.trim()}>
          Send
        </button>
      </div>
    </div>
  );
}
