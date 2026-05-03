/**
 * voice.js — Módulo de Voz da AMÉLIA v3.0
 * ==========================================
 *
 * Correções desta versão:
 *  ✅ Bug do áudio: o microfone NÃO captura mais a fala da AMÉLIA.
 *     Solução: o reconhecimento só começa APÓS a síntese terminar (evento onend).
 *              Enquanto o sistema fala, o microfone fica inativo.
 *  ✅ Emergência sem senha; demais senhas padronizadas: U001, M001, L001
 *  ✅ Compatível com o novo fluxo de botões do script.js
 *
 * Usa Web Speech API (nativa, gratuita).
 * Melhor suporte: Chrome/Edge. Firefox não suporta SpeechRecognition.
 */

// ── URL da API (detecta ambiente automaticamente) ─────────────
const API_BASE_URL = (
    window.location.hostname === "localhost" ||
    window.location.hostname === "127.0.0.1"
) ? "http://localhost:8000" : "";

// ── Estado do módulo de voz ───────────────────────────────────
const voiceState = {
    isRecording:  false,
    recognition:  null,
    synthesis:    window.speechSynthesis,
    transcript:   "",
    apiOnline:    false,
    isSpeaking:   false,   // ← NOVO: true enquanto a AMÉLIA estiver falando
    speechToken:   0,
    echoBlockUntil: 0,
    echoGuardMs:   900,
    pendingStartTimer: null,
};

function _voicePlainText(text) {
    return String(text ?? "")
        .replace(/<[^>]*>/g, " ")
        .replace(/&nbsp;/g, " ")
        .replace(/&amp;/g, "&")
        .replace(/&lt;/g, "<")
        .replace(/&gt;/g, ">")
        .replace(/\s+/g, " ")
        .trim();
}

function _voiceCleanText(text) {
    return _voicePlainText(text)
        .replace(/[\u{1F000}-\u{1FAFF}\u{2600}-\u{27BF}\uFE0F\u200D]/gu, "")
        .replace(/[≤]/g, "menor ou igual a ")
        .replace(/[≥]/g, "maior ou igual a ")
        .replace(/[–—]/g, ". ")
        .replace(/[•·]/g, ". ")
        .replace(/[^\p{L}\p{N}\s.,;:!?%/()-]/gu, " ")
        .replace(/\s+/g, " ")
        .trim();
}

function _voiceCleanTranscript(text) {
    return _voicePlainText(text)
        .replace(/[\u{1F000}-\u{1FAFF}\u{2600}-\u{27BF}\uFE0F\u200D]/gu, "")
        .replace(/[^\p{L}\p{N}\s.,;:!?%/()-]/gu, " ")
        .replace(/\s+/g, " ")
        .trim();
}

function _voiceIsSpeechActive() {
    return !!(
        voiceState.isSpeaking ||
        (voiceState.synthesis && (voiceState.synthesis.speaking || voiceState.synthesis.pending))
    );
}

function _voiceBlockEcho(ms = voiceState.echoGuardMs) {
    voiceState.echoBlockUntil = Math.max(voiceState.echoBlockUntil, Date.now() + ms);
}

function _voiceClearPendingStart() {
    if (voiceState.pendingStartTimer) {
        clearTimeout(voiceState.pendingStartTimer);
        voiceState.pendingStartTimer = null;
    }
}

function _voiceCancelSpeechOutput() {
    voiceState.speechToken++;
    if (voiceState.synthesis) voiceState.synthesis.cancel();
    voiceState.isSpeaking = false;
    _voiceBlockEcho();
}

function _voiceDedupeTranscript(text) {
    const words = _voiceCleanTranscript(text).split(/\s+/).filter(Boolean);
    const out = [];

    for (let i = 0; i < words.length;) {
        let skipped = false;
        const maxSize = Math.min(8, Math.floor((words.length - i) / 2));

        for (let size = maxSize; size >= 2; size--) {
            const a = words.slice(i, i + size).join(" ").toLowerCase();
            const b = words.slice(i + size, i + size * 2).join(" ").toLowerCase();
            if (a && a === b) {
                out.push(...words.slice(i, i + size));
                i += size * 2;
                skipped = true;
                break;
            }
        }

        if (!skipped) out.push(words[i++]);
    }

    return out.join(" ");
}


// ═══════════════════════════════════════════════════════════════
// STATUS DA API
// ═══════════════════════════════════════════════════════════════

async function checkAPIStatus() {
    const dot    = document.getElementById("api-indicator");
    const status = document.getElementById("api-status");
    try {
        const res = await fetch(`${API_BASE_URL}/api/health`,
            { signal: AbortSignal.timeout(3000) });
        if (res.ok) {
            voiceState.apiOnline = true;
            if (dot)    dot.textContent    = "OK";
            if (status) status.textContent = "IA Online (97.74% acurácia)";
        } else throw new Error();
    } catch {
        voiceState.apiOnline = false;
        if (dot)    dot.textContent    = "OFF";
        if (status) status.textContent = "Modo local (sem backend)";
    }
}


// ═══════════════════════════════════════════════════════════════
// SÍNTESE DE VOZ (TEXTO → FALA)
// ═══════════════════════════════════════════════════════════════

/**
 * Fala um texto em voz alta.
 *
 * ╔══ CORREÇÃO DO BUG DE ÁUDIO ══════════════════════════════╗
 * ║  voiceState.isSpeaking = true  enquanto a AMÉLIA fala.  ║
 * ║  O reconhecimento de voz checa essa flag antes de       ║
 * ║  processar qualquer resultado — descarta tudo que       ║
 * ║  chegou enquanto o sistema estava falando.              ║
 * ║  Além disso, se o usuário clicar em "Falar" enquanto    ║
 * ║  a AMÉLIA ainda fala, esperamos o onend antes de        ║
 * ║  ativar o microfone.                                    ║
 * ╚═══════════════════════════════════════════════════════════╝
 *
 * @param {string}   text   Texto a falar (HTML é removido)
 * @param {Function} onEnd  Callback chamado quando terminar (opcional)
 */
function speak(text, onEnd = null) {
    if (!voiceState.synthesis) {
        if (onEnd) onEnd();
        return;
    }
    if (voiceState.isRecording) {
        if (onEnd) onEnd();
        return;
    }

    // Remove HTML, emojis e símbolos que a síntese costuma pronunciar mal.
    const clean = _voiceCleanText(text);
    if (!clean) {
        voiceState.isSpeaking = false;
        if (onEnd) onEnd();
        return;
    }

    const speechToken = ++voiceState.speechToken;
    voiceState.isSpeaking = true;   // ← microfone vai ignorar entrada
    _voiceBlockEcho();
    voiceState.synthesis.cancel();

    const utt     = new SpeechSynthesisUtterance(clean);
    utt.lang      = "pt-BR";
    utt.rate      = 0.92;
    utt.pitch     = 1.05;
    utt.volume    = 1.0;

    // Escolhe voz feminina em pt-BR se disponível
    const _setVoice = () => {
        const voices = voiceState.synthesis.getVoices();
        const ptFem  = voices.find(v =>
            v.lang.startsWith("pt") && v.name.toLowerCase().includes("female"));
        const ptAny  = voices.find(v => v.lang.startsWith("pt"));
        const chosen = ptFem || ptAny;
        if (chosen) utt.voice = chosen;
    };
    voiceState.synthesis.getVoices().length > 0
        ? _setVoice()
        : voiceState.synthesis.addEventListener("voiceschanged", _setVoice, { once: true });

    utt.onend = () => {
        // Aguarda 400 ms extras de margem de segurança antes de
        // liberar o microfone — garante que o eco do alto-falante
        // já se dissipou antes de começar a gravar.
        _voiceBlockEcho();
        setTimeout(() => {
            if (speechToken !== voiceState.speechToken) return;
            voiceState.isSpeaking = false;
            if (onEnd) onEnd();
        }, voiceState.echoGuardMs);
    };

    utt.onerror = () => {
        if (speechToken !== voiceState.speechToken) return;
        _voiceBlockEcho();
        voiceState.isSpeaking = false;
        if (onEnd) onEnd();
    };

    voiceState.synthesis.speak(utt);
}


// ═══════════════════════════════════════════════════════════════
// RECONHECIMENTO DE VOZ (FALA → TEXTO)
// ═══════════════════════════════════════════════════════════════

function checkVoiceSupport() {
    const ok = "SpeechRecognition" in window || "webkitSpeechRecognition" in window;
    if (!ok) {
        const btn = document.getElementById("btn-voice");
        if (btn) {
            btn.disabled      = true;
            btn.title         = "Reconhecimento de voz não suportado. Use Chrome.";
            btn.style.opacity = "0.4";
        }
    }
    return ok;
}

function _createRecognition() {
    const SR  = window.SpeechRecognition || window.webkitSpeechRecognition;
    const rec = new SR();

    rec.lang            = "pt-BR";
    rec.continuous      = true;
    rec.interimResults  = true;
    rec.maxAlternatives = 1;

    rec.onstart = () => {
        voiceState.isRecording = true;
        _updateMicBtn(true);
        _setFeedback("Ouvindo... fale agora e clique em parar quando terminar.");
    };

    rec.onresult = (event) => {
        // ╔══ CORREÇÃO DO BUG ══════════════════════════════════╗
        // ║  Se a AMÉLIA ainda estiver falando, descarta todos  ║
        // ║  os resultados — eles são eco da própria fala.      ║
        // ╚════════════════════════════════════════════════════╝
        if (_voiceIsSpeechActive() || Date.now() < voiceState.echoBlockUntil) return;

        const finalParts = [];
        const interimParts = [];
        for (let i = 0; i < event.results.length; i++) {
            const seg = _voiceCleanTranscript(event.results[i][0].transcript);
            if (!seg) continue;
            event.results[i].isFinal
                ? finalParts.push(seg)
                : interimParts.push(seg);
        }

        voiceState.transcript = _voiceDedupeTranscript(finalParts.join(" "));
        const interimText = _voiceDedupeTranscript(interimParts.join(" "));

        // Mostra em tempo real no input
        const inp = document.getElementById("chat-input");
        if (inp) inp.value = [voiceState.transcript, interimText].filter(Boolean).join(" ");
    };

    rec.onerror = (event) => {
        const msgs = {
            "no-speech":     "Não detectei sua voz. Tente falar mais perto do microfone.",
            "audio-capture": "Microfone não encontrado.",
            "not-allowed":   "Permissão de microfone negada. Habilite nas configurações do navegador.",
            "network":       "Erro de rede.",
        };
        showToast(msgs[event.error] || `Erro no microfone: ${event.error}`, "error");
        stopRecording();
    };

    rec.onend = () => {
        // Reinicia automaticamente se ainda estiver em modo gravação
        if (voiceState.isRecording && voiceState.recognition === rec && !_voiceIsSpeechActive()) {
            const delay = Math.max(0, voiceState.echoBlockUntil - Date.now());
            setTimeout(() => {
                if (voiceState.isRecording && voiceState.recognition === rec) {
                    try { rec.start(); } catch { /* ignora se já parou */ }
                }
            }, delay);
        }
    };

    return rec;
}


// ═══════════════════════════════════════════════════════════════
// CONTROLE DE GRAVAÇÃO
// ═══════════════════════════════════════════════════════════════

function startRecording() {
    if (!checkVoiceSupport()) return;

    if (voiceState.isRecording || voiceState.recognition) return;

    if (_voiceIsSpeechActive()) {
        _setFeedback("Preparando microfone...");
        _voiceCancelSpeechOutput();
    }

    const delay = Math.max(0, voiceState.echoBlockUntil - Date.now());
    if (delay > 0) {
        _voiceClearPendingStart();
        voiceState.pendingStartTimer = setTimeout(() => {
            voiceState.pendingStartTimer = null;
            _doStartRecording();
        }, delay);
        return;
    }

    _doStartRecording();
}

function _doStartRecording() {
    if (voiceState.isRecording || voiceState.recognition) return;
    _voiceClearPendingStart();
    voiceState.transcript  = "";
    voiceState.recognition = _createRecognition();
    try {
        voiceState.recognition.start();
    } catch (err) {
        voiceState.recognition = null;
        voiceState.isRecording = false;
        _updateMicBtn(false);
        showToast("Não foi possível iniciar o microfone: " + err.message, "error");
    }
}

function stopRecording() {
    _voiceClearPendingStart();
    voiceState.isRecording = false;
    if (voiceState.recognition) {
        try { voiceState.recognition.stop(); } catch { /* ja encerrado */ }
        voiceState.recognition = null;
    }
    _updateMicBtn(false);
    _setFeedback("Gravação encerrada.");
}

/**
 * Toggle do botão de voz.
 * Ao parar, processa o texto transcrito como uma resposta de texto livre.
 */
function toggleRecording() {
    if (voiceState.isRecording) {
        stopRecording();
        const inp = document.getElementById("chat-input");
        const t = (voiceState.transcript || inp?.value || "").trim();
        if (t) {
            // Injeta no input e envia como texto (aproveita toda a lógica do script.js)
            if (inp) {
                inp.value    = t;
                inp.disabled = false;
            }
            // Chama sendMessage() do script.js — ele extrai as features corretamente
            if (typeof sendMessage === "function") sendMessage();
        } else {
            _setFeedback("Nenhum áudio detectado. Tente novamente.");
        }
    } else if (voiceState.pendingStartTimer) {
        _voiceClearPendingStart();
        _setFeedback("Microfone cancelado.");
    } else {
        startRecording();
    }
}


// ═══════════════════════════════════════════════════════════════
// ATUALIZAÇÃO DA INTERFACE
// ═══════════════════════════════════════════════════════════════

function _updateMicBtn(recording) {
    const btn = document.getElementById("btn-voice");
    if (!btn) return;
    btn.textContent = recording ? "Stop" : "Mic";
    btn.title       = recording ? "Parar gravação" : "Clique para falar";
    btn.setAttribute("aria-label", recording ? "Parar gravação de voz" : "Iniciar gravação de voz");
    btn.classList.toggle("recording", recording);
}

function _setFeedback(msg) {
    const el = document.getElementById("voice-feedback");
    if (el) el.textContent = msg;
}


// ═══════════════════════════════════════════════════════════════
// INICIALIZAÇÃO
// ═══════════════════════════════════════════════════════════════
document.addEventListener("DOMContentLoaded", () => {
    checkVoiceSupport();
});
