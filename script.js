/* ═══════════════════════════════════════════════════════════════
   script.js — AMÉLIA v3.0
   Chat estruturado: botões de escolha + multi-seleção + voz/texto
   Emergencia sem senha; demais fluxos: U001 / M001 / L001
   ═══════════════════════════════════════════════════════════════ */

// ── Estado global ─────────────────────────────────────────────
let currentUser = null;

// chatData armazena tanto os textos (para PDF) quanto os valores
// numéricos/booleanos (para o modelo ML)
let chatData = {
    currentStep:    0,
    answers:        {},   // texto humano de cada resposta (para PDF)
    features:       {},   // valores estruturados para o modelo ML
    classification: null,
    password:       null,
    apiResult:      null,
    resultColor:    null,
    resultText:     "",
    waitTime:       "",
};

// ── Definição das perguntas ────────────────────────────────────
//
// type: "text"    → só digitação / voz
// type: "single"  → botões de escolha única (+ digitação/voz disponíveis)
// type: "multi"   → checkboxes de múltipla escolha (+ digitação/voz)
// type: "skip"    → igual a text, mas com botão "Não tenho mais informações"
//
const QUESTIONS = [
    // ── 0. Saudação ───────────────────────────────────────────
    {
        id:   "greeting",
        text: "Olá! Eu sou a AMÉLIA, sua assistente de triagem médica.\n\nPara começar, como você está se sentindo de forma geral?",
        type: "text",
    },

    // ── 1. Nível de dor ───────────────────────────────────────
    {
        id:   "pain_level",
        text: "Em uma escala de <strong>0 a 10</strong>, qual é o nível da sua dor ou desconforto?",
        type: "single",
        hint: "Ou digite / fale um número de 0 a 10",
        options: [
            { label: "😊 0 — Sem dor",          value: 0  },
            { label: "😌 1–2 — Muito leve",      value: 1  },
            { label: "🙁 3–4 — Leve",            value: 3  },
            { label: "😣 5–6 — Moderada",        value: 5  },
            { label: "😫 7–8 — Forte",           value: 7  },
            { label: "😭 9–10 — Insuportável",   value: 9  },
        ],
    },

    // ── 2. Sintomas (multi-seleção) ───────────────────────────
    {
        id:   "symptoms",
        text: "Selecione <strong>todos os sintomas</strong> que você está sentindo agora:",
        type: "multi",
        hint: "Ou descreva com suas palavras digitando / falando",
        confirmLabel: "Confirmar seleção",
        options: [
            { label: "🌡️ Febre",                        key: "fever"                },
            { label: "😮‍💨 Falta de ar",                 key: "shortness_of_breath"  },
            { label: "💔 Dor no peito",                  key: "chest_pain"           },
            { label: "🤮 Vômito ou náusea intensa",      key: "vomiting"             },
            { label: "🤕 Dor de cabeça muito forte",     key: "severe_headache"      },
            { label: "🤧 Reação alérgica",               key: "allergic_reaction"    },
            { label: "🩸 Sangramento",                   key: "bleeding"             },
            { label: "😵 Desmaio ou confusão mental",    key: "altered_consciousness"},
            { label: "🚑 Trauma ou acidente",            key: "trauma"               },
            { label: "✋ Nenhum dos acima",              key: "none"                 },
        ],
    },

    // ── 3. Duração ────────────────────────────────────────────
    {
        id:   "duration",
        text: "Há quanto tempo esses sintomas começaram?",
        type: "single",
        hint: "Ou escreva / fale: ex. '2 horas', '3 dias'",
        options: [
            { label: "⚡ Há minutos",          value: 0.25 },
            { label: "🕐 Menos de 6 horas",    value: 3    },
            { label: "🕕 6 a 24 horas",        value: 12   },
            { label: "📅 1 a 3 dias",          value: 48   },
            { label: "📅 4 a 7 dias",          value: 120  },
            { label: "📆 Mais de 1 semana",    value: 200  },
        ],
    },

    // ── 4. Informações adicionais ─────────────────────────────
    {
        id:       "additional",
        text:     "Tem mais alguma informação importante?\n(Medicamentos em uso, alergias conhecidas, histórico médico…)",
        type:     "skip",
        skipLabel:"Não tenho mais informações",
    },
];

// ── Perguntas cujas respostas extraímos features diretas ───────
const SYMPTOM_KEY_TO_FEATURE = {
    fever:                "fever",
    shortness_of_breath:  "shortness_of_breath",
    chest_pain:           "chest_pain",
    vomiting:             "vomiting",
    severe_headache:      "severe_headache",
    allergic_reaction:    "allergic_reaction",
    bleeding:             "bleeding",
    altered_consciousness:"altered_consciousness",
    trauma:               "trauma",
};


// ═══════════════════════════════════════════════════════════════
// INICIALIZAÇÃO
// ═══════════════════════════════════════════════════════════════
document.addEventListener("DOMContentLoaded", () => {
    const saved = localStorage.getItem("ameliaUser");
    if (saved) {
        try {
            currentUser = JSON.parse(saved);
            _updateHeader();
        } catch { localStorage.removeItem("ameliaUser"); }
    }
    _initMasks();
    _fixLogo();
    _revealScreenStagger(document.querySelector(".screen.active"));
});


// ═══════════════════════════════════════════════════════════════
// NAVEGAÇÃO
// ═══════════════════════════════════════════════════════════════
function showScreen(name) {
    document.querySelectorAll(".screen").forEach(s => s.classList.remove("active"));
    const el = document.getElementById(`screen-${name}`);
    if (el) {
        el.classList.add("active");
        _revealScreenStagger(el);
    }
    if (name === "painel" && currentUser) {
        checkAPIStatus();   // declarado em voice.js
        initChat();
    }
}

function _revealScreenStagger(screen) {
    if (!screen) return;
    const items = screen.querySelectorAll("[data-stagger]");
    items.forEach((item, i) => {
        item.style.transitionDelay = `${Math.min(i * 55, 420)}ms`;
        item.classList.add("is-visible");
    });
}

function _updateHeader() {
    const nav = document.getElementById("nav-buttons");
    if (!nav) return;
    const nome = currentUser.nome.split(" ")[0];
    nav.innerHTML = `
        <span style="font-weight:600;color:var(--c-blue)">Olá, ${nome}</span>
        <button class="btn btn-ghost" onclick="showScreen('painel')">Triagem</button>
        <button class="btn btn-ghost" onclick="showScreen('sobre')">Sobre</button>
        <button class="btn btn-ghost" onclick="_logout()">Sair</button>
    `;
}

function _logout() {
    currentUser = null;
    localStorage.removeItem("ameliaUser");
    location.reload();
}


// ═══════════════════════════════════════════════════════════════
// MÁSCARAS DE INPUT
// ═══════════════════════════════════════════════════════════════
function _initMasks() {
    document.querySelectorAll("#cad-cpf, #login-id").forEach(el => {
        el.addEventListener("input", e => {
            let v = e.target.value.replace(/\D/g, "").slice(0, 11);
            v = v.replace(/(\d{3})(\d)/, "$1.$2")
                 .replace(/(\d{3})(\d)/, "$1.$2")
                 .replace(/(\d{3})(\d{1,2})$/, "$1-$2");
            e.target.value = v;
        });
    });
    const sus = document.getElementById("cad-sus");
    if (sus) sus.addEventListener("input", e => {
        let v = e.target.value.replace(/\D/g, "").slice(0, 15);
        v = v.replace(/(\d{3})(\d)/, "$1 $2")
             .replace(/(\d{4})(\d)/, "$1 $2")
             .replace(/(\d{4})(\d)/, "$1 $2");
        e.target.value = v;
    });
    const tel = document.getElementById("cad-telefone");
    if (tel) tel.addEventListener("input", e => {
        let v = e.target.value.replace(/\D/g, "").slice(0, 11);
        v = v.replace(/(\d{2})(\d)/, "($1) $2")
             .replace(/(\d{5})(\d)/, "$1-$2");
        e.target.value = v;
    });
}


// ═══════════════════════════════════════════════════════════════
// VALIDAÇÕES
// ═══════════════════════════════════════════════════════════════
function _validateCPF(cpf) {
    cpf = cpf.replace(/\D/g, "");
    if (cpf.length !== 11 || /^(\d)\1{10}$/.test(cpf)) return false;
    let s = 0;
    for (let i = 0; i < 9; i++) s += +cpf[i] * (10 - i);
    let d1 = 11 - (s % 11); if (d1 > 9) d1 = 0;
    s = 0;
    for (let i = 0; i < 10; i++) s += +cpf[i] * (11 - i);
    let d2 = 11 - (s % 11); if (d2 > 9) d2 = 0;
    return +cpf[9] === d1 && +cpf[10] === d2;
}
function _validateSUS(sus) { return /^\d{15}$/.test(sus.replace(/\D/g, "")); }
async function _hash(pwd) {
    const buf = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(pwd));
    return [...new Uint8Array(buf)].map(b => b.toString(16).padStart(2,"0")).join("");
}


// ═══════════════════════════════════════════════════════════════
// CADASTRO
// ═══════════════════════════════════════════════════════════════
async function handleCadastro(e) {
    e.preventDefault();
    const nome    = document.getElementById("cad-nome").value.trim();
    const cpf     = document.getElementById("cad-cpf").value;
    const sus     = document.getElementById("cad-sus").value;
    const nasc    = document.getElementById("cad-nascimento").value;
    const sexo    = document.getElementById("cad-sexo").value;
    const tel     = document.getElementById("cad-telefone").value;
    const senha   = document.getElementById("cad-senha").value;
    const confirm = document.getElementById("cad-senha-confirm").value;

    if (!_validateCPF(cpf))  { showToast("CPF inválido.", "error"); return; }
    if (!_validateSUS(sus))  { showToast("Cartão SUS inválido (15 dígitos).", "error"); return; }
    if (senha !== confirm)   { showToast("As senhas não coincidem.", "error"); return; }
    if (!sexo)               { showToast("Selecione o sexo biológico.", "error"); return; }

    const users = JSON.parse(localStorage.getItem("ameliaUsers") || "[]");
    const cpfC  = cpf.replace(/\D/g,"");
    const susC  = sus.replace(/\D/g,"");
    if (users.find(u => u.cpf === cpfC || u.sus === susC)) {
        showToast("CPF ou SUS já cadastrado.", "error"); return;
    }

    const h = await _hash(senha);
    users.push({ nome, cpf: cpfC, sus: susC, nascimento: nasc, sexo, telefone: tel, h });
    localStorage.setItem("ameliaUsers", JSON.stringify(users));
    showToast("Conta criada! Faça login.", "success");
    document.getElementById("form-cadastro").reset();
    setTimeout(() => showScreen("login"), 1400);
}


// ═══════════════════════════════════════════════════════════════
// LOGIN
// ═══════════════════════════════════════════════════════════════
async function handleLogin(e) {
    e.preventDefault();
    const id   = document.getElementById("login-id").value.replace(/\D/g,"");
    const pwd  = document.getElementById("login-senha").value;
    const users= JSON.parse(localStorage.getItem("ameliaUsers") || "[]");
    const h    = await _hash(pwd);
    const user = users.find(u => (u.cpf===id || u.sus===id) && u.h===h);

    if (user) {
        currentUser = user;
        localStorage.setItem("ameliaUser", JSON.stringify(user));
        showToast("Bem-vindo(a), " + user.nome.split(" ")[0] + "!", "success");
        _updateHeader();
        setTimeout(() => showScreen("painel"), 900);
    } else {
        showToast("CPF/SUS ou senha incorretos.", "error");
    }
}


// ═══════════════════════════════════════════════════════════════
// CHAT — INICIALIZAÇÃO
// ═══════════════════════════════════════════════════════════════
function initChat() {
    chatData = {
        currentStep: 0,
        answers:     {},
        features: {
            // defaults — sobrescritos à medida que o paciente responde
            pain_level:            0,
            fever:                 false,
            shortness_of_breath:   false,
            chest_pain:            false,
            altered_consciousness: false,
            bleeding:              false,
            duration_hours:        24,
            vomiting:              false,
            severe_headache:       false,
            allergic_reaction:     false,
            trauma:                false,
        },
        classification: null,
        password:       null,
        apiResult:      null,
        resultColor:    null,
        resultText:     "",
        waitTime:       "",
    };

    document.getElementById("chat-messages").innerHTML = "";
    document.getElementById("chat-input-area").style.display = "";
    document.getElementById("chat-actions").style.display    = "none";

    const fb = document.getElementById("voice-feedback");
    if (fb) fb.textContent = "";

    // Reativa o input
    _enableInput();

    setTimeout(() => {
        _showQuestion(QUESTIONS[0]);
    }, 400);
}

function _ameliaAvatarHTML() {
    return `<div class="av"><img src="AmeliaCVPS.png" alt="AMELIA" onerror="this.remove();this.parentElement.textContent='A';"></div>`;
}

function _plainText(text) {
    return String(text ?? "")
        .replace(/<[^>]*>/g, " ")
        .replace(/&nbsp;/g, " ")
        .replace(/&amp;/g, "&")
        .replace(/&lt;/g, "<")
        .replace(/&gt;/g, ">")
        .replace(/\s+/g, " ")
        .trim();
}

function _stripEmoji(text) {
    return String(text ?? "")
        .replace(/[\u{1F000}-\u{1FAFF}\u{2600}-\u{27BF}\uFE0F\u200D]/gu, "")
        .replace(/\s+/g, " ")
        .trim();
}

function _recordText(text) {
    return _stripEmoji(_plainText(text))
        .replace(/[✅✔]/g, "")
        .replace(/\s+/g, " ")
        .trim();
}

function _speechText(text) {
    return _recordText(text)
        .replace(/[≤]/g, "menor ou igual a ")
        .replace(/[≥]/g, "maior ou igual a ")
        .replace(/[–—]/g, ". ")
        .replace(/[•·]/g, ". ")
        .replace(/[^\p{L}\p{N}\s.,;:!?%/()-]/gu, " ")
        .replace(/\s+/g, " ")
        .trim();
}

function _normText(text) {
    return _recordText(text)
        .toLowerCase()
        .normalize("NFD")
        .replace(/[\u0300-\u036f]/g, "");
}

function _pdfSafe(text) {
    return _recordText(text)
        .replace(/[≤]/g, "<=")
        .replace(/[≥]/g, ">=")
        .replace(/[–—]/g, "-")
        .replace(/→/g, "->")
        .replace(/[•·]/g, "-")
        .replace(/…/g, "...")
        .normalize("NFD")
        .replace(/[\u0300-\u036f]/g, "")
        .replace(/[^\x20-\x7E]/g, "")
        .replace(/\s+/g, " ")
        .trim();
}


// ═══════════════════════════════════════════════════════════════
// CHAT — RENDERIZAÇÃO DE PERGUNTAS
// ═══════════════════════════════════════════════════════════════

/** Exibe a pergunta atual com a UI adequada ao seu tipo. */
function _showQuestion(q) {
    // Fala a pergunta SOMENTE se o microfone não estiver gravando
    // (fix do bug de áudio — voice.js gerencia isso também)
    const speaking = typeof voiceState !== "undefined"
        && (voiceState.isRecording || voiceState.pendingStartTimer);
    if (!speaking && typeof speak === "function") {
        // Remove HTML para a síntese de voz
        speak(_speechText(q.text));
    }

    _botMsgWithExtras(q);
    _enableInput();

    // Atualiza placeholder do input conforme o tipo
    const inp = document.getElementById("chat-input");
    if (inp) {
        inp.placeholder = q.hint
            || (q.type === "text" || q.type === "skip"
                ? "Digite sua resposta ou clique no microfone"
                : "Ou escreva / fale sua resposta…");
    }
}

/**
 * Insere a mensagem do bot + os controles interativos (botões, checkboxes).
 * Usa o indicador de "digitando…" antes de mostrar o conteúdo.
 */
function _botMsgWithExtras(q) {
    const box = document.getElementById("chat-messages");

    // Indicador de digitação
    const typing = document.createElement("div");
    typing.className = "message message-bot";
    typing.innerHTML = `${_ameliaAvatarHTML()}
        <div class="typing-indicator"><span></span><span></span><span></span></div>`;
    box.appendChild(typing);
    _scrollBottom();

    setTimeout(() => {
        typing.remove();

        // Balão de texto
        const msg = document.createElement("div");
        msg.className = "message message-bot";
        msg.innerHTML = `${_ameliaAvatarHTML()}
            <div class="message-content">${q.text.replace(/\n/g,"<br>")}</div>`;
        box.appendChild(msg);

        // Controles extras conforme tipo
        if (q.type === "single") {
            box.appendChild(_buildSingleChoiceUI(q));
        } else if (q.type === "multi") {
            box.appendChild(_buildMultiChoiceUI(q));
        } else if (q.type === "skip") {
            box.appendChild(_buildSkipUI(q));
        }

        _scrollBottom();
    }, 900 + Math.random() * 300);
}

/** Botões de escolha única (pain_level, duration) */
function _buildSingleChoiceUI(q) {
    const wrap = document.createElement("div");
    wrap.className = "choice-wrap";
    wrap.id = `choices-${q.id}`;

    q.options.forEach(opt => {
        const btn = document.createElement("button");
        btn.className = "choice-btn";
        btn.textContent = opt.label;
        btn.onclick = () => _handleSingleChoice(q, opt, wrap);
        wrap.appendChild(btn);
    });

    return wrap;
}

/** Grid de checkboxes + botão confirmar (symptoms) */
function _buildMultiChoiceUI(q) {
    const wrap = document.createElement("div");
    wrap.className = "multi-wrap";
    wrap.id = `multi-${q.id}`;

    const grid = document.createElement("div");
    grid.className = "multi-grid";

    const selectedKeys = new Set();

    q.options.forEach(opt => {
        const label = document.createElement("label");
        label.className = "multi-item";

        const cb = document.createElement("input");
        cb.type  = "checkbox";
        cb.value = opt.key;

        cb.onchange = () => {
            // "Nenhum dos acima" desmarca todos os outros
            if (opt.key === "none" && cb.checked) {
                grid.querySelectorAll("input[type=checkbox]").forEach(c => {
                    if (c.value !== "none") { c.checked = false; selectedKeys.delete(c.value); }
                });
                selectedKeys.add("none");
            } else if (opt.key !== "none") {
                // Desmarcar "nenhum" se marcar qualquer outro
                const noneBox = grid.querySelector("input[value=none]");
                if (noneBox) { noneBox.checked = false; selectedKeys.delete("none"); }
                cb.checked ? selectedKeys.add(opt.key) : selectedKeys.delete(opt.key);
            }
            label.classList.toggle("multi-item--selected", cb.checked);
        };

        const span = document.createElement("span");
        span.textContent = opt.label;

        label.appendChild(cb);
        label.appendChild(span);
        grid.appendChild(label);
    });

    const confirmBtn = document.createElement("button");
    confirmBtn.className = "btn btn-solid choice-confirm";
    confirmBtn.textContent = q.confirmLabel || "Confirmar seleção";
    confirmBtn.onclick = () => _handleMultiConfirm(q, selectedKeys, wrap);

    wrap.appendChild(grid);
    wrap.appendChild(confirmBtn);
    return wrap;
}

/** Botão "Não tenho mais informações" para pergunta skip */
function _buildSkipUI(q) {
    const wrap = document.createElement("div");
    wrap.className = "choice-wrap";
    wrap.id = `skip-${q.id}`;

    const btn = document.createElement("button");
    btn.className  = "choice-btn choice-btn--secondary";
    btn.textContent = q.skipLabel || "Não tenho mais informações";
    btn.onclick = () => {
        _lockChoiceUI(wrap);
        _processAnswer(q, btn.textContent, {});
    };
    wrap.appendChild(btn);
    return wrap;
}


// ═══════════════════════════════════════════════════════════════
// CHAT — PROCESSAMENTO DE RESPOSTAS
// ═══════════════════════════════════════════════════════════════

/** Chamado ao clicar em um botão de escolha única */
function _handleSingleChoice(q, opt, wrap) {
    _lockChoiceUI(wrap, opt.label);

    const featureUpdates = {};

    if (q.id === "pain_level") {
        featureUpdates.pain_level = opt.value;
    } else if (q.id === "duration") {
        featureUpdates.duration_hours = opt.value;
    }

    _processAnswer(q, opt.label, featureUpdates);
}

/** Chamado ao confirmar seleção múltipla de sintomas */
function _handleMultiConfirm(q, selectedKeys, wrap) {
    if (selectedKeys.size === 0) {
        showToast("Selecione ao menos uma opção ou descreva por texto/voz.", "warning");
        return;
    }

    // Monta texto legível
    const labels = [];
    const featureUpdates = {};

    q.options.forEach(opt => {
        if (selectedKeys.has(opt.key)) {
            labels.push(opt.label);
            const feat = SYMPTOM_KEY_TO_FEATURE[opt.key];
            if (feat) featureUpdates[feat] = true;
        }
    });

    // "Nenhum dos acima" → todas features false
    if (selectedKeys.has("none")) {
        Object.keys(SYMPTOM_KEY_TO_FEATURE).forEach(k => featureUpdates[k] = false);
    }

    const text = selectedKeys.has("none")
        ? "Nenhum sintoma adicional"
        : labels.join(", ");

    _lockChoiceUI(wrap, text);
    _processAnswer(q, text, featureUpdates);
}

/** Bloqueia a UI de escolha após seleção, evitando duplo clique */
function _lockChoiceUI(wrap, selectedLabel) {
    if (selectedLabel) {
        wrap.innerHTML = `<div class="choice-selected">Selecionado: ${_esc(_recordText(selectedLabel))}</div>`;
    } else {
        wrap.style.display = "none";
    }
}

/**
 * Ponto central de processamento de qualquer resposta
 * (botão, multi-select ou texto livre / voz).
 *
 * @param {object} q              - Pergunta atual
 * @param {string} textAnswer     - Texto legível para o usuário (e PDF)
 * @param {object} featureUpdates - Valores estruturados para o modelo ML
 */
function _processAnswer(q, textAnswer, featureUpdates) {
    const cleanAnswer = _recordText(textAnswer) || "Sem resposta registrada";

    // Salva texto legível
    chatData.answers[q.id] = cleanAnswer;

    // Atualiza features estruturadas
    Object.assign(chatData.features, featureUpdates);

    // Exibe mensagem do usuário no chat
    addUserMessage(cleanAnswer);

    // Avança para próxima pergunta
    chatData.currentStep++;
    _nextQuestion();
}

/** Vai para a próxima pergunta ou finaliza */
function _nextQuestion() {
    const acks = [
        "Entendo. Obrigada por compartilhar.",
        "Sinto muito que esteja passando por isso.",
        "Agradeço pela confiança.",
        "Compreendo. Vamos continuar.",
        "Anotado. Mais uma pergunta.",
    ];

    if (chatData.currentStep < QUESTIONS.length) {
        setTimeout(() => {
            _botMsg(acks[Math.floor(Math.random() * acks.length)]);
            setTimeout(() => {
                _showQuestion(QUESTIONS[chatData.currentStep]);
            }, 1100);
        }, 500);
    } else {
        _finishChat();
    }
}


// ═══════════════════════════════════════════════════════════════
// CHAT — INPUT DE TEXTO (digitação e voz)
// ═══════════════════════════════════════════════════════════════

function sendMessage() {
    const inp = document.getElementById("chat-input");
    const msg = inp.value.trim();
    if (!msg) return;
    inp.value = "";
    _disableInput();

    const q = QUESTIONS[chatData.currentStep];
    if (!q) return;

    // Fecha qualquer UI de choice que ainda esteja aberta
    const choiceUI = document.getElementById(`choices-${q.id}`)
                  || document.getElementById(`multi-${q.id}`)
                  || document.getElementById(`skip-${q.id}`);
    if (choiceUI) _lockChoiceUI(choiceUI);

    // Extrai features do texto para cada tipo de pergunta
    const featureUpdates = _extractFeaturesFromText(q.id, msg);
    _processAnswer(q, msg, featureUpdates);
}

/**
 * Tenta extrair valores estruturados de uma resposta em texto livre.
 * Garante que mesmo respostas digitadas alimentem o modelo corretamente.
 */
function _extractFeaturesFromText(questionId, text) {
    const t = _normText(text);
    const updates = {};

    if (questionId === "pain_level") {
        const m = t.match(/\b(10|[0-9])\b/);
        updates.pain_level = m ? parseInt(m[1]) :
            /insuportavel|horrivel|terrivel|agonia/i.test(t) ? 10 :
            /muito forte|intensa|severa/i.test(t) ? 8 :
            /forte|consideravel/i.test(t) ? 7 :
            /moderada|media|razoavel/i.test(t) ? 5 :
            /leve|fraca|pouca/i.test(t) ? 3 :
            /sem dor|nao doi|nenhuma/i.test(t) ? 0 : 5;

    } else if (questionId === "symptoms") {
        updates.fever                = /febre|febril|temperatura alta/i.test(t);
        updates.shortness_of_breath  = /falta de ar|sem ar|dificuldade.*respir|sufocando|dispneia/i.test(t);
        updates.chest_pain           = /dor no peito|aperto.*peito|dor toracica|coracao doendo|infarto/i.test(t);
        updates.vomiting             = /vomito|vomitando|nausea intensa|ansia/i.test(t);
        updates.severe_headache      = /dor de cabeca|cefaleia|enxaqueca|cabeca latejando/i.test(t);
        updates.allergic_reaction    = /alergi|anafilaxia|urticaria|coceira.*corpo|picada|inchaco/i.test(t);
        updates.bleeding             = /sangue|sangrando|sangramento|hemorragia/i.test(t);
        updates.altered_consciousness= /desmaiei|desmaio|convulsao|desorientado|confuso|perdi.*consciencia/i.test(t);
        updates.trauma               = /acidente|cai|queda|bateu|bati|trauma|colisao|machucou|fratura/i.test(t);

    } else if (questionId === "duration") {
        let dur = 24;
        const mSem = t.match(/(\d+(?:[.,]\d+)?)\s*semana/);
        const mDia = t.match(/(\d+(?:[.,]\d+)?)\s*dia/);
        const mHor = t.match(/(\d+(?:[.,]\d+)?)\s*hora/);
        const mMin = t.match(/(\d+(?:[.,]\d+)?)\s*minuto/);
        if (mSem)            dur = parseFloat(mSem[1].replace(",",".")) * 168;
        else if (mDia)       dur = parseFloat(mDia[1].replace(",",".")) * 24;
        else if (mHor)       dur = Math.max(0.5, parseFloat(mHor[1].replace(",",".")));
        else if (mMin)       dur = Math.max(0.25, parseFloat(mMin[1].replace(",",".")) / 60);
        else if (/agora|minutos/.test(t))  dur = 0.25;
        else if (/hoje cedo|manha/.test(t)) dur = 4;
        else if (/ontem/.test(t))           dur = 24;
        else if (/dias/.test(t))            dur = 72;
        else if (/semanas/.test(t))         dur = 336;
        updates.duration_hours = dur;
    }

    return updates;
}

function _enableInput() {
    const inp = document.getElementById("chat-input");
    const btn = document.getElementById("chat-send");
    const avoidMobileKeyboard = window.matchMedia
        && window.matchMedia("(max-width: 640px), (pointer: coarse)").matches;
    if (inp) {
        inp.disabled = false;
        if (!avoidMobileKeyboard) inp.focus();
    }
    if (btn)   btn.disabled = false;
    if (inp) inp.onkeypress = ev => { if (ev.key === "Enter") sendMessage(); };
}

function _disableInput() {
    const inp = document.getElementById("chat-input");
    const btn = document.getElementById("chat-send");
    if (inp) inp.disabled = true;
    if (btn) btn.disabled = true;
}


// ═══════════════════════════════════════════════════════════════
// CHAT — FINALIZAÇÃO E CLASSIFICAÇÃO
// ═══════════════════════════════════════════════════════════════

function _finishChat() {
    _disableInput();
    setTimeout(() => {
        _botMsg("Obrigada pelas informações! ⏳ Classificando com IA…");
        setTimeout(async () => {
            const online = typeof voiceState !== "undefined" && voiceState.apiOnline;
            if (online) {
                await _classifyViaAPI();
            } else {
                _classifyLocal();
            }
        }, 1800);
    }, 700);
}

// ── Via API (usa features estruturadas coletadas pelos botões) ─
async function _classifyViaAPI() {
    try {
        const age = currentUser ? _calcAge(currentUser.nascimento) : 30;
        const sex = currentUser?.sexo || "M";
        const cpf = currentUser?.cpf  || "00000000000";

        // Monta descrição textual completa para o prontuário
        const description = Object.entries(chatData.answers)
            .map(([k, v]) => `${k}: ${v}`)
            .join(". ");

        const structuredPayload = {
            cpf,
            age,
            sex,
            description,
            ...chatData.features,          // todos os valores estruturados dos botões
            duration_hours: chatData.features.duration_hours ?? 24,
        };

        let res = await fetch(`${API_BASE_URL}/api/triagem`, {
            method:  "POST",
            headers: { "Content-Type": "application/json" },
            body:    JSON.stringify({
                texto: description,
                cpf,
                age,
                sex,
                use_bert: true,
            }),
        });

        // Fallback para o contrato antigo caso o backend NLP nao esteja publicado.
        if (!res.ok) {
            res = await fetch(`${API_BASE_URL}/api/classify`, {
                method:  "POST",
                headers: { "Content-Type": "application/json" },
                body:    JSON.stringify(structuredPayload),
            });
        }

        if (!res.ok) throw new Error("HTTP " + res.status);

        const result = await res.json();
        const color = result.classification.color;
        const password = color === "red" ? null : result.password;
        chatData.apiResult      = result;
        chatData.password       = password;
        chatData.classification = _colorToLetter(color);
        _showResult(
            color,
            result.classification.explanation,
            password,
            result.classification.confidence,
            result.classification.wait_time
        );
    } catch (err) {
        console.warn("API falhou, usando classificação local:", err);
        _classifyLocal();
    }
}

// ── Classificação local (fallback sem backend) ─────────────────
function _classifyLocal() {
    const f = chatData.features;
    const pain = f.pain_level ?? 5;

    let color;
    if (f.altered_consciousness
        || (f.shortness_of_breath && f.chest_pain)
        || pain >= 9
        || (f.allergic_reaction && f.shortness_of_breath)
        || (f.trauma && f.bleeding && pain >= 7)) {
        color = "red";
    } else if (pain >= 7 || f.shortness_of_breath || f.chest_pain
               || f.bleeding || f.allergic_reaction
               || (f.fever && pain >= 6)
               || (f.severe_headache && pain >= 7)) {
        color = "orange";
    } else if (pain >= 4 || f.fever || f.vomiting || f.severe_headache) {
        color = "yellow";
    } else {
        color = "green";
    }

    const exps = {
        red:    "Emergência — atendimento imediato por médico(a) ou enfermeiro(a).",
        orange: "Urgente — atendimento em breve.",
        yellow: "Moderado — fila prioritária.",
        green:  "Pouco urgente — fila regular.",
    };
    const waits = { red:"Imediato", orange:"<= 10 min", yellow:"<= 30 min", green:"<= 2 h" };

    chatData.classification = _colorToLetter(color);
    chatData.password       = _localPwd(color);
    _showResult(color, exps[color], chatData.password, null, waits[color]);
}

function _riskPresentation(color) {
    const map = {
        red:    { label:"Emergência",     instruction:"Atendimento imediato por médico(a) ou enfermeiro(a).", rgb:[214,59,59] },
        orange: { label:"Urgente",        instruction:"Atendimento em breve.", rgb:[249,115,22] },
        yellow: { label:"Moderado",       instruction:"Fila prioritária.", rgb:[234,179,8] },
        green:  { label:"Pouco urgente",  instruction:"Fila regular.", rgb:[22,163,74] },
    };
    return map[color] || map.green;
}

/** Exibe o resultado final no chat e lê em voz alta */
function _showResult(color, explanation, password, confidence, waitTime) {
    const risk = _riskPresentation(color);
    const isEmergency = color === "red";
    const cleanExplanation = _recordText(explanation) || `${risk.label}. ${risk.instruction}`;
    const prioClass = { red:"prio-red", orange:"prio-orange",
                        yellow:"prio-yellow", green:"prio-green" }[color] || "prio-green";
    const finalPassword = isEmergency ? null : password;

    chatData.resultColor = color;
    chatData.resultText  = cleanExplanation;
    chatData.waitTime    = waitTime || "";
    chatData.password    = finalPassword;

    const confLine = confidence != null
        ? `<small style="color:var(--c-muted)">Confiança da IA: ${(confidence*100).toFixed(1)}%</small>`
        : `<small style="color:var(--c-muted)">Classificação local (backend offline)</small>`;
    const resultBlock = isEmergency
        ? `<div class="emergency-alert">
               <div class="emergency-icon">!</div>
               <div class="emergency-text">
                   <strong>Sem senha de espera</strong><br>
                   Atendimento imediato por médico(a) ou enfermeiro(a).
               </div>
           </div>`
        : `<div class="senha-card senha-card--${color}">
               <span>Sua senha de atendimento</span>
               <strong>${_esc(finalPassword)}</strong>
               <small>${risk.label}</small>
           </div>`;

    _botMsg(`
        <span class="prio-badge ${prioClass}">${_esc(cleanExplanation)}</span><br>
        Tempo de espera: <strong>${_esc(waitTime || "—")}</strong><br><br>
        ${resultBlock}<br>
        ${confLine}<br>
        Baixe o prontuário PDF e apresente no guichê, se solicitado.
    `);

    if (typeof speak === "function") {
        const spoken = isEmergency
            ? `Triagem concluída. Emergência. Atendimento imediato por médico ou enfermeiro. Sem senha de espera.`
            : `Triagem concluída. ${cleanExplanation}. Tempo de espera: ${waitTime}. Sua senha é: ${String(finalPassword).split("").join(" ")}.`;
        speak(spoken);
    }

    document.getElementById("chat-input-area").style.display = "none";
    document.getElementById("chat-actions").style.display    = "flex";
}


// ═══════════════════════════════════════════════════════════════
// MENSAGENS DO CHAT
// ═══════════════════════════════════════════════════════════════

function _botMsg(html) {
    const box = document.getElementById("chat-messages");
    const typing = document.createElement("div");
    typing.className = "message message-bot";
    typing.innerHTML = `${_ameliaAvatarHTML()}
        <div class="typing-indicator"><span></span><span></span><span></span></div>`;
    box.appendChild(typing);
    _scrollBottom();

    setTimeout(() => {
        typing.remove();
        const d = document.createElement("div");
        d.className = "message message-bot";
        d.innerHTML = `${_ameliaAvatarHTML()}
            <div class="message-content">${html}</div>`;
        box.appendChild(d);
        _scrollBottom();
    }, 850 + Math.random() * 300);
}

// Alias público usado por voice.js
function addBotMessage(html)  { _botMsg(html); }

function addUserMessage(text) {
    const box = document.getElementById("chat-messages");
    const d   = document.createElement("div");
    d.className = "message message-user";
    d.innerHTML = `<div class="message-content">${_esc(text)}</div>`;
    box.appendChild(d);
    _scrollBottom();
}


// ═══════════════════════════════════════════════════════════════
// GERAÇÃO DE PDF
// ═══════════════════════════════════════════════════════════════
function generatePDF() {
    if (typeof window.jspdf === "undefined") {
        showToast("Recarregue a página para gerar o PDF.", "error"); return;
    }
    const { jsPDF } = window.jspdf;
    const doc  = new jsPDF();
    const now  = new Date();
    const date = now.toLocaleDateString("pt-BR");
    const time = now.toLocaleTimeString("pt-BR");
    const W    = 210, M = 18;
    const color = chatData.resultColor || "green";
    const risk = _riskPresentation(color);
    const [rr, gg, bb] = risk.rgb;
    const safe = _pdfSafe;

    const ensurePage = (needed = 18) => {
        if (y + needed > 270) {
            doc.addPage();
            y = 20;
        }
    };

    // Cabeçalho azul
    doc.setFillColor(13, 71, 161);
    doc.rect(0,0,W,38,"F");
    doc.setFont("helvetica","bold"); doc.setFontSize(20); doc.setTextColor(255,255,255);
    doc.text(safe("PRONTUARIO DE TRIAGEM"), W/2, 16, {align:"center"});
    doc.setFont("helvetica","normal"); doc.setFontSize(11);
    doc.text(safe("A.M.E.L.I.A - Sistema de Triagem por Inteligencia Artificial"), W/2, 27, {align:"center"});

    // Dados do paciente
    let y = 48;
    doc.setFontSize(8); doc.setTextColor(100,116,139);
    doc.text(safe("DADOS DO PACIENTE"), M, y); y += 5;
    doc.setFontSize(10); doc.setTextColor(26,32,53); doc.setFont("helvetica","normal");
    doc.text(safe(`Nome: ${currentUser?.nome || "-"}`), M, y); y += 6;
    doc.text(safe(`CPF: ${_maskCPF(currentUser?.cpf || "")}`), M, y);
    doc.text(safe(`SUS: ${_maskSUS(currentUser?.sus || "")}`), 110, y); y += 6;
    doc.text(safe(`Data/Hora: ${date} as ${time}`), M, y);

    y += 8; doc.setDrawColor(221,227,238); doc.line(M,y,W-M,y); y += 7;

    // Sintomas estruturados
    doc.setFontSize(8); doc.setTextColor(100,116,139); doc.text(safe("SINTOMAS IDENTIFICADOS"), M, y); y += 5;
    doc.setFontSize(10); doc.setTextColor(26,32,53);

    const f = chatData.features;
    const symptomRows = [
        [`Nivel de dor: ${f.pain_level ?? "-"}/10`, `Duracao: ${f.duration_hours ?? "-"}h`],
        [`Febre: ${f.fever?"Sim":"Não"}`,             `Falta de ar: ${f.shortness_of_breath?"Sim":"Não"}`],
        [`Dor no peito: ${f.chest_pain?"Sim":"Não"}`, `Vômito: ${f.vomiting?"Sim":"Não"}`],
        [`Cefaleia intensa: ${f.severe_headache?"Sim":"Não"}`,`Sangramento: ${f.bleeding?"Sim":"Não"}`],
        [`Reação alérgica: ${f.allergic_reaction?"Sim":"Não"}`,`Trauma/Acidente: ${f.trauma?"Sim":"Não"}`],
        [`Consciência alterada: ${f.altered_consciousness?"Sim":"Não"}`, ""],
    ];
    symptomRows.forEach(([a,b]) => {
        doc.setFont("helvetica","normal");
        doc.text(safe(a), M, y);
        if (b) doc.text(safe(b), 110, y);
        y += 6;
    });

    y += 4; doc.line(M,y,W-M,y); y += 7;

    // Relatos textuais
    doc.setFontSize(8); doc.setTextColor(100,116,139); doc.text(safe("RELATO DO PACIENTE"), M, y); y += 5;
    QUESTIONS.forEach((q, i) => {
        const ans = chatData.answers[q.id];
        if (!ans || q.id === "greeting") return;
        const lines = doc.splitTextToSize(safe(`${_plainText(q.text)} -> ${ans}`), W-M*2);
        doc.setFontSize(9); doc.setTextColor(26,32,53); doc.setFont("helvetica","normal");
        ensurePage(lines.length * 5 + 8);
        doc.text(lines, M, y); y += lines.length * 5 + 3;
    });

    // Resultado
    ensurePage(48);
    y += 4; doc.line(M,y,W-M,y); y += 8;
    doc.setFillColor(rr, gg, bb);
    doc.rect(M, y, W - M * 2, 34, "F");
    doc.setTextColor(255,255,255);
    doc.setFont("helvetica","bold"); doc.setFontSize(11);
    doc.text(safe(`CLASSIFICACAO: ${risk.label.toUpperCase()}`), M + 5, y + 8);
    doc.setFontSize(color === "red" ? 15 : 20);
    const resultLine = color === "red"
        ? "SEM SENHA - ATENDIMENTO IMEDIATO"
        : `SENHA: ${chatData.password || "-"}`;
    doc.text(safe(resultLine), M + 5, y + 21);
    doc.setFont("helvetica","normal"); doc.setFontSize(9);
    doc.text(safe(chatData.resultText || risk.instruction), M + 5, y + 29);
    y += 42;
    doc.setTextColor(26,32,53);
    doc.setFont("helvetica","bold"); doc.setFontSize(10);
    doc.text(safe(`TEMPO DE ESPERA: ${chatData.waitTime || "-"}`), M, y);

    // Rodapé
    doc.setFontSize(8); doc.setTextColor(150,150,150); doc.setFont("helvetica","italic");
    doc.text(safe("Apresente no guiche, se solicitado - A.M.E.L.I.A v3.0 - FeNaDANTE 2025"), W/2, 285, {align:"center"});

    const fn = `Prontuario_${safe(currentUser?.nome||"paciente").replace(/\s+/g,"_")}_${date.replace(/\//g,"-")}.pdf`;
    doc.save(fn);
    showToast("Prontuário baixado!", "success");
}


// ═══════════════════════════════════════════════════════════════
// AUXILIARES
// ═══════════════════════════════════════════════════════════════

// Contadores locais de senha (fallback sem backend)
const _pwdCnt = { U:0, M:0, L:0 };
function _localPwd(color) {
    if (color === "red") return null;
    const p = color==="orange" ? "U" : color==="yellow" ? "M" : "L";
    _pwdCnt[p]++;
    return `${p}${String(_pwdCnt[p]).padStart(3,"0")}`;
}

function _colorToLetter(c) {
    return c==="red"||c==="orange" ? "U" : c==="yellow" ? "M" : "L";
}

function _calcAge(d) {
    if (!d) return 30;
    const b = new Date(d), t = new Date();
    const a = t.getFullYear() - b.getFullYear();
    return t < new Date(t.getFullYear(), b.getMonth(), b.getDate()) ? a-1 : a;
}

function _maskCPF(c) {
    if (c.length!==11) return c;
    return `${c.slice(0,3)}.${c.slice(3,6)}.${c.slice(6,9)}-${c.slice(9)}`;
}
function _maskSUS(s) {
    if (s.length!==15) return s;
    return `${s.slice(0,3)} ${s.slice(3,7)} ${s.slice(7,11)} ${s.slice(11)}`;
}
function _esc(str) {
    return String(str)
        .replace(/&/g,"&amp;").replace(/</g,"&lt;")
        .replace(/>/g,"&gt;").replace(/"/g,"&quot;");
}
function _scrollBottom() {
    const el = document.getElementById("chat-messages");
    if (el) el.scrollTop = el.scrollHeight;
}
function _fixLogo() {
    const l = document.getElementById("logo-img");
    if (!l) return;
    l.onerror = () => {
        l.style.display = "none";
        const lt = document.querySelector(".brand-name");
        if (lt) lt.textContent = "A.M.E.L.I.A";
    };
}
function resetChat() {
    showToast("Iniciando nova triagem…", "success");
    setTimeout(initChat, 450);
}
function showToast(msg, type="success") {
    const t = document.getElementById("toast");
    t.textContent = msg;
    t.className = `toast ${type} show`;
    clearTimeout(t._tid);
    t._tid = setTimeout(() => t.classList.remove("show"), 3200);
}
