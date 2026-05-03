/**
 * animations.js — AMÉLIA Premium Interaction Layer
 * ==================================================
 * Implementa física de mola, hover magnético, cursor customizado,
 * animações em cascata e paralaxe — tudo a 60fps via RAF.
 *
 * Filosofia: nenhuma animação usa `ease` genérico.
 * Toda curva é derivada de comportamento físico real.
 */

'use strict';

/* ─────────────────────────────────────────────────────────────
   1. CUSTOM CURSOR — segue o mouse com mola amortecida (spring)
   Rigidez: 0.12 | Amortecimento: implícito pelo lerp
   ───────────────────────────────────────────────────────────── */
class SpringCursor {
  constructor() {
    this.el      = document.getElementById('cursor');
    this.dot     = document.getElementById('cursor-dot');
    this.pos     = { x: window.innerWidth / 2, y: window.innerHeight / 2 };
    this.target  = { x: window.innerWidth / 2, y: window.innerHeight / 2 };
    this.raf     = null;
    this.visible = false;

    // Não renderizar em touch-only devices
    if (!this.el || window.matchMedia('(pointer: coarse)').matches) return;

    this._bind();
    this._tick();
  }

  _bind() {
    document.addEventListener('mousemove', e => {
      this.target.x = e.clientX;
      this.target.y = e.clientY;
      if (!this.visible) {
        this.visible = true;
        this.el.style.opacity = '1';
        if (this.dot) this.dot.style.opacity = '1';
      }
    });

    document.addEventListener('mouseleave', () => {
      this.visible = false;
      this.el.style.opacity = '0';
      if (this.dot) this.dot.style.opacity = '0';
    });

    // Cursor incha em elementos interativos
    const interactives = 'button, a, input, select, [data-magnetic], [role=button], label';
    document.addEventListener('mouseover', e => {
      if (e.target.closest(interactives)) {
        this.el.classList.add('cursor--hover');
      } else {
        this.el.classList.remove('cursor--hover');
      }
    });
  }

  _tick() {
    /* Spring lerp: mola com rigidez 0.12 (suave) para o anel externo
       O dot central usa rigidez 0.6 (rígido) para sensação de precisão */
    this.pos.x += (this.target.x - this.pos.x) * 0.10;
    this.pos.y += (this.target.y - this.pos.y) * 0.10;

    if (this.el) {
      this.el.style.transform =
        `translate3d(${this.pos.x}px, ${this.pos.y}px, 0)`;
    }
    // Dot segue direto (snap) — contraste entre os dois dá sensação de massa
    if (this.dot) {
      this.dot.style.transform =
        `translate3d(${this.target.x}px, ${this.target.y}px, 0)`;
    }

    this.raf = requestAnimationFrame(() => this._tick());
  }
}


/* ─────────────────────────────────────────────────────────────
   2. MAGNETIC HOVER — elementos atraem o cursor
   Simulação de campo magnético: força decai com a distância
   ───────────────────────────────────────────────────────────── */
function initMagnetic() {
  document.querySelectorAll('[data-magnetic]').forEach(el => {
    const strength = parseFloat(el.dataset.magnetic) || 0.35;
    let rect = null;

    const onEnter = () => { rect = el.getBoundingClientRect(); };

    const onMove = e => {
      if (!rect) rect = el.getBoundingClientRect();
      const cx = rect.left + rect.width  / 2;
      const cy = rect.top  + rect.height / 2;
      const dx = e.clientX - cx;
      const dy = e.clientY - cy;

      /* Decaimento gaussiano: força máxima no centro,
         atenua suavemente nas bordas */
      const dist    = Math.sqrt(dx * dx + dy * dy);
      const maxDist = Math.max(rect.width, rect.height);
      const factor  = Math.max(0, 1 - dist / maxDist);

      // Transição instantânea enquanto em cima (segue o mouse)
      el.style.transition = 'transform 0.08s linear';
      el.style.transform  = `translate(${dx * strength * factor}px, ${dy * strength * factor}px)`;
    };

    const onLeave = () => {
      /* Retorno com mola: overshoot (1.56) → settle */
      el.style.transition = 'transform 0.65s cubic-bezier(0.34, 1.56, 0.64, 1)';
      el.style.transform  = '';
      rect = null;
      setTimeout(() => { el.style.transition = ''; }, 700);
    };

    el.addEventListener('mouseenter', onEnter);
    el.addEventListener('mousemove',  onMove);
    el.addEventListener('mouseleave', onLeave);
  });
}


/* ─────────────────────────────────────────────────────────────
   3. STAGGER ENTRANCE — Intersection Observer
   Elementos aparecem em diagonal (não todos juntos)
   Atraso: 80ms por índice filho
   ───────────────────────────────────────────────────────────── */
function initStagger() {
  const io = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      if (!entry.isIntersecting) return;

      const children = entry.target.querySelectorAll('[data-stagger]');
      children.forEach((child, i) => {
        /* Cada filho herda o índice como variável CSS.
           O CSS usa --i para calcular o delay, permitindo
           reuso do mesmo keyframe com timings diferentes. */
        child.style.setProperty('--i', i);
        child.classList.add('is-visible');
      });

      io.unobserve(entry.target);
    });
  }, { threshold: 0.08, rootMargin: '0px 0px -40px 0px' });

  document.querySelectorAll('[data-stagger-group]').forEach(g => io.observe(g));
}


/* ─────────────────────────────────────────────────────────────
   4. HERO TEXT SPLIT — quebra cada letra em <span>
   Permite animação letra por letra com delay em cascata diagonal
   ───────────────────────────────────────────────────────────── */
function splitHeroLines() {
  document.querySelectorAll('.hero__line').forEach((line, lineIdx) => {
    const raw  = line.textContent.trim();
    const html = [...raw].map((ch, i) => {
      /* Índice diagonal: combina linha + coluna para criar
         onda que vai da esquerda-cima para direita-baixo */
      const diagIdx = lineIdx * 3 + i;
      const char = ch === ' ' ? '&nbsp;' : ch;
      return `<span class="char" aria-hidden="true"
                style="--char:${diagIdx}">${char}</span>`;
    }).join('');
    line.innerHTML = html;
    /* Texto real mantido em aria-label para acessibilidade */
    line.setAttribute('aria-label', raw);
  });
}


/* ─────────────────────────────────────────────────────────────
   5. ORB PARALLAX — orbs do hero reagem ao movimento do mouse
   Intensidade diferente por orb cria ilusão de profundidade (z-layers)
   ───────────────────────────────────────────────────────────── */
function initOrbParallax() {
  const orbs  = document.querySelectorAll('.orb');
  const hero  = document.querySelector('.hero__visual');
  if (!hero || !orbs.length) return;

  let raf = null;
  let mx  = 0, my = 0;

  document.addEventListener('mousemove', e => {
    /* Normaliza mouse em relação ao centro da viewport (-1 a +1) */
    mx = (e.clientX / window.innerWidth  - 0.5) * 2;
    my = (e.clientY / window.innerHeight - 0.5) * 2;
  });

  const tick = () => {
    orbs.forEach((orb, i) => {
      /* Cada orb tem profundidade diferente: z-layer implícito */
      const depth = (i + 1) * 6;
      const tx    = mx * depth;
      const ty    = my * depth * 0.6; // Y tem menos deslocamento (perspectiva)

      orb.style.transform =
        `translate(${tx}px, ${ty}px)`;
    });
    raf = requestAnimationFrame(tick);
  };

  tick();
}


/* ─────────────────────────────────────────────────────────────
   6. TILT CARDS — painéis inclinam ligeiramente com o mouse
   Simula cartão físico sendo segurado e girado
   ───────────────────────────────────────────────────────────── */
function initTiltCards() {
  document.querySelectorAll('[data-tilt]').forEach(card => {
    const maxTilt = parseFloat(card.dataset.tilt) || 6; // graus máximos

    card.addEventListener('mousemove', e => {
      const rect = card.getBoundingClientRect();
      const cx   = rect.left + rect.width  / 2;
      const cy   = rect.top  + rect.height / 2;
      const rx   = ((e.clientY - cy) / (rect.height / 2)) * -maxTilt;
      const ry   = ((e.clientX - cx) / (rect.width  / 2)) *  maxTilt;

      card.style.transition = 'transform 0.1s linear';
      card.style.transform  =
        `perspective(600px) rotateX(${rx}deg) rotateY(${ry}deg) scale(1.015)`;
    });

    card.addEventListener('mouseleave', () => {
      /* Retorno suave com mola — ligeiro oversshoot antes de estabilizar */
      card.style.transition = 'transform 0.8s cubic-bezier(0.34, 1.56, 0.64, 1)';
      card.style.transform  = '';
    });
  });
}


/* ─────────────────────────────────────────────────────────────
   7. REACTIVE INPUT — inputs pulsam levemente ao receber foco
   Feedback tátil: o campo "respira" quando ativado
   ───────────────────────────────────────────────────────────── */
function initReactiveInputs() {
  document.querySelectorAll('.field input, .field select').forEach(inp => {
    inp.addEventListener('focus', () => {
      inp.parentElement?.classList.add('field--active');
    });
    inp.addEventListener('blur', () => {
      inp.parentElement?.classList.remove('field--active');
    });
  });
}


/* ─────────────────────────────────────────────────────────────
   8. RE-INIT — chamado quando o DOM muda (mudança de tela)
   Garante que novos elementos também tenham magnetic/tilt
   ───────────────────────────────────────────────────────────── */
function reinitInteractions() {
  initMagnetic();
  initTiltCards();
  initStagger();
  initReactiveInputs();
}

/* Observa mudanças no DOM (script.js altera o header, cria
   novos botões, etc.) e re-aplica as interações */
const _domObserver = new MutationObserver(mutations => {
  const hasNewElements = mutations.some(m => m.addedNodes.length > 0);
  if (hasNewElements) {
    // Debounce: espera 50ms para o DOM estabilizar antes de re-init
    clearTimeout(_domObserver._tid);
    _domObserver._tid = setTimeout(reinitInteractions, 50);
  }
});


/* ─────────────────────────────────────────────────────────────
   INIT — executado após DOM carregado
   ───────────────────────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', () => {
  new SpringCursor();
  initMagnetic();
  initStagger();
  splitHeroLines();
  initOrbParallax();
  initTiltCards();
  initReactiveInputs();

  _domObserver.observe(document.body, {
    childList:  true,
    subtree:    true,
    attributes: false,
  });
});

/* Expõe reinit globalmente para script.js chamar após update de header */
window.ameliaReinit = reinitInteractions;
