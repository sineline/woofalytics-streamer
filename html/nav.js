/**
 * Woofalytics — shared nav component.
 * Drop <script src="/nav.js"></script> anywhere in <body> and this
 * self-injects a sticky top navbar with consistent links + live bark status.
 */
(function () {
  // ── Styles ──────────────────────────────────────────────────────────────────
  const css = `
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700&display=swap');

    .woof-nav {
      position: fixed; top: 0; left: 0; right: 0; height: 60px;
      background: rgba(7, 8, 15, 0.82);
      backdrop-filter: blur(18px); -webkit-backdrop-filter: blur(18px);
      border-bottom: 1px solid rgba(255,255,255,0.07);
      display: flex; align-items: center;
      padding: 0 24px; gap: 0;
      z-index: 9999;
      font-family: 'Outfit', sans-serif;
      box-shadow: 0 2px 24px rgba(0,0,0,0.4);
    }

    .woof-nav-brand {
      display: flex; align-items: center; gap: 8px;
      font-size: 1.05rem; font-weight: 700;
      text-decoration: none; color: #e2e8f0;
      flex-shrink: 0; margin-right: 20px;
      letter-spacing: -0.3px;
    }
    .woof-nav-brand em { color: #f59e0b; font-style: normal; }
    .woof-nav-brand .paw { font-size: 1.2rem; }

    .woof-nav-links {
      display: flex; gap: 2px;
    }
    .woof-nav-link {
      color: #64748b;
      text-decoration: none;
      font-size: 0.84rem; font-weight: 500;
      padding: 6px 13px; border-radius: 8px;
      border: 1px solid transparent;
      transition: color 0.18s, border-color 0.18s, background 0.18s;
      white-space: nowrap;
    }
    .woof-nav-link:hover {
      color: #f59e0b;
      border-color: rgba(245,158,11,0.25);
      background: rgba(245,158,11,0.05);
    }
    .woof-nav-link.woof-active {
      color: #f59e0b;
      border-color: rgba(245,158,11,0.5);
      background: rgba(245,158,11,0.08);
    }

    .woof-nav-sep {
      width: 1px; height: 22px;
      background: rgba(255,255,255,0.08);
      margin: 0 10px; flex-shrink: 0;
    }

    .woof-nav-status {
      margin-left: auto;
      display: flex; align-items: center; gap: 8px;
      background: rgba(255,255,255,0.04);
      border: 1px solid rgba(255,255,255,0.08);
      border-radius: 99px;
      padding: 5px 14px 5px 10px;
      font-size: 0.79rem; font-weight: 500; color: #e2e8f0;
      flex-shrink: 0;
      transition: border-color 0.3s, background 0.3s;
    }
    .woof-nav-status.woof-barking {
      border-color: rgba(239,68,68,0.5);
      background: rgba(239,68,68,0.08);
    }

    .woof-sdot {
      width: 8px; height: 8px; border-radius: 50%;
      background: #22c55e;
      transition: background 0.3s;
      flex-shrink: 0;
    }
    .woof-sdot.woof-barking {
      background: #ef4444;
      animation: woof-blink 0.45s infinite alternate;
    }
    @keyframes woof-blink { from{opacity:1;transform:scale(1)} to{opacity:0.3;transform:scale(0.7)} }

    /* Offset page content so it doesn't hide under the fixed nav */
    body { padding-top: 68px !important; }

    @media(max-width: 640px) {
      .woof-nav-link { padding: 5px 8px; font-size: 0.76rem; }
      .woof-nav-brand { margin-right: 10px; }
    }
  `;

  const styleEl = document.createElement('style');
  styleEl.textContent = css;
  document.head.appendChild(styleEl);

  // ── Active link detection ────────────────────────────────────────────────────
  const path = window.location.pathname;
  const LINKS = [
    { href: '/',          label: 'Dashboard' },
    { href: '/analytics', label: 'Analytics' },
    { href: '/library',   label: 'Library'   },
    { href: '/stream',    label: '📺 Stream'  },
    { href: '/train',     label: '🧠 Train'   },
    { href: '/debug',     label: 'Debug'     },
    { href: '/config',    label: 'Config'    },
    { href: '/rec',       label: 'Record'    },
  ];

  const linksHTML = LINKS.map(l => {
    const active = (path === l.href || (l.href !== '/' && path.startsWith(l.href)));
    return `<a href="${l.href}" class="woof-nav-link${active ? ' woof-active' : ''}">${l.label}</a>`;
  }).join('');

  // ── DOM injection ────────────────────────────────────────────────────────────
  const nav = document.createElement('nav');
  nav.className = 'woof-nav';
  nav.id = 'woof-main-nav';
  nav.innerHTML = `
    <a class="woof-nav-brand" href="/">
      <span class="paw">🐾</span>
      Woof<em>alytics</em>
    </a>
    <div class="woof-nav-sep"></div>
    <div class="woof-nav-links">${linksHTML}</div>
    <div class="woof-nav-status" id="woof-nav-status">
      <div class="woof-sdot" id="woof-sdot"></div>
      <span id="woof-stext">Listening</span>
    </div>
  `;

  // Insert as very first child of <body>
  document.body.insertBefore(nav, document.body.firstChild);

  // ── Live bark status polling ──────────────────────────────────────────────────
  async function pollNav() {
    try {
      const r = await fetch('/api/bark');
      const d = await r.json();
      const barking = (d.bark_probability || 0) >= 0.88;
      const dot    = document.getElementById('woof-sdot');
      const txt    = document.getElementById('woof-stext');
      const status = document.getElementById('woof-nav-status');
      if (!dot) return;
      if (barking) {
        dot.className    = 'woof-sdot woof-barking';
        txt.textContent  = '🔊 Barking!';
        status.className = 'woof-nav-status woof-barking';
      } else {
        dot.className    = 'woof-sdot';
        txt.textContent  = 'Listening';
        status.className = 'woof-nav-status';
      }
    } catch (_) {}
  }

  pollNav();
  setInterval(pollNav, 2500);
})();
