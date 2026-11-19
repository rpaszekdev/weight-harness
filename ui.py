"""Browser chat for hello-llm. Run: python ui.py, then open http://localhost:8765

Stdlib server: GET / serves the page, POST /chat streams the model's reply token by token.
"""
import html
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from model import generate, load, summary

PORT = 8765
MAX_BODY = 4096
MAX_CHARS = 300

PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>hello-llm</title>
<style>
:root{--bg:#f6f5f1;--panel:#fff;--ink:#1d1d1b;--muted:#77756f;--line:#e3e1da;--me:#1d1d1b;--me-ink:#fff;--accent:#2f6f4f}
@media (prefers-color-scheme:dark){:root{--bg:#121211;--panel:#1b1b1a;--ink:#ecebe6;--muted:#8f8d86;--line:#2c2b29;
  --me:#ecebe6;--me-ink:#121211;--accent:#7fc59b}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 system-ui,-apple-system,sans-serif;
  height:100dvh;display:flex;justify-content:center}
main{width:100%;max-width:680px;display:flex;flex-direction:column;padding:16px;min-height:0}
h1{font:600 18px ui-monospace,Menlo,monospace;margin:0}
header p{margin:2px 0 12px;color:var(--muted);font-size:13px}
#log{flex:1;overflow-y:auto;display:flex;flex-direction:column;gap:8px;padding:12px;
  background:var(--panel);border:1px solid var(--line);border-radius:12px}
.msg{max-width:85%;padding:8px 12px;border-radius:12px;white-space:pre-wrap;overflow-wrap:anywhere}
.me{align-self:flex-end;background:var(--me);color:var(--me-ink)}
.bot{align-self:flex-start;border:1px solid var(--line);font:14px/1.45 ui-monospace,Menlo,monospace}
.bot.typing::after{content:"\\258D";color:var(--accent);animation:blink 1s steps(2) infinite}
@keyframes blink{50%{opacity:0}}
.meta{align-self:flex-start;color:var(--muted);font-size:11px;margin-top:-4px}
#chips{display:flex;flex-wrap:wrap;gap:6px;margin:10px 0}
#chips button{border:1px solid var(--line);background:var(--panel);color:var(--ink);border-radius:999px;
  padding:4px 10px;font:inherit;font-size:13px;cursor:pointer}
form{display:flex;gap:8px}
input{flex:1;min-width:0;padding:10px 12px;border-radius:10px;border:1px solid var(--line);
  background:var(--panel);color:var(--ink);font:inherit}
form button{padding:10px 16px;border:0;border-radius:10px;background:var(--accent);color:var(--bg);
  font:600 14px system-ui;cursor:pointer}
button:disabled{opacity:.5;cursor:default}
</style></head>
<body><main>
<header><h1>hello-llm</h1><p>__STATS__</p></header>
<div id="log" aria-live="polite"><div class="meta">no training: 703 facts, every weight set by hand. ask about countries, animals or opposites.</div></div>
<div id="chips"></div>
<form id="form"><input id="q" placeholder="say something..." maxlength="__MAX__" autocomplete="off" autofocus>
<button id="send">send</button></form>
</main>
<script>
const log = document.getElementById("log"), q = document.getElementById("q"), send = document.getElementById("send");
const EXAMPLES = ["hello", "what is the capital of peru?", "countries in europe", "which animals live in the ocean?",
                  "is ottawa in europe?", "what currency does japan use?", "capitals u know", "is a lion a herbivore?",
                  "which countries use the euro?", "capital of wakanda?"];
let busy = false;

function add(cls, text) {
  const d = document.createElement("div");
  d.className = cls;
  d.textContent = text;
  log.append(d);
  log.scrollTop = log.scrollHeight;
  return d;
}

async function ask(text) {
  if (busy) return;
  busy = send.disabled = true;
  add("msg me", text || "(empty)");
  const bot = add("msg bot typing", ""), t0 = performance.now();
  try {
    const r = await fetch("/chat", {method: "POST", headers: {"Content-Type": "application/json"},
                                    body: JSON.stringify({message: text})});
    if (!r.ok) throw new Error("server answered " + r.status);
    const reader = r.body.getReader(), dec = new TextDecoder();
    for (;;) {
      const {done, value} = await reader.read();
      if (done) break;
      bot.textContent = (bot.textContent + dec.decode(value, {stream: true})).trimStart();
      log.scrollTop = log.scrollHeight;
    }
    add("meta", `${bot.textContent.length} chars, ${Math.round(performance.now() - t0)} ms`);
  } catch (e) {
    bot.textContent = "error: " + e.message + " (is ui.py still running?)";
  }
  bot.classList.remove("typing");
  busy = send.disabled = false;
  q.focus();
}

for (const ex of EXAMPLES) {
  const b = document.createElement("button");
  b.type = "button";
  b.textContent = ex;
  b.onclick = () => ask(ex);
  document.getElementById("chips").append(b);
}
document.getElementById("form").onsubmit = e => {
  e.preventDefault();
  const text = q.value;
  q.value = "";
  ask(text);
};
</script></body></html>
"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != "/":
            return self.send_error(404)
        body = self.server.page
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path != "/chat":
            return self.send_error(404)
        n = int(self.headers.get("Content-Length") or 0)
        if not 0 < n <= MAX_BODY:
            return self.send_error(413 if n else 400)
        try:
            msg = json.loads(self.rfile.read(n))["message"]
            assert isinstance(msg, str)
        except (ValueError, KeyError, TypeError, AssertionError):
            return self.send_error(400, 'expected JSON {"message": "..."}')
        self.send_response(200)  # HTTP/1.0: the body ends when the connection closes, so chars stream
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.end_headers()
        for tok in generate(self.server.weights, msg[:MAX_CHARS]):
            self.wfile.write(tok.encode())
            self.wfile.flush()

    def log_message(self, *args):
        pass


def main():
    try:
        weights = load()
    except FileNotFoundError:
        raise SystemExit("no weights.npz yet: run python build.py first")
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    server.weights = weights
    server.page = PAGE.replace("__STATS__", html.escape(summary(weights))).replace("__MAX__", str(MAX_CHARS)).encode()
    print(f"hello-llm chat on http://localhost:{PORT}  (ctrl-c stops)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
