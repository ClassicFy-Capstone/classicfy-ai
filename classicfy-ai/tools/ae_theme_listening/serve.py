"""Serve the listening page from this file's directory, regardless of cwd."""

import argparse
import json
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from string import Template
from urllib.parse import parse_qs, urlsplit


ROOT = Path(__file__).resolve().parent
STORAGE_PREFIX = "classicfy-theme-choice-v2-"
MAX_TRANSFER_BYTES = 6 * 1024 * 1024


def script_json(value):
    return json.dumps(value, ensure_ascii=True).replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def parse_transfer(body):
    if len(body) > MAX_TRANSFER_BYTES:
        raise ValueError("Transfer too large")
    fields = parse_qs(body.decode("utf-8"), strict_parsing=True)
    if set(fields) != {"state", "next"} or any(len(values) != 1 for values in fields.values()):
        raise ValueError("Invalid transfer fields")
    state, path = json.loads(fields["state"][0]), fields["next"][0]
    if not isinstance(state, dict) or any(not isinstance(key, str) or not key.startswith(STORAGE_PREFIX)
                                          or not isinstance(value, str) for key, value in state.items()):
        raise ValueError("Only listening response storage can be transferred")
    parsed = urlsplit(path)
    if not path.startswith("/") or path.startswith("//") or parsed.scheme or parsed.netloc or "\\" in path or any(ord(c) < 32 for c in path):
        raise ValueError("Invalid return path")
    return state, path


def migration_page(port):
    # Top-level POST avoids third-party localStorage partitioning in an iframe.
    # Only this app's keys are read; their original copies remain untouched.
    return '''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="referrer" content="strict-origin-when-cross-origin">
<title>Classicfy 주소 이동</title><p id="message">재생 가능한 localhost 주소로 이동합니다. 기존 응답을 보존합니다.</p>
<a id="continue" href="/">localhost에서 계속</a><script>
const canonical=__CANONICAL__,prefix=__PREFIX__;
const next=location.pathname+location.search+location.hash;
document.querySelector('#continue').href=canonical+next;
try {
  const state={};
  for(let i=0;i<localStorage.length;i++){const key=localStorage.key(i);if(key.startsWith(prefix))state[key]=localStorage.getItem(key);}
  if(!Object.keys(state).length)location.replace(canonical+next);
  else {
    const form=document.createElement('form');form.method='POST';form.action=canonical+'/__origin_transfer';
    for(const [name,value] of Object.entries({state:JSON.stringify(state),next})){const input=document.createElement('input');input.type='hidden';input.name=name;input.value=value;form.append(input);}
    document.body.append(form);form.submit();
  }
}catch(_){document.querySelector('#message').textContent='이전 주소의 응답을 읽지 못했습니다. 원래 저장본은 남아 있습니다. 아래 링크로 계속할 수 있습니다.';}
</script></html>'''.replace("__CANONICAL__", script_json(f"http://localhost:{port}")).replace("__PREFIX__", script_json(STORAGE_PREFIX))


def transfer_page(state, path):
    return Template('''<!doctype html><html lang="ko"><meta charset="utf-8"><title>Classicfy 응답 복사</title>
<p id="message">기존 응답을 복사하고 있습니다.</p><a id="continue" href="/">localhost에서 계속</a><script>
const state=$state,next=$next,backupKey='classicfy-theme-origin-backups-v1';
document.querySelector('#continue').href=next;
try {
  const raw=localStorage.getItem(backupKey);let backups={};
  if(raw){try{backups=JSON.parse(raw);if(!backups||typeof backups!=='object'||Array.isArray(backups))throw Error();}catch(_){backups={unreadable_previous_backup:[raw]};}}
  const answerCount=value=>{try{const parsed=JSON.parse(value);return parsed&&parsed.responses&&typeof parsed.responses==='object'&&!Array.isArray(parsed.responses)?Object.keys(parsed.responses).length:null;}catch(_){return null;}};
  const writes=[];
  for(const [key,value] of Object.entries(state)){
    const current=localStorage.getItem(key);
    if(current===null)writes.push([key,value]);
    else if(current!==value){
      const preferOld=answerCount(current)===0&&answerCount(value)>0,copy=preferOld?current:value;
      if(!Array.isArray(backups[key]))backups[key]=[];if(!backups[key].includes(copy))backups[key].push(copy);
      if(preferOld)writes.push([key,value]);
    }
  }
  if(Object.keys(backups).length)localStorage.setItem(backupKey,JSON.stringify(backups));
  for(const [key,value] of writes)localStorage.setItem(key,value);
  location.replace(next);
}catch(_){document.querySelector('#message').textContent='응답 복사를 완료하지 못했습니다. 127.0.0.1 주소의 원래 응답은 보존돼 있습니다. 아래 링크로 계속할 수 있습니다.';}
</script></html>''').substitute(state=script_json(state), next=script_json(path))


class ListeningHandler(SimpleHTTPRequestHandler):
    def is_ip_origin(self):
        return urlsplit("http://" + self.headers.get("Host", "")).hostname == "127.0.0.1"

    def write_html(self, value):
        body = value.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Referrer-Policy", "strict-origin-when-cross-origin")
        self.end_headers()
        self.wfile.write(body)

    def redirect_to_localhost(self):
        self.send_response(302)
        self.send_header("Location", f"http://localhost:{self.server.server_port}{self.path}")
        self.send_header("Content-Length", "0")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

    def do_GET(self):
        if self.is_ip_origin():
            if urlsplit(self.path).path in {"/", "/index.html"}:
                self.write_html(migration_page(self.server.server_port))
            else:
                self.redirect_to_localhost()
            return
        super().do_GET()

    def do_HEAD(self):
        if self.is_ip_origin():
            self.redirect_to_localhost()
        else:
            super().do_HEAD()

    def do_POST(self):
        port = self.server.server_port
        if self.path != "/__origin_transfer":
            self.send_error(404)
            return
        if self.headers.get("Host") != f"localhost:{port}" or self.headers.get("Origin") != f"http://127.0.0.1:{port}":
            self.send_error(403)
            return
        if self.headers.get("Content-Type", "").split(";", 1)[0] != "application/x-www-form-urlencoded":
            self.send_error(415)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= MAX_TRANSFER_BYTES:
                self.send_error(413)
                return
            state, path = parse_transfer(self.rfile.read(length))
        except (ValueError, UnicodeError):
            self.send_error(400)
            return
        self.write_html(transfer_page(state, path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    missing = [name for name in ("index.html", "data.js", "embed_checks.js", "study.js", "player.js", "app.js", "style.css")
               if not (ROOT / name).is_file()]
    if missing:
        parser.error(f"Missing listening page files in {ROOT}: {', '.join(missing)}")
    handler = partial(ListeningHandler, directory=str(ROOT))
    try:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    except OSError as error:
        parser.exit(1, f"Could not start the listening server: {error}\n")
    with server:
        print(f"Listening page: http://localhost:{args.port}/", flush=True)
        print(f"Files: {ROOT} (Ctrl+C to stop)", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
