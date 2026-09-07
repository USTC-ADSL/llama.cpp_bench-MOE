#!/usr/bin/env python3
"""Loopback-only deterministic browser task; records only this test's state."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json, pathlib, time, urllib.parse

ROOT=pathlib.Path(__file__).resolve().parents[1]
OUT=ROOT/'docs/Motivation_Exp/Static_App/evidence'
PAGE='''<!doctype html><html lang="en"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Library workshop task</title>
<style>body{font:20px sans-serif;margin:20px;background:#fff;color:#123}button,input{font:22px sans-serif;padding:18px;margin:10px 0;width:95%}#status{font-weight:bold;color:#065}p{line-height:1.5}</style>
<h1>Library workshop</h1><div id="status">READING — no draft prepared</div>
<button onclick="arm()">Prepare workshop draft</button><input id="draft" placeholder="Unsubmitted volunteer note" autocomplete="off">
<button onclick="verify()">Verify and continue task</button>
<h2>Saturday volunteer briefing</h2>
<p>The public library repair workshop runs from 09:00 until 13:00. Visitors take their belongings home on the same day. The purpose is to teach one practical repair skill.</p>
<p>Mira teaches sewing, Daniel checks bicycles, Helen advises on safety, and Sam manages arrivals. Electrical repairs are outside the authorised scope. Keep noisy activities away from the neighbouring reading session between 10:00 and 11:00.</p>
<p>Prepare a volunteer note before switching to the calculator to estimate the cost of materials. Return to this page afterwards and verify that the unsubmitted note is still ready.</p>
<script>
const run=new URLSearchParams(location.search).get('run')||'preflight';
const generation=crypto.randomUUID?crypto.randomUUID():String(Date.now());
let armed=false, continued=0;
function emit(event){fetch('/event',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({run,generation,event,armed,continued,draft_ok:document.querySelector('#draft').value==='Mira: sewing; Daniel: bicycles; close at 13:00',visible:!document.hidden,t_ms:performance.now(),time_origin:performance.timeOrigin})}).catch(()=>{});}
function arm(){document.querySelector('#draft').value='Mira: sewing; Daniel: bicycles; close at 13:00';armed=true;document.querySelector('#status').textContent='DRAFT READY — not submitted';emit('armed');}
function verify(){continued++;let ok=armed&&document.querySelector('#draft').value==='Mira: sewing; Daniel: bicycles; close at 13:00';document.querySelector('#status').textContent=ok?'TASK CONTINUES — draft preserved':'STATE LOST — recreate draft';emit('verified');}
document.addEventListener('visibilitychange',()=>{emit('visibility');if(!document.hidden)requestAnimationFrame(()=>requestAnimationFrame(()=>emit('visible_painted')));});
window.addEventListener('pageshow',()=>requestAnimationFrame(()=>requestAnimationFrame(()=>emit('pageshow_painted'))));emit('load');
</script></html>'''

class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args): pass
    def do_GET(self):
        if urllib.parse.urlsplit(self.path).path!='/task':self.send_error(404);return
        data=PAGE.encode();self.send_response(200);self.send_header('Content-Type','text/html; charset=utf-8');self.send_header('Cache-Control','no-store');self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
    def do_POST(self):
        if self.path!='/event':self.send_error(404);return
        n=int(self.headers.get('Content-Length','0'))
        if n>4096:self.send_error(413);return
        body=json.loads(self.rfile.read(n)); body['host_monotonic']=time.monotonic();body['host_unix']=time.time()
        with (OUT/'app-events.jsonl').open('a') as f:f.write(json.dumps(body)+'\n')
        self.send_response(204);self.end_headers()

if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    print('Loopback test page http://127.0.0.1:18765/task; no external requests',flush=True)
    ThreadingHTTPServer(('127.0.0.1',18765),Handler).serve_forever()
