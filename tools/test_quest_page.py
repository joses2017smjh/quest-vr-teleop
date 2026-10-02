"""Run quest_bridge.html the way the headset would, without a headset.

The page is the hardest part of this rig to debug: a single uncaught error in
the frame loop leaves both panels black, and nothing on the PC says why. Two of
those cost an afternoon each (an undefined `hud`, then `status` colliding with
window.status), and both would have been caught in a second by this file.

It evaluates the page's <script> in QuickJS against stubbed browser objects,
then drives the paths the headset drives: the transport fallback, the XR frame
loop, and a real status payload.

  uv venv /tmp/jsenv && uv pip install --python /tmp/jsenv/bin/python quickjs
  /tmp/jsenv/bin/python tools/test_quest_page.py

With the bridge and run_teleop running, this also feeds the page whatever the
robot is broadcasting right now:

  /tmp/jsenv/bin/python tools/test_quest_page.py --live
"""

from __future__ import annotations

import argparse
import json
import subprocess
import random
import math
import re
import sys
from pathlib import Path

PAGE = Path(__file__).resolve().parent.parent / "scripts/teleop/quest_bridge.html"

# Stand-ins for everything the page asks the browser for. Where a browser would
# quietly do something surprising, these copy the surprise: `status` is defined
# as the DOMString property it really is, so assigning an object to it stores
# "[object Object]" here exactly as it does in Quest Browser.
STUBS = r"""
var __calls = [], __beacons = [], __posts = [], __glCalls = {}, __timers = [];
var __gl = {};
["createShader","createProgram","createBuffer","createTexture"].forEach(function(n){
  __gl[n]=function(){ __glCalls[n]=(__glCalls[n]||0)+1; return {id:n}; }; });
["shaderSource","compileShader","attachShader","linkProgram","bindBuffer","bufferData","bindTexture",
 "texParameteri","texImage2D","pixelStorei","useProgram","enableVertexAttribArray","vertexAttribPointer",
 "activeTexture","uniform1i","uniformMatrix4fv","uniform4f","drawArrays","bindFramebuffer","clearColor","clear",
 "viewport","disable","enable","blendFunc","uniformMatrix3fv","uniform3f","depthMask",
 "disableVertexAttribArray"].forEach(function(n){
   __gl[n]=function(){ __glCalls[n]=(__glCalls[n]||0)+1; }; });
__gl.getShaderParameter=function(){ return true; }; __gl.getProgramParameter=function(){ return true; };
__gl.getShaderInfoLog=function(){ return ""; }; __gl.getProgramInfoLog=function(){ return ""; };
__gl.getAttribLocation=function(){ return 0; }; __gl.getUniformLocation=function(){ return {}; };
__gl.makeXRCompatible=function(){ return Promise.resolve(); };
["VERTEX_SHADER","FRAGMENT_SHADER","COMPILE_STATUS","LINK_STATUS","ARRAY_BUFFER","STATIC_DRAW",
 "TEXTURE_2D","TEXTURE_MIN_FILTER","TEXTURE_MAG_FILTER","LINEAR","TEXTURE_WRAP_S","TEXTURE_WRAP_T",
 "CLAMP_TO_EDGE","RGBA","UNSIGNED_BYTE","UNPACK_PREMULTIPLY_ALPHA_WEBGL","TEXTURE0","FLOAT",
 "TRIANGLE_STRIP","COLOR_BUFFER_BIT","DEPTH_BUFFER_BIT","DEPTH_TEST","CULL_FACE","BLEND","ONE",
 "ONE_MINUS_SRC_ALPHA","FRAMEBUFFER","TRIANGLES","BYTE"].forEach(function(n,i){ __gl[n]=i+1; });

function Ctx(){ this.font=""; this.fillStyle=""; this.strokeStyle=""; this.lineWidth=0;
                this.textAlign=""; this.textBaseline=""; this.globalAlpha=1; }
["clearRect","fillRect","fillText","beginPath","moveTo","lineTo","arcTo","arc","closePath","fill",
 "stroke","drawImage","save","restore","translate","rotate","scale","setTransform","strokeText",
 "rect","quadraticCurveTo","bezierCurveTo","roundRect","clip"]
  .forEach(function(n){ Ctx.prototype[n]=function(){}; });
Ctx.prototype.createLinearGradient=function(){ return {addColorStop:function(){}}; };
Ctx.prototype.measureText=function(t){ return { width: String(t == null ? "" : t).length * 9 }; };

function El(){ this.style={}; this.content="8080"; this.width=0; this.height=0; this.hidden=true;
  this.textContent=""; this.className=""; this.disabled=false; this.naturalWidth=0; this.naturalHeight=0;
  this.classList={add:function(){},remove:function(){},toggle:function(){}}; this.dataset={}; }
El.prototype.addEventListener=function(){}; El.prototype.appendChild=function(){};
El.prototype.removeAttribute=function(){}; El.prototype.setAttribute=function(){};
El.prototype.getContext=function(kind){ return kind === "2d" ? new Ctx() : __gl; };
var document={ getElementById:function(){ return new El(); }, querySelector:function(){ return new El(); },
  createElement:function(){ return new El(); }, addEventListener:function(){}, hidden:false, body:new El() };
// The microphone the headset hands over. One that has been taken away still gives
// a live track and still delivers buffers, so `level` decides what is inside them.
var __mic = { level: 0, rate: 48000, asks: [], node: null, track: null, audio: [] };
__mic.fire = function(name){
  var t = __mic.track;
  if (t && t.on && t.on[name]) { t.readyState = name === "ended" ? "ended" : t.readyState; t.on[name](); }
};
__mic.feed = function(buffers){
  var samples = new Float32Array(4096);
  for (var i = 0; i < samples.length; i++) samples[i] = __mic.level * (i % 2 ? 1 : -1);
  var buffer = { getChannelData: function(){ return samples; } };
  for (var b = 0; b < buffers; b++) {
    if (__mic.node && __mic.node.onaudioprocess) __mic.node.onaudioprocess({ inputBuffer: buffer });
  }
};
function AudioCtxStub(){
  var self = this;
  this.sampleRate = __mic.rate; this.state = "running"; this.destination = {};
  this.createMediaStreamSource = function(){ return { connect:function(){}, disconnect:function(){} }; };
  this.createScriptProcessor = function(){
    __mic.node = { onaudioprocess:null, connect:function(){}, disconnect:function(){} };
    return __mic.node;
  };
  this.createGain = function(){ return { gain:{value:1}, connect:function(){}, disconnect:function(){} }; };
  this.resume = function(){ self.state = "running"; };
  this.close = function(){ self.state = "closed"; };
}
var window={ AudioContext:AudioCtxStub, addEventListener:function(){}, isSecureContext:true };
var navigator={ xr:undefined, userAgent:"QuestStub", permissions:null,
                mediaDevices:{ getUserMedia:function(want){
                  __mic.asks.push(JSON.stringify(want));
                  __mic.track = { readyState:"live", muted:false, enabled:true, label:"Quest mic",
                                  on:{}, addEventListener:function(n,f){ this.on[n]=f; },
                                  stop:function(){} };
                  var track = __mic.track;
                  return Promise.resolve({ getAudioTracks:function(){ return [track]; },
                                           getTracks:function(){ return [track]; } });
                } } };
var location={ protocol:"http:", host:"10.0.0.46:8080", hostname:"10.0.0.46",
  origin:"http://10.0.0.46:8080", href:"http://10.0.0.46:8080/", search:"" };
var __now = 100000;
var performance={ now:function(){ return __now; } };
var localStorage={ getItem:function(){ return null; }, setItem:function(){} };
function WebSocket(url){ __calls.push("ws " + url); this.readyState=0; }
WebSocket.OPEN=1; WebSocket.CLOSED=3;
function EventSource(url){ __calls.push("sse " + url); this.readyState=0; this.close=function(){}; }
function Image(){ var self=this; this.naturalWidth=0; this.naturalHeight=0;
  Object.defineProperty(this, "src", { set:function(v){ __beacons.push(v); self._src=v; },
                                       get:function(){ return self._src; } }); }
function URLSearchParams(s){ this.get=function(){ return null; }; this.has=function(){ return false; }; }
function setTimeout(fn, ms){ __timers.push([fn, ms]); return __timers.length; }
function clearTimeout(){} function setInterval(){ return 1; }
function fetch(url, opts){ __posts.push([url, opts && opts.body]); return Promise.resolve({}); }
var console={ log:function(){}, warn:function(){}, error:function(){} };
function XRWebGLLayer(){ this.framebuffer={};
  this.getViewport=function(){ return {x:0,y:0,width:1000,height:1000}; }; }

// `status` is a DOMString on window, not a free name: anything assigned to it
// is stringified. The page must therefore never keep robot state in it.
var __status = "";
Object.defineProperty(globalThis, "status", { get:function(){ return __status; },
                                             set:function(v){ __status = String(v); } });
"""

# One XR frame, the shape the Quest delivers: two views, two controllers, every
# button down, so every input path runs.
FRAME = r"""
var fakeView = {
  projectionMatrix: new Float32Array([1,0,0,0, 0,1,0,0, 0,0,-1,-1, 0,0,-0.2,0]),
  transform: { inverse: { matrix: new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]) } },
};
var fakeViewer = { transform: { position:{x:0,y:1.6,z:0}, orientation:{x:0,y:0,z:0,w:1} },
                   views: [fakeView, fakeView] };
function pad(hand, x, y) {
  return { handedness: hand, targetRayMode: "tracked-pointer", hand: null, gripSpace: {},
           gamepad: { buttons: [{value:0.9,pressed:true},{pressed:true},{pressed:false},
                                {pressed:true},{pressed:true},{pressed:true}],
                      axes: [0, 0, x, y] } };
}
var fakeFrame = {
  session: { requestAnimationFrame:function(){}, inputSources:[pad("left",0.5,-0.5), pad("right",-0.4,0.4)],
             end:function(){ __calls.push("session.end"); },
             renderState:{ baseLayer: new XRWebGLLayer() } },
  getViewerPose: function(){ return fakeViewer; },
  getPose: function(){ return { transform:{ matrix:new Float32Array([1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]) },
                                emulatedPosition:false }; },
};
session = fakeFrame.session; refSpace = {}; passthrough = true; sessionLabel = "AR (alpha-blend)";
for (var i = 0; i < 6; i++) { __now += 140; onFrame(__now, fakeFrame); }
"""

SCENARIOS = [
    ("no robot yet", "screenReady = false;"),
    ("screen share live",
     "screenReady = true; screenImg.naturalWidth = 1280; screenImg.naturalHeight = 720;"),
    ("a full status", """screenReady = false; linkOpen = true;
        handleServerMessage(JSON.stringify({type:"status", state:"ARMED", hz:50, sim:false,
          calibrated:3, scale:0.5, joints:[{n:"L pitch",tag:"OK",tau:1.2,lim:4,max:6,cal:true}],
          log:["hello"], lines:["a","b"], hint:"h", alert:""}));"""),
    ("a status with fields missing",
     'screenReady = false; linkOpen = true; handleServerMessage(JSON.stringify({type:"status", state:"CAL"}));'),
    ("a status with wrong types", """screenReady = false; linkOpen = true;
        handleServerMessage(JSON.stringify({type:"status", state:"CAL", lines:"nope", joints:null, log:7}));"""),
]


def page_script() -> str:
    return re.search(r"<script>(.*)</script>", PAGE.read_text(), re.S).group(1)


def fresh(quickjs, script):
    ctx = quickjs.Context()
    ctx.eval(STUBS)
    ctx.eval(script)
    return ctx


def pump(ctx, rounds: int = 50) -> None:
    """Let the promises settle: getUserMedia is awaited, and nothing runs until then."""
    for _ in range(rounds):
        if not ctx.execute_pending_job():
            return


def dictating(quickjs, script, level: float):
    """A page with the microphone open and one utterance spoken into it."""
    ctx = fresh(quickjs, script)
    ctx.eval("ws.onopen(); ws.readyState = WebSocket.OPEN;"
             "ws.send = function(b){ if (typeof b !== 'string') __mic.audio.push(b.byteLength); };"
             f"__mic.level = {level}; enableVoice(); 1;")
    pump(ctx)
    ctx.eval("startTalking(); __mic.feed(3); stopTalking(); 1;")
    pump(ctx)
    return ctx


def live_status(port: int = 8080) -> str | None:
    """Whatever run_teleop is broadcasting right now, as the bridge last sent it."""
    import urllib.request

    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/state.json", timeout=4) as response:
            state = json.loads(response.read().decode())
    except Exception as exc:                       # the bridge is simply not running
        print(f"  (no live status: {exc})")
        return None
    if not state.get("last_status"):
        print("  (the bridge is running but run_teleop has sent it nothing)")
        return None
    age = state.get("last_status_age")
    print(f"  (live status is {age:.1f}s old; "
          f"{state['pages_websocket']} page(s) on WebSocket, {state['pages_http']} on HTTP)")
    return json.dumps(state["last_status"])


def robot_model_json() -> str | None:
    """The headset's robot model, building it with the project venv if needed."""
    path = Path.home() / ".cache/bhl/robot_model.json"
    if not path.exists():
        venv = PAGE.parents[2] / ".venv/bin/python"
        subprocess.run([str(venv), str(PAGE.parent / "robot_model.py")], capture_output=True)
    return path.read_text() if path.exists() else None


def pinocchio_links(q: dict, links: list) -> dict | None:
    """Where Pinocchio (the teleop IK's own library) puts these links, for joint angles q."""
    venv = PAGE.parents[2] / ".venv/bin/python"
    urdf = PAGE.parents[2] / ("source/berkeley_humanoid_lite_assets/data/robots/berkeley_humanoid/"
                              "berkeley_humanoid_lite/urdf/berkeley_humanoid_lite.urdf")
    code = f"""
import json, pinocchio as pin
model = pin.buildModelFromUrdf({str(urdf)!r})
data = model.createData()
q = pin.neutral(model)
for name, value in {q!r}.items():
    q[model.joints[model.getJointId(name)].idx_q] = value
pin.forwardKinematics(model, data, q)
pin.updateFramePlacements(model, data)
print(json.dumps({{link: list(data.oMf[model.getFrameId(link)].translation) for link in {links!r}}}))
"""
    done = subprocess.run([str(venv), "-c", code], capture_output=True, text=True)
    if done.returncode != 0:
        return None
    return json.loads(done.stdout.strip().splitlines()[-1])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--live", action="store_true", help="also test the status the robot is sending now")
    args = parser.parse_args()
    try:
        import quickjs
    except ImportError:
        print("quickjs is missing. Make a throwaway venv (python3 -m venv needs a\n"
              "package this box does not have, so use uv, which the project already uses):\n"
              "  uv venv /tmp/jsenv && uv pip install --python /tmp/jsenv/bin/python quickjs\n"
              "  /tmp/jsenv/bin/python tools/test_quest_page.py --live")
        return 2

    script = page_script()
    failures = []

    try:
        ctx = fresh(quickjs, script)
    except Exception as exc:
        print(f"FAIL  the page does not even load: {str(exc).splitlines()[0]}")
        return 1
    print("ok    the page loads")

    if "/diag?" not in ctx.eval("JSON.stringify(__beacons)"):
        failures.append("the page does not report itself to /diag on load")
    else:
        print("ok    it reports itself to the bridge on load")

    for name, setup in SCENARIOS:
        ctx = fresh(quickjs, script)
        ctx.eval("__glCalls = {};")
        ctx.eval(setup)
        try:
            ctx.eval(FRAME)
        except Exception as exc:
            failures.append(f"the frame loop throws with {name}: {str(exc).splitlines()[0]}")
            continue
        drew = ctx.eval("__glCalls.drawArrays || 0")
        painted = ctx.eval("drawError")
        if painted:
            failures.append(f"the panel shows an error with {name}: {painted[:120]}")
        elif drew < 4:
            failures.append(f"only {drew} panels drawn with {name} (the headset would see black)")
        else:
            print(f"ok    {drew} panels drawn with {name}")

    # The transport the headset actually ends up on: ws:// is refused from a
    # page the browser treats as secure, so the fallback has to carry everything.
    ctx = fresh(quickjs, script)
    ctx.eval("__calls = []; var pending = __timers.slice(); __timers = [];"
             "for (var i = 0; i < pending.length; i++) { if (pending[i][1] >= 2000) pending[i][0](); }")
    if "sse /events" not in ctx.eval("JSON.stringify(__calls)") or ctx.eval("transport") != "http":
        failures.append("the page does not fall back to server-sent events when the WebSocket fails")
    else:
        ctx.eval("events.onopen(); send({ page:'t', left:{}, right:{}, ev:ev, stick:[0,0] });")
        if "/input" not in ctx.eval("JSON.stringify(__posts)"):
            failures.append("poses are not POSTed on the HTTP fallback")
        else:
            print("ok    it falls back to server-sent events and POSTs poses")
        ctx.eval("""events.onmessage({ data: JSON.stringify({type:"status", state:"ARMED", hz:50,
            lines:["x"], joints:[], log:["l"], scale:1, calibrated:0, sim:false}) });""")
        if not ctx.eval("statusFresh()"):
            failures.append("a status arriving over server-sent events is ignored")
        else:
            print("ok    a status over server-sent events reaches the panels")

    ctx = fresh(quickjs, script)
    ctx.eval("ws.onopen(); ws.readyState = WebSocket.OPEN;"
             "ws.send = function(m){ __calls.push('sent'); };"
             "send({ page:'t', left:{}, right:{}, ev:ev, stick:[0,0] });")
    if "sent" not in ctx.eval("JSON.stringify(__calls)"):
        failures.append("poses are not sent when the WebSocket does open")
    else:
        print("ok    it sends poses over the WebSocket when that works")

    ctx = fresh(quickjs, script)
    # ---- panels: all three up at once on an arc; looking at one focuses it.
    ARC = """
      function head(x, z, yaw, pitch, y){
        var cy = Math.cos(yaw/2), sy = Math.sin(yaw/2), cp = Math.cos((pitch||0)/2), sp = Math.sin((pitch||0)/2);
        return { position:{x:x, y:(y === undefined ? 1.6 : y), z:z},
                 orientation:{ x:cy*sp, y:cp*sy, z:-sy*sp, w:cy*cp } };
      }
      function lookAt(name){
        var pl = placementOf(name);
        var dx = pl.x - arc.head.x, dy = pl.y - arc.head.y, dz = pl.z - arc.head.z;
        return head(arc.head.x, arc.head.z, Math.atan2(-dx, -dz), Math.atan2(dy, Math.hypot(dx, dz)), arc.head.y);
      }
      function hold(h, ms){ for (var t = 0; t < ms; t += 20) { __now += 20; updateArc(h); updateGaze(__now); } }
      function gazeFor(name, ms){ hold(lookAt(name), ms); }
      arc.focus = "camera"; arc.init = false;
      updateArc(head(0, 0, 0, 0));
    """
    # Looking only swaps panels in the developer view (the sensors panel and the Claude
    # sessions); calibration mode (PC screen, camera, robot panel) switches with a left stick click.
    DEV = ("setView('developer'); sessionList = ['M', 'CAM']; refreshPanels(); arc.focus = 'sensors';"
           " arc.slots = arrange(); arc.init = false; updateArc(head(0, 0, 0, 0)); 1;")

    uv = ctx.eval("JSON.stringify([cameraUv('left'), cameraUv('right')])")
    if uv != "[[1,1,-0.5,-1],[0.5,1,-0.5,-1]]":
        failures.append(f"flipped stereo halves wrong: {uv}")
    else:
        print("ok    the upside-down stereo camera gives each eye its own half, turned and swapped")

    # Every panel, every frame: the focused one bright with its outline, the rest dim.
    ctx = fresh(quickjs, script)
    ctx.eval("""__now += 9000;                          // no toast left
      var __draws = [];
      var __realDraw = gfx.draw;
      gfx.draw = function(mvp, tex, uv, tint){ __draws.push([tex, tint]); return __realDraw(mvp, tex, uv, tint); };""")
    ctx.eval(FRAME)
    ctx.eval("""
      var perEye = __draws.slice(-10);
      var names = perEye.map(function(d){ return d[0] === controlsTexture ? "controls"
        : d[0] === hudTexture ? "status"
        : (d[0] === cameraTexture || d[0] === camOffTexture) ? "camera"
        : (d[0] === screenTexture || d[0] === logTexture) ? "screen"
        : (d[0] === visionTexture || d[0] === visOffTexture) ? "vision"
        : d[0] === whiteTexture ? "outline" : "?"; });
      var bright = perEye.filter(function(d){ return d[0] !== whiteTexture && d[0] !== controlsTexture && !d[1]; }).length;
      var dim = perEye.filter(function(d){ return d[1] === DIM && d[0] !== controlsTexture; }).length;
      var outline = perEye.filter(function(d){ return d[0] === whiteTexture && d[1] === ACCENT; }).length;
      var offline = perEye.some(function(d){ return d[0] === camOffTexture; })
        && perEye.some(function(d){ return d[0] === visOffTexture; });""")
    drawn = json.loads(ctx.eval("JSON.stringify(names)"))
    if sorted(n for n in drawn if n not in ("outline", "controls")) != ["camera", "screen", "status", "vision", "vision"]:
        failures.append(f"not every panel is drawn each frame: {drawn}")
    elif "controls" not in drawn:
        failures.append(f"with buttons held, the controls strip is not shown: {drawn}")
    elif ctx.eval("bright") != 3 or ctx.eval("dim") != 2 or ctx.eval("outline") != 4:
        failures.append(f"focus is not shown: {ctx.eval('bright')} bright, {ctx.eval('dim')} dim, "
                        f"{ctx.eval('outline')} outline strips (want 3 - the camera and its two vision "
                        "pictures - 2 and 4)")
    elif not ctx.eval("offline"):
        failures.append("with no camera, the camera panel and its vision pictures do not say so")
    else:
        print("ok    all three panels (and the vision pair) are drawn; the focused one bright and outlined, the others dim")
        print("ok    with no camera frames, the camera panel and its vision pair say they are waiting")

    # Looking picks the panel (developer view) - but only a look, not a glance on the way past.
    ctx = fresh(quickjs, script)
    ctx.eval(ARC)
    ctx.eval(DEV)
    ctx.eval("gazeFor('s:M', 150); gazeFor('sensors', 100); 1;")
    glance = ctx.eval("arc.focus")
    # the whole arc slides after a pick, so the head follows the picked panel to the middle
    ctx.eval("gazeFor('s:M', 400); hold(head(0, 0, 0, 0), 600); 1;")
    looked = ctx.eval("arc.focus")
    ctx.eval("gazeFor('s:CAM', 400); 1;")
    third = ctx.eval("arc.focus")
    ctx.eval("""hold(head(0, 0, 0, 0), 600);           // settle, looking at the middle
      var a = placementOf(arc.slots[1]), b = placementOf(arc.slots[2]);
      var edgeA = a.yaw - Math.atan2(a.size.w / 2, a.reach), edgeB = b.yaw + Math.atan2(b.size.w / 2, b.reach);
      hold(head(0, 0, (edgeA + edgeB) / 2, 0), 600); 1;""")
    between = ctx.eval("arc.focus")
    if glance != "sensors":
        failures.append(f"a 150 ms glance at session M stole the focus ({glance})")
    elif looked != "s:M" or third != "s:CAM":
        failures.append(f"looking does not move the focus (M -> {looked}, CAM -> {third})")
    elif between != "s:CAM":
        failures.append(f"looking at the gap between two panels changed the focus to {between}")
    else:
        print("ok    looking at a panel focuses it; a glance does not, and the gaps change nothing")

    # The picked panel slides into the middle and the arc slides with it, keeping its
    # order; staying on the spot it came from picks nothing else.
    ctx = fresh(quickjs, script)
    ctx.eval(ARC)
    ctx.eval(DEV)
    ctx.eval("var was = targetOf('s:CAM').yaw; gazeFor('s:CAM', 330); 1;")
    moved_in = ctx.eval("Math.abs(wrapAngle(targetOf('s:CAM').yaw - arc.yaw))")
    kept_order = ctx.eval("JSON.stringify(arc.slots)")
    sensors_left = ctx.eval("wrapAngle(targetOf('sensors').yaw - arc.yaw) > 0.3 && wrapAngle(targetOf('s:M').yaw - arc.yaw) < -0.3")
    sliding = ctx.eval("Math.abs(wrapAngle(placementOf('s:CAM').yaw - targetOf('s:CAM').yaw))")
    ctx.eval("hold(head(0, 0, was, 0), 2000); 1;")          # keep looking where CAM was
    stayed = ctx.eval("arc.focus")
    ctx.eval("hold(head(0, 0, 0, 0), 500); gazeFor('sensors', 400); 1;")   # away, and back
    back = ctx.eval("arc.focus")
    smaller = ctx.eval("sizeOf('s:M').w < sizeOf('sensors').w && sizeOf('s:CAM').w < sizeOf('sensors').w")
    if moved_in > 1e-9:
        failures.append(f"the picked panel does not take the middle place ({moved_in:.2f} rad off)")
    elif kept_order != '["sensors","s:CAM","s:M"]' or not sensors_left:
        failures.append(f"the arc did not keep its order around the picked panel: {kept_order}")
    elif sliding < 0.05:
        failures.append("the swap jumps instead of sliding")
    elif stayed != "s:CAM":
        failures.append(f"looking at where CAM was picks what slid there ({stayed} became main)")
    elif back != "sensors":
        failures.append(f"after looking away, the old main screen cannot be picked again ({back})")
    elif not smaller:
        failures.append("the panels at the side are not smaller previews of the main screen")
    else:
        print("ok    a picked panel slides into the middle, the arc keeps its order around it, no ping-pong")
        print("ok    after looking away the old main can be picked back; the side panels are smaller previews")

    # Calibration mode: looking around never swaps; a left stick click brings the next
    # screen in (PC screen, camera, robot panel, round again); held, it leaves.
    ctx = fresh(quickjs, script)
    ctx.eval(ARC)
    ctx.eval("""gazeFor('screen', 800); gazeFor('status', 800); var looked = arc.focus;
      toggleDetails(); gazeFor('camera', 1000); var kept = arc.focus;
      function lpad(down){ return { handedness:"left", targetRayMode:"tracked-pointer", hand:null, gripSpace:{},
        gamepad:{ buttons:[{value:0,pressed:false},{pressed:false},{pressed:false},{pressed:down},
        {pressed:false},{pressed:false}], axes:[0,0,0,0] } }; }
      function clickL(){ readButtons([lpad(true)]); __now += 150; readButtons([lpad(false)]); __now += 50; }
      var cycled = [];
      for (var i = 0; i < 3; i++) { clickL(); cycled.push(arc.focus); }
      session = { end: function(){ __calls.push("session.end"); } };
      var clickLeft = __calls.indexOf("session.end") >= 0;
      readButtons([lpad(true)]); __now += 1100; readButtons([lpad(true)]); readButtons([lpad(false)]);
      var heldLeft = __calls.indexOf("session.end") >= 0; 1;""")
    cycled = json.loads(ctx.eval("JSON.stringify(cycled)"))
    if ctx.eval("looked") != "camera":
        failures.append(f"in calibration mode, looking at a panel swapped it in ({ctx.eval('looked')})")
    elif ctx.eval("kept") != "status":
        failures.append(f"staring at the camera undid the robot panel the left trigger brought up ({ctx.eval('kept')})")
    elif cycled != ["screen", "camera", "status"]:
        failures.append(f"left stick clicks do not cycle PC screen, camera, robot panel: {cycled}")
    elif ctx.eval("clickLeft") or not ctx.eval("heldLeft"):
        failures.append("a left stick click must switch screens, and only holding it leaves the headset view")
    else:
        print("ok    calibration mode: looking around never swaps; a left stick click brings the next screen in")
        print("ok    the left trigger's robot panel holds while you look around; holding the left stick leaves")

    # Growing one panel pushes its neighbours out; no two ever overlap.
    ctx = fresh(quickjs, script)
    ctx.eval(ARC)
    ctx.eval("""
      function gaps(){ var off = arcOffsets(), out = [];
        for (var i = 0; i + 1 < arc.slots.length; i++) {
          var a = arc.slots[i], b = arc.slots[i + 1];
          out.push((off[a] - Math.atan2(sizeOf(a).w / 2, reachOf(a))) - (off[b] + Math.atan2(sizeOf(b).w / 2, reachOf(b))));
        } return out; }
      var before = arcOffsets().screen, worst = 1;
      for (var i = 0; i < 80; i++) { scalePanel(0.9, 0.05); gaps().forEach(function(g){ worst = Math.min(worst, g); }); }
      var after = arcOffsets().screen;
      setFocus("status", "test");
      for (var i = 0; i < 80; i++) { scalePanel(0.9, 0.05); movePanel(-0.9, 0, 0.05); gaps().forEach(function(g){ worst = Math.min(worst, g); }); }
    """)
    if ctx.eval("spots.camera.scale") <= 1.5:
        failures.append("the focused panel does not grow when resized")
    elif abs(ctx.eval("after")) <= abs(ctx.eval("before")):
        failures.append("growing the camera panel did not push the PC screen outward")
    elif ctx.eval("worst") < ctx.eval("ARC_GAP") - 1e-9:
        failures.append(f"two panels overlapped while resizing (gap {ctx.eval('worst'):.3f} rad)")
    else:
        print(f"ok    resizing pushes the neighbours out; the gap never drops below "
              f"{ctx.eval('ARC_GAP'):.2f} rad")

    # Each panel keeps its own place and size.
    ctx = fresh(quickjs, script)
    ctx.eval(ARC)
    ctx.eval("""setFocus("status", "test");
      for (var i = 0; i < 40; i++) { movePanel(0.9, 0.9, 0.05); scalePanel(0.9, 0.05); }""")
    other = ctx.eval("JSON.stringify([spots.screen.dd, spots.screen.dy, spots.screen.scale, "
                     "spots.camera.dd, spots.camera.dy, spots.camera.scale])")
    if other != "[0,0,1,0,0,1]":
        failures.append(f"moving the robot panel moved the others too: {other}")
    elif ctx.eval("spots.status.dd <= 0 || spots.status.dy <= 0 || spots.status.scale <= 1"):
        failures.append("the focused panel did not move or grow")
    else:
        print("ok    the stick moves and resizes only the focused panel")

    # Fixed in the room: turning and walking leave the panels put; recentring brings
    # the focused one straight ahead.
    ctx = fresh(quickjs, script)
    ctx.eval(ARC)
    ctx.eval("""var was = JSON.stringify(PANEL_ORDER.map(function(n){ var p = placementOf(n); return [p.x, p.z]; }));
      for (var i = 0; i < 40; i++) updateArc(head(0.5, 0.5, 0.9, 0.2));
      var now = JSON.stringify(PANEL_ORDER.map(function(n){ var p = placementOf(n); return [p.x, p.z]; }));""")
    stayed = ctx.eval("was === now")
    ctx.eval("setFocus('status', 'test'); arc.init = false; updateArc(head(0.5, 0.5, 0.9, 0)); 1;")
    ahead = ctx.eval("Math.abs(wrapAngle(placementOf('status').yaw - 0.9))")
    if not stayed:
        failures.append("the panels move when you turn your head or step")
    elif ahead > 1e-6:
        failures.append(f"recentring does not put the focused panel straight ahead ({ahead:.3f} rad off)")
    else:
        print("ok    the panels stay put in the room; recentring brings the focused one straight ahead")

    # Sitting down after placing them standing: they keep their place but turn to
    # face the new eye line, and holding the right stick resets the focused one.
    ctx = fresh(quickjs, script)
    ctx.eval(ARC)
    ctx.eval("""var standing = placementOf('camera');
      for (var i = 0; i < 20; i++) updateArc(head(0, 0, 0, 0, 1.15));
      var seated = placementOf('camera');""")
    moved = ctx.eval("Math.abs(seated.x - standing.x) + Math.abs(seated.y - standing.y) + Math.abs(seated.z - standing.z)")
    turned = ctx.eval("Math.abs(seated.tilt - standing.tilt)")
    if moved > 1e-9:
        failures.append("sitting down drags the panels with you")
    elif turned < 0.05:
        failures.append(f"the panels do not turn to face you when you sit ({turned:.3f} rad)")
    else:
        ctx.eval("""for (var i = 0; i < 40; i++) { movePanel(0.9, 0.9, 0.05); scalePanel(0.9, 0.05); }
          __now += 5000;
          function stickPad(down){ return { handedness:"right", targetRayMode:"tracked-pointer", hand:null,
            gripSpace:{}, gamepad:{ buttons:[{value:0,pressed:false},{pressed:false},{pressed:false},
            {pressed:down},{pressed:false},{pressed:false}], axes:[0,0,0,0] } }; }
          readButtons([stickPad(true)]); __now += 900; readButtons([stickPad(true)]); 1;""")
        if not ctx.eval("spots.camera.dy === 0 && spots.camera.dd === 0 && spots.camera.scale === 1"):
            failures.append("holding the right stick does not put the focused panel back to its default")
        else:
            print(f"ok    panels turn to face you when you sit ({turned:.2f} rad) and reset on a held stick")

    # A question pulls the focus to the robot panel and says where it is; A is only
    # the robot's yes and never moves the focus.
    ctx = fresh(quickjs, script)
    ctx.eval(ARC)
    ctx.eval("""
      function apad(down){ return { handedness:"right", targetRayMode:"tracked-pointer", hand:null,
        gripSpace:{}, gamepad:{ buttons:[{value:0,pressed:false},{pressed:false},{pressed:false},
        {pressed:false},{pressed:down},{pressed:false}], axes:[0,0,0,0] } }; }
      function clickA(){ readButtons([apad(true)]); readButtons([apad(false)]); }
      clickA(); __now += 100; clickA();""")
    after_a = ctx.eval("arc.focus")
    ctx.eval("""linkOpen = true;
      handleServerMessage(JSON.stringify({type:"status", state:"CAL", hz:50, lines:[], joints:[], log:[]}));
      watchPrompts(); 1;""")
    if after_a != "camera":
        failures.append(f"A twice moved the focus to {after_a}; A is only the robot's yes now")
    elif ctx.eval("arc.focus") != "status":
        failures.append("a calibration question does not pull the focus to the robot panel")
    elif "main screen" not in ctx.eval("toastText") or ctx.eval("Math.abs(wrapAngle(targetOf('status').yaw - arc.yaw)) > 1e-9"):
        failures.append(f"the question's toast does not say where to look: {ctx.eval('toastText')!r}")
    else:
        print(f"ok    a question brings the robot panel to the middle and says so: {ctx.eval('toastText')!r}")

    # Pointing: the ray has to land where the screen panel actually is, and the
    # trigger has to become a mouse button rather than push-to-talk.
    ctx = fresh(quickjs, script)
    ctx.eval(ARC)
    ctx.eval("""
      setFocus("screen", "test"); arc.init = false; updateArc(head(0, 0, 0, 0));   // screen straight ahead
      function aimAt(yaw, pitch){ return mul(translate(0, 1.6, 0), mul(rotY(yaw), rotX(pitch))); }
      ws.onopen(); ws.readyState = WebSocket.OPEN;
      var __sent = [];
      ws.send = function(m){ if (typeof m === "string") __sent.push(m); };
    """)
    straight = json.loads(ctx.eval("JSON.stringify(rayHit(aimAt(0, 0)))"))
    aside = json.loads(ctx.eval("JSON.stringify(rayHit(aimAt(0.25, 0)))"))
    away = ctx.eval("JSON.stringify(rayHit(aimAt(1.4, 0)))")
    ctx.eval("setFocus('camera', 'test'); arc.init = false; updateArc(head(0, 0, 0, 0)); 1;")
    off_side = ctx.eval("var pl = placementOf('screen');"
                        "JSON.stringify(rayHit(aimAt(pl.yaw, Math.atan2(pl.y - 1.6, pl.reach))))")
    ctx.eval("setFocus('screen', 'test'); arc.init = false; updateArc(head(0, 0, 0, 0)); 1;")
    if not straight:
        failures.append("a controller pointed straight at the PC screen misses it entirely")
    elif abs(straight["u"] - 0.5) > 0.001 or not (0 <= straight["v"] <= 1):
        failures.append(f"straight ahead lands at u={straight['u']:.3f} v={straight['v']:.3f}, not the middle")
    elif not aside or abs(aside["u"] - 0.5) < 0.02:
        failures.append("turning the wrist does not move the hit across the panel")
    elif away != "null":
        failures.append("a controller pointed away from the panel still reports a hit")
    elif off_side == "null":
        failures.append("the PC screen cannot be pointed at while it sits to the side of the arc")
    else:
        ctx.eval("pointer.on = true; updatePointer(aimAt(0, 0), 0.9, 0); 1;")
        mice = [json.loads(m) for m in json.loads(ctx.eval("JSON.stringify(__sent)"))
                if json.loads(m).get("type") == "mouse"]
        if not mice:
            failures.append("pointing sends nothing to the PC")
        elif mice[-1].get("down") != 1 or not (0 <= mice[-1].get("u", -1) <= 1):
            failures.append(f"the click is not reported properly: {mice[-1]}")
        else:
            print(f"ok    the ray lands on the PC screen wherever it sits on the arc, and the trigger clicks "
                  f"(u={straight['u']:.2f} v={straight['v']:.2f})")

    # The headset takes the microphone back whenever something else wants it, which
    # is what happened mid-session; the page has to notice and ask for it again.
    ctx = dictating(quickjs, script, 0.4)
    ctx.eval("__mic.asks = []; __mic.fire('ended'); 1;")
    if ctx.eval("voiceReady"):
        failures.append("a microphone the headset took away is still reported as ready")
    else:
        ctx.eval("var pending = __timers.pop(); if (pending) pending[0](); 1;")
        pump(ctx)
        if ctx.eval("__mic.asks.length") != 1:
            failures.append("the page does not ask for the microphone again after the headset drops it")
        elif not ctx.eval("voiceReady"):
            failures.append("the microphone is asked for again but never comes back")
        else:
            print("ok    a microphone the headset drops is taken back automatically")

    # Replacing the microphone stops the old track; if that stop is mistaken for a
    # fresh loss, every retake starts another one and the microphone never settles.
    ctx = dictating(quickjs, script, 0.4)
    ctx.eval("var stale = __mic.track; __mic.asks = []; __mic.fire('ended'); 1;")
    ctx.eval("var pending = __timers.pop(); if (pending) pending[0](); 1;")
    pump(ctx)
    ctx.eval("__mic.track = stale; __mic.fire('ended'); __mic.track = null; 1;")   # the old one, on its way out
    ctx.eval("var after = __timers.pop(); if (after) after[0](); 1;")
    pump(ctx)
    if ctx.eval("__mic.asks.length") != 1:
        failures.append(f"a stale track ending starts another retake "
                        f"({ctx.eval('__mic.asks.length')} microphones asked for, wanted 1)")
    else:
        print("ok    a replaced microphone track cannot start a retake loop")

    # Dictation. A muted headset microphone still delivers buffers, full of zeros,
    # and the audio alone cannot be told from a quiet room - the page has to say so.
    ctx = dictating(quickjs, script, 0.4)
    if not ctx.eval("voiceReady"):
        failures.append("the microphone cannot be opened at all")
    elif ctx.eval("__mic.audio.length") != 3:
        failures.append(f"a spoken utterance sent {ctx.eval('__mic.audio.length')} audio buffers, not 3")
    elif "voice-silent" in ctx.eval("JSON.stringify(__beacons)"):
        failures.append("a microphone that is working is reported as silent")
    elif ctx.eval("voiceLevel") <= 0:
        failures.append("the level meter stays flat while the microphone is being heard")
    elif ctx.eval("__mic.audio[0]") != 2730:            # 4096 frames of 48 kHz, at 16 kHz, 16-bit
        failures.append(f"a 48 kHz buffer arrives as {ctx.eval('__mic.audio[0]')} bytes, not 2730")
    else:
        print(f"ok    a live microphone sends 16 kHz audio and moves the meter "
              f"({ctx.eval('meter(voiceLevel)')})")

    ctx = dictating(quickjs, script, 0.0)
    if "voice-silent" not in ctx.eval("JSON.stringify(__beacons)"):
        failures.append("a microphone sending pure silence is not reported to the bridge")
    elif ctx.eval("__mic.asks.length") != 2:
        failures.append("the page does not reopen the microphone after a silent utterance")
    elif "\"echoCancellation\":false" not in ctx.eval("__mic.asks[1]").replace(" ", ""):
        failures.append(f"the microphone is reopened with the same processing: {ctx.eval('__mic.asks[1]')}")
    else:
        print("ok    a silent microphone is reported and reopened without echo cancellation")

    # ---- hands-free dictation
    def listening_page():
        ctx = fresh(quickjs, script)
        ctx.eval("var __said = [];"
                 "ws.onopen(); ws.readyState = WebSocket.OPEN;"
                 "ws.send = function(b){ if (typeof b !== 'string') __mic.audio.push(b.byteLength);"
                 "                       else __said.push(JSON.parse(b)); };"
                 "__mic.level = 0.4; enableVoice(); 1;")
        pump(ctx)
        return ctx

    def fed(ctx, setup: str) -> int:
        ctx.eval(setup + "; __mic.audio = []; __mic.feed(2); 1;")
        return ctx.eval("__mic.audio.length")

    # The microphone streams with nobody holding the trigger, so the wake word can be
    # heard - but not while one of Claude's replies is being read aloud, which would
    # otherwise wake it with its own words; a sentence already under way keeps going.
    ctx = listening_page()
    idle = fed(ctx, "")
    during = fed(ctx, "sayingUntil = performance.now() + 5000")
    mid = fed(ctx, "handleServerMessage(JSON.stringify({type:'voice', state:'listening', text:'wake', wake:'voice'}))")
    off = fed(ctx, "handleServerMessage(JSON.stringify({type:'voice', state:'heard', wake:'voice'}));"
                   "sayingUntil = 0; wakeOn = false")
    if idle != 2:
        failures.append(f"hands-free: the microphone does not stream while waiting ({idle} of 2 buffers)")
    elif during != 0:
        failures.append("hands-free: the microphone streams while Claude's reply is read out (it would wake itself)")
    elif mid != 2:
        failures.append("hands-free: a sentence under way stops streaming when a reply starts playing")
    elif off != 0:
        failures.append("hands-free off still streams the microphone while nobody holds the trigger")
    elif ctx.eval("wakeWord") != "voice":
        failures.append(f"the page does not learn the wake word from voice_typer ({ctx.eval('wakeWord')!r})")
    else:
        print("ok    hands-free streams while waiting, pauses for Claude's replies, and switches off")

    # A beep when it starts listening because you said the word, and one when it stops.
    ctx = fresh(quickjs, script)
    ctx.eval("""var __beeps = []; beep = function(up){ __beeps.push(up); };
      function note(state, text){ handleServerMessage(JSON.stringify({type:'voice', state:state, text:text||'', wake:'voice'})); }
      note('listening', 'trigger'); note('heard');            // held trigger: only the end
      note('listening', 'wake'); note('partial', 'open'); note('heard');
      note('listening', 'trigger'); note('cancelled'); 1;""")
    beeps = ctx.eval("JSON.stringify(__beeps)")
    if beeps != "[false,true,false]":
        failures.append(f"beeps wrong: {beeps} (want an end beep for a trigger sentence, "
                        "start + end for a wake-word one, nothing for a cancelled tap)")
    elif ctx.eval("typerListening"):
        failures.append("the page still thinks voice_typer is listening after a cancelled tap")
    else:
        print("ok    a beep when the wake word starts a sentence and when listening stops; none for a tap")

    # A stop is always sent, even after the headset took the microphone away mid-sentence:
    # without it, voice_typer kept listening until the next press wiped the sentence.
    ctx = listening_page()
    ctx.eval("startTalking(); voiceReady = false; stopTalking('released'); 1;")
    stops = ctx.eval("JSON.stringify(__said.filter(function(m){ return m.type === 'voice' && m.action === 'stop'; }))")
    if stops == "[]":
        failures.append("no stop is sent when the microphone was taken away mid-sentence")
    else:
        print("ok    the stop is sent even when the headset took the microphone mid-sentence")

    # The grip: never starts dictation; only ends it where grip + trigger mean something
    # to the robot (armed: the claw; calibrating: the hold).
    ctx = listening_page()
    ctx.eval("robotStatus = {state:'STOPPED'}; talkControl(0.9, false, null); talkControl(0.9, true, null); 1;")
    stopped_idle = not ctx.eval("voiceActive")
    ctx.eval("robotStatus = {state:'ARMED'}; talkControl(0.9, true, null); 1;")
    stopped_armed = not ctx.eval("voiceActive")
    ctx.eval("talkControl(0.0, false, null); talkControl(0.9, true, null); 1;")
    started_by_grip = ctx.eval("voiceActive")
    if stopped_idle:
        failures.append("closing the hand on the grip cuts dictation off while the robot is not armed")
    elif not stopped_armed:
        failures.append("the grip does not end dictation while armed (it would record while you drive the claw)")
    elif started_by_grip:
        failures.append("squeezing the grip starts dictation")
    else:
        print("ok    the grip only cuts dictation while armed or calibrating, and never starts it")

    # Pressing the trigger to talk twice in a row must not switch hands-free off (it did).
    ctx = listening_page()
    ctx.eval("""function press(ms){ talkControl(0.9, false, null); __now += ms; talkControl(0.0, false, null); }
      press(1500); __now += 300; press(1200);          // said something, then something else
      __now += 2000; press(900); __now += 200; press(100);   // a long press, then a quick tap
      1;""")
    if not ctx.eval("wakeOn"):
        failures.append("pressing the trigger to talk (twice, or long then short) switched hands-free off")
    else:
        print("ok    talking with the trigger twice in a row leaves hands-free on")

    # Right trigger twice: hands-free on and off.
    ctx = listening_page()
    was = ctx.eval("wakeOn")
    ctx.eval("talkControl(0.9, false, null); talkControl(0.0, false, null);"
             "__now += 150; talkControl(0.9, false, null); talkControl(0.0, false, null); 1;")
    if not was or ctx.eval("wakeOn"):
        failures.append(f"right trigger x2 does not toggle hands-free (was {was}, now {ctx.eval('wakeOn')})")
    elif "Hands-free off" not in ctx.eval("toastText"):
        failures.append(f"turning hands-free off says nothing: {ctx.eval('toastText')!r}")
    else:
        print("ok    right trigger x2 turns hands-free off (it starts on), and says so")

    # ---- the 3D robot in the robot panel
    model_json = robot_model_json()
    if model_json is None:
        print("skip  the 3D robot: no model (run scripts/teleop/robot_model.py with the project venv)")
    else:
        model = json.loads(model_json)
        size = model["positions"][1] + model["normals"][1]
        STATUS = """{type:"status", state:"CAL", hz:50, sim:false, calibrated:3, scale:1,
          lines:["L elbow: WATCH IT MOVE", "It should bend the elbow.", "A = yes", "B = no"],
          log:[], hint:"4.0 Nm was not enough; trying 4.5 Nm", alert:"",
          joints:[
            {n:"L pitch 1", u:"arm_left_shoulder_pitch_joint", q:-0.5, qc:-0.5, tag:"OK 3.5", cal:true, tau:1, lim:3, max:5,
             p:{kp:30, kd:2, fl:2, hr:1.8, g:0.95, flip:false, when:"2026-09-24T11:13", last:{ok:true, at:3.5, moved:0.97, err:0.05}}},
            {n:"L elbow 7", u:"arm_left_elbow_pitch_joint", q:-0.8, qc:-1.0, tag:"TEST", test:true, cal:false, tau:3.9, lim:4, max:5.5,
             id:7, p:{kp:30, kd:2, fl:1.5, hr:1.0, g:1.0, flip:true, when:"", last:{ok:false, at:4.0, moved:0.02, err:0.4, why:"not moving"}}},
            {n:"L wrist 9", u:"arm_left_elbow_roll_joint", q:0.3, qc:0.3, tag:"BLOCKED", cal:false, tau:0, lim:1, max:2.5, p:{}},
            {n:"R pitch 2", u:"arm_right_shoulder_pitch_joint", q:0.0, qc:0, tag:"", cal:false, tau:0, lim:2, max:6, p:{}}
          ]}"""

        def robot_page():
            ctx = fresh(quickjs, script)
            ctx.eval(f"setupRobot({model_json}, new ArrayBuffer({size})); linkOpen = true;"
                     f"handleServerMessage(JSON.stringify({STATUS})); 1;")
            return ctx

        # The page's joint maths against Pinocchio (what the IK itself uses): if the
        # headset poses the robot differently from the real kinematics, it lies.
        rng = random.Random(7)
        q = {j["name"]: rng.uniform(j["lower"], j["upper"]) for j in model["joints"]
             if j["type"] == "revolute" and j["name"].startswith("arm_")}
        reference = pinocchio_links(q, [j["child"] for j in model["joints"] if j["name"] in q])
        ctx = robot_page()
        mine = json.loads(ctx.eval(f"""var w = robotFK({json.dumps(q)}); var out = {{}};
          Object.keys(w).forEach(function(k){{ out[k] = [w[k][12], w[k][13], w[k][14]]; }}); JSON.stringify(out)"""))
        if reference is None:
            print("skip  robot kinematics vs Pinocchio (no Pinocchio in the project venv)")
        else:
            worst = max(math.dist(mine[link], at) for link, at in reference.items())
            if worst > 1e-4:
                failures.append(f"the headset's robot poses its arms differently from Pinocchio (off by {worst * 1000:.1f} mm)")
            else:
                print(f"ok    the headset's robot matches Pinocchio's kinematics for a random pose "
                      f"({len(reference)} arm links, within {worst * 1e6:.1f} µm)")

        # Colours: the joint under test glows, OK is green, a problem red, untested grey;
        # a hand takes its wrist's colour; the legs stay dark.
        colour = json.loads(ctx.eval("JSON.stringify(robotPose(0))"))
        def of(link):
            return json.loads(ctx.eval(f"JSON.stringify(colourOf('{link}', {json.dumps(colour)}))"))
        elbow, pitch, wrist = of("arm_left_elbow_pitch"), of("arm_left_shoulder_pitch"), of("arm_left_elbow_roll")
        hand, untested, leg = of("arm_left_hand_link"), of("arm_right_shoulder_pitch"), of("leg_left_knee_pitch")
        if not (elbow[0] > 0.9 and elbow[1] > 0.6):
            failures.append(f"the joint being calibrated does not glow: {elbow}")
        elif not (pitch[1] > pitch[0] and pitch[1] > pitch[2]):
            failures.append(f"a joint that passed is not green: {pitch}")
        elif not (wrist[0] > wrist[1] and wrist[0] > wrist[2]):
            failures.append(f"a blocked joint is not red: {wrist}")
        elif hand != wrist:
            failures.append("the hand does not take its wrist's colour")
        elif untested != json.loads(ctx.eval("JSON.stringify(NEUTRAL)")) or leg != json.loads(ctx.eval("JSON.stringify(BODY)")):
            failures.append(f"untested arm joints or the legs are coloured: {untested}, {leg}")
        else:
            print("ok    the joint being calibrated glows; OK green, problems red, untested grey, hands follow wrists")

        # Drawn in the frame loop, one draw per part per eye, and the robot panel keeps
        # to the question: no joint table, no hint, the A / B lines.
        ctx = robot_page()
        ctx.eval("""__glCalls = {}; var __texts = [];
          hudCtx.fillText = function(t){ __texts.push(String(t)); };
          var __tri = 0, __realDrawArrays = __gl.drawArrays;
          __gl.drawArrays = function(mode){ if (mode === __gl.TRIANGLES) __tri += 1; return __realDrawArrays.apply(null, arguments); };""")
        ctx.eval(FRAME)
        parts = sum(1 for link in model["links"].values() if link["mesh"])
        texts = json.loads(ctx.eval("JSON.stringify(__texts)"))
        if ctx.eval("__tri") != parts * 12:
            failures.append(f"the robot is not drawn part by part in both eyes ({ctx.eval('__tri')} draws, want {parts * 12})")
        elif ctx.eval("drawError"):
            failures.append(f"drawing the robot panel failed: {ctx.eval('drawError')[:120]}")
        elif any("L pitch 1" in t for t in texts) or any("4.0 Nm was not enough" in t for t in texts):
            failures.append("the robot panel still shows the joint table or the hint beside the 3D robot")
        elif not any("WATCH IT MOVE" in t for t in texts) or not any("B = no" in t for t in texts):
            failures.append("the robot panel lost the question or its A / B lines")
        else:
            print(f"ok    the 3D robot is drawn ({parts} parts per eye); the panel keeps only the question and A / B")

        # The left trigger: the details of the joint under test, in words, on the main screen.
        ctx = robot_page()
        ctx.eval(ARC)
        ctx.eval("""var __texts = []; hudCtx.fillText = function(t){ __texts.push(String(t)); };
          function lpad(down){ return { handedness:"left", targetRayMode:"tracked-pointer", hand:null,
            gripSpace:{}, gamepad:{ buttons:[{value:down?1:0,pressed:down},{pressed:false},{pressed:false},
            {pressed:false},{pressed:false},{pressed:false}], axes:[0,0,0,0] } }; }
          readButtons([lpad(true)]); readButtons([lpad(false)]);
          drawHud({left:null, right:null, passthrough:true, inXr:true, label:"AR"}); 1;""")
        texts = " | ".join(json.loads(ctx.eval("JSON.stringify(__texts)")))
        if not ctx.eval("showDetails"):
            failures.append("the left trigger does not open the details")
        elif ctx.eval("Math.abs(wrapAngle(targetOf('status').yaw - arc.yaw)) > 1e-9"):
            failures.append("opening the details does not bring the robot panel to the middle")
        elif "CAN ID 7" not in texts or "FLIPPED" not in texts or "failed at 4.0 N·m: not moving" not in texts:
            failures.append(f"the details do not explain the joint under test: {texts[:300]}")
        elif "between 1.5 and 5.5 N·m" not in texts:
            failures.append(f"the power numbers are not explained: {texts[:300]}")
        else:
            print("ok    L trigger: the joint's numbers in words (power 1.5 to 5.5 N·m, flipped, last test failed)")
            ctx.eval("readButtons([lpad(true)]); readButtons([lpad(false)]); 1;")
            if ctx.eval("showDetails"):
                failures.append("pressing the left trigger again does not go back to the robot")
            else:
                print("ok    and again: back to the robot")

        # What is active glows, and a pose to hold is demonstrated (the reach check).
        ctx = robot_page()
        ctx.eval("""handleServerMessage(JSON.stringify(Object.assign({}, %s, {
            state:"CAL", active:["arm_left_shoulder_pitch_joint","arm_right_shoulder_pitch_joint"],
            pose:{arm_left_shoulder_pitch_joint:-1.5, arm_right_shoulder_pitch_joint:1.5},
            joints:%s.joints.map(function(j){ return Object.assign({}, j, {test:false, tag:""}); })})));
          for (var i = 0; i < 30; i++) robotPose(i * 16); 1;""" % (STATUS, STATUS))
        colour = json.loads(ctx.eval("JSON.stringify(robotPose(0))"))
        glowing = colour.get("arm_left_shoulder_pitch", [0, 0, 0])
        shown = ctx.eval("robot.q.arm_left_shoulder_pitch_joint")
        if not (glowing[0] > 0.9 and glowing[1] > 0.6):
            failures.append(f"a joint the status marks active does not glow: {glowing}")
        elif abs(shown + 1.5) > 0.05:
            failures.append(f"the robot does not demonstrate the pose to hold (left pitch {shown:.2f}, want -1.50)")
        else:
            print("ok    the reach check's pose is demonstrated by the robot, with both arms glowing")

        # A x2 swaps the stereo camera for the depth + flow grid, but never while a question is up.
        ctx = robot_page()
        ctx.eval("""function apad(down){ return { handedness:"right", targetRayMode:"tracked-pointer", hand:null,
            gripSpace:{}, gamepad:{ buttons:[{value:0,pressed:false},{pressed:false},{pressed:false},
            {pressed:false},{pressed:down},{pressed:false}], axes:[0,0,0,0] } }; }
          function clickA(){ readButtons([apad(true)]); readButtons([apad(false)]); }
          clickA(); __now += 100; clickA(); 1;""")
        during_question = ctx.eval("showGrid")          # the fixture is mid-calibration
        ctx.eval("""handleServerMessage(JSON.stringify({type:"status", state:"ARMED", lines:[], joints:[], log:[],
            camera:"still"}));
          __now += 2000; clickA(); __now += 100; clickA(); 1;""")
        armed = ctx.eval("showGrid")
        grid_url = ctx.eval("visionUrl()")
        grid_said = ctx.eval("toastText")
        camera_asked = ctx.eval("ev.cam")
        ctx.eval("""handleServerMessage(JSON.stringify({type:"status", state:"ARMED", lines:[], joints:[], log:[],
            camera:"head"})); 1;""")
        if during_question:
            failures.append("A x2 during a calibration question switches to depth + flow (each A is an answer there)")
        elif not armed or grid_url != "/depthflow.jpg":
            failures.append(f"A x2 does not switch to the depth + flow grid (showGrid {armed}, asks {grid_url})")
        elif "RAFT" not in grid_said:
            failures.append(f"switching to depth + flow is not announced: {grid_said!r}")
        elif camera_asked:
            failures.append("A x2 still asks run_teleop to switch the camera")
        elif "Camera follows your head" not in ctx.eval("toastText"):
            failures.append(f"the camera switching to head mode is not announced: {ctx.eval('toastText')!r}")
        else:
            print("ok    A x2 switches to the depth + flow grid (never during a question), and says so")

    # "agent, ..." commands: look-to-centre off and on, the camera, and nonsense.
    ctx = fresh(quickjs, script)
    ctx.eval(ARC)
    def command(c, said=""):
        ctx.eval("handleServerMessage(JSON.stringify({type:'voice', state:'command', text:%s, command:%s})); 1;"
                 % (json.dumps(said), json.dumps(c)))
    command({"do": "set", "feature": "swap", "on": True}, "start the looking swap")
    in_calibration = ctx.eval("toastText")
    ctx.eval(DEV)
    command({"do": "set", "feature": "swap", "on": False}, "stop the looking swap")
    ctx.eval("gazeFor('s:M', 600); 1;")
    kept = ctx.eval("arc.focus")
    command({"do": "set", "feature": "swap", "on": True})
    ctx.eval("gazeFor('sensors', 100); gazeFor('s:M', 600); 1;")
    swapped = ctx.eval("arc.focus")
    ctx.eval("robotStatus = {state:'ARMED', camera:'track', lines:[], joints:[], log:[]}; statusAt = performance.now(); 1;")
    command({"do": "camera", "mode": "head"}, "camera follow my head")
    cam = ctx.eval("ev.camhead")
    command({"do": "unknown"}, "make me a sandwich")
    if "left stick" not in in_calibration:
        failures.append(f"asking calibration mode to swap by looking does not point to the left stick: {in_calibration!r}")
    elif kept != "sensors":
        failures.append(f"with look-to-centre switched off, looking still swapped the panels ({kept})")
    elif swapped != "s:M":
        failures.append("switching look-to-centre back on does not bring it back")
    elif cam != 1:
        failures.append("agent 'camera follow my head' does not ask run_teleop for head mode")
    elif "agent help" not in ctx.eval("toastText"):
        failures.append(f"a phrase with no command is not answered: {ctx.eval('toastText')!r}")
    else:
        print("ok    agent commands: look-to-centre off and back on, camera head mode, and nonsense is answered")

    # ---- Claude sessions and the developer view
    ctx = fresh(quickjs, script)
    ctx.eval(ARC)
    ctx.eval("""
      var __said = [];
      ws.onopen(); ws.readyState = WebSocket.OPEN;
      ws.send = function(m){ if (typeof m === "string") __said.push(JSON.parse(m)); };
      fetch = function(url){
        var body = url.indexOf("/sessions.json") === 0
          ? {sessions: [{name: "M", lines: ["> hello", "working on it"]}, {name: "CAM", lines: ["Error: nope"]}]}
          : {lidar: {t: 1, age: 0.1, points: [[0, 1.5], [90, 2.0], [180, 0.8], [270, 3.1]]},
             imu: {t: 1, age: 0.1, roll: -20.7, pitch: 82.8, yaw: -116.6, acc: [-0.99, -0.04, 0.12],
                   gyro: [0.1, 0.0, -0.2], mag: [1, 2, 3], temp: 27.6}};
        return Promise.resolve({ ok: true, json: function(){ return Promise.resolve(body); } });
      };
      pollData(1e6); updateVoiceTarget(1e6); 1;""")
    pump(ctx)
    last_target = "JSON.stringify(__said.filter(function(m){ return m.action === 'target'; }).slice(-1))"
    normal = json.loads(ctx.eval("JSON.stringify(PANEL_ORDER)"))
    normal_target = ctx.eval(last_target)
    ctx.eval("setView('developer'); polledSessions = 0; polledSensors = 0; pollData(2e6); 1;")
    pump(ctx)
    dev = json.loads(ctx.eval("JSON.stringify(PANEL_ORDER)"))
    ctx.eval("arc.init = false; updateArc(head(0, 0, 0, 0)); var atCam = lookAt('s:CAM'); hold(atCam, 400); updateVoiceTarget(__now); 1;")
    # (a glance on the way past a session must not take dictation: checked by staying on CAM below)
    looked = ctx.eval(last_target)
    ctx.eval("""hold(head(0, 0, 0, 0), 800);               // follow it to the middle
      gazeFor('sensors', 400);                               // then look at the sensors where they now are
      for (var i = 0; i < 30; i++) { __now += 50; updateArc(lookAt('sensors')); updateGaze(__now); updateVoiceTarget(__now); }
      __now += 6000; updateVoiceTarget(__now); 1;""")
    stayed = ctx.eval("voiceSession")
    ctx.eval("__glCalls = {};")
    ctx.eval(FRAME)
    drawn = ctx.eval("!!(pieceCanvas.all && canvasPanels['s:CAM']) && !pieceError")
    ctx.eval("setView('normal'); pollData(3e6); updateVoiceTarget(3e6 + 6000); 1;")
    back = json.loads(ctx.eval("JSON.stringify(PANEL_ORDER)"))
    back_target = ctx.eval(last_target)
    if normal != ["screen", "camera", "status"] or '"session":"@"' not in normal_target.replace(" ", ""):
        failures.append(f"the normal view shows sessions, or does not dictate to the PC screen's: {normal} {normal_target}")
    elif dev != ["sensors", "s:M", "s:CAM"]:
        failures.append(f"the developer view is not the sensors panel and the sessions: {dev}")
    elif '"session":"CAM"' not in looked.replace(" ", ""):
        failures.append(f"looking at session CAM does not send dictation there: {looked}")
    elif stayed != "CAM":
        failures.append(f"looking away from the sessions moved dictation to {stayed!r}; it should stay on CAM")
    elif not drawn or ctx.eval("drawError"):
        failures.append(f"the sensors or session panels were not drawn ({ctx.eval('drawError')[:80]} {ctx.eval('pieceError')[:80]})")
    elif back != ["screen", "camera", "status"] or '"session":"@"' not in back_target.replace(" ", ""):
        failures.append(f"back in the normal view the sessions stay, or dictation does not follow the PC: {back} {back_target}")
    else:
        print("ok    normal view: no session panels, and dictation goes to the session the PC's screen shows")
        print("ok    developer view: the sensors panel and the sessions; dictation follows your gaze and stays")

    # ---- the controls strip: what you are pressing, and moving with no grip held
    ctx = fresh(quickjs, script)
    ctx.eval(ARC)
    ctx.eval("""linkOpen = true;
      handleServerMessage(JSON.stringify({type:"status", state:"ARMED", swap:true, lines:[], joints:[], log:[]}));
      function gpad(held){ var b = []; for (var i = 0; i < 6; i++) b.push({pressed: held.indexOf(i) >= 0,
        value: held.indexOf(i) >= 0 ? 1 : 0}); return {buttons: b, axes: [0, 0, 0, 0]}; }
      readPad("L", gpad([1]));                               // the left grip
      var driving = controlsNote(__now)[0];
      var lit = padNow.L.grip && !padNow.L.x;
      readPad("L", gpad([]));
      function at(x){ return {tracked: true, button_pressed: false, pose: [[1,0,0,x],[0,1,0,0],[0,0,1,0],[0,0,0,1]]}; }
      for (var i = 0; i < 20; i++) { __now += 50; watchNoGrip(__now, {left: at(i * 0.02), right: null}); }
      var nudge = controlsNote(__now);
      pointer.on = true; readPad("L", gpad([1]));             // pointing at the PC, grip squeezed
      var pointing = controlsNote(__now);
      pointer.on = false; readPad("L", gpad([]));
      var shown = controlsMatrix(__now) !== null;
      // a footer on the robot panel: flush under it, and it follows when the panel moves
      function footerY(){ var m = controlsMatrix(__now).model; return m[13]; }
      function panelBottom(){ var pl = placementOf("status"); return pl.y - Math.cos(pl.tilt) * pl.size.h / 2; }
      var under = footerY() < panelBottom() && panelBottom() - footerY() < 0.08;
      setFocus("status", "test"); arc.snap = true; updateArc(head(0, 0, 0, 0));
      var before = footerY();
      spots.status.dy += 0.2; arc.snap = true; updateArc(head(0, 0, 0, 0));
      var follows = Math.abs(footerY() - before - 0.2) < 0.02 && controlsMatrix(__now).host === "status";
      handleServerMessage(JSON.stringify({type:"status", state:"STOPPED", lines:[], joints:[], log:[]}));
      __now += 5000;
      var hidden = controlsMatrix(__now) === null; 1;""")
    driving = ctx.eval("driving")
    nudge = json.loads(ctx.eval("JSON.stringify(nudge)"))
    if not ctx.eval("lit"):
        failures.append("a held grip does not light up on the controls strip")
    elif "RIGHT arm" not in driving or "mirror" not in driving:
        failures.append(f"in mirror mode the left grip should say it drives the robot's RIGHT arm: {driving!r}")
    elif not nudge[1] or "no GRIP" not in nudge[0]:
        failures.append(f"moving 30 cm with no grip held while armed is not pointed out: {nudge}")
    elif not ctx.eval("shown") or not ctx.eval("hidden"):
        failures.append("the controls strip is not shown while armed, or stays up once stopped and idle")
    elif "POINTING AT THE PC" not in json.loads(ctx.eval("JSON.stringify(pointing)"))[0]:
        failures.append("with pointer mode on, a squeezed grip is not explained (the arms ignore it)")
    elif not ctx.eval("under") or not ctx.eval("follows"):
        failures.append("the controls strip is not a footer under the robot panel that moves with it")
    else:
        print(f"ok    the controls strip lights what you press, and says which arm a grip drives: {driving!r}")
        print(f"ok    moving with no grip held while armed: {nudge[0]!r}")
        print("ok    it is a footer under the robot panel, and moves with it")

    # ---- the order: the sensors panel on the left, the Claude sessions on its right from the
    # one used last to the one used longest ago; the main screen in the middle, wherever it is
    ctx = fresh(quickjs, script)
    ctx.eval(ARC)
    ctx.eval("""
      fetch = function(url){
        var body = url.indexOf("/sessions.json") === 0
          ? {sessions: [{name: "M", lines: []}, {name: "A", lines: []}, {name: "IL", lines: []}]}
          : {lidar: null, imu: null};
        return Promise.resolve({ ok: true, json: function(){ return Promise.resolve(body); } });
      };
      setView('developer'); pollData(1e6); 1;""")
    pump(ctx)
    order = lambda: json.loads(ctx.eval("JSON.stringify(arc.slots)"))
    centred = lambda name: ctx.eval(f"Math.abs(wrapAngle(targetOf('{name}').yaw - arc.yaw)) < 1e-9")
    start = order()
    ctx.eval("setFocus('s:A', 'test'); hold(head(0, 0, 0, 0), 600); 1;")
    after_a, a_mid = order(), centred("s:A")
    ctx.eval("setFocus('sensors', 'test'); hold(head(0, 0, 0, 0), 600); 1;")
    after_sensors, sensors_mid = order(), centred("sensors")
    if start != ["sensors", "s:M", "s:A", "s:IL"]:
        failures.append(f"the arc does not start as the sensors panel, then the sessions: {start}")
    elif after_a != ["sensors", "s:A", "s:M", "s:IL"] or not a_mid:
        failures.append(f"picking session A does not bring it to the front, M right behind: {after_a}")
    elif after_sensors != after_a or not sensors_mid or ctx.eval("lastSession") != "A":
        failures.append(f"looking back at the sensors reshuffled the arc or lost session A: {after_sensors}")
    else:
        print("ok    the sensors panel on the left, the sessions on its right, the last used in front")
        print("ok    the panel you pick slides to the middle and the arc keeps that order around it")

    # ---- the sensors panel: every sensor in one panel; B breaks the piece you look at into
    # smaller panels, A puts them back, and in the developer view neither reaches the robot.
    # Driven the way the headset drives it: frames with the head pointed at a piece.
    SENSOR_RIG = """
      var SENSORS = {lidar: {t: 5, age: 0.1, points: [[0, 1.5], [45, 0.4], [90, 2.0], [180, 0.8], [270, 3.1], [300, 0.03]]},
                     imu: {t: 5, age: 0.2, roll: -20.7, pitch: 82.8, yaw: -116.6, acc: [-0.99, -0.04, 0.12],
                           gyro: [0.1, 0.0, -0.2], mag: [3342, 605, -4878], temp: 25.7}};
      fetch = function(url){
        var body = url.indexOf("/sessions.json") === 0 ? {sessions: [{name: "M", lines: []}]} : SENSORS;
        return Promise.resolve({ ok: true, json: function(){ return Promise.resolve(body); } });
      };
      var JOINTS = [];
      ["L", "R"].forEach(function(s, a){ ["pitch", "roll", "yaw", "elbow", "wrist"].forEach(function(k, i){
        JOINTS.push({n: s + " " + k + " " + (a * 5 + i + 1), q: 0.1 * i, qc: 0.12 * i, e: 0, tau: 0.5 * i, lim: 2, max: 5,
                     tag: i === 2 ? "LAG" : "", cal: true, test: false, id: a * 5 + i + 1, u: "arm_" + k,
                     p: {kp: 30, kd: 2, fl: 2, hr: 1.75, g: 0.95, flip: i === 4, when: "2026-09-24",
                         last: {ok: i !== 3, why: "stuck", at: 3.5, moved: 0.97}}});
      }); });
      var STATE = "STOPPED";
      function sendStatus(){ handleServerMessage(JSON.stringify({type: "status", state: STATE, hz: 39, motors: false,
        calibrated: 10, scale: 0.58, camera: "track", lines: [], joints: JOINTS, log: [], alert: ""})); }
      var eyeView = { projectionMatrix: new Float32Array([1,0,0,0, 0,1,0,0, 0,0,-1,-1, 0,0,-0.2,0]),
                      transform: { inverse: { matrix: new Float32Array([1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1]) } } };
      var looking = head(0, 0, 0, 0), held = { left: [], right: [] }, __tex = [];
      function qpad(hand, down){ var b = []; for (var i = 0; i < 6; i++) b.push({pressed: down.indexOf(i) >= 0, value: down.indexOf(i) >= 0 ? 1 : 0});
        return { handedness: hand, targetRayMode: "tracked-pointer", hand: null, gripSpace: {}, gamepad: { buttons: b, axes: [0, 0, 0, 0] } }; }
      function step(ms){
        __now += ms;
        sendStatus();
        var frame = { session: { requestAnimationFrame: function(){}, inputSources: [qpad("left", held.left), qpad("right", held.right)],
                                 end: function(){}, renderState: { baseLayer: new XRWebGLLayer() } },
                      getViewerPose: function(){ return { transform: looking, views: [eyeView, eyeView] }; },
                      getPose: function(){ return { transform: { matrix: new Float32Array([1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1]) }, emulatedPosition: false }; } };
        session = frame.session; refSpace = {}; passthrough = true;
        __tex = [];
        onFrame(__now, frame);
      }
      function run(ms){ for (var t = 0; t < ms; t += 20) step(20); }
      function press(which){ held.right = [which === "a" ? 4 : 5]; step(20); held.right = []; step(20); }
      function pieces(){ return JSON.stringify(sensorPieces()); }
      function lookAtPiece(id){       // the head pointed so its ray (eyes a little lower) lands mid-piece
        var pl = placementOf("sensors"), cell = sensorShape().cells[id], r = mul(rotY(pl.yaw), rotX(-pl.tilt));
        var u = (cell.x + cell.w / 2 - 0.5) * pl.size.w, v = (0.5 - cell.y - cell.h / 2) * pl.size.h;
        var p = [pl.x + r[0] * u + r[4] * v, pl.y + r[1] * u + r[5] * v, pl.z + r[2] * u + r[6] * v];
        var dx = p[0] - arc.head.x, dy = p[1] - arc.head.y, dz = p[2] - arc.head.z;
        looking = head(arc.head.x, arc.head.z, Math.atan2(-dx, -dz), Math.atan2(dy, Math.hypot(dx, dz)) + PIECE_LOOK_DROP, arc.head.y);
      }
      var __drawSensors = gfx.draw;
      gfx.draw = function(mvp, tex, uv, tint){ __tex.push(tex); return __drawSensors(mvp, tex, uv, tint); };
      linkOpen = true;
    """
    ctx = fresh(quickjs, script)
    ctx.eval(ARC)
    ctx.eval(SENSOR_RIG)
    ctx.eval("setView('developer'); pollData(1e6); 1;")
    pump(ctx)
    ctx.eval("__beacons = []; run(400); 1;")
    pump(ctx)
    one = json.loads(ctx.eval("JSON.stringify({ order: PANEL_ORDER, focus: arc.focus, pieces: sensorPieces(),"
                              " draws: __tex.filter(function(t){ return pieceCanvas.all && t === pieceCanvas.all.texture; }).length,"
                              " w: sizeOf('sensors').w, cam: camOn, err: pieceError + drawError })"))
    ctx.eval("press('b'); run(300); 1;")
    six = json.loads(ctx.eval("JSON.stringify({ pieces: sensorPieces(), w: sizeOf('sensors').w, toast: toastText,"
                              " draws: __tex.length, ev: [ev.a, ev.b], err: pieceError })"))
    cells = json.loads(ctx.eval("JSON.stringify(sensorShape().cells)"))
    overlap = [f"{a}/{b}" for a in cells for b in cells if a < b
               and cells[a]["x"] < cells[b]["x"] + cells[b]["w"] and cells[b]["x"] < cells[a]["x"] + cells[a]["w"]
               and cells[a]["y"] < cells[b]["y"] + cells[b]["h"] and cells[b]["y"] < cells[a]["y"] + cells[a]["h"]]
    outside = [k for k, c in cells.items() if c["x"] < -1e-9 or c["y"] < -1e-9
               or c["x"] + c["w"] > 1 + 1e-9 or c["y"] + c["h"] > 1 + 1e-9]
    ctx.eval("lookAtPiece('imu'); run(400); 1;")
    picked = ctx.eval("sensorNav.sel")
    outline = ctx.eval("pieceTiles(placementOf('sensors'), true).border.length")
    ctx.eval("press('b'); run(300); 1;")
    imu = json.loads(ctx.eval("pieces()"))
    ctx.eval("lookAtPiece('imu:tilt'); run(400); press('b'); press('b'); run(500); 1;")   # a double press: once
    alone = json.loads(ctx.eval("pieces()"))
    sharp = ctx.eval("pieceCanvas['imu:tilt'].k")
    ctx.eval("press('b'); run(600); 1;")
    back_all = json.loads(ctx.eval("pieces()"))
    # down to one joint and back up with A, a level at a time
    ctx.eval("press('b'); run(300); lookAtPiece('arms'); run(400); press('b'); run(600); press('b'); run(600); 1;")
    joints = json.loads(ctx.eval("pieces()"))
    titles = json.loads(ctx.eval("JSON.stringify(sensorPieces().map(sensorTitle))"))
    ctx.eval("press('a'); run(600); 1;")
    up1 = json.loads(ctx.eval("JSON.stringify([sensorPieces(), sensorNav.sel])"))
    ctx.eval("press('a'); run(600); 1;")
    up2 = json.loads(ctx.eval("JSON.stringify([sensorPieces(), sensorNav.sel])"))
    ctx.eval("press('a'); run(600); 1;")
    up3 = json.loads(ctx.eval("pieces()"))
    robot_heard = json.loads(ctx.eval("JSON.stringify([ev.a, ev.b])"))
    if one["order"] != ["sensors", "s:M"] or one["focus"] != "sensors" or one["pieces"] != ["all"]:
        failures.append(f"the developer view does not open on one sensors panel: {one}")
    elif one["draws"] != 2 or one["err"]:
        failures.append(f"'All sensors' is not drawn once per eye per frame, without errors: {one}")
    elif not one["cam"]:
        failures.append("'All sensors' shows the camera, but the camera does not stream for it")
    elif six["pieces"] != ["camera", "lidar", "imu", "arms", "you", "progs"] or "6 panels" not in six["toast"]:
        failures.append(f"B does not break 'All sensors' into one panel per sensor, and say so: {six}")
    elif not six["w"] > one["w"] or overlap or outside:
        failures.append(f"broken apart, the panel should grow, its pieces neither overlapping nor outside it: "
                        f"{one['w']:.2f} -> {six['w']:.2f} m, overlap {overlap}, outside {outside}")
    elif picked != "imu" or outline != 4:
        failures.append(f"looking at the IMU piece does not pick and outline it: picked {picked}, {outline} outline strips")
    elif imu != ["imu:tilt", "imu:acc", "imu:gyro", "imu:mag"]:
        failures.append(f"B on the IMU does not break it into its parts: {imu}")
    elif alone != ["imu:tilt"] or sharp != 1.4:
        failures.append(f"B B on a part should show it alone (once) and sharper: {alone}, canvas x{sharp}")
    elif back_all != ["all"]:
        failures.append(f"B on a piece shown alone does not put everything back in one panel: {back_all}")
    elif joints != ["j:0", "j:1", "j:2", "j:3", "j:4"] or titles[0] != "Left pitch (CAN 1)":
        failures.append(f"B, B, B (looking at the arms) does not reach the left arm's joints: {joints} {titles}")
    elif up1 != [["arm:L", "arm:R"], "arm:L"] or up2 != [["camera", "lidar", "imu", "arms", "you", "progs"], "arms"] \
            or up3 != ["all"]:
        failures.append(f"A does not go back up a level at a time, the piece you came from picked: {up1} {up2} {up3}")
    elif robot_heard != [0, 0]:
        failures.append(f"in the developer view A and B reached the robot (A x{robot_heard[0]}, B x{robot_heard[1]})")
    else:
        print("ok    developer view: one sensors panel, 'All sensors', drawn once per eye; B breaks it into one panel")
        print("      per sensor (it grows, nothing overlaps), looking picks and outlines a piece, B breaks that into")
        print("      its parts, the smallest alone and sharper, then all in one again; B B counts once")
        print("ok    B B B reaches the left arm's joints; A goes back a level at a time; the robot heard no A or B")

    # every piece draws, with live-looking data, with none, and with nonsense
    bad = []
    for name, setup in (("live data", ""),
                        ("no data", "sensors = {lidar: null, imu: null}; robotStatus = null; sensorHands = {left: null, right: null};"),
                        ("nonsense", "sensors = {lidar: {t: 1, age: 'x', points: 'nope'}, imu: {t: 2, age: null, acc: 'x', gyro: [null],"
                                     " mag: {}, roll: 'a'}}; robotStatus = {type: 'status', state: 'ARMED', hz: 'x', motors: 1, lines: [],"
                                     " log: [], joints: [null, {n: 7, q: 'x', p: null}, {}]}; statusAt = performance.now();"
                                     " noteSensors(); notePage(performance.now(), __now);")):
        ctx.eval(setup + "; 1;")
        for k in (1, 1.4):
            errors = json.loads(ctx.eval(f"""JSON.stringify(Object.keys(SENSOR_TREE).filter(function(id){{ return id !== 'camera'; }})
              .map(function(id){{ pieceError = ''; drawPiece(id, {k}); return pieceError; }}).filter(Boolean))"""))
            bad += [f"{name}: {e[:140]}" for e in errors]
    if bad:
        failures.append(f"sensors panel pieces fail to draw: {bad[:3]}")
    else:
        print(f"ok    all {ctx.eval('Object.keys(SENSOR_TREE).length - 1')} pieces draw with live data, with none, and with nonsense")

    # the camera piece is the stereo stream itself, each eye its own half; its tasks' pictures
    # are asked for one at a time only while their pieces are up, and the camera only while shown
    ctx = fresh(quickjs, script)
    ctx.eval(ARC)
    ctx.eval(SENSOR_RIG)
    ctx.eval("setView('developer'); run(200); camReady = true; camImg.naturalWidth = 1280; camImg.naturalHeight = 480;"
             " press('b'); run(300); 1;")
    eyes = json.loads(ctx.eval("JSON.stringify([pieceUv('camera', 'left'), pieceUv('camera', 'right'), cameraEvery()])"))
    cam_drawn = ctx.eval("__tex.filter(function(t){ return t === cameraTexture; }).length")
    ctx.eval("lookAtPiece('camera'); run(400); __beacons = []; press('b'); run(700); 1;")
    tasks = json.loads(ctx.eval("pieces()"))
    asked = [b for b in json.loads(ctx.eval("JSON.stringify(__beacons)")) if "/vision.jpg" in b or "/depthflow.jpg" in b]
    ctx.eval("""var __texts = [], __fill = Ctx.prototype.fillText;
      Ctx.prototype.fillText = function(t){ __texts.push(String(t)); };
      drawPiece('cam:track', 1); var waiting = __texts.join('|');
      piecePics['cam:track'].img.naturalWidth = 1280; piecePics['cam:track'].img.naturalHeight = 480;
      var before = pieceCanvas['cam:track'].at; piecePics['cam:track'].img.onload(); __texts = []; run(40);
      var shown = __texts.join('|'); Ctx.prototype.fillText = __fill; 1;""")
    redrawn = ctx.eval("pieceCanvas['cam:track'].at > before && !pieceError && waiting.indexOf('waiting for person_track') >= 0"
                       " && shown.indexOf('a box for the person') >= 0 && shown.indexOf('waiting for person_track') < 0")
    ctx.eval("press('a'); press('a'); run(600); lookAtPiece('imu'); run(400); press('b'); __beacons = []; run(1500); 1;")
    after = [b for b in json.loads(ctx.eval("JSON.stringify(__beacons)")) if ".jpg" in b or ".mjpg" in b]
    streaming = ctx.eval("camOn")
    uv_ok = (eyes[0] is not None and eyes[1] is not None and abs(abs(eyes[0][3]) - (640 / 480) / (800 / 560)) < 1e-6
             and abs(eyes[0][2]) == 0.5 and eyes[0] != eyes[1] and eyes[2] == 33)
    if not uv_ok or cam_drawn != 2:
        failures.append(f"the camera piece is not the live stereo stream, each eye its half, trimmed to the piece: "
                        f"{eyes}, drawn {cam_drawn} times")
    elif tasks != ["cam:track", "cam:depth"]:
        failures.append(f"B on the camera does not break it into the tasks that run on it: {tasks}")
    elif sorted(a.split("?")[0] for a in asked) != ["/depthflow.jpg", "/vision.jpg"]:
        failures.append(f"the tracker's and depth's pictures are not asked for one at a time: {asked}")
    elif not redrawn:
        failures.append("a tracker picture that arrives is not drawn into its piece")
    elif after or streaming:
        failures.append(f"with the IMU's parts up, pictures or the camera are still asked for: {after}, camera {streaming}")
    else:
        print("ok    the camera piece is the live stereo stream (each eye its half, trimmed to fit); B on it shows the")
        print("      tracker and depth + flow, their pictures asked for one at a time only while shown, and the")
        print("      camera streams only while 'All sensors' or the camera is up")

    # a question from the robot: A and B answer it as everywhere, and the panel stays put;
    # calibration mode keeps B B for the PC pointer; spoken commands find the sensors panel
    ctx = fresh(quickjs, script)
    ctx.eval(ARC)
    ctx.eval(SENSOR_RIG)
    ctx.eval("setView('developer'); run(200); STATE = 'CAL'; run(400); press('b'); run(300); 1;")
    asked_robot = json.loads(ctx.eval("JSON.stringify([ev.b, sensorPieces(), toastText])"))
    ctx.eval("STATE = 'STOPPED'; run(1500); setView('normal'); arc.init = false; run(100); press('b'); run(100); press('b'); run(100); 1;")
    pointing = ctx.eval("pointer.on")
    ctx.eval("press('b'); run(100); press('b'); run(100); 1;")         # and off again
    said = lambda c, text: ctx.eval("handleServerMessage(JSON.stringify({type: 'voice', state: 'command', text: %s, command: %s})); run(100); 1;"
                                    % (json.dumps(text), json.dumps(c)))
    said({"do": "view", "view": "developer"}, "Developer view.")
    dev_focus = json.loads(ctx.eval("JSON.stringify([view, arc.focus])"))
    said({"do": "focus", "panel": "camera"}, "Show camera.")
    cam_alone = json.loads(ctx.eval("JSON.stringify([sensorPieces(), toastText])"))
    said({"do": "focus", "panel": "screen"}, "Show screen.")
    no_screen = ctx.eval("toastText")
    said({"do": "view", "view": "developer"}, "All sensors.")
    all_again = json.loads(ctx.eval("pieces()"))
    if asked_robot[0] != 1 or asked_robot[1] != ["all"] or "calibration mode" not in asked_robot[2]:
        failures.append(f"while the robot asks, B should answer it, leave the panel, and say where the question is: {asked_robot}")
    elif not pointing or ctx.eval("pointer.on"):
        failures.append("in calibration mode B B no longer points at the PC (and back)")
    elif dev_focus != ["developer", "sensors"]:
        failures.append(f"'agent, developer view' does not bring the sensors panel up: {dev_focus}")
    elif cam_alone[0] != ["camera"] or "stereo camera" not in cam_alone[1]:
        failures.append(f"'agent, show camera' in the developer view does not show the camera piece: {cam_alone}")
    elif "calibration mode" not in no_screen:
        failures.append(f"'agent, show screen' in the developer view does not say where the screen is: {no_screen!r}")
    elif all_again != ["all"]:
        failures.append(f"'agent, all sensors' does not put everything back in one panel: {all_again}")
    else:
        print("ok    while the robot asks, B answers it and the sensors panel stays; calibration mode keeps B B for")
        print("      the PC pointer; 'agent, developer view', 'show camera', 'all sensors' find the sensors panel")

    # ---- a session's panel: its whole conversation, long lines wrapped rather than cut,
    # and a strip under it: listening, the words as they come, sent, and Claude on it
    ctx = fresh(quickjs, script)
    ctx.eval(ARC)
    ctx.eval("""
      var __texts = [], __claude = { state: "idle", status: "Baked for 10s · done 6:53 PM" };
      Ctx.prototype.fillText = function(t){ __texts.push(String(t)); };
      var long = "● " + "the conversation goes on and on ".repeat(12) + "END";
      fetch = function(url){
        var body = url.indexOf("/sessions.json") === 0
          ? {sessions: [{name: "M", lines: ["❯ hello"], state: "idle", status: "", typed: "", topic: ""},
                        {name: "A", lines: ["● Update(scripts/teleop/servos.py)", long],
                         state: __claude.state, status: __claude.status, typed: "", topic: "Demo video scripts"}]}
          : {lidar: null, imu: null};
        return Promise.resolve({ ok: true, json: function(){ return Promise.resolve(body); } });
      };
      lastSession = "A"; voiceSession = "A"; setView('developer'); pollData(1e6); 1;""")
    pump(ctx)
    rows = json.loads(ctx.eval("JSON.stringify(wrapLine(long, 85))"))
    ctx.eval("__texts = []; drawSessionPanel('A'); 1;")
    drawn_text = json.loads(ctx.eval("JSON.stringify(__texts)"))
    ctx.eval("__texts = []; drawSessionPanel('M'); 1;")
    drawn_m = json.loads(ctx.eval("JSON.stringify(__texts)"))
    strip = lambda name: json.loads(ctx.eval(f"JSON.stringify(sessionStrip('{name}', __now))"))
    voice = lambda message: ctx.eval(f"handleServerMessage(JSON.stringify({json.dumps(message)})); refreshStrips(); 1;")
    voice({"type": "voice", "state": "listening", "text": "wake"})
    voice({"type": "voice", "state": "partial", "text": "make the camera panel bigger"})
    listening, listening_m = strip("A"), strip("M")
    redrawn = ctx.eval("canvasPanels['s:A'].strip")
    voice({"type": "voice", "state": "heard", "text": "5 s of silence"})
    heard = strip("A")
    voice({"type": "voice", "state": "sent", "text": "make the camera panel bigger"})
    sent = strip("A")
    ctx.eval("""__claude = { state: "working", status: "Gallivanting… (3s · ↓ 1.2k tokens)" };
      __now += 1500; pollData(2e6); 1;""")
    pump(ctx)
    working = strip("A")
    ctx.eval("__claude = { state: 'idle', status: 'Baked for 10s · done 6:53 PM' }; sentTo.A.took = false;"
             "sentTo.A.status = __claude.status; __now += 9000; pollData(3e6); 1;")
    pump(ctx)
    stuck = strip("A")
    if max(map(len, rows)) > 85 or " ".join(r.strip() for r in rows) != ctx.eval("long").replace("  ", " "):
        failures.append(f"a long terminal line is not wrapped into the panel's width, whole: {rows}")
    elif not any("END" in t for t in drawn_text):
        failures.append("the end of a long line never reaches the session panel")
    elif "Claude A · Demo video scripts" not in drawn_text or "Claude M · new session" not in drawn_m:
        failures.append(f"a session panel is not headed with what it is about: {drawn_text[:1]} / {drawn_m[:1]}")
    elif not (listening[0]["text"].startswith("● LISTENING") and "camera panel bigger" in listening[0].get("words", "")):
        failures.append(f"the session dictation goes to does not say it is listening, with the words: {listening[0]}")
    elif "LISTENING" in listening_m[0]["text"] or "LISTENING" not in (redrawn or ""):
        failures.append(f"LISTENING shows on the wrong session, or the panel was not redrawn: M={listening_m[0]}")
    elif "writing it down" not in heard[0]["text"] or "waiting for Claude A to start" not in sent[0]["text"]:
        failures.append(f"the strip does not follow the sentence to Claude: {heard[0]['text']!r} / {sent[0]['text']!r}")
    elif working[0]["text"] != "✓ Sent “make the camera panel bigger”" or "is working · Gallivanting" not in working[1]["text"]:
        failures.append(f"the strip does not say Claude took the sentence up: {working}")
    elif "has not started" not in stuck[0]["text"]:
        failures.append(f"a sentence Claude never started on is not pointed out: {stuck[0]['text']!r}")
    else:
        print("ok    a session panel is headed with what it is about, wraps long lines instead of cutting them,")
        print("      and shows the end of the conversation")
        print("ok    its strip: LISTENING with the words, then sent, then Claude working on it (or not started)")

    # ---- the agent's menu: "agent, help" opens it in front of you, fixed in the room, and
    # "agent, help" again (or three minutes) closes it. It shows the last phrase heard,
    # nothing is picked through it, and every phrase it offers does what it says in
    # voice_typer's own parser. Toasts wrap instead of being cut at 35 characters.
    sys.path.insert(0, str(PAGE.parent))
    from voice_typer import parse_command
    ctx = fresh(quickjs, script)
    ctx.eval(ARC)
    agent = lambda c, said: ctx.eval("handleServerMessage(JSON.stringify({type:'voice', state:'command', "
                                     f"text:{json.dumps(said)}, command:{json.dumps(c)}}})); 1;")
    ctx.eval("arc.init = false; updateArc(head(0, 0, 0.3, 0)); 1;")
    agent({"do": "help"}, "Help.")
    opened, card_yaw = ctx.eval("helpCard.open"), ctx.eval("helpCard.yaw")
    ctx.eval(FRAME)
    drawn = ctx.eval("!!helpCard.texture && helpCard.key !== '' && !drawError")
    gazes = json.loads(ctx.eval("helpCard.open = false; var __g1 = gazeTarget(); helpCard.open = true;"
                                "JSON.stringify([__g1, gazeTarget()])"))
    agent({"do": "unknown"}, "Exit the Volipur view.")
    heard = ctx.eval("helpKey()")
    agent({"do": "help"}, "Help.")
    closed = not ctx.eval("helpCard.open")
    agent({"do": "help"}, "Help.")
    ctx.eval("helpCard.at -= 200000;")
    ctx.eval(FRAME)
    timed_out = not ctx.eval("helpCard.open")
    agent({"do": "close"}, "Close.")                  # nothing open: harmless
    closed_idle = not ctx.eval("helpCard.open")
    agent({"do": "help"}, "Help.")
    agent({"do": "close"}, "Close.")
    closed_by_close = not ctx.eval("helpCard.open")
    menu = json.loads(ctx.eval("JSON.stringify(AGENT_MENU)"))
    phrases = [(phrase, wanted) for group in menu for item in group["items"] for phrase, wanted in item["tests"]]
    pressed = " | ".join(item["press"] for group in json.loads(ctx.eval("JSON.stringify(BUTTON_MENU)"))
                         for item in group["items"])
    missing = [b for b in ("X ×3", "Y ×3", "A ×2", "B ×2", "A · B", "grip", "L stick", "R stick", "L trigger",
                           "R trigger", "hold Y") if b not in pressed]
    # (what the menu promises must match; anything more the parser adds - the words said - may differ)
    wrong = [f"'agent, {phrase}' does {parse_command(phrase)}, not {wanted}" for phrase, wanted in phrases
             if any(parse_command(phrase).get(key) != value for key, value in wanted.items())]
    long_toast = ("Agent: calibration mode: PC screen, camera, robot - L stick click switches; "
                  "you talk to the session on the PC screen")
    toast_lines = json.loads(ctx.eval(f"JSON.stringify(wrapWords(toastCtx, {json.dumps(long_toast)}, TOAST_W - 48, 3))"))
    if not opened or abs(card_yaw - 0.3) > 1e-6 or not drawn:
        failures.append(f"'agent, help' does not put the menu in front of you and draw it "
                        f"(open {opened}, yaw {card_yaw}, drawn {drawn}, {ctx.eval('drawError')[:80]})")
    elif gazes[0] is None or gazes[1] is not None:
        failures.append(f"a panel is picked through the open menu (closed: {gazes[0]}, open: {gazes[1]})")
    elif "Volipur" not in heard or "false" not in heard:
        failures.append(f"the menu does not show a phrase that was not a command: {heard[-60:]!r}")
    elif not closed or not timed_out:
        failures.append(f"the menu does not close on 'agent, help' ({closed}) or after three minutes ({timed_out})")
    elif not closed_by_close or not closed_idle:
        failures.append(f"'agent, close' does not put the menu away ({closed_by_close}), "
                        f"or opens it when nothing was open ({closed_idle})")
    elif wrong:
        failures.append("the menu offers phrases that do something else: " + "; ".join(wrong))
    elif missing:
        failures.append(f"the buttons card leaves out {missing}")
    elif " ".join(toast_lines) != long_toast or len(toast_lines) < 2:
        failures.append(f"a long toast is cut instead of wrapped: {toast_lines}")
    else:
        print("ok    'agent, help' puts two cards in front of you - every button, every spoken command, the last")
        print("      phrase heard - picks no panel through them, and puts them away again (or by itself)")
        print(f"ok    every phrase on the menu does what it says in voice_typer's parser ({len(phrases)} phrases)")
        print("ok    toasts wrap onto up to three lines instead of being cut at 35 characters")

    # ---- the vision pictures: both eyes, side by side, just above the camera panel
    ctx = fresh(quickjs, script)
    ctx.eval(ARC)
    geometry = """JSON.stringify((function(){
      var cam = placementOf('camera'), pics = visionPlacements();
      return { camW: cam.size.w, camYaw: cam.yaw, top: cam.y + Math.cos(cam.tilt) * cam.size.h / 2,
               pics: pics.map(function(p){ return { x: p.x, z: p.z, w: p.size.w, yaw: p.yaw, uv: p.uv,
                 bottom: p.y - Math.cos(p.tilt) * p.size.h / 2 }; }) };
    })())"""
    ctx.eval("__now += 9000; arc.focus = 'camera'; arc.init = false; updateArc(head(0, 0, 0, 0)); 1;")
    main_geo = json.loads(ctx.eval(geometry))
    ctx.eval("setFocus('screen', 'test'); hold(head(0, 0, 0, 0), 1500); 1;")    # the camera slides aside
    aside = json.loads(ctx.eval(geometry))
    ctx.eval("""setFocus('camera', 'test'); hold(head(0, 0, 0, 0), 1500);
      var pics = visionPlacements(), mid = { x: (pics[0].x + pics[1].x) / 2, y: pics[0].y, z: (pics[0].z + pics[1].z) / 2 };
      var dx = mid.x - arc.head.x, dy = mid.y - arc.head.y, dz = mid.z - arc.head.z;
      hold(head(0, 0, Math.atan2(-dx, -dz), Math.atan2(dy, Math.hypot(dx, dz))), 1500); 1;""")
    looked_up = ctx.eval("arc.focus")
    ctx.eval("""var __visDraws = 0, __drawVia = gfx.draw;
      gfx.draw = function(mvp, tex, uv, tint){
        if (tex === visionTexture || tex === visOffTexture) __visDraws += 1;
        return __drawVia(mvp, tex, uv, tint); };
      __beacons = []; __now += 9000; 1;""")
    ctx.eval(FRAME)
    asked = [b for b in json.loads(ctx.eval("JSON.stringify(__beacons)")) if "/vision.jpg" in str(b)]
    draws_on = ctx.eval("__visDraws")
    ctx.eval("visImg.naturalWidth = 1280; visImg.naturalHeight = 480; visImg.onload(); __beacons = []; 1;")
    ctx.eval(FRAME)                                    # a picture came: shown, and the next asked for
    again = [b for b in json.loads(ctx.eval("JSON.stringify(__beacons)")) if "/vision.jpg" in str(b)]
    shown = ctx.eval("visHave")
    command({"do": "set", "feature": "vision", "on": False}, "vision off")
    said_off = ctx.eval("toastText")
    ctx.eval("__beacons = []; __visDraws = 0; __now += 9000; 1;")
    ctx.eval(FRAME)
    asked_off = [b for b in json.loads(ctx.eval("JSON.stringify(__beacons)")) if "/vision.jpg" in str(b)]
    draws_off = ctx.eval("__visDraws")
    left, right = main_geo["pics"]
    if len(main_geo["pics"]) != 2 or min(left["bottom"], right["bottom"]) < main_geo["top"] + 0.01:
        failures.append(f"the vision pictures are not above the camera panel: {main_geo}")
    elif not (right["x"] - left["x"] >= left["w"] + 0.02 and left["uv"] == [0, 0, 0.5, 1] and right["uv"] == [0.5, 0, 0.5, 1]):
        failures.append(f"the two eyes are not side by side, left then right, with a gap: {main_geo['pics']}")
    elif abs(2 * left["w"] + 0.025 - 0.8 * main_geo["camW"]) > 0.01:
        failures.append(f"the pair is not sized from the camera panel: {2 * left['w']:.2f} m for {main_geo['camW']:.2f} m")
    elif abs(aside["pics"][0]["yaw"] - aside["camYaw"]) > 1e-6 or abs(aside["camYaw"]) < 0.3 \
            or aside["pics"][0]["w"] >= left["w"] or min(p["bottom"] for p in aside["pics"]) < aside["top"]:
        failures.append(f"the pictures do not follow the camera panel to the side: {aside}")
    elif looked_up != "camera":
        failures.append(f"looking up at the vision pictures swapped the panels (main is now {looked_up})")
    elif len(asked) != 1:
        failures.append(f"the page should ask for one /vision.jpg at a time while the pictures are up: {asked}")
    elif not again or not shown:
        failures.append(f"a picture that arrives is not shown ({shown}), or the next is not asked for ({again})")
    elif asked_off or "vision off" not in said_off:
        failures.append(f"'agent, vision off' does not stop the stream ({asked_off}, {said_off!r})")
    elif draws_on != 2 * 2 * 6 or draws_off:
        failures.append(f"the vision pictures should be drawn 2 per eye per frame (24), then none: {draws_on}, {draws_off}")
    else:
        print("ok    both eyes sit side by side above the camera panel, follow it aside, and looking up there swaps nothing")
        print("ok    /vision.jpg is asked for one picture at a time while shown; 'agent, vision off' stops both")

    # ---- A x2: the depth + flow grid takes the stereo camera's place on the arc, bigger
    ctx = fresh(quickjs, script)
    ctx.eval(ARC)

    def requests(kind):
        return [b for b in json.loads(ctx.eval("JSON.stringify(__beacons)")) if kind in str(b)]
    where = """JSON.stringify((function(){ var t = targetOf('camera'), h = t.w / aspectOf('camera');
      return { w: t.w, bottom: t.dy - h / 2, reach: reachOf('camera') }; })())"""
    ctx.eval("__now += 9000; arc.focus = 'camera'; arc.init = false; updateArc(head(0, 0, 0, 0)); 1;")
    ctx.eval(FRAME)
    ctx.eval("visImg.naturalWidth = 1280; visImg.naturalHeight = 480; visImg.onload(); __now += 400; 1;")
    ctx.eval(FRAME)                                    # the eyes are up, and the next eyes asked for
    streaming = ctx.eval("camOn")
    camera = json.loads(ctx.eval(where))
    ctx.eval("toggleGrid(); 1;")
    grid = json.loads(ctx.eval(where))                 # measured together: FRAME's sticks resize panels
    stood_in = ctx.eval("visHave")
    ctx.eval("visImg.onload(); __beacons = []; 1;")    # ... and they land after A x2
    late = ctx.eval("visNew")
    ctx.eval(FRAME)
    grid_asked, eyes_asked = requests("/depthflow.jpg"), requests("/vision.jpg")
    still_streaming = ctx.eval("camOn")
    ctx.eval("""visImg.naturalWidth = 960; visImg.naturalHeight = 720; visImg.onload();
      var __draws = [], __drawVia = gfx.draw;
      gfx.draw = function(mvp, tex, uv, tint){
        __draws.push({ grid: tex === visionTexture, cam: tex === cameraTexture, uv: uv || null });
        return __drawVia(mvp, tex, uv, tint); }; 1;""")
    ctx.eval(FRAME)
    draws = [d for d in json.loads(ctx.eval("JSON.stringify(__draws)")) if d["grid"] or d["cam"]]
    called = ctx.eval("panelName('camera')")
    ctx.eval("gfx.draw = __drawVia; toggleGrid(); __beacons = []; __now += 400; 1;")
    ctx.eval(FRAME)
    back, camera_back = requests("/vision.jpg"), requests("/camera.mjpg")
    ctx.eval("showVision = false; toggleGrid(); __beacons = []; __now += 400; 1;")
    ctx.eval(FRAME)
    without_eyes = requests("/depthflow.jpg")
    if stood_in or late:
        failures.append(f"after A x2 the eyes stand in for the depth grid (visHave {stood_in}, late picture kept {late})")
    elif len(grid_asked) != 1 or eyes_asked:
        failures.append(f"after A x2 the page should ask for /depthflow.jpg only: {grid_asked} {eyes_asked}")
    elif not streaming or still_streaming:
        failures.append(f"the stereo camera should stream until A x2 and stop after it ({streaming}, {still_streaming})")
    elif draws != [{"grid": True, "cam": False, "uv": None}] * (2 * 6):     # per eye, in each of FRAME's 6
        failures.append(f"the grid should be drawn once per eye, whole and flat, and nothing else of the camera: {draws}")
    elif abs(grid["w"] - max(camera["w"], min(1.25 * camera["w"], 1.55 * camera["reach"]))) > 0.01 \
            or grid["w"] <= camera["w"] or abs(grid["bottom"] - camera["bottom"]) > 0.005:
        failures.append(f"the grid should be 1.25x the camera panel (at most ~75 deg wide), growing up from its "
                        f"bottom edge: {camera} -> {grid}")
    elif called != "depth + flow":
        failures.append(f"the grid's place is still called {called!r}")
    elif len(back) != 1 or not camera_back:
        failures.append(f"A x2 again does not bring the eyes and the stereo camera back: {back} {camera_back}")
    elif len(without_eyes) != 1:
        failures.append(f"with 'vision off', A x2 does not show the grid: {without_eyes}")
    else:
        print("ok    A x2 puts depth over RAFT flow (/depthflow.jpg) in the stereo camera's place, 1.25x wider")
        print("      (at most ~75 deg of view), growing up from its bottom edge; the camera stream and the small")
        print("      pair stop, a late eyes picture is dropped, and A x2 again brings them back; 'vision off' too")

    if args.live:
        payload = live_status()
        if payload:
            ctx = fresh(quickjs, script)
            ctx.eval("linkOpen = true; screenReady = false; __glCalls = {};")
            try:
                ctx.eval("handleServerMessage(" + json.dumps(payload) + ");")
                ctx.eval(FRAME)
            except Exception as exc:
                failures.append(f"the robot's own status breaks the page: {str(exc).splitlines()[0]}")
            else:
                if ctx.eval("drawError"):
                    failures.append(f"the robot's own status paints an error: {ctx.eval('drawError')[:120]}")
                else:
                    print(f"ok    the robot's live status renders "
                          f"({ctx.eval('robotStatus.joints.length')} joints, "
                          f"{ctx.eval('__glCalls.drawArrays')} panels drawn)")
                    print(f"      page would show: {ctx.eval('ui.robot.textContent')}")

    print()
    for failure in failures:
        print("FAIL  " + failure)
    print("the page is sound" if not failures else f"{len(failures)} problem(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
