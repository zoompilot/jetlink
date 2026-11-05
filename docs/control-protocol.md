# Control protocol

How the apps and the web page talk to the server. There is no socket: each
runs the server in its own process and uses `ServerController` (in
`JetlinkKit/Sources/JetlinkServer`) directly.

| Who | How |
| --- | --- |
| Mac and iPhone apps | Swift calls: `handle(_:)` runs a `ControlCommand`, `events` delivers each `ControlEvent` |
| Android app | Each command as a JSON object through JNI (`Native.command`), a JSON reply back; the screens draw a snapshot built from the events |
| Web page (Jetson, PC) | Relays events to browsers as Server-Sent Events; a signed-in page sends some commands as JSON over HTTP |

It has no version number: every caller is built with the server it talks to.
The JSON below is what
`ControlEvent.jsonLine()` writes and the page streams: one object per line,
snake_case keys, sorted. An optional field with no value is left out, and a
reader takes a missing key as `null`; the examples below show such fields as
`<value>|null`.

<a id="the-status-page"></a>
## The web page

`jetlink-server --status-port P --web-auth FILE` (the installed service uses
5600 and `/etc/jetlink/web-auth.json`) serves a page to any browser on the
network. With the password file, everything but the page and the sign-in needs
the sign-in cookie; without it the page is read-only and the `/api/` controls
answer 403.

| Path | What |
| --- | --- |
| `GET /` | the page |
| `GET /api/session` | `{"auth": "none"\|"required", "signed_in"}`; renews a cookie over a day old |
| `POST /api/login` | `{"password"}`: sets the cookie, or 401; 429 with `retry_after` while held up |
| `POST /api/logout` | clears the cookie on this browser |
| `POST /api/logout-all` | a new signing key: every other browser signs in again |
| `POST /api/password` | `{"current", "new"}`: a new password and key, a new cookie for this browser |
| `GET /events` | Server-Sent Events: on connect the latest `hello`, `server`, `link`, `engine`, `inventory`, `catalog`, `download` and `benchmark`, the last two minutes of `stats`, then `host` and one `hw` a second while a page is open |
| `GET /logs` | the server's last 300 log lines, as text |
| `POST /api/command` | one of the [commands](#commands) below: `status`, `catalog`, `download`, `cancel_download`, `prepare`, `forget`, `inventory`, `benchmark`, `cancel_benchmark`; the reply's `ok` false is a 409 |
| `GET /api/system` | `settings`: the installer's answers that apply here, in its words; the version, keep-awake, the last run |
| `POST /api/settings` | `{"power": "always"\|"switched", "comma_poweroff": "yes"\|"no", "desktop": "on"\|"off", "autostart": "yes"\|"no"}`, only those that apply: runs `jetlink setup --set ...` |
| `POST /api/action` | `{"action": "restart"\|"reboot"\|"poweroff"\|"update"\|"check_update"\|"keep_awake", "seconds"}` |
| `GET /api/task` | the settings or update run: `kind`, `state` (`running`, `done`, `failed`, `lost`), `exit`, its last lines |

- **Cookie:** `jetlink_session`, HttpOnly, SameSite=Strict, 400 days. It is an
  expiry, a nonce and an HMAC-SHA256 under the file's key, so the server keeps
  nothing and a restart signs nobody out.
- **Password file:** root's, 0600: PBKDF2-HMAC-SHA256 (210,000 rounds) of the
  password, its salt, and the 32-byte signing key. The server reads it again
  when it changes. `jetlink-server web-password --file F [--generate]` writes
  one.
- **Guessing:** five tries an address, then a wait that doubles up to five
  minutes; two password checks a second at most from everyone together, one
  at a time.
- **Other sites:** a POST needs `Content-Type: application/json`, the header
  `X-Jetlink: 1` (no other site can send it without a preflight, which is never
  answered) and, when the browser sends one, an `Origin` naming this host.
- **Driving:** while the comma is connected and frames came in the last 10 s,
  `prepare`, `download`, `forget`, `benchmark`, settings, updates, restarts,
  reboots and power-off answer 409.
- **Runs:** settings and updates run the installer as the transient unit
  `jetlink-web-task` (`systemd-run`, nice 10), outside the server's own unit,
  since they restart it. Its output and exit status are in `/run/jetlink-web/`.
  Keep-awake is `jetlink caffeinate -t` as `jetlink-web-awake`, so it outlasts
  a restart too; Check for updates is `jetlink update --check`.

```bash
curl -c jar -H 'X-Jetlink: 1' -H 'Content-Type: application/json' \
  -d '{"password": "<password>"}' http://<name>.local:5600/api/login
curl -b jar -N http://<name>.local:5600/events
```

`host` (hostname, board, OS, kernel, GPU) and `hw` (CPU, memory, GPU,
temperatures, power, fan, disk) come from the Linux host; other hosts leave
them out.

## Events

`{"event": "<name>", "t": <unix time, float seconds>, ...}`. Placeholders in
angle brackets.

```jsonc
{"event":"server","t":0,"state":"serving","detail":"","backend":"trt",
 "runtime_version":"10.16.2.10","device":"Orin-sm87"}
// state: "serving" | "stopping". backend, runtime_version and device are what
// the hello to the comma carries.

{"event":"link","t":0,"state":"connected","detail":"","peer":"usb","medium":"usb3","client":"modeld"}
// state: "waiting" | "connected" | "disconnected". medium, when connected:
// "usb3" | "usb2" | "usb1" | "usb" (speed unknown) | "tcp", from the comma's
// hello (a phone's cable is TCP over USB). client: the hello's name, "modeld"
// while driving or "provision" fetching the model; absent before a hello.
// Transitions only.

{"event":"engine","t":0,"state":"building","sha256":"<sha256>","detail":"",
 "stage":"build","frac":0.42,"msg":"","load_only":false}
// state: "none" | "building" | "loading" | "ready" | "failed". Every state
// change, and progress at most 4 times a second.

{"event":"stats","t":0,"frames":20,"fps":19.9,"slow":0,"window_s":1.0,
 "total_ms":{"mean":17.0,"p99":17.4,"max":18.1},"served_ms":{"mean":17.4,"p99":17.9,"max":18.6},
 "gpu_ms":{"mean":15.7},"stages_ms":{"queue":0.5,"gpu":15.7,"other":0.8,"send":0.4}}
// Once a second while frames arrive. total_ms: arrival to reply ready;
// served_ms adds the send; stages_ms are means that add up to served_ms.mean.
// slow: frames whose total was over 60 ms.

{"event":"inventory","t":0,"loaded":"<sha256>|null","last_loaded":"<sha256>|null",
 "models":[{"sha256":"<sha256>","bytes":765953504,"path":"<cache>/models/<sha16>.onnx",
            "name":"Cinque Terre V3 Model","ref":"<ref>|null"}],
 "artifacts":[{"sha256":"<sha256>","key":"<sha16>.ort1.29.0.coreml-Apple_M1_Pro",
               "path":"<cache>/engines/<sha16>.ort1.29.0.coreml-Apple_M1_Pro.ortcache",
               "bytes":2300000000,"backend":"ort","runtime_version":"1.29.0","device":"coreml-Apple_M1_Pro",
               "built_at":"<ISO 8601>|null","build_seconds":8.2,"checkpoint":"<id>|null","current":true}],
 "disk":{"models_bytes":765953504,"engines_bytes":2300000000,"free_bytes":120000000000}}
// A Mac's. current: this server would load that artifact. After builds,
// loads, downloads, imports, forget, and the inventory command.

{"event":"catalog","t":0,"fetched_at":1757440000.0,"url":"<model index url>",
 "default_ref":"<ref>","error":"<why>|null",
 "models":[{"name":"Cinque Terre V3 Model","short_name":"CTV3M","ref":"<ref>",
            "build_time":"<ISO 8601>","index":13,"sha256":"<sha256>","bytes":765953504,
            "folder":"Master Models"}]}
// The model index's list, in its order: index counts down from the number
// listed, so sorting by it, highest first, keeps that order. sha256 and bytes
// are always set; a server from before the index sent null for a ref it had
// not resolved, and no folder. fetched_at is when the kept index was fetched.
// default_ref is the index's own default when it lists it, else Cinque
// Terre V3, which is also what a server with no index yet sends.
// error is set when a refresh failed or the index served could not be used,
// and this is the kept list.

{"event":"download","t":0,"sha256":"<sha256>","ref":"<ref>|null","state":"progress",
 "frac":0.42,"bytes":321000000,"total":765953504,"rate_bps":41000000.0,"detail":"","source":"<url>"}
// state: "started" | "progress" (at most 4 a second) | "done" | "failed" | "cancelled".

{"event":"import","t":0,"path":"<file>","state":"hashing","frac":0.3,"sha256":"<sha256>|null","detail":""}
// state: "hashing" | "copying" | "done" | "failed".

{"event":"benchmark","t":0,"state":"running","elapsed":12.0,"total":60.0,"frames":240,
 "frame":{...},"report":{...}|null,"detail":""}
// state: "running" | "done" | "cancelled" | "failed"; report when done.

{"event":"shutdown_request","t":0,"reason":"<why>"}
// The comma asked a phone or Mac to power off; the apps refuse and say so.

{"event":"hello","t":0,"version":"0.7.0"}
// The web page's first event: the server's version, as
// `jetlink-server --version` prints it.
```

## Commands

`{"cmd": "<name>", ...arguments}`, answered by one reply:
`{"ok": true, ...extras}` or `{"ok": false, "error": "<sentence>"}`.

| cmd | arguments | reply extras | behaviour |
| --- | --- | --- | --- |
| `status` | | | re-sends `server`, `link`, `engine`, `inventory`, `catalog` |
| `catalog` | `refresh` (default false) | `queued: true` | fetches the model index if asked, or if none is kept or the kept one is over an hour old, then emits `catalog` |
| `download` | `ref` or `sha256` | `sha256` | queues a download (one at a time, in order); `download` events follow. Refused for a model the model index does not know |
| `cancel_download` | `sha256` | | cancels it and removes the `.part` |
| `import` | `path` | `queued: true` | hashes and copies the file in; `import` events, then `inventory` |
| `prepare` | `sha256`, `frame_skip` (default 4), `ref` (the catalog row chosen) | `state`; `sha256` when downloading | builds if needed and loads it. With no model file or engine on disk it downloads first, sized from the model index, and replies `state: "downloading"`. It is a pick: the comma takes it at its next hello unless it picked another since |
| `unload` | | | releases the loaded engine |
| `forget` | `sha256`, `artifacts` (default true), `model` (default false) | | unloads it if loaded, deletes its engines and/or download, then `inventory` |
| `inventory` | | | emits `inventory` |
| `benchmark` | `seconds` (default 60, up to 3600) | `queued: true` | runs the loaded model at the comma's pace with no comma; `benchmark` events. Refused while a comma is connected |
| `cancel_benchmark` | | | stops it |
| `shutdown` | | | emits `server` with `stopping`; the app stops the server |
