# Transport floor attribution: where the ~30ms addon `ping` cost lives

Measurement-only probe. No product code changed. Reproduce with
`python3 benchmarks/probe_transport_floor.py` (raw data in `transport_floor.json`).
All numbers are p50 unless noted; 100 samples after 5 warmup per experiment.

## Attribution table

| Layer | What it isolates | p50 (ms) | p90 | p99 |
| --- | --- | ---: | ---: | ---: |
| Python client + loopback wire | In-process pure-Python echo, same NDJSON client path (json encode/decode, sendall, recv-until-newline) | 0.07 | 0.08 | 0.11 |
| Wolfram SocketListen + handler stack | Headless `wolframscript` echo, same handler shape (`SourceSocket`+`DataBytes`), NO front end, NO dispatch | 1.93 | 2.68 | 4.13 |
| Live addon `ping` (persistent socket) | Full path inside the front-end-bonded kernel | 31.92 | 35.54 | 49.17 |
| Live addon `ping` (fresh socket/request) | Adds TCP connection setup | 33.96 | 36.86 | 50.83 |

Derived split of the ~32ms floor:

| Component | p50 (ms) | Share |
| --- | ---: | ---: |
| Python client + wire | 0.07 | 0.2% |
| WL socket stack over client+wire (1.93 - 0.07) | 1.86 | 5.8% |
| Front-end kernel residue (31.92 - 1.93) | **29.99** | **94%** |
| TCP connect setup (fresh - persistent, 33.96 - 31.92) | 2.04 | (per-connection, not in the persistent floor) |

The client and the wire are noise. Wolfram's own socket + async-handler
machinery, measured in a plain headless kernel, is only ~2ms. **~94% of the
floor is residue that appears only when the identical SocketListen mechanism
runs inside the kernel that is bonded to the Mathematica front end.** `ping`
does no notebook work, no `CurrentValue`, no dispatch of substance (`cmdPing`
is a 4-key constant), so the residue is not the addon's code - it is the
latency before the socket handler gets to run at all.

## Pipelining: the ~30ms is per-handler-invocation, not per-message

Send K=10 `ping`s back-to-back on one socket, then read all 10 (100 batches):

| Metric | p50 (ms) | p90 | p99 |
| --- | ---: | ---: | ---: |
| Per-request (batch_total / 10) | 5.91 | 7.91 | 11.16 |
| Batch total (10 requests) | 59.14 | 79.06 | 111.56 |

Ten sequential pings would cost ~320ms (10 x 32). Pipelined they cost ~59ms and
the per-request figure collapses from ~32ms to ~6ms. The 10 requests arrive in
one socket-read event, the addon's `handleConnection` splits the buffer and
`Map`s over all 10 in a single handler invocation, so the fixed ~28-30ms
scheduling wait is paid roughly **once per invocation**, not once per message.
This is the decisive evidence: the floor is the wait for the handler to be
scheduled on the front-end kernel, not the cost of processing a message.

## Payload sweep: base floor is fixed, growth beyond it is serialization

`execute_code` returning `StringRepeat["x", N]` (persistent socket, 50 iters):

| Response size | wire bytes | p50 (ms) |
| --- | ---: | ---: |
| 1 KB | 2,278 | 35.23 |
| 10 KB | 20,278 | 37.61 |
| 100 KB | 200,278 | 59.96 |

A ~35ms floor is present even at 1KB (about 3ms above bare `ping`, which is
`execute_code`'s own parse + evaluate + `ToString[InputForm]` + `AbsoluteTiming`
overhead). Going from ~2KB to ~200KB of response adds ~25ms, i.e. ~0.13ms/KB.
Loopback wire time for 200KB is well under 1ms, so that growth is **response
serialization in the kernel** (`ExportString[..,"RawJSON"]`, `ToString` of a
100KB string, `truncateLargeResult` scan, `BinaryWrite`), not transport. RTT
does scale with response size, but only above the fixed scheduling floor and
only through serialization, not the wire.

## Conclusion

**The binding constraint is front-end-kernel async-task scheduling latency, a
fixed ~28-30ms wait before the SocketListen handler runs.** It is not in the
Python client (0.07ms), not on the loopback wire, not in TCP setup (~2ms), not
in Wolfram's SocketListen mechanism itself (~2ms headless), and not in the
addon's dispatch or JSON code. The addon's socket server lives inside the
kernel that is bonded to the front end; that kernel spends its time servicing
the front-end MathLink loop and only services its async socket task between
service cycles, and that cadence is ~30ms (about 33Hz). Every single-command
round trip waits one cadence tick.

**Is a ~5ms floor plausible? Only by moving the socket listener off the
front-end-bonded kernel.** The headless kernel proves the same SocketListen +
handler mechanism answers in ~2ms, so ~5ms (2ms socket + a few ms dispatch) is
realistic for pure-kernel commands (`ping`, `execute_code`, math, variable
introspection) if they are served by a dedicated headless "gateway" kernel that
owns the socket, and the front end is driven only when a command actually
touches a notebook. It is **not** achievable while the listener stays in the
front-end kernel: nothing in the addon or the transport can be tuned away
because the ~30ms is the front end's kernel-service interval, not work.

### Recommendation

1. **Cheapest win, no re-architecture, available today: batch.** `batch_commands`
   already routes N sub-commands through one `handleConnection` invocation, so N
   commands cost ~one scheduling tick instead of N. The pipelining result
   (per-request ~6ms vs ~32ms) is the ceiling this unlocks. Any agent workflow
   that issues several addon calls in sequence should prefer one batch.
2. **Structural win, if single-command latency must drop to ~5ms: a gateway
   kernel.** Run the SocketListen server in a headless kernel (~2ms floor) that
   answers pure-kernel commands directly and forwards only notebook-touching
   commands to the front end. This is a significant change to the addon
   architecture and out of scope for this measurement.
3. **Do not chase micro-optimizations in the client, JSON, or dispatch path.**
   They sum to under 2ms and cannot move the floor.

### Caveats

- The Python echo runs in an in-process thread, so its ~0.07ms includes GIL
  contention; it is an upper bound on the client+wire floor, and it is already
  negligible.
- The ~30ms residue is inferred as front-end-kernel scheduling by elimination
  (headless SocketListen is ~2ms; the addon's `ping` does no other work). It was
  not confirmed by instrumenting the addon, which was out of scope (no product
  code changes).
- Machine was quiet during the run; absolute numbers will drift on a loaded host,
  but the layer ratios (client << WL socket << front-end residue) are the point.
