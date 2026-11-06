# Host stall diagnostics

This opt-in diagnostic records long-running **host call boundaries**, not the GPU's current
shader, a driver-hang cause, or a fix for Mario Kart World (#6). It is disabled by default.
Use a build containing this change and set `EDEN_NX2_STALL_DIAGNOSTICS=1` when launching it.
Do not manually clear the shader cache or change split/game settings for the diagnostic run.
The flag does not change cache compatibility. A different binary baseline can invalidate a cache
independently: preserve the original files and check the loaded pipeline count when interpreting
a cached-startup result.

At the first Vulkan pipeline-cache construction in the process it starts one 180-second window.
The normal log receives a diagnostic-origin UTC/Unix/steady timestamp and host PID/native TID.
This timestamp is **not process creation time**. Correlate future kernel messages using both
time (including timezone) and PID/TID; an older reset message cannot be assigned to a run from
this new logging alone.

The sampler checks only fixed host records once per second. It reports phases active for at
least one second, repeats an operation at most every five seconds, and stops after 256 snapshot
rows or 180 seconds. It has 32 concurrent thread slots. Overflow, expiry and saturation are
explicit; subsequent or unrecorded phases are unknown. It does not restart when another game
is launched in the same process. Fast operations need not appear.

The reported phases distinguish native instruction capture and primary/fallback lowering,
pipeline preparation, the actual Vulkan graphics/compute pipeline-create calls, pipeline-build
waits inside scheduler callbacks, queue drain versus execution-lock acquisition, submission-lock
acquisition and the actual queue-submit calls, and existing counter/semaphore/fence waits.
Role describes the call path, not an inferred OS thread name. A compute build/wait key is the
compute shader key; graphics build/wait uses the graphics pipeline key. A native capture/lower
address is the known program base. `key_or_0`/`address_or_0` zero means not recorded;
`stage_or_minus1=-1` means unknown. No stage is inferred for a whole graphics pipeline.

`submitted_at_entry` is the last successful submit tracked for that semaphore while diagnostics
were active. `cached_completed_at_entry` is an existing cached GPU tick. Both are entry snapshots,
may be stale, and zero can mean no recorded progress. They are not extra driver queries. A submit
return does not mean GPU execution completed. A long pipeline-build wait with a matching worker
inside pipeline creation attributes the host dependency, not shader correctness or a driver bug.

Scopes restore the preceding phase on normal or exceptional exit. There are no success claims
based merely on scope destruction. The sampler copies records before logging, never takes GPU
or cache locks, adds no guest reads/Vulkan polling, and does not alter the existing waits. It
flushes only emitted bounded diagnostics through the existing logger to preserve buffered output.
Logging and host bookkeeping can change timing; this is not a timing-neutral measurement.

The sampler joins on normal process teardown. Its one fixed recorder intentionally has process
lifetime so a driver thread returning later cannot release a scope into freed storage. No sampler
or recorder is created with the flag unset (or set to anything other than exactly `1`).
