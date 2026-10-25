# Layercake — 3-minute judges video

**Format:** 6 slides, 180 seconds total. The PowerPoint advances automatically; record the narration below over the slides. Timings are 30 / 28 / 32 / 32 / 30 / 28 seconds. Slide 4 uses a screenshot from the captured live run.

## 0:00–0:30 — One model. Three devices.
What if a group of devices could combine their memory and compute to run one language model? We’re building Layercake: software that breaks a model into layer ranges, then runs those ranges across a peer-to-peer mesh. For our HackOHIO proof, an M2 Pro, an M4 iPad, and an RTX 5060 Ti worked on one model. The idea is to extend usable capacity by combining devices, while measuring the speed tradeoff of passing work between them.

## 0:30–0:58 — The bottleneck is memory
A large model can demand more memory than a team wants to dedicate to one device. Layercake assigns contiguous model layers to different peers, so each one loads only its slice. The layers together still form one model: each stage receives the hidden state from the previous stage and passes its result onward. That gives us a practical way to explore pooled capacity without asking every device to store the full model.

## 0:58–1:30 — Prompt in. Activations across.
Here’s the user flow. The user opens the browser chat on the coordinator and enters a prompt. The coordinator runs its layers, then forwards the intermediate activations to the next peer. The iPad computes its assigned range and forwards onward to the GPU. The final stage picks the next token, which returns to the coordinator. This repeats token by token until the response is complete. The dashboard makes the stages and live execution visible, so we can debug what each device actually did.

## 1:30–2:02 — We ran it on real hardware
This is a captured execution view, not a mockup. In this run, a fixed split of Qwen3.8-27B Q5_K_M covered all 64 decoder layers: layers 0 through 27 on the M2 Pro, 28 through 31 on the M4 iPad, and 32 through 63 on the RTX 5060 Ti. The run logged one prefill and 42 decode steps. It is a feasibility demonstration on three physical platforms; the split was configured in advance, not selected automatically.

## 2:02–2:32 — The engineering is in the handoff
The interesting work is coordinating very different machines and runtimes. Our shared core is written in Rust; the stage workers use native Metal and CUDA execution. Iroh provides peer discovery and encrypted QUIC connections, with relay paths when direct connections are unavailable. We record stage timings and tensor transfer sizes to make the handoffs measurable. The run's recorded decode rate was 3.6030 decode steps per second. That is a run-specific tradeoff, not the main promise; our next challenge is improving end-to-end speed and automating placement.

## 2:32–3:00 — A new way to think about compute
Layercake is an early but working proof that useful model capacity can come from a mesh of devices, not just one expensive box. The same idea could let a lab, classroom, or small team combine hardware they already have. We’ve built the distributed path and put it under real hardware. Next we want smarter layer placement, faster transfers, and robust recovery when a peer disappears. We’re Layercake: make the mesh the machine.

## Accuracy note for presenters
The recorded run uses a fixed validation split. Automatic placement is not enabled for that run. The source records 43 phase entries: one prefill and 42 decode steps. The 3.6030 decode-steps/second value comes from the artifact's `decode_steps_per_second` field in `experiments/model-validation/results/qwen38-p2p-ring-2026-10-24/live-after-rejection.json`; it is run-specific, not a general performance benchmark. Layercake is early development; mobile native inference remains in progress.
