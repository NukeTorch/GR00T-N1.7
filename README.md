# NVIDIA GR00T N1.7 on NukeTorch

NVIDIA's GR00T N1.7 robot policy (fine-tuned on LIBERO-10) served on one Mac with NukeTorch 1.0.5, beside a text model. One request is one observation (two camera frames, the robot's state and an instruction); the answer is one chunk of actions. Requests that are waiting when the model is free are answered together, up to 16 in one model call.

**[Video demo](https://youtu.be/A4xSc33-w7s)** · **[Detailed benchmark report](https://nuketorch.ai/reports/groot-four-m5-max.pdf)**

## Benchmark summary

One MacBook Pro M5 Max with 48 GB, a text model loaded beside the policy. Measured beside a text model that is not the one in the commands below; speed beside the 35B was not measured. The built-in benchmark sends one observation it generates itself (two 256×256 camera frames of noise, a state vector and an instruction), so it needs no data set. Best of five runs.

| Test | M5 Max |
|---|---:|
| One request at a time (median of 256) | 44.8 ms |
| 16 clients at once: chunks of actions a second | 37.2 |
| Response time under that load (median) | 428 ms |

On an earlier build we also replayed the recorded observations of our four-Mac report (LIBERO-10 episode 0, two cameras) through this server on one of the report's Macs, with our earlier test scripts. Those scripts are not in the package, so these rows cannot be repeated with the built-in benchmark:

| Test | The report, per Mac | One Mac, this server |
|---|---:|---:|
| 48 clients at once, 2 cameras: chunks a second (completions in a 10-second window) | 38.4 | 38.4 |
| Response time under that load: half of the requests within | 1,275 ms | 1,280 ms |
| One request at a time | 44.3 ms | 45.0 ms |

- **Same answers.** For the same observation and the same seed the server returns the same actions, bit for bit, run after run. A request answered alone and the same request answered inside a batch differ in the last bits; the benchmark knows both answers and its report says how many were answered alone.
- **Memory.** GR00T adds about 5.5 GB beside a text model (the report's 8.04 GiB is the worker process). Loading it alone is not in this release.
- No robot is moved and no simulator runs. The numbers say how fast the policy answers, not how well a robot does its task.
- Let the Mac cool before you measure: in our runs a batch of 16 requests took 406 ms on a rested Mac and 480 ms thirty seconds into a run on a warm one.

**Paper:** [GR00T N1.7 Concurrent Policies on Four M5 Max Macs](https://nuketorch.ai/reports/groot-four-m5-max.pdf) (our four-Mac report: 152 policy chunks a second across four MacBook Pros; this package runs on one Mac and lets you check the per-Mac part).

## Multi Machines: NukeTorch Enterprise

Our pipeline parallelism spreads one model across many machines connected by a single cable in between. We used it for Qwen3.5-9B LoRA fine-tuning and inference of a larger model; the fine-tuning reached 93.0% pipeline efficiency against four single machines.

| Benchmark | Performance | Compared with |
|---|---:|---|
| **Robotics:** Nubotics 1.7B, one MacBook Pro M5 Max, one robot at a time | **53.2** chunks/s (18.81 ms per call) | 1.22× NVIDIA's own H100 figure for GR00T N1.7 on LIBERO-10 (43.7 chunks/s, 22.9 ms, one request at a time) |
| **Robotics:** GR00T N1.7, one MacBook Pro M5 Max, one request at a time | **22.6** chunks/s (44.34 ms per call) | 0.63× NVIDIA's own H100 figure (35.9 chunks/s)² |
| **Robotics:** GR00T N1.7, four M5 Max, 192 concurrent policies | **152.07** chunks/s | 2.45× the fastest published H100 result (62.2 chunks/s), which is for one request at a time¹ |

¹ The H100 figures are published results timed one request at a time; no H100 rate with many requests at once is published. Ours: 48 clients per Mac, FP16; a longer run measured 136.83 chunks/s.
² Ours: one request at a time; the test pooled 200 timed calls across 10 fresh processes, with 100 warm-up calls excluded; camera preparation and action decoding are included. NVIDIA's: its TensorRT pipeline on an H100 80 GB, per its hardware recommendation guide.

NukeTorch Enterprise P1.0.0 (Multi Machines) is by invitation. Ask for an evaluation license:

[![Request a NukeTorch Enterprise evaluation license](https://img.shields.io/badge/NukeTorch%20Enterprise-Request%20an%20evaluation%20license-121F47?style=for-the-badge)](mailto:contact@nuketorch.ai?subject=NukeTorch%20Enterprise%20evaluation%20request&body=Name%3A%0AOrganization%3A%0AMacs%20%28how%20many%2C%20which%20chips%2C%20memory%29%3A%0AModel%20or%20workload%3A%0A)

## Install

Use `curl`, not a browser. This release is not yet notarized by Apple, and macOS blocks un-notarized programs downloaded with a browser.

```sh
curl -fsSL https://github.com/NukeTorch/nuketorch/releases/download/1.0.5/install.sh | sh
```

The script downloads the package, checks its SHA-256, shows the license and asks you to accept it, then unpacks it into `~/NukeTorch`. Nothing is installed outside that folder. To remove NukeTorch, delete the folder.

Requirements: a Mac with an M5 Max or M4 Max chip and macOS 27.2 or later. macOS 26 is not supported. Other chips: coming. We tested on 48 GB.

## Get the models

The policy bundle holds its makers' files unchanged: NVIDIA's checkpoint from the report (`nvidia/gr00t17-lerobot-libero_10-640`), NVIDIA's base model configuration, and the tokenizer files from Qwen that GR00T's backbone uses. The policy runs beside a text model, so download one too.

The first line gives you the `hf` command (or: `brew install hf`).

```sh
python3 -m venv ~/NukeTorch/hf-env && ~/NukeTorch/hf-env/bin/pip install -U huggingface_hub
~/NukeTorch/hf-env/bin/hf download NukeTorch/GR00T-N1.7-LIBERO-10 --local-dir ~/NukeTorch/models/GR00T-N1.7-LIBERO-10
~/NukeTorch/hf-env/bin/hf download NukeTorch/Qwen3.6-35B-A3B-4bit-DFlash2 --local-dir ~/NukeTorch/models/Qwen3.6-35B-A3B-4bit-DFlash2
```

Model card: https://huggingface.co/NukeTorch/GR00T-N1.7-LIBERO-10

## Run

```sh
cd ~/NukeTorch/nuketorch-1.0.5-macos-arm64
./nuketorch-serve http \
  --model ~/NukeTorch/models/Qwen3.6-35B-A3B-4bit-DFlash2/Qwen3.6-35B-A3B-4bit \
  --draft-model ~/NukeTorch/models/Qwen3.6-35B-A3B-4bit-DFlash2/Qwen3.6-35B-A3B-DFlash2 \
  --policy-model ~/NukeTorch/models/GR00T-N1.7-LIBERO-10 \
  --port 8080
```

The route is `POST /v1/nuketorch/policy/act`. It takes observations already prepared for the model; a route that takes raw camera frames comes in an update.

## Repeat our numbers

```sh
./nuketorch-serve bench ~/NukeTorch/models/GR00T-N1.7-LIBERO-10 \
  --text-model ~/NukeTorch/models/Qwen3.6-35B-A3B-4bit-DFlash2
./nuketorch-serve report
```

`report` sets your run beside ours. Other commands: `bench --list` (what each test does), `bench --inputs` (every stored input with its license and source), `selftest` (the robot-policy kernels are listed as not covered in this release).

You are responsible for the safe use of this model with a real robot.

## License

NukeTorch is free for personal use, schools and universities, and businesses and other organizations (not government entities or international organizations) that with their affiliates have less than US$10 million in revenue and fewer than 250 people, on up to 250 machines, each working alone. Everyone else may evaluate it for 30 days, then needs a commercial license: contact@nuketorch.ai. You may not offer its function itself to others: access to its API, or running, serving or fine-tuning models for them. Terms are in [LICENSE.md](https://github.com/NukeTorch/nuketorch/blob/1.0.5/LICENSE.md).

GR00T N1.7 is licensed by NVIDIA Corporation under the NVIDIA Open Model License. Built on NVIDIA Cosmos.

**Powered by NukeTorch** · https://nuketorch.ai · Back to [NukeTorch/nuketorch](https://github.com/NukeTorch/nuketorch)
