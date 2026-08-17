# GPU box

Development happens on the laptop; everything runs on a rented GPU. This file is the
provisioning contract and the record of what the environment actually turned out to be.

## What we're using

**RunPod, L4 24 GB, Community Cloud (~$0.44/hr)** for all development, with a short
**A100** rental at the very end for the M9 benchmark only.

The L4 is not a compromise for M0–M8 — every milestone before the final load test is about
correctness and per-request cost, and both transfer. Only the concurrency ceiling needs
the bigger card, and only for a few hours.

| GPU | VRAM | ~$/hr (community) | Used for |
|---|---|---|---|
| L4 | 24 GB | 0.40–0.50 | M0–M8: bring-up, export, backends, gateway, observability |
| A100 80 GB | 80 GB | 1.20–1.90 | M9 only |

## The constraint that shaped everything: RunPod pods are containers

A RunPod pod is not a VM. It is a Docker container on RunPod's hardware, running with the
default capability set. Verified on the first box:

```
/proc/1/cgroup   →  12:misc:/docker/9c54c01b7653...   ← inside a container
CapEff:             00000000a80425fb                  ← no CAP_SYS_ADMIN
/var/run/docker.sock → absent                         ← no Docker to talk to
```

Consequences, all of which the repo now accounts for:

- **No Docker, no docker-compose.** `docker/` is retained for VM deploys and as the
  dependency reference; the RunPod path is `scripts/provision.sh` + `scripts/services.sh`.
- **Triton must come from the pod image**, not from a `docker pull` — hence starting the
  pod *from* `nvcr.io/nvidia/tritonserver:24.08-py3`.
- **No DCGM exporter** (wants privileges a pod lacks). Triton's own `nv_gpu_*` metrics
  cover it.
- **RunPod's `ssh.runpod.io` proxy gives a shell but ignores remote commands, and does no
  port forwarding or rsync.** `scripts/rpod.sh` drives it over stdin. Exposing TCP 22
  gives a direct-SSH endpoint with full functionality; RunPod's HTTPS proxy handles the
  web UIs without any tunnel.

## Deploying the pod

1. **Deploy** → Community Cloud → **L4**
2. **Change Template** → Custom:
   - **Container Image:** `nvcr.io/nvidia/tritonserver:24.08-py3`
   - **Container Disk:** 60 GB (image is ~15 GB; TRT builds need scratch)
   - **Volume Disk:** 20 GB at `/workspace`
   - **Expose HTTP Ports:** `8000,3000,9090,16686`
   - **Expose TCP Ports:** `22`
3. NGC images do not start `sshd`. Paste into **Container Start Command**:

```bash
bash -c 'apt-get update -qq && apt-get install -y -qq openssh-server && mkdir -p /run/sshd ~/.ssh && ssh-keygen -A && echo "$PUBLIC_KEY" > ~/.ssh/authorized_keys && chmod 700 ~/.ssh && chmod 600 ~/.ssh/authorized_keys && /usr/sbin/sshd -D'
```

4. On the pod:

```bash
git clone <repo> /workspace/streaming-tts-serving
cd /workspace/streaming-tts-serving && bash scripts/provision.sh
```

## Reaching the UIs

RunPod proxies exposed HTTP ports over HTTPS — no SSH tunnel needed:

| URL | Service |
|---|---|
| `https://<POD_ID>-3000.proxy.runpod.net` | Grafana |
| `https://<POD_ID>-9090.proxy.runpod.net` | Prometheus |
| `https://<POD_ID>-16686.proxy.runpod.net` | Jaeger |
| `https://<POD_ID>-8000.proxy.runpod.net/v2/health/ready` | Triton health |

## Where things live

| Path | Persists across pods? | Contents |
|---|---|---|
| `/workspace` | **yes** (volume) | repo, venv, models, TRT engines, Prometheus data |
| `/` | no | the image; anything written here dies with the pod |

Put anything expensive to rebuild on `/workspace`. `provision.sh` already does.

## Cost discipline

**Terminate the pod when you stop working — do not merely stop it.** RunPod bills storage
on stopped pods, and provisioning is fully scripted, so a rebuild is ~10 minutes
unattended. Expected total for the whole project is roughly $10–15 if this is followed,
and several times that if it is not.

Running load tests **on the pod itself** is deliberate: it measures server-side
time-to-first-audio without WAN jitter in the number. The tradeoff is that it excludes
real network latency, which is noted wherever the results are reported.
