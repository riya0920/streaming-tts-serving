# docker/ — for VM deploys only

These files assume a machine where **you can run Docker**: a real VM (Lambda Labs, an
EC2/GCE GPU instance, or your own box). There, `docker compose up` brings the whole
stack — Triton, OTel collector, Jaeger, Prometheus, Grafana, DCGM exporter — up together.

```bash
docker compose -f docker/docker-compose.yml up -d
```

**They do not work on RunPod.** A RunPod pod is itself a Docker container running with
the default capability set — no `CAP_SYS_ADMIN`, no `/var/run/docker.sock` — so it
cannot start containers of its own:

```
/proc/1/cgroup  →  12:misc:/docker/9c54c01b7653...
CapEff:            00000000a80425fb
```

For RunPod, the pod is started **from** `nvcr.io/nvidia/tritonserver:24.08-py3`
directly, so the pod *is* the Triton container, and the observability services install
as static binaries. That path is `scripts/provision.sh` + `scripts/services.sh`.

Both paths serve the same architecture and the same `model_repo/`. The only difference
is packaging: compose composes containers, `services.sh` supervises processes. Keeping
both means the project is not welded to one provider — and the Dockerfile is still the
reference for exactly which system and Python dependencies the serving environment needs.

## Differences to remember

| | VM + compose | RunPod pod |
|---|---|---|
| Service addressing | compose DNS (`triton:8001`) | `localhost` |
| GPU metrics | DCGM exporter | Triton's built-in `nv_gpu_*` |
| Jaeger OTLP port | 4317 (own network namespace) | 5317 (shares the pod with the collector) |
| Reaching the UIs | SSH tunnel | RunPod HTTPS proxy |
