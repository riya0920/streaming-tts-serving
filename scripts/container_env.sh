#!/usr/bin/env bash
# Import the container image's environment into the current shell.
#
# A Docker image's ENV belongs to PID 1. Interactive logins get it because the image's
# bashrc/profile is set up for it, but a shell spawned by sshd does NOT — it starts from
# sshd's default environment. So over SSH, /opt/tritonserver/bin is missing from PATH,
# NVIDIA_TRITON_SERVER_VERSION is unset, and LD_LIBRARY_PATH lacks the CUDA and TensorRT
# directories, which makes tritonserver and trtexec appear not to exist.
#
# Reading /proc/1/environ recovers them. Source this from anything that runs over SSH.
#
#   source scripts/container_env.sh

if [ -r /proc/1/environ ]; then
  _pid1_path=""
  while IFS= read -r -d '' _kv; do
    case "$_kv" in
      PATH=*)
        _pid1_path="${_kv#PATH=}"
        ;;
      LD_LIBRARY_PATH=*)
        # Prepend rather than replace: keep anything the caller already set.
        export LD_LIBRARY_PATH="${_kv#LD_LIBRARY_PATH=}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        ;;
      NVIDIA_*|TRITON_*|CUDA_*|PYTHONPATH=*)
        export "${_kv}"
        ;;
    esac
  done < /proc/1/environ

  # Union the two PATHs, image first, dropping duplicates.
  if [ -n "$_pid1_path" ]; then
    _merged="$_pid1_path"
    _IFS_SAVE="$IFS"; IFS=:
    for _d in $PATH; do
      case ":$_merged:" in *":$_d:"*) ;; *) _merged="$_merged:$_d" ;; esac
    done
    IFS="$_IFS_SAVE"
    export PATH="$_merged"
  fi
  unset _kv _pid1_path _merged _d _IFS_SAVE
fi

# trtexec lives outside the image PATH in the Triton images.
[ -d /usr/src/tensorrt/bin ] && export PATH="$PATH:/usr/src/tensorrt/bin"
