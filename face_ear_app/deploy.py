"""Deploy face_ear_app to target device via paramiko (no sshpass required)."""
from __future__ import annotations

import sys
from pathlib import Path

import paramiko

DEVICE     = "172.30.101.184"
USER       = "root"
PASS       = "oelinux123"
REMOTE_DIR = "/data/local/tmp/face_ear_app"

HERE        = Path(__file__).resolve().parent
FACEMAP_APP = HERE.parent / "facemap_app"
DLC_DIR     = HERE.parent / "facemap_3dmm-qnn_dlc-w8a8"

# Files from this directory.
OWN_FILES = [
    HERE / "ear_demo.py",
    HERE / "run_demo.sh",
]

# Runtime + shim sources from facemap_app (shared, no duplication needed).
FACEMAP_FILES = [
    FACEMAP_APP / "snpe_runtime.py",
    FACEMAP_APP / "snpe_shim.cpp",
    FACEMAP_APP / "snpe_shim.h",
    FACEMAP_APP / "build.sh",
    FACEMAP_APP / "VarSetup",
    FACEMAP_APP / "meanFace.npy",
    FACEMAP_APP / "shapeBasis.npy",
    FACEMAP_APP / "blendShape.npy",
    FACEMAP_APP / "face_det_w8a8.dlc",
    FACEMAP_APP / "face_det_w8a8_prepared.dlc",
]

# 3DMM model DLCs.
DLC_FILES = [
    DLC_DIR / "facemap_3dmm.dlc",
    DLC_DIR / "facemap_3dmm_prepared.dlc",
]


def _ssh() -> paramiko.SSHClient:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(DEVICE, username=USER, password=PASS, timeout=15)
    return client


def _run(client: paramiko.SSHClient, cmd: str) -> str:
    _, stdout, stderr = client.exec_command(cmd)
    out = stdout.read().decode()
    err = stderr.read().decode()
    rc  = stdout.channel.recv_exit_status()
    if rc != 0:
        raise RuntimeError(f"remote command failed (rc={rc}): {cmd}\n{err.strip()}")
    return out


def main() -> None:
    # Verify all required files exist before connecting.
    all_files = OWN_FILES + FACEMAP_FILES + DLC_FILES
    missing = [f for f in all_files if not f.exists()]
    if missing:
        for f in missing:
            print(f"ERROR: missing {f}")
        sys.exit(1)

    print(f"connecting to {USER}@{DEVICE} …")
    client = _ssh()

    print(f"preparing {REMOTE_DIR} …")
    _run(client, "mount -o remount,exec /data 2>/dev/null || true")
    _run(client, f"mkdir -p {REMOTE_DIR}")

    print(f"transferring {len(all_files)} file(s) …")
    sftp = client.open_sftp()
    for local in all_files:
        print(f"  {local.name}")
        sftp.put(str(local), f"{REMOTE_DIR}/{local.name}")
    sftp.close()

    print("setting permissions …")
    _run(client, f"chmod +x {REMOTE_DIR}/run_demo.sh {REMOTE_DIR}/build.sh")

    print("building libsnpe_shim.so on device …")
    out = _run(client,
               f"cd {REMOTE_DIR} && "
               f"QNN_INCLUDE=/data/local/tmp/snpeexample/include/SNPE sh build.sh")
    print(out.strip() or "  done")

    client.close()
    print("\ndone.  on device run:")
    print(f"  ssh {USER}@{DEVICE}")
    print(f"  {REMOTE_DIR}/run_demo.sh --video-index 0")
    print(f"  {REMOTE_DIR}/run_demo.sh --image /data/local/tmp/mediapipe_hand/zidane.jpg")


if __name__ == "__main__":
    main()
