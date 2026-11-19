#!/usr/bin/env bash
# CRIU action script (--action-script) for criu-snapshot.sh.
#
# post-dump: CRIU runs this before its --leave-running rollback deletes the link_remap.* files
# it created for unlinked-but-mapped files (e.g. Python's multiprocessing semaphore in
# /dev/shm). Copy them under ${IMG_DIR}/remaps/ with their paths so restore can put them
# back; without them the images are not restorable (files-reg.c: "Can't link ... link_remap").
set -euo pipefail

[[ "${CRTOOLS_SCRIPT_ACTION:-}" == "post-dump" ]] || exit 0
[[ -n "${IMG_DIR:-}" ]] || exit 0

dest="${IMG_DIR}/remaps"
mkdir -p "${dest}"
shopt -s nullglob
for f in /dev/shm/link_remap.* /tmp/link_remap.* /root/link_remap.* /link_remap.*; do
  cp -a --parents "${f}" "${dest}/"
done
echo "saved link remaps: $(find "${dest}" -type f | wc -l)"
