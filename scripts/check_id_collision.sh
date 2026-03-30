#!/usr/bin/env bash
# scripts/check_id_collision.sh
# 사용법: ./scripts/check_id_collision.sh
# 종료코드: 0 = 충돌 없음, 1 = 충돌 발견

set -euo pipefail

SPEC_DIR="spec"
DUPLICATES=$(
  grep -roE '\[[A-Z]+-[A-Z]*-?[0-9]+[a-z]?\]' "$SPEC_DIR"/*.md \
  | sed 's/.*://' \
  | sort \
  | uniq -d
)

if [ -n "$DUPLICATES" ]; then
  echo "ERROR: 중복 ID 발견:"
  echo "$DUPLICATES"
  for id in $DUPLICATES; do
    echo "  $id 위치:"
    grep -rn "$id" "$SPEC_DIR"/*.md | head -5
  done
  exit 1
fi

echo "OK: ID 충돌 없음."
exit 0
