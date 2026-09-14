#!/usr/bin/env bash
# QSA 경로에 fp8_e4m3 메인 KV 캐시를 허용합니다 (TROUBLESHOOTING §15-16).
#
# 출처: vLLM 이슈 #54426 의 첨부 패치 (gist Nanetnounou/2e6486dd033bf077a9b8fdfcb6e1f7c8).
#   작성자 환경이 우리와 동일합니다 — GB10 DGX Spark sm_121, vLLM 0.1.dev20073+g8e685d198,
#   같은 이미지. 그래서 17개 hunk 가 오프셋 -1~-7 로 그대로 붙습니다.
#   상류 PR 은 #55557 이며 아직 미병합이고, 그쪽은 모듈명이 qwen4_exp 로 바뀌어
#   우리 빌드에는 적용되지 않습니다(파일이 크게 재구성됨).
#
# 왜 필요한가: Qwen3.8-Flash-Next 의 풀어텐션 12개 층은 QSA(희소 어텐션)를 쓰는데
#   그 구현이 BF16 KV 를 하드 요구해서 --kv-cache-dtype 이 전부 막혀 있습니다(§15-15).
#   QSA 는 flash-attn 커널을 쓰지 않고 자체 Triton 커널을 쓰므로 부모 클래스의
#   양자화 거부는 실제 제약이 아닙니다. 패치는 그 거부를 우회하고 Triton 커널 안에서
#   _cast_kv_tile 로 역양자화합니다.
#
# 🔴 앵커가 어긋나면 일부러 non-zero 로 죽습니다. 조용히 BF16 으로 뜨면
#    벤치 결과를 잘못 해석하게 됩니다 (fix_ple_fp8_gate.py 와 같은 방침).
set -euo pipefail

SP=/usr/local/lib/python3.12/dist-packages
PATCH=/patches/qsa_fp8_kv_sm121.patch
MARK="$SP/vllm/models/qwen3_8_flash_next/nvidia/.qsa_fp8_kv.applied"

if [ -f "$MARK" ]; then
  echo "[apply_qsa_fp8_kv] 이미 적용됨 — 건너뜁니다"
  exit 0
fi

if [ ! -f "$PATCH" ]; then
  echo "[apply_qsa_fp8_kv] 🔴 $PATCH 없음 — ./patches 마운트를 확인하세요" >&2
  exit 1
fi

cd "$SP"
if ! patch -p1 --dry-run --silent < "$PATCH"; then
  echo "[apply_qsa_fp8_kv] 🔴 패치가 적용되지 않습니다. 이미지가 바뀌었는지 확인하세요." >&2
  echo "[apply_qsa_fp8_kv]    기대 이미지: vllm/vllm-openai@sha256:3b0e188f... (2026-08-26)" >&2
  exit 1
fi

patch -p1 --silent < "$PATCH"

# 🔴 검증을 마커보다 먼저 합니다. 마커를 먼저 쓰면, 검증이 실패해 컨테이너가 재시작될 때
#    두 번째 기동이 "이미 적용됨"으로 건너뛰어 실패가 가려집니다 (2026-09-09 실제로 겪음).
# 🔴 모듈을 단독 import 하면 안 됩니다 — qsa.py <-> model.py 가 순환 임포트라
#    vLLM 정상 경로 밖에서는 ImportError 가 납니다. 파일 내용으로 확인합니다.
QSA_PY="$SP/vllm/models/qwen3_8_flash_next/nvidia/qsa.py"
OPS_PY="$SP/vllm/models/qwen3_8_flash_next/nvidia/ops/qsa.py"
fail=0
grep -q '"fp8_e4m3",' "$QSA_PY"                      || { echo "[apply_qsa_fp8_kv] 🔴 백엔드 dtype 목록 미반영" >&2; fail=1; }
grep -q 'float8_e4m3fn' "$QSA_PY"                    || { echo "[apply_qsa_fp8_kv] 🔴 캐시 재해석 미반영" >&2; fail=1; }
grep -q '_cast_kv_tile' "$OPS_PY"                    || { echo "[apply_qsa_fp8_kv] 🔴 커널 역양자화 미반영" >&2; fail=1; }
grep -q 'block_n = max(16, block_n // 2)' "$OPS_PY"  || { echo "[apply_qsa_fp8_kv] 🔴 sm_121 공유메모리 완화 미반영" >&2; fail=1; }
if [ "$fail" -ne 0 ]; then
  echo "[apply_qsa_fp8_kv] 🔴 검증 실패 — 마커를 쓰지 않고 종료합니다" >&2
  exit 1
fi

touch "$MARK"
echo "[apply_qsa_fp8_kv] ✅ 적용 및 검증 완료 — QSA 가 fp8_e4m3 KV 를 받습니다"
