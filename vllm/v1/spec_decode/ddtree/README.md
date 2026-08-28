# DDTree for vLLM — 작업 트리 (PR 준비용)

DDTree (Diffusion Draft Tree, arXiv:2604.12989) 를 vLLM 에 구현한 것.
블록 디퓨전 드래프터의 위치별 분포로 best-first 트리를 만들고, 조상만 보이는
어텐션 마스크로 한 번의 forward 에 여러 후보를 동시에 검증한다.

참조 구현: https://github.com/liranringel/ddtree (MIT)

## 베이스

    vLLM 0.26.1rc1.dev1177+ga9a17e709   (도커 이미지 vllm/vllm-openai:nightly)

`patches/*.patch` 는 **이 커밋의 파일에 대한 차분**이다. 다른 리비전에 올릴 때는
재생성해야 한다. 원본은 이미지에서 뽑았다:

    cid=$(docker create vllm/vllm-openai:nightly)
    docker cp "$cid:/usr/local/lib/python3.12/dist-packages/vllm/<경로>" .

## 구성

    patches/    기존 vLLM 파일 8개에 대한 차분 (합계 ~1000줄)
    new/        새로 추가하는 파일
      vllm/v1/spec_decode/ddtree/   트리 빌드·런타임·KV 압축·GDN 트리 헬퍼
      csrc/ddtree/                  GDN 트리 CUDA 커널 + JIT 로더
    tests/      검증 스크립트

### 패치별 요지

| 패치 | 파일 | 내용 |
|---|---|---|
| 0001 | `v1/attention/backends/flashinfer.py` | prefill `plan()` 에 `custom_mask` 전달, 마스크 공급자 등록 |
| 0002 | `v1/worker/gpu_model_runner.py` | 훅 6곳 — 초기화·스텝 시작·RoPE 치환·수용/압축·propose 가로채기 |
| 0003 | `v1/spec_decode/dflash.py` | `drafter_k` 를 체크포인트 `block_size-1` 에서 읽음 |
| 0004 | `v1/spec_decode/llm_base_proposer.py` | 드래프터 logits 를 트리 빌드용으로 보관 |
| 0005 | `model_executor/models/qwen3_dflash2.py` | conv `block_size` 를 체크포인트에서 읽음 |
| 0006 | `.../mamba/gdn/qwen_gdn_linear_attn.py` | GDN 트리 경로 연결, 8토큰 상한 해제 |
| 0007 | `.../ops/fused_sigmoid_gating.py` | Triton SSM 에 `tree_parent_indices` |
| 0008 | `.../mamba/ops/causal_conv1d.py` | conv **update** 커널이 조상 열에서 윈도를 읽게 수정 |

## 현재 상태

동작 확인:

  - T=0 (greedy) 무손실 — 폭 32 이하, 하이브리드(Qwen3.8-27B AWQ / Qwen3.5-4B)와
    순수 어텐션(Qwen3-0.6B) 양쪽
  - 순수 어텐션 + ngram 드래프터에서 **순정 vLLM 대비 22% 단축** (5.13s → 4.20s,
    base 대비 1.63배)
  - 트리 빌드가 참조 구현과 동일 (`tests/t4_tree_build.py`, 12/12)
  - 분기 노드의 로짓이 그 조상 경로를 단독 실행한 결과와 일치
    (`tests/t20_node_context.py`, 48/48)
  - GDN 트리 CUDA 커널: 임의 슬롯·accepted 1~33·T=96·분기 깊이 11 에서 오차 0
    (`tests/t18_branch.py`)

## 🔴 업스트림 전에 반드시 해결할 것

1. **정확성 기준을 모델 종류별로 나눠야 한다.** 아래 §정확성 기준 참조.
   커널 자체는 fp64 로 검증했고 결함이 없다 — 고칠 코드가 아니라 바꿀 기준이다.
2. **디버그 발판 제거.** `VLLM_DDTREE_GDN_CHECK`(CUDA vs Triton 상태 교차검증),
   `VLLM_DDTREE_TRACE`, `lp_top`/`dyn_mode`/`e_chain` 필드, 각종 통계 히스토그램.
   측정용이라 프로덕션 경로에 있으면 안 된다.
3. **모듈 배치와 import.** 새 모듈들이 서로를 평면 이름으로 import 한다
   (`from ddtree_tree import ...`). 패키지 상대 import 로 바꿔야 한다.
   `gpu_model_runner` 패치와 `qwen_gdn_linear_attn` 패치가 `/work`, `/work/cuda`
   를 `sys.path` 에 넣는 부분도 제거 대상이다.
4. **CUDA 커널 빌드.** 지금은 `torch.utils.cpp_extension.load` 로 JIT 컴파일하고
   sm_86 gencode 가 하드코딩돼 있다. vLLM 빌드 시스템에 편입해야 한다.
5. **V2 모델 러너 미지원.** 훅이 전부 V1 기준이라 `VLLM_USE_V2_MODEL_RUNNER=0`
   가 필요하다. DFlash 드래프터 중 SWA/full 혼합인 것들은 V2 전용이라 현재는
   등가 변환(§17) 없이는 못 쓴다.
6. **다중 요청에서 짧은 드래프트 미지원.** 확신 시 드래프트를 짧게 내는 경로는
   요청이 하나일 때만 켜진다. 한 스텝의 드래프트 폭을 텐서 하나로 공유하기 때문.
7. **T>0 미지원.** 트리 거부 샘플링이 없다. 현재는 greedy 전용.

## 알아둘 사실 (측정으로 확인)

  - 트리가 이기는 조건은 **드래프터가 불확실할 때**다. 확신하는 드래프터
    (DFlash2) 에서는 같은 예산의 사슬이 항상 낫다. 순수 어텐션+ngram 에서도
    반복 텍스트(확신)는 사슬이, 비반복 텍스트(불확실)는 트리가 이긴다.
  - best-first 의 가중치는 위치별 확률의 **곱**이라 독립을 전제한다. 실측에서
    DFlash2 는 자기 정확도를 과소평가하고 오차가 깊이에 따라 커져(깊이1 1.04배 →
    깊이14 4.29배) 깊이 10 도달 확률을 108배 과소평가한다. 그래서 예산이 얕은
    형제로 샌다. `depth_bonus`(가중치에 깊이당 상수를 **더한다**) 로 완화할 수
    있다. 곱하면 단조 변환이라 무연산이다.

자세한 실험 기록: `../docs/DDTREE-SCOPING.md` §15~17


## 정확성 기준 — 하이브리드에서는 base 비트 일치가 성립하지 않는다

트리 폭을 넓히면 하이브리드 모델(GDN)에서 간헐적으로 base greedy 와 다른 토큰이
나온다. 폭 40 이상에서 처음 관측됐지만 **원인은 폭이 아니라 수용 경로의 깊이**다
(수용이 깊이 8까지 내려간 스텝에서만 발생). 추적 끝에 커널 버그가 아님을 확인했다.

### fp64 참조로 확인한 것 (`tests/ddtree/t23_fp64ref.py`, `t26_tri_vs_fp64.py`)

실제 실행에서 덤프한 입력(부모 배열·슬롯·초기 상태 포함, 깊이 8 스텝 포함)으로
커널 수식을 fp64 로 다시 구현해 대조했다.

    CUDA   상태 vs fp64            0.000000
    Triton 상태 vs fp64            0.000000
    CUDA   출력 vs fp64            0.003560
    Triton 출력(fp32 유지) vs fp64  0.000803
    Triton→bf16 저장 vs fp64       0.003560   ← CUDA 와 동일
    fp64 를 bf16 으로 저장만        0.003560   ← 저장 자체의 하한

**재귀 상태는 양쪽 다 fp64 와 정확히 일치**하고, **출력 오차는 bf16 출력 텐서의
이론적 하한과 정확히 같다**. 커널은 더 정확해질 수 없다.

🔴 Triton 이 0.000803 으로 좋아 보이는 건 그 경로만 fp32 로 남겨 비교했기
   때문이다. 저장 조건을 맞추면 두 구현은 동일한 하한에 있다.

### 그래서 무엇이 다른가

두 구현은 **bf16 1 ULP 만큼** 다르다 (0.003559 — 위 하한과 같은 값). GDN 은 재귀
계층이라 상태가 누적되고 재정규화가 없어서, 24개 GDN 계층 × 수십 스텝에 걸쳐 이
차이가 쌓이면 깊은 트리에서 결정이 뒤집힌다. 논리 오류가 아니라 **산술 순서 차이의
증폭**이다.

순정 vLLM flat DFlash 조차 base 와 갈린다 (토큰 114, 로짓 격차 0.000 = 완전 동점).

### 기준

  - **순수 어텐션 모델**: base greedy 와 **비트 일치**가 성립한다. KV 가 토큰별로
    독립이라 트리 마스크만 맞으면 된다. 실측으로 확인됨 (Qwen3-0.6B).
  - **하이브리드(GDN) 모델**: base 비트 일치는 성립하지 않는다. **순정 flat 스펙
    디코딩을 기준선**으로 삼거나 허용오차 비교를 써야 한다.
  - 비트 일치가 꼭 필요하면 `VLLM_GDN_DECODE_KERNEL=triton` 으로 융합 CUDA 커널을
    끈다 (실측 무손실, 대신 느리다).

### 이탈을 판정하는 법 (`tests/ddtree/t19_tiegap.py`)

base 의 위치별 top-1/top-2 로짓 격차를 재서 분류한다. 격차가 분포 하위 5% 미만이면
수치적 동점이고, 중앙값 근처면 진짜 이탈이다.

🔴 "출력이 자연스러우니 수치겠지" 도, "🔴 표시가 떴으니 버그" 도 근거가 안 된다.
   이 프로젝트에서 두 방향 모두로 오판한 적이 있다. 격차를 재기 전에는 판정하지 않는다.
