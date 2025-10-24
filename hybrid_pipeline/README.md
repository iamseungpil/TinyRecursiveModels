# Hybrid LLaMA + TRM Pipeline for ARC-AGI

**완전히 재구성된 hybrid pipeline - gpt_integration 의존성 제거**

## 📁 구조

```
hybrid_pipeline/
├── gpt_oss_port/              # GPT-OSS 로직 모듈화 (LLM + planner + verifier)
│   ├── llm.py                 # TextReasoningModule (LLaMA wrapper)
│   ├── planner.py             # ARCPlanner (multi-attempt reasoning)
│   ├── verifier.py            # GridVerifier (validation + feedback)
│   ├── grid_utils.py          # Grid formatting utilities
│   ├── dataset_access.py      # Wrapper for existing dataset/ module
│   ├── run_baseline.py        # CLI baseline runner (LLM-only)
│   └── tests/
│       ├── test_planner.py    # Planner unit tests
│       └── test_grid_utils.py # Grid utils tests
│
├── adapters/                  # LLM↔TRM interface
│   ├── text_to_latent.py      # TextToLatentAdapter (LLaMA → TRM)
│   ├── latent_to_text.py      # LatentToTextAdapter (TRM → LLaMA)
│   ├── feedback_formatter.py  # Grid → feedback text
│   └── tests/
│       └── test_projection.py # Adapter tests
│
├── trm_pretrain/              # TRM pretraining (single GPU)
│   ├── train_trm.py           # Simplified pretrain script
│   └── eval_trm.py            # TRM evaluation
│
├── experiments/               # Joint training orchestration
│   ├── run_joint_training.py  # Main joint training (LLaMA + TRM)
│   ├── config_joint.yaml      # Configuration file
│   ├── run_trm_pretrain.sh    # TRM pretrain script
│   ├── run_trm_eval.sh        # TRM eval script
│   ├── run_joint_training.sh  # Joint training script
│   └── run_baseline.sh        # Baseline script
│
└── docs/
    └── README.md              # This file
```

## 🔑 핵심 설계 원칙

### ✅ 코드 중복 제거
- 모든 TRM/dataset 코드는 기존 `/home/ubuntu/TinyRecursiveModels/models/` 및 `dataset/`에서 import
- **gpt_integration 의존성 완전 제거**

### ✅ 모듈화
- `gpt_oss_port`: LLM 추론 + planning 로직
- `adapters`: LLM↔TRM 인터페이스
- `trm_pretrain`: TRM 사전학습
- `experiments`: 전체 오케스트레이션

### ✅ dataclass 기반 설정
- `JointModelConfig`: dataclass로 정의
- `asdict()` 사용해 직렬화

### ✅ 그래디언트 흐름
- `.detach()` 없음
- Adapter + TRM 학습 가능
- LLaMA 동결 (선택적)

## 🚀 빠른 시작

### 1. 데이터 준비

```bash
cd /home/ubuntu/TinyRecursiveModels/dataset
python build_arc_dataset.py \
    --input_file_prefix /path/to/arc \
    --output_dir /data/arc/processed \
    --subsets training evaluation \
    --test_set_name evaluation \
    --num_aug 100
```

### 2. Baseline 실행 (LLM-only)

```bash
cd /home/ubuntu/TinyRecursiveModels/hybrid_pipeline/experiments
./run_baseline.sh \
    /path/to/arc_agi \
    /data/trm/baseline_results.json \
    evaluation \
    3
```

**Note**: Baseline uses original ARC JSON files via dataset_access, not preprocessed data.

### 3. TRM 사전학습

```bash
./run_trm_pretrain.sh \
    /data/arc/processed \
    /data/trm/pretrain \
    cuda:0
```

### 4. TRM 평가

```bash
./run_trm_eval.sh \
    /data/trm/pretrain/checkpoint_step_5000.pt \
    /data/arc/processed \
    cuda:0
```

### 5. 조인트 학습 (LLaMA + TRM)

```bash
./run_joint_training.sh \
    --data_path /data/arc/processed \
    --output_dir /data/trm/joint_training \
    --trm_checkpoint /data/trm/pretrain/checkpoint_step_5000.pt \
    --device cuda:0 \
    --max_attempts 16
```

## 🧪 테스트 실행

```bash
# Adapter 테스트
cd /home/ubuntu/TinyRecursiveModels/hybrid_pipeline/adapters/tests
python test_projection.py

# Planner 테스트
cd ../../gpt_oss_port/tests
python test_planner.py

# Grid utils 테스트
python test_grid_utils.py
```

## 📊 모니터링 (WandB)

- **TRM Pretraining**: `arc-trm-pretrain`
- **Joint Training**: `arc-hybrid-joint`

로그된 메트릭:
- `loss`: 평균 손실
- `exact_match`: 정확한 일치 비율
- `shape_match`: Grid 크기 일치 비율
- `cell_accuracy`: Cell-level 정확도
- `avg_attempts`: 평균 시도 횟수
- `avg_trm_steps`: TRM 평균 추론 스텝
- `avg_reasoning_length`: 평균 추론 텍스트 길이

## 📝 주요 변경 사항 (이전 버전 대비)

| 항목 | 이전 | 현재 |
|------|------|------|
| LLM 모듈 | `gpt_integration.models.text_reasoning` | `gpt_oss_port.llm` |
| Planner | 없음 | `gpt_oss_port.planner.ARCPlanner` |
| Verifier | 없음 | `gpt_oss_port.verifier.GridVerifier` |
| Config | Class with `to_dict()` | `@dataclass` with `asdict()` |
| TRM carry | 직접 생성 + while 루프 | `model.initial_carry()` + `model()` |
| Import 경로 | `gpt_integration` | `gpt_oss_port` + `adapters` |

## 🔧 설정 파일

`experiments/config_joint.yaml` 편집:

```yaml
# Data
data_path: "/data/arc/processed"

# LLaMA
llama_model: "meta-llama/Llama-3.2-8B-Instruct"
llama_frozen: true

# TRM
trm_checkpoint: "/data/trm/pretrain/checkpoint_step_5000.pt"
trm_hidden_size: 512
trm_halt_max_steps: 16

# Training
batch_size: 1
max_attempts: 16
epochs: 10
lr: 0.0001
```

## 📚 API 참조

### `gpt_oss_port.llm.TextReasoningModule`

```python
llm = TextReasoningModule(
    model_name="meta-llama/Llama-3.2-8B-Instruct",
    freeze=True,
    device="cuda"
)

# Generate with optional latent prefix
z_init, text = llm.generate_latent(
    problem_text="Solve this puzzle",
    latent_prefix=None,  # Optional [hidden_size] tensor
    max_length=128
)
```

### `gpt_oss_port.planner.ARCPlanner`

```python
planner = ARCPlanner(
    llm_module=llm,
    max_attempts=16
)

# Multi-attempt solving
results = planner.multi_attempt_solve(
    problem_description="...",
    verifier_fn=verifier_function
)
```

### `adapters.text_to_latent.TextToLatentAdapter`

```python
adapter = TextToLatentAdapter(
    llm_hidden_size=4096,
    trm_hidden_size=512,
    trm_seq_len=900
)

z_H, z_L = adapter(llm_hidden_state)  # [batch, llm_dim] → [batch, seq, trm_dim]
```

## 🐛 문제 해결

### CUDA OOM
- `--batch_size 1` 사용
- `--max_attempts 8`로 감소

### Import 오류
- `sys.path`에 `/home/ubuntu/TinyRecursiveModels` 포함 확인
- `PYTHONPATH` 설정

### TRM 학습 안 됨
- 사전학습된 체크포인트 사용
- Learning rate 확인 (`--lr 1e-4`)

## 📞 파일 위치

| 컴포넌트 | 위치 |
|----------|------|
| 데이터 | `/data/arc/processed/` |
| TRM 체크포인트 | `/data/trm/pretrain/` |
| 조인트 체크포인트 | `/data/trm/joint_training/` |
| 로그 | `/data/trm/*/train_*.log` |

## ⚠️ 중요 사항

1. **모든 경로는 CLI 인자로 전달 필수**
2. **gpt_integration 의존성 없음**
3. **기존 models/ 및 dataset/ 재사용**
4. **compileall 통과 검증 완료**

---

**상태**: ✅ Production-ready
**최종 업데이트**: 2025-10-13
**Python 파일**: 18개
**Shell 스크립트**: 4개
