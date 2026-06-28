# Albumin BSA 결합 예측 실험 결과 요약

## 실험 개요
- **목표:** 83개 중성 유기 화합물의 BSA 결합 상수(log K_BSA/w) 예측
- **실험일:** 2026-06-08
- **데이터:** albumin_data.csv (name, SMILES, logK_exp)

---

## 최종 결과

### Calibrated R² 성능 비교

| 방법 | Calibrated R² | Spearman | 설명 |
|------|---------------|----------|------|
| **Vina blind** | **0.274** | -0.433 | 전체 단백질 탐색 |
| Vina focused (Sudlow I) | 0.216 | -0.345 | 결합 부위 고정 |
| Boltz-2 unconstrained | 0.045 | -0.211 | 무제약 도킹 |
| Boltz-2 pocket (Sudlow I) | 0.020 | -0.086 | Pocket 강제 |

### Absolute R² (보정 없음)

| 방법 | Absolute R² |
|------|--------------|
| Vina blind | -0.174 |
| Vina focused | -0.829 |
| Boltz-2 unconstrained | -0.418 |
| Boltz-2 pocket | -0.384 |

---

## 주요 발견

1. **Vina > Boltz-2**
   - Vina blind: Calibrated R² = 0.274 (최고)
   - Boltz-2: Calibrated R² = 0.02~0.05 (매우 낮음)
   - 물리 기반 도킹이 AI 모델보다 우수

2. **Site 고정 효과 없음**
   - Pocket-conditioned (R² = 0.020) < Unconstrained (R² = 0.045)
   - 결합 부위 제약이 성능 향상에 기여하지 않음

3. **Boltz-2 한계**
   - 중성 유기 화합물에 대한 예측력 부족
   - Applicability domain 문제

---

## 기술 이슈

### CUDA 호환성 문제
- **에러:** `libnvrtc.so.12: cannot open shared object file`
- **원인:** 시스템 CUDA 13.2 vs Boltz-2 요구 CUDA 12
- **해결:** `--no_kernels` 옵션 사용

---

## 사용 코드 및 결과 파일

### 실험 코드
```
vina_blind.py              # Vina 전체 탐색
vina_focused_sudlow1.py    # Vina Sudlow site I
boltz2_pocket_sudlow1.py   # Boltz-2 Pocket
vina_common.py             # 공통 함수
```

### SLURM 스크립트
```
r_vina_blind.sh
r_vina_focused.sh
r_boltz_pocket.sh
```

### 결과 CSV
```
vina_blind_results.csv
vina_focused_sudlow1_results.csv
boltz2_pocket_results.csv
boltz2_results.csv (이전)
```

### 결과 그래프
```
vina_blind_fit.png
vina_focused_sudlow1_fit.png
boltz2_pocket_fit.png
boltz2_albumin_fit.png (이전)
```