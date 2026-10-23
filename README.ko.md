# ReflectoSim

[![Tests](https://github.com/NaNIsaac/ReflectoSim/actions/workflows/ci.yml/badge.svg)](https://github.com/NaNIsaac/ReflectoSim/actions/workflows/ci.yml) [![License: MIT](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

기계학습용 현실적 합성 X선 반사율(XRR) 데이터: 층 구조와 기기에 대한 prior, 계수 잡음, 그리고 다시 만드는 방법을 기록하는 데이터셋 형식.

측정 곡선과 닮은 시뮬레이션 곡선이 필요한 XRR 역문제 모델을 학습하거나 시험하는 사람을 위한 도구다. 두께에 묶인 거칠기와 이웃 사이 최소 대비를 갖춘 무작위 다층막, 지정한 위치의 자연 산화막, 가우시안 분해능, footprint, Kα₂, 2θ offset, 배경, Poisson 계수를 만든다. 곡선은 피팅 엔진 [ReflectoFit](https://github.com/NaNIsaac/ReflectoFit)의 순방향 모델에서 나오므로, 이 곡선으로 학습한 네트워크와 같은 측정의 피팅이 같은 물리를 쓴다. 이 패키지는 데이터 분포만 더한다.

상태: 초기 개발 단계.

## 설치

아직 PyPI에 없다. clone한 뒤 [uv](https://docs.astral.sh/uv/)로(ReflectoFit은 GitHub에서 받는다):

```sh
git clone https://github.com/NaNIsaac/ReflectoSim
cd ReflectoSim
uv sync
```

## 첫 실행

예제 prior로 Si 위 SiO₂ 위의 막 1–3개짜리 곡선 1000개를 만들고 파일을 요약한다:

<!-- test -->
```sh
uv run reflectosim sim --prior examples/lab_films.toml --n 1000 --seed 20261024,1 -o train.npz
uv run reflectosim describe train.npz
```

`train.npz`에는 `q`, `counts`, `reflectivity`(잡음 전), `counts_scale`, 파라미터마다 배열 하나(`layer0.thickness`, `oxide.sld`, `tth_offset`, ...; 층이 적은 곡선은 NaN)가 있다. 메타데이터에는 prior, seed, 버전이 있고, `reflectosim.priors.from_metadata`로 prior를 되살리면 같은 seed에서 같은 파일이 나온다. HDF5는 `.h5` 이름을 쓴다(`uv sync --extra hdf5`).

## Prior

prior는 TOML 파일이고 `examples/lab_films.toml`에 모든 항목이 있다. 자연 산화막 위치에는 기본값이 없다: `oxide = "under"`는 기판 위, `"over"`는 막들 위, `"none"`은 산화막 없음. 두 순서의 곡선은 수십 % 다르므로 모든 prior가 선택을 명시한다.

## Python

```python
import numpy as np
from reflectofit import tth_to_q
from reflectosim import InstrumentPrior, StackPrior, simulate

stack = StackPrior(
    n_layers=(1, 3),
    thickness=(20, 250),
    sld=(5, 80),
    roughness=(2, 10),
    min_contrast=4.0,
    oxide="none",
)
instrument = InstrumentPrior(
    sigma_q=(5e-4, 3e-3), tth_offset=(-0.01, 0.01), log_background=(-8.5, -7.0), log_counts=(6, 8)
)
batch = simulate(
    stack, instrument, tth_to_q(np.linspace(0.05, 8, 500)), n=10_000, seed=(20261024, 1)
)
batch.counts, batch.reflectivity, batch.params["layer0.thickness"]
```

## 개발

```sh
uv run pytest
```

## 라이선스

MIT
