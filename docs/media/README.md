# README media

`curves.png` is drawn by `make_figure.py` from this commit's code, with no measured data:

```sh
uv run --with matplotlib docs/media/make_figure.py
```

It draws four curves from `examples/lab_films.toml` with `simulate(..., n=4, seed=(20261024, 7))` on
the CLI's default grid (2θ 0.05–8°, 500 points). Lines are `reflectivity` (before noise), points are
`counts / counts_scale`; each curve is shifted three decades below the one above it. Running the
command again gives the same file byte for byte (checked on Windows, matplotlib 3.11.2).
