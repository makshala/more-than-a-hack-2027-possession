# More Than A Hack 2027 — Possession Prediction

Deterministyczny potok predykcji posiadania piłki (`SEED=2027`).

## Szybki start
1. Umieść surowe pliki meczowe w `data/raw/` (katalog jest ignorowany przez Git).
2. Wygeneruj syntetyczne kontrakty do testów: `make mock`
3. Uruchom pełny potok: `make pipeline`
4. Zweryfikuj reprodukowalność (2x uruchomienie + sha256): `make reproduce`
