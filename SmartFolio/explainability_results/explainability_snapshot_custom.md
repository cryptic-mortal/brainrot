# Explainability Snapshot (custom)

- Generated at: 2025-11-08T22:10:44+00:00
- Model checkpoint: SmartFolio/checkpoints/ppo_hgat_custom_20251108_121539.zip
- Dataset directory: /home/pushpendras0026/inter/SmartFolio/dataset_default/data_train_predict_custom/1_hy

- Surrogate global R²: 0.0459 (~4.6% of allocation variance explained)

## Top Stocks by Tree Importance
| Rank | Ticker | Stock Index | Avg Weight |
| --- | --- | --- | --- |
| 1 | HINDUNILVR.NS | 37 | 1.34% |
| 2 | SHREECEM.NS | 80 | 1.28% |
| 3 | BAJFINANCE.NS | 7 | 1.27% |
| 4 | WIPRO.NS | 95 | 1.26% |
| 5 | GRSE.NS | 30 | 1.19% |
| 6 | ASHOKLEY.NS | 3 | 1.19% |
| 7 | CUMMINSIND.NS | 24 | 1.19% |
| 8 | NESTLEIND.NS | 62 | 1.19% |
| 9 | RVNL.NS | 77 | 1.17% |
| 10 | SIEMENS.NS | 81 | 1.16% |

## Attention Semantic Mix
| Semantic Channel | Share |
| --- | --- |
| Positive | 35.17% |
| Negative | 23.70% |
| Industry | 22.30% |
| Self | 18.84% |

## Strongest Attention Edges
### Industry channel
| Rank | Source | Source Index | Target | Target Index | Attention |
| --- | --- | --- | --- | --- | --- |
| 1 | MRF.NS | 59 | GOLDBEES.NS | 29 | 4.20% |
| 2 | MRF.NS | 59 | CESC.NS | 19 | 4.14% |
| 3 | MRF.NS | 59 | NIFTYBEES.NS | 63 | 4.10% |
| 4 | MRF.NS | 59 | MON100.NS | 57 | 4.05% |
| 5 | IDFCFIRSTB.NS | 39 | SIEMENS.NS | 81 | 1.95% |

### Positive channel
| Rank | Source | Source Index | Target | Target Index | Attention |
| --- | --- | --- | --- | --- | --- |
| 1 | SUZLON.NS | 84 | PAGEIND.NS | 67 | 18.76% |
| 2 | TRIDENT.NS | 92 | PAGEIND.NS | 67 | 17.97% |
| 3 | RELIANCE.NS | 75 | PAGEIND.NS | 67 | 17.30% |
| 4 | MANAPPURAM.NS | 53 | PAGEIND.NS | 67 | 17.25% |
| 5 | ICICIBANK.NS | 38 | PAGEIND.NS | 67 | 17.15% |

### Negative channel
| Rank | Source | Source Index | Target | Target Index | Attention |
| --- | --- | --- | --- | --- | --- |
| 1 | INFY.NS | 44 | MRF.NS | 59 | 49.52% |
| 2 | RELIANCE.NS | 75 | MRF.NS | 59 | 48.23% |
| 3 | BHARTIARTL.NS | 12 | MRF.NS | 59 | 48.21% |
| 4 | ICICIBANK.NS | 38 | MRF.NS | 59 | 48.04% |
| 5 | AXISBANK.NS | 6 | MRF.NS | 59 | 47.82% |
