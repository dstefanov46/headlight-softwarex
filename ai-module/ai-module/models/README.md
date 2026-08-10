# Legacy and Experimental Models

The modules in this directory (`AdapGL.py`, `ASTGCN.py`, `DCRNN.py`, `GCN.py`)
are retained legacy/experimental implementations. They are not instantiated by
the primary `train_model.py --once --controller headlight` path or by
`live_inference.py`.

The active implementation is `headlight.models.HeadlightForecastModel`.
