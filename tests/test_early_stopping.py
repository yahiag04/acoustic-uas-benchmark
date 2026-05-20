from counter_uas.training.early_stopping import EarlyStopper


def test_early_stopper_signals_stop_after_patience_exhausted():
    stopper = EarlyStopper(patience=2, mode="max")

    assert stopper.update(0.5) is False  # improvement
    assert stopper.update(0.4) is False  # 1 epoch without improvement
    assert stopper.update(0.45) is False  # 2 epochs without improvement
    assert stopper.update(0.45) is True  # patience exhausted


def test_early_stopper_resets_on_improvement():
    stopper = EarlyStopper(patience=1, mode="max")

    stopper.update(0.5)
    stopper.update(0.4)  # 1 without improvement
    assert stopper.update(0.6) is False  # improvement resets counter
    assert stopper.update(0.55) is False  # 1 without improvement
    assert stopper.update(0.55) is True  # patience exhausted


def test_early_stopper_min_mode_tracks_decreasing_metric():
    stopper = EarlyStopper(patience=1, mode="min")

    assert stopper.update(1.0) is False
    assert stopper.update(0.5) is False  # improvement
    assert stopper.update(0.6) is False  # 1 without improvement
    assert stopper.update(0.7) is True
