import pandas as pd

from wds_sentinel.experiments.split import split_masks


def test_partitions_do_not_overlap():
    idx = pd.date_range("2018-01-01", "2019-12-31 23:55", freq="5min")
    masks = split_masks(idx, horizon_steps=12)
    train, val, test = masks["train"], masks["validation"], masks["test"]
    assert not (train & val).any()
    assert not (val & test).any()
    assert not (train & test).any()


def test_embargo_excludes_last_horizon_rows_of_train_and_validation():
    idx = pd.date_range("2018-01-01", "2019-12-31 23:55", freq="5min")
    masks = split_masks(idx, horizon_steps=12)
    from wds_sentinel.experiments.split import TRAIN_END, VAL_END

    embargo_train = idx[(idx > TRAIN_END - pd.Timedelta(minutes=5 * 12)) & (idx <= TRAIN_END)]
    assert not masks["train"].loc[embargo_train].any()

    embargo_val = idx[(idx > VAL_END - pd.Timedelta(minutes=5 * 12)) & (idx <= VAL_END)]
    assert not masks["validation"].loc[embargo_val].any()


def test_test_partition_is_full_2019_and_unembargoed():
    idx = pd.date_range("2018-01-01", "2019-12-31 23:55", freq="5min")
    masks = split_masks(idx, horizon_steps=12)
    test_idx = idx[masks["test"]]
    assert test_idx.min() == pd.Timestamp("2019-01-01 00:00:00")
    assert test_idx.max() == pd.Timestamp("2019-12-31 23:55:00")


# --- split_masks_detection ---
from wds_sentinel.experiments.split import split_masks_detection, VAL_START, TEST_START


def test_detection_split_partitions_do_not_overlap():
    idx = pd.date_range("2018-01-01", "2019-12-31 23:55", freq="5min")
    masks = split_masks_detection(idx, window=pd.Timedelta("1h"))
    train, val, test = masks["train"], masks["validation"], masks["test"]
    assert not (train & val).any()
    assert not (val & test).any()
    assert not (train & test).any()


def test_detection_split_embargoes_start_of_validation_and_test():
    idx = pd.date_range("2018-01-01", "2019-12-31 23:55", freq="5min")
    window = pd.Timedelta("1h")
    masks = split_masks_detection(idx, window=window)

    embargo_val = idx[(idx >= VAL_START) & (idx < VAL_START + window)]
    assert not masks["validation"].loc[embargo_val].any()

    embargo_test = idx[(idx >= TEST_START) & (idx < TEST_START + window)]
    assert not masks["test"].loc[embargo_test].any()
