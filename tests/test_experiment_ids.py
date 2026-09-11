from wds_sentinel.experiments.ids import config_hash, make_experiment_id


def test_config_hash_deterministic():
    assert config_hash("a: 1") == config_hash("a: 1")


def test_config_hash_differs_on_content_change():
    assert config_hash("a: 1") != config_hash("a: 2")


def test_experiment_id_unique_over_time():
    id1 = make_experiment_id("a: 1")
    id2 = make_experiment_id("a: 1")
    # Same config -> same config-hash component, but timestamp differs.
    assert id1 != id2
