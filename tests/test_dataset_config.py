import unittest

import run


class _Base:
    def __init__(self, dataset='MMBench', skip_noimg=True):
        self.kwargs = dict(dataset=dataset, skip_noimg=skip_noimg)


class _ForwardsAll(_Base):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class _ForwardsWithExtra(_Base):
    def __init__(self, nframe=8, **kwargs):
        super().__init__(**kwargs)
        self.kwargs['nframe'] = nframe


class _NoForward(_Base):
    def __init__(self, dataset='X'):
        super().__init__(dataset=dataset)


class TestAcceptedInitParams(unittest.TestCase):

    def test_kwargs_only_init_inherits_parent_params(self):
        self.assertEqual(run._accepted_init_params(_ForwardsAll), {'dataset', 'skip_noimg'})

    def test_own_params_are_merged_with_parent_params(self):
        self.assertEqual(
            run._accepted_init_params(_ForwardsWithExtra), {'dataset', 'skip_noimg', 'nframe'})

    def test_init_without_kwargs_stops_the_walk(self):
        self.assertEqual(run._accepted_init_params(_NoForward), {'dataset'})


class TestBuildDatasetFromConfig(unittest.TestCase):

    def setUp(self):
        import vlmeval.dataset
        self._module = vlmeval.dataset
        self._module._ConfigProbe = _ForwardsWithExtra

    def tearDown(self):
        del self._module._ConfigProbe

    def test_config_keys_reach_a_kwargs_only_constructor(self):
        cfg = {'d': {'class': '_ConfigProbe', 'dataset': 'MMMU_Pro_10c', 'nframe': 4, 'bogus': 1}}
        ds = run.build_dataset_from_config(cfg, 'd')
        self.assertEqual(ds.kwargs, dict(dataset='MMMU_Pro_10c', skip_noimg=True, nframe=4))

    def test_strict_mode_only_rejects_truly_unknown_keys(self):
        cfg = {'d': {'class': '_ConfigProbe', 'dataset': 'MMMU_Pro_10c', 'bogus': 1}}
        with self.assertRaisesRegex(ValueError, 'bogus'):
            run.build_dataset_from_config(cfg, 'd', strict=True)
        del cfg['d']['bogus']
        ds = run.build_dataset_from_config(cfg, 'd', strict=True)
        self.assertEqual(ds.kwargs['dataset'], 'MMMU_Pro_10c')


if __name__ == '__main__':
    unittest.main()
