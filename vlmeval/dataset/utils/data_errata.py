"""Index-specific errata for the pinned TSV releases reported in Issue #1754."""

import warnings

ISSUE_1754_DATA_ERRATA = {
    'MMBench_DEV_EN': ('b6caf1133a01c6bb705cf753bb527ed8', (1770, 1849, 1852)),
    'MMBench_DEV_CN': ('08b8fc3324a5ed74155350f57be69fbd', (1770, 1849, 1852, 1879, 1880)),
    'MMBench_dev_ar': ('4271b4a0d0200e1a86380a878e0d64a4', (710, 1849, 1852)),
    'MMBench_dev_cn': ('2ed5135326fed02c8e51ea50dda8222f', (1770, 1849, 1852, 1879, 1880)),
    'MMBench_dev_en': ('d9ab776fc018b3d45785e9a5c23431c2', (1770, 1849, 1852)),
    'MMBench_dev_pt': ('4ddfbcd27ef12444b908c03831cd0295', (710, 712, 1849, 1852)),
    'MMBench_dev_ru': ('5ba1171ff2e68f80637bf78349e402a5', (710, 712, 1849, 1852)),
    'MMBench_dev_tr': ('4fab39d501389d3d6cc90264bb708f11', (712, 1849, 1852)),
    'MMMB_ar': ('f3a18b6385f1d9701840aa42de27aead', (1141, 1286, 1634)),
    'MMMB_cn': ('13ed82fa89730037292fcaa27f08f430', (1141, 1286, 1634)),
    'MMMB_pt': ('548ea2b3bb2da991790386f0015d30d1', (
        12, 26, 31, 61, 63, 138, 153, 154, 162, 227, 242, 249, 270, 288,
        302, 306, 309, 311, 314, 337, 338, 356, 369, 371, 375, 406, 419,
        458, 1141, 1286, 1547, 1634, 1723,
    )),
    'MMMB_tr': ('0733739d43090327975294292bc5cd67', (1141, 1286, 1634)),
    'MTVQA_TEST': ('d87c17dbab934b7cd89c0a3c1c5657f4', (
        743, 744, 745, 746, 747, 1720, 1721, 2034, 2037, 2065, 2138, 2364,
        2367, 2863, 3093, 3124, 3125, 3126, 3127, 3796, 3797, 3798, 3799,
        3800, 3820, 3821, 5044, 5046, 5047, 5048, 5853, 5855, 5856,
    )),
    'SEEDBench2_Plus': ('e32d3216dc4f452b0fe497a52015d1fd', (1592,)),
    'AI2D_TEST': ('0f593e0d1c7df9a3d69bf1f947e71975', (225703,)),
    'AI2D_TEST_NO_MASK': ('fd8f463634d4fe9fbd23b876e8eea5be', (225703,)),
    'MMT-Bench_VAL': ('8dd4b730f53dbf9c3aed90ca31c928e0', (8279, 8289, 8293, 20554)),
    'MMT-Bench_VAL_MI': ('c7d7b998eb5cd9aa36c7d4f721472462', (8279, 8289, 8293, 20554)),
}


def filter_unscorable_samples(data, dataset, dataset_md5):
    """Exclude known samples that cannot be scored reliably.

    These rows contain conflicting reference answers for identical inputs,
    duplicated correct options (including translation collisions), or only
    one available answer option. Keeping them can penalize valid answers or
    award trivial credit, distorting evaluation scores. See Issue #1754:
    https://github.com/open-compass/VLMEvalKit/issues/1754

    Only listed indices from matching MD5-pinned releases are excluded,
    along with their MMBench circular variants; other rows are unchanged.
    """
    errata = ISSUE_1754_DATA_ERRATA.get(dataset)
    if errata is None or dataset_md5 != errata[0]:
        return data

    indices = data['index'].astype(int)
    excluded = indices.isin(errata[1])
    if dataset.startswith('MMBench_'):
        group_indices = data['g_index'].astype(int) if 'g_index' in data else indices % 1000000
        excluded |= group_indices.isin(errata[1])

    if not excluded.any():
        return data
    warnings.warn(
        f'Excluding {int(excluded.sum())} known invalid samples from {dataset} (Issue #1754).',
        stacklevel=2,
    )
    return data.loc[~excluded].copy()
