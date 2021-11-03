from meddlr.data.slice_dataset import SliceData


class MockSliceDataset(SliceData):
    def __init__(self, dataset_dicts):
        super().__init__(dataset_dicts, None)

    def __getitem__(self, item):
        ex = self.examples[item]
        return ex["file_name"], ex["slice_id"], ex["is_unsupervised"]


class _MockDataset:
    def __init__(self, groups=None):
        if groups is not None:
            self.build_examples(groups)
        else:
            self.examples = [
                {"letter": "A", "number": 1},
                {"letter": "A", "number": 2},
                {"letter": "A", "number": 3},
                {"letter": "B", "number": 1},
                {"letter": "B", "number": 2},
                {"letter": "B", "number": 3},
                {"letter": "C", "number": 1},
                {"letter": "C", "number": 4},
            ]

    def build_examples(self, groups):
        examples = []
        for g, (num_sup, num_unsup) in groups.items():
            if not isinstance(g, (tuple, list)):
                g = (g,)
            base_dict = {f"field{idx}": _g for idx, _g in enumerate(g)}
            exs = [{"_is_unsupervised": False} for _ in range(num_sup)] + [
                {"_is_unsupervised": True} for _ in range(num_unsup)
            ]
            for ex in exs:
                ex.update(base_dict)
            examples.extend(exs)
        self.examples = examples

    def get_supervised_idxs(self):
        return [idx for idx, ex in enumerate(self.examples) if not ex["_is_unsupervised"]]

    def get_unsupervised_idxs(self):
        return [idx for idx, ex in enumerate(self.examples) if ex["_is_unsupervised"]]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]
