from typing import List


class TextVidDataset:
    def __init__(self, instances, captions=None) -> None:
        self.instances = instances
        self.captions = captions

    def __getitem__(self, i):
        return self.instances[i]

    def __len__(self):
        return len(self.instances)

    def punctuate(self, s: str):
        return s if s.endswith(".") else (s + '.')
    
    def get_text(self, index: int) -> List[str]:
        raise NotImplementedError()