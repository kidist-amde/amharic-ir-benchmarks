from ColBERT_AM.infra.run import Run
import os
import ujson

from ColBERT_AM.evaluation.loaders import load_queries


class Queries:
    def __init__(self, path=None, data=None):
        self.path = path

        if data:
            assert isinstance(data, dict), type(data)

        self._load_data(data) or self._load_file(path)
    
    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data.items())

    def provenance(self):
        return self.path
    
    def toDict(self):
        return {'provenance': self.provenance()}

    def _load_data(self, data):
        """
        Load queries from a dictionary.
        """
        if data is None:
            return None

        self.data = {}
        self._qas = {}

        for qid, content in data.items():
            if isinstance(content, dict) and "query" in content:
                self.data[str(qid)] = content["query"]
                self._qas[str(qid)] = content  # Store full query info
            else:
                self.data[str(qid)] = content  # Store as string

        if not self._qas:
            self._qas = None  # Ensure it's not empty

        return True

    def _load_file(self, path):
        """
        Load queries from a JSONL file in the Amharic MS MARCO format.
        Each line should have {"query_id": int, "headline": str}
        """
        self.data = {}
        self._qas = {}

        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    qa = ujson.loads(line.strip())

                    # Ensure query_id exists and is stored as an integer
                    if "query_id" not in qa or "headline" not in qa:  #  Fix: Changed "query" to "headline"
                        print(f"Warning: Skipping invalid query line: {line.strip()}")
                        continue

                    qid = int(qa["query_id"])  #  Convert query_id to integer
                    self.data[qid] = qa["headline"]  #  Store as "headline"
                    self._qas[qid] = qa  # Store full query object

                except Exception as e:
                    print(f" Error parsing line: {line.strip()} | Error: {e}")

        return self.data


    def qas(self):
        return dict(self._qas) if self._qas else {}

    def __getitem__(self, key):
        return self.data[key]

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    def items(self):
        return self.data.items()

    def save(self, new_path):
        assert new_path.endswith('.tsv')
        assert not os.path.exists(new_path), new_path

        with Run().open(new_path, 'w') as f:
            for qid, content in self.data.items():
                content = f'{qid}\t{content}\n'
                f.write(content)
            
            return f.name

    def save_qas(self, new_path):
        assert new_path.endswith('.json')
        assert not os.path.exists(new_path), new_path

        with open(new_path, 'w') as f:
            for qid, qa in self._qas.items():
                qa["query_id"] = qid  # Ensure correct format
                f.write(ujson.dumps(qa, ensure_ascii=False) + '\n')

    @classmethod
    def cast(cls, obj):
        if obj is None:
            return None

        if isinstance(obj, str):
            return cls(path=obj)  # Load from file

        if isinstance(obj, dict):  # If already in dictionary format
            return cls(data=obj)

        if isinstance(obj, cls):
            return obj  # Already an instance

        assert False, f"obj has type {type(obj)} which is not compatible with cast()"
