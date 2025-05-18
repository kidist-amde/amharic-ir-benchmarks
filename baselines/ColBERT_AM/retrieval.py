from ColBERT_AM.data import Queries
from ColBERT_AM.infra import Run, RunConfig, ColBERTConfig
from ColBERT_AM import Searcher

if __name__ == "__main__":
    with Run().context(RunConfig(nranks=1, experiment="amharic_passage_retrieval")):
        # Ensure the correct paths are set
        index_path = "experiments/default/indexes/amharic_index.nbits=2"
        checkpoint_path = "experiments/default/checkpoints/ColBERT-Base-Amharic"


        
        collection_path = "dataset/processed/msmarco-amharic-news_dataset/collection.tsv"
        queries_path = "dataset/processed/msmarco-amharic-news_dataset/queries_dev.jsonl"

        output_ranking = f"{index_path}/Roberta-Medium-amharic_ranking.tsv"

        # Debug: Print collection path
        print(f"DEBUG: Collection Path -> {collection_path}")

        # set explicit config
        config = ColBERTConfig(
            root=index_path,
            checkpoint=checkpoint_path,
            collection=collection_path,  
            index_name="amharic_index.nbits=2"
        )

        config.collection = collection_path

        # Extra Debugging to ensure collection is not None
        if config.collection is None:
            raise ValueError("ERROR: `config.collection` is None! Fix this before passing to Searcher.")

        print(f"DEBUG: `config.collection` successfully set -> {config.collection}")

        searcher = Searcher(index="amharic_index.nbits=2", config=config)

        # Debug: Ensure searcher initialized correctly
        print(" Searcher initialized successfully!")

        queries = Queries(queries_path)
        ranking = searcher.search_all(queries, k=100)
        ranking.save(output_ranking)

    print(f"Retrieval complete! Results saved at: {output_ranking}")
