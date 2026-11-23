import pandas as pd

df = pd.read_csv("results_v2.csv")

for _, row in df.iterrows():
    print("\n" + "=" * 60)
    print("Depth:", row["depth"])
    print("Trial:", row["trial"])
    print("Expected:", row["expected"])
    print("Actual:", row["response"])
    print("Retrieval correct:", bool(row["retrieval_correct"]))
    print("Prompt processing:", row["prompt_eval_sec"], "sec")
    print("Generation:", row["generation_sec"], "sec")
    print("Model loading:", row["load_sec"], "sec")
