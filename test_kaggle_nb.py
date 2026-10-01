import json, os
BASE = "/home/prudhvi/Hackathons/Predicting_Electric_Vehicle_Purchases"
os.environ["LOCAL_DATA"] = BASE + "/Data"
nb = json.load(open(BASE + "/kaggle_ev_pipeline.ipynb"))
codes = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
for i, c in enumerate(codes):
    compile(c, f"cell{i}", "exec")
print("all", len(codes), "code cells compile", flush=True)
g = {}
codes[0] = codes[0].replace("RUN_CATBOOST = True", "RUN_CATBOOST = False").replace(
    "DNN_SEEDS = [42, 123, 456, 789, 2024]", "DNN_SEEDS = [42]")
codes[5] = codes[5].replace("max_epochs=50", "max_epochs=1")
for i in [0, 1, 2, 3, 4]:
    exec(codes[i], g)
    print("cell", i, "ok", flush=True)
exec(codes[5], g)
print("DNN 1-epoch OOF", g["roc_auc_score"](g["y"], g["d_oof"]), flush=True)
