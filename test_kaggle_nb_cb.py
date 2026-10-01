import json, os
BASE = "/home/prudhvi/Hackathons/Predicting_Electric_Vehicle_Purchases"
os.environ["LOCAL_DATA"] = BASE + "/Data"
nb = json.load(open(BASE + "/kaggle_ev_pipeline.ipynb"))
codes = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
g = {}
codes[0] = (codes[0].replace("RUN_DNN = True", "RUN_DNN = False")
            .replace("CB_SEEDS = [42, 123, 456]", "CB_SEEDS = [42, 123]")
            .replace('WORK = "/kaggle/working" if os.path.isdir("/kaggle/working") else "."', 'WORK = "/tmp"'))
codes[4] = codes[4].replace("iterations=40000", "iterations=100")
for i in [0, 1, 2, 3, 4, 6, 7]:
    exec(codes[i], g)
    print("cell", i, "ok", flush=True)
print(open("/tmp/submission.csv").read()[:200])
