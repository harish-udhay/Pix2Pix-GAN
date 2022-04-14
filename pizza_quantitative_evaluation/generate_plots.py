import json
import matplotlib.pyplot as plt

with open("./out/is.json") as f:
    inception_score_dict = json.load(f)

with open("./out/fid.json") as f:
    fid_score_dict = json.load(f)

keys = sorted([int(key) for key in inception_score_dict.keys()])

#fig, ax = plt.subplots()
plt.title("Inception Score on Real Pizza Dataset")
plt.xlabel("Epochs")
plt.ylabel("Score")
plt.plot(keys, [inception_score_dict.get(str(key))[0] for key in keys], label = "Inception Score")
plt.show()

plt.title("FID Score on Real Pizza Dataset")
plt.xlabel("Epochs")
plt.ylabel("Score")
plt.plot(keys, [fid_score_dict.get(str(key)) for key in keys], label = "FID Score")
plt.show()
