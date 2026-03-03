import matplotlib.pyplot as plt 

with open("results_v2", "r") as file:
    lines = file.read().split("\n")

def transform_line(line):
    if line == "":
        return ()
    
    name, nb_param, percentage = line.split(" ; ")
    nb_param = float(nb_param.replace(",", ""))
    percentage = float(percentage[:-1])

    return name, nb_param, percentage
    
results = [[]]

for line in lines:
    print(line)
    result = transform_line(line)
    if len(result) != 3:
        results.append([])
    else:
        results[-1].append(result)

print(results)

plt.figure(figsize=(10,8))
colors = ["purple", "navy", "red"]
n = max(len(results), len(colors))
for models, color in zip(results[:n], colors[:n]):
    for name, nb_param, percentage in models:
        plt.scatter(nb_param, percentage, c=color, marker="x")
        plt.annotate(name, (nb_param, percentage), c=color, xytext=(5, 5), textcoords="offset points", ha="left", fontsize=7)

plt.axhline(y=90, color="red", linestyle="--")
plt.grid()
plt.title("scatter plot accuracy with respect to the score")
plt.xlabel("score")
plt.ylabel("accuracy (%)")
plt.savefig("scatter_results_v2.png")