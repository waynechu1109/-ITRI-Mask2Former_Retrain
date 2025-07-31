import re
import matplotlib.pyplot as plt

log_path = "ckpt/merged/self_train_8/log.txt"

iterations = []
total_losses = []

pattern = re.compile(r"iter:\s*(\d+)\s+total_loss:\s*([\d.]+)")

with open(log_path, "r") as f:
    for line in f:
        match = pattern.search(line)
        if match:
            iter_num = int(match.group(1))
            total_loss = float(match.group(2))
            iterations.append(iter_num)
            total_losses.append(total_loss)
 
# plot
plt.figure(figsize=(10, 5))
plt.scatter(iterations, total_losses, marker='o', linewidth=0.8)
plt.xlabel("Iteration")
plt.ylabel("Total Loss")
plt.title("Training Loss Curve")
plt.grid(True)
plt.tight_layout()
plt.show()
