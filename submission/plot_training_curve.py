import matplotlib.pyplot as plt

epochs = [1,2,3,4,5]
f1_scores = [0.8923,0.8958,0.8984,0.9002,0.8981]

plt.figure()

plt.plot(epochs, f1_scores, marker='o')

plt.xlabel("Epoch")
plt.ylabel("Switch F1")
plt.title("Training Performance Across Epochs")

plt.xticks(epochs)

plt.savefig("training_curve.png", dpi=300)

plt.show()