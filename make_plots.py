import json
import seaborn as sns
import matplotlib.pyplot as plt
import os

sns.set_theme()

def parse_args():
    parser = ArgumentParser(description='CIFAR')
    parser.add_argument('--path', type=str, help='path to json file', required=True)
    parser.add_argument('--out_dir', type=str, help='path to output directory, must exist beforehand', required=True)
    return parser.parse_args()

# Take in JSON file and plot and save losses/accs
def plot_data(path, out_dir):
  with open(path) as json_file:
    data = json.load(json_file)
    
    epochs = range(len(data["train loss"]))
    plt.plot(epochs, data["train loss"], label="Train Loss")
    plt.plot(epochs, data["test loss"], label="Test Loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title("Losses")
    plt.legend()
    
    plt.savefig(os.path.join(out_dir, "losses.png"))
    plt.show()
    plt.clf()

    plt.plot(epochs, data["train acc"], label="Train Accuracy")
    plt.plot(epochs, data["test acc"], label="Test Accuracy")
    plt.xlabel('Epoch')
    
    plt.ylabel('Accuracy')
    
    plt.title("Accuracies")
    plt.legend()
    plt.savefig(os.path.join(out_dir, "acc.png"))
    plt.show()

if __name__ == "__main__":
  args = parse_args()
  plot_data(args.path, args.out_dir)

