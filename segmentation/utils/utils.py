import csv
import matplotlib.pyplot as plt


def parse_paths(split_path):
    train_path, val_path, test_path = [], [], []

    split = []
    with open(split_path, 'r') as f:
        for row in csv.DictReader(f):
            split.append(row)


    for row in split:
        paths = [row['img_path'], row['sem_path'], row['inst_path']]

        match row['split']:
            case 'train':
                train_path.append(paths)
            case 'dev':
                val_path.append(paths)
            case 'test':
                test_path.append(paths)

    return train_path, val_path, test_path


def trainable_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f'Params: {params:.1f}M')


def test_predicts(model, test_dataset, path, device='cpu'):
    for i, (X, y) in enumerate(test_dataset):
        pred = model(X.unsqueeze(0).to(device)).cpu().detach().squeeze(0)

        fig, axes = plt.subplots(ncols=3, figsize=(14, 14))
        axes[0].imshow(X.permute(1, 2, 0))
        axes[0].set_title('Image')
        axes[1].imshow(y.permute(1, 2, 0))
        axes[1].set_title('Target')
        axes[2].imshow(pred.permute(1, 2, 0))
        axes[2].set_title('Predict')

        plt.savefig(f"{path}predict_{i}.png")
        plt.close()
