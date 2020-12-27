import cv2
import matplotlib.pyplot as plt


def visualize_transpose(images, transform):
    """Plot images and their transformations"""
    fig = plt.figure(figsize=(16, 16))
    for i, im in enumerate(images):
        ax = fig.add_subplot(1, 5, i + 1, xticks=[], yticks=[])
        im = transform(image=im)["image"].numpy().transpose(1, 2, 0)
        plt.imshow(im)


def show_images(train, path, name_mapping):
    """画像表示"""
    selected_images = []
    fig = plt.figure(figsize=(16, 16))
    for class_id, class_name in name_mapping.items():
        for i, (idx, row) in enumerate(
            train.loc[train["label"] == class_id].sample(4).iterrows()
        ):
            ax = fig.add_subplot(5, 4, class_id * 4 + i + 1, xticks=[], yticks=[])
            img = cv2.imread(f"{path}{row['image_id']}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.imshow(img)
            ax.set_title(f"Image: {row['image_id']}. Label: {row['label']}")
            if i == 0:
                selected_images.append(img)
    return selected_images


def show_pred_diff_images(oof_preds_df, name_mapping):
    """予測外した画像表示"""
    selected_image_paths = []
    fig = plt.figure(figsize=(16, 16))
    c = 1
    for class_id1, class_name1 in name_mapping.items():
        for class_id2, class_name2 in name_mapping.items():
            if class_id1 != class_id2:
                try:
                    img_path = (
                        oof_preds_df.loc[
                            (oof_preds_df["target"] == class_id1)
                            & (oof_preds_df["prediction"] == class_id2)
                        ]
                        .sort_values("logit", ascending=False)["file_path"]
                        .values[0]
                    )

                    ax = fig.add_subplot(5, 4, c, xticks=[], yticks=[])
                    img = cv2.imread(img_path)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    plt.imshow(img)
                    ax.set_title(
                        f"Correct class: {class_id1}. Predicted class: {class_id2}"
                    )
                    c += 1
                    selected_image_paths.append(img_path)
                except IndexError:
                    print(f"target={class_id1}, prediction={class_id2} data none")
                    pass
    return selected_image_paths
