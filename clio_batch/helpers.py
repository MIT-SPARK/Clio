"""Helper methods for clustering."""
import torch
from PIL import Image
import cv2
import yaml
import matplotlib.pyplot as plt
import numpy as np
import copy
import networkx as nx


def default_device(use_cuda=True):
    """Get default device to use for pytorch."""
    return "cuda" if torch.cuda.is_available() and use_cuda else "cpu"


class ClipHandler:
    """Wrapper around various CLIP models."""

    def __init__(self, model_name, device=None, pretrained="laion2B_s32B_b79K"):
        """Construct a wrapper around CLIP."""
        print(model_name)
        self.device = default_device() if device is None else device
        self.use_open_clip = model_name == "ViT-H-14"
        if self.use_open_clip:
            import open_clip
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained, device=self.device)
            self.tokenizer = open_clip.get_tokenizer(model_name)
        else:
            import clip
            self.model, self.preprocess = clip.load(model_name, device=self.device)
            self.tokenizer = clip.tokenize

    def get_text_clip_features(self, strings):
        """Compute text embeddings for all strings in list."""
        texts = torch.cat([self.tokenizer(string)
                           for string in strings]).to(self.device)
        with torch.no_grad():
            # Access the underlying model when using DataParallel
            text_features = self.model.encode_text(texts)
        return text_features.cpu().numpy()

    def process_and_encode_image(self, rgb_img):
        """Compute image CLIP embedding."""
        pil_img = Image.fromarray(rgb_img)
        image = self.preprocess(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            return np.squeeze(self.model.encode_image(image).cpu().numpy())


def compute_sim_to_tasks(v1, v2, canonical_features=None, use_lerf_canonical=True):
    # if using multiple tasks, task MUST be v1
    if type(v1) == list:
        v1 = copy.deepcopy(v1)

    v1 = np.array(v1)
    model_size = v1.shape[-1]
    v1 = v1.reshape((-1, model_size))
    v2 = np.array(v2).reshape((-1, model_size))

    v1_norm = np.linalg.norm(v1, axis=v1.ndim - 1)
    v2_norm = np.linalg.norm(v2, axis=v2.ndim - 1)
    v1 = v1/(v1_norm[:, np.newaxis])
    v2 = v2/(v2_norm[:, np.newaxis])

    if use_lerf_canonical and canonical_features is not None:
        canonical_features = canonical_features.reshape((-1, model_size))
        canonical_norm = np.linalg.norm(
            canonical_features, axis=canonical_features.ndim - 1)
        canonical_features = canonical_features/(canonical_norm[:, np.newaxis])
        canonical_term = np.max(np.exp(canonical_features @ v2.T), axis=0, keepdims=True)
        result = np.exp(v1 @ v2.T) / (canonical_term + np.exp(v1 @ v2.T))

    else:
        result = (v1 @ v2.T)
        
    return result


def compute_cosine_sim(v1, v2, canonical_features=None, use_lerf_canonical=True, get_assignments=False):
    # for each v2, pick the v1 that maximizes the sim and return that value
    sims = compute_sim_to_tasks(
        v1, v2, canonical_features=canonical_features, use_lerf_canonical=use_lerf_canonical)
    best_vals = np.max(sims, axis=0, keepdims=True)

    best_idxes = np.argmax(sims, axis=0)

    if get_assignments:
        return best_vals, best_idxes
    else:
        return best_vals


def parse_tasks_from_yaml(yaml_file):
    prompts = []
    with open(yaml_file, "r") as stream:
        prompts = [prompt for prompt in yaml.safe_load(stream)]
    return prompts


def crop_image(frame, bbox, padding=4):
    frame = frame.copy()
    x_min, y_min, x_max, y_max = bbox

    # Check and adjust padding to avoid going beyond the image borders
    image_height, image_width, _ = frame.shape
    left_padding = min(padding, x_min)
    top_padding = min(padding, y_min)
    right_padding = min(padding, image_width - x_max)
    bottom_padding = min(padding, image_height - y_max)

    # Apply the adjusted padding
    x_min -= left_padding
    y_min -= top_padding
    x_max += right_padding
    y_max += bottom_padding

    x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)

    cropped_image = frame[y_min:y_max, x_min:x_max, :]
    return cropped_image


def apply_mask(cv_img, mask):
    _, binary_mask = cv2.threshold(mask.astype(np.uint8), .5, 255, cv2.THRESH_BINARY)

    # Use OpenCV's bitwise_and to apply the mask
    result_img = cv2.bitwise_and(cv_img, cv_img, mask=binary_mask)
    return result_img


def create_obj_masks(masks, bboxes, clusters):
    if len(masks) == 0:
        return None
    mask_shape = masks[0].shape
    obj_masks = []
    boxes = []
    for cluster in clusters:
        obj_mask = np.zeros(mask_shape, dtype=bool)
        obj_box = None
        for mask_idx in cluster:
            obj_mask = obj_mask | masks[mask_idx]
            if bboxes is not None:
                cluster_box = bboxes[mask_idx].copy()
                obj_box = merge_boxes(obj_box, cluster_box)
        obj_masks.append(obj_mask)
        boxes.append(obj_box)
    if bboxes is None:
        return np.array(obj_masks)
    return np.array(obj_masks), np.array(boxes)


def visualize_clustering(G, cv_image, masks, bboxes, clusters, ax, scale=100.0):
    import matplotlib
    print(clusters)
    print(len(masks))
    ax.clear()
    masks = masks.copy()
    bboxes = bboxes.copy()
    masks, bboxes = create_obj_masks(masks, bboxes, clusters)

    node_images = []
    for node, mask in enumerate(masks):
        masked_img = apply_mask(
            cv_image, mask)
        masked_img = crop_image(masked_img, bboxes[node])
        node_images.append(masked_img)
    print(len(masks))
    # Use Kamada-Kawai layout
    pos = nx.kamada_kawai_layout(G, weight='weight.0')  # Specify the edge weight attribute (the first element)
    edge_weights = [scale*weight for _, _, weight in G.edges(data='weight')]
    edge_collection = nx.draw_networkx_edges(G, pos, edge_color=edge_weights, edge_cmap=plt.cm.inferno, edge_vmin=min(edge_weights), edge_vmax=max(edge_weights), width=2.0)
    cbar = plt.colorbar(edge_collection, label='Edge Weight')

    # Draw edges with weights as labels
    edge_labels = {(u, v): f'{data["weight"]*scale:.2f}' for u, v, data in G.edges(data=True)} 
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')

    # Draw nodes with images
    index = 0
    for node, (x, y) in pos.items():
        masked_img = node_images[index]
        img = Image.fromarray(masked_img)
        img = img.resize((50, 50))  # Adjust the size of the displayed image
        img_array = np.array(img)
        imagebox = matplotlib.offsetbox.OffsetImage(img_array, zoom=1.0, resample=True, clip_path=None)
        ab = matplotlib.offsetbox.AnnotationBbox(imagebox, (x, y), frameon=False, pad=0.0)
        ax.add_artist(ab)
        index += 1

    # # Show the plot
    # plt.axis('off')  # Hide the axis
    # plt.show()
