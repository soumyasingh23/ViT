import argparse
import sys
import torch
from PIL import Image
from torchvision import transforms
import numpy as np
import cv2

from vit_rollout import VITAttentionRollout
from vit_grad_rollout import VITAttentionGradRollout

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cuda', action='store_true', default=True,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image_path', type=str, default='./examples/both.png',
                        help='Input image path')
    parser.add_argument('--head_fusion', type=str, default='max',
                        help='How to fuse the attention heads for attention rollout. \
                        Can be mean/max/min')
    parser.add_argument('--discard_ratio', type=float, default=0.9,
                        help='How many of the lowest 14x14 attention paths should we discard')
    parser.add_argument('--category_index', type=int, default=None,
                        help='The category index for gradient rollout')
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print("Using GPU")
    else:
        print("Using CPU")

    return args

def show_mask_on_image(img, mask):
    img = np.float32(img) / 255
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

if __name__ == '__main__':
    args = get_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)   
    
    model = ViT(
    image_size=64,
    patch_size=4,
    output_size=64,
    dim=1024,
    depth=4,
    heads=16,
    mlp_dim=2048,
    channels=2,
    dropout=0.1,
    emb_dropout=0.1
    ).to(device)

    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    checkpoint_path = f'checkpoint/model_epoch_60.pt'
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    true_dir = 'data/true'
    pred_dir = 'data/pred'
    os.makedirs(true_dir, exist_ok=True)
    os.makedirs(pred_dir, exist_ok=True)
    y_pred = np.load('y_pred.npy')
    y_true = np.load('y_true.npy')

    for i in range(y_pred.shape[0]):
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Save predicted image
        pred_path = os.path.join(pred_dir, f'predicted_{i}.png')
        plt.imsave(pred_path, (y_pred[i].cpu().detach().numpy()).squeeze())
        
        # Save true image
        true_path = os.path.join(true_dir, f'true_{i}.png')
        plt.imsave(true_path, (y_true[i].cpu().detach().numpy()).squeeze())
        
        plt.close(fig)
        img = Image.open(args.image_path)
        img = img.resize((224, 224))
        input_tensor = transform(img).unsqueeze(0)
        if args.use_cuda:
            input_tensor = input_tensor.cuda()

        if args.category_index is None:
            print("Doing Attention Rollout")
            attention_rollout = VITAttentionRollout(model, head_fusion=args.head_fusion, 
                discard_ratio=args.discard_ratio)
            mask = attention_rollout(input_tensor)
            name = "attention_rollout_{:.3f}_{}.png".format(args.discard_ratio, args.head_fusion)
        else:
            print("Doing Gradient Attention Rollout")
            grad_rollout = VITAttentionGradRollout(model, discard_ratio=args.discard_ratio)
            mask = grad_rollout(input_tensor, args.category_index)
            name = "grad_rollout_{}_{:.3f}_{}.png".format(args.category_index,
                args.discard_ratio, args.head_fusion)


        np_img = np.array(img)[:, :, ::-1]
        mask = cv2.resize(mask, (np_img.shape[1], np_img.shape[0]))
        mask = show_mask_on_image(np_img, mask)
        cv2.imshow("Input Image", np_img)
        cv2.imshow(name, mask)
        cv2.imwrite("input.png", np_img)
        cv2.imwrite(name, mask)
        cv2.waitKey(-1)