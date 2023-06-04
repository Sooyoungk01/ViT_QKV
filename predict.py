import config
import ImageTransformer

import torch
import torch.nn as nn
import numpy as np
from PIL import Image 
import albumentations as alb

import torch.cuda.nvtx as nvtx

def predict(image_path):
    model = ImageTransformer.ViT(
        patch_height = 16,
        patch_width = 16,
        embedding_dims = 768,
        dropout = 0.1,
        heads = 4,
        num_layers = 4,
        forward_expansion = 4,
        max_len = int((32*32)/(16*16)),
       # max_len = int((512*512)/(16*16)), 
        layer_norm_eps = 1e-5,
        num_classes = 10,
    )

    model.load_state_dict(torch.load(config.Model_Path))
    model.eval()

    image = np.array(Image.open(image_path).convert('RGB'))
    transform = alb.Compose([
        alb.Resize(config.image_height, config.image_width, always_apply=True),
        alb.Normalize(config.mean, config.std, always_apply=True)
    ])

    image = transform(image=image)['image']
    
    image = torch.tensor(image, dtype=torch.float)
    image = image.permute(2,0,1)
    patches = image.reshape(-1, image.shape[0], image.shape[1], image.shape[2])

    idx_to_class = {
        0: 'airplane',
        1: 'automobile',
        2: 'bird',
        3: 'cat',
        4: 'deer',
        5: 'dog',
        6: 'frog',
        7: 'horse',
        8: 'ship',
        9: 'truck'
    }
    for i in range(11):
        with torch.no_grad():
            nvtx.range_push("Forward " + str(i))
            output = model(patches)
            nvtx.range_pop()
    
    prediction_class = torch.softmax(output, dim=-1)[0].argmax(dim=-1).item()
    prediction = idx_to_class[prediction_class]
    print(f'THE IMAGE CONTAINS A {prediction.upper()}')

if __name__ == "__main__":
    image_path = './cifar10/test/dog/blenheim_spaniel_s_000431.png'
    predict(image_path)

