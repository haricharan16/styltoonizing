import os
import torch
import cv2
import numpy as np
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
from torchvision.models import vgg19

device = torch.device('gpu') if torch.cuda.is_available() else 'cpu'

# Function to initialize and load weights for the models
def load_model_weights(models_dir):
    import torch  # Ensure you have PyTorch imported
    print(f"Loading model weights from directory: {models_dir}")

    # Paths to the weight files
    best_content_encoder_path = os.path.join(models_dir, "best_content_encoder.pth")
    best_style_encoder_path = os.path.join(models_dir, "best_style_encoder.pth")
    best_decoder_path = os.path.join(models_dir, "best_decoder.pth")

    print("Checking for model files...")
    print(f"Content encoder path: {best_content_encoder_path}")
    print(f"Style encoder path: {best_style_encoder_path}")
    print(f"Decoder path: {best_decoder_path}")

    if not os.path.exists(best_content_encoder_path):
        raise FileNotFoundError(f"Content encoder weights not found: {best_content_encoder_path}")
    if not os.path.exists(best_style_encoder_path):
        raise FileNotFoundError(f"Style encoder weights not found: {best_style_encoder_path}")
    if not os.path.exists(best_decoder_path):
        raise FileNotFoundError(f"Decoder weights not found: {best_decoder_path}")

    print("Initializing VGG-style encoder...")
    content_encoder, style_encoder = vgg_style_encoder()
    print(f"Content Encoder: {content_encoder}")
    print(f"Style Encoder: {style_encoder}")

    print("Initializing Style Transfer Net...")
    decoder = style_transfer_net()
    print(f"Decoder: {decoder}")

    print("Loading weights...")
    content_encoder.load_state_dict(torch.load(best_content_encoder_path, map_location=torch.device('cpu')))
    style_encoder.load_state_dict(torch.load(best_style_encoder_path, map_location=torch.device('cpu')))
    decoder.load_state_dict(torch.load(best_decoder_path, map_location=torch.device('cpu')))

    content_encoder.eval()
    style_encoder.eval()
    decoder.eval()

    print("Model weights loaded successfully.")
    return content_encoder, style_encoder, decoder


# --- Image Processing Functions ---
def load_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at the provided path: {image_path}")
    return img

def resize_image(img, target_width=600):
    height, width = img.shape[:2]
    aspect_ratio = width / height
    target_height = int(target_width / aspect_ratio)
    return cv2.resize(img, (target_width, target_height))

def display_image(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.axis('off')
    plt.show()

def convert_to_grayscale(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.medianBlur(gray, 5)

def sharpen_image(img):
    sharpen_kernel = np.array([[-1, -1, -1],
                               [-1, 9, -1],
                               [-1, -1, -1]])
    return cv2.filter2D(img, -1, sharpen_kernel)

def detect_edges_canny(img_gray):
    edges = cv2.Canny(img_gray, threshold1=50, threshold2=150)
    return cv2.bitwise_not(edges)

def apply_bilateral_filter(img, no_of_filters=2):
    color = img.copy()
    for _ in range(no_of_filters):
        color = cv2.bilateralFilter(color, 9, 150, 150)  # Reduced smoothing
    return color

def blend_with_original(img, cartoon, alpha):
    return cv2.addWeighted(img, alpha, cartoon, 1 - alpha, 0)

def blend_edges_with_image(img, edges):
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(img, 0.9, edges_colored, 0.1, 0)

def gamma_correction(img, gamma=1.2):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(img, table)

def cartoonize_image(img, edges):
    return cv2.bitwise_and(img, img, mask=edges)

# --- Cartoonization Process ---
def process_image(image_path):
    img = load_image(image_path)
    img_resized = resize_image(img)
    print("Original Image")
    display_image(img_resized)

    # Cartoonization Steps
    gray_img = convert_to_grayscale(img_resized)
    sharpened_img = sharpen_image(img_resized)

    edges = detect_edges_canny(gray_img)
    filtered_img = apply_bilateral_filter(sharpened_img)

    # Cartoonize and blend with the original image
    cartoon_img = cartoonize_image(filtered_img, edges)
    cartoon_with_original_colors = blend_with_original(img_resized, cartoon_img, alpha=0.25)
    blended_with_edges = blend_edges_with_image(cartoon_with_original_colors, edges)

    # Enhance details with gamma correction
    enhanced_cartoon = gamma_correction(blended_with_edges, gamma=0.85)

    # Display final cartoonized image
    print("Cartoon Image")
    display_image(enhanced_cartoon)

    return enhanced_cartoon

# --- Style Transfer Network ---
def vgg_style_encoder():
    vgg = models.vgg19(pretrained=True).features
    style_features = nn.Sequential(*list(vgg.children())[:9])  # Up to relu3_1
    content_features = nn.Sequential(*list(vgg.children())[:21])  # Up to relu4_1

    for param in style_features.parameters():
        param.requires_grad = False
    for param in content_features.parameters():
        param.requires_grad = False

    return content_features, style_features

def cross_fusion_attention(high_level_features, low_level_features):
    # Ensure the inputs are PyTorch tensors
    if isinstance(high_level_features, np.ndarray):
        high_level_features = torch.tensor(high_level_features, device=device)
    if isinstance(low_level_features, np.ndarray):
        low_level_features = torch.tensor(low_level_features, device=device)

    gap = nn.AdaptiveAvgPool2d(1).to(device)  # Adaptive Average Pooling
    high_level_features_gap = gap(high_level_features)
    low_level_features_gap = gap(low_level_features)

    # Define convolution layers
    conv = nn.Conv2d(high_level_features_gap.size(1), low_level_features_gap.size(1), kernel_size=1, padding=1).to(device)
    fca = torch.sigmoid(conv(high_level_features_gap)) * low_level_features_gap

    conv_cross_attention = nn.Conv2d(fca.size(1), 512, kernel_size=1, stride=1, padding=1).to(device)
    bn_cross_attention = nn.BatchNorm2d(512).to(device)
    relu_cross_attention = nn.ReLU().to(device)
    cross_attention_features = relu_cross_attention(bn_cross_attention(conv_cross_attention(fca)))

    max_pool = nn.AdaptiveMaxPool2d(1).to(device)
    avg_pool = nn.AdaptiveAvgPool2d(1).to(device)
    max_pooled_features = max_pool(cross_attention_features)
    avg_pooled_features = avg_pool(cross_attention_features)

    combined_pooled_features = torch.cat((max_pooled_features, avg_pooled_features), dim=1)
    sigmoid_output = torch.sigmoid(combined_pooled_features)
    conv_reduce_channels = nn.Conv2d(combined_pooled_features.size(1), max_pooled_features.size(1), kernel_size=1).to(device)
    sigmoid_output = conv_reduce_channels(sigmoid_output)

    final_output = high_level_features_gap * sigmoid_output
    return final_output


def style_transfer_net():
    decoder = nn.Sequential(
        nn.Conv2d(512, 256, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(256, 128, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(128, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 3, kernel_size=3, padding=1),
        nn.Tanh()
    )
    return decoder

def discriminator():
    layers = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(512, 1),
        nn.Sigmoid()
    )
    return layers

def joint_loss_with_dft(alpha, beta, gamma, generated, content, style, disc):
    adversarial_loss = nn.BCEWithLogitsLoss()
    l1_loss = nn.L1Loss()

    real_score = disc(content)
    fake_score = disc(generated)
    loss_adv = torch.sum(torch.mean(torch.log(real_score))) + torch.sum(torch.mean(1 - torch.log( fake_score)))

    loss_l1 = l1_loss(generated, style)

    def frequency_loss(img1, img2):
        dft1 = torch.fft.fft2(img1)
        dft2 = torch.fft.fft2(img2)
        mag1 = torch.abs(dft1)
        mag2 = torch.abs(dft2)
        N,M = img1.shape[2], img1.shape[3]
        return (torch.sum(torch.abs(mag1 - mag2)**2))/(N*M)

    loss_dft = frequency_loss(generated, style)
    total_loss = alpha * loss_adv + beta * loss_l1 + gamma * loss_dft
    return total_loss
    
def save_and_show_image(tensor_image, save_path):
    # Convert the tensor to CPU for saving as image
    image = tensor_image.squeeze(0).detach().cpu().numpy()
    image = np.transpose(image, (1, 2, 0))  # Convert CHW to HWC
    image = (image * 255).clip(0, 255).astype(np.uint8)  # Rescale and cast
    print(save_path)
    Image.fromarray(image).save(save_path)
    print("SAAAAVED")
    # display_image(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    

# --- Cartoons and Style Transfer Main Function ---
def cartoon_and_style_transfer(original_image_path, style_image_path, cartoon_output_path, style_output_path, models_dir):
    """
    Process the input image, cartoonize it, and apply style transfer.

    Parameters:
    - original_image_path: str, path to the input content image.
    - style_image_path: str, path to the style image.
    - cartoon_output_path: str, path to save the cartoonized image.
    - style_output_path: str, path to save the stylized image.
    - models_dir: str, path to the directory containing model weights.
    """
    # Step 1: Load pre-trained model weights
    content_encoder, style_encoder, decoder = load_model_weights(models_dir)

    # Step 2: Cartoonize the image
    cartoon_image = process_image(original_image_path)
    cv2.imwrite(cartoon_output_path, cv2.cvtColor(cartoon_image, cv2.COLOR_RGB2BGR))
    print("Cartoonized Image:")
    display_image(cartoon_image)

    # Step 3: Preprocess cartoon image
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    print("Transformed")
    cartoon_pil = Image.fromarray(cartoon_image)
    cartoon_tensor = transform(cartoon_pil).unsqueeze(0).to(device)

    # Step 4: Preprocess style image
    style_image = Image.fromarray(load_image(style_image_path))
    style_tensor = transform(style_image).unsqueeze(0).to(device)
    print("Style tensor gen")
    # Step 5: Extract features
    content_features = content_encoder(cartoon_tensor)
    style_features = style_encoder(style_tensor)
    print("Style feat gen")
    # Step 6: Apply cross-fusion attention and decode
    fused_features = cross_fusion_attention(content_features, style_features)
    final_image = decoder(fused_features)
    print("Final image gen")
    # Step 7: Save and display the stylized image
    save_and_show_image(final_image, style_output_path)
    print("Saved")
    return cartoon_output_path, style_output_path

