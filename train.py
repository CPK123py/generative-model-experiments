import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torch.nn.utils import spectral_norm
import matplotlib.pyplot as plt
from PIL import Image
import os


class Generator(nn.Module):
    def __init__(self, noise_dim, channels=3):
        super(Generator, self).__init__()
        self.noise_dim = noise_dim  # Store noise_dim as class attribute
        self.model = nn.Sequential(
            # Initial dense layer - flatten the noise first
            nn.Flatten(),  # Add this layer to flatten the input
            nn.Linear(noise_dim, 256 * 4 * 4),
            nn.ReLU(),
            nn.Unflatten(1, (256, 4, 4)),
            
            # Rest remains the same
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            
            nn.ConvTranspose2d(32, channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

# Modified Discriminator for RGB images
class Discriminator(nn.Module):
    def __init__(self, channels=3):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            # Convolutional layers
            nn.Conv2d(channels, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            
            # Flatten and dense layer
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

def train_gan(dataloader, generator, discriminator, num_epochs, device):
    criterion = nn.BCELoss()
    optim_G = optim.Adam(generator.parameters(), lr=0.0006, betas=(0.5, 0.999))
    optim_D = optim.Adam(discriminator.parameters(), lr=0.0006, betas=(0.5, 0.999))
    
    G_losses = []
    D_losses = []
    
    for epoch in range(num_epochs):
        for batch_idx, real_images in enumerate(dataloader):
            batch_size = real_images.size(0)
            real_images = real_images.to(device)
            
            # Train Discriminator
            optim_D.zero_grad()
            label_real = torch.ones(batch_size, 1).to(device)
            label_fake = torch.zeros(batch_size, 1).to(device)
            
            output_real = discriminator(real_images)
            d_real_loss = criterion(output_real, label_real)
            
            # Modified noise generation
            noise = torch.randn(batch_size, noise_dim).to(device)  # Remove the extra dimensions
            fake_images = generator(noise)
            output_fake = discriminator(fake_images.detach())
            d_fake_loss = criterion(output_fake, label_fake)
            
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            optim_D.step()
            
            # Train Generator
            optim_G.zero_grad()
            output_fake = discriminator(fake_images)
            g_loss = criterion(output_fake, label_real)
            g_loss.backward()
            optim_G.step()
            
            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch}/{num_epochs}] Batch [{batch_idx}/{len(dataloader)}] "
                      f"D_loss: {d_loss.item():.4f} G_loss: {g_loss.item():.4f}")
            
            G_losses.append(g_loss.item())
            D_losses.append(d_loss.item())
        
        if epoch % 5 == 0:
            save_sample_images(generator, epoch, device)
    
    return G_losses, D_losses

# Hyperparameters
noise_dim = 100
image_size = 64
batch_size = 32
num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load MNIST dataset
# Replace the ImageFolderDataset class and dataset loading part with this:

# Simple transform for color images
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# Load images directly
image_dir = "char/jenna"  # your image directory
image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
dataset = []

for img_file in image_files:
    img_path = os.path.join(image_dir, img_file)
    image = Image.open(img_path).convert('RGB')
    if transform:
        image = transform(image)
    dataset.append(image)

dataset = torch.stack(dataset)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize models
generator = Generator(noise_dim).to(device)
discriminator = Discriminator().to(device)

# Loss and optimizers
criterion = nn.BCELoss()
optim_G = optim.Adam(generator.parameters(), lr=0.0006, betas=(0.5, 0.999))
optim_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))


def save_sample_images(generator, epoch, device):
    generator.eval()
    with torch.no_grad():
        # Modified noise generation
        sample_noise = torch.randn(16, noise_dim).to(device)  # Remove extra dimensions
        fake_images = generator(sample_noise)
        # Rescale images to 0-1 range
        fake_images = (fake_images + 1) / 2.0
        fake_images = fake_images.cpu()
        
        fig, axs = plt.subplots(4, 4, figsize=(8, 8))
        for i in range(4):
            for j in range(4):
                # Move channels to the end for plotting
                img = fake_images[i*4 + j].permute(1, 2, 0)
                axs[i, j].imshow(img)
                axs[i, j].axis('off')
        
        plt.savefig(f'fake_images_epoch_{epoch}.png')
        plt.close()
    generator.train()

# Train the GAN
if __name__ == "__main__":
    G_losses, D_losses = train_gan(dataloader, generator, discriminator, num_epochs, device)
    
    # Plot loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(G_losses, label='Generator Loss')
    plt.plot(D_losses, label='Discriminator Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('gan_loss_curves.png')
    plt.close()