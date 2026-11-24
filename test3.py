import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.nn.utils import spectral_norm
import matplotlib.pyplot as plt
from PIL import Image
import os
import numpy as np

class Discriminator(nn.Module):
    def __init__(self, channels=3):
        super(Discriminator, self).__init__()
        
        def discriminator_block(in_channels, out_channels, normalize=True):
            layers = [
                spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)),
                nn.LeakyReLU(0.2),
                nn.Dropout2d(0.3)
            ]
            if normalize:
                layers.append(nn.BatchNorm2d(out_channels))  # Switched back to BatchNorm2d
            return layers

        self.model = nn.Sequential(
            *discriminator_block(channels, 32, normalize=False),  # 64x64 -> 32x32
            *discriminator_block(32, 64),                        # 32x32 -> 16x16
            *discriminator_block(64, 128),                       # 16x16 -> 8x8
            *discriminator_block(128, 256),                      # 8x8 -> 4x4
            nn.Flatten(),
            spectral_norm(nn.Linear(256 * 4 * 4, 1))
        )
        
        # Self-attention layer after the second block
        self.attention = SelfAttention(64)

    def forward(self, x):
        # Track the size of feature maps
        features = []
        for i, layer in enumerate(self.model):
            x = layer(x)
            if i == 3:  # After second conv block
                x = self.attention(x)
            features.append(x.shape if isinstance(x, torch.Tensor) else None)
        return x

class Generator(nn.Module):
    def __init__(self, noise_dim, channels=3):
        super(Generator, self).__init__()
        
        def generator_block(in_channels, out_channels):
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2)
            )

        self.project = nn.Sequential(
            nn.Linear(noise_dim, 256 * 4 * 4),
            nn.LeakyReLU(0.2)
        )

        self.main = nn.Sequential(
            generator_block(256, 128),    # 4x4 -> 8x8
            generator_block(128, 64),     # 8x8 -> 16x16
            generator_block(64, 32),      # 16x16 -> 32x32
            nn.ConvTranspose2d(32, channels, kernel_size=4, stride=2, padding=1),  # 32x32 -> 64x64
            nn.Tanh()
        )

    def forward(self, x):
        x = self.project(x)
        x = x.view(x.size(0), 256, 4, 4)
        return self.main(x)

class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query = spectral_norm(nn.Conv2d(in_channels, in_channels // 8, 1))
        self.key = spectral_norm(nn.Conv2d(in_channels, in_channels // 8, 1))
        self.value = spectral_norm(nn.Conv2d(in_channels, in_channels, 1))
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, H, W = x.size()
        
        q = self.query(x).view(batch_size, -1, H * W).permute(0, 2, 1)
        k = self.key(x).view(batch_size, -1, H * W)
        v = self.value(x).view(batch_size, -1, H * W)
        
        attention = torch.bmm(q, k)
        attention = torch.softmax(attention, dim=-1)
        
        out = torch.bmm(v, attention.permute(0, 2, 1))
        out = out.view(batch_size, C, H, W)
        
        return self.gamma * out + x


class GANTrainer:
    def __init__(self, generator, discriminator, device, noise_dim):
        self.generator = generator
        self.discriminator = discriminator
        self.device = device
        self.noise_dim = noise_dim
        
        # Wasserstein GAN with gradient penalty
        self.g_optimizer = optim.AdamW(generator.parameters(), lr=1e-4, betas=(0.5, 0.999))
        self.d_optimizer = optim.AdamW(discriminator.parameters(), lr=4e-4, betas=(0.5, 0.999))
        
        # Learning rate schedulers
        self.g_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.g_optimizer, T_max=200)
        self.d_scheduler = optim.lr_scheduler.CosineAnnealingLR(self.d_optimizer, T_max=200)

    def gradient_penalty(self, real_samples, fake_samples):
        batch_size = real_samples.size(0)
        alpha = torch.rand(batch_size, 1, 1, 1).to(self.device)
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        
        d_interpolates = self.discriminator(interpolates)
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones_like(d_interpolates),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        
        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def train_step(self, real_images, accumulation_steps=2):
        batch_size = real_images.size(0)
        real_images = real_images.to(self.device)

        # Train Discriminator
        for _ in range(2):  # Multiple D updates per G update
            self.d_optimizer.zero_grad()
            
            # Real images
            d_real = self.discriminator(real_images)
            
            # Generate fake images
            z = torch.randn(batch_size, self.noise_dim).to(self.device)
            fake_images = self.generator(z)
            d_fake = self.discriminator(fake_images.detach())
            
            # Wasserstein loss
            d_loss = -torch.mean(d_real) + torch.mean(d_fake)
            
            # Gradient penalty
            gp = self.gradient_penalty(real_images, fake_images.detach())
            d_loss = d_loss + 10.0 * gp
            
            d_loss.backward()
            self.d_optimizer.step()

        # Train Generator
        self.g_optimizer.zero_grad()
        
        fake_images = self.generator(z)
        g_loss = -torch.mean(self.discriminator(fake_images))
        
        g_loss.backward()
        self.g_optimizer.step()

        return d_loss.item(), g_loss.item()

def save_sample_images(generator, epoch, device, noise_dim, save_dir='samples'):
    os.makedirs(save_dir, exist_ok=True)
    generator.eval()
    with torch.no_grad():
        noise = torch.randn(16, noise_dim).to(device)
        fake_images = generator(noise)
        fake_images = (fake_images + 1) / 2.0
        
        fig, axs = plt.subplots(4, 4, figsize=(10, 10))
        for i in range(4):
            for j in range(4):
                img = fake_images[i*4 + j].cpu().permute(1, 2, 0)
                axs[i, j].imshow(img)
                axs[i, j].axis('off')
        
        plt.savefig(f'{save_dir}/epoch_{epoch}.png', dpi=300, bbox_inches='tight')
        plt.close()
    generator.train()
# [Previous imports remain the same...]

class ImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        
        # Check if directory exists
        if not os.path.exists(image_dir):
            raise FileNotFoundError(f"Directory not found: {image_dir}")
            
        # Get image files with case-insensitive extension check
        self.image_files = [
            f for f in os.listdir(image_dir) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp'))
        ]
        
        # Check if any images were found
        if not self.image_files:
            raise RuntimeError(f"No image files found in {image_dir}")
        
        print(f"Found {len(self.image_files)} images in {image_dir}")
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.image_files[idx])
        try:
            image = Image.open(img_name).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_name}: {str(e)}")
            # Return a blank image in case of error
            image = Image.new('RGB', (64, 64), 'black')
            
        if self.transform:
            try:
                image = self.transform(image)
            except Exception as e:
                print(f"Error applying transform to {img_name}: {str(e)}")
                # Return a blank tensor in case of error
                image = torch.zeros((3, 64, 64))
        
        return image

if __name__ == "__main__":
    # Training configuration
    noise_dim = 100
    image_size = 64
    batch_size = 32
    num_epochs = 500
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data preprocessing
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # Initialize dataset and dataloader with error handling
    image_dir = "char/Gwen Tennyson"  # your image directory
    try:
        # Print absolute path for debugging
        abs_path = os.path.abspath(image_dir)
        print(f"Looking for images in: {abs_path}")
        
        dataset = ImageDataset(image_dir=image_dir, transform=transform)
        
        if len(dataset) == 0:
            raise RuntimeError("Dataset is empty")
            
        if len(dataset) < batch_size:
            print(f"Warning: Dataset size ({len(dataset)}) is smaller than batch_size ({batch_size})")
            batch_size = len(dataset)
            print(f"Adjusted batch_size to {batch_size}")
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # Set to 0 for debugging
            pin_memory=True,
            drop_last=True
        )
        
        if len(dataloader) == 0:
            raise RuntimeError("No batches in dataloader")
            
        print(f"Dataset size: {len(dataset)} images")
        print(f"Number of batches: {len(dataloader)}")
        
    except Exception as e:
        print(f"Error setting up dataset: {str(e)}")
        print("Please check the following:")
        print("1. The directory path is correct")
        print("2. The directory contains image files")
        print("3. You have read permissions for the directory")
        print("\nCurrent working directory:", os.getcwd())
        print("Directory contents:", os.listdir() if os.path.exists('.') else "Cannot read directory")
        raise SystemExit(1)

    # Initialize models
    generator = Generator(noise_dim).to(device)
    discriminator = Discriminator().to(device)
    trainer = GANTrainer(generator, discriminator, device, noise_dim)
    
    # Training history
    history = {
        'd_losses': [],
        'g_losses': []
    }
    
    # Training loop with error handling
    try:
        for epoch in range(num_epochs):
            epoch_d_losses = []
            epoch_g_losses = []
            
            for batch_idx, real_images in enumerate(dataloader):
                d_loss, g_loss = trainer.train_step(real_images)
                epoch_d_losses.append(d_loss)
                epoch_g_losses.append(g_loss)
                
                if batch_idx % 5 == 0:
                    print(f"Epoch [{epoch}/{num_epochs}] Batch [{batch_idx}/{len(dataloader)}] "
                          f"D_loss: {d_loss:.4f} G_loss: {g_loss:.4f}")
            
            # Calculate average epoch losses with error checking
            if epoch_d_losses and epoch_g_losses:
                avg_d_loss = sum(epoch_d_losses) / len(epoch_d_losses)
                avg_g_loss = sum(epoch_g_losses) / len(epoch_g_losses)
                history['d_losses'].append(avg_d_loss)
                history['g_losses'].append(avg_g_loss)
                
                print(f"Epoch [{epoch}/{num_epochs}] "
                      f"Avg D_loss: {avg_d_loss:.4f} Avg G_loss: {avg_g_loss:.4f}")
            else:
                print(f"Warning: No losses recorded for epoch {epoch}")
            
            # Step the learning rate schedulers
            trainer.g_scheduler.step()
            trainer.d_scheduler.step()
            
            # Save sample images
            if epoch % 5 == 0:
                save_sample_images(generator, epoch, device, noise_dim)
            
            # Save models periodically
            if epoch % 50 == 0:
                torch.save({
                    'epoch': epoch,
                    'generator_state_dict': generator.state_dict(),
                    'discriminator_state_dict': discriminator.state_dict(),
                    'g_optimizer_state_dict': trainer.g_optimizer.state_dict(),
                    'd_optimizer_state_dict': trainer.d_optimizer.state_dict(),
                    'g_scheduler_state_dict': trainer.g_scheduler.state_dict(),
                    'd_scheduler_state_dict': trainer.d_scheduler.state_dict(),
                    'history': history
                }, f'checkpoint_epoch_{epoch}.pt')
                
    except Exception as e:
        print(f"Error during training: {str(e)}")
        # Save emergency checkpoint
        torch.save({
            'epoch': epoch,
            'generator_state_dict': generator.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'g_optimizer_state_dict': trainer.g_optimizer.state_dict(),
            'd_optimizer_state_dict': trainer.d_optimizer.state_dict(),
            'g_scheduler_state_dict': trainer.g_scheduler.state_dict(),
            'd_scheduler_state_dict': trainer.d_scheduler.state_dict(),
            'history': history
        }, 'emergency_checkpoint.pt')
        raise
    
    # Plot training history
    plt.figure(figsize=(10, 5))
    plt.plot(history['g_losses'], label='Generator Loss')
    plt.plot(history['d_losses'], label='Discriminator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('training_history.png')
    plt.close()