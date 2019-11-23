"""
  author: Sierkinhane
  since: 2019-11-7 19:29:06
  description: this code was based on GAN in actions(https://github.com/GANs-in-Action/gans-in-action)
"""
import torch
import torch.nn as nn
import mnist
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict
import matplotlib.pyplot as plt
import math
# np.random.seed(1)
# torch.random.manual_seed(1)

# 1. generator
class Generator(nn.Module):
	def __init__(self):
		super(Generator, self).__init__()
		self.embedding = nn.Embedding(n_class, n_class)
		self.model =  nn.Sequential(
			nn.Linear(100+n_class, 128),
			nn.ReLU(inplace=True),
			nn.Linear(128, 28*28*1),
			nn.Sigmoid()
			)

	def forward(self, labels, z):
		embedding = self.embedding(labels)
		con_info = torch.cat([embedding, z], -1)

		return self.model(con_info)
		

# 2. discriminator
class Discriminator(nn.Module):
	def __init__(self):
		super(Discriminator, self).__init__()
		self.embedding = nn.Embedding(n_class, n_class)
		self.model = nn.Sequential(
			nn.Linear(28*28*1+n_class, 128),
			nn.ReLU(inplace=True),
			nn.Linear(128, 1),
			nn.Sigmoid())
	
	def forward(self, labels, gen_imgs):
		embedding = self.embedding(labels)
		con_info = torch.cat([embedding, gen_imgs], -1)

		return self.model(con_info)
		

def sample_images(generator, iteration, device, image_grid_rows=4, image_grid_columns=4):

	labels = [r for _ in range(image_grid_columns) for r in range(image_grid_rows)]
	labels = torch.LongTensor(labels).to(device)
	z = torch.from_numpy(np.random.normal(0, 1, (image_grid_rows*image_grid_columns, 100)).astype(np.float32)).to(device)
	gen_imgs = generator(labels, z)
	# Generate images from random noise
	gen_imgs = gen_imgs.cpu().detach().numpy().reshape(-1, 28, 28)

	# Rescale image pixel values to [0, 1]
	# gen_imgs = 0.5 * gen_imgs + 0.5

	# Set image grid
	fig, axs = plt.subplots(image_grid_rows,
							image_grid_columns,
							figsize=(4, 4),
							sharey=True,
							sharex=True)

	cnt = 0
	for i in range(image_grid_rows):
		for j in range(image_grid_columns):
			# Output a grid of images
			axs[i, j].imshow(gen_imgs[cnt, :, :])
			axs[i, j].axis('off')
			plt.savefig(str(iteration+1)+'_generated.jpg')
			cnt += 1
	plt.close()

# 3. training
def train(generator, discriminator, device, iterations, batch_size, sample_interval, lr):

	generator.to(device)
	discriminator.to(device)

	# load the mnist dataset
	(X_train, Label), (_, _) = mnist.load_data()

	# rescale [0, 255] graysacle pixel values to [-1. 1]
	X_train = X_train / 255.0
	X_train = np.expand_dims(X_train, axis=3)

	# labels for real and fake images
	real = torch.from_numpy(np.ones((batch_size, 1), dtype=np.float32)).to(device)
	fake = torch.from_numpy(np.zeros((batch_size, 1), dtype=np.float32)).to(device)

	# define optimizer 
	optimizer_G = torch.optim.Adam(params=generator.parameters(), lr=lr)
	optimizer_D = torch.optim.Adam(params=discriminator.parameters(), lr=lr)
	scheduler_G = torch.optim.lr_scheduler.MultiStepLR(optimizer_G, milestones=[10000, 15000], gamma=0.1)
	scheduler_D = torch.optim.lr_scheduler.MultiStepLR(optimizer_D, milestones=[10000, 15000], gamma=0.1)
	for iteration in range(iterations):

		# train the discriminator
		idx = np.random.randint(0, X_train.shape[0], batch_size)
		imgs = X_train[idx].astype(np.float32)
		labels = torch.LongTensor(Label[idx]).to(device)
		imgs_t = torch.from_numpy(imgs).reshape(-1, 28*28).to(device)
		z = torch.from_numpy(np.random.normal(0, 1, (batch_size, 100)).astype(np.float32)).to(device)

		gen_imgs = generator(labels, z)

		d_scores_real = discriminator(labels, imgs_t)
		d_scores_fake = discriminator(labels, gen_imgs)
		
		# you can choose this loss function
		# d_loss_real = F.binary_cross_entropy_with_logits(d_scores_real, real)
		# d_loss_fake = F.binary_cross_entropy_with_logits(d_scores_fake, fake)
		# d_loss = 0.5*torch.add(d_loss_real, d_loss_fake)

		# or consistent with Goodfellow
		d_loss = -torch.mean(torch.log(d_scores_real) + torch.log(1. - d_scores_fake))

		optimizer_D.zero_grad()
		d_loss.backward()
		optimizer_D.step()

		
		#train generator
		z = torch.from_numpy(np.random.normal(0, 1, (batch_size, 100)).astype(np.float32)).to(device)
		gen_imgs = generator(labels, z)
		g_scores_real = discriminator(labels, gen_imgs)

		# you can choose this loss function
		# g_loss = F.binary_cross_entropy_with_logits(g_scores_real, real)

		# or consistent with Goodfellow
		g_loss = -torch.mean(torch.log(g_scores_real))
		# g_loss = torch.mean(torch.log(1. - g_scores_real))

		optimizer_G.zero_grad()
		g_loss.backward()
		optimizer_G.step()

		if (iteration + 1) % sample_interval == 0:

			# Output training progress
			print("iteration {0}, d_loss {1:.4f},  g_loss {2:.4f}".format(iteration+1, d_loss, g_loss))

			# Output a sample of generated image
			sample_images(generator, iteration, device)
		scheduler_D.step()
		scheduler_G.step()

if __name__ == '__main__':
	
	iterations = 20000
	lr = 0.001
	batch_size = 128
	sample_interval = 1000
	n_class = 10 # 0-9

	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	generator = Generator()
	discriminator = Discriminator()
	
	train(generator, discriminator, device, iterations, batch_size, sample_interval, lr)

