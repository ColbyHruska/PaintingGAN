from PIL import Image
from matplotlib import pyplot
import os
import numpy as np

def save_image(arr, path=os.path.join(os.path.dirname(__file__), f"./out/")):
	with Image.fromarray((arr * 127.5 + 127.5).astype(np.uint8)) as im:
		dir = path
		n = len(os.listdir(dir))
		im.save(os.path.join(dir, f"{n}.png"))

def save_plot(examples, n=4, path=os.path.join(os.path.dirname(__file__), f"./out/")):
	examples = examples * 0.5 + 0.5
	file = os.path.join(path, f"{len(os.listdir(path))}.png")
	for i in range(n * n):
		pyplot.subplot(n, n, 1 + i)
		pyplot.axis('off')
		pyplot.imshow(examples[i])
	pyplot.savefig(file, dpi = 130)
	pyplot.close('all')
	pyplot.close()

def save_graphs(batches, d_loss, g_loss, acc, fid):
	batches = range(1, batches + 1)
	pyplot.plot(batches, d_loss, 'r', label='Discriminator Loss')
	pyplot.plot(batches, g_loss, 'g', label='Generator Loss')

	pyplot.title('GAN Training')
	pyplot.xlabel('Batches')
	pyplot.ylabel('Loss')

	
	pyplot.plot(batches, g_loss, 'b', label='Accuracy')
	pyplot.plot(batches, g_loss, '#000000', label='Accuracy')