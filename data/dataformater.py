from PIL import Image

n = 79400
for i in range(79400 + 1):
	print(i)
	with Image.open(f"./data/{i}.png") as im:
		background = Image.new("RGB", (128, 128), (0, 0, 0))
		background.paste(im)
		background.save(f"./dataNew/{i}.png")