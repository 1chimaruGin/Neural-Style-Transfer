import matplotlib.pyplot as plt
import glob
from PIL import Image
import torchvision.transforms as transforms

def load_image(loader, imag):
      image = Image.open(imag)
      image = loader(image)
      return image

w, h = 300, 225
loader = transforms.Compose([
        transforms.Resize((w, h)),
    ])

fig=plt.figure(figsize=(10, 10))
columns = 3
rows = 3
for i, img in enumerate(glob.glob('./Foo/*')):
    img = load_image(loader, img)
    fig.add_subplot(rows, columns, i+1)
    if (i+1) % 3 == 1:
        plt.title('Content')
    elif (i+1) % 3 == 2:
        plt.title('Style')
    else:
        plt.title('NST')

    plt.axis('off')
    plt.imshow(img)
plt.show()