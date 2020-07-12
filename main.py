import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image
import argparse


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('-c', '--content', type=str, default='images/waterfall.jpg', help='Content dir')
  parser.add_argument('-s', '--style', type=str, default='images/paint.jpg', help='Style dir')
  parser.add_argument('-o', '--output', type=str, default='output/generated.png', help='Output dir')
  args = parser.parse_args()

  class vgg(nn.Module):
    def __init__(self):
        super(vgg, self).__init__()
        self.layer_no = ['0', '5', '18', '19', '28']
        self.model = models.vgg19(pretrained=True).features[:29]

    def forward(self, x):
        features = []
        for i, layer in enumerate(self.model):
            x = layer(x)

            if str(i) in self.layer_no:
                features.append(x)

        return features

  def load_image(loader, imag):
      image = Image.open(imag)
      image = loader(image).unsqueeze(0)
      image = image.to(device)
      return image

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  w, h = 300, 225
  loader = transforms.Compose([
        transforms.Resize((w, h)),
        transforms.ToTensor()
    ])

  content = load_image(loader, args.content)
  style = load_image(loader, args.style)

  model = vgg().to(device).eval()
  generated = content.clone().requires_grad_(True)

  total_step = 7000
  alpha = 1
  beta = 0.01

  optimizer = optim.Adam([generated], lr=1e-3)

  for step in range(total_step):
    generated_feature = model(generated)
    content_feature = model(content)
    style_feature = model(style)

    style_loss = content_loss = 0

    for gen, cont, sty in zip(generated_feature, content_feature, style_feature): 
        batch_size, channel, height, width = gen.shape

        content_loss += torch.mean((gen - cont)**2)
        G = gen.view(channel, height*width).mm(gen.view(channel, height*width).t())
        A = sty.view(channel, height*width).mm(sty.view(channel, height*width).t())

        style_loss += torch.mean((G-A)**2)
    
    total_loss = alpha*content_loss + beta*style_loss
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if step % 200 ==0:
        print(total_loss)
        save_image(generated, args.output)