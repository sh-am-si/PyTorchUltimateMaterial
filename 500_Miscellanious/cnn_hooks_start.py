# %% packages
from torchvision import transforms
from PIL import Image

# %% Data Prep
image_path = "kiki.jpg"
image = Image.open(image_path)
transformations = transforms.Compose(
    [transforms.Resize((224, 224)), transforms.ToTensor()]
)

X = transformations(image).unsqueeze(0)
X.shape
# %%
