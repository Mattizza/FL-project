import numpy as np
import random
from PIL import Image
import cv2
from tqdm import tqdm


class StyleApplier:

    def __init__(self, L=0.1, size=(1024, 512)):
        self.styles_bank = [] #banca degli stili
        self.styles_names = [] #nome di ogni stile
        self.L = L #grandezza percentuale di metà del lato della finestra (0-1). Quella che nel paper viene chiamata Beta
        self.size = size #size (W,H) to which resize images before style transfer
        self.sizes = None
        self.cv2 = False

    def add_style_to_bank(self, style, style_name):
        self.styles_bank.append(style)
        self.styles_names.append(style_name)
    
    def preprocess(self, x):
        if isinstance(x, np.ndarray):
            x = cv2.resize(x, self.size, interpolation=cv2.INTER_CUBIC)
            self.cv2 = True
        else:
            x = x.resize(self.size, Image.BICUBIC)
        x = np.asarray(x, np.float32)
        x = x[:, :, ::-1]
        x = x.transpose((2, 0, 1))
        return x.copy()

    def deprocess(self, x, size):
        if self.cv2:
            x = cv2.resize(np.uint8(x).transpose((1, 2, 0))[:, :, ::-1], size, interpolation=cv2.INTER_CUBIC)
        else:
            x = Image.fromarray(np.uint8(x).transpose((1, 2, 0))[:, :, ::-1])
            x = x.resize(size, Image.BICUBIC)
        return x

    def apply_style(self, image):
        return self._apply_style(image)

    #given a pil image, it returns a pil image with a random style from the style bank (self.styles)
    def _apply_style(self, img):

        if len(self.styles_bank) > 0:
            n = random.randint(0, len(self.styles_bank) - 1)
            style = self.styles_bank[n]
        else:
            style = self.styles_bank[0]

        W, H = img.size
        img_np = self.preprocess(img)

        fft_np = np.fft.fft2(img_np, axes=(-2, -1))
        amp, pha = np.abs(fft_np), np.angle(fft_np)
        amp_shift = np.fft.fftshift(amp, axes=(-2, -1))
        h1, h2, w1, w2 = self.sizes
        amp_shift[:, h1:h2, w1:w2] = style
        amp_ = np.fft.ifftshift(amp_shift, axes=(-2, -1))

        fft_ = amp_ * np.exp(1j * pha)
        img_np_ = np.fft.ifft2(fft_, axes=(-2, -1))
        img_np_ = np.real(img_np_)
        img_np__ = np.clip(np.round(img_np_), 0., 255.)

        img_with_style = self.deprocess(img_np__, (W, H))

        return img_with_style
    
    def delete_styles(self):
        self.styles_bank = []
        self.styles_names = []

    def set_win_sizes(self, sizes):
        self.sizes = sizes
