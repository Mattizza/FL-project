import numpy as np
from PIL import Image
import cv2
from tqdm import tqdm

class StyleExtractor:

    def __init__(self, dataset):
        self.dataset = dataset
        self.avg_style = None #avg style
        self.n_images_per_style  = len(self.dataset) #num. di immagini usate per calcolare lo stile medio
        
        self.L = None #grandezza percentuale di metà del lato della finestra (0-1). Quella che nel paper viene chiamata Beta
        self.size = (1024, 512) #size (W,H) to which resize images before style transfer
        self.sizes = None #Coordinate della finestra
        self.b = None #num di pxls = metà del lato della finestra
                    #b == 0 --> 1x1, b == 1 --> 3x3, b == 2 --> 5x5, ...

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

    def extract_avg_style(self, b, L = 0.1):
        
        self.L = L
        
        if b != None:
            self.b = b

        if self.n_images_per_style < 0:
            return
        
        styles = [] #collect the style of each image

        #Questo ciclo estrae lo stile da un numero = self.n_images_per_style immagini
        for sample, _ in tqdm(self.dataset, total=self.n_images_per_style):
            
            image = self.preprocess(sample)
            styles.append(self._extract_style(image))

        #Qui viene calcolato lo stile medio
        if self.n_images_per_style > 1:
            styles = np.stack(styles, axis=0)
            style = np.mean(styles, axis=0)
            self.avg_style = style

        elif self.n_images_per_style == 1:
            self.avg_style = styles[0]

        return self.avg_style, self.sizes

    # Data una singola immagine ne estrae lo stile
    def _extract_style(self, img_np):
        fft_np = np.fft.fft2(img_np, axes=(-2, -1))
        amp = np.abs(fft_np)
        amp_shift = np.fft.fftshift(amp, axes=(-2, -1))
        if self.sizes is None:
            self.sizes = self.compute_size(amp_shift)
        h1, h2, w1, w2 = self.sizes
        style = amp_shift[:, h1:h2, w1:w2]
        return style
    
    #Restituisce gli indici riga e colonna della finestra a cui applicare FDA
    def compute_size(self, amp_shift):
        _, h, w = amp_shift.shape
        b = (np.floor(np.amin((h, w)) * self.L)).astype(int) if self.b is None else self.b
        c_h = np.floor(h / 2.0).astype(int)
        c_w = np.floor(w / 2.0).astype(int)
        h1 = c_h - b
        h2 = c_h + b + 1
        w1 = c_w - b
        w2 = c_w + b + 1
        return h1, h2, w1, w2
    
    def get_sizes(self):
        if self.sizes != None:
            return self.sizes
        else:
            print("Error!")
