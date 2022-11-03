import numpy as np
import cv2 as cv

class Retinex(object):

    def get_ksize(self,sigma):
        return int(((sigma - 0.8) / 0.15) + 2.0)

    def get_gaussian_blur(self,img, ksize=0, sigma=5):
        if ksize == 0:
            ksize = self.get_ksize(sigma)
        sep_k = cv.getGaussianKernel(ksize, sigma)
        return cv.filter2D(img, -1, np.outer(sep_k, sep_k))

    def ssr(self,img, sigma):
        result = np.log10(img) - np.log10(self.get_gaussian_blur(img, ksize=0, sigma=sigma) + 1.0)
        return result

    def msr(self, img, sigma_scales=[15, 80, 250]):
        msr = np.zeros(img.shape)
        for sigma in sigma_scales:
            msr = msr + self.ssr(img, sigma)
        msr = msr / len(sigma_scales)
        return msr

    def color_balance(self, img, low_per, high_per):
        tot_pix = img.shape[1] * img.shape[0]
        low_count = tot_pix * low_per / 100
        high_count = tot_pix * (100 - high_per) / 100

        ch_list = []
        if len(img.shape) == 2:
            ch_list = [img]
        else:
            ch_list = cv.split(img)

        cs_img = []
        for i in range(len(ch_list)):
            ch = ch_list[i]
            cum_hist_sum = np.cumsum(cv.calcHist([ch], [0], None, [256], (0, 256)))
            li, hi = np.searchsorted(cum_hist_sum, (low_count, high_count))
            if (li == hi):
                cs_img.append(ch)
                continue
            lut = np.array([0 if i < li
                            else (255 if i > hi else round((i - li) / (hi - li) * 255))
                            for i in np.arange(0, 256)], dtype='uint8')
            cs_ch = cv.LUT(ch, lut)
            cs_img.append(cs_ch)

        if len(cs_img) == 1:
            return np.squeeze(cs_img)
        elif len(cs_img) > 1:
            return cv.merge(cs_img)
        return None

    def msrcr(self, image, sigma_scales=[15, 80, 250], alpha=125, beta=46, G=192, b=-30, low_per=1, high_per=1):
        img2 = np.array(image, dtype=np.float64) + 1.0
        msr_img = self.msr(img2, sigma_scales)
        crf = beta * (np.log10(alpha * img2) - np.log10(np.sum(img2, axis=2, keepdims=True)))
        msrcr = G * (msr_img * crf - b)
        msrcr = cv.normalize(msrcr, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8UC3)
        msrcr = self.color_balance(msrcr, low_per, high_per)
        return msrcr