import numpy as np

class PhotoZMetrics:
    def __init__(self, z_spec, z_phot):
        self.z_spec = np.asarray(z_spec)
        self.z_phot = np.asarray(z_phot)
        self.dz = (self.z_spec-self.z_phot) / (1 + self.z_spec)
    def delta(self,threshold=0.3):
        return np.count_nonzero(abs(self.dz)<0.3)/len(self.dz)
        
    def bias(self):
       return np.mean(self.z_spec-self.z_phot)
    def sigma(self):
        return np.std(self.dz)

    def sigma_nmad(self):
        return 1.48 *  np.median(abs(self.dz))
    def outlier_fraction(self, threshold=0.15):
        return np.mean(np.abs(self.dz) > threshold)
    def summary(self):
        return {
            "bias": self.bias(),
            "sigma": self.sigma(),
            "sigma_NMAD": self.sigma_nmad(),
            "delta(0.3)": self.delta(),
            "outlier_fraction(0.15)": self.outlier_fraction()
        }

