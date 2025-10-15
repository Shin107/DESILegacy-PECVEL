import matplotlib.pyplot as plt
import numpy as np
import logging
from catboost import CatBoostRegressor  
model = CatBoostRegressor()  
from catboost_cv import PhotoZCatBoostPipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from matplotlib.colors import LogNorm
from Metrics import PhotoZMetrics  # Your custom metrics class
import argparse
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
parser = argparse.ArgumentParser(description="CatBoost PhotoZ Pipeline")
parser.add_argument("--gpu", action="store_true", default=False, help="Use GPU if available")
parser.add_argument("--mode", choices=['train', 'continue', 'evaluate'], default='train',
                    help="Pipeline mode")
parser.add_argument("--model_path", type=str, help="Path to saved model")
parser.add_argument("--iterations", type=int, default=5000, help="Number of iterations")
parser.add_argument("--depth", type=int, default=6, help="Tree depth")
parser.add_argument("--learning_rate", type=float, default=0.01, help="Learning rate")
parser.add_argument('--plot',type=str,default=None,help='Plot type: 2d_hist_valid, 2d_hist_test, 2d_hist_complete')
args = parser.parse_args()
pipeline = PhotoZCatBoostPipeline(use_gpu=args.gpu)

DR1, legacy = pipeline.load_astronomical_data()
legacy = pipeline.compute_magnitudes(legacy)
legacy_filtered, DR1_filtered = pipeline.filter_data(legacy, DR1)
X, y = pipeline.prepare_ml_data(legacy_filtered, DR1_filtered)
pipeline.create_data_splits(X, y)


class LegacyPhotoz_metric:
    def __init__(self):
        self.Z_legacy = None
        self.Z_DR1 = None
    def redshift_extraction(self, legacy, DR1):
        self.Z_legacy = legacy['Z_PHOT_MEAN']
        self.Z_DR1 = DR1['Z'] 
    def plot_2d_hist(self,bins=50, save_path='legacy_2d_hist.png'):
        """Plot 2D histogram"""
        x = self.Z_DR1
        y = self.Z_legacy
        hist, xedges, yedges = np.histogram2d(x, y, bins=bins)
        plt.figure(figsize=(8, 6))
        plt.imshow(hist.T, origin='lower', aspect='auto', 
                   extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                   norm=LogNorm(), cmap='viridis')
        plt.colorbar(label='Counts')
        plt.xlabel('Spectroscopic Redshift (Z_DR1)')
        plt.ylabel('Photometric Redshift (Z_legacy)')
        plt.title('2D Histogram of Legacy vs DR1 Redshifts')
        plt.plot([0, max(xedges[-1], yedges[-1])], [0, max(xedges[-1], yedges[-1])], 'r--')
        plt.savefig(save_path)
        plt.close()
        logging.info(f"2D histogram saved to {save_path}")
    def evaluate(self):
        logger.info("Evaluating Legacy PhotoZ performance")
        results={}
        results={'MAE': mean_absolute_error(self.Z_legacy, self.Z_DR1),
                         'MSE': mean_squared_error(self.Z_legacy, self.Z_DR1),
                         'R2': r2_score(self.Z_legacy, self.Z_DR1)}
        logger.info(f"Evaluation results: {results}")
        metrics = PhotoZMetrics(self.Z_DR1, self.Z_legacy)
        print('PhotoZ Metrics Summary:', metrics.summary())

        




metric=LegacyPhotoz_metric()
metric.redshift_extraction(legacy_filtered, DR1_filtered)
#
# 
# 












metric.plot_2d_hist(save_path='legacy_2d_hist.png')
metric.evaluate()
# Just evaluate existing model
# pipeline.load_pretrained_model(args.model_path)
# pipeline.evaluate_model()
# if args.plot == '2d_hist_valid':
#     pipeline.plot_2d_hist_validation()
# elif args.plot == '2d_hist_test':
#     pipeline.plot_2d_hist_test()
# elif args.plot == '2d_hist_complete':
#     pipeline.plot_2d_hist_complete()
