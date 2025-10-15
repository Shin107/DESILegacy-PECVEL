import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostRegressor, Pool
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy import stats
import warnings
from catboost_cv import PhotoZCatBoostPipeline
warnings.filterwarnings('ignore')

# Try to import SHAP (optional)
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Warning: SHAP not available. Some feature importance plots will be skipped.")

class CatBoostVisualizer:
    """
    A comprehensive visualization class for CatBoost regression models
    """
    
    def __init__(self, model_path=None, model=None, feature_names=None):
        """
        Initialize the visualizer with a CatBoost model
        
        Parameters:
        -----------
        model_path : str, optional
            Path to saved CatBoost model file
        model : CatBoostRegressor, optional
            Pre-loaded CatBoost model
        feature_names : list, optional
            List of feature names for better plots
        """
        if model_path is not None:
            self.model = CatBoostRegressor()
            self.model.load_model(model_path)
            print(f"Loaded model from {model_path}")
        elif model is not None:
            self.model = model
            print("Using provided model")
        else:
            raise ValueError("Either model_path or model must be provided")
        
        self.feature_names = feature_names
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.predictions_train = None
        self.predictions_val = None
        self.predictions_test = None
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
    def load_data(self, X_train, y_train, X_val=None, y_val=None, X_test=None, y_test=None):
        """
        Load training, validation, and test data
        
        Parameters:
        -----------
        X_train, y_train : array-like
            Training features and targets
        X_val, y_val : array-like, optional
            Validation features and targets
        X_test, y_test : array-like, optional
            Test features and targets
        """
        self.X_train = np.array(X_train)
        self.y_train = np.array(y_train)
        
        if X_val is not None and y_val is not None:
            self.X_val = np.array(X_val)
            self.y_val = np.array(y_val)
            
        if X_test is not None and y_test is not None:
            self.X_test = np.array(X_test)
            self.y_test = np.array(y_test)
            
        # Generate predictions
        self._generate_predictions()
        
    def _generate_predictions(self):
        """Generate predictions for all loaded datasets"""
        if self.X_train is not None:
            self.predictions_train = self.model.predict(self.X_train)
            
        if self.X_val is not None:
            self.predictions_val = self.model.predict(self.X_val)
            
        if self.X_test is not None:
            self.predictions_test = self.model.predict(self.X_test)
    
    def plot_training_metrics(self, figsize=(15, 5)):
        """
        Plot training metrics from the model (loss, learning rate, etc.)
        """
        try:
            # Get training history
            evals_result = self.model.get_evals_result()
            
            fig, axes = plt.subplots(1, 3, figsize=figsize)
            
            # Plot 1: Training loss
            if 'learn' in evals_result:
                train_metric = list(evals_result['learn'].values())[0]
                axes[0].plot(train_metric, label='Train', linewidth=2)
                
            if 'validation' in evals_result:
                val_metric = list(evals_result['validation'].values())[0]
                axes[0].plot(val_metric, label='Validation', linewidth=2)
                
            axes[0].set_xlabel('Iterations')
            axes[0].set_ylabel('RMSE')
            axes[0].set_title('Training History')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Plot 2: Feature importances
            feature_importance = self.model.get_feature_importance()
            feature_names = self.feature_names or [f'Feature_{i}' for i in range(len(feature_importance))]
            
            axes[1].barh(range(len(feature_importance)), feature_importance)
            axes[1].set_yticks(range(len(feature_importance)))
            axes[1].set_yticklabels(feature_names)
            axes[1].set_xlabel('Importance')
            axes[1].set_title('Feature Importance')
            
            # Plot 3: Model parameters
            params = self.model.get_all_params()
            key_params = ['iterations', 'depth', 'learning_rate', 'l2_leaf_reg']
            param_values = [params.get(p, 0) for p in key_params]
            
            axes[2].bar(key_params, param_values)
            axes[2].set_title('Key Hyperparameters')
            axes[2].tick_params(axis='x', rotation=45)
            plt.savefig('metrics_1.png')
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Could not plot training metrics: {e}")
    
    def plot_predictions_vs_actual(self, figsize=(15, 5)):
        """
        Plot predictions vs actual values for all datasets
        """
        datasets = []
        if self.y_train is not None and self.predictions_train is not None:
            datasets.append(('Train', self.y_train, self.predictions_train))
        if self.y_val is not None and self.predictions_val is not None:
            datasets.append(('Validation', self.y_val, self.predictions_val))
        if self.y_test is not None and self.predictions_test is not None:
            datasets.append(('Test', self.y_test, self.predictions_test))
        
        n_datasets = len(datasets)
        if n_datasets == 0:
            print("No data loaded. Use load_data() first.")
            return
        
        fig, axes = plt.subplots(1, n_datasets, figsize=(5*n_datasets, 5))
        if n_datasets == 1:
            axes = [axes]
        
        for i, (name, y_true, y_pred) in enumerate(datasets):
            # Scatter plot
            axes[i].scatter(y_true, y_pred, alpha=0.6, s=20)
            
            # Perfect prediction line
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            axes[i].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            
            axes[i].set_xlabel('True Values')
            axes[i].set_ylabel('Predictions')
            axes[i].set_title(f'{name} Set\nRMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        plt.savefig('acvsreal_1.png')
        plt.tight_layout()
        plt.show()
    
    def plot_residuals(self, figsize=(15, 10)):
        """
        Plot residual analysis for all datasets
        """
        datasets = []
        if self.y_train is not None and self.predictions_train is not None:
            datasets.append(('Train', self.y_train, self.predictions_train))
        if self.y_val is not None and self.predictions_val is not None:
            datasets.append(('Validation', self.y_val, self.predictions_val))
        if self.y_test is not None and self.predictions_test is not None:
            datasets.append(('Test', self.y_test, self.predictions_test))
        
        n_datasets = len(datasets)
        if n_datasets == 0:
            print("No data loaded. Use load_data() first.")
            return
        
        fig, axes = plt.subplots(2, n_datasets, figsize=(5*n_datasets, 10))
        if n_datasets == 1:
            axes = axes.reshape(-1, 1)
        
        for i, (name, y_true, y_pred) in enumerate(datasets):
            residuals = y_true - y_pred
            
            # Residuals vs predictions
            axes[0, i].scatter(y_pred, residuals, alpha=0.6, s=20)
            axes[0, i].axhline(y=0, color='r', linestyle='--')
            axes[0, i].set_xlabel('Predictions')
            axes[0, i].set_ylabel('Residuals')
            axes[0, i].set_title(f'{name} Set - Residuals vs Predictions')
            axes[0, i].grid(True, alpha=0.3)
            
            # Histogram of residuals
            axes[1, i].hist(residuals, bins=50, alpha=0.7, density=True)
            
            # Normal distribution overlay
            mu, sigma = stats.norm.fit(residuals)
            x = np.linspace(residuals.min(), residuals.max(), 100)
            axes[1, i].plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label=f'Normal (μ={mu:.3f}, σ={sigma:.3f})')
            
            axes[1, i].set_xlabel('Residuals')
            axes[1, i].set_ylabel('Density')
            axes[1, i].set_title(f'{name} Set - Residual Distribution')
            axes[1, i].legend()
            axes[1, i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('residual_1.png')
        plt.show()
    
    def plot_feature_importance(self, figsize=(12, 8), top_n=None):
        """
        Plot detailed feature importance analysis
        """
        # Get different types of importance
        importance_types = ['PredictionValuesChange', 'LossFunctionChange', 'FeatureImportance']
        
        fig, axes = plt.subplots(1, len(importance_types), figsize=figsize)
        if len(importance_types) == 1:
            axes = [axes]
        
        feature_names = self.feature_names or [f'Feature_{i}' for i in range(self.model.get_feature_count())]
        
        for i, imp_type in enumerate(importance_types):
            try:
                if imp_type == 'FeatureImportance':
                    importance = self.model.get_feature_importance()
                else:
                    # For pool-based importance, we need data
                    if self.X_train is not None:
                        pool = Pool(self.X_train, self.y_train)
                        importance = self.model.get_feature_importance(pool, type=imp_type)
                    else:
                        continue
                
                # Sort by importance
                sorted_idx = np.argsort(importance)[::-1]
                if top_n is not None:
                    sorted_idx = sorted_idx[:top_n]
                
                sorted_importance = importance[sorted_idx]
                sorted_features = [feature_names[j] for j in sorted_idx]
                
                # Plot
                y_pos = np.arange(len(sorted_features))
                axes[i].barh(y_pos, sorted_importance)
                axes[i].set_yticks(y_pos)
                axes[i].set_yticklabels(sorted_features)
                axes[i].set_xlabel('Importance')
                axes[i].set_title(f'{imp_type}')
                axes[i].grid(True, alpha=0.3)
                
            except Exception as e:
                print(f"Could not compute {imp_type}: {e}")
                axes[i].text(0.5, 0.5, f'Error: {imp_type}\nnot available', 
                           transform=axes[i].transAxes, ha='center', va='center')
        plt.savefig('feature_1.png')
        plt.tight_layout()
        plt.show()
    
    def plot_shap_analysis(self, figsize=(15, 10)):
        """
        Plot SHAP (SHapley Additive exPlanations) analysis
        """
        if not SHAP_AVAILABLE:
            print("SHAP is not installed. Please install with: pip install shap")
            return
        
        if self.X_train is None:
            print("No training data loaded. Use load_data() first.")
            return
        
        try:
            # Create SHAP explainer
            explainer = shap.Explainer(self.model)
            
            # Use a sample of data for speed (SHAP can be slow)
            sample_size = min(1000, len(self.X_train))
            sample_idx = np.random.choice(len(self.X_train), sample_size, replace=False)
            X_sample = self.X_train[sample_idx]
            
            # Calculate SHAP values
            print("Calculating SHAP values (this may take a while)...")
            shap_values = explainer(X_sample)
            
            # Create subplots
            fig, axes = plt.subplots(2, 2, figsize=figsize)
            
            # Summary plot
            plt.subplot(2, 2, 1)
            shap.summary_plot(shap_values, X_sample, feature_names=self.feature_names, show=False)
            plt.title('SHAP Summary Plot')
            
            # Bar plot
            plt.subplot(2, 2, 2)
            shap.summary_plot(shap_values, X_sample, feature_names=self.feature_names, plot_type="bar", show=False)
            plt.title('SHAP Feature Importance')
            
            # Waterfall plot for first instance
            plt.subplot(2, 2, 3)
            shap.waterfall_plot(shap_values[0], show=False)
            plt.title('SHAP Waterfall (First Instance)')
            
            # Partial dependence plot for most important feature
            plt.subplot(2, 2, 4)
            feature_importance = np.abs(shap_values.values).mean(0)
            most_important_feature = np.argmax(feature_importance)
            
            shap.partial_dependence_plot(
                most_important_feature, self.model.predict, X_sample,
                feature_names=self.feature_names, show=False
            )
            plt.title(f'Partial Dependence - {self.feature_names[most_important_feature] if self.feature_names else f"Feature {most_important_feature}"}')
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"SHAP analysis failed: {e}")
    
    def plot_learning_curves(self, figsize=(12, 8)):
        """
        Plot learning curves to analyze overfitting/underfitting
        """
        try:
            evals_result = self.model.get_evals_result()
            
            fig, axes = plt.subplots(1, 2, figsize=figsize)
            
            # Plot 1: Loss curves
            iterations = range(len(list(evals_result['learn'].values())[0]))
            
            if 'learn' in evals_result:
                train_metric = list(evals_result['learn'].values())[0]
                axes[0].plot(iterations, train_metric, label='Training Loss', linewidth=2)
                
            if 'validation' in evals_result:
                val_metric = list(evals_result['validation'].values())[0]
                axes[0].plot(iterations, val_metric, label='Validation Loss', linewidth=2)
                
                # Mark best iteration
                best_iteration = self.model.get_best_iteration()
                if best_iteration is not None and best_iteration < len(val_metric):
                    axes[0].axvline(x=best_iteration, color='red', linestyle='--', 
                                   label=f'Best Iteration ({best_iteration})')
            
            axes[0].set_xlabel('Iterations')
            axes[0].set_ylabel('Loss (RMSE)')
            axes[0].set_title('Learning Curves')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Plot 2: Overfitting analysis
            if 'learn' in evals_result and 'validation' in evals_result:
                train_loss = np.array(list(evals_result['learn'].values())[0])
                val_loss = np.array(list(evals_result['validation'].values())[0])
                
                # Calculate gap between train and validation
                gap = val_loss - train_loss
                axes[1].plot(iterations, gap, linewidth=2, color='purple')
                axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
                axes[1].fill_between(iterations, 0, gap, alpha=0.3, color='purple')
                
                axes[1].set_xlabel('Iterations')
                axes[1].set_ylabel('Validation Loss - Training Loss')
                axes[1].set_title('Overfitting Analysis')
                axes[1].grid(True, alpha=0.3)
                
                # Add text annotation
                final_gap = gap[-1]
                if final_gap > 0.01:
                    axes[1].text(0.7, 0.9, 'Possible Overfitting', transform=axes[1].transAxes, 
                               bbox=dict(boxstyle="round,pad=0.3", facecolor="orange", alpha=0.7))
                else:
                    axes[1].text(0.7, 0.9, 'Good Generalization', transform=axes[1].transAxes,
                               bbox=dict(boxstyle="round,pad=0.3", facecolor="green", alpha=0.7))
            plt.savefig('lc_1.png')
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            print(f"Could not plot learning curves: {e}")
    
    def plot_error_analysis(self, figsize=(15, 10)):
        """
        Detailed error analysis across different ranges
        """
        if self.y_test is None or self.predictions_test is None:
            print("No test data available for error analysis.")
            return
        
        y_true = self.y_test
        y_pred = self.predictions_test
        errors = np.abs(y_true - y_pred)
        relative_errors = errors / (y_true + 1e-8)  # Avoid division by zero
        
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        
        # 1. Errors vs True Values
        axes[0, 0].scatter(y_true, errors, alpha=0.6, s=20)
        axes[0, 0].set_xlabel('True Values')
        axes[0, 0].set_ylabel('Absolute Error')
        axes[0, 0].set_title('Errors vs True Values')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Relative errors vs True Values
        axes[0, 1].scatter(y_true, relative_errors, alpha=0.6, s=20)
        axes[0, 1].set_xlabel('True Values')
        axes[0, 1].set_ylabel('Relative Error')
        axes[0, 1].set_title('Relative Errors vs True Values')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Error distribution
        axes[0, 2].hist(errors, bins=50, alpha=0.7, density=True)
        axes[0, 2].axvline(np.mean(errors), color='red', linestyle='--', label=f'Mean: {np.mean(errors):.4f}')
        axes[0, 2].axvline(np.median(errors), color='orange', linestyle='--', label=f'Median: {np.median(errors):.4f}')
        axes[0, 2].set_xlabel('Absolute Error')
        axes[0, 2].set_ylabel('Density')
        axes[0, 2].set_title('Error Distribution')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Binned analysis
        n_bins = 10
        bins = np.linspace(y_true.min(), y_true.max(), n_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        binned_errors = []
        
        for i in range(n_bins):
            mask = (y_true >= bins[i]) & (y_true < bins[i + 1])
            if mask.sum() > 0:
                binned_errors.append(np.mean(errors[mask]))
            else:
                binned_errors.append(0)
        
        axes[1, 0].bar(bin_centers, binned_errors, width=(bins[1] - bins[0]) * 0.8)
        axes[1, 0].set_xlabel('True Value Bins')
        axes[1, 0].set_ylabel('Mean Absolute Error')
        axes[1, 0].set_title('Error by Value Range')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Quantile-Quantile plot
        from scipy import stats
        residuals = y_true - y_pred
        stats.probplot(residuals, dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q Plot (Normality Check)')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Outlier analysis
        # Define outliers as points with errors > 3 standard deviations
        error_threshold = np.mean(errors) + 3 * np.std(errors)
        outliers = errors > error_threshold
        
        axes[1, 2].scatter(y_true[~outliers], y_pred[~outliers], alpha=0.6, s=20, label='Normal')
        axes[1, 2].scatter(y_true[outliers], y_pred[outliers], alpha=0.8, s=40, color='red', label=f'Outliers ({outliers.sum()})')
        
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        axes[1, 2].plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
        axes[1, 2].set_xlabel('True Values')
        axes[1, 2].set_ylabel('Predictions')
        axes[1, 2].set_title('Outlier Detection')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        plt.savefig('error_anal_1.png')
        plt.tight_layout()
        plt.show()
        
        # Print summary statistics
        print("\nError Analysis Summary:")
        print("-" * 30)
        print(f"Mean Absolute Error: {np.mean(errors):.6f}")
        print(f"Median Absolute Error: {np.median(errors):.6f}")
        print(f"Standard Deviation of Errors: {np.std(errors):.6f}")
        print(f"Max Error: {np.max(errors):.6f}")
        print(f"95th Percentile Error: {np.percentile(errors, 95):.6f}")
        print(f"Number of Outliers: {outliers.sum()} ({100*outliers.sum()/len(outliers):.2f}%)")
    
    def generate_comprehensive_report(self, save_plots=False, output_dir="./plots"):
        """
        Generate all visualization plots in one go
        """
        print("Generating comprehensive CatBoost model analysis...")
        
        if save_plots:
            import os
            os.makedirs(output_dir, exist_ok=True)
            # You could save plots here with plt.savefig()
        
        # 1. Training metrics
        print("\n1. Training Metrics...")
        self.plot_training_metrics()
        
        # 2. Predictions vs Actual
        print("\n2. Predictions vs Actual...")
        self.plot_predictions_vs_actual()
        
        # 3. Residual Analysis
        print("\n3. Residual Analysis...")
        self.plot_residuals()
        
        # 4. Feature Importance
        print("\n4. Feature Importance...")
        self.plot_feature_importance()
        
        # 5. Learning Curves
        print("\n5. Learning Curves...")
        self.plot_learning_curves()
        
        # 6. Error Analysis
        print("\n6. Detailed Error Analysis...")
        self.plot_error_analysis()
        
        # 7. SHAP Analysis (if available)
        if SHAP_AVAILABLE:
            print("\n7. SHAP Analysis...")
            self.plot_shap_analysis()
        
        print("\nComprehensive analysis complete!")


# Example usage
def example_usage():
    """
    Example of how to use the CatBoostVisualizer
    """
    # Load your model
    pipeline = PhotoZCatBoostPipeline(use_gpu=False)
    DR1, legacy = pipeline.load_astronomical_data()
    legacy = pipeline.compute_magnitudes(legacy)
    legacy_filtered, DR1_filtered = pipeline.filter_data(legacy, DR1)
    X, y = pipeline.prepare_ml_data(legacy_filtered, DR1_filtered)
    visualizer = CatBoostVisualizer(model_path="catboost_model_new.cbm", 
                                   feature_names=['MAG_GR', 'MAG_G', 'MAG_RZ', 'MAG_R', 
                                                'MAG_RW1', 'MAG_ZW1', 'MAG_W1W2'])
    
    # Load your data (X_train, y_train, X_val, y_val, X_test, y_test)
    visualizer.load_data(X, y)
    
    # Generate all plots
    visualizer.generate_comprehensive_report()
    
    # Or generate specific plots
    #visualizer.plot_predictions_vs_actual()
    # visualizer.plot_feature_importance()
    # visualizer.plot_residuals()
    
    pass

if __name__ == "__main__":
    example_usage()
