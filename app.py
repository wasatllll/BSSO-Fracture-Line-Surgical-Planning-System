import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import joblib
import os

st.set_page_config(page_title="BSSO Fracture Line Surgical Planning System", layout="wide")

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

class FractureAnalysis:
    def __init__(self, config=None):
        self.default_config = {
            'analysis_features': ['Depth  of A', 'LLBCE'],
            'grid_size': 100,
            'probability_threshold': 0.8,
            'font_family': 'Times New Roman',
            'font_size': 12,
            'figure_dpi': 300
        }
        self.config = {**self.default_config, **(config or {})}
        plt.rcParams['font.family'] = self.config['font_family']
        plt.rcParams['font.size'] = self.config['font_size']
        plt.rcParams['axes.unicode_minus'] = False
        sns.set_style("whitegrid")
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.class_names = None
        
    def load_model(self, model_path='result/best_model.pkl', 
                   scaler_path='result/scaler.pkl', 
                   encoder_path='result/label_encoder.pkl'):
        try:
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            self.label_encoder = joblib.load(encoder_path)
            self.class_names = self.label_encoder.classes_
            st.success("Model loaded successfully!")
            st.write(f"Classes: {self.class_names}")
            return True
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            return False
    
    def create_prediction_grid(self, feature_ranges, grid_size=None):
        if grid_size is None:
            grid_size = self.config['grid_size']
        feature1, feature2 = self.config['analysis_features']
        x1_min, x1_max = feature_ranges[feature1]
        x2_min, x2_max = feature_ranges[feature2]
        x1_range = np.linspace(x1_min, x1_max, grid_size)
        x2_range = np.linspace(x2_min, x2_max, grid_size)
        X1, X2 = np.meshgrid(x1_range, x2_range)
        st.write(f"Grid: {feature1} vs {feature2}, size {grid_size}x{grid_size}")
        return X1, X2, x1_range, x2_range
    
    def predict_on_grid(self, X1, X2, constant_values):
        feature1, feature2 = self.config['analysis_features']
        grid_size = X1.shape[0]
        prediction_data = []
        for i in range(grid_size):
            for j in range(grid_size):
                sample = []
                for feature in ['LLBCE', 'PMBT', 'MRT', 'Depth  of A']:
                    if feature == feature1:
                        sample.append(X1[i, j])
                    elif feature == feature2:
                        sample.append(X2[i, j])
                    else:
                        sample.append(constant_values[feature])
                prediction_data.append(sample)
        prediction_data = np.array(prediction_data)
        if self.scaler is not None:
            prediction_data = self.scaler.transform(prediction_data)
        probabilities = self.model.predict_proba(prediction_data)
        prob_grids = []
        for class_idx in range(probabilities.shape[1]):
            prob_grid = probabilities[:, class_idx].reshape(grid_size, grid_size)
            prob_grids.append(prob_grid)
        return prob_grids
    
    def plot_3d_surface(self, X1, X2, prob_grids, feature_ranges, threshold=None):
        if threshold is None:
            threshold = self.config['probability_threshold']
        feature1, feature2 = self.config['analysis_features']
        fig = plt.figure(figsize=(6*len(self.class_names), 5))
        for class_idx, class_name in enumerate(self.class_names):
            ax = fig.add_subplot(1, len(self.class_names), class_idx + 1, projection='3d')
            surf = ax.plot_surface(X1, X2, prob_grids[class_idx], 
                                  cmap='RdYlBu_r', alpha=0.8, linewidth=0, antialiased=True)
            high_prob_mask = prob_grids[class_idx] > threshold
            if np.any(high_prob_mask):
                X1_high = X1[high_prob_mask]
                X2_high = X2[high_prob_mask]
                prob_high = prob_grids[class_idx][high_prob_mask]
                ax.scatter(X1_high, X2_high, prob_high, color='red', s=50, alpha=1.0, label=f'P > {threshold}')
            ax.set_xlabel(feature1, labelpad=10)
            ax.set_ylabel(feature2, labelpad=10)
            ax.set_zlabel('Probability', labelpad=10)
            ax.set_title(f'Class {class_name}', fontsize=12, pad=10)
            ax.legend()
            ax.view_init(elev=30, azim=45)
            fig.colorbar(surf, ax=ax, shrink=0.5, aspect=15)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    
    def plot_contour_maps(self, X1, X2, prob_grids, threshold=None):
        if threshold is None:
            threshold = self.config['probability_threshold']
        feature1, feature2 = self.config['analysis_features']
        fig, axes = plt.subplots(1, len(self.class_names), figsize=(6*len(self.class_names), 5))
        if len(self.class_names) == 1:
            axes = [axes]
        for class_idx, class_name in enumerate(self.class_names):
            ax = axes[class_idx]
            contour = ax.contourf(X1, X2, prob_grids[class_idx], levels=20, cmap='RdYlBu_r', alpha=0.8)
            CS = ax.contour(X1, X2, prob_grids[class_idx], levels=[0.2, 0.4, 0.6, 0.8], colors='black', linewidths=0.5)
            ax.clabel(CS, inline=True, fontsize=8)
            high_prob_mask = prob_grids[class_idx] > threshold
            if np.any(high_prob_mask):
                X1_high = X1[high_prob_mask]
                X2_high = X2[high_prob_mask]
                ax.scatter(X1_high, X2_high, color='red', s=10, alpha=0.6, label=f'P > {threshold}')
            ax.set_xlabel(feature1)
            ax.set_ylabel(feature2)
            ax.set_title(f'Class {class_name}', fontsize=12)
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.colorbar(contour, ax=ax)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    
    def analyze_high_probability_regions(self, X1, X2, prob_grids, threshold=None):
        if threshold is None:
            threshold = self.config['probability_threshold']
        feature1, feature2 = self.config['analysis_features']
        
        st.markdown(f"**High Probability Region Analysis (Threshold: {threshold})**")
        
        n_classes = len(self.class_names)
        cols = st.columns(n_classes)
        
        for class_idx, (class_name, col) in enumerate(zip(self.class_names, cols)):
            with col:
                high_prob_mask = prob_grids[class_idx] > threshold
                high_prob_count = np.sum(high_prob_mask)
                total_points = X1.size
                
                if high_prob_count > 0:
                    high_prob_ratio = high_prob_count / total_points * 100
                    X1_high = X1[high_prob_mask]
                    X2_high = X2[high_prob_mask]
                    
                    st.markdown(f"**Class {class_name}:**")
                    st.write(f"Points: {high_prob_count}/{total_points} ({high_prob_ratio:.2f}%)")
                    st.write(f"{feature1}: [{X1_high.min():.3f}, {X1_high.max():.3f}]")
                    st.write(f"{feature2}: [{X2_high.min():.3f}, {X2_high.max():.3f}]")
                    st.write(f"{feature1} mean: {X1_high.mean():.3f}±{X1_high.std():.3f}")
                    st.write(f"{feature2} mean: {X2_high.mean():.3f}±{X2_high.std():.3f}")
                else:
                    st.markdown(f"**Class {class_name}:**")
                    st.write(f"No regions with P > {threshold}")
    
    def run_analysis(self, feature_ranges, constant_values, **kwargs):
        if kwargs:
            self.config.update(kwargs)
        if not self.load_model():
            return
        X1, X2, _, _ = self.create_prediction_grid(feature_ranges)
        prob_grids = self.predict_on_grid(X1, X2, constant_values)
        
        st.subheader("3D Probability Surfaces")
        self.plot_3d_surface(X1, X2, prob_grids, feature_ranges)
        
        st.subheader("Contour Probability Maps")
        self.plot_contour_maps(X1, X2, prob_grids)
        
        st.subheader("Statistical Analysis")
        self.analyze_high_probability_regions(X1, X2, prob_grids)
        
        st.success("Analysis completed!")


# Streamlit UI
st.title("🔬 BSSO Fracture Line Surgical Planning System")

st.sidebar.header("⚙️ Configuration")

# Fixed features
feat1 = 'Depth  of A'
feat2 = 'LLBCE'

st.sidebar.markdown(f"**X-axis:** {feat1}")
st.sidebar.markdown(f"**Y-axis:** {feat2}")

st.sidebar.subheader("Feature Ranges")
feature_ranges = {}
col1, col2 = st.sidebar.columns(2)
with col1:
    feature_ranges[feat1] = [
        st.number_input(f"{feat1} Min", value=0.5, step=0.1),
        st.number_input(f"{feat1} Max", value=3.0, step=0.1)
    ]
with col2:
    feature_ranges[feat2] = [
        st.number_input(f"{feat2} Min", value=-4.0, step=0.1),
        st.number_input(f"{feat2} Max", value=8.0, step=0.1)
    ]

st.sidebar.subheader("Other Feature Constants")
constant_values = {
    'PMBT': st.sidebar.number_input("PMBT", value=3.2, step=0.1),
    'MRT': st.sidebar.number_input("MRT", value=9.5, step=0.1)
}

grid_size = st.sidebar.slider("Grid Density", 50, 200, 100, 10)
threshold = st.sidebar.slider("Probability Threshold", 0.0, 1.0, 0.8, 0.05)

if st.button("🚀 Run Analysis", type="primary"):
    with st.spinner("Analyzing..."):
        analyzer = FractureAnalysis(config={
            'analysis_features': [feat1, feat2],
            'grid_size': grid_size,
            'probability_threshold': threshold
        })
        analyzer.run_analysis(feature_ranges=feature_ranges, constant_values=constant_values)

# Footer description
st.markdown("---")
st.markdown(
    "This system calculates high-probability parameter intervals, means, and standard deviations "
    "to assist in surgical planning by analyzing the influence of surgical variables on fracture patterns."
)