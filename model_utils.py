import numpy as np
import joblib
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import os

class FractureModel:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.class_names = None
    
    def load_model(self, model_path, scaler_path, encoder_path):
        """加载模型和相关工具"""
        try:
            for path in [model_path, scaler_path, encoder_path]:
                if not os.path.exists(path):
                    print(f"文件不存在: {path}")
                    return False
            
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            self.label_encoder = joblib.load(encoder_path)
            self.class_names = self.label_encoder.classes_
            print(f"模型加载成功，类别: {self.class_names}")
            return True
        except Exception as e:
            print(f"加载失败: {e}")
            return False
    
    def create_prediction_grid(self, feat1, feat2, range1, range2, grid_size):
        """创建预测网格"""
        x1_min, x1_max = range1
        x2_min, x2_max = range2
        
        x1_range = np.linspace(x1_min, x1_max, grid_size)
        x2_range = np.linspace(x2_min, x2_max, grid_size)
        X1, X2 = np.meshgrid(x1_range, x2_range)
        
        return X1, X2, x1_range, x2_range
    
    def predict_on_grid(self, X1, X2, feat1, feat2, constant_values, feature_order):
        """在网格上进行预测"""
        grid_size = X1.shape[0]
        prediction_data = []
        
        for i in range(grid_size):
            for j in range(grid_size):
                sample = []
                for feature in feature_order:
                    if feature == feat1:
                        sample.append(X1[i, j])
                    elif feature == feat2:
                        sample.append(X2[i, j])
                    else:
                        sample.append(constant_values.get(feature, 0))
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
    
    def plot_3d_surface_plotly(self, X1, X2, prob_grids, feat1, feat2, threshold):
        """使用Plotly绘制3D曲面"""
        n_classes = len(self.class_names)
        
        # 只有1个类别时特殊处理
        if n_classes == 1:
            specs = [[{'type': 'surface'}]]
        else:
            specs = [[{'type': 'surface'}] * n_classes]
        
        fig = make_subplots(
            rows=1, 
            cols=max(n_classes, 1),
            subplot_titles=[f'Class: {str(name)}' for name in self.class_names],
            specs=specs
        )
        
        colorscale = 'RdYlBu_r'
        
        for idx, (class_name, prob_grid) in enumerate(zip(self.class_names, prob_grids)):
            # 关键修复：确保所有值都是Python原生类型
            col_idx = int(idx) + 1
            
            fig.add_trace(
                go.Surface(
                    x=X1.astype(float), 
                    y=X2.astype(float), 
                    z=prob_grid.astype(float),
                    colorscale=colorscale,
                    showscale=(idx == 0),
                    name=str(class_name),  # 确保是字符串
                    opacity=0.8
                ),
                row=1, col=col_idx
            )
            
            # 高概率点
            high_prob_mask = prob_grid > threshold
            if np.any(high_prob_mask):
                fig.add_trace(
                    go.Scatter3d(
                        x=X1[high_prob_mask].astype(float),
                        y=X2[high_prob_mask].astype(float),
                        z=prob_grid[high_prob_mask].astype(float),
                        mode='markers',
                        marker=dict(size=3, color='red'),
                        name=f'>{threshold}'  # 简化名称
                    ),
                    row=1, col=col_idx
                )
        
        fig.update_layout(
            title='3D概率分布曲面',
            height=500
        )
        
        return fig
    
    def plot_contour_plotly(self, X1, X2, prob_grids, feat1, feat2, threshold):
        """使用Plotly绘制等高线图"""
        n_classes = len(self.class_names)
        
        fig = make_subplots(
            rows=1,
            cols=max(n_classes, 1),
            subplot_titles=[f'Class: {str(name)}' for name in self.class_names]
        )
        
        for idx, (class_name, prob_grid) in enumerate(zip(self.class_names, prob_grids)):
            col_idx = int(idx) + 1
            
            fig.add_trace(
                go.Contour(
                    x=X1[0].astype(float), 
                    y=X2[:, 0].astype(float), 
                    z=prob_grid.astype(float),
                    colorscale='RdYlBu_r',
                    showscale=(idx == 0),
                    name=str(class_name),
                    contours=dict(
                        start=0, end=1, size=0.05,
                        showlabels=True,
                        labelfont=dict(size=10)
                    )
                ),
                row=1, col=col_idx
            )
            
            high_prob_mask = prob_grid > threshold
            if np.any(high_prob_mask):
                fig.add_trace(
                    go.Scatter(
                        x=X1[high_prob_mask].astype(float),
                        y=X2[high_prob_mask].astype(float),
                        mode='markers',
                        marker=dict(size=4, color='red', opacity=0.6),
                        name=f'>{threshold}'
                    ),
                    row=1, col=col_idx
                )
        
        fig.update_layout(
            height=400
        )
        
        return fig
    
    def display_statistics(self, X1, X2, prob_grids, threshold):
        """显示高概率区域统计信息"""
        st.subheader("高概率区域统计")
        
        for class_idx, class_name in enumerate(self.class_names):
            high_prob_mask = prob_grids[class_idx] > threshold
            high_prob_count = np.sum(high_prob_mask)
            total_points = X1.size
            
            if high_prob_count > 0:
                ratio = high_prob_count / total_points * 100
                X1_high = X1[high_prob_mask]
                X2_high = X2[high_prob_mask]
                
                with st.expander(f"Class {class_name} 详情"):
                    col1, col2, col3 = st.columns(3)
                    col1.metric("高概率点数", f"{int(high_prob_count)}")
                    col2.metric("占比", f"{float(ratio):.2f}%")
                    col3.metric("平均概率", f"{float(prob_grids[class_idx][high_prob_mask].mean()):.3f}")
            else:
                st.warning(f"Class {class_name}: 无高概率区域")