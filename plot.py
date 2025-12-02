import argparse
import ast
import json
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# 设置英文绘图样式
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10
})

class TrainingVisualizer:
    """Training Process Visualization Tool"""
    
    def __init__(self, log_file: str):
        """
        Initialize visualization tool
        
        Args:
            log_file: Path to training log file
        """
        self.log_file = Path(log_file)
        if not self.log_file.exists():
            raise FileNotFoundError(f"Log file not found: {log_file}")
        
        # Read and parse log data
        self.data = self.parse_log_file()
        
        # Set plotting style
        self.set_plot_style()
        
    def parse_log_file(self) -> pd.DataFrame:
        """
        Parse training log file
        
        Returns:
            DataFrame containing all training metrics
        """
        print(f"Parsing log file: {self.log_file}")
        
        records = []
        
        with open(self.log_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                # Try to extract dictionary from mixed format lines
                # This handles lines like: "38%|███▊      | 1500/3916 [1:29:26<2:05:46,  3.12s/it]{'eval_loss': 1.022974...}"
                
                # First, try to find dictionary pattern in the line
                dict_pattern = r'\{[^{}]*\}'  # Match {...} pattern
                matches = re.findall(dict_pattern, line)
                
                if not matches:
                    # If no dictionary found, skip this line
                    continue
                
                # Use the first dictionary found (there should be only one)
                dict_str = matches[0]
                
                # Clean up the dictionary string if needed
                # Some logs might have extra characters before/after the dict
                dict_str = dict_str.strip()
                
                # Extract timestamp or progress info if present (for debugging)
                progress_info = line.replace(dict_str, '').strip()
                if progress_info:
                    # Extract percentage if available
                    percentage_match = re.search(r'(\d+)%', progress_info)
                    if percentage_match:
                        progress_percent = int(percentage_match.group(1))
                        # Optional: store progress percentage
                        # We'll add it to the record later if needed
                
                try:
                    # Try to parse as Python dictionary using ast.literal_eval
                    record = ast.literal_eval(dict_str)
                    
                    # Ensure it's a dictionary
                    if not isinstance(record, dict):
                        print(f"Warning: Line {line_num} parsed but not a dict: {dict_str[:50]}...")
                        continue
                    
                    # Add progress info if available
                    if percentage_match:
                        record['progress_percent'] = progress_percent
                    
                    records.append(record)
                    
                except (SyntaxError, ValueError) as e:
                    # Try to handle as JSON (replace single quotes with double)
                    try:
                        json_str = dict_str.replace("'", '"')
                        # Handle Python-style booleans and None
                        json_str = json_str.replace('True', 'true').replace('False', 'false').replace('None', 'null')
                        # Handle NaN values
                        json_str = json_str.replace('nan', 'null')
                        
                        record = json.loads(json_str)
                        
                        # Add progress info if available
                        if percentage_match:
                            record['progress_percent'] = progress_percent
                            
                        records.append(record)
                        
                    except json.JSONDecodeError as json_err:
                        print(f"Warning: Could not parse line {line_num}: {dict_str[:50]}...")
                        print(f"  Error: {json_err}")
                        continue
        
        if not records:
            raise ValueError("No valid data records found in log file")
        
        print(f"Successfully parsed {len(records)} records")
        
        # Convert to DataFrame
        df = pd.DataFrame(records)
        
        # Clean up column names (remove any whitespace)
        df.columns = [col.strip() for col in df.columns]
        
        # Ensure epoch column exists and is numeric
        if 'epoch' in df.columns:
            df['epoch'] = pd.to_numeric(df['epoch'], errors='coerce')
        
        # Sort data by epoch if available, otherwise by index
        if 'epoch' in df.columns and not df['epoch'].isna().all():
            df = df.sort_values('epoch')
        
        # Create step column (sequential number for each record)
        df['step'] = np.arange(len(df))
        
        # Determine record type (train or eval)
        df['record_type'] = 'train'
        
        # Check for evaluation metrics
        eval_indicators = ['eval_loss', 'eval_runtime', 'eval_samples_per_second']
        has_eval_metrics = any(indicator in df.columns for indicator in eval_indicators)
        
        if has_eval_metrics:
            # Mark evaluation records (where eval_loss is not null)
            if 'eval_loss' in df.columns:
                df.loc[df['eval_loss'].notna(), 'record_type'] = 'eval'
            # Alternative: if there's no eval_loss but other eval metrics exist
            elif any(col in df.columns for col in eval_indicators):
                # Create a combined indicator
                eval_mask = df[eval_indicators].notna().any(axis=1)
                df.loc[eval_mask, 'record_type'] = 'eval'
        
        print(f"Data columns: {list(df.columns)}")
        print(f"Training records: {len(df[df['record_type']=='train'])}")
        print(f"Evaluation records: {len(df[df['record_type']=='eval'])}")
        
        # Display sample data
        print("\nSample data (first 5 records):")
        print(df.head())
        
        return df
    
    def set_plot_style(self):
        """Set plotting style"""
        sns.set_style("whitegrid")
        sns.set_context("notebook", font_scale=1.1)
        
        # Define color scheme
        self.colors = {
            'train_loss': '#FF6B6B',  # Red
            'eval_loss': '#4ECDC4',   # Teal
            'grad_norm': '#FFA726',   # Orange
            'learning_rate': '#66BB6A',  # Green
            'train': '#5C6BC0',       # Blue
            'eval': '#AB47BC',        # Purple
            'background': '#F5F5F5'
        }
        
        # Define line styles
        self.line_styles = {
            'train': '-',
            'eval': '--'
        }
    
    def plot_training_curves(self, save_path: Optional[str] = None, dpi: int = 150):
        """
        Plot training curves
        
        Args:
            save_path: Save path, if None then display plot
            dpi: Plot resolution
        """
        fig = plt.figure(figsize=(16, 10))
        fig.patch.set_facecolor(self.colors['background'])
        
        # 1. Loss curves
        ax1 = plt.subplot(2, 2, 1)
        self._plot_loss_curves(ax1)
        
        # 2. Gradient norm
        ax2 = plt.subplot(2, 2, 2)
        self._plot_grad_norm(ax2)
        
        # 3. Learning rate
        ax3 = plt.subplot(2, 2, 3)
        self._plot_learning_rate(ax3)
        
        # 4. Training speed (if evaluation runtime data exists)
        ax4 = plt.subplot(2, 2, 4)
        self._plot_training_speed(ax4)
        
        plt.suptitle('Training Process Analysis', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor=fig.get_facecolor())
            print(f"Plot saved to: {save_path}")
        else:
            plt.show()
        
        plt.close(fig)
    
    def _plot_loss_curves(self, ax):
        """Plot loss curves"""
        ax.set_facecolor(self.colors['background'])
        
        # Extract training loss data
        train_data = self.data[self.data['record_type'] == 'train']
        
        if 'loss' in train_data.columns and len(train_data) > 0:
            # Use epoch if available, otherwise use step
            if 'epoch' in train_data.columns and not train_data['epoch'].isna().all():
                x_values = train_data['epoch']
                x_label = 'Epoch'
            else:
                x_values = train_data['step']
                x_label = 'Step'
            
            # Filter out NaN values
            valid_mask = train_data['loss'].notna()
            if valid_mask.any():
                ax.plot(x_values[valid_mask], train_data['loss'][valid_mask], 
                       color=self.colors['train_loss'], 
                       linewidth=2.5, 
                       label='Training Loss', 
                       marker='o', markersize=5, markevery=max(1, sum(valid_mask)//20))
            
            # Add trend line if enough data points
            if sum(valid_mask) > 5:
                try:
                    z = np.polyfit(train_data['step'][valid_mask], train_data['loss'][valid_mask], 2)
                    p = np.poly1d(z)
                    ax.plot(x_values[valid_mask], p(train_data['step'][valid_mask]), 
                           color=self.colors['train_loss'], 
                           linewidth=1.5, 
                           linestyle='--', 
                           alpha=0.7, 
                           label='Trend Line')
                except:
                    pass
        
        # Extract evaluation loss data
        eval_data = self.data[self.data['record_type'] == 'eval']
        
        if 'eval_loss' in eval_data.columns and len(eval_data) > 0:
            # Use epoch if available, otherwise use step
            if 'epoch' in eval_data.columns and not eval_data['epoch'].isna().all():
                x_values = eval_data['epoch']
            else:
                x_values = eval_data['step']
            
            # Filter out NaN values
            valid_mask = eval_data['eval_loss'].notna()
            if valid_mask.any():
                ax.plot(x_values[valid_mask], eval_data['eval_loss'][valid_mask], 
                       color=self.colors['eval_loss'], 
                       linewidth=2.5, 
                       label='Validation Loss', 
                       marker='s', markersize=6, markevery=max(1, sum(valid_mask)//10))
        
        ax.set_xlabel(x_label if 'x_label' in locals() else 'Step')
        ax.set_ylabel('Loss Value')
        ax.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        # Add average loss label if training data exists
        if 'loss' in train_data.columns and train_data['loss'].notna().any():
            avg_loss = train_data['loss'].mean()
            ax.annotate(f'Avg Train Loss: {avg_loss:.4f}', 
                       xy=(0.05, 0.05), xycoords='axes fraction',
                       fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9))
    
    def _plot_grad_norm(self, ax):
        """Plot gradient norm"""
        ax.set_facecolor(self.colors['background'])
        
        train_data = self.data[self.data['record_type'] == 'train']
        
        if 'grad_norm' in train_data.columns and len(train_data) > 0:
            # Use epoch if available, otherwise use step
            if 'epoch' in train_data.columns and not train_data['epoch'].isna().all():
                x_values = train_data['epoch']
                x_label = 'Epoch'
            else:
                x_values = train_data['step']
                x_label = 'Step'
            
            # Filter out NaN values
            valid_mask = train_data['grad_norm'].notna()
            if valid_mask.any():
                ax.plot(x_values[valid_mask], train_data['grad_norm'][valid_mask], 
                       color=self.colors['grad_norm'], 
                       linewidth=2.5, 
                       label='Gradient Norm',
                       marker='^', markersize=5, markevery=max(1, sum(valid_mask)//20))
            
                # Add moving average if enough data points
                if sum(valid_mask) > 10:
                    window = min(10, sum(valid_mask) // 5)
                    moving_avg = train_data['grad_norm'][valid_mask].rolling(window=window, center=True).mean()
                    ax.plot(x_values[valid_mask], moving_avg, 
                           color='darkorange', 
                           linewidth=2, 
                           linestyle='--', 
                           label=f'{window}-step Moving Avg')
            
            ax.set_xlabel(x_label)
            ax.set_ylabel('Gradient Norm')
            ax.set_title('Gradient Norm Variation', fontsize=14, fontweight='bold')
            ax.legend(loc='upper right', framealpha=0.9)
            ax.grid(True, alpha=0.3)
            
            # Add statistics if data exists
            if valid_mask.any():
                max_grad = train_data['grad_norm'][valid_mask].max()
                min_grad = train_data['grad_norm'][valid_mask].min()
                avg_grad = train_data['grad_norm'][valid_mask].mean()
                
                ax.annotate(f'Max: {max_grad:.2f}\nMin: {min_grad:.2f}\nAvg: {avg_grad:.2f}', 
                           xy=(0.05, 0.95), xycoords='axes fraction',
                           fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9),
                           verticalalignment='top')
        else:
            ax.text(0.5, 0.5, 'No Gradient Norm Data', 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=12)
            ax.set_title('Gradient Norm Variation', fontsize=14, fontweight='bold')
    
    def _plot_learning_rate(self, ax):
        """Plot learning rate variation"""
        ax.set_facecolor(self.colors['background'])
        
        train_data = self.data[self.data['record_type'] == 'train']
        
        if 'learning_rate' in train_data.columns and len(train_data) > 0:
            # Use epoch if available, otherwise use step
            if 'epoch' in train_data.columns and not train_data['epoch'].isna().all():
                x_values = train_data['epoch']
                x_label = 'Epoch'
            else:
                x_values = train_data['step']
                x_label = 'Step'
            
            # Filter out NaN values
            valid_mask = train_data['learning_rate'].notna()
            if valid_mask.any():
                lr_values = train_data['learning_rate'][valid_mask]
                
                # Check for non-positive values (avoid log errors)
                if (lr_values > 0).all():
                    ax.semilogy(x_values[valid_mask], lr_values, 
                              color=self.colors['learning_rate'], 
                              linewidth=2.5,
                              label='Learning Rate',
                              marker='d', markersize=5, markevery=max(1, sum(valid_mask)//20))
                    ax.set_ylabel('Learning Rate (log scale)')
                else:
                    ax.plot(x_values[valid_mask], lr_values, 
                           color=self.colors['learning_rate'], 
                           linewidth=2.5,
                           label='Learning Rate',
                           marker='d', markersize=5, markevery=max(1, sum(valid_mask)//20))
                    ax.set_ylabel('Learning Rate')
            
            ax.set_xlabel(x_label)
            ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
            ax.legend(loc='upper right', framealpha=0.9)
            ax.grid(True, alpha=0.3)
            
            # Add learning rate range if data exists
            if valid_mask.any():
                initial_lr = lr_values.iloc[0] if len(lr_values) > 0 else 0
                final_lr = lr_values.iloc[-1] if len(lr_values) > 0 else 0
                if initial_lr > 0:
                    reduction = (initial_lr - final_lr) / initial_lr * 100
                else:
                    reduction = 0
                
                ax.annotate(f'Initial: {initial_lr:.2e}\nFinal: {final_lr:.2e}\nReduction: {reduction:.1f}%', 
                           xy=(0.05, 0.95), xycoords='axes fraction',
                           fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9),
                           verticalalignment='top')
        else:
            ax.text(0.5, 0.5, 'No Learning Rate Data', 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=12)
            ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    
    def _plot_training_speed(self, ax):
        """Plot training/validation speed"""
        ax.set_facecolor(self.colors['background'])
        
        eval_data = self.data[self.data['record_type'] == 'eval']
        
        # Check for speed metrics
        speed_metrics = ['eval_samples_per_second', 'eval_steps_per_second', 
                        'train_samples_per_second', 'train_steps_per_second']
        
        # Find available metrics
        available_metrics = [m for m in speed_metrics if m in self.data.columns]
        
        if available_metrics:
            # Use first available speed metric
            metric = available_metrics[0]
            
            if metric.startswith('eval'):
                data = eval_data
                color = self.colors['eval']
            else:
                data = self.data[self.data['record_type'] == 'train']
                color = self.colors['train']
            
            if len(data) > 0:
                # Use epoch if available, otherwise use step
                if 'epoch' in data.columns and not data['epoch'].isna().all():
                    x_values = data['epoch']
                    x_label = 'Epoch'
                else:
                    x_values = data['step']
                    x_label = 'Step'
                
                # Filter out NaN values
                valid_mask = data[metric].notna()
                if valid_mask.any():
                    ax.plot(x_values[valid_mask], data[metric][valid_mask], 
                           color=color, 
                           linewidth=2.5,
                           label=metric.replace('_', ' ').title(),
                           marker='*', markersize=6, markevery=max(1, sum(valid_mask)//10))
                    
                    ax.set_xlabel(x_label)
                    ax.set_ylabel('Speed (samples/sec)')
                    ax.set_title('Training/Evaluation Speed', fontsize=14, fontweight='bold')
                    ax.legend(loc='upper right', framealpha=0.9)
                    ax.grid(True, alpha=0.3)
                    
                    # Add statistics
                    avg_speed = data[metric][valid_mask].mean()
                    max_speed = data[metric][valid_mask].max()
                    
                    ax.annotate(f'Avg: {avg_speed:.1f}\nMax: {max_speed:.1f}', 
                               xy=(0.05, 0.95), xycoords='axes fraction',
                               fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9),
                               verticalalignment='top')
                    return
        
        # If no speed data, show runtime information
        if 'eval_runtime' in eval_data.columns and len(eval_data) > 0:
            # Use epoch if available, otherwise use step
            if 'epoch' in eval_data.columns and not eval_data['epoch'].isna().all():
                x_values = eval_data['epoch']
                x_label = 'Epoch'
            else:
                x_values = eval_data['step']
                x_label = 'Step'
            
            # Filter out NaN values
            valid_mask = eval_data['eval_runtime'].notna()
            if valid_mask.any():
                ax.plot(x_values[valid_mask], eval_data['eval_runtime'][valid_mask], 
                       color=self.colors['eval'], 
                       linewidth=2.5,
                       label='Evaluation Runtime',
                       marker='*', markersize=6, markevery=max(1, sum(valid_mask)//10))
                
                ax.set_xlabel(x_label)
                ax.set_ylabel('Runtime (seconds)')
                ax.set_title('Evaluation Runtime', fontsize=14, fontweight='bold')
                ax.legend(loc='upper right', framealpha=0.9)
                ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No Training Speed Data', 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=12)
            ax.set_title('Training/Evaluation Speed', fontsize=14, fontweight='bold')
    
    def plot_detailed_analysis(self, save_path: Optional[str] = None, dpi: int = 150):
        """
        Plot detailed analysis charts
        
        Args:
            save_path: Save path, if None then display plot
            dpi: Plot resolution
        """
        fig = plt.figure(figsize=(18, 10))
        fig.patch.set_facecolor(self.colors['background'])
        
        # 1. Loss distribution (box plot)
        ax1 = plt.subplot(2, 3, 1)
        self._plot_loss_distribution(ax1)
        
        # 2. Gradient norm distribution (box plot)
        ax2 = plt.subplot(2, 3, 2)
        self._plot_grad_norm_distribution(ax2)
        
        # 3. Learning rate vs loss correlation
        ax3 = plt.subplot(2, 3, 3)
        self._plot_lr_loss_correlation(ax3)
        
        # 4. Training vs validation loss comparison
        ax4 = plt.subplot(2, 3, 4)
        self._plot_train_eval_comparison(ax4)
        
        # 5. Convergence analysis
        ax5 = plt.subplot(2, 3, 5)
        self._plot_convergence_analysis(ax5)
        
        # 6. Training statistics summary
        ax6 = plt.subplot(2, 3, 6)
        self._plot_statistics_summary(ax6)
        
        plt.suptitle('Training Process Detailed Analysis', fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            detailed_path = str(save_path).replace('.png', '_detailed.png')
            plt.savefig(detailed_path, dpi=dpi, bbox_inches='tight', facecolor=fig.get_facecolor())
            print(f"Detailed analysis plot saved to: {detailed_path}")
        else:
            plt.show()
        
        plt.close(fig)
    
    def _plot_loss_distribution(self, ax):
        """Plot loss distribution"""
        ax.set_facecolor(self.colors['background'])
        
        train_data = self.data[self.data['record_type'] == 'train']
        
        if 'loss' in train_data.columns and train_data['loss'].notna().any():
            # Filter out NaN values
            loss_values = train_data['loss'].dropna()
            
            if len(loss_values) > 0:
                # Create box plot
                bp = ax.boxplot(loss_values, patch_artist=True, widths=0.6)
                
                # Set box plot colors
                bp['boxes'][0].set_facecolor(self.colors['train_loss'])
                bp['boxes'][0].set_alpha(0.7)
                
                # Add jittered scatter plot to show data distribution
                x = np.random.normal(1, 0.04, size=len(loss_values))
                ax.scatter(x, loss_values, alpha=0.6, color=self.colors['train_loss'], 
                          s=30, edgecolors='black', linewidth=0.5)
                
                ax.set_xticklabels(['Training Loss'])
                ax.set_ylabel('Loss Value')
                ax.set_title('Training Loss Distribution', fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'No Training Loss Data', 
                       horizontalalignment='center', verticalalignment='center',
                       transform=ax.transAxes, fontsize=12)
                ax.set_title('Training Loss Distribution', fontsize=14, fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'No Training Loss Data', 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=12)
            ax.set_title('Training Loss Distribution', fontsize=14, fontweight='bold')
    
    def _plot_grad_norm_distribution(self, ax):
        """Plot gradient norm distribution"""
        ax.set_facecolor(self.colors['background'])
        
        train_data = self.data[self.data['record_type'] == 'train']
        
        if 'grad_norm' in train_data.columns and train_data['grad_norm'].notna().any():
            # Filter out NaN values
            grad_values = train_data['grad_norm'].dropna()
            
            if len(grad_values) > 0:
                # Create box plot
                bp = ax.boxplot(grad_values, patch_artist=True, widths=0.6)
                
                # Set box plot colors
                bp['boxes'][0].set_facecolor(self.colors['grad_norm'])
                bp['boxes'][0].set_alpha(0.7)
                
                # Add jittered scatter plot to show data distribution
                x = np.random.normal(1, 0.04, size=len(grad_values))
                ax.scatter(x, grad_values, alpha=0.6, color=self.colors['grad_norm'], 
                          s=30, edgecolors='black', linewidth=0.5)
                
                ax.set_xticklabels(['Gradient Norm'])
                ax.set_ylabel('Norm Value')
                ax.set_title('Gradient Norm Distribution', fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'No Gradient Norm Data', 
                       horizontalalignment='center', verticalalignment='center',
                       transform=ax.transAxes, fontsize=12)
                ax.set_title('Gradient Norm Distribution', fontsize=14, fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'No Gradient Norm Data', 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=12)
            ax.set_title('Gradient Norm Distribution', fontsize=14, fontweight='bold')
    
    def _plot_lr_loss_correlation(self, ax):
        """Plot learning rate vs loss correlation"""
        ax.set_facecolor(self.colors['background'])
        
        train_data = self.data[self.data['record_type'] == 'train']
        
        has_lr = 'learning_rate' in train_data.columns and train_data['learning_rate'].notna().any()
        has_loss = 'loss' in train_data.columns and train_data['loss'].notna().any()
        
        if has_lr and has_loss and len(train_data) > 0:
            # Filter out NaN values
            valid_mask = train_data['learning_rate'].notna() & train_data['loss'].notna()
            if valid_mask.any():
                lr_values = train_data['learning_rate'][valid_mask]
                loss_values = train_data['loss'][valid_mask]
                
                # Create scatter plot
                scatter = ax.scatter(lr_values, loss_values,
                                   c=train_data['step'][valid_mask] if 'step' in train_data.columns else range(len(lr_values)),
                                   cmap='viridis', alpha=0.7, s=50, edgecolors='black', linewidth=0.5)
                
                ax.set_xlabel('Learning Rate')
                ax.set_ylabel('Training Loss')
                ax.set_title('Learning Rate vs Loss Correlation', fontsize=14, fontweight='bold')
                
                # Add color bar
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label('Training Step')
                
                # If enough data points, add trend line
                if len(lr_values) > 5:
                    try:
                        # Use log scale for fitting
                        x_log = np.log10(lr_values + 1e-10)
                        
                        # Polynomial fit
                        coef = np.polyfit(x_log, loss_values, 2)
                        poly1d_fn = np.poly1d(coef)
                        
                        # Generate fit line
                        x_sorted = np.sort(x_log)
                        ax.plot(10**x_sorted, poly1d_fn(x_sorted), 
                               color='red', linewidth=2, linestyle='--', 
                               label='Trend Line')
                        
                        ax.legend(framealpha=0.9)
                    except:
                        pass
                
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'No Learning Rate or Loss Data', 
                       horizontalalignment='center', verticalalignment='center',
                       transform=ax.transAxes, fontsize=12)
                ax.set_title('Learning Rate vs Loss Correlation', fontsize=14, fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'No Learning Rate or Loss Data', 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=12)
            ax.set_title('Learning Rate vs Loss Correlation', fontsize=14, fontweight='bold')
    
    def _plot_train_eval_comparison(self, ax):
        """Plot training vs validation loss comparison"""
        ax.set_facecolor(self.colors['background'])
        
        train_data = self.data[self.data['record_type'] == 'train']
        eval_data = self.data[self.data['record_type'] == 'eval']
        
        has_train_loss = 'loss' in train_data.columns and train_data['loss'].notna().any()
        has_eval_loss = 'eval_loss' in eval_data.columns and eval_data['eval_loss'].notna().any()
        
        if has_train_loss and has_eval_loss:
            # Use bar chart to compare average losses
            categories = ['Training Loss', 'Validation Loss']
            train_loss_mean = train_data['loss'].mean()
            eval_loss_mean = eval_data['eval_loss'].mean()
            values = [train_loss_mean, eval_loss_mean]
            
            bars = ax.bar(categories, values, color=[self.colors['train_loss'], self.colors['eval_loss']],
                         edgecolor='black', linewidth=1.5)
            
            # Add values on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
            
            ax.set_ylabel('Average Loss')
            ax.set_title('Training vs Validation Loss', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Calculate and display gap
            gap = eval_loss_mean - train_loss_mean
            if train_loss_mean > 0:
                gap_percent = (gap / train_loss_mean) * 100
            else:
                gap_percent = 0
            
            ax.annotate(f'Gap: {gap:.4f} ({gap_percent:.1f}%)', 
                       xy=(0.5, 0.95), xycoords='axes fraction',
                       fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9),
                       horizontalalignment='center')
            
        elif has_train_loss or has_eval_loss:
            # Only one type of data
            if has_train_loss:
                ax.bar(['Training Loss'], [train_data['loss'].mean()], 
                      color=self.colors['train_loss'], edgecolor='black', linewidth=1.5)
                ax.set_title('Training Loss', fontsize=14, fontweight='bold')
            else:
                ax.bar(['Validation Loss'], [eval_data['eval_loss'].mean()], 
                      color=self.colors['eval_loss'], edgecolor='black', linewidth=1.5)
                ax.set_title('Validation Loss', fontsize=14, fontweight='bold')
            
            ax.set_ylabel('Average Loss')
            ax.grid(True, alpha=0.3, axis='y')
        else:
            ax.text(0.5, 0.5, 'No Training or Validation Loss Data', 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=12)
            ax.set_title('Training vs Validation Loss', fontsize=14, fontweight='bold')
    
    def _plot_convergence_analysis(self, ax):
        """Plot convergence analysis"""
        ax.set_facecolor(self.colors['background'])
        
        train_data = self.data[self.data['record_type'] == 'train']
        
        if 'loss' in train_data.columns and train_data['loss'].notna().any() and len(train_data) > 10:
            # Filter out NaN values
            loss_values = train_data['loss'].dropna().values
            
            # Calculate relative changes
            if len(loss_values) > 1:
                relative_changes = np.abs(np.diff(loss_values) / (loss_values[:-1] + 1e-10))
                
                # Moving average to reduce noise
                window = min(5, len(relative_changes) // 3)
                if window > 1:
                    relative_changes_smooth = pd.Series(relative_changes).rolling(window=window, center=True).mean().values
                else:
                    relative_changes_smooth = relative_changes
                
                # Get corresponding steps
                steps = train_data['step'].values[1:train_data['step'].shape[0]] if len(relative_changes) > 0 else []
                
                if len(steps) > 0 and len(relative_changes_smooth) > 0:
                    ax.plot(steps, relative_changes_smooth, 
                           color='purple', linewidth=2.5,
                           label='Loss Relative Change Rate',
                           marker='.', markersize=8, markevery=max(1, len(steps)//20))
                    
                    # Add convergence threshold line
                    threshold = 0.001  # 0.1% change threshold
                    ax.axhline(y=threshold, color='red', linestyle='--', alpha=0.8, linewidth=2,
                              label=f'Convergence Threshold ({threshold})')
                    
                    # Mark convergence point
                    below_threshold = relative_changes_smooth < threshold
                    if below_threshold.any():
                        first_convergence = np.where(below_threshold)[0]
                        if len(first_convergence) > 0:
                            conv_step = steps[first_convergence[0]]
                            conv_value = relative_changes_smooth[first_convergence[0]]
                            ax.scatter([conv_step], [conv_value], color='red', s=100, zorder=5,
                                      label=f'Convergence Point (Step {conv_step})')
                    
                    ax.set_xlabel('Step')
                    ax.set_ylabel('Loss Change Rate')
                    ax.set_title('Convergence Analysis', fontsize=14, fontweight='bold')
                    ax.legend(loc='upper right', framealpha=0.9)
                    ax.grid(True, alpha=0.3)
                else:
                    ax.text(0.5, 0.5, 'Insufficient Data for Convergence Analysis', 
                           horizontalalignment='center', verticalalignment='center',
                           transform=ax.transAxes, fontsize=12)
                    ax.set_title('Convergence Analysis', fontsize=14, fontweight='bold')
            else:
                ax.text(0.5, 0.5, 'Insufficient Data for Convergence Analysis', 
                       horizontalalignment='center', verticalalignment='center',
                       transform=ax.transAxes, fontsize=12)
                ax.set_title('Convergence Analysis', fontsize=14, fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'No Training Loss Data for Convergence Analysis', 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes, fontsize=12)
            ax.set_title('Convergence Analysis', fontsize=14, fontweight='bold')
    
    def _plot_statistics_summary(self, ax):
        """Plot statistics summary"""
        # Clear axis
        ax.clear()
        ax.axis('off')
        ax.set_facecolor(self.colors['background'])
        
        # Collect statistics
        stats_text = "Training Process Statistics Summary\n\n"
        
        # Total records
        stats_text += f"Total Records: {len(self.data)}\n"
        
        # Training records statistics
        train_data = self.data[self.data['record_type'] == 'train']
        if len(train_data) > 0:
            stats_text += f"\nTraining Records: {len(train_data)}\n"
            
            if 'loss' in train_data.columns and train_data['loss'].notna().any():
                stats_text += f"  Loss - Average: {train_data['loss'].mean():.4f}\n"
                stats_text += f"         Min: {train_data['loss'].min():.4f}\n"
                stats_text += f"         Max: {train_data['loss'].max():.4f}\n"
            
            if 'grad_norm' in train_data.columns and train_data['grad_norm'].notna().any():
                stats_text += f"  Gradient Norm - Average: {train_data['grad_norm'].mean():.2f}\n"
        
        # Evaluation records statistics
        eval_data = self.data[self.data['record_type'] == 'eval']
        if len(eval_data) > 0:
            stats_text += f"\nEvaluation Records: {len(eval_data)}\n"
            
            if 'eval_loss' in eval_data.columns and eval_data['eval_loss'].notna().any():
                stats_text += f"  Loss - Average: {eval_data['eval_loss'].mean():.4f}\n"
                stats_text += f"         Min: {eval_data['eval_loss'].min():.4f}\n"
                stats_text += f"         Max: {eval_data['eval_loss'].max():.4f}\n"
            
            if 'eval_runtime' in eval_data.columns and eval_data['eval_runtime'].notna().any():
                total_eval_time = eval_data['eval_runtime'].sum()
                stats_text += f"  Total Evaluation Time: {total_eval_time:.1f} seconds\n"
        
        # Training duration estimation
        if 'epoch' in self.data.columns and self.data['epoch'].notna().any():
            max_epoch = self.data['epoch'].max()
            stats_text += f"\nTotal Epochs: {max_epoch:.2f}\n"
        
        # Add text to plot
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
               fontsize=10, verticalalignment='top', family='monospace',
               bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.9, edgecolor='black'))
        
        ax.set_title('Statistics Summary', fontsize=14, fontweight='bold', pad=20)
    
    def export_statistics(self, output_file: str):
        """
        Export statistics to CSV file
        
        Args:
            output_file: Output file path
        """
        # Calculate various statistics
        stats = {}
        
        # Basic statistics
        stats['total_records'] = len(self.data)
        stats['train_records'] = len(self.data[self.data['record_type'] == 'train'])
        stats['eval_records'] = len(self.data[self.data['record_type'] == 'eval'])
        
        # Training loss statistics
        train_data = self.data[self.data['record_type'] == 'train']
        print(f"Train Loss is {train_data['loss']}")
        if 'loss' in train_data.columns and train_data['loss'].notna().any():
            stats['train_loss_mean'] = train_data['loss'].mean()
            stats['train_loss_min'] = train_data['loss'].min()
            stats['train_loss_max'] = train_data['loss'].max()
            stats['train_loss_std'] = train_data['loss'].std()
            stats['train_loss_final'] = train_data['loss'].iloc[-1] if len(train_data) > 0 else 0
        
        # Validation loss statistics
        eval_data = self.data[self.data['record_type'] == 'eval']
        if 'eval_loss' in eval_data.columns and eval_data['eval_loss'].notna().any():
            stats['eval_loss_mean'] = eval_data['eval_loss'].mean()
            stats['eval_loss_min'] = eval_data['eval_loss'].min()
            stats['eval_loss_max'] = eval_data['eval_loss'].max()
            stats['eval_loss_std'] = eval_data['eval_loss'].std()
            stats['eval_loss_final'] = eval_data['eval_loss'].iloc[-1] if len(eval_data) > 0 else 0
        
        # Gradient norm statistics
        if 'grad_norm' in train_data.columns and train_data['grad_norm'].notna().any():
            stats['grad_norm_mean'] = train_data['grad_norm'].mean()
            stats['grad_norm_min'] = train_data['grad_norm'].min()
            stats['grad_norm_max'] = train_data['grad_norm'].max()
        
        # Learning rate statistics
        if 'learning_rate' in train_data.columns and train_data['learning_rate'].notna().any():
            stats['learning_rate_initial'] = train_data['learning_rate'].iloc[0] if len(train_data) > 0 else 0
            stats['learning_rate_final'] = train_data['learning_rate'].iloc[-1] if len(train_data) > 0 else 0
        
        # Save to CSV
        stats_df = pd.DataFrame([stats])
        stats_df.to_csv(output_file, index=False)
        print(f"Statistics exported to: {output_file}")
        
        return stats


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Training Process Visualization Tool')
    parser.add_argument('--log_file', type=str, required=True, 
                       help='Path to training log file')
    parser.add_argument('--output_dir', type=str, default='./training_plots',
                       help='Output directory path')
    parser.add_argument('--dpi', type=int, default=150,
                       help='Plot resolution')
    parser.add_argument('--detailed', action='store_true',
                       help='Generate detailed analysis plots')
    parser.add_argument('--export_stats', action='store_true',
                       help='Export statistics to CSV')
    parser.add_argument('--model_name', type=str, default='Unknown_Model',
                       help='Model name for plot titles')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Initialize visualization tool
        visualizer = TrainingVisualizer(args.log_file)
        
        # Generate basic plots
        output_file = output_dir / f"training_curves_{args.model_name.replace('/', '_')}.png"
        visualizer.plot_training_curves(save_path=str(output_file), dpi=args.dpi)
        
        # If requested, generate detailed analysis plots
        if args.detailed:
            visualizer.plot_detailed_analysis(save_path=str(output_file), dpi=args.dpi)
        
        # If requested, export statistics
        if args.export_stats:
            stats_file = output_dir / f"training_stats_{args.model_name.replace('/', '_')}.csv"
            stats = visualizer.export_statistics(str(stats_file))
            print(f"\nTraining Statistics Summary:")
            for key, value in stats.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.4f}")
                else:
                    print(f"  {key}: {value}")
        
        print(f"\nVisualization completed successfully!")
        print(f"Output directory: {output_dir}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()