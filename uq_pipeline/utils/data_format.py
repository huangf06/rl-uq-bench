"""
Data Format Utilities
Unified data reading/writing functions for UQ pipeline.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Union
import pickle
import lzma


def save_dataframe(df: pd.DataFrame, path: Path, compression: str = 'xz') -> None:
    """
    Save DataFrame to compressed file.
    
    Args:
        df: DataFrame to save
        path: Output file path
        compression: Compression method ('xz', 'gzip', or None)
    """
    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save with appropriate compression
    if compression == 'xz':
        df.to_pickle(path, compression='xz')
    elif compression == 'gzip':
        df.to_pickle(path, compression='gzip')
    else:
        df.to_pickle(path)


def load_dataframe(path: Path) -> pd.DataFrame:
    """
    Load DataFrame from compressed file with NumPy compatibility handling.
    
    Args:
        path: Path to compressed DataFrame file
        
    Returns:
        Loaded DataFrame
    """
    try:
        # 首先尝试直接读取（适用于相同NumPy版本）
        return pd.read_pickle(path)
    except (ModuleNotFoundError, ImportError) as e:
        if 'numpy._core' in str(e):
            # NumPy版本兼容性问题 - 尝试兼容性读取
            import warnings
            warnings.warn(f"NumPy compatibility issue detected, attempting fallback loading for {path}")
            
            try:
                # 方法1: 尝试使用pickle直接加载
                import lzma
                with lzma.open(path, 'rb') as f:
                    # 设置兼容性pickle协议
                    import pickle
                    data = pickle.load(f)
                    if isinstance(data, pd.DataFrame):
                        return data
                    else:
                        raise ValueError("Loaded data is not a DataFrame")
                        
            except Exception as fallback_error:
                # 方法2: 如果是eval数据集，尝试重新生成
                if 'eval_dataset' in str(path):
                    raise FileNotFoundError(
                        f"Cannot load evaluation dataset {path} due to NumPy compatibility. "
                        f"Please regenerate the dataset with current environment or upgrade NumPy. "
                        f"Original error: {e}"
                    )
                else:
                    raise RuntimeError(
                        f"Failed to load {path} with NumPy compatibility fallback. "
                        f"Original error: {e}, Fallback error: {fallback_error}"
                    )
        else:
            # 其他类型的错误，直接抛出
            raise


def save_json(data: Dict[str, Any], path: Path, indent: int = 2) -> None:
    """
    Save dictionary to JSON file with NumPy type handling.
    
    Args:
        data: Dictionary to save
        path: Output file path
        indent: JSON indentation level
    """
    # Ensure parent directory exists
    path.parent.mkdir(parents=True, exist_ok=True)
    
    def numpy_json_serializer(obj):
        """JSON serializer for NumPy types"""
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, '__dict__'):
            # 对于其他对象，尝试字符串化
            return str(obj)
        else:
            return str(obj)
    
    with open(path, 'w') as f:
        json.dump(data, f, indent=indent, default=numpy_json_serializer)


def load_json(path: Path) -> Dict[str, Any]:
    """
    Load dictionary from JSON file.
    
    Args:
        path: Path to JSON file
        
    Returns:
        Loaded dictionary
    """
    with open(path, 'r') as f:
        return json.load(f)


def save_numpy_compressed(array: np.ndarray, path: Path) -> None:
    """
    Save NumPy array to compressed file.
    
    Args:
        array: NumPy array to save
        path: Output file path (.xz extension recommended)
    """
    # TODO: Implement compressed NumPy array saving
    pass


def load_numpy_compressed(path: Path) -> np.ndarray:
    """
    Load NumPy array from compressed file.
    
    Args:
        path: Path to compressed array file
        
    Returns:
        Loaded NumPy array
    """
    # TODO: Implement compressed NumPy array loading
    pass


def save_pickle_compressed(obj: Any, path: Path) -> None:
    """
    Save Python object to compressed pickle file.
    
    Args:
        obj: Object to save
        path: Output file path (.xz extension recommended)
    """
    # TODO: Implement compressed pickle saving
    pass


def load_pickle_compressed(path: Path) -> Any:
    """
    Load Python object from compressed pickle file.
    
    Args:
        path: Path to compressed pickle file
        
    Returns:
        Loaded object
    """
    # TODO: Implement compressed pickle loading
    pass


def save_q_values(q_values: Union[np.ndarray, pd.DataFrame], path: Path, 
                 metadata: Optional[Dict[str, Any]] = None) -> None:
    """
    Save Q-values array or DataFrame with optional metadata.
    
    Args:
        q_values: Q-values array or DataFrame to save
        path: Output file path (should end with .xz)
        metadata: Optional metadata dictionary
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if isinstance(q_values, pd.DataFrame):
        # Add metadata as DataFrame attributes if provided
        if metadata:
            for key, value in metadata.items():
                q_values.attrs[key] = value
        q_values.to_pickle(path, compression='xz')
    else:
        # Save numpy array with metadata
        data_to_save = {
            'q_values': q_values,
            'metadata': metadata or {}
        }
        with lzma.open(path, 'wb') as f:
            pickle.dump(data_to_save, f)


def load_q_values(path: Path) -> tuple:
    """
    Load Q-values array and metadata.
    
    Args:
        path: Path to Q-values file
        
    Returns:
        Tuple of (q_values_array, metadata_dict)
    """
    try:
        # Try to load as DataFrame first
        df = pd.read_pickle(path, compression='xz')
        if isinstance(df, pd.DataFrame):
            metadata = dict(df.attrs) if hasattr(df, 'attrs') else {}
            return df, metadata
    except:
        pass
    
    # Try to load as pickle with metadata
    try:
        with lzma.open(path, 'rb') as f:
            data = pickle.load(f)
            return data['q_values'], data.get('metadata', {})
    except Exception as e:
        raise ValueError(f"Failed to load Q-values from {path}: {e}")


def save_metrics_csv(metrics_df: pd.DataFrame, path: Path) -> None:
    """
    Save metrics DataFrame to CSV file.
    
    Args:
        metrics_df: Metrics DataFrame
        path: Output CSV file path
    """
    # TODO: Implement metrics CSV saving
    pass


def load_metrics_csv(path: Path) -> pd.DataFrame:
    """
    Load metrics DataFrame from CSV file.
    
    Args:
        path: Path to CSV file
        
    Returns:
        Loaded metrics DataFrame
    """
    # TODO: Implement metrics CSV loading
    pass


def save_performance_metrics(performance_data: Dict[str, Any], path: Path) -> None:
    """
    Save performance metrics to JSON file.
    
    Args:
        performance_data: Performance metrics dictionary
        path: Output JSON file path
    """
    # Convert numpy types to JSON-serializable
    serializable_metrics = {}
    for key, value in performance_data.items():
        if isinstance(value, np.ndarray):
            serializable_metrics[key] = value.tolist()
        elif isinstance(value, (np.integer, np.floating)):
            serializable_metrics[key] = value.item()
        elif hasattr(value, 'item'):  # Handle numpy scalars
            serializable_metrics[key] = value.item()
        else:
            serializable_metrics[key] = value
    
    save_json(serializable_metrics, path)


def load_performance_metrics(path: Path) -> Dict[str, Any]:
    """
    Load performance metrics from JSON file.
    
    Args:
        path: Path to performance JSON file
        
    Returns:
        Performance metrics dictionary
    """
    return load_json(path)


def save_calibration_params(params: Dict[str, Any], path: Path) -> None:
    """
    Save calibration parameters to JSON file.
    
    Args:
        params: Calibration parameters dictionary
        path: Output JSON file path
    """
    # TODO: Implement calibration parameters saving
    pass


def load_calibration_params(path: Path) -> Dict[str, Any]:
    """
    Load calibration parameters from JSON file.
    
    Args:
        path: Path to calibration parameters file
        
    Returns:
        Calibration parameters dictionary
    """
    # TODO: Implement calibration parameters loading
    pass


def verify_file_integrity(path: Path, expected_type: str = 'auto') -> bool:
    """
    Verify file integrity and format.
    
    Args:
        path: Path to file to verify
        expected_type: Expected file type ('json', 'csv', 'xz', 'auto')
        
    Returns:
        True if file is valid and readable
    """
    # TODO: Implement file integrity verification
    pass


def get_file_size_mb(path: Path) -> float:
    """
    Get file size in megabytes.
    
    Args:
        path: Path to file
        
    Returns:
        File size in MB
    """
    if path.exists():
        return path.stat().st_size / (1024 * 1024)
    else:
        return 0.0