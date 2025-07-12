import os
import sys
import logging
import time
import json
import yaml
from datetime import datetime


class Logger:
    """
    Unified logging class for the ECG-FM system.
    Supports both file and console output and tracks all parameters
    to ensure reproducibility of results.
    """
    
    def __init__(self, output_dir, name="ecgfm", console_level=logging.INFO, file_level=logging.DEBUG):
        """
        Initialize a new logger.
        
        Args:
            output_dir: Directory where log files will be saved
            name: Logger name and prefix for log files
            console_level: Logging level for console output
            file_level: Logging level for file output
        """
        self.output_dir = output_dir
        self.name = name
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Generate log filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(output_dir, f"{name}_{timestamp}.log")
        
        # Configure logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)  # Capture all messages
        self.logger.handlers = []  # Remove any existing handlers
        
        # Formatter for log messages
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(console_level)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(file_level)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
        # Variables for performance tracking
        self.timers = {}
        
        self.logger.info(f"Logger initialized. Log file: {self.log_file}")
    
    def log_config(self, config):
        """
        Record the complete configuration in the log file.
        
        Args:
            config: Configuration object or dictionary
        """
        if hasattr(config, '__dict__'):
            config_dict = config.__dict__
        else:
            config_dict = config
            
        self.logger.info("=== CONFIGURATION ===")
        
        # Format configuration in YAML for better readability
        formatted_config = yaml.dump(config_dict, default_flow_style=False)
        
        # Log each line with indentation for better readability
        for line in formatted_config.split('\n'):
            if line.strip():
                self.logger.info(f"  {line}")
        
        self.logger.info("=====================")
        
        # Also save the configuration to a separate JSON file for reference
        config_file = os.path.join(self.output_dir, f"{self.name}_config.json")
        with open(config_file, 'w') as f:
            json.dump(config_dict, f, indent=2, default=str)
    
    def start_timer(self, task_name):
        """
        Start a timer for a specific task.
        
        Args:
            task_name: Name of the task to time
        """
        self.timers[task_name] = time.time()
        self.logger.info(f"Starting task: {task_name}")
    
    def stop_timer(self, task_name):
        """
        Stop the timer for a task and log the elapsed time.
        
        Args:
            task_name: Task name
            
        Returns:
            Duration in seconds
        """
        if task_name not in self.timers:
            self.logger.warning(f"Timer not found for task: {task_name}")
            return 0
        
        elapsed = time.time() - self.timers[task_name]
        hours, remainder = divmod(elapsed, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        time_str = f"{int(hours)}h {int(minutes)}m {seconds:.2f}s"
        self.logger.info(f"Completed task: {task_name} - Time: {time_str}")
        
        return elapsed
    
    def info(self, message):
        """Log an INFO level message."""
        self.logger.info(message)
    
    def debug(self, message):
        """Log a DEBUG level message."""
        self.logger.debug(message)
    
    def warning(self, message):
        """Log a WARNING level message."""
        self.logger.warning(message)
    
    def error(self, message):
        """Log an ERROR level message."""
        self.logger.error(message)
    
    def critical(self, message):
        """Log a CRITICAL level message."""
        self.logger.critical(message)
    
    def exception(self, message):
        """Log an exception with stack trace."""
        self.logger.exception(message)
    
    def log_step(self, step_name, details=None):
        """
        Log the start of a new processing phase.
        
        Args:
            step_name: Name of the phase
            details: Optional details about the phase
        """
        separator = "=" * 50
        self.logger.info(separator)
        self.logger.info(f"PHASE: {step_name}")
        if details:
            self.logger.info(f"Details: {details}")
        self.logger.info(separator)
    
    def log_result(self, title, result_dict):
        """
        Log a set of results in a structured format.
        
        Args:
            title: Title for the results
            result_dict: Dictionary with results
        """
        self.logger.info(f"=== {title} ===")
        
        # Format and log each result
        for key, value in result_dict.items():
            # Handle numeric values with appropriate formatting
            if isinstance(value, float):
                formatted_value = f"{value:.6f}"
            else:
                formatted_value = str(value)
                
            self.logger.info(f"  {key}: {formatted_value}")
        
        self.logger.info("=" * (len(title) + 8))  # Closing line
    
    def log_memory_usage(self):
        """Log current memory usage."""
        try:
            import psutil
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            self.logger.info(f"Current memory usage: {memory_mb:.2f} MB")
            return memory_mb
        except ImportError:
            self.logger.warning("psutil not installed, cannot log memory usage")
            return None
    
    def log_model_summary(self, model_name, model_params, metrics):
        """
        Log a summary of a model's performance.
        
        Args:
            model_name: Name of the model
            model_params: Dictionary of model parameters
            metrics: Dictionary of performance metrics
        """
        self.logger.info(f"=== MODEL SUMMARY: {model_name} ===")
        
        # Log parameters
        self.logger.info("Parameters:")
        for param, value in model_params.items():
            self.logger.info(f"  {param}: {value}")
        
        # Log metrics
        self.logger.info("Performance metrics:")
        for metric, value in metrics.items():
            if isinstance(value, float):
                self.logger.info(f"  {metric}: {value:.6f}")
            else:
                self.logger.info(f"  {metric}: {value}")
        
        self.logger.info("=" * (len(model_name) + 16))  # Closing line
        
        
    def log_metrics_comparison(self, title, metrics_dict, sort_by=None):
        """
        Log a comparison of metrics across multiple models.
        
        Args:
            title: Title for the comparison
            metrics_dict: Dictionary with model names as keys and dictionaries of metrics as values
            sort_by: Optional metric to sort by (descending order)
        """
        self.logger.info(f"=== {title} ===")
        
        # Extract all unique metric names
        metric_names = set()
        for model_metrics in metrics_dict.values():
            metric_names.update(model_metrics.keys())
        
        # If sort_by is specified, sort the models by that metric
        if sort_by and sort_by in metric_names:
            sorted_models = sorted(
                metrics_dict.items(), 
                key=lambda x: x[1].get(sort_by, 0), 
                reverse=True
            )
        else:
            sorted_models = list(metrics_dict.items())
        
        # Create a header row
        header = ["Model"] + list(metric_names)
        header_str = " | ".join(header)
        self.logger.info(f"  {header_str}")
        self.logger.info(f"  {'-' * len(header_str)}")
        
        # Log each model's metrics
        for model_name, model_metrics in sorted_models:
            row = [model_name]
            for metric in metric_names:
                if metric in model_metrics:
                    value = model_metrics[metric]
                    if isinstance(value, float):
                        formatted_value = f"{value:.4f}"
                    else:
                        formatted_value = str(value)
                    row.append(formatted_value)
                else:
                    row.append("-")
            
            row_str = " | ".join(row)
            self.logger.info(f"  {row_str}")
        
        self.logger.info("=" * (len(title) + 8))  # Closing line