from .XGBoostClassifier import XGBoostClassifier
from .RFClassifier import RandomForestClassifier
from .SVMClassifier import SVMClassifier
from .NNClassifier import NNClassifier
from .DecisionTreeClassifier import DecisionTreeClassifier
from .LogisticRegressionClassifier import LogisticRegressionClassifier


class ClassifierFactory:
    """
    Factory class for creating classifier instances.
    """
    
    @staticmethod
    def create_classifier(classifier_type, logger=None):
        """
        Factory method for creating classifier instances.
        
        Args:
            classifier_type (str): Type of classifier to create
                ('xgboost', 'random_forest', 'svm', 'nn')
            logger: Logger instance for tracking progress (optional)
                
        Returns:
            Instance of the requested classifier
            
        Raises:
            ValueError: If the classifier type is not supported
        """
        if logger:
            logger.info(f"Creating classifier of type: {classifier_type}")
        
        classifier = None
        
        if classifier_type.lower() == 'xgboost':
            classifier = XGBoostClassifier(logger=logger)
        elif classifier_type.lower() in ['random_forest', 'rf']:
            classifier = RandomForestClassifier(logger=logger)
        elif classifier_type.lower() == 'svm':
            classifier = SVMClassifier(logger=logger)
        elif classifier_type.lower() in ['nn', 'neural_network']:
            classifier = NNClassifier(logger=logger)
        elif classifier_type.lower() in ['dt', 'decision_tree', 'decision tree', 'decisiontree']:
            classifier = DecisionTreeClassifier(logger=logger)
        elif classifier_type.lower() in ['logistic_regression', 'lr']:
            classifier = LogisticRegressionClassifier(logger=logger)
        else:
            error_msg = f"Unsupported classifier type: {classifier_type}"
            if logger:
                logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Log successful creation
        if logger:
            logger.debug(f"Successfully created {classifier_type} classifier")
        
        return classifier