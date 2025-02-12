import numpy as np

class LinearRegression:
    def __init__(self):
        self.theta = None

    def normal_equation(self, X: np.ndarray, y: np.ndarray):
        """Calcula os parâmetros usando a equação normal.
        
        Args:
            X: Matriz de features (n amostras x m features)
            y: Vetor target (n amostras)
        
        Returns:
            theta: Vetor de parâmetros (m+1,)
        """
        # Adiciona a coluna de 1's para o bias
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        self.theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
        return self.theta

    def predict(self, X: np.ndarray):
        """Realiza as previsões usando os parâmetros calculados.
        
        Args:
            X: Matriz de features (n amostras x m features)
        
        Returns:
            Predições: Vetor de previsões
        """
        X_b = np.c_[np.ones((X.shape[0], 1)), X]
        return X_b.dot(self.theta)