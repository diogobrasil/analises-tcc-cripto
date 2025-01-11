# feature_engineering.py
##Código base, feito com auxilio do gpt e com base nos vídeos presentes no outro arquivo. Falta mudar muita coisa, testar a efetividade, além de pegar as ideias dos vídeos (uns são meio grandes, ent infelizmente não consegui ver todos ainda)
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif

class FeatureEngineering:
    def __init__(self):
        self.imputers = {}
        self.scaler = None
        self.encoders = {}
        self.selected_features = None

    def handle_missing_values(self, df, strategy="mean"):
        #esse código serve para tratar os valores ausentes!!
        for column in df.columns:
            imputer = SimpleImputer(strategy=strategy)
            df[column] = imputer.fit_transform(df[[column]])
            self.imputers[column] = imputer
        return df

    def encode_categorical(self, df, categorical_columns):
        #Codificar variáveis categóricas com One-Hot Encoding (irei estudar mais sobre isso para saber como funfa, apenas vi sendo usado)
        for column in categorical_columns:
            encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
            encoded = encoder.fit_transform(df[[column]])
            encoded_df = pd.DataFrame(
                encoded, 
                columns=[f"{column}_{cat}" for cat in encoder.categories_[0]]
            )
            df = pd.concat([df.drop(column, axis=1), encoded_df], axis=1)
            self.encoders[column] = encoder
        return df

    def scale_features(self, df, numeric_columns):
        #Realiza o processo de Escalar variáveis numéricas
        self.scaler = StandardScaler()
        df[numeric_columns] = self.scaler.fit_transform(df[numeric_columns])
        return df

    def feature_selection(self, df, target, k=10):
        #seleciona as melhores caracteristicas
        selector = SelectKBest(score_func=f_classif, k=k)
        X_new = selector.fit_transform(df, target)
        self.selected_features = df.columns[selector.get_support()]
        return pd.DataFrame(X_new, columns=self.selected_features)

    def transform(self, df):
        #aplica as transformações predefinidas ao conjunto de dados
        for column, imputer in self.imputers.items():
            if column in df.columns:
                df[column] = imputer.transform(df[[column]])
        for column, encoder in self.encoders.items():
            if column in df.columns:
                encoded = encoder.transform(df[[column]])
                encoded_df = pd.DataFrame(
                    encoded, 
                    columns=[f"{column}_{cat}" for cat in encoder.categories_[0]]
                )
                df = pd.concat([df.drop(column, axis=1), encoded_df], axis=1)
        if self.scaler:
            numeric_columns = [col for col in df.columns if np.issubdtype(df[col].dtype, np.number)]
            df[numeric_columns] = self.scaler.transform(df[numeric_columns])
        if self.selected_features is not None:
            df = df[self.selected_features]
        return df

 # !! Adicionar elementos faltantes (pesquisar sobre multicolinearidade e implementações nos vídeos)
