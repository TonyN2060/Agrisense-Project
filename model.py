import pandas as pd
import numpy as np
from typing import Dict , Tuple , Optional
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
from sklearn.metrics import f1_score ,classification_report, accuracy_score
from sklearn.model_selection import cross_val_score


Q10_params ={
    'tomato' : {
        'crop_name' : 'Tomato' ,
        'T_ref' : 13.0,
        'SL_ref' : 336.0,
        'Q10': 2.5 ,
        'T_optimal' : (10.0 , 15.0) ,
        'H_optimal' : (85.0 ,95.0) ,
        'CO2_optimal' : (300 , 400)
    
    },

    'banana' : {
        'crop_name' : 'Banana' ,
        'T_ref' : 13.5 ,
        'SL_ref' : 720.0 ,
        'Q10' : 2.2 ,
        'T_optimal' : (13.0 ,14.0) ,
        'H_optimal' : (85.0 , 95.0) ,
        'CO2_optimal' : (300 , 450) ,

    } , 

    'orange' : {
        'crop_name' : 'Orange' ,
        'T_ref' : 5.0 ,
        'SL_ref' : 2160.0 ,
        'Q10' : 2.0 ,
        'T_optimal' : (3.0 , 8.0) ,
        'H_optimal' : (85.0 , 90.0) ,
        'C02_optimal' : (300 ,500) 

    },

    'pineapple' : {
        'crop_name' : 'Pineapple' ,
        'T_ref' : 7.0 ,
        'SL_ref' : 960.0 ,
        'Q10' : 2.3 ,
        'T_optimal' : (7.0 , 13.0),
        'H_optimal' : (85.0 , 90.0),
        'CO2_optimal':(300 , 400)
    }
}

#DATA PROCESSOR 
class CropAnalyzer :

    """
    Identify optimal range for crops
    Train a random Forest Classifier to predict crop status
    Calibrate Q10 params with real world data
    """

    def __init__(self , csv_path :str):
        self_df = pd.read_csv(csv_path)
        self.clean_data()
        self.classifier = None 
        self.label_encoder = LabelEncoder()
        self.optimal_ranges = {}

    def clean_data(self):
        self.df['Fruit'] = self.df['Fruit'].str.lower().str.strip()

        self.df['Class'] = self.df['Class'].str.lower().str.strip()
        self.df['Class_Binary'] = (self.df['Class']=='good').astype(int)

        self.df.rename(columns = {
            'Temp':'temperature' ,
            'Humid (%)' : 'humidity' ,
            'Light (Fux)': 'light' ,
            'CO2 (ppm)' : 'co2' ,
            'Fruit' : 'crop'
        } ,inplace = True)

        print(f"Loaded {len(self.df)} records for the following crops {self.df['crops'].unique()}")


    def analyze_optimal_ranges(self) -> Dict :
        """
        Extracts optimal ranges (crops in good condition)
        """    
        for crop in self.df['crop'].unique():
            crop_data = self.df[self.df['crop'] == crop]
            good_data = crop_data[crop_data['Class_Binary'] ==1]


        self.optimal_ranges[crop] = {
            'temperature' : {
                'mean' : good_data['temperature'].mean(),
                'std' : good_data['temperature'].std(),
                'min' : good_data['temperature'].quantile(0.1),
                'max' : good_data['temperature'].quantile(0.9),
            },
            'humidity' : {
                'mean' : good_data['humidity'].mean(),
                'std' : good_data['humidity'].std(),
                'min' : good_data['humidity'].quantile(0.1),
                'max' : good_data['humidity'].quantile(0.9),
            },
            'co2' : {
                'mean' : good_data['co2'].mean(),
                'std' : good_data['co2'].std(),
                'min' : good_data['co2'].quantile(0.1),
                'max' : good_data['co2'].quantile(0.9)
            },
            'light' :{
                'mean' : good_data['light'].mean(),
                'std' : good_data['light'].std(),
                'min' : good_data['light'].quantile(0.1) ,
                'max' : good_data['light'].quantile (0.9),
            }
        }

        return self.optimal_ranges
    
    def print_optimal_ranges(self):
        print("\n" + "="*70)
        print("EMPIRICAL OPTIMAL RANGES (from your CSV data)")
        print("="*70)
        
        for crop, ranges in self.optimal_ranges.items():
            print(f"\n{crop.upper()}:")
            print(f"  Temperature: {ranges['temperature']['min']:.1f} - {ranges['temperature']['max']:.1f}°C")
            print(f"  Humidity:    {ranges['humidity']['min']:.1f} - {ranges['humidity']['max']:.1f}%")
            print(f"  CO2:         {ranges['co2']['min']:.0f} - {ranges['co2']['max']:.0f} ppm")
            print(f"  Light:       {ranges['light']['min']:.1f} - {ranges['light']['max']:.1f} lux")
    
                    

    def train_classifier(self ,use_optuna :bool = True , n_trials :int = 100) -> float :
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import classification_report , accuracy_score

        #we're choosing to have our inputs / targets in dataframe format for easy debugging
        feature_cols = ['temperature' , 'humidity' , 'co2' , 'light' ]
        X = self.df[feature_cols]
        y = self.df['Class_Binary']

        #adding the crops as a categorical_feature
        crop_encoded = self.label_encoder.fit_transform(self.df['crop'])
        X['crop_encoded'] = crop_encoded

        #train/test splitting
        X_train , X_test , y_train , y_test = train_test_split(X ,y, test_size = 0.25,
                                                               random_state = 619,
                                                               stratify = y)


        if use_optuna :
            print(f"Hypeparameter tuning with optuna running {n_trials}")

            import optuna
            from optuna.samplers import TPESampler

            self.X_train = X_train 
            self.X_test  = X_test
            self.y_train = y_train
            self.y_test = y_test

            study = optuna.create_study(
                direction = 'maximize' ,
                sampler = TPESampler(seed = 42),
                study_name = 'Random_Forest_Optimizer'
            )

            study.optimize(
                self._optuna_objective ,
                n_trials = n_trials ,
                show_progress_bar = True ,
                n_jobs = -1 #using all cpu cores

            )


            print(f"Optimization Complete")
            print(f"Best trial is {study.best_trial.number}")
            print(f"Best accuracy is :{study.best_value}")
            for key , value in study.best_params.items() :
                print(f"{key} : {value} ")

            #training the final model with the best parameters :
            best_params  = study.best_params
            self.classifier = RandomForestClassifier(
                n_estimators = best_params['n_estimators'],
                max_depth = best_params['max_depth'],
                min_samples_split = best_params['min_samples_split'],
                min_samples_leaf = best_params['min_samples_leaf'],
                max_features = best_params['max_features'],
                bootstrap = best_params['bootstrap'],
                random_state = 42,
                class_weight = 'balanced',
                n_jobs = -1

            )    
            
            self.optuna_study = study

        else :
            print("Training with default params")


            self.classifier = RandomForestClassifier(
                n_estimators = 100 ,
                max_depth = 10 ,
                random_state = 42 ,
                class_weight = 'balanced' ,
                n_jobs = -1

            )

            self.classifier.fit(X_train , y_train)

            y_pred = self.classifier.predict(X_test)
            y_pred_proba = self.classifier.predict_proba(X_test)

            accuracy = accuracy_score (y_test , y_pred)
            f1 = f1_score(y_test ,y_pred)


            print(f"Test Accuracy is : {accuracy}")
            print(f"Test F1 score is {f1}")

            cv_scores = cross_val_score(self.classifier , X_train , y_train , cv = 5 , scoring  = 'accuracy')
            print(f"CV Accuracy is {cv_scores.mean()}")

            print("\nClassification Report :")
            print(classification_report(y_test ,y_pred , target_names = ['Bad' , 'Good']))

            feature_names = feature_cols + ['crop_type']
            importances = self.classifier.feature_importances_

            print("\nFeature Importance:")
            for name, importance in sorted(zip(feature_names ,importances),
                                           key = lambda x : x[1] ,reverse = True) :
                print(f"{name} :{importance}")


            if hasattr(self , 'X_train'):
                del self._X_train , self._y_train ,self._X_test , self._y_test

            return accuracy


    def optuna_objective(self, trial) : 
        #objective_function for hyperparameter optimization

        params = {
            'n_estimators' : trial.suggest_int('n_estimators' , 50 , 500) ,
            'max_depth'  : trial.suggest_int('max_depth' ,  3 , 30) ,
            'min_samples_split' : trial.suggest_int('min_samples_split' , 2 , 20) ,
            'min_samples_leaf' : trial.suggest_int('min_samples_leaf' , 1, 10) ,
            'max_features' : trial.suggest_categorical('max_features' , ['sqrt' ,'log2', None]),
            'bootstrap' : trial.suggest_categorical('bootstrap' , [True , False]),
        }            

        model  =  RandomForestClassifier(
            **params,
            random_state = 42 ,
            class_weight = 'balanced',
            n_jobs = -1
        )

        model.fit(self._X_train , self._y_train)

        y_pred = model.predict(X_test)
        score = f1_score(self._y_test, y_pred)

        return score
    
    def plot_optuna_results(self):
        """
        Plot Optuna optimization results.
        Requires matplotlib and optuna.visualization.
        """
        if not hasattr(self, 'optuna_study'):
            print("No Optuna study found. Train with use_optuna=True first.")
            return
        
        try:
            import matplotlib.pyplot as plt
            from optuna.visualization import plot_optimization_history, plot_param_importances
            
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))
            
            # Plot 1: Optimization history
            ax1 = axes[0]
            trials = self.optuna_study.trials
            ax1.plot([t.number for t in trials], [t.value for t in trials], 'o-')
            ax1.axhline(y=self.optuna_study.best_value, color='r', linestyle='--', 
                       label=f'Best: {self.optuna_study.best_value:.4f}')
            ax1.set_xlabel('Trial Number')
            ax1.set_ylabel('F1 Score')
            ax1.set_title('Optimization History')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Parameter importances
            ax2 = axes[1]
            importances = optuna.importance.get_param_importances(self.optuna_study)
            params = list(importances.keys())
            values = list(importances.values())
            ax2.barh(params, values)
            ax2.set_xlabel('Importance')
            ax2.set_title('Hyperparameter Importance')
            ax2.grid(True, alpha=0.3, axis='x')
            
            plt.tight_layout()
            plt.savefig('optuna_optimization_results.png', dpi=150, bbox_inches='tight')
            print("\n✓ Optimization plots saved to 'optuna_optimization_results.png'")
            plt.show()
            
        except ImportError:
            print("Install matplotlib for visualization: pip install matplotlib")
        except Exception as e:
            print(f"Error plotting results: {e}")

    def predict_quality(self , crop : str , temp : float , humidity : float , 
                        co2 : float  ,light : float) -> Tuple[str , float]:
        """
        Predicts of the conditions will produce good/ bad outcomes
        Returns (prediction,confidence)
        """        

        if self.classifier is None :
            raise ValueError("Classifier Not trained, Call the classifier first")
        
        crop_code = self.label_encoder.transform([crop.lower()])[0]
        X = np.array([[temp , humidity , co2 , light , crop_code]])

        prediction = self.classifier.predict(X)[0]
        probability = self.classifier.predict_proba(X)[0]

        result = "GOOD" if prediction == 1 else "BAD"
        confidence = probability [prediction]

        return result , confidence
    
    def save_model(self , path :  str = 'empirical_classifier.pkl'):
        import os
        os.makedirs(os.path.dirname(path) , exist_ok = True)
        joblib.dump({
            'classifier' : self.classifier, 
            'label_encoder' : self.label_encoder , 
            'optimal-rangers' : self.optimal_ranges
        }, path)

        print(f"Model saved to {path}")

    def load_model(self, path: str = 'models/empirical_classifier.pkl'):
        """Load a trained classifier."""
        data = joblib.load(path)
        self.classifier = data['classifier']
        self.label_encoder = data['label_encoder']
        self.optimal_ranges = data['optimal_ranges']
        print(f"Model loaded from {path}")  


        ##HYBRID SYSTEM _COMBINES PHYSICS + EMPRICAL ML   


class HybridCropDatabase :

    """
    Integrates our Q10  models with empirical ML Preds
    """        
    
    def __init__(self , csv_path :  str):
        self.q10_params = Q10_params 
        self.empirical =  CropAnalyzer(csv_path)

        #Analyzing
        print("Analyzing empirical data")
        self.empirical.analyze_optimal_ranges()
        self.empirical.print_optimal_ranges()

        print("\nTraining classifier")
        self.empiricial.train_classifier()

    def get_crop_params(self , crop :str) -> Dict :
        crop_key = crop.lower()


        if crop_key not in self.q10_params :
            raise ValueError(f"{crop } unknown / unavailable")


        params = self.q10_params[crop_key].copy()   

        if crop_key in self.empirical.optimal_ranges :
            emp_ranges = self.empirical.optimal_ranges[crop_key]

            params['T_optimal'] = (
                emp_ranges['temperature']['min'] ,
                emp_ranges['temperature']['max']

            )
            params['H_optimal'] = (
                emp_ranges['humidity']['min'],
                emp_ranges['humidity']['max']
            )

            params['CO2_optimal'] = (
                emp_ranges['co2']['min'],
                emp_ranges['co2']['max']
            )
            
            params['empirical_calibrated'] = True
        else :
            params['empirical_calibrated'] = False 

        return params

    def evaluate_conditions(self , crop:str , temp : float , humidity : float ,
                            co2 : float , light : float) -> Dict:
        """
        Evaluating....
        """    
        crop_key = crop.lower()
        params = self.get_crop_params(crop)

        temp_ok = params['T_optimal'][0] <= temp <= params['T_optimal'][1]
        humidity_ok = params['H_optimal'][0] <= humidity <= params['H_optimal'][1]
        co2_ok = params['CO2_optimal'][0] <= co2 <= params['CO2_optimal'][1]physics_score = sum([temp_ok, humidity_ok, co2_ok]) / 3.0
        
        # 2. ML-based prediction
        ml_prediction, ml_confidence = self.empirical.predict_quality(
            crop, temp, humidity, co2, light
        )
        
        # 3. Combined assessment
        if physics_score >= 0.67 and ml_prediction == "GOOD":
            overall_status = "OPTIMAL"
            confidence = ml_confidence
        elif physics_score >= 0.67 or ml_prediction == "GOOD":
            overall_status = "ACCEPTABLE"
            confidence = ml_confidence * 0.7
        else:
            overall_status = "POOR"
            confidence = ml_confidence
        
        return {
            'crop': crop,
            'overall_status': overall_status,
            'confidence': confidence,
            'physics_check': {
                'temperature_ok': temp_ok,
                'humidity_ok': humidity_ok,
                'co2_ok': co2_ok,
                'score': physics_score
            },
            'ml_prediction': {
                'prediction': ml_prediction,
                'confidence': ml_confidence
            },
            'recommendations': self._generate_recommendations(
                crop, temp, humidity, co2, params
            )
        }
    
    def _generate_recommendations(self, crop: str, temp: float, 
                                 humidity: float, co2: float, params: Dict) -> list:
        """Recommendations."""
        recommendations = []
        
        if temp < params['T_optimal'][0]:
            recommendations.append(f" Temperature too low ({temp}°C). Raise to {params['T_optimal'][0]}°C")
        elif temp > params['T_optimal'][1]:
            recommendations.append(f" Temperature too high ({temp}°C). Lower to {params['T_optimal'][1]}°C")
        
        if humidity < params['H_optimal'][0]:
            recommendations.append(f" Humidity too low ({humidity}%). Increase to {params['H_optimal'][0]}%")
        elif humidity > params['H_optimal'][1]:
            recommendations.append(f" Humidity too high ({humidity}%). Reduce to {params['H_optimal'][1]}%")
        
        if co2 < params['CO2_optimal'][0]:
            recommendations.append(f" CO2 too low ({co2} ppm). Increase ventilation management")
        elif co2 > params['CO2_optimal'][1]:
            recommendations.append(f" CO2 too high ({co2} ppm). Improve ventilation")
        
        if not recommendations:
            recommendations.append("✅ All parameters within optimal range")
        
        return recommendations



        







