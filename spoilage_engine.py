import numpy as np 
import pandas as pd 
from typing import Dict , List , Tuple 

class spoilage_engine :
    """
    Implementing Q10 rate law for spoilage modelling
    Q10 = factor by which the reaction rate increases for 10 degrees rise

    """

    def __init__ (self ,crop_params :Dict) :
        """
        crop_params = {
        'crop name': str ,
        'T_ref': float ,
        'Q10' : float ,
        'T_optimal' : tuple [float ,float] -(min , max)
        } 
        """

        self.crop_name = crop_params['crop_name']
        self.T_ref = crop_params['T_ref']
        self.Q10 = crop_params['Q10']
        self.SL_ref = crop_params['SL_ref']
        self.T_optimal = crop_params['T_optimal']

        self.total_flu = 0.0 #fractional_life_used
        self.timestep = 5 #mins 

    def calculate_flu (self , T_current : float) ->float:
        """
        Calculating fractional life used for the current timestep
        FLU = (timestep/SL_ref)*Q10^((T_current - T_ref)/10)
        FLU here is how ,much of the product is consumed ,scaled up by how fast/ slow the spoilage occurs at the current temperature

        """
        delta_T = T_current - self.T_ref

        #Q10 rate change with temperature
        rate_multiplier = self.Q10 ** (delta_T /10.0)

        #Fractional life consumed in this timestep : 
        timestep_hours  =  self.timestep / 60 
        flu = (timestep_hours / self.SL_ref)*rate_multiplier 


        return flu

    def update_spoilage (self , T_current : float) -> Dict :
        """
        update cumulative spoilage and return current status

        """
        flu = self.calculate_flu(T_current)
        self.total_flu += flu 

        #remaining shelf life in percentage

        rsl_percent = max (0 , (1-self.total_flu)*100)

        #remaining in shelf life in hours 

        rsl_hours = max(0 , (1-self.total_flu)*self.SL_ref )

        if rsl_percent <= 0 :
            status = "SPOILED"
        elif rsl_percent <= 20 :
            status = 'CRITICAL'
        elif rsl_percent <= 50 :
            status = 'WARNING'
        else:
            status = 'GOOD'


        return{

            'flu_current' : flu ,
            'flu_total' : self.total_flu ,
            'rsl_percent': rsl_percent ,
            'rsl_hours' : rsl_hours ,
            'status' : status ,
            'temperature' : T_current ,
            'in_optimal_range' :  self.T_optimal[0] <= T_current <= self.T_optimal[1]
        }                           

    def reset(self):
            """
            Reset for new shipment batch
            """
            self.total_flu = 0.0


