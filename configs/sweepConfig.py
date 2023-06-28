# First we define our configuration
sweep_config = {
    
    'method': 'random',    # We specify the search strategy.
    
    'metric': {            # We set the metric and the goal (minimize/maximize).
        'name': 'loss',
        'goal': 'minimize'
        },

    # Here we need to set all the parameters we want to tune. Notice that the key
    # 'parameters' is mandatory and can't be omitted.
    'parameters': {
        #'value': x
        # \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
        # Hyperparameters that need additional hyperparameters, e.g. optimizer and scheduler.
        # In this case we need to specify both the name and the settings related to each.        
        'optimizer': {
            'parameters': {
                           'name': {'values': ['Adam', 'SGD']},
                           'settings': {'parameters': {                           
                                                       'lr'      : {'max': 0.1,   
                                                                    'min': 0.01},
                                                       'momentum': {'max': 1,
                                                                    'min': 0}
                                                       }
                                        }
                           }
                      },

        'scheduler': {
            'parameters':{
                          'name'  : {'values': ['ConstantLR', 'ExponentialLR']},
                          'settings': {'parameters':{      
                                                     'gamma' : {'max': 1,
                                                                'min': 0.1},
                                                     'factor': {'max': 1,
                                                                'min': 0.1}}
                                       }
                          }
                      }
        }
    }