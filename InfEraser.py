import numpy as np
import numpy as np

def inf_eraser(gaus_comp):
    def inf_eraser(gaus_comp):
        '''
        Funzione che trova gli infiniti nel ndarray e li sostituisce con il massimo o il minimo valore della colonna in cui si trovano.
        '''
        inf_index_max = []
        inf_index_min = []

        ###searching for inf values: 
        for j in range(0,4):
            for i in range(0,len(gaus_comp),1):
                if gaus_comp[i,j] == np.inf:
                    gaus_comp[i,j] = 0      #replace inf with 0
                    inf_index_max.append(([i,j]))   #save index of inf values

                if gaus_comp[i,j] == -np.inf:
                    inf_index_min.append(([i,j]))   #save index of -inf values
                    gaus_comp[i,j] = 0    #replace -inf with 0


        if inf_index_max:  #if there are inf values
            max = np.max(gaus_comp,axis=0)   #max value of each column
            for i in range(0,len(inf_index_max),1):  #for each inf value
                gaus_comp[inf_index_max[i][0],inf_index_max[i][1]] = max[inf_index_max[i][1]]      #replace inf with max value of the column

        if inf_index_min:
            min = np.min(gaus_comp,axis=0)     #min value of each column
            for i in range(0,len(inf_index_min),1):   #for each -inf value
                gaus_comp[inf_index_min[i][0],inf_index_min[i][1]] = min[inf_index_min[i][1]]       #replace -inf with min value of the column
        return gaus_comp
