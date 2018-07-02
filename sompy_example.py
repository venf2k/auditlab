import numpy as np
from sompy.sompy import SOMFactory
import pandas as pd

# lettura dei dati dal csv
input_data = pd.read_csv("data/DatasetNuovoConAAeBFiltrato.csv", delimiter = ";")
trunc_data = input_data.drop(["Ticker","Issuer","Rating"],axis=1)

# Normalizzazione dataset per colonne
#trunc_data = (trunc_data - trunc_data.min() ) / ( trunc_data.max() - trunc_data.min())

#Ottengo la lista degli Header, la salvo in names e li droppo dal file del dataset
data=trunc_data
names = list(trunc_data.columns.values)
print( "FEATURES: ", ", ".join(names))
data = data.values

#Alleno la SOM
#msz = calculate_msz(data)
sm = SOMFactory().build(data, normalization = 'var', initialization='random', component_names=names)
sm.train(n_job=1, verbose='info', train_rough_len=2, train_finetune_len=300)

#Calcolo dell'errore topografico e di quantizzazione
topographic_error = sm.calculate_topographic_error()
quantization_error = np.mean(sm._bmu[1])
print ("Topographic error = %s; Quantization error = %s" % (topographic_error, quantization_error))

#Visualizzazione delle component planes
from sompy.visualization.mapview import View2D
view2D  = View2D(10,10,"rand data",text_size=10)
view2D.show(sm, col_sz=4, which_dim="all", desnormalize=True)

#Visualizzazione delle BMUHitsview
from sompy.visualization.bmuhits import BmuHitsView
vhts  = BmuHitsView(4,4,"Hits Map",text_size=12)
vhts.show(sm, anotate=True, onlyzeros=False, labelsize=12, cmap="Greys", logaritmic=False)

#Visualizzazione delle HitMapView
from sompy.visualization.hitmap import HitMapView
sm.cluster(4) #Indica il numero di cluster per il raggruppamento
hits  = HitMapView(20,20,"Clustering",text_size=12)
a=hits.show(sm)

