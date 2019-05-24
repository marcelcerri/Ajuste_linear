# -*- coding: utf-8 -*-
"""
Created on Wed May 22 15:10:19 2019

@author: Marcel Otavio Cerri
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
#importando os dados
importado = pd.read_excel("media_desvio_dados.xlsx")
importado_np = importado.values
x = importado_np[:,0]
y1 = importado_np[:,1]
y2 = importado_np[:,2]
y3 = importado_np[:,3]

a = np.array([y1,y2,y3])
b = np.transpose(a)

y_media = np.mean(b, axis=1)
y_std = np.std(b, axis =1)

mean, sigma = np.mean(b, axis = 1), np.std(b, axis = 1)

conf_int = stats.norm.interval(0.95, loc=mean, 
    scale=sigma)

menor = np.array(conf_int[0])
maior = np.array(conf_int[1])

 #ajustando o modelo linear
modelo_linear = np.polyfit(x,y_media,1)
modelo_y = np.polyval(modelo_linear, x)
## Comando para determinar o tamanho segundo o qual os textos grafados no gráfico serão impressos na tela:
SMALL_SIZE = 14                        
MEDIUM_SIZE = 20                       
BIGGER_SIZE = 20   
plt.rc('font', size=SMALL_SIZE)          
plt.rc('axes', titlesize=SMALL_SIZE)     
plt.rc('axes', labelsize=SMALL_SIZE)    
plt.rc('xtick', labelsize=SMALL_SIZE)    
plt.rc('ytick', labelsize=SMALL_SIZE)    
plt.rc('legend', fontsize=SMALL_SIZE)    
plt.rc('figure', titlesize=BIGGER_SIZE)  

#construindo o gráfico
f = plt.figure()     
ax = f.add_subplot(111)                                                
lns1 = ax.plot(x,y_media,'o',label='Dados experimentais')    
lns2 = ax.plot(x,modelo_y,'b',linewidth=2,label='Modelo Linear')
ax.set_title("Ajuste linear de pontos experimentais") 
ax.fill_between(x, menor, maior, alpha=0.3, label = 'Intervalo de confiança 95%') 
ax.set_xlabel('Valores de x',weight='bold')               
ax.set_ylabel('Valores de y', weight='bold')
ax.grid(True)                                                   
f.set_figheight(5)                                                 
f.set_figwidth(8)
ax.legend(loc=0)                                                   
f.patch.set_facecolor('white')                                       
plt.style.use('default') 
plt.savefig('Linearizacao_com_media.png', dpi=400)  
plt.show()   
#Estimando o coeficiente de determinação
yresid = y_media - modelo_y
SQresid = sum(yresid**2)
y_tot = y_media - np.mean(y_media)
SQtotal = sum(y_tot**2)
R2 = 1 - SQresid/SQtotal
#Impressão dos resultados
print('O coeficiente angular é {:.4f}'.format(modelo_linear[0]))
print('O coeficiente linear é {:.4f}'.format(modelo_linear[1]))
print('O coeficiente de determinação do ajuste é {:.4f}'.format(R2))
#Escrevendo o arquivo de saída xlsx
df_concents= pd.DataFrame({'Valores de x': x, 'Valores de y media':y_media, 'y do modelo':modelo_y, 'coeficiente angular':modelo_linear[0], 'coeficiente linear':modelo_linear[1]})
with pd.ExcelWriter('Saída_de_dados_linear_media.xlsx') as writer:
    df_concents.to_excel(writer, sheet_name="Output_concent")
    writer.save()