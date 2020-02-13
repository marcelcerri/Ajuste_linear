# -*- coding: utf-8 -*-
"""
Created on Wed May 22 15:10:19 2019

@author: Marcel Otavio Cerri
"""

#importando as bibliotecas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#importando os dados
importado = pd.read_excel("entrada_dados_linear.xlsx")
importado_np = importado.values
x = importado_np[:,0]
y = importado_np[:,1]

#ajustando o modelo linear
modelo_linear = np.polyfit(x,y,1)
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
lns1 = ax.plot(x,y,'o',label='Dados experimentais')    
lns2 = ax.plot(x,modelo_y,'red',linewidth=2,label='Modelo Linear')
ax.set_title("Ajuste linear de pontos experimentais")  
ax.set_xlabel('Valores de x',weight='bold')               
ax.set_ylabel('Valores de y', weight='bold')
ax.grid(True)                                                   
f.set_figheight(5)                                                 
f.set_figwidth(8)
ax.legend(loc=0)                                                   
f.patch.set_facecolor('white')                                       
plt.style.use('default') 
plt.savefig('Linearizacao.png', dpi=500)  
plt.show()   

#Estimando o coeficiente de determinação
yresid = y - modelo_y
SQresid = sum(yresid**2)
y_tot = y - np.mean(y)
SQtotal = sum(y_tot**2)
R2 = 1 - SQresid/SQtotal

#Impressão dos resultados
print('O coeficiente angular é {:.4f}'.format(modelo_linear[0]))
print('O coeficiente linear é {:.4f}'.format(modelo_linear[1]))
print('O coeficiente de determinação do ajuste é {:.4f}'.format(R2))

#Escrevendo o arquivo de saída xlsx
df_concents= pd.DataFrame({'Valores de x': x, 'Valores de y':y, 'y do modelo':modelo_y, 'coeficiente angular':modelo_linear[0], 'coeficiente linear':modelo_linear[1]})
with pd.ExcelWriter('Saída_de_dados_linear.xlsx') as writer:
    df_concents.to_excel(writer, sheet_name="Output_concent")
    writer.save()