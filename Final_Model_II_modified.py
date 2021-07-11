
import numpy as np
from sklearn.metrics import mean_squared_error
import pandas as pd



N = 10000000


#data feeding
df = pd.read_csv("Covid_India_II.csv", header = 0) # dataset
# df.head()


available = len(df.index)
# print("Available", available, "days")


start_date = df['Date'].iloc[0]   # first index take push korlam array te.
exposed= df['Exposed'].values.tolist()
isolated = df['Isolated'].values.tolist()
infected = df['Infected'].values.tolist()  # shob gula ek ekta col k tar nijeder moddhe rakhlam..
Di = df['Di'].values.tolist()
Ri = df['Ri'].values.tolist()
unwilling = df['Unwilling'].values.tolist() 
Du = df['Du'].values.tolist() # shob gula ek ekta col k tar nijeder moddhe rakhlam.. 
Ru = df['Ru'].values.tolist() # shob gula ek ekta col k tar nijeder moddhe rakhlam..



#Helper Function

t_max = available + 15

date = np.array('2020-03-24', dtype=np.datetime64) 
dates = date + np.arange(t_max)
str_dates = []
for i in dates:
    str_dates.append(str(i))
# print(str_dates)
# print(len(str_dates))

def seird_model(init_vals, params, t):

    DATE_0, S_0, E_0, Is_0, I_0, Di_0, Ri_0, U_0, Du_0, Ru_0 = init_vals  # jehetu ajk r real data niye kaj kortesi tai date takeo mention korte hobe.
    DATE, S, E, Is, I, Di, Ri, U, Du, Ru = [DATE_0] , [S_0] , [E_0] ,  [Is_0] , [I_0] , [Di_0], [Ri_0] , [U_0], [Du_0] , [Ru_0]
    
    alpha, rho, gamma, delta, miu, beta, sigma, tao = params
    
    ##### Create next t days ######
    date = np.array(DATE, dtype=np.datetime64)
    dates = date + np.arange(len(t))
    str_dates = []
    for i in dates:
        str_dates.append(str(i))
        
    ##### End creating t days #####
    for tic in t[1:]:                # 1st din er initial value deya ache tai 1 theke shuru kore baki shob add dilam.

        DATE.append(str_dates[tic])

        next_S = S[-1] - ( alpha * (S[-1]  * E[-1] / N ) )  # Susceptible
        next_E = E[-1] + ( alpha * (S[-1]  * E[-1] / N ) ) - ( rho * Is[-1] ) - ( gamma * I[-1] )  - ( beta * U[-1] ) # Exposed
        next_Is = Is[-1] + (rho *  Is[-1] )  #isolation
        next_I = I[-1] + ( gamma * I[-1] ) - ( delta * I[-1] ) - ( miu * I[-1] ) # Infected
        next_Di = Di[-1] + ( delta * I[-1] ) 
        next_Ri = Ri[-1] + ( miu * I[-1] )
        next_U = U[-1] + ( beta * U[-1] ) - ( sigma * U[-1] ) - ( tao * U[-1] )  # Unknown 
        next_Du = Du[-1] + ( sigma * U[-1] )
        next_Ru = Ru[-1] + ( tao * U[-1] )
        
        
        S.append(next_S)
        E.append(next_E)
        Is.append(next_Is)
        I.append(next_I)
        Di.append(next_Di)
        Ri.append(next_Ri)
        U.append(next_U)
        Du.append(next_Du)
        Ru.append(next_Ru)
        

    return np.stack([DATE, S, E, Is, I, Di, Ri, U, Du, Ru]).T



# Spliting the data for one day rolling window approach

train_min = 1  # diffrence ber korar jnno ami 1 din er diffrence k niye cal kortesi
train_max = available + 1

exposed_train = []
isolated_train = []
infected_train = []
Di_train = []
Ri_train = []
unwilling_train = []
Du_train = []
Ru_train = []


for i in range(train_min+1, train_max):
    j = i - 2
    exposed_train.append(exposed[j:i])
    isolated_train.append(isolated[j:i])
    infected_train.append(infected[j:i])
    Di_train.append(Di[j:i])
    Ri_train.append(Ri[j:i])
    unwilling_train.append(unwilling[j:i])
    Du_train.append(Du[j:i])
    Ru_train.append(Ru[j:i])


# for i in range(len(exposed_train)): 
#     print(i,exposed_train[i])





t2 = np.arange(0, 15, 1)   # 0 - 3 PORJNTO JABE.. R PROTTEKBAR 1 KORE KORE BARBE. 0 initial value tar mane 0 theke shuru hoye aro 2 din porjnto simulation cholbe.
# print(t2)



last5_vals = []
last5_params = []


t_incub = 15  # Assumption 30 days

for sample in range(len(exposed_train)):
    
    e_0 = exposed_train[sample][0]
    e_1 = exposed_train[sample][1]
    e_diff = e_1 - e_0
    
    is_0 = isolated_train[sample][0]
    is_1 = isolated_train[sample][1]
    is_diff = is_1 - is_0

    i_0 = infected_train[sample][0]
    i_1 = infected_train[sample][1]
    i_diff = i_1 - i_0
    
    di_0 = Di_train[sample][0]
    di_1 = Di_train[sample][1]
    di_diff = di_1 - di_0
    
    ri_0 = Ri_train[sample][0]
    ri_1 = Ri_train[sample][1]
    ri_diff = ri_1 - ri_0
    
    u_0 = unwilling_train[sample][0]
    u_1 = unwilling_train[sample][1]
    u_diff = u_1 - u_0

    du_0 = Du_train[sample][0]
    du_1 = Du_train[sample][1]
    du_diff = du_1 - du_0

    ru_0 = Ru_train[sample][0]
    ru_1 = Ru_train[sample][1]
    ru_diff = ru_1 - ru_0
    


    s_0 = N - is_0 - di_0 - ri_0 - du_0 - ru_0 
    
    alpha = (( e_diff + is_diff + i_diff + u_diff + di_diff + ri_diff + du_diff + ru_diff) / (s_0 * e_0 )) * N  # alpha
    rho = ( is_diff / is_0 )      # rho
    gamma = ( i_diff + di_diff + ri_diff) / i_0    # gamma
    delta = ( di_diff / i_0 )   # delta
    miu =  ( ri_diff / i_0 )     # miu
    beta =  ( ( u_diff + du_diff + ru_diff ) / u_0 )     # beta
    sigma = ( du_diff / u_0 )     # sigma
    tao = ( ru_diff / u_0 )      # tao
    

    
    init_vals = str_dates[sample], s_0, e_0, is_0, i_0, di_0, ri_0, u_0, du_0, ru_0  
    params = alpha, rho, gamma, delta, miu, beta, sigma, tao
    pred = seird_model(init_vals, params, t2) # t2 cz etar upor porer day gula depend kortese.


    # Store Last five cases for predecting future
    if ((len(exposed_train) - sample ) <= 15):
        print(str_dates[sample])
        last5_vals.append(init_vals)
        last5_params.append(params)




t7 = np.arange(0, 13, 1)  #13 din er data ashbe
# print(t7)

for z in range(1, len(last5_params)+1):
        results = seird_model(last5_vals[z-1], last5_params[z-1], t7)
        t_z = np.arange(7 - z, 14 - z, 1)
        print("Predictions using", last5_vals[z-1][0], "parameters\n")
        print("------------------------------------------------------------------------------------------------------------------")
        print("Date", "\t\tUncovered", "Du", "\t Ru", "\tExposed","Isolated", "Infected", " Di", "\t Ri" )
        print("--------------------------------------------------------------------------------------------------------------------")
        for day in t_z:
            print(results[day][0],"\t",int(float(results[day][2])),"\t",int(float(results[day][3])),"\t",int(float(results[day][4])),"\t", int(float(results[day][5])),  int(float(results[day][6])),"\t",int(float(results[day][7])),"\t",int(float(results[day][8])),"\t",int(float(results[day][9])))
        print("---------------------------------------------------------------------------------------------------------------\n\n")