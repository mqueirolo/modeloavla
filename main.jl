using JuMP, SDDP, Clp, Base.Test,JLD

#Parámetros:

                #Costo de transacción.
                #Si quiero quedar sobre el índice, a>1.
S=10             #Horizonte de simulación (escenarios en años).
N=4             #cantidad de activos
cvar_tolerancia=0.4 #toleracia al riesgo
cvar_confianza=0.1

rf=1.029  #activo libre de riesgo
cash_init=100.0 # al principio solo se tiene cash_init de caja , nada en activos
p_init=100.0 # precio inicial de todos los activos (inventado)
tasa_reaseguradora_credito=0.9
tasa_reaseguradora_garantia=0.7
MA_credito=0.5*min(cash_init,cash_init/((1-tasa_reaseguradora_credito)*S)) #condicion para que plata inicial pueda cubrir maximas perdidas todos los años sin pedir prestado
MA_garantia=0.5*min(cash_init,cash_init/((1-tasa_reaseguradora_garantia)*S)) #condicion para que plata inicial pueda cubrir maximas perdidas todos los años sin pedir prestado

prima_ingreso_credito=0.005
prima_ingreso_garantia=0.03

prima_reaseguradora_credito=tasa_reaseguradora_credito*prima_ingreso_credito
prima_reaseguradora_garantia=tasa_reaseguradora_garantia*prima_ingreso_garantia
prob_siniestralidad_credito=[0.9792,0.00116,0.0185,0.00114]
prob_siniestralidad_garantia=[0.99011,0,0.00968,0.00021]
#prob_siniestralidad=[prob_siniestralidad_credito,prob_siniestralidad_garantia]
grado_siniestralidad_credito=[0.0,0.5,0.75,1]*(1-tasa_reaseguradora_credito)*MA_credito
# grado_siniestralidad_credito_aux=[0,0.5,0.75,1]*(1-tasa_reaseguradora_credito)
grado_siniestralidad_garantia=[0.0,0.5,0.75,1]*(1-tasa_reaseguradora_garantia)*MA_garantia
# grado_siniestralidad_garantia_aux=[0,0.5,0.75,1]*(1-tasa_reaseguradora_garantia)
prob_siniestralidad_1=prob_siniestralidad_credito
prob_siniestralidad_2=prob_siniestralidad_garantia
grado_siniestralidad_1=[0.0,0.5,0.75,1]
grado_siniestralidad_2=[0.0,0.5,0.75,1]

prob_siniestralidad=zeros(Float64, length(prob_siniestralidad_1)*length(prob_siniestralidad_2))
grado_siniestralidad=zeros(Float64, length(prob_siniestralidad_1)*length(prob_siniestralidad_2))

for i = 1:length(prob_siniestralidad_1)
    for j = 1:length(prob_siniestralidad_2)
        a=length(prob_siniestralidad_1)*(i-1)+j
        prob_siniestralidad[a]=prob_siniestralidad_1[i]*prob_siniestralidad_2[j]
        grado_siniestralidad[a]=grado_siniestralidad_1[i]*(1-tasa_reaseguradora_credito)*MA_credito+grado_siniestralidad_2[j]*(1-tasa_reaseguradora_garantia)*MA_garantia
    end
end

#tasa_deuda=0.005104
tasa_deuda=2.0

###################################### generacion de precios ##################################
prob=zeros(Float64, 2^N) #el ultimo es el indice
jump_aux=zeros(Float64,2^N,N)
uuuu=transpose([1, 1, 1, 1])
uduu=transpose([1, -1, 1, 1])
uudu=transpose([1, 1, -1, 1])
uuud=transpose([1, 1, 1, -1])
uddu=transpose([1, -1, -1, 1])
uudd=transpose([1, 1, -1, -1])
udud=transpose([1, -1, 1, -1])
duud=transpose([-1, 1, 1, -1])
dddd=transpose([-1, -1, -1, -1])
uddd=transpose([1, -1, -1, -1])
dudd=transpose([-1, 1, -1, -1])
ddud=transpose([-1, -1, 1, -1])
dddu=transpose([-1, -1, -1, 1])
dudu=transpose([-1, 1, -1, 1])
duuu=transpose([-1, 1, 1, 1])
dduu=transpose([-1, -1, 1, 1])
ud=zeros(Float64, 2^N, N)
ud=[uuuu;uduu;uudu;uuud;uddu;uudd;udud;duud;dddd;uddd;
dudd;ddud;dddu;dudu;duuu;dduu]
correl=[[1.00 0.945 0.738 -0.068];
[0.945 1.00 0.901 0.201];
[0.738 0.901 1.00 0.537];
[-0.068 0.201 0.537 1.00]]

drift=[0.0703,0.0684,0.0665,0.0661]
sigma=[0.0963,0.0474,0.0300,0.0251]
delta=drift-0.5*sigma.^2
aux=repmat(delta',2^N,1)
for i = 1:2^N
for l = 1:N-1
         for m = (l+1):N
                 if(ud[i,l]>ud[i,m])
                        prob[i]=prob[i]-correl[l,m]
                 elseif(ud[i,l]<ud[i,m])
                        prob[i]=prob[i]-correl[l,m]
                 else
                        prob[i]=prob[i]+correl[l,m]
                 end
         end
end
prob[i]=(1/(2^N))*(1+prob[i])
jump_aux[i,:]=sigma.*ud[i,:]

end
R=exp.(jump_aux+aux)
#aa = Array{Tuple{Float64, Float64,Float64,Float64},1}

transiciones_aux=repmat(prob,1,2^N)
transiciones_aux=transiciones_aux'


m = SDDPModel(
                sense           = :Min,
                stages          = (S+1),
                objective_bound = -10000000.0,
                solver          = ClpSolver(),
                risk_measure = EAVaR(lambda=cvar_tolerancia, beta=cvar_confianza),
                markov_transition =  Array{Float64, 2}[
                [ 1.0 ]',
                prob',
                transiciones_aux,
                transiciones_aux,
                transiciones_aux,
                transiciones_aux,
                transiciones_aux,
                transiciones_aux,
                transiciones_aux,
                transiciones_aux,#10
                transiciones_aux,
                transiciones_aux,
                transiciones_aux,
                transiciones_aux,
                transiciones_aux,
                transiciones_aux,
                transiciones_aux,
                transiciones_aux,
                transiciones_aux,
                transiciones_aux,#20
                transiciones_aux,
                transiciones_aux,
                transiciones_aux,
                transiciones_aux,
                transiciones_aux
                ]

                                                    ) do sp, t,i


        @state(sp, Xt[j=1:N]>=0, X0==0.0)
        @state(sp, CASHt>=0.0, CASH0==cash_init)
        @state(sp, Deuda_pasadat>=0.0, Deuda_pasada0==0.0)

        @variables(sp, begin

                BUYt[j=1:N]>=0.0
                SELLt[j=1:N]>=0.0
                DEUDAt>=0.0
                # Monto Asegurado incluirlo como variable de estado y realizar el ploteo para hacer seguimiento del ratio entre el monto asegurado y la riqueza en tiempo t
                # @state(sp, MAt>=0, MA0==0)
        end)

        if t==1

               @constraint(sp,CASHt == CASH0+ DEUDAt -sum(BUYt[j] for j in 1:N) + sum(SELLt[j] for j in 1:N))
               @constraint(sp,balanceXt[j=1:N], Xt[j]== X0[j] + BUYt[j] - SELLt[j])
               @constraint(sp, Deuda_pasadat == DEUDAt)
               @constraint(sp, CASHt+sum(Xt[j] for j in 1:N) -(1+tasa_deuda)*Deuda_pasada0 >= 0.0) #liquidez


        elseif t==S+1

               @constraint(sp,CASHt == CASH0*rf + DEUDAt-(1+tasa_deuda)*Deuda_pasada0)
               @constraint(sp,balanceXt[j=1:N], Xt[j]== X0[j]*R[i,j])
               @constraint(sp, fijo_buy[j=1:N], BUYt[j]==0.0)
               @constraint(sp, fijo_sell[j=1:N], SELLt[j]==0.0)

               #@constraint(sp, fijo_deuda, DEUDAt==0)
               #@constraint(sp,balance_equalt[j=1:N], Xt[j]== (1/N)*sum(X0[j]*R[i,j] for j in 1:N))
        else
                @rhsnoise(sp, C=-grado_siniestralidad, CASHt -  CASH0*rf -DEUDAt+(1+tasa_deuda)*Deuda_pasada0+ sum(BUYt[j] for j in 1:N) - sum(SELLt[j] for j in 1:N)+ (prima_reaseguradora_credito-prima_ingreso_credito)*MA_credito+ (prima_reaseguradora_garantia-prima_ingreso_garantia)*MA_garantia == C)
                setnoiseprobability!(sp, prob_siniestralidad)
                #@rhsnoise(sp, C2=-grado_siniestralidad_garantia, CASHt -  CASH0*rf -DEUDAt+(1+tasa_deuda)*Deuda_pasada0+ sum(BUYt[j] for j in 1:N) - sum(SELLt[j] for j in 1:N)+ (prima_reaseguradora_garantia-prima_ingreso_garantia)*MA_garantia == C2)
                #setnoiseprobability!(sp, prob_siniestralidad_garantia)
                @constraint(sp,balanceXt[j=1:N], Xt[j] == X0[j]*R[i,j] + BUYt[j] - SELLt[j])
                @constraint(sp, Deuda_pasadat == DEUDAt)
                @constraint(sp, CASHt+sum(Xt[j] for j in 1:N) -(1+tasa_deuda)*Deuda_pasada0 >= 0.0) #liquidez

                #@constraint(sp,balance_equalt[j=1:N], Xt[j]== (1/N)*sum(X0[j]*R[i,j] for j in 1:N))
                #@constraint(sp,balance_cash0, CASHt== 0.0)
                #@constraint(sp, def==0.0)
               #@constraint(sp, exc==0.0)
        end

        if t==S+1
            @stageobjective(sp,  +DEUDAt -CASHt - sum(Xt[j] for j in 1:N))
        else
            @stageobjective(sp,  0.0)
        end


end

status=solve(m,
iteration_limit = 100,
log_file="example.log")

srand(124)
SIMN = 500
sim = simulate(m, SIMN,[:Xt,:CASHt,:BUYt,:SELLt,:markov,:DEUDAt])

lb=getbound(m)

#################### usar la solucion para visualizar una politica ########################
precio1=Array{Float64}(SIMN,S+1)
precio2=Array{Float64}(SIMN,S+1)
precio3=Array{Float64}(SIMN,S+1)
precio4=Array{Float64}(SIMN,S+1)
peso1=Array{Float64}(SIMN,S)
peso2=Array{Float64}(SIMN,S)
peso3=Array{Float64}(SIMN,S)
peso4=Array{Float64}(SIMN,S)
peso_cash=Array{Float64}(SIMN,S)
riqueza=Array{Float64}(SIMN,S+1)
riqueza_indice=Array{Float64}(SIMN,S+1)
riqueza_benchmark=Array{Float64}(SIMN,S+1)
funcion_objetivo=Array{Float64}(SIMN)
ratioMAriqueza=Array{Float64}(SIMN,S+1)
debt=Array{Float64}(SIMN,S+1)
cash=Array{Float64}(SIMN,S+1)

#retrono cartera en t=2 significa retornos entre 1 y 2. t=1 no guadara nada
retorno_cartera=Array{Float64}(SIMN,S+1)
retorno_geometrico=Array{Float64}(SIMN)
volatilidad=Array{Float64}(SIMN)
retorno_benchmark=Array{Float64}(SIMN,S+1)


for i=1:SIMN
               funcion_objetivo[i]=-sim[i][:objective]

           for t=1:S+1

                    riqueza[i,t]=0
                    for j=1:N
                        riqueza[i,t]=riqueza[i,t]+sim[i][:Xt][t][j]
                        ratioMAriqueza[i,t]=(MA_credito+MA_garantia)/riqueza[i,t]
                    end
                    if t==1
                        riqueza[i,t]=riqueza[i,t]+sim[i][:CASHt][t]+sim[i][:DEUDAt][t]
                        ratioMAriqueza[i,t]=(MA_credito+MA_garantia)/riqueza[i,t]
                        debt[i,t]=0.0
                        cash[i,t]=cash_init
                    else
                        riqueza[i,t]=riqueza[i,t]+sim[i][:CASHt][t]-sim[i][:DEUDAt][t]-(1+tasa_deuda)*sim[i][:DEUDAt][t-1]
                        ratioMAriqueza[i,t]=(MA_credito+MA_garantia)/riqueza[i,t]
                        debt[i,t]=sim[i][:DEUDAt][t]
                        cash[i,t]=sim[i][:CASHt][t]
                    end
                    if t==1

                           precio1[i,t]=p_init
                           precio2[i,t]=p_init
                           precio3[i,t]=p_init
                           precio4[i,t]=p_init
                     else
                           h=sim[i][:markov][t]


                           precio1[i,t]=precio1[i,t-1]*R[h,1]
                           precio2[i,t]=precio2[i,t-1]*R[h,2]
                           precio3[i,t]=precio3[i,t-1]*R[h,3]
                           precio4[i,t]=precio4[i,t-1]*R[h,4]

                           for j=1:N

                                 retorno_cartera[i,t]=riqueza[i,t]/riqueza[i,t-1]-1
                        #         retorno_benchmark[i,t]=retorno_benchmark[i,t]+(R[h,j]-1)
                           end

                           #riqueza_benchmark[i,t]=riqueza_benchmark[i,t-1]*(1+retorno_benchmark[i,t])


                     end

                     if t<=S

                         peso1[i,t]=sim[i][:Xt][t][1]/riqueza[i,t]
                         peso2[i,t]=sim[i][:Xt][t][2]/riqueza[i,t]
                         peso3[i,t]=sim[i][:Xt][t][3]/riqueza[i,t]
                         peso4[i,t]=sim[i][:Xt][t][4]/riqueza[i,t]
                         peso_cash[i,t]=(1-peso1[i,t]-peso2[i,t]-peso3[i,t]-peso4[i,t])

                     end



               end
               if (riqueza[i,S+1]/riqueza[i,1])<0
                   println(riqueza[i,S+1])
                   println(riqueza[i,1])
                   println(i)
               else
                   retorno_geometrico[i]=(riqueza[i,S+1]/riqueza[i,1])^(1/S)-1
               end
               volatilidad[i]=std(retorno_cartera[i,2:S+1])

end

aux1=sort(retorno_geometrico)
percentil_numero=Int(ceil(cvar_confianza*SIMN))
percentil=aux1[percentil_numero]
cvar=mean(aux1[1:percentil_numero])
metricas_riqueza=[mean(retorno_geometrico),mean(volatilidad),cvar]


plt = SDDP.newplot()
h=0
caminos=SIMN
SDDP.addplot!(plt, 1:caminos, 1:S,(i, t)->peso1[i,t], title="x1", ylabel=string(cvar_tolerancia))
SDDP.addplot!(plt, 1:caminos, 1:S,(i, t)->peso2[i,t], title="x2", ylabel="")
SDDP.addplot!(plt, 1:caminos, 1:S,(i, t)->peso3[i,t], title="x3", ylabel="")
SDDP.addplot!(plt, 1:caminos, 1:S,(i, t)->peso4[i,t], title="x4", ylabel="")
SDDP.addplot!(plt, 1:caminos, 1:S,(i, t)->peso_cash[i,t], title="cash", ylabel="")
SDDP.addplot!(plt, 1:caminos, 1:S+1,(i, t)->riqueza[i,t], title="riqueza", ylabel="")
SDDP.addplot!(plt, 1:caminos, 1:S+1,(i, t)->ratioMAriqueza[i,t], title="ratio", ylabel="")

#SDDP.addplot!(plt, 1:caminos, 1:S,(i, t)->retorno_cartera[i,t], title="ret_cartera", ylabel="")
SDDP.addplot!(plt, 1:caminos, 1:S+1,(i, t)->precio1[i,t], title="price1", ylabel="")
SDDP.addplot!(plt, 1:caminos, 1:S+1,(i, t)->precio2[i,t], title="price2", ylabel="")
SDDP.addplot!(plt, 1:caminos, 1:S+1,(i, t)->precio3[i,t], title="price3", ylabel="")
SDDP.addplot!(plt, 1:caminos, 1:S+1,(i, t)->precio4[i,t], title="pindice", ylabel="")
SDDP.addplot!(plt, 1:caminos, 1:S+1,(i, t)->debt[i,t], title="deuda", ylabel="")
SDDP.addplot!(plt, 1:caminos, 1:S+1,(i, t)->cash[i,t], title="caja", ylabel="")

SDDP.show("simulacion.html", plt)

writedlm("analisis_peso1.txt", hcat(peso1,peso2,peso3,peso4,peso_cash))
writedlm("analisis_precios.txt", hcat(precio1,precio2,precio3,precio4))
writedlm("analisis_riqueza.txt", riqueza)
writedlm("analisis_caja.txt", hcat(cash,debt))
writedlm("analisis_ratio.txt", ratioMAriqueza)
