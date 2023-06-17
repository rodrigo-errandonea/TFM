library("rstan")

options(mc.cores = parallel::detectCores())

rstan_options(auto_write = TRUE)

library(tidyverse) # para  ggplot2, dplyr, purrr, etc.
theme_set(theme_bw(base_size = 14)) #  ggplot2 theme

library(mvtnorm)

#para dibujar las posteriors
library("bayesplot")
library("rstanarm")

set.seed(12407) # para comparar resultados

#setwd("~/misdocumentos/rodrigo/master/TFM")
#tabla1 <- read.csv("~/misdocumentos/rodrigo/master/TFM/tabla1.csv")

setwd("~/Documents/rodrigo/TFM")
tabla1 <- read.csv("~/Documents/rodrigo/TFM/tabla1.csv")

datatot <- tabla1%>%select(-X)%>%select(-Genero)

datatot$Age<-datatot$Age/100
datatot$mx<-datatot$mx*30

seleccion<-sample(nrow(datatot),5800)

dat<-datatot[seleccion,]

test<-datatot%>%filter(!row_number() %in% seleccion)

seleccion2<-sample(nrow(test),100)

new_full<-test[seleccion2,]

new<-new_full%>%select(-SAI)

 
#################################################################################

#vamos a crear un file stan con el modelo

multinivel <- "
data {
  int N_obs; // numero de observaciones
  int N_pts; // numero de estados
  int K; // numero ode predictores + intercepts
  int gid[N_obs]; // vector de indetificador de estados
  matrix[N_obs, K] x; // matriz de predictores
  real y[N_obs]; // vector resultados
  
  int<lower=0> N_new;     // numero predicciones
  matrix[N_new, K] x_new;  // datos para predecir
}

parameters {
  vector[K] beta_p[N_pts]; // ordenada al origen y pendientes de cada grupo
  vector<lower=0>[K] sigma_p; // sd para la ordenada al origen y las pendientes
  vector[K] beta; //  hyper-priors para la ordenada al origen y las pendientes
  corr_matrix[K] Omega; // matriz de correlation 
}

model {
  vector[N_obs] mu;
  // priors
  beta ~ normal(0, 1);
  Omega ~ lkj_corr(2);
  sigma_p ~ exponential(1);
  beta_p ~ multi_normal(beta, quad_form_diag(Omega, sigma_p));
  
  // likelihood
  for(i in 1:N_obs) {
    mu[i] = x[i] * (beta_p[gid[i]]) ; // * aca significa multiplicacion de matrices
  }
  y ~ exponential(mu);
}
generated quantities {
  vector[N_new] y_new;
  vector[N_new] mu_new;
  
  for(i in 1:N_new){
    mu_new[i] = x_new[i] * (beta_p[gid[i]]) ; // * aca significa multiplicacion de matrices
    y_new[i] = exponential_rng(mu_new[i]);}

}

"
cat(multinivel,file="model2b.stan")


dat2 <- list(
  N_obs = nrow(dat),
  N_pts = max(as.numeric(dat$grupo)),
  K = 2, # predictores
  gid = as.numeric(dat$grupo),
  x = matrix(c(dat$Age, dat$mx) ,ncol = 2),
  y =  dat$SAI,
  N_new=nrow(new),
  x_new=matrix(c(new$Age, new$mx) ,ncol = 2)
)


#condiciones iniciales
ci<-
  function(){
    list(beta=runif(2,0,2),sigma_p=runif(2,0,1),beta_p=matrix(runif(32,0,2),ncol=2),
         Omega=diag(1,2,2))
  }

fit <- stan(file = "model2b.stan", data = dat2, iter=5000,init=ci,chains = 12)


fit <- stan(file = "model2b.stan", data = dat2, iter=5000,warmup=4000,init=ci,chains = 12,
            control = list(adapt_window=10, adapt_delta = .99999,max_treedepth = 15) )

print(fit)

plot(fit)

###############################################################################
### DIAGNOSTICOS 
rstan::check_divergences(fit)
###############################################################################

#sacar todos los resulatdos como un data frame
resultados<- as.data.frame(summary(fit)[[1]])

#############################################################################
##### El primer indice corresponde al grupo y el segundo al predictor
##############################################################################

#en las beta[id grupo, parametro(x_i)]
for(i in 1:nrow(resultados)-1){
  cat(rownames(resultados)[i], resultados[i,1], " \n")
}

traceplot(fit, pars = c("beta"), inc_warmup = TRUE, nrow = 3)

#pairs(fit, pars = c("beta_p[6,1]", "beta_p[6,2]"))

pairs(fit, pars = c("beta"))

pairs(fit, pars = c("sigma","lp__"))
par(mfrow=c(3,2))
#vemos correlaciones por grupo
pairs(fit, pars = c("beta_p[1,1]", "beta_p[1,2]","beta_p[2,1]", "beta_p[2,2]","beta_p[3,1]",
                    "beta_p[3,2]"))

#vemos correlaciones por predictor 
pairs(fit, pars = c("beta_p[1,1]", "beta_p[2,1]","beta_p[3,1]", "beta_p[4,1]",
                    "beta_p[5,1]","beta_p[6,1]"))


plot(fit,pars=c("beta_p[1,1]", "beta_p[1,2]", "beta_p[2,1]","beta_p[2,2]",
                "beta_p[3,1]", "beta_p[3,2]", "beta_p[4,1]","beta_p[4,2]",
                "beta_p[5,1]", "beta_p[5,2]", "beta_p[6,1]","beta_p[6,2]",
                "beta_p[7,1]", "beta_p[7,2]", "beta_p[8,1]","beta_p[8,2]"))
#posteriors
p <- plot(fit, pars = "beta_p", ci_level = 0.5, prob_outer = 0.9)
p + ggplot2::ggtitle("Posterior medians \n with 50% and 90% intervals")


plot(fit, plotfun = "hist", pars = "theta", include = FALSE)

# vemos las posterior completas de un grupo 


#mejores plots para las  posteriors 
color_scheme_set("brightblue")
posterior <- as.matrix(fit)
plot_title <- ggtitle("Posterior distributions with medians and 80% intervals")
mcmc_areas(posterior, pars = c("beta_p[1,1]", "beta_p[1,2]","beta_p[2,1]", "beta_p[2,2]",
                               "beta_p[3,1]","beta_p[3,2]"),prob = 0.8) + plot_title

plot_title <- ggtitle("Posterior distributions with medians and 80% intervals")
mcmc_areas(posterior, pars = c("beta_p[4,1]", "beta_p[4,2]","beta_p[5,1]","beta_p[5,2]", 
                               "beta_p[6,1]","beta_p[6,2]"),prob = 0.8) + plot_title


plot_title <- ggtitle("Posterior distributions with medians and 80% intervals")
mcmc_areas(posterior, pars = c("beta[1]","beta[2]"),prob = 0.8) + plot_title


