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

setwd("~/Documents/rodrigo/TFM")
tabla1 <- read.csv("~/Documents/rodrigo/TFM/tabla1.csv")
 
datatot <- tabla1%>%select(-X)%>%select(-Genero)

datatot$Age<-datatot$Age/100
datatot$mx<-datatot$mx*30

seleccion<-sample(nrow(datatot),4000)

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
  real<lower=0> sigma; // sigma de la distribucion final
  
  vector[N_new] y_new; // predicciones
  
}

model {
  vector[N_obs] mu;
  vector[N_new] mu_new;
  
  // priors
  beta ~ normal(0, 1);
  Omega ~ lkj_corr(2);
  sigma_p ~ exponential(1);
  sigma ~ exponential(1);
  beta_p ~ multi_normal(beta, quad_form_diag(Omega, sigma_p));
  
  // likelihood
  for(i in 1:N_obs) {
    mu[i] = x[i] * (beta_p[gid[i]]); // *  significa multiplicacion de matrices
  }
  // para las predicciones
  for(i in 1:N_new) {
    mu_new[i] = x_new[i] * (beta_p[gid[i]]); // *  significa multiplicacion de matrices
  }
  
  
  y ~ normal(exp(-mu),sigma); //observaciones
  
  y_new ~ normal(exp(-mu_new),sigma);// predicciones
}

"
cat(multinivel,file="model1b.stan")

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

fit <- stan(file = "model1b.stan",
            data = dat2, iter=5000,chains = 8)

fit <- stan(file = "model1b.stan", data = dat2, iter=12000, warmup=8400,chains = 12,
            control = list(adapt_delta = .999999,stepsize = 0.005,max_treedepth = 13))


fit <- stan(file = "model1b.stan", data = dat2, iter=5000,warmup=4000,chains = 12,
            control = list(adapt_window=10, adapt_delta = .999999,max_treedepth = 17) )


print(fit)

plot(fit)


plot(fit,pars=c("beta_p[1,1]", "beta_p[1,2]", "beta_p[2,1]","beta_p[2,2]",
                "beta_p[3,1]", "beta_p[3,2]", "beta_p[4,1]","beta_p[4,2]",
                "beta_p[5,1]", "beta_p[5,2]", "beta_p[6,1]","beta_p[6,2]",
                "beta_p[7,1]", "beta_p[7,2]", "beta_p[8,1]","beta_p[8,2]"))

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

pairs(fit,pars=c("beta_p[1,1]", "beta_p[1,2]", "beta_p[2,1]","beta_p[2,2]"))

pairs(fit, pars = c("beta"))

pairs(fit, pars = c("sigma","lp__"))


#vemos correlaciones por grupo
pairs(fit, pars = c("beta_p[1,1]", "beta_p[1,2]","beta_p[2,1]", "beta_p[2,2]","beta_p[3,1]",
                    "beta_p[3,2]"))

#vemos correlaciones por predictor 
pairs(fit, pars = c("beta_p[1,1]", "beta_p[2,1]","beta_p[3,1]", "beta_p[4,1]",
                    "beta_p[5,1]","beta_p[6,1]"))

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


