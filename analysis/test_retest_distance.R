library(lme4)
library(lmerTest)
library(DHARMa)

df = read.table("data/distances_test_retest.csv", header = TRUE, sep ="," )

model = lmer(formula = distance ~ same_id + (1 | id_test) + (1 | id_retest),
             data = df,
             control = lmerControl(optimizer = "bobyqa"))

summary(model)

sim_res <- simulateResiduals(model)
plot(sim_res)
check_overdispersion(model)
check_outliers(model)
check_collinearity(model)
r2(model)
isSingular(model)

model_transf = lmer(formula = dist_sq_zs ~ same_id + (1 | id_test) + (1 | id_retest),
                    data = df)
summary(model_transf)

sim_res <- simulateResiduals(model_transf)
plot(sim_res)
check_overdispersion(model_transf)
check_outliers(model_transf)
check_collinearity(model_transf)
r2(model_transf)
isSingular(model_transf)

model_summary <-  summary(model)