library(lme4)
library(lmerTest)
library(DHARMa)
library(performance)

df = read.table("data/pivot_table.csv", header = TRUE, sep ="," )

model0 <- glmer(correctness_1 ~ correctness_0*validity_0 + (1 | id) + (1 | enc_task),
                  data = df,
                  glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 10e6)),
                  family = "binomial")

summary(model0)

model1 <- glmer(correctness_1 ~ correctness_0*validity_0*energy_sqrt_pc_0 + (1 | id) + (1 | enc_task),
                         data = df,
                         glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 10e6)),
                         family = "binomial")

summary(model1)

model2 <- glmer(correctness_1 ~ correctness_0*validity_0 + (correctness_0 + validity_0 | id) + (correctness_0 | enc_task),
                  data = df,
                  glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 10e6)),
                  family = "binomial")
summary(model2)

model3 <- glmer(correctness_1 ~ correctness_0*validity_0 + (correctness_0 * validity_0 | id) + (correctness_0 | enc_task),
                data = df,
                glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 10e6)),
                family = "binomial")
summary(model3)

model4 <- glmer(correctness_1 ~ correctness_0*validity_0*energy_sqrt_pc_0 + (validity_0 + correctness_0 + energy_sqrt_pc_0 | id) + (correctness_0 + energy_sqrt_pc_0 | enc_task),
              data = df,
              glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 10e6)),
              family = "binomial")

summary(model4)

model5 <- glmer(correctness_1 ~ correctness_0*validity_0*energy_sqrt_pc_0 + (validity_0 + correctness_0 + energy_sqrt_pc_0 | id) + (correctness_0 | enc_task),
                data = df,
                glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 10e6)),
                family = "binomial")
summary(model5)

model6 <- glmer(correctness_1 ~ correctness_0*validity_0*energy_sqrt_pc_0 + (validity_0 + correctness_0 | id) + (correctness_0 + energy_sqrt_pc_0 | enc_task),
                data = df,
                glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 10e6)),
                family = "binomial")
summary(model6)

model7 <- glmer(correctness_1 ~ correctness_0*validity_0*energy_sqrt_pc_0 - correctness_0:validity_0:energy_sqrt_pc_0 + (validity_0 + correctness_0 | id) + (correctness_0 + energy_sqrt_pc_0 | enc_task),
                data = df,
                glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 10e6)),
                family = "binomial")
summary(model7)

sim_res <- simulateResiduals(model7)
plot(sim_res)
check_overdispersion(model7)
check_collinearity(model7)
r2(model7)

sim_res <- simulateResiduals(model0)
plot(sim_res)
check_overdispersion(model0)
check_collinearity(model0)
r2(model1) #diagonistics indicate underdispersion

model8 <- glmer(correctness_1 ~ correctness_0*validity_0*energy_sqrt_pc_0 - correctness_0:validity_0:energy_sqrt_pc_0 + task_success_zs_0 + (validity_0 + correctness_0 | id) + (correctness_0 + energy_sqrt_pc_0 | enc_task),
                data = df,
                glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 10e6)),
                family = "binomial")
summary(model8)

model9 <- glmer(correctness_1 ~ correctness_0*validity_0*task_success_zs_0 + (validity_0 + correctness_0 | id) + (correctness_0 | enc_task),
                data = df,
                glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 10e6)),
                family = "binomial")
summary(model9)

model10 <- glmer(correctness_1 ~ correctness_0*validity_0 + task_success_zs_0 + (validity_0 + correctness_0 | id) + (correctness_0 | enc_task),
                 data = df,
                 glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 10e6)),
                 family = "binomial")
summary(model10)

model11 <- glmer(correctness_1 ~ correctness_0*validity_0*energy_sqrt_pc_0*task_success_zs_0 + (validity_0 + correctness_0 | id) + (correctness_0 + energy_sqrt_pc_0 | enc_task),
                data = df,
                glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 10e6)),
                family = "binomial")
summary(model11)

anova(model8,model10)

sim_res <- simulateResiduals(model10)
plot(sim_res)
check_overdispersion(model10)
check_collinearity(model10)
r2(model10)

sim_res <- simulateResiduals(model8)
plot(sim_res)
check_overdispersion(model8)
check_collinearity(model8)
r2(model8)

best <- model8
base <- model10

best_summary <- summary(best)
base_summary <- summary(base)

comparison <- anova(best, base)

delta_aic <- comparison["base", "AIC"] - comparison["best", "AIC"]

best_perf <- performance::model_performance(best)
base_perf <- performance::model_performance(base)



