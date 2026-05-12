library(lme4)
library(lmerTest)
library(DHARMa)
library(performance)

df = read.table("data/pivot_table.csv", header = TRUE, sep ="," )

model <- glmer(correctness_1 ~ correctness_0*validity_0*E_One_sqrt_pc_0 - correctness_0:validity_0:E_One_sqrt_pc_0 + task_success_zs_0 + (validity_0 + correctness_0 | id) + (correctness_0 + E_One_sqrt_pc_0| enc_task),
                data = df,
                glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 10e6)),
                family = "binomial")
summary(model) #divergent

model2 <- glmer(correctness_1 ~ correctness_0*validity_0*E_One_sqrt_pc_0 - correctness_0:validity_0:E_One_sqrt_pc_0 + task_success_zs_0 + (validity_0 + correctness_0 | id) + (correctness_0 | enc_task),
               data = df,
               glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 10e6)),
               family = "binomial")
summary(model)
model_simple <- glmer(correctness_1 ~ correctness_0 + validity_0 + E_One_sqrt_pc_0 + task_success_zs_0 + (validity_0 + correctness_0 | id) + (correctness_0| enc_task),
                      data = df,
                      glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 10e6)),
                      family = "binomial")
summary(model_simple)