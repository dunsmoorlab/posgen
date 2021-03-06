## script for mixed modeling generalization data (summer 2019)
## compare AIC/BIC of linear and nonlinear models across tertile 
## and experimental group

## installing lmer package
#setwd("/Users/ach3377/Db_lpl/Dunsmoor Lab/posgen")
setwd("C:/Users/ACH/Dropbox (LewPeaLab)/Dunsmoor Lab/posgen")
library("lme4")
library(lmtest)

## parse that there data

pg<-read.csv("posgen_data.csv")

pg_1 <- subset(pg, subset = phase == 1)
pg_2 <- subset(pg,subset = phase == 2)
pg_3 <- subset(pg,subset = phase == 3)

fg<-read.csv("feargen_data.csv")

fg_1 <- subset(fg, subset = phase == 1)
fg_2 <- subset(fg,subset = phase == 2)
fg_3 <- subset(fg,subset = phase == 3)

##posgen modeling

lm_pg_full <- lmer(pg$scr~pg$face+(1|pg$subject), REML=FALSE)

lm_pg_1 <- lmer(pg_1$scr~pg_1$face+(1|pg_1$sub), REML=FALSE)
lm_pg_2 <- lmer(pg_2$scr~pg_2$face+(1|pg_2$sub), REML=FALSE)
lm_pg_3 <- lmer(pg_3$scr~pg_3$face+(1|pg_3$sub), REML=FALSE)

lm_pg_full_poly <- lmer(pg$scr~poly(pg$face,2)+(1|pg$sub), REML=FALSE)

lm_pg_1_poly <- lmer(pg_1$scr~poly(pg_1$face,2)+(1|pg_1$sub), REML=FALSE)
lm_pg_2_poly <- lmer(pg_2$scr~poly(pg_2$face,2)+(1|pg_2$sub), REML=FALSE)
lm_pg_3_poly <- lmer(pg_3$scr~poly(pg_3$face,2)+(1|pg_3$sub), REML=FALSE)


##feargen modeling

lm_fg_full <- lmer(fg$scr~fg$face+(1|pg$subject), REML=FALSE)

lm_fg_1 <- lmer(fg_1$scr~fg_1$face+(1|fg_1$sub), REML=FALSE)
lm_fg_2 <- lmer(fg_2$scr~fg_2$face+(1|fg_2$sub), REML=FALSE)
lm_fg_3 <- lmer(fg_3$scr~fg_3$face+(1|fg_3$sub), REML=FALSE)

lm_fg_full_poly <- lmer(fg$scr~poly(fg$face,2)+(1|pg$sub), REML=FALSE)

lm_fg_1_poly <- lmer(fg_1$scr~poly(fg_1$face,2)+(1|fg_1$sub), REML=FALSE)
lm_fg_2_poly <- lmer(fg_2$scr~poly(fg_2$face,2)+(1|fg_2$sub), REML=FALSE)
lm_fg_3_poly <- lmer(fg_3$scr~poly(fg_3$face,2)+(1|fg_3$sub), REML=FALSE)


## print out summary stats. AIC & BIC will be reported
## general trend of better poly fit for posgen, better
## linear fit for feargen

print(summary(lm_pg_full)$AICtab['BIC'])
print(summary(lm_pg_full_poly)$AICtab['BIC'])
lrtest(lm_pg_full_poly,lm_pg_full)

print(summary(lm_pg_1)$AICtab['BIC'])
print(summary(lm_pg_1_poly)$AICtab['BIC'])
lrtest(lm_pg_1_poly,lm_pg_1)

print(summary(lm_pg_2)$AICtab['BIC'])
print(summary(lm_pg_2_poly)$AICtab['BIC'])
lrtest(lm_pg_2_poly,lm_pg_2)

print(summary(lm_pg_3)$AICtab['BIC'])
print(summary(lm_pg_3_poly)$AICtab['BIC'])
lrtest(lm_pg_3_poly,lm_pg_3)

print(summary(lm_fg_full)$AICtab['BIC'])
print(summary(lm_fg_full_poly)$AICtab['BIC'])
lrtest(lm_fg_full_poly,lm_fg_full)

print(summary(lm_fg_1)$AICtab['BIC'])
print(summary(lm_fg_1_poly)$AICtab['BIC'])
lrtest(lm_fg_1_poly,lm_fg_1)

print(summary(lm_fg_2)$AICtab['BIC'])
print(summary(lm_fg_2_poly)$AICtab['BIC'])
lrtest(lm_fg_2_poly,lm_fg_2)

print(summary(lm_fg_3)$AICtab['BIC'])
print(summary(lm_fg_3_poly)$AICtab['BIC'])
lrtest(lm_fg_3_poly,lm_fg_3)
