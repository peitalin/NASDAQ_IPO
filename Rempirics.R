
require(sandwich)
require(lmtest)
require(plm)
require(nlme)
source("clmclx.R")

df <- data.table::fread("df.csv", colClasses=c(cik="character", SIC="character", Year="factor"))
dfu <- data.table::fread("df_update.csv", colClasses=c(cik="character", SIC="character", Year="factor"))

getVariance <- function(sigma) { sigma^2 }

getIOTKEY <- function(iotkey, df) {
    cat("Demeaned IOTKEY\n")
    df[[iotkey]] - mean(df[[iotkey]])
}


################################
INITIAL_RETURNS_REGRESSIONS <- function() {}
################################

dfa <- df[df$close_return < 200]
# Remove Baidu and Dicerna
attach(dfa)


# IOTKEY <- getIOTKEY("IoT_15day_CASI_news", dfa)
# IOTKEY <- getIOTKEY("IoT_15day_CASI_all", dfa)
IOTKEY <- getIOTKEY("IoT_15day_CASI_weighted_finance", dfa)


PRICE_SIGNAL <- priceupdate_up
PRICE_SIGNAL.type <- "priceupdate_up"
PRICE_SIGNAL <- pct_final_revision_up
PRICE_SIGNAL.type <- "pct_final_revision_up"

eq9 <- close_return ~ log(days_from_s1_to_listing) +
    underwriter_rank_avg + VC + share_overhang + log(proceeds) +
    EPS + log(1+sales) +
    M3_indust_rets + M3_initial_returns +
    Year +
    confidential_IPO +
    # foreign +
    # FF49_industry +
    media_listing +
    IOTKEY +
    I(IOTKEY^2) +
    number_of_price_updates_up +
    number_of_price_updates_down +
    PRICE_SIGNAL +
    IOTKEY:PRICE_SIGNAL

# ####### Cluster Robust OLS
m09 <- lm(eq9, data=dfa)
# # coeftest(m09, vcov=vcovHC(m09, type="HC1"))

clx(m09, 1, FF49_industry)
# # clx(m09, 1, underwriter_rank_single)
# # mclx(m09, 1, FF49_industry, Year)
# summary(m09)$r.squared


### Hierarchical Linear Models

m9.lme1 <- lme(eq9, random = ~ 1 | FF49_industry)
# m9.lme2 <- lme(eq9, random = list(~ 1 | FF49_industry, ~ M3_initial_returns | FF49_industry))
# m9.lme3 <- lme(eq9, random = list(~ 1 | FF49_industry, ~ PRICE_SIGNAL | FF49_industry))
# m9.lme4 <- lme(eq9, random = list(~ 1 | FF49_industry, ~ IOTKEY | FF49_industry), method="ML")


require(lme4)
library(lmerTest)
m9.lmer1 <- lmer(update(eq9, ~ . + (1 | FF49_industry)))
m9.lmer2 <- lmer(update(eq9, ~ . + (1 + M3_initial_returns | FF49_industry)))
m9.lmer4 <- lmer(update(eq9, ~ . + (1 + IOTKEY | FF49_industry)))

summary(m9.lme1)
summary(m9.lmer1)
summary(m9.lmer2)
summary(m9.lmer4)

print(PRICE_SIGNAL.type)







################################
FRP_REGRESSIONS <- function(){}
################################
dfa <- dfu
# dfa <- df
dfa <- dfu[df$close_return < 200]
attach(dfa)

# IOTKEY <- getIOTKEY("IoT_15day_CASI_news", dfa)
# IOTKEY <- getIOTKEY("IoT_15day_CASI_all", dfa)
IOTKEY <- getIOTKEY("IoT_15day_CASI_weighted_finance", dfa)


PRICE_SIGNAL <- priceupdate_up

eq1 <- percent_final_price_revision ~
    log(days_from_s1_to_listing) +
    underwriter_rank_avg + VC + share_overhang +
    log(proceeds) + log(1+sales) + EPS +
    M3_indust_rets + M3_initial_returns +
    Year +
    confidential_IPO +
    # foreign +
    media_listing +
    IOTKEY +
    I(IOTKEY^2) +
    PRICE_SIGNAL +
    IOTKEY:PRICE_SIGNAL

# ####### Cluster Robust OLS
m1 <- lm(eq1, data=dfa)
clx(m1, 1, FF49_industry)


require(lme4)
library(lmerTest)
### Hierarchical Linear Model
m1.lme1 <- lme(eq1, random = ~ 1 | FF49_industry)

m1.lmer1 <- lmer(update(eq1, ~ . + (1 | FF49_industry)))
m1.lmer4 <- lmer(update(eq1, ~ . + (1 + IOTKEY | FF49_industry)))

# summary(m1.lme1)
# ranef(m1.lme)

summary(m1.lmer1)
summary(m1.lmer4)

print(PRICE_SIGNAL.type)




################################
TIMING_PRICE_UPDATE_REGRESSIONS <- function(){}
################################
dfa <- dfu
# dfa <- df
# dfa <- dfu[df$close_return < 200]
dfa <- dfu[df$delay_in_price_update != 1]
attach(dfa)

# IOTKEY <- getIOTKEY("IoT_15day_CASI_news", dfa)
# IOTKEY <- getIOTKEY("IoT_15day_CASI_all", dfa)
IOTKEY <- getIOTKEY("IoT_15day_CASI_weighted_finance", dfa)


eq2 <- percent_first_price_update ~
    log(days_from_s1_to_listing) +
    delay_in_price_update +
    underwriter_rank_avg + VC + share_overhang +
    log(proceeds) + log(1+sales) + EPS +
    M3_indust_rets + M3_initial_returns +
    Year +
    confidential_IPO +
    # foreign +
    media_listing +
    IOTKEY +
    I(IOTKEY^2)


# ####### Cluster Robust OLS
m2 <- lm(eq2, data=dfa)
clx(m2, 1, FF49_industry)


require(lme4)
library(lmerTest)
### Hierarchical Linear Model
m2.lme1 <- lme(eq2, random = ~ 1 | FF49_industry)

m2.lmer1 <- lmer(update(eq2, ~ . + (1 | FF49_industry)))
m2.lmer4 <- lmer(update(eq2, ~ . + (1 + IOTKEY | FF49_industry)))

# summary(m1.lme1)
# ranef(m1.lme)

summary(m2.lmer1)
summary(m2.lmer4)





################################
TIMING_FPR_REGRESSIONS <- function(){}
################################
# dfa <- dfu
dfa <- df[(df$close_return < 200) & (df$delay_in_price_update < 1)]
# dfa <- df[df$close_return < 200]
# dfa <- df[df$delay_in_price_update != 1]
attach(dfa)

# IOTKEY <- getIOTKEY("IoT_15day_CASI_news", dfa)
# IOTKEY <- getIOTKEY("IoT_15day_CASI_all", dfa)
IOTKEY <- getIOTKEY("IoT_15day_CASI_weighted_finance", dfa)

PRICE_SIGNAL <- pct_final_revision_up
DEPVAR <- close_return

# PRICE_SIGNAL <- priceupdate_up
# DEPVAR <- percent_final_price_revision

eq3 <- DEPVAR ~
    log(days_from_s1_to_listing) +
    delay_in_price_update +
    underwriter_rank_avg + VC + share_overhang +
    log(proceeds) + log(1+sales) + EPS +
    M3_indust_rets + M3_initial_returns +
    Year +
    confidential_IPO +
    # foreign +
    media_listing +
    IOTKEY +
    I(IOTKEY^2) +
    PRICE_SIGNAL +
    IOTKEY:PRICE_SIGNAL


# ####### Cluster Robust OLS
m3 <- lm(eq3, data=dfa)
clx(m3, 1, FF49_industry)


require(lme4)
library(lmerTest)
### Hierarchical Linear Model

m3.lmer1 <- lmer(update(eq3, ~ . + (1 | FF49_industry)))
m3.lmer4 <- lmer(update(eq3, ~ . + (1 + IOTKEY | FF49_industry)))

summary(m3.lmer1)
summary(m3.lmer4)



# ############### SURVIVAL REG
# # dfa <- dfu
# dfa <- df[(df$close_return < 200) & (df$amendment != 0)]
# # dfa <- df[df$close_return < 200]
# # dfa <- df[df$delay_in_price_update != 1]
# dfa$amends <- sapply(dfa$amendment, function(x) { ifelse(x==-1, 0, 1) })
# attach(dfa)


# library(mlogit)

# ddata <- mlogit.data(df, id="cik", shape="wide", choice="amendment")

# eq5 <- amends ~
#     underwriter_rank_avg + VC + share_overhang +
#     log(proceeds) + log(1+sales) + log(market_cap) +
#     liab_over_assets + EPS +
#     M3_indust_rets + M3_initial_returns +
#     delay_in_price_update + log(days_from_s1_to_listing) +
#     IoT_15day_CASI_weighted_finance

# m5 <- glm(eq5, family=binomial(link=logit), dfa)
# summary(m5)

# m5 <- mlogit(eq5, ddata)






