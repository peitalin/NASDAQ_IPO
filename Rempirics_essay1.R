
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
INITIAL_RETURNS_REGRESSIONS <- function() {
################################

dfa <- df[(df$close_return < 200)]
# dfa <- df[(df$close_return < 200) & (df$original_price_is_range == TRUE)]
# Remove Baidu and Dicerna
attach(dfa)


PRICE_SIGNAL <- pct_final_revision_up
PRICE_SIGNAL.type <- "pct_final_revision_up"

# IOTKEY <- getIOTKEY("IoT_15day_CASI_news", dfa)
# IOTKEY <- getIOTKEY("IoT_15day_CASI_all", dfa)
IOTKEY <- getIOTKEY("IoT_15day_CASI_weighted_finance", dfa)
#### Now partialling out media_listing from CASI
## regress CASI ~ Media_listing and save residuals to use as variable instead of CASI
IOTKEY = lm(IoT_15day_CASI_weighted_finance ~ media_listing )$residuals
# IOTKEY = lm(IoT_15day_CASI_weighted_finance ~ media_listing + I(media_listing^2))$residuals

# stddev_prices_10day
# stddev_prices_5day
eq9 <- close_return ~ log(days_from_s1_to_listing) +
    underwriter_rank_avg + VC + share_overhang + log(proceeds) +
    EPS + log(1+sales) +
    M3_indust_rets + M3_initial_returns +
    Year +
    # confidential_IPO +
    # foreign +
    # underwriter_syndicate_size +
    FF49_industry +
    media_listing +
    IOTKEY +
    I(IOTKEY^2) +
    number_of_price_updates_up +
    number_of_price_updates_down +
    pct_final_revision_up +
    pct_final_revision_down +
    # IOTKEY:underwriter_rank_avg
    IOTKEY:pct_final_revision_up +
    IOTKEY:pct_final_revision_down


######## Cluster Robust OLS
m09 <- lm(eq9, data=dfa)
clx(m09, 1, FF49_industry)

# m9.lme1 <- lme(eq9, random = ~ 1 | FF49_industry)
# m9.lme2 <- lme(eq9, random = ~ 1 | IoT_entity_type)
# summary(m9.lme2)

require(lme4)
library(lmerTest)
m9.lmer1 <- lmer(update(eq9, ~ . + (1 | FF49_industry)))
# m9.lmer2 <- lmer(update(eq9, ~ . + (1 | IoT_entity_type)))

summary(m9.lmer1)
# summary(m9.lmer2)
# summary(m9.lmer4)

# print(PRICE_SIGNAL.type)
}







################################
FRP_REGRESSIONS <- function() {
################################
# dfa <- dfu
# dfa <- df
dfa <- dfu[df$close_return < 200]
attach(dfa)
PRICE_SIGNAL <- priceupdate_up


# IOTKEY <- getIOTKEY("IoT_15day_CASI_news", dfa)
# IOTKEY <- getIOTKEY("IoT_15day_CASI_all", dfa)
IOTKEY <- getIOTKEY("IoT_15day_CASI_weighted_finance", dfa)
##### NOw partialling out media for CASI
### by regression CASI ~ media_listing and saving residuals
IOTKEY <- lm(IoT_15day_CASI_weighted_finance ~ media_listing)$residuals


eq1 <- percent_final_price_revision ~
    log(days_from_s1_to_listing) +
    underwriter_syndicate_size +
    underwriter_rank_avg + VC + share_overhang +
    log(proceeds) + log(1+sales) + EPS +
    M3_indust_rets + M3_initial_returns +
    Year +
    # FF49_industry +
    # confidential_IPO +
    # foreign +
    media_listing +
    IOTKEY +
    I(IOTKEY^2) +
    PRICE_SIGNAL +
    IOTKEY:PRICE_SIGNAL
    # IOTKEY:underwriter_rank_avg

# ####### Cluster Robust OLS
m1 <- lm(eq1, data=dfa)
clx(m1, 1, FF49_industry)


require(lme4)
library(lmerTest)
### Hierarchical Linear Model

m1.lmer1 <- lmer(update(eq1, ~ . + (1 | FF49_industry)))
summary(m1.lmer1)

### Hierarchical Linear Model
# m1.lme1 <- lme(eq1, random = ~ 1 | FF49_industry)
# summary(m1.lme1)
# ranef(m1.lme)

}





################################
1st_PRICEAMENDENT_REGRESSIONS <- function() {
################################
# dfa <- dfu
# dfa <- df
dfa <- dfu[df$close_return < 200]
attach(dfa)
PRICE_SIGNAL <- priceupdate_up


# IOTKEY <- getIOTKEY("IoT_15day_CASI_news", dfa)
# IOTKEY <- getIOTKEY("IoT_15day_CASI_all", dfa)
IOTKEY <- getIOTKEY("IoT_15day_CASI_weighted_finance", dfa)
##### NOw partialling out media for CASI
### by regression CASI ~ media_listing and saving residuals
IOTKEY <- lm(IoT_15day_CASI_weighted_finance ~ media_listing)$residuals


eq3 <- priceupdate_down ~
    log(days_from_s1_to_listing) +
    underwriter_rank_avg + VC + share_overhang +
    log(proceeds) + log(1+sales) + EPS +
    M3_indust_rets + M3_initial_returns +
    Year +
    FF49_industry +
    # confidential_IPO +
    # foreign +
    media_listing +
    IOTKEY +
    I(IOTKEY^2) +
    IOTKEY:media_listing
    # IOTKEY:underwriter_rank_avg

# ####### Cluster Robust OLS
m3 <- lm(eq3, data=dfa)
clx(m3, 1, FF49_industry)


require(lme4)
library(lmerTest)
### Hierarchical Linear Model

m3.lmer1 <- lmer(update(eq3, ~ . + (1 | FF49_industry)))
summary(m3.lmer1)

### Hierarchical Linear Model
# m1.lme1 <- lme(eq1, random = ~ 1 | FF49_industry)
# summary(m1.lme1)
# ranef(m1.lme)

}










################################
TIMING_PRICE_UPDATE_REGRESSIONS <- function(){
################################
# dfa <- dfu
# dfa <- df
# dfa <- dfu[df$close_return < 200]
dfa <- df[df$delay_in_price_update != 1]
attach(dfa)

# IOTKEY <- getIOTKEY("IoT_15day_CASI_news", dfa)
# IOTKEY <- getIOTKEY("IoT_15day_CASI_all", dfa)
IOTKEY <- getIOTKEY("IoT_15day_CASI_weighted_finance", dfa)
IOTKEY <- lm(IoT_15day_CASI_weighted_finance ~ media_listing )$residuals

eq2 <- percent_first_price_update ~
    log(days_from_s1_to_listing) +
    delay_in_price_update +
    underwriter_rank_avg + VC + share_overhang +
    log(proceeds) + log(1+sales) + EPS +
    M3_indust_rets + M3_initial_returns +
    Year +
    underwriter_syndicate_size +
    FF49_industry +
    # confidential_IPO +
    # foreign +
    media_listing +
    IOTKEY +
    I(IOTKEY^2)


# ####### Cluster Robust OLS
m2 <- lm(eq2, data=dfa)
clx(m2, 1, FF49_industry)


require(lme4)
### Hierarchical Linear Model - LME
m2.lme1 <- lme(eq2, random = ~ 1 | FF49_industry)
# summary(m2.lme1)
# ranef(m2.lme)



library(lmerTest)
m2.lmer1 <- lmer(update(eq2, ~ . + (1 | FF49_industry)))
# m2.lmer4 <- lmer(update(eq2, ~ . + (1 + IOTKEY | FF49_industry)))
summary(m2.lmer1)
summary(m2.lmer4)

}










################################
PERSUASION <- function() {
################################
library(mlogit)

dfa <- df[(df$delay_in_price_update < 1) & (df$prange_change_pct != "NA")]
attach(dfa)
IOTKEY <- getIOTKEY("IoT_15day_CASI_weighted_finance", dfa)
IOTKEY <- lm(IoT_15day_CASI_weighted_finance ~ media_listing )$residuals

DELAY <- delay_in_price_update

eq4 <- prange_change_pct ~
    underwriter_syndicate_size +
    underwriter_rank_avg +
    VC + share_overhang +
    log(days_from_s1_to_listing) +
    log(proceeds) + log(1+sales) + EPS +
    M3_indust_rets + M3_initial_returns +
    Year +
    FF49_industry +
    # confidential_IPO +
    # foreign +
    media_listing +
    IOTKEY +
    I(IOTKEY^2) +
    DELAY +
    pct_first_price_change +
    pct_first_price_change_up
    # pct_first_price_change_down

# ####### Cluster Robust OLS
m4 <- lm(eq4, data=dfa)
clx(m4, 1, FF49_industry)


require(lme4)
library(lmerTest)
m4.lmer1 <- lmer(update(eq4, ~ . + (1 | FF49_industry)))
# m4.lmer4 <- lmer(update(eq4, ~ . + (1 + IOTKEY | FF49_industry)))

summary(m4.lmer1)
# summary(m4.lmer4)
}

