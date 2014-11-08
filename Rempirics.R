
require(sandwich)
require(lmtest)
require(plm)
source("clmclx.R")


df <- data.table::fread("df.csv", colClasses=c(cik="character", SIC="character", Year="factor"))
dfu <- data.table::fread("df_update.csv", colClasses=c(cik="character", SIC="character", Year="factor"))
attach(df)

Y <- df$percent_final_price_revision

## Control Variables
eq <- Y ~ log(days_from_s1_to_listing) +
				underwriter_rank_avg + VC +
				number_of_price_updates +
				Year +
				share_overhang +
				log(proceeds) +
				# log(market_cap) +
				# liab_over_assets +
				EPS +
				M3_indust_rets

m01 <- lm(eq, data=df)
eq <- update(eq, ~ . + priceupdate_up + priceupdate_down)
m03 <- lm(eq, data=df)


# Main independent variable
# IOTKEY <- df$IoT_30day_CASI_weighted_finance
IOTKEY <- df$IoT_15day_CASI_weighted_finance
eq4 <- update(eq, ~ . + IOTKEY + I(IOTKEY**2))
m04 <- lm(eq4, data=df)

eq5 <- update(eq4, ~ . + IOTKEY:priceupdate_up + IOTKEY:priceupdate_down)
m05 <- lm(eq5, data=df)
eq6 <- update(eq5, ~ . + IOTKEY:VC)
m06 <- lm(eq6, data=df)




# formul <- Y ~ underwriter_rank_avg + VC +
# 			share_overhang +
# 			log(proceeds) +
# 			EPS +
# 			M3_indust_rets +
# 			M3_initial_returns +
# 			priceupdate_up + priceupdate_down + prange_change_plus +
# 			delay_in_price_update + log(1 + days_from_s1_to_listing) +
# 			IOTKEY +
# 			IOTKEY * priceupdate_up +
# 			IOTKEY * priceupdate_down +
# 			# IOTKEY * prange_change_plus +
# 			# IOTKEY * log(1 + days_to_first_price_change) +
# 			IOTKEY * log(1 + days_from_s1_to_listing) +
# 			IOTKEY * delay_in_price_update
# m11 <- lm(formul, data=df)


eq7 <- close_return ~ log(days_from_s1_to_listing) + number_of_price_updates +
	underwriter_rank_avg +
	VC + share_overhang + log(proceeds) +
    Year + EPS + M3_indust_rets + M3_initial_returns +
	pct_final_revision_up +
    pct_final_revision_down +
	IOTKEY + I(IOTKEY^2) +
	pct_final_revision_up:IOTKEY +
	IOTKEY:VC
	# pct_final_revision_up:underwriter_rank_avg
m07 <- lm(eq7, data=df)

summary(m07)
coeftest(m07, vcov=vcovHC(m07, type="HC1"))


# require(survival)
# Y <- Surv(df$days_to_first_price_change)
# formul2 <- with(dfu,
# 			(Y ~ underwriter_rank_avg + VC +
# 			share_overhang +
# 			log(proceeds) +
# 			log(market_cap) +
# 			liab_over_assets +
# 			EPS +
# 			M3_indust_rets +
# 			M3_initial_returns +
# 			# priceupdate_up + priceupdate_down + prange_change_plus +
# 			# delay_in_price_update + log(1 + days_from_s1_to_listing) +
# 			dfu$IoT_15day_CASI_weighted_finance))
# 			# IOTKEY * priceupdate_up +
# 			# IOTKEY * priceupdate_down +
# 			# IOTKEY * prange_change_plus +
# 			# IOTKEY * log(1 + days_to_first_price_change) +
# 			# IOTKEY * log(1 + days_from_s1_to_listing) +
# 			# IOTKEY * delay_in_price_update))
# m12 <- coxph(formul2)

