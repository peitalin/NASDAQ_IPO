
require(sandwich)
require(lmtest)
require(quantreg)
require(plm)
source("clmclx.R")


df <- data.table::fread("df.csv", colClasses=c(cik="character", SIC="character", Year="factor"))
dfu <- data.table::fread("df_update.csv", colClasses=c(cik="character", SIC="character", Year="factor"))
attach(df)


### Final Price Revision Regressions
## Control Variables
eq <- percent_final_price_revision ~ log(days_from_s1_to_listing) +
				underwriter_rank_avg +
				VC +
				number_of_price_updates_up +
				number_of_price_updates_down +
				Year +
				share_overhang +
				log(proceeds) +
				EPS +
				M3_indust_rets +
				M3_initial_returns

m01 <- lm(eq, data=df)
eq <- update(eq, ~ . + priceupdate_up + priceupdate_down)
m03 <- lm(eq, data=df)


# Main independent variable
# IOTKEY <- df$IoT_30day_CASI_weighted_finance
IOTKEY <- df$IoT_15day_CASI_weighted_finance
# IOTKEY <- df$IoT_30day_CASI_news
# IOTKEY <- df$IoT_15day_CASI_news

eq4 <- update(eq, ~ . + IOTKEY + I(IOTKEY**2))
m04 <- lm(eq4, data=df)

eq5 <- update(eq4, ~ . + IOTKEY:priceupdate_up + IOTKEY:priceupdate_down)
m05 <- lm(eq5, data=df)
eq6 <- update(eq5, ~ . + IOTKEY:VC)
m06 <- lm(eq6, data=df)

# coeftest(m06, vcov=vcovHC(m06, type="HC1"))
clx(m06, 1, FF49_industry)
mclx(m06, 1, FF49_industry, underwriter_rank_single)
summary(m06)$r.squared


### INITIAL RETURNS REGRESSIONS
################################
IOTKEY <- df$IoT_15day_CASI_news
IOTKEY <- df$IoT_15day_CASI_all
IOTKEY <- df$IoT_15day_CASI_weighted_finance

eq9 <- close_return ~ log(days_from_s1_to_listing) +
	underwriter_rank_avg + VC + share_overhang + log(proceeds) +
	Year + EPS + M3_indust_rets + M3_initial_returns +
	IOTKEY +
	I(IOTKEY^2) +
	# FF49_industry +
	# pct_final_revision_up +
	# pct_final_revision_down +
	pct_first_price_change_up +
	# pct_first_price_change_down +
	number_of_price_updates_up +
	number_of_price_updates_down +

	# IOTKEY:pct_final_revision_up
	# IOTKEY:pct_final_revision_down
	IOTKEY:pct_first_price_change_up
	# IOTKEY:pct_first_price_change_down
	# IOTKEY:number_of_price_updates_up +
	# IOTKEY:number_of_price_updates_down

m09 <- lm(eq9, data=df)
# coeftest(m07, vcov=vcovHC(m07, type="HC1"))

clx(m09, 1, FF49_industry)
# mclx(m09, 1, FF49_industry, underwriter_rank_single)
mclx(m09, 1, FF49_industry, Year)
summary(m09)$r.squared







############ PRICE UPDATE REG
# IOTKEY <- df$IoT_15day_CASI_news
# IOTKEY <- df$IoT_15day_CASI_weighted_finance
# dup <- df[!is.na(df$size_of_first_price_update)]
# pupdate_eq1 <-  "percent_first_price_update ~ Year + log(days_from_s1_to_listing) + underwriter_rank_avg + VC + confidential_IPO + share_overhang + log(proceeds) + log(market_cap) + log(1+sales) + liab_over_assets + EPS + M3_indust_rets + M3_initial_returns + delay_in_price_update + IOTKEY + I(IOTKEY^2) + IOTKEY:VC"
# p01 <- lm(pupdate_eq1, data=dup)
# clx(p01, 1, dup$FF49_industry)
# mclx(p01, 1, dup$FF49_industry, dup$underwriter_rank_single)





# # QUANTILE REGRESSIONS
# require(quantreg)
# FRP <- percent_final_price_revision
# FRP <- pct_final_revision_up
# # FRP <- pct_final_revision_down
# IR <- close_return

# plotvar <- "IOTKEY"
# # plotvar <- "pct_final_revision_up:IOTKEY"
# plot(FRP, IR, cex=0.25, type="n", xlab=plotvar, ylab="Initial Returns (%)")
# points(FRP, IR, cex=0.5, col="blue")
# abline(rq(IR ~ FRP, tau=0.5), col="blue")
# abline(lm(IR ~ FRP), lty=2, col="red")
# # taus <- c(0.05, 0.10, 0.25, 0.75, 0.90, 0.95)

# taus <- seq(0.1, 0.9, 0.1)
# get_variable <- function(model, varname) {
# 	coefs <- model$coefficients
# 	for(i in seq_along(1:length(coefs))) {
# 		if(names(coefs[i]) == varname) {
# 			return(i)
# 		}
# 	}
# }

# for(i in seq_along(1:length(taus))) {
# 	M <- rq(eq9, tau=taus[i])
# 	coef <- M$coefficients
# 	abline(coef[1], coef[get_variable(M, plotvar)], col="gray")
# }




rqeq <- close_return ~ log(days_from_s1_to_listing) +
	underwriter_rank_avg + VC + share_overhang + log(proceeds) +
	Year + EPS + M3_indust_rets + M3_initial_returns

rqeq1 <- update(rqeq, ~ . + IOTKEY + I(IOTKEY^2) + number_of_price_updates_up + number_of_price_updates_down)
rqeq2 <- update(rqeq1, ~ . + pct_final_revision_up + pct_final_revision_up:IOTKEY)
rqeq2 <- update(rqeq1, ~ . + pct_first_price_change_up + pct_first_price_change_up:IOTKEY)
# rqeq2 <- update(rqeq1, ~ . + number_of_price_updates_up:IOTKEY)
taus <- c(0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90)
taus <- seq(0.05, 0.95, 0.05)


quantreg_print <- function(i, last=1) {
	qr <- rq(rqeq2, tau=taus[i])
	tail(summary(qr)$coefficients, last)
}


for(i in seq_along(1:length(taus))) {
	cat('\n\n', paste("Tau:", taus[i]), '\n')
	print(quantreg_print(i))
}




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

